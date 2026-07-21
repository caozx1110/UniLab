# GHTracking numba acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in numba (`prange` over the env axis) acceleration layer to `GHTrackingEnv.update_state` (reward + obs + termination), keeping the existing numpy path as fallback + parity baseline.

**Architecture:** New self-contained module `gh_tracking_numba.py` mirroring `motion_tracking/g1/motion_tracking_numba.py`: per-term `@njit(fastmath=True, nogil=True)` scalar device functions, one fused `@njit(parallel=True)` kernel iterating `for i in prange(n)`, and a `GHTrackingNumbaAccelerator` class. Wired into `update_state` behind a config flag. Stateful history-buffer roll + noise stay in numpy (rng stream); the kernel does only per-env-independent math.

**Tech Stack:** Python, numpy, numba (already a dep, `pyproject.toml:15`), pytest.

## Global Constraints

- **Always use `uv run`, not python.** (CLAUDE.md)
- **Precision path A:** kernel is **float32 + `fastmath=True`**. Parity is a tolerance, NOT bit-identical: `rtol=1e-4, atol=1e-5` for reward/obs, exact array-equal for `terminated` (matches `tests/envs/test_g1_motion_tracking_numba.py`).
- **numpy path is never deleted** — it is the fallback and the parity baseline. Default `numba_acceleration=False`.
- **Scope:** reward + obs + termination only. `before_update` (motion slice, force/admittance) and `set_ctrl` are OUT of scope.
- **Contract:** `obs_groups_spec` stays `{"policy":450,"priv":717,"priv_critic":3}`; reward stays a 3-vector `[impedance, tracking, loco]` summed by GAE. Kernel output must match the numpy path's group order and slice layout exactly.
- Follow the in-repo precedent verbatim: optional import guard, `is_available()`/`unsupported_terms()`, `from_env(env, num_threads)`, `compute_update_state(...)`, `set_num_threads` at call entry, frozen-dataclass result. Reference: `src/unilab/envs/motion_tracking/g1/motion_tracking_numba.py`.

---

### Task 1: Module scaffold, config flag, and wiring seam (numpy-delegating accelerator)

Establishes the opt-in path end-to-end with the accelerator delegating to the existing numpy computation, so the seam is proven correct before any kernel exists. Default off; numpy path untouched.

**Files:**
- Create: `src/unilab/envs/gh_tracking/gh_tracking_numba.py`
- Modify: `src/unilab/envs/gh_tracking/config.py` (add two `GHTrackingCfg` fields)
- Modify: `src/unilab/envs/gh_tracking/env.py` (`__init__` build accelerator; `update_state` branch)
- Test: `tests/envs/gh_tracking/test_numba_accel.py`

**Interfaces:**
- Produces:
  - `is_available() -> bool` (numba importable)
  - `unsupported_terms(groups: dict) -> frozenset[str]` (reward terms with no kernel yet; Task 1 returns `frozenset()` since delegation supports all)
  - `@dataclass(frozen=True) GHNumbaResult(reward_vec: np.ndarray, obs: dict[str, np.ndarray], terminated: np.ndarray)`
  - `class GHTrackingNumbaAccelerator` with `from_env(cls, env, num_threads: int | None) -> GHTrackingNumbaAccelerator` and `compute_update_state(self, env) -> GHNumbaResult`
- Consumes (Task 1 delegation): `env._before_update()`, `env._compute_reward()`, `env._compute_obs()`, `env.termination`, `env._episode_length`, `env._cfg`, `env._motion_finished()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/gh_tracking/test_numba_accel.py
import numpy as np
import pytest
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking import gh_tracking_numba as NB


def _make_env(numba: bool):
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    cfg = GHTrackingCfg()
    cfg.numba_acceleration = numba
    return GHTrackingEnv(cfg, num_envs=8, backend_type="mujoco")


def test_config_defaults_off():
    cfg = GHTrackingCfg()
    assert cfg.numba_acceleration is False
    assert cfg.numba_num_threads is None


def test_accelerator_delegation_matches_numpy_update_state():
    # numba path (delegating) must equal the numpy path bit-for-bit in Task 1,
    # because the accelerator just calls the same numpy functions.
    np.random.seed(0)
    env_np = _make_env(numba=False)
    s_np = env_np.init_state()
    s_np = env_np.step(np.zeros((8, env_np._backend.num_actuators), dtype=np.float32))

    np.random.seed(0)
    env_nb = _make_env(numba=True)
    assert env_nb._numba_accelerator is not None
    s_nb = env_nb.init_state()
    s_nb = env_nb.step(np.zeros((8, env_nb._backend.num_actuators), dtype=np.float32))

    for g in ("policy", "priv", "priv_critic"):
        np.testing.assert_allclose(s_nb.obs[g], s_np.obs[g], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(s_nb.reward, s_np.reward, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(s_nb.terminated, s_np.terminated)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_accel.py -v`
Expected: FAIL — `AttributeError: 'GHTrackingCfg' object has no attribute 'numba_acceleration'` / module import error.

- [ ] **Step 3: Add config fields**

In `src/unilab/envs/gh_tracking/config.py`, add to `GHTrackingCfg` (place near other top-level scalar fields):

```python
    numba_acceleration: bool = False
    numba_num_threads: int | None = None
```

- [ ] **Step 4: Create the scaffold module (numpy-delegating accelerator)**

```python
# src/unilab/envs/gh_tracking/gh_tracking_numba.py
"""Optional numba (prange over env axis) acceleration for GHTrackingEnv.update_state.

Mirrors motion_tracking/g1/motion_tracking_numba.py. Path A: float32 + fastmath;
parity is rtol=1e-4/atol=1e-5, not bit-identical. Task 1 delegates to the numpy
path to prove the wiring; later tasks move reward/obs/termination into a kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange, set_num_threads  # noqa: F401
    _NUMBA = True
except ImportError:  # pragma: no cover
    njit = prange = set_num_threads = None  # type: ignore[assignment]
    _NUMBA = False


def is_available() -> bool:
    return _NUMBA


def unsupported_terms(groups: dict) -> frozenset[str]:
    # Task 1 delegates to numpy, so every term is "supported". Later tasks
    # narrow this to terms without a kernel translation.
    return frozenset()


@dataclass(frozen=True)
class GHNumbaResult:
    reward_vec: np.ndarray          # (N, 3) [impedance, tracking, loco]
    obs: dict[str, np.ndarray]      # policy (N,450) / priv (N,717) / priv_critic (N,3)
    terminated: np.ndarray          # (N,) bool


class GHTrackingNumbaAccelerator:
    def __init__(self, num_threads: int | None) -> None:
        self.num_threads = num_threads

    @classmethod
    def from_env(cls, env, num_threads: int | None) -> "GHTrackingNumbaAccelerator":
        if not _NUMBA:
            raise RuntimeError(
                "numba_acceleration=True but numba is not importable; "
                "install numba or set numba_acceleration=False"
            )
        return cls(num_threads=num_threads)

    def compute_update_state(self, env) -> GHNumbaResult:
        # Task 1: delegate to the numpy path (proves the seam). Tasks 2-4 replace
        # the body with the fused kernel.
        if self.num_threads is not None and _NUMBA:
            set_num_threads(self.num_threads)
        reward_vec = env._compute_reward()            # (N,3), writes _cum_error
        obs = env._compute_obs()                      # dict of 3 groups
        from unilab.envs.gh_tracking.terminations import apply_terminate_gate
        terminated = apply_terminate_gate(
            env.termination.terminated(), env._episode_length)[:, 0]
        return GHNumbaResult(reward_vec=reward_vec, obs=obs, terminated=terminated)
```

- [ ] **Step 5: Wire into env `__init__` and `update_state`**

In `src/unilab/envs/gh_tracking/env.py` `__init__`, after the reward manager / obs manager are built and before `_init_domain_randomization`, add:

```python
        self._numba_accelerator = None
        if getattr(cfg, "numba_acceleration", False):
            from unilab.envs.gh_tracking.gh_tracking_numba import (
                GHTrackingNumbaAccelerator,
            )
            self._numba_accelerator = GHTrackingNumbaAccelerator.from_env(
                self, num_threads=getattr(cfg, "numba_num_threads", None)
            )
```

Replace the body of `update_state` (env.py:447-467) so the shared pre-steps run, then branch:

```python
    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._before_update()
        self.termination.update(self._cum_error)  # 1-step _cum_error lag
        acc = getattr(self, "_numba_accelerator", None)
        if acc is not None:
            res = acc.compute_update_state(self)
            reward_vec, obs, terminated = res.reward_vec, res.obs, res.terminated
        else:
            reward_vec = self._compute_reward()
            obs = self._compute_obs()
            terminated = apply_terminate_gate(
                self.termination.terminated(), self._episode_length)[:, 0]
        reward = reward_vec.sum(axis=-1)
        truncated = compute_truncation(
            self._episode_length, self._cfg.termination.max_episode_length,
            self._motion_finished(),
        )[:, 0]
        info = dict(state.info)
        info["reward_vec"] = reward_vec
        return state.replace(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info
        )
```

(Confirm `apply_terminate_gate` / `compute_truncation` are already imported at the top of env.py; they are used in the current `update_state`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_accel.py -v`
Expected: PASS (3 tests). The delegating accelerator reproduces the numpy result exactly.

- [ ] **Step 7: Regression — numpy path untouched**

Run: `uv run pytest tests/envs/gh_tracking/ -q`
Expected: PASS (all existing gh_tracking tests; default numba off).

- [ ] **Step 8: Commit**

```bash
git add src/unilab/envs/gh_tracking/gh_tracking_numba.py \
        src/unilab/envs/gh_tracking/config.py \
        src/unilab/envs/gh_tracking/env.py \
        tests/envs/gh_tracking/test_numba_accel.py
git commit -m "feat(gh_tracking): numba accel scaffold + wiring seam (delegating)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Reward kernel — scalar device functions + fused `prange` reward kernel

Translate every reward term (`rewards.py`) into `@njit(inline="always", fastmath=True, cache=True, nogil=True)` scalar `*_i(...)` functions and a `@njit(parallel=True, ...)` kernel that fills `reward_vec` (N,3). Replace only the reward computation inside `compute_update_state`; obs/termination still delegate to numpy.

**Files:**
- Modify: `src/unilab/envs/gh_tracking/gh_tracking_numba.py`
- Test: `tests/envs/gh_tracking/test_numba_reward.py`

**Interfaces:**
- Consumes: reward context `env._rc` (dict built by `_compute_reward`; capture the same inputs the numpy term_fns read — see `env._build_reward_manager` env.py:397-442 for the exact key→function map, and `_REWARD_GROUPS`/`_REWARD_SIGMA` env.py:59-105 for weights and sigmas).
- Produces: `_reward_kernel(...) -> None` (writes into a pre-alloced `(N,3)` float32 array); private helper `self._compute_reward_vec(env) -> np.ndarray (N,3)`.

**Translation table (each is a 1:1 scalar port of an in-repo numpy function; source is the spec):**

| kernel `*_i` fn | numpy source | group / weight |
|---|---|---|
| `survival_i` | `rewards.survival` (rewards.py:75) | loco 5.0 |
| `action_rate_l2_i` | `rewards.action_rate_l2` (rewards.py:80) | loco 0.1 |
| `impact_force_l2_i` | `rewards.impact_force_l2` (rewards.py:93) | loco 4.0 |
| `feet_slip_i` | `rewards.feet_slip` (rewards.py:108) | loco 2.0 |
| `joint_vel_l2_i` | `rewards.joint_vel_l2` (rewards.py:127) | loco 5e-4 |
| `joint_pos_limits_i` | `rewards.joint_pos_limits` | loco 1.0 |
| `feet_air_time_ref` | precomputed `s["feet_air_time_reward"]` (pass-through column) | loco 10.0 |
| `lower_keypoint_tracking_i` | `rewards.lower_keypoint_tracking` | tracking 2.0 |
| `root_pos_tracking_i` | `rewards.root_pos_tracking` | tracking 0.5 |
| `root_rot_tracking_i` | `rewards.root_rot_tracking` | tracking 0.5 |
| `root_vel_tracking_i` | `rewards.root_vel_tracking` | tracking 1.0 (+ ang_vel 1.0) |
| `joint_pos_tracking_i` | `rewards.joint_pos_tracking` | tracking 1.0 |
| `joint_vel_tracking_i` | `rewards.joint_vel_tracking` | tracking 0.5 |
| `force_reward_i` | `rewards.force_reward` | impedance 2.0 |
| `force_exd_penalty_i` | `rewards.force_exd_penalty` | impedance 6.0 |
| `force_target_tracking_i` | `rewards.force_target_tracking` | impedance 2.0 |
| `force_target_vel_tracking_i` | `rewards.force_target_vel_tracking` | impedance 1.0 |
| `keypoint_tracking_imp_i` | `rewards.keypoint_tracking_imp` | impedance 2.0 |

`calc_exp_sigma` (rewards.py:18) becomes an inlined `_exp_reward(err, sigma_arr)` device fn (see motion_tracking `_exp_reward` motion_tracking_numba.py:87 for the exact pattern). Group value = `sum(weight*term) * current_factor(=1.0) * step_dt(=ctrl_dt)`, per `RewardManager.compute` (rewards.py:53-63). Column order in `reward_vec`: `[impedance, tracking, loco]` (dict insertion order of `_REWARD_GROUPS`).

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/gh_tracking/test_numba_reward.py
import numpy as np
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking.env import GHTrackingEnv


def _rollout_reward(numba: bool, steps: int = 3):
    np.random.seed(0)
    cfg = GHTrackingCfg()
    cfg.numba_acceleration = numba
    env = GHTrackingEnv(cfg, num_envs=16, backend_type="mujoco")
    s = env.init_state()
    rewards = []
    for _ in range(steps):
        s = env.step(np.zeros((16, env._backend.num_actuators), dtype=np.float32))
        rewards.append(s.info["reward_vec"].copy())
    return np.stack(rewards)


def test_reward_vec_parity_numba_vs_numpy():
    r_np = _rollout_reward(numba=False)
    r_nb = _rollout_reward(numba=True)
    # per-group 3-vector parity across the rollout
    np.testing.assert_allclose(r_nb, r_np, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_reward.py -v`
Expected: FAIL — reward still delegates to numpy so it PASSES trivially UNTIL Step 3 swaps in the kernel; to make the test meaningful first assert the kernel is actually used. Add to the accelerator a `self._reward_from_kernel = False` flag set True once implemented, and assert it in the test:

```python
def test_reward_uses_kernel():
    cfg = GHTrackingCfg(); cfg.numba_acceleration = True
    env = GHTrackingEnv(cfg, num_envs=4, backend_type="mujoco")
    assert env._numba_accelerator._reward_from_kernel is True
```

Expected: FAIL — attribute is False/absent.

- [ ] **Step 3: Implement scalar device fns + fused reward kernel**

Add to `gh_tracking_numba.py` (guarded by `if _NUMBA:` so import without numba still works). Implement each `*_i` per the translation table above, the `_exp_reward` helper, and:

```python
if _NUMBA:
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _reward_kernel(
        # per-env input arrays (all float32, shape (N, ...)) captured from env._rc,
        # + weight/sigma constant arrays, + output reward_vec (N,3):
        ...,
        reward_vec,
    ):
        n = reward_vec.shape[0]
        for i in prange(n):
            imp = (2.0 * force_reward_i(...) + 6.0 * force_exd_penalty_i(...)
                   + 2.0 * force_target_tracking_i(...) + 1.0 * force_target_vel_tracking_i(...)
                   + 2.0 * keypoint_tracking_imp_i(...))
            trk = (2.0 * lower_keypoint_tracking_i(...) + 0.5 * root_pos_tracking_i(...)
                   + 0.5 * root_rot_tracking_i(...) + 1.0 * root_vel_tracking_i(...)
                   + 1.0 * root_ang_vel_tracking_i(...) + 1.0 * joint_pos_tracking_i(...)
                   + 0.5 * joint_vel_tracking_i(...))
            loco = (5.0 * survival_i(...) + 4.0 * impact_force_l2_i(...)
                    + 2.0 * feet_slip_i(...) + 5e-4 * joint_vel_l2_i(...)
                    + 0.1 * action_rate_l2_i(...) + 10.0 * feet_air_time_ref[i]
                    + 1.0 * joint_pos_limits_i(...))
            reward_vec[i, 0] = imp * step_dt
            reward_vec[i, 1] = trk * step_dt
            reward_vec[i, 2] = loco * step_dt
```

Wire `self._compute_reward_vec(env)`: build the float32 input arrays from `env._rc` (populate `env._rc` by calling the existing numpy `_compute_reward` producers OR refactor `_compute_reward` to expose the context; the minimal path is to let `_compute_reward` still build `_rc` and _cum_error, then feed `_rc` arrays to the kernel — do NOT double-run the numpy reward math). Set `self._reward_from_kernel = True`. In `compute_update_state`, replace `reward_vec = env._compute_reward()` with `reward_vec = self._compute_reward_vec(env)`.

> Note: `_compute_reward` also produces `_cum_error` (consumed by priv_critic obs + termination). Preserve that side-effect — either keep the `_cum_error` producer lines in numpy and only move the group aggregation to the kernel, or reproduce the `_cum_error` writes in the kernel. Simpler + lower-risk: keep `_cum_error` production in numpy, move only the exp-reward term math + aggregation into the kernel.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_reward.py -v`
Expected: PASS — `test_reward_uses_kernel` and `test_reward_vec_parity_numba_vs_numpy` both pass (parity within rtol=1e-4).

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/gh_tracking/gh_tracking_numba.py tests/envs/gh_tracking/test_numba_reward.py
git commit -m "feat(gh_tracking): fused prange reward kernel (numba, fp32)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Observation kernel — scalar obs device fns + fused `prange` obs kernel

Translate the obs term functions (`observations.py`) into scalar `*_i` device fns and a kernel filling `policy` (N,450), `priv` (N,717), `priv_critic` (N,3). Stateful buffer roll + noise stay in numpy: call `self.obs_manager.update(state)` (numpy roll + rng noise) FIRST, then pass the already-rolled buffer arrays + gathered backend state into the kernel, which does only the per-env stateless assembly.

**Files:**
- Modify: `src/unilab/envs/gh_tracking/gh_tracking_numba.py`
- Test: `tests/envs/gh_tracking/test_numba_obs.py`

**Interfaces:**
- Consumes: the `ObsState` built by `env._build_obs_state()` (env.py:657-707, 26 fields) + the obs_manager's rolled buffers (`pol_angvel.buffer`, `pol_joint.buffer`, `priv_*`, `root_ema.linvel_w`, `contact.history`) + `history_steps` index arrays + `actuator_offset`.
- Produces: `_obs_kernel(...)` writing `policy`/`priv`/`priv_critic` output arrays; `self._compute_obs_dict(env) -> dict[str,np.ndarray]`; `self._obs_from_kernel = True`.

**Translation table (each `*_i` is a scalar port; source functions in `observations.py`):** every free function `observations.py:166-329` (`body_height`, `applied_action_obs`, `prev_actions_obs`, `boot_indicator_state_obs`, `command_obs`, `target_joint_pos_obs`, `target_projected_gravity_b`, `target_pos_b_obs`, `target_linvel_b_obs`, `relative_quat_obs`, `current_keypoint_b`, `current_keypoint_vel_b`, `target_keypoints_diff_b_obs`, `force_priv_obs`) plus the `.compute(env_ids)` readers of the four telemetry buffer classes. Slice offsets are fixed by the numpy `ObservationManager.compute` (observations.py:438-480) — replicate the exact `[start:end]` layout the comments document (policy `[0:1][1:23][23:168]...`; priv `[0:15][15:30]...[688:717]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/gh_tracking/test_numba_obs.py
import numpy as np
from unilab.envs.gh_tracking.config import GHTrackingCfg
from unilab.envs.gh_tracking.env import GHTrackingEnv


def _rollout_obs(numba: bool, steps: int = 3):
    np.random.seed(0)
    cfg = GHTrackingCfg(); cfg.numba_acceleration = numba
    env = GHTrackingEnv(cfg, num_envs=16, backend_type="mujoco")
    s = env.init_state()
    out = []
    for _ in range(steps):
        s = env.step(np.zeros((16, env._backend.num_actuators), dtype=np.float32))
        out.append({g: s.obs[g].copy() for g in ("policy", "priv", "priv_critic")})
    return out


def test_obs_uses_kernel():
    cfg = GHTrackingCfg(); cfg.numba_acceleration = True
    env = GHTrackingEnv(cfg, num_envs=4, backend_type="mujoco")
    assert env._numba_accelerator._obs_from_kernel is True


def test_obs_parity_numba_vs_numpy():
    a = _rollout_obs(numba=False)
    b = _rollout_obs(numba=True)
    for da, db in zip(a, b):
        for g in ("policy", "priv", "priv_critic"):
            np.testing.assert_allclose(db[g], da[g], rtol=1e-4, atol=1e-5)
```

Note: noise terms use the obs_manager rng; because both paths call `obs_manager.update` in numpy with the same seed sequence, noise matches. Confirm the numba branch still calls `obs_manager.update` exactly once per step (same call-site as numpy path).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_obs.py -v`
Expected: FAIL — `_obs_from_kernel` absent (obs still delegates to numpy).

- [ ] **Step 3: Implement obs device fns + fused obs kernel + wire**

Add the `*_i` device fns and `_obs_kernel` per the translation table. In `_compute_obs_dict(env)`: (1) `state = env._build_obs_state()`; (2) `env.obs_manager.update(state)` (numpy roll + noise, unchanged); (3) gather buffer arrays + `state` fields into float32 kernel inputs; (4) run `_obs_kernel`; (5) return `{"policy":..., "priv":..., "priv_critic":...}`. Set `self._obs_from_kernel = True`. In `compute_update_state`, replace `obs = env._compute_obs()` with `obs = self._compute_obs_dict(env)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_obs.py -v`
Expected: PASS (parity within rtol=1e-4).

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/gh_tracking/gh_tracking_numba.py tests/envs/gh_tracking/test_numba_obs.py
git commit -m "feat(gh_tracking): fused prange obs kernel (numba, fp32)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fuse reward + obs + termination into one kernel + full parity test

Merge the reward and obs kernels into a single `prange` pass (one `for i in prange(n)` computing reward_vec, obs rows, and termination gate for env i) to eliminate the double prange dispatch, and add the full `update_state` parity test.

**Files:**
- Modify: `src/unilab/envs/gh_tracking/gh_tracking_numba.py`
- Test: `tests/envs/gh_tracking/test_numba_accel.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 2-3.
- Produces: single `_update_state_kernel(...)` writing reward_vec/policy/priv/priv_critic/terminated; `compute_update_state` runs it once.

- [ ] **Step 1: Write the failing test (assert single fused kernel + full parity)**

```python
def test_update_state_full_parity_multistep():
    import numpy as np
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    def rollout(numba):
        np.random.seed(0)
        cfg = GHTrackingCfg(); cfg.numba_acceleration = numba
        env = GHTrackingEnv(cfg, num_envs=32, backend_type="mujoco")
        s = env.init_state(); frames = []
        for _ in range(10):
            s = env.step(np.zeros((32, env._backend.num_actuators), dtype=np.float32))
            frames.append((s.obs["policy"].copy(), s.obs["priv"].copy(),
                           s.obs["priv_critic"].copy(), s.reward.copy(),
                           s.terminated.copy()))
        return frames

    a, b = rollout(False), rollout(True)
    for (pa, va, ca, ra, ta), (pb, vb, cb, rb, tb) in zip(a, b):
        np.testing.assert_allclose(pb, pa, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(vb, va, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cb, ca, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(rb, ra, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(tb, ta)


def test_single_fused_kernel():
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    cfg = GHTrackingCfg(); cfg.numba_acceleration = True
    env = GHTrackingEnv(cfg, num_envs=4, backend_type="mujoco")
    assert env._numba_accelerator._fused is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/envs/gh_tracking/test_numba_accel.py::test_single_fused_kernel -v`
Expected: FAIL — `_fused` absent.

- [ ] **Step 3: Implement the fused kernel**

Combine the reward + obs bodies into one `_update_state_kernel` with a single `for i in prange(n)`; compute the termination gate inline (`apply_terminate_gate` logic per env — port from `terminations.apply_terminate_gate`). Set `self._fused = True`. `compute_update_state` builds all inputs once, calls the single kernel, returns `GHNumbaResult`.

- [ ] **Step 4: Run full parity + regression**

Run: `uv run pytest tests/envs/gh_tracking/ -q`
Expected: PASS (all, numba on tests within tolerance, numpy default untouched).

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/gh_tracking/gh_tracking_numba.py tests/envs/gh_tracking/test_numba_accel.py
git commit -m "feat(gh_tracking): single fused update_state kernel + full parity

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Speedup measurement + docs

Measure iter time and CPU utilization with numba on vs off, at num_envs=4096, and record it.

**Files:**
- Create: `docs/superpowers/notes/2026-07-21-gh-numba-speedup.md` (force-add; docs/superpowers is gitignored)

- [ ] **Step 1: Measure numpy baseline**

Run (reuse the profiler; it defaults numba off):
`uv run python /tmp/gh_profile.py`
Record: iter time, collect time, update_state ms.

- [ ] **Step 2: Measure numba on**

Create `/tmp/gh_profile_numba.py` = copy of `/tmp/gh_profile.py` with the compose overrides extended by `"env.numba_acceleration=true"` (and optionally `env.numba_num_threads=8`). Run:
`uv run python /tmp/gh_profile_numba.py`
Record the same fields + rerun once to exclude JIT-compile of the first call (`cache=True` amortizes).

- [ ] **Step 3: Measure CPU utilization delta**

Adapt `/tmp/cpu_probe.py` with `env.numba_acceleration=true`; compare avg busy cores during collect vs the 2.2/32 baseline.

- [ ] **Step 4: Write results doc + commit**

```bash
git add -f docs/superpowers/notes/2026-07-21-gh-numba-speedup.md
git commit -m "docs(gh_tracking): numba update_state speedup measurement

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** architecture (Task 1 module+wiring), fp32+fastmath path A (Tasks 2-4 kernels), reward+obs+termination scope (Tasks 2/3/4), config opt-in + numpy fallback retained (Task 1), stateful buffer boundary (Task 3 keeps roll+noise in numpy), parity tests rtol=1e-4 (Tasks 2-4), fallback guard (Task 1 `from_env` raise), speedup measurement (Task 5), out-of-scope items untouched. ✅

**Placeholder scan:** kernel `*_i` bodies are specified by exact in-repo numpy source (`path:line` + translation tables + reward weights/sigmas) — mechanical translations, not TODOs. The `...` in kernel signatures denote the per-env input arrays enumerated in each task's Interfaces/translation table. No "add error handling"/"TBD".

**Type consistency:** `GHNumbaResult(reward_vec, obs, terminated)` used consistently; `_reward_from_kernel`/`_obs_from_kernel`/`_fused` flags introduced and asserted; `compute_update_state`/`from_env` signatures stable across tasks; reward column order `[impedance, tracking, loco]` matches `_REWARD_GROUPS` insertion order everywhere.
