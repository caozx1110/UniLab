# GHTracking numba acceleration — design

**Date:** 2026-07-21
**Branch:** feat/gh-mujoco-migration
**Status:** approved (architecture), pending implementation plan

## Problem

`GHTrackingEnv.update_state` (reward + obs + termination) is single-threaded
float64 numpy. Measured on this machine (5090 GPU + 32-core CPU, num_envs=4096,
horizon=32, train phase, avg of 5 iters excl warmup):

| segment | s/iter | note |
|---|---|---|
| iter total | 5.88 | fps ~22k |
| collect | 5.06 (86%) | dominated by env.step |
| ├ policy.act | 0.03 | GPU |
| ├ env.step | 4.92 | CPU |
| │  ├ physics (mj_step) | 1.26 | multi-threaded C, ~24 cores at peak |
| │  ├ **update_state (obs+reward)** | **1.69** | **single-core numpy** |
| │  ├ set_ctrl (per-substep loop) | 0.79 | single-core (out of scope) |
| │  ├ astype fp64→fp32 | 0.44 | single-core (out of scope) |
| │  └ reset_done | 0.52 | single-core |
| train (train_op) | 0.82 | GPU |

CPU utilization during collect measured at **27% system / avg 2.2 of 32 cores
busy** — because ~79% of the iter is single-core numpy or GPU work with no
overlap; only physics lights up multiple cores. `update_state` is the largest
addressable single-core block, and every obs/reward term is per-env independent
(proven by commit 4b065c8e), so it is embarrassingly parallel along the env
axis.

## Chosen approach

Port the existing, in-repo numba acceleration pattern (used by
`motion_tracking/g1` and `locomotion/g1`) to gh_tracking. A fused
`@njit(parallel=True, fastmath=True, nogil=True)` kernel iterates
`for i in prange(n)`, computing each env's reward vector + priv_critic + policy/
priv obs rows + termination in parallel across cores. Simulation stays on CPU
(UniLab tenet — never move sim to GPU).

**Decisions locked with user:**
- **Precision: A — float32 + fastmath.** User accepts fp32 precision. This
  matches the existing motion_tracking numba path. It does NOT preserve the
  current bit-identical (max_abs_diff=0.0) self-parity; acceptance moves to a
  tolerance (`rtol=1e-4, atol=1e-5`, same as motion_tracking).
- **Scope: reward + obs + termination only** ("the safe part", fused kernel with
  a proven precedent). `before_update` (motion slice gather + force/admittance,
  stateful/per-substep) and `set_ctrl` (inside backend.step, per-substep) are
  explicitly deferred to later phases.
- **Opt-in: config field** — `GHTrackingCfg.numba_acceleration: bool = False` +
  `numba_num_threads: int | None = None`, mirroring `tracking.py`. numpy path is
  retained unchanged as fallback and parity baseline.
- **Immediately measure speedup after implementation.**

## Architecture

### New module: `src/unilab/envs/gh_tracking/gh_tracking_numba.py`
Mirrors `motion_tracking/g1/motion_tracking_numba.py`:
- Optional import guard: `try: from numba import njit, prange, get_thread_id,
  set_num_threads except ImportError: set to None`.
- Per-term scalar device functions `@njit(inline="always", fastmath=True,
  cache=True, nogil=True)` — one `*_i(...)` per obs/reward term, each a scalar
  translation of the corresponding vectorized function in `observations.py` /
  `rewards.py`. Each keeps a `.py_func` reachable for the math-parity unit test.
- One fused kernel `@njit(parallel=True, fastmath=True, cache=True, nogil=True)`
  with `for i in prange(n):` writing into pre-allocated output arrays
  (reward_vec (N,3), policy (N,450), priv (N,717), priv_critic (N,3),
  terminated (N,)).
- `class GHTrackingNumbaAccelerator`:
  - `is_available()` / `unsupported_terms()` — refuse or fall back cleanly.
  - `from_env(env, num_threads)` — capture indices/constants once (cold path).
  - `compute_update_state(...) -> GHNumbaResult` — the hot entrypoint returning
    reward_vec, obs dict, terminated.

### Modified: `src/unilab/envs/gh_tracking/config.py`
Add to `GHTrackingCfg`: `numba_acceleration: bool = False`,
`numba_num_threads: int | None = None`.

### Modified: `src/unilab/envs/gh_tracking/env.py`
- `__init__`: build `self._numba_accelerator` when `cfg.numba_acceleration`
  (mirrors `tracking.py:584-591`), else `None`.
- `update_state`: branch `if self._numba_accelerator is not None:` → call the
  accelerator; `else:` → existing numpy path, unchanged (before_update /
  _compute_reward / _compute_obs).

**The numpy path is not deleted.** numba is a parallel optional layer.

## Stateful history buffers — boundary

gh obs depends on stateful telemetry buffers (`HistoryBuffer`, `JointPosHistory`,
`RootLinVelEMA`, `ContactForceHistory`) that roll each control step and inject
noise via an rng stream. To keep the rng sequence and noise parity intact:

- **Buffer roll (`np.roll`/shift) and noise injection stay in numpy**, outside
  the kernel. They are cheap relative to the geometric transforms and carry rng
  state.
- The kernel receives the already-rolled buffer arrays and the per-env data, and
  does only the per-env stateless math (quat-inverse-apply, keypoint diffs,
  norms, exp-rewards, gather-and-flatten).

This confines numba to the pure, per-env-independent arithmetic and leaves all
stateful / rng-carrying logic on the existing numpy code, minimizing parity risk.

## Data flow

```
update_state (numba branch):
  before_update()                      # numpy, unchanged (motion slice, force)
  history buffers .update()            # numpy roll + noise, unchanged (rng)
  gather backend states                # numpy (get_base_pos/quat/body_pos...)
  → kernel(prange n): reward_vec, obs rows, terminated   # numba, multi-core
  termination counter / truncation     # numpy, unchanged
  state.replace(obs, reward, terminated, truncated, info)
```

## Testing

1. **Math parity (device func level):** each `*_i.py_func(...)` vs its numpy
   counterpart on small inputs — `pytest.approx(rel=1e-3, abs=1e-6)`.
2. **update_state parity (kernel level):** fixed-seed synthetic env, numba vs
   numpy path, `reward` and `obs` groups `rtol=1e-4, atol=1e-5`; `terminated`
   exact array-equal. Both no-noise and (if feasible) fixed-noise variants,
   mirroring `test_g1_motion_tracking_numba.py`.
3. **Fallback guard:** `numba_acceleration=True` with numba absent → clean error
   or documented fallback; unsupported reward terms rejected.
4. **Speedup measurement:** fixed-seed rollout, numba on vs off, report iter
   time + CPU core utilization delta. Success = meaningful multi-core speedup on
   the update_state segment with parity within tolerance.
5. **Regression:** existing gh_tracking + gh_distill_ppo suites pass with numba
   off (default), confirming the numpy path is untouched.

## Out of scope (later phases)

- `before_update` (motion slice gather, force curriculum, admittance).
- `set_ctrl` per-substep loop inside `backend.step`.
- astype fp64→fp32 copy (#3) and set_ctrl bug hunt (#2) — separate follow-ups.
- Async/double-buffered collect pipeline (structural, largest potential, separate).
- Any fp32-ification of the numpy path itself.

## Risks

- **fastmath breaks bit-identical self-parity** — accepted; acceptance is now a
  tolerance, documented in tests and commit.
- **Noise/rng parity** — mitigated by keeping roll+noise in numpy.
- **Large translation surface** — obs has ~26 terms; each needs a scalar kernel
  function. Motion_tracking's file is ~1275 lines; expect comparable. Mitigated
  by per-term unit parity tests catching translation errors early.
- **First-call JIT compile latency** (`cache=True` amortizes across runs).
