"""GHDistillPPO 3-phase runner (Phase 10.10).

Custom trainer-style runner (HoraDistillationTrainer template, but fully GH-semantic):
builds the live GHTrackingEnv + GHDistillPolicy + GH VecNorm + GHDistillTrainer, inherits
the prior-phase checkpoint (teacher->student hard copy on train->adapt), collects on-policy
rollouts through the phase's rollout policy, and runs the phase's ``train_op`` each outer
iteration. Emits the sim2sim ``contract_snapshot`` via ExperimentTracker and saves the
GH-schema checkpoint (no optimizer) itself — it does NOT use the rsl_rl OnPolicyRunner save
path (which persists optimizer state).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.gh_distill_ppo.checkpoint import (
    apply_meta_to_trainer,
    build_env_state,
    load_gh_checkpoint,
    meta_from_trainer,
    save_gh_checkpoint,
)
from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy
from unilab.algos.gh_distill_ppo.trainer import GHDistillTrainer, GHDistillTrainerCfg
from unilab.algos.gh_distill_ppo.vecnorm import VecNorm
from unilab.training.common import create_env, ensure_registries

_OBS_GROUPS = ("policy", "priv", "priv_critic")


def _algo(cfg) -> Any:
    return cfg.algo


def trainer_cfg_from(algo) -> GHDistillTrainerCfg:
    g = lambda k, d: getattr(algo, k, d) if not hasattr(algo, "get") else algo.get(k, d)
    return GHDistillTrainerCfg(
        lr=float(g("lr", 3e-4)),
        ppo_epochs=int(g("ppo_epochs", 5)),
        num_minibatches=int(g("num_minibatches", 8)),
        estimator_epochs=int(g("estimator_epochs", 2)),
        clip_param=float(g("clip_param", 0.2)),
        gamma=float(g("gamma", 0.99)),
        lam=float(g("lam", 0.95)),
        desired_kl=float(g("desired_kl", 0.01)),
        entropy_start=float(g("entropy_start", 0.005)),
        entropy_end=float(g("entropy_end", 0.002)),
        reg_lambda_max=float(g("reg_lambda_max", 0.2)),
    )


class GHDistillRunner:
    def __init__(self, cfg, device: str = "cpu", *, env_cfg_override: dict | None = None,
                 log_dir: str | Path | None = None) -> None:
        ensure_registries()
        self.cfg = cfg
        self.device = device
        algo = _algo(cfg)
        self.phase = str(algo.phase)

        override = {"student_train": bool(algo.student_train)}
        if env_cfg_override:
            override.update(env_cfg_override)
        self.env = create_env(
            cfg, num_envs=int(algo.num_envs),
            sim_backend=str(cfg.training.sim_backend), env_cfg_override=override,
        )
        dims = dict(self.env.obs_groups_spec)
        self.policy = GHDistillPolicy(obs_dims=dims).to(device)
        self.vecnorm = VecNorm(dims).to(device)
        self.vecnorm.train() if algo.vecnorm == "train" else self.vecnorm.eval()
        self.trainer = GHDistillTrainer(self.policy, trainer_cfg_from(algo))

        # inherit the prior-phase checkpoint (teacher->student hard copy handled in load)
        ckpt_path = getattr(algo, "checkpoint_path", None)
        if ckpt_path:
            res = load_gh_checkpoint(
                ckpt_path, policy=self.policy, vecnorm=self.vecnorm, target_phase=self.phase)
            if res["meta_restored"] is not None:
                apply_meta_to_trainer(self.trainer, res["meta_restored"])

        self.log_dir = Path(log_dir) if log_dir else None
        self._tracker = None

    # --- sim2sim contract snapshot (ExperimentTracker) --- #
    def start(self) -> None:
        if self.log_dir is None:
            return
        from unilab.training.experiment import ExperimentTracker
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tracker = ExperimentTracker(
            root_dir=Path.cwd(), log_dir=self.log_dir,
            algo_name=str(getattr(_algo(self.cfg), "algo_log_name", "gh_distill")),
            task_name=str(self.cfg.training.task_name),
            sim_backend=str(self.cfg.training.sim_backend),
            training_cfg=self.cfg.training, full_cfg=self.cfg,
        )
        self._tracker.start()

    # --- rollout collection --- #
    def _to_torch(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {g: torch.as_tensor(obs[g], dtype=torch.float32, device=self.device) for g in _OBS_GROUPS}

    def collect_rollout(self, state, horizon: int) -> tuple[dict, Any]:
        pol, pri, prc, act, loc, scl, logp, rew, done, is_init = ([] for _ in range(10))
        prev_done = torch.ones(self.env.num_envs, dtype=torch.bool)  # t=0 is an episode start
        for _ in range(horizon):
            obs = self._to_torch(state.obs)
            if self.vecnorm.training:
                self.vecnorm.update(obs)
            nobs = self.vecnorm.normalize(obs)
            with torch.no_grad():
                action, loc_t, scale_t, logp_t = self.policy.act(nobs, self.phase, sample=True)
            pol.append(nobs["policy"]); pri.append(nobs["priv"]); prc.append(nobs["priv_critic"])
            act.append(action); loc.append(loc_t); scl.append(scale_t); logp.append(logp_t)
            is_init.append(prev_done.clone())
            state = self.env.step(action.cpu().numpy())
            # scalar reward = 3-group reward_vec.sum(-1) (env sums it into state.reward) = GH GAE input
            rew.append(torch.as_tensor(state.reward, dtype=torch.float32, device=self.device))
            d = torch.as_tensor(state.terminated | state.truncated, dtype=torch.bool)
            done.append(d); prev_done = d
        stack = lambda xs: torch.stack(xs, dim=0)
        rollout = {
            "policy": stack(pol), "priv": stack(pri), "priv_critic": stack(prc),
            "action": stack(act), "loc": stack(loc), "scale": stack(scl),
            "sample_log_prob": stack(logp), "reward": stack(rew),
            "done": stack(done), "is_init": stack(is_init),
        }
        return rollout, state

    # --- main loop --- #
    def learn(self, num_iterations: int | None = None) -> dict:
        algo = _algo(self.cfg)
        total = int(num_iterations if num_iterations is not None else algo.max_iterations)
        horizon = int(getattr(algo, "train_every", getattr(algo, "rollout_length", 32)))
        save_interval = int(getattr(algo, "save_interval", max(1, total)))
        state = self.env.init_state()
        info: dict = {}
        for it in range(total):
            progress = it / max(total, 1)
            rollout, state = self.collect_rollout(state, horizon)
            self.trainer.step_schedule(progress)
            info = self.trainer.train_op(rollout, self.phase, progress)
            if self.log_dir is not None and (it + 1) % save_interval == 0:
                self.save(self.log_dir / f"model_{it + 1}.pt")
        if self.log_dir is not None:
            self.save(self.log_dir / "checkpoint_final.pt")
        return info

    def save(self, path) -> None:
        save_gh_checkpoint(
            path, policy=self.policy, vecnorm=self.vecnorm, env_state=build_env_state(self.env),
            cfg=None, last_phase=self.phase, meta=meta_from_trainer(self.trainer),
        )

    # --- in-sim evaluate (GH train.py:205-208) --- #
    def evaluate(self, num_steps: int = 20) -> dict:
        """Run the phase's eval rollout policy in-sim and report mean reward."""
        eval_policy = self.policy.get_rollout_policy("eval")
        self.vecnorm.eval()
        state = self.env.init_state()
        rewards = []
        for _ in range(num_steps):
            with torch.no_grad():
                action = eval_policy(self.vecnorm.normalize(self._to_torch(state.obs)))
            state = self.env.step(action.cpu().numpy())
            rewards.append(float(np.mean(state.reward)))
        return {"eval/mean_reward": float(np.mean(rewards)), "eval/steps": num_steps}

    def close(self) -> None:
        self.env.close()
