"""T10.11: end-to-end train->adapt->finetune closed loop (synthetic fixture).

Verifies PIPELINE CORRECTNESS (= migration correctness) — all green:
- 3-phase chaining with checkpoint auto-inheritance (T10.9/T10.10),
- teacher->student hard copy on train->adapt,
- VecNorm train-learn / adapt+finetune inherit+freeze,
- per-phase trainable sets (T10.8) through the live runner,
- finetune checkpoint -> get_rollout_policy("eval") produces an in-sim evaluate number,
- obs/reward/dim/contract integrity + the reward_vec.sum()==state.reward wiring.

TRAINING CONVERGENCE (policy learning GH motion, reward curve rising) is ⏳ pending real
retargeted data — a synthetic fixture exercises the pipeline, not learning quality.
"""
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset

_CONF = str(Path(__file__).resolve().parents[3] / "conf" / "gh_distill")


def _multi_dataset(tmp_path):
    # multi-dataset weighted fixture (interx/lafan/amass names, GH 0.4/0.2/0.4 weights)
    for nm, seed in (("interx", 0), ("lafan", 1), ("amass", 2)):
        write_synthetic_dataset(str(tmp_path / nm), clip_lengths=[120, 200], seed=seed)
    dirs = [str(tmp_path / nm) for nm in ("interx", "lafan", "amass")]
    return {"motion": {"dirs": dirs, "weights": [0.4, 0.2, 0.4]}}


def _cfg(phase):
    with initialize_config_dir(config_dir=_CONF, version_base="1.3"):
        return compose(config_name="config", overrides=[
            f"phase={phase}", "task=gh_tracking/mujoco",
            "algo.num_envs=4", "algo.num_minibatches=2", "algo.ppo_epochs=1",
            "algo.estimator_epochs=1", "algo.train_every=8"])


def _grad_set(policy):
    names = ("encoder_priv", "adapt_module", "actor_teacher", "actor_student", "critic")
    return {n for n in names
            if any(p.grad is not None and torch.any(p.grad != 0) for p in getattr(policy, n).parameters())}


def test_reward_vec_sums_to_scalar_reward(tmp_path):
    """Last accounting item: env 3-group reward_vec.sum(-1) == state.reward (trainer's GAE input)."""
    from unilab.envs.gh_tracking.config import GHTrackingCfg
    from unilab.envs.gh_tracking.env import GHTrackingEnv

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120], seed=0)
    cfg = GHTrackingCfg()
    cfg.motion.dirs = [str(tmp_path / "interx")]
    cfg.motion.weights = [1.0]
    env = GHTrackingEnv(cfg, num_envs=4, backend_type="mujoco")
    env.init_state()
    state = env.step(np.zeros((4, 29)))
    np.testing.assert_allclose(state.info["reward_vec"].sum(axis=-1), state.reward, rtol=0, atol=1e-6)
    env.close()


def test_end_to_end_three_phase_chain_and_evaluate(tmp_path):
    from unilab.algos.gh_distill_ppo.runner import GHDistillRunner

    override = _multi_dataset(tmp_path)

    # --- phase 1: train (vecnorm learns; teacher PPO + estimator) ---
    train = GHDistillRunner(_cfg("train"), device="cpu", env_cfg_override=override)
    assert train.vecnorm.training                              # online decay-EMA
    train.learn(num_iterations=2)
    assert _grad_set(train.policy) == {"encoder_priv", "actor_teacher", "critic", "adapt_module"}
    vecnorm_count = float(getattr(train.vecnorm, "_policy__count"))
    assert vecnorm_count > 0                                   # vecnorm actually updated
    teacher_ref = [p.clone() for p in train.policy.actor_teacher.parameters()]
    train_ckpt = tmp_path / "train_final.pt"
    train.save(train_ckpt)
    train.close()

    # --- phase 2: adapt (inherit train ckpt; teacher->student hard copy; vecnorm frozen) ---
    adapt_cfg = _cfg("adapt")
    adapt_cfg.algo.checkpoint_path = str(train_ckpt)
    adapt = GHDistillRunner(adapt_cfg, device="cpu", env_cfg_override=override)
    assert not adapt.vecnorm.training                          # frozen
    torch.testing.assert_close(float(getattr(adapt.vecnorm, "_policy__count")), vecnorm_count)  # inherited
    for ps, pt in zip(adapt.policy.actor_student.parameters(), teacher_ref):
        torch.testing.assert_close(ps, pt)                    # hard copy
    adapt.learn(num_iterations=2)
    assert _grad_set(adapt.policy) == {"adapt_module"}         # estimator only
    adapt_ckpt = tmp_path / "adapt_final.pt"
    adapt.save(adapt_ckpt)
    adapt.close()

    # --- phase 3: finetune (inherit adapt ckpt; student PPO; NOT train ckpt so no re-copy) ---
    ft_cfg = _cfg("finetune")
    ft_cfg.algo.checkpoint_path = str(adapt_ckpt)
    ft = GHDistillRunner(ft_cfg, device="cpu", env_cfg_override=override)
    assert not ft.vecnorm.training
    ft.learn(num_iterations=2)
    assert _grad_set(ft.policy) == {"actor_student", "adapt_module", "critic"}
    ft_ckpt = tmp_path / "finetune_final.pt"
    ft.save(ft_ckpt)

    # --- in-sim evaluate from the finetune policy (GH train.py:205-208) ---
    metrics = ft.evaluate(num_steps=10)
    assert "eval/mean_reward" in metrics and np.isfinite(metrics["eval/mean_reward"])
    assert metrics["eval/steps"] == 10
    ft.close()


def test_finetune_checkpoint_has_gh_schema_no_optimizer(tmp_path):
    from unilab.algos.gh_distill_ppo.runner import GHDistillRunner

    override = _multi_dataset(tmp_path)
    ft = GHDistillRunner(_cfg("finetune"), device="cpu", env_cfg_override=override)
    ft.learn(num_iterations=1)
    path = tmp_path / "ft.pt"
    ft.save(path)
    blob = torch.load(path, weights_only=False)
    assert set(blob) == {"wandb", "policy", "env", "cfg", "vecnorm"}
    assert not any("optim" in str(k).lower() for k in blob) and not any(
        "optim" in str(k).lower() for k in blob["policy"])
    assert blob["env"]["obs_groups_spec"] == {"policy": 450, "priv": 717, "priv_critic": 3}
    ft.close()
