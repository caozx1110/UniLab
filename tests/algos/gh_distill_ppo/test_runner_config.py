"""T10.10: 3-phase runner + Hydra config.

Config composition asserts the per-phase override group (§一:47-52). The runner smoke
builds the live GHTrackingEnv + policy/vecnorm/trainer, runs one rollout + train_op, and
round-trips a GH-schema checkpoint (no optimizer).
"""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from unilab.envs.gh_tracking.motion_dataset import write_synthetic_dataset

_CONF = str(Path(__file__).resolve().parents[3] / "conf" / "gh_distill")


def test_config_composes_per_phase():
    expected = {
        "train": ("train", False, 32),
        "adapt": ("eval", True, 16),
        "finetune": ("eval", True, 32),
    }
    with initialize_config_dir(config_dir=_CONF, version_base="1.3"):
        for phase, (vecnorm, student_train, train_every) in expected.items():
            cfg = compose(config_name="config",
                          overrides=[f"phase={phase}", "task=gh_tracking/mujoco"])
            assert cfg.training.task_name == "GHTracking"
            assert cfg.algo.phase == phase
            assert cfg.algo.vecnorm == vecnorm
            assert bool(cfg.algo.student_train) is student_train
            assert int(cfg.algo.train_every) == train_every
        # finetune-specific knobs (GH cfg/exp/finetune.yaml)
        ft = compose(config_name="config", overrides=["phase=finetune", "task=gh_tracking/mujoco"])
        assert float(ft.algo.lr) == 1e-4
        assert float(ft.algo.entropy_start) == 0.002 and float(ft.algo.entropy_end) == 0.0005


def _cfg(phase, num_envs, tmp_path):
    with initialize_config_dir(config_dir=_CONF, version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=[f"phase={phase}", "task=gh_tracking/mujoco",
                                 f"algo.num_envs={num_envs}", "algo.num_minibatches=2",
                                 "algo.ppo_epochs=1", "algo.estimator_epochs=1"])
    return cfg


def test_runner_smoke_train_iteration_and_checkpoint(tmp_path):
    from unilab.algos.gh_distill_ppo.runner import GHDistillRunner

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120, 200], seed=0)
    cfg = _cfg("train", 4, tmp_path)
    override = {"motion": {"dirs": [str(tmp_path / "interx")], "weights": [1.0]}}

    runner = GHDistillRunner(cfg, device="cpu", env_cfg_override=override)
    assert runner.phase == "train" and runner.vecnorm.training  # vecnorm=train -> updating

    info = runner.learn(num_iterations=1)                         # rollout + train_op
    assert "kl" in info                                          # train phase ran a PPO update

    ckpt = tmp_path / "train_final.pt"
    runner.save(ckpt)
    blob = torch.load(ckpt, weights_only=False)
    assert set(blob) == {"wandb", "policy", "env", "cfg", "vecnorm"}
    assert not any("optim" in str(k).lower() for k in blob["policy"])
    assert blob["policy"]["last_phase"] == "train"
    runner.close()


def test_adapt_runner_loads_train_ckpt_and_freezes_vecnorm(tmp_path):
    from unilab.algos.gh_distill_ppo.runner import GHDistillRunner

    write_synthetic_dataset(str(tmp_path / "interx"), clip_lengths=[120, 200], seed=1)
    override = {"motion": {"dirs": [str(tmp_path / "interx")], "weights": [1.0]}}

    train = GHDistillRunner(_cfg("train", 4, tmp_path), device="cpu", env_cfg_override=override)
    train.learn(num_iterations=1)
    ckpt = tmp_path / "train_final.pt"
    train.save(ckpt)
    teacher_ref = [p.clone() for p in train.policy.actor_teacher.parameters()]
    train.close()

    cfg = _cfg("adapt", 4, tmp_path)
    cfg.algo.checkpoint_path = str(ckpt)
    adapt = GHDistillRunner(cfg, device="cpu", env_cfg_override=override)
    assert not adapt.vecnorm.training                             # adapt -> frozen vecnorm
    # teacher->student hard copy on loading the train checkpoint
    for ps, pt in zip(adapt.policy.actor_student.parameters(), teacher_ref):
        torch.testing.assert_close(ps, pt)
    adapt.close()
