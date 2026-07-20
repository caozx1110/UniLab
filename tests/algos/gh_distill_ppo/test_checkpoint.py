"""T10.9: GH-schema checkpoint — {wandb,policy,env,cfg,vecnorm}, NO optimizer,
teacher->student hard copy, _meta same-phase-only restore, VecNorm inheritance.
"""
import torch

from unilab.algos.gh_distill_ppo.checkpoint import (
    load_gh_checkpoint,
    meta_from_trainer,
    save_gh_checkpoint,
)
from unilab.algos.gh_distill_ppo.policy import GHDistillPolicy
from unilab.algos.gh_distill_ppo.trainer import GHDistillTrainer
from unilab.algos.gh_distill_ppo.vecnorm import VecNorm

_DIMS = {"policy": 450, "priv": 717, "priv_critic": 3}


def _meta(**kw):
    m = dict(current_lr=3e-4, entropy_coef=0.005, reg_lambda=0.1, progress=0.5,
             num_updates=7, world_size=1)
    m.update(kw)
    return m


def _env_state():
    return {"obs_groups_spec": _DIMS, "action_dim": 29, "num_envs": 4}


def test_outer_keys_and_no_optimizer(tmp_path):
    p, vn = GHDistillPolicy(), VecNorm(_DIMS)
    path = tmp_path / "train_final.pt"
    save_gh_checkpoint(path, policy=p, vecnorm=vn, env_state=_env_state(),
                       cfg={"phase": "train"}, last_phase="train", meta=_meta())
    ckpt = torch.load(path, weights_only=False)

    assert set(ckpt) == {"wandb", "policy", "env", "cfg", "vecnorm"}
    # NO optimizer state anywhere (dump keys)
    assert not any("optim" in str(k).lower() for k in ckpt)
    assert not any("optim" in str(k).lower() for k in ckpt["policy"])
    # policy holds the 5 nets + last_phase + _meta
    assert set(ckpt["policy"]) == {
        "encoder_priv", "adapt_module", "actor_teacher", "actor_student", "critic",
        "last_phase", "_meta"}
    assert set(ckpt["policy"]["_meta"]) == {
        "current_lr", "entropy_coef", "reg_lambda", "progress", "num_updates", "world_size"}


def test_teacher_to_student_hard_copy_on_train_load(tmp_path):
    src = GHDistillPolicy()
    path = tmp_path / "train_final.pt"
    save_gh_checkpoint(path, policy=src, vecnorm=VecNorm(_DIMS), env_state=_env_state(),
                       cfg={}, last_phase="train", meta=_meta())

    dst = GHDistillPolicy()
    # before load, student != teacher (independent orthogonal init)
    st0 = [p.clone() for p in dst.actor_student.parameters()]
    load_gh_checkpoint(path, policy=dst, vecnorm=VecNorm(_DIMS), target_phase="adapt")

    # after loading a 'train' checkpoint: student weights == teacher weights (hard copy)
    for ps, pt in zip(dst.actor_student.parameters(), dst.actor_teacher.parameters()):
        torch.testing.assert_close(ps, pt)
    # and they actually changed from the fresh init
    assert any(not torch.equal(a, b) for a, b in zip(st0, dst.actor_student.parameters()))


def test_no_hard_copy_when_last_phase_not_train(tmp_path):
    src = GHDistillPolicy()
    path = tmp_path / "adapt_final.pt"
    save_gh_checkpoint(path, policy=src, vecnorm=VecNorm(_DIMS), env_state=_env_state(),
                       cfg={}, last_phase="adapt", meta=_meta())
    dst = GHDistillPolicy()
    # make teacher/student clearly different so a copy would be detectable
    with torch.no_grad():
        for p in dst.actor_student.parameters():
            p.add_(1.0)
    stu_before = [p.clone() for p in dst.actor_student.parameters()]
    load_gh_checkpoint(path, policy=dst, vecnorm=VecNorm(_DIMS), target_phase="finetune")
    # loaded from adapt checkpoint -> student overwritten by the SAVED student, NOT hard-copied
    for p_now, p_src in zip(dst.actor_student.parameters(), src.actor_student.parameters()):
        torch.testing.assert_close(p_now, p_src)


def test_meta_restored_only_on_matching_phase(tmp_path):
    p = GHDistillPolicy()
    path = tmp_path / "train_final.pt"
    save_gh_checkpoint(path, policy=p, vecnorm=VecNorm(_DIMS), env_state=_env_state(),
                       cfg={}, last_phase="train", meta=_meta(progress=0.9, num_updates=42))

    same = load_gh_checkpoint(path, policy=GHDistillPolicy(), vecnorm=VecNorm(_DIMS),
                              target_phase="train")
    assert same["meta_restored"] is not None
    assert same["meta_restored"]["progress"] == 0.9 and same["meta_restored"]["num_updates"] == 42

    diff = load_gh_checkpoint(path, policy=GHDistillPolicy(), vecnorm=VecNorm(_DIMS),
                              target_phase="adapt")   # phase changed -> progress resets
    assert diff["meta_restored"] is None


def test_vecnorm_stats_inherited_and_frozen(tmp_path):
    vn = VecNorm(_DIMS)
    vn.update({"policy": torch.ones(16, 450) * 3.0,
               "priv": torch.ones(16, 717), "priv_critic": torch.ones(16, 3)})
    saved_sum = getattr(vn, "_policy__sum").clone()
    path = tmp_path / "train_final.pt"
    save_gh_checkpoint(path, policy=GHDistillPolicy(), vecnorm=vn, env_state=_env_state(),
                       cfg={}, last_phase="train", meta=_meta())

    vn2 = VecNorm(_DIMS)
    vn2.eval()                                  # adapt/finetune load frozen
    load_gh_checkpoint(path, policy=GHDistillPolicy(), vecnorm=vn2, target_phase="adapt")
    torch.testing.assert_close(getattr(vn2, "_policy__sum"), saved_sum)   # inherited
    vn2.update({"policy": torch.zeros(16, 450), "priv": torch.zeros(16, 717),
                "priv_critic": torch.zeros(16, 3)})
    torch.testing.assert_close(getattr(vn2, "_policy__sum"), saved_sum)   # frozen (no change)
