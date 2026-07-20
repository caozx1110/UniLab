"""Task 9.6: GH tracking owner-config assembly (DENYLIST fields, cross-backend).

Validates the task owner YAMLs directly (OmegaConf.load): DENYLIST fields explicit
and byte-identical across mujoco/motrix (sim2sim contract), task_name/sim_backend set.
The full `algo=gh_distill` Hydra compose + sim2sim wiring are Phase 10 (no gh_distill
algo config group yet), so this asserts the contract artifact, not a full compose.
"""
from pathlib import Path

from omegaconf import OmegaConf

_CONF = Path(__file__).resolve().parents[3] / "conf" / "gh_distill" / "task" / "gh_tracking"


def test_owner_yamls_exist_task_name_and_backend():
    mj = OmegaConf.load(_CONF / "mujoco.yaml")
    mo = OmegaConf.load(_CONF / "motrix.yaml")
    assert mj.training.task_name == "GHTracking"
    assert mo.training.task_name == "GHTracking"
    assert mj.training.sim_backend == "mujoco"
    assert mo.training.sim_backend == "motrix"


def test_denylist_fields_explicit():
    mj = OmegaConf.load(_CONF / "mujoco.yaml")
    assert set(mj.algo.obs_groups.keys()) == {"policy", "priv", "priv_critic"}
    # DENYLIST action_scale at the sim2sim convention path (env.control_config.action_scale)
    assert set(mj.env.control_config.action_scale.keys()) == {
        ".*elbow_joint", ".*shoulder.*", ".*wrist.*", ".*hip_roll.*", ".*hip_yaw.*",
        ".*hip_pitch.*", ".*knee.*", ".*waist.*", ".*ankle.*",
    }
    assert "sampling_mode" in mj.env


def test_denylist_subtree_identical_across_backends():
    """sim2sim contract: DENYLIST fields must be identical on both backends."""
    mj = OmegaConf.load(_CONF / "mujoco.yaml")
    mo = OmegaConf.load(_CONF / "motrix.yaml")
    assert OmegaConf.to_container(mj.algo.obs_groups) == OmegaConf.to_container(mo.algo.obs_groups)
    assert OmegaConf.to_container(mj.env.control_config.action_scale) == OmegaConf.to_container(
        mo.env.control_config.action_scale
    )
    assert mj.env.sampling_mode == mo.env.sampling_mode
