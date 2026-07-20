"""Test GHTrackingEnv backend construction with add_body_sensors=True."""
import numpy as np

# Foot bodies carry the netcontact_/contactfound_ sensors (Phase 2) and are
# among the add_body_sensors tracking keypoints, so all three force-system
# queries resolve on them. The world body (id 0) has none of these sensors.
_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def test_backend_constructed_with_add_body_sensors_true():
    """Backend must enable body sensors for force-system contact/pose queries (Phase 5/6)."""
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    from unilab.envs.gh_tracking.config import GHTrackingCfg

    cfg = GHTrackingCfg()
    env = GHTrackingEnv(cfg, num_envs=2, backend_type="mujoco")

    foot_ids = env._backend.get_body_ids(_FEET)

    # add_body_sensors=True enables body-pose tracking sensors (Phase 2 记账:
    # get_body_pos_w requires them). If it were False this query would raise.
    body_pos = env._backend.get_body_pos_w(foot_ids)
    assert body_pos.shape == (2, 2, 3), "body-pose tracking sensors must be queryable"

    # Phase 1 真名: net-contact force + contact-found on the feet (Phase 5 force system).
    forces = env._backend.get_body_net_contact_force_w(foot_ids)
    contact_state = env._backend.get_body_contact_state(foot_ids)
    assert forces.shape == (2, 2, 3), "foot net contact force should be queryable"
    assert contact_state.shape == (2, 2), "foot contact-found state should be queryable"

    env.close()


def test_obs_groups_spec_is_three_groups():
    """obs_groups_spec must be {"policy":450, "priv":717, "priv_critic":3} (DENYLIST field)."""
    from unilab.envs.gh_tracking.env import GHTrackingEnv
    from unilab.envs.gh_tracking.config import GHTrackingCfg

    cfg = GHTrackingCfg()
    env = GHTrackingEnv(cfg, num_envs=2, backend_type="mujoco")

    spec = env.obs_groups_spec
    assert set(spec.keys()) == {"policy", "priv", "priv_critic"}
    assert spec["policy"] == 450
    assert spec["priv"] == 717
    assert spec["priv_critic"] == 3

    env.close()
