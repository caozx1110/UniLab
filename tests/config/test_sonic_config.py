from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


CONF_DIR = Path(__file__).parents[2] / "conf" / "sonic"


def test_sonic_release_enables_official_critic_rms() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config_release",
            overrides=["task=sonic_g1_tracking/mujoco"],
        )

    assert cfg.algo.critic_obs_normalization is True
