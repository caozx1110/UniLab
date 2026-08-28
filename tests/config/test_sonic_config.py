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


def test_sonic_release_uses_bounded_full_manifest_motion_cache() -> None:
    """The full-data release profile must avoid the benchmark cache ceiling.

    This is intentionally a config-level contract: the loader still enforces
    both limits at runtime, while the benchmark profile remains on its small
    128-entry/512 MiB policy.
    """

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config_release",
            overrides=["task=sonic_g1_tracking/mujoco"],
        )

    params = cfg.env.commands.motion.params
    assert params.motion_cache_size == 4096
    assert params.motion_cache_max_size == 4096
    assert params.motion_cache_max_bytes == 4 * 1024**3


def test_sonic_benchmark_keeps_conservative_motion_cache() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config_benchmark",
            overrides=["task=sonic_g1_tracking/mujoco"],
        )

    params = cfg.env.commands.motion.params
    assert params.motion_cache_size == "auto"
    assert params.motion_cache_max_size == 128
    assert params.motion_cache_max_bytes == 512 * 1024**2
