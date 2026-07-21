"""GH tracking environment configuration.

Owner-layer config for the live GHTrackingEnv (Phase 9). Numeric defaults are the
blueprint/USD-confirmed values where pinned (GH_to_UniLab_MuJoCo_migration.md §五/§八);
values marked ``# TODO(parity)`` are structural placeholders whose exact GH value is
pending real-data / deeper-source calibration (DP2 synthetic-fixture regime).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg


@dataclass
class GHAssetCfg:
    """Asset identity fields consumed at backend construction (cold path).

    ``base_name`` names the floating-base body used by the backend for
    baselink pose/velocity sensors (e.g. ``get_base_ang_vel_world``). g1_gh's
    root body is ``pelvis`` (assets/robots/g1_gh/robot.xml), matching the G1
    convention (envs/locomotion/g1/base.py Asset.base_name).
    """

    base_name: str = "pelvis"
    keyframe_name: str = "stand"  # scene_flat.xml <key name="stand"> (default pose + ctrl)


@dataclass
class GHMotionCfg:
    """Weighted multi-dataset motion sampling (Phase 3 WeightedMotionDataset).

    ``dirs`` defaults empty: with no real data on disk yet (DP2) the env skips
    motion-dataset construction so it still builds for unit tests; the owner YAML
    (Task 9.6) and data-backed tests set real/synthetic dataset dirs + weights.
    """

    dirs: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    max_step: int = 1000
    sample_once: bool = True
    zero_init_prob: float = 0.0  # TODO(parity): GH sample_init zero-start probability
    seed: int = 0


@dataclass
class GHActionCfg:
    """29-dim residual action pipeline (Phase 4 GHActionPipeline). G1_gentle.yaml:23-27."""

    max_delay: int = 4
    alpha_range: tuple[float, float] = (0.9, 0.9)
    alpha_wide: tuple[float, float] = (0.8, 1.0)
    alpha_jit_scale: float = 0.025
    boot_protect: bool = True


@dataclass
class GHControlConfigCfg:
    """DENYLIST (sim2sim ``env.control_config.action_scale``): GH per-joint residual
    action scaling. Exact from G1_gentle.yaml:13-22 (re.fullmatch regex -> per-joint
    scale; every joint matches exactly one pattern)."""

    action_scale: dict = field(default_factory=lambda: {
        ".*elbow_joint": 1.0,
        ".*shoulder.*": 1.0,
        ".*wrist.*": 1.0,
        ".*hip_roll.*": 0.25,
        ".*hip_yaw.*": 0.25,
        ".*hip_pitch.*": 0.5,
        ".*knee.*": 0.5,
        ".*waist.*": 0.25,
        ".*ankle.*": 0.5,
    })


@dataclass
class GHInitNoiseCfg:
    """D2 reset init-state noise, exact from G1_gentle.yaml:48-54. Applied per field
    as ``randn().clamp(-1,1) * value`` (root_pos z-noise clamped >= 0)."""

    root_pos: float = 0.03
    root_ori: float = 0.1
    root_lin_vel: float = 0.1
    root_ang_vel: float = 0.1
    joint_pos: float = 0.1
    joint_vel: float = 0.1


@dataclass
class GHRewardCfg:
    """Static reward constants from G1_gentle.yaml (per-term weights live in env
    _REWARD_GROUPS; per-term sigmas in env _REWARD_SIGMA)."""

    joint_pos_limits_soft_factor: float = 0.9  # :134
    feet_air_time_thres: float = 0.8           # :133


@dataclass
class GHForceCfg:
    """Compliant external-force system (Phase 5 ForceSystem). Blueprint §五.

    ``compliance`` selects the GH force variant (motion_tracking.py:1045-1051):
    True = gentle admittance force (default); False+max_force>0 = extreme random
    perturbation; False+max_force<=0 = no_force (pure action tracking)."""

    num_force_bodies: int = 6
    max_force: float = 30.0
    net_force_limit: float = 30.0
    net_torque_limit: float = 20.0
    force_alpha: float = 1.0
    compliance: bool = True
    seed: int = 0


@dataclass
class GHTerminationCfg:
    """cum-error termination (Phase 7 CumErrorTermination). Blueprint §八."""

    thres: float = 1.0
    min_steps: int = 50  # strict > -> 51 consecutive over-threshold steps
    max_episode_length: int = 1000  # GH timeout (episode_length >= 1000)


@dataclass
class GHObsCfg:
    """Observation-manager telemetry parameters (Phase 6 ObservationManager)."""

    boot_indicator_max: int = 25
    seed: int = 0


@dataclass
class GHDRCfg:
    """GH domain randomization (Phase 7 GHDomainRand). Ranges live in the component."""

    seed: int = 0


@registry.envcfg("GHTracking")
@dataclass
class GHTrackingCfg(EnvCfg):
    """GH tracking environment configuration."""

    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "g1_gh" / "scene_flat.xml")
    ))
    asset: GHAssetCfg = field(default_factory=GHAssetCfg)
    sampling_mode: str = "sample_once"  # DENYLIST (sim2sim): motion episode sampling mode
    student_train: bool = False  # train->teacher(current frame); adapt/finetune->student 50-cache
    motion: GHMotionCfg = field(default_factory=GHMotionCfg)
    action: GHActionCfg = field(default_factory=GHActionCfg)
    control_config: GHControlConfigCfg = field(default_factory=GHControlConfigCfg)
    init_noise: GHInitNoiseCfg = field(default_factory=GHInitNoiseCfg)
    force: GHForceCfg = field(default_factory=GHForceCfg)
    termination: GHTerminationCfg = field(default_factory=GHTerminationCfg)
    reward: GHRewardCfg = field(default_factory=GHRewardCfg)
    obs: GHObsCfg = field(default_factory=GHObsCfg)
    domain_rand: GHDRCfg = field(default_factory=GHDRCfg)
    sim_dt: float = 0.02 / 4.0  # 200 Hz sim (4 substeps × 50 Hz ctrl)
    ctrl_dt: float = 0.02  # 50 Hz control
    numba_acceleration: bool = False
    numba_num_threads: int | None = None
