"""Play back a GHDistill checkpoint in MuJoCo viewer.

Loads gh checkpoint (policy + vecnorm + last_phase), constructs GHTrackingEnv,
runs eval rollout with the phase-appropriate policy (train→teacher, adapt/finetune→student),
and visualizes in mujoco.viewer (4 robots side-by-side by default).

Usage:
    uv run scripts/play_gh_mujoco.py --checkpoint <path> [--num_envs 4] [--phase auto]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco.viewer

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training.common import ensure_registries
from unilab.algos.gh_distill_ppo.runner import GHDistillRunner
from unilab.algos.gh_distill_ppo.checkpoint import load_gh_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to gh checkpoint (.pt)")
    p.add_argument("--num_envs", type=int, default=4, help="Number of robots to visualize (default 4)")
    p.add_argument(
        "--phase",
        choices=["auto", "train", "adapt", "finetune"],
        default="auto",
        help="Which policy to use. 'auto' (default) uses checkpoint's last_phase: train→teacher, adapt/finetune→student.",
    )
    p.add_argument("--steps", type=int, default=None, help="Max steps to run (default: infinite loop)")
    return p.parse_args()


def _mujoco_visual_xml_paths(env):
    """Infer parent scene + robot base XMLs from env.cfg.scene/asset (reuse visualize_task_env logic)."""
    from pathlib import Path
    scene_xml = Path(env.cfg.scene.model_file)
    # Assume robot base is in same dir; g1_gh uses scene_flat.xml + g1_gh/robot.xml sibling
    robot_xml = scene_xml.parent / "robot.xml"
    if not robot_xml.exists():
        robot_xml = scene_xml  # fallback: use scene itself as both
    return scene_xml, robot_xml


def _stitch_replicas(parent_scene_xml: Path, robot_base_xml: Path, env_origins: np.ndarray):
    """Attach (N-1) extra robot instances at env_origins (reuse visualize_task_env stitching)."""
    import xml.etree.ElementTree as ET
    num_envs = len(env_origins)
    parent_tree = ET.parse(parent_scene_xml)
    parent_root = parent_tree.getroot()
    worldbody = parent_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"scene XML {parent_scene_xml} has no <worldbody>")

    robot_tree = ET.parse(robot_base_xml)
    robot_body = robot_tree.getroot().find("worldbody").find("body")
    if robot_body is None:
        raise ValueError(f"robot XML {robot_base_xml} has no <worldbody><body>")

    # First replica is already in parent_scene; add N-1 more at offsets
    for i in range(1, num_envs):
        replica = ET.fromstring(ET.tostring(robot_body))
        replica.set("name", f"{robot_body.get('name', 'robot')}_{i}")
        pos = env_origins[i]
        replica.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        worldbody.append(replica)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        parent_tree.write(f, encoding="unicode")
        stitched_path = f.name
    return mujoco.MjModel.from_xml_path(stitched_path)


def main():
    args = parse_args()
    ensure_registries()

    # --- Load checkpoint metadata to get last_phase ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    last_phase = ckpt["policy"]["last_phase"]
    print(f"[play_gh_mujoco] Checkpoint last_phase: {last_phase}")

    # Decide which policy to use for rollout
    if args.phase == "auto":
        # train→teacher (adapt_module→actor_teacher), adapt/finetune→student (adapt_module→actor_student)
        # The GHDistillPolicy.get_rollout_policy("eval") always uses student; for train we want teacher.
        rollout_phase = last_phase if last_phase == "train" else "eval"
    else:
        rollout_phase = "eval" if args.phase != "train" else "train"

    print(f"[play_gh_mujoco] Rollout phase: {rollout_phase}")
    if last_phase == "train" and rollout_phase != "train":
        print("[WARNING] train-phase checkpoint but using student policy — student may be untrained!")

    # --- Construct runner (env + policy + vecnorm) ---
    with initialize_config_dir(version_base="1.3", config_dir="/home/a/ws/unilabsim/UniLab/conf/gh_distill"):
        cfg = compose(
            config_name="config",
            overrides=[f"phase={last_phase}", f"algo.num_envs={args.num_envs}"],
        )
    OmegaConf.set_struct(cfg, False)

    runner = GHDistillRunner(cfg, device="cpu", log_dir=None)  # CPU for simplicity; can be cuda
    load_gh_checkpoint(args.checkpoint, runner.policy, runner.vecnorm, target_phase=last_phase, strict=True)
    runner.policy.eval()

    eval_policy = runner.policy.get_rollout_policy(rollout_phase)
    env = runner.env
    num_envs = args.num_envs
    ctrl_dt = float(env.cfg.ctrl_dt)

    print(f"[play_gh_mujoco] Env: {num_envs} envs, ctrl_dt={ctrl_dt}s")

    # --- Set up MuJoCo viewer (reuse visualize_task_env stitching) ---
    parent_xml, robot_xml = _mujoco_visual_xml_paths(env)
    env_origins = env._spawn.origins_for(np.arange(num_envs))

    if num_envs > 1 and not env_origins[:, :2].any():
        print(
            f"[play_gh_mujoco] NOTE: all {num_envs} robots will overlap at origin "
            "(env has no terrain spawn offset)."
        )

    decoder_model = mujoco.MjModel.from_xml_path(str(parent_xml))
    decoder_data = mujoco.MjData(decoder_model)
    nq_per = int(decoder_model.nq)
    nv_per = int(decoder_model.nv)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    if num_envs == 1:
        viz_model = decoder_model
    else:
        viz_model = _stitch_replicas(parent_xml, robot_xml, env_origins)
    viz_data = mujoco.MjData(viz_model)

    print("[play_gh_mujoco] Initializing env...")
    state = env.init_state()

    print("[play_gh_mujoco] Opening MuJoCo viewer — close window or press Esc to quit.")
    step_count = 0
    max_steps = args.steps if args.steps else float("inf")

    with mujoco.viewer.launch_passive(viz_model, viz_data) as viewer:
        while viewer.is_running() and step_count < max_steps:
            t0 = time.perf_counter()

            # Get action from eval policy (deterministic for eval/student, sampled for train/teacher)
            obs_torch = runner._to_torch(state.obs)
            obs_norm = runner.vecnorm.normalize(obs_torch)
            with torch.no_grad():
                action_torch = eval_policy(obs_norm)
            action = action_torch.cpu().numpy()

            # Step env
            state = env.step(action)
            step_count += 1

            # Update viewer qpos/qvel from env physics state
            phys = env.get_physics_state_snapshot()
            for i in range(num_envs):
                mujoco.mj_setState(decoder_model, decoder_data, phys[i].astype(np.float64), state_spec)
                viz_data.qpos[i * nq_per : (i + 1) * nq_per] = decoder_data.qpos
                viz_data.qvel[i * nv_per : (i + 1) * nv_per] = decoder_data.qvel

            mujoco.mj_forward(viz_model, viz_data)
            viewer.sync()

            # Maintain ctrl_dt pacing
            elapsed = time.perf_counter() - t0
            sleep = ctrl_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    print(f"[play_gh_mujoco] Ran {step_count} steps. Closing.")
    runner.close()


if __name__ == "__main__":
    main()
