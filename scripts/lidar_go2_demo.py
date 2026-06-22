"""Interactive Go2 LiDAR demo on mujoco-uni 3.8.0.post1's batched raycaster.

This reproduces MuJoCo-LiDAR's ``examples/unitree_go2.py --backend warp``
(interactive MuJoCo window + Go2 walking via an ONNX policy + a mid360 LiDAR
point cloud colored by height), but the LiDAR ranges are produced by
``mujoco.batch_env.BatchEnvPool.multi_ray`` -- the batched ``mj_multiRay``
primitive added in mujoco-uni 3.8.0.post1 (UniLab issue-627) -- instead of the
Warp ray-casting backend.

It is a *validation* demo for the post1 raycaster, and is fully self-contained
inside UniLab: the scene/model XMLs, the ONNX policy and the mid360 scan pattern
are bundled under ``scripts/lidar_demo_assets/`` (all referenced via paths
relative to this file), and the robot meshes are read from UniLab's own
``src/unilab/assets/robots/{go2,g1}/assets`` (referenced relatively from the
bundled model XMLs). It does NOT depend on the MuJoCo-LiDAR project and contains
no absolute paths.

Prereqs: UniLab env synced with mujoco-uni==3.8.0.post1 (see pyproject + uv sync)
and a desktop session (e.g. DISPLAY=:1).

Run (call the venv python directly for a clean exit code):

    ./.venv/bin/python scripts/lidar_go2_demo.py
    ./.venv/bin/python scripts/lidar_go2_demo.py --stand --lidar mid360

`uv run --no-sync python scripts/lidar_go2_demo.py` also works, but in this env
`uv run` triggers a harmless non-zero (120) exit during interpreter shutdown
(an onnxruntime/native-lib finalization quirk under uv run); the demo itself
runs fine. Prefer the direct venv-python form above.

Camera: mouse drag = rotate, scroll = zoom, right-drag = pan.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as rt
from mujoco.batch_env import BatchEnvPool

# NOTE: do NOT `import matplotlib` here. mujoco-uni's bundled native libs and
# matplotlib's segfault when matplotlib is imported after mujoco in this env, so
# the height colormap below is a dependency-free vectorized HSV->RGB instead.

_JOINT_NUM = 12
# Bundled, self-contained demo assets next to this script. Robot meshes are not
# duplicated here: the model XMLs under models/ reference UniLab's own
# src/unilab/assets/robots/{go2,g1}/assets via relative paths.
_ASSETS = Path(__file__).resolve().parent / "lidar_demo_assets"


# ---------------------------------------------------------------------------
# Scan patterns (vendored minimal copy of MuJoCo-LiDAR's scan_gen, so this demo
# does not need to install the `mujoco-lidar` package into UniLab's env, which
# would pull official `mujoco` and clash with `mujoco-uni`).
# ---------------------------------------------------------------------------
class LivoxGenerator:
    """Roll through a precomputed Livox mid360 ray pattern, 24000 rays / frame."""

    def __init__(self, npy_path: Path, samples: int = 24000):
        self.ray_angles = np.load(npy_path.as_posix())  # (N, 2): [:,0]=theta, [:,1]=phi
        self.samples = samples
        self.n_rays = len(self.ray_angles)
        self.curr = 0

    def sample_ray_angles(self) -> tuple[np.ndarray, np.ndarray]:
        if self.curr + self.samples > self.n_rays:
            part1 = self.ray_angles[self.curr :]
            part2 = self.ray_angles[: self.samples - len(part1)]
            out = np.concatenate([part1, part2], axis=0)
            self.curr = self.samples - len(part1)
        else:
            out = self.ray_angles[self.curr : self.curr + self.samples]
            self.curr += self.samples
        return out[:, 0], out[:, 1]


def generate_airy96() -> tuple[np.ndarray, np.ndarray]:
    """Robosense Airy-96 uniform grid pattern (static)."""
    phi = np.deg2rad(np.linspace(0.0, 89.5, 96))
    theta = np.deg2rad(np.linspace(-180.0, 180.0, 900))
    tg, pg = np.meshgrid(theta, phi)
    return tg.flatten(), pg.flatten()


def angles_to_local_dirs(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(theta, phi) sensor-frame angles -> unit direction vectors (nray, 3)."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    return np.stack([cp * ct, cp * st, sp], axis=1).astype(np.float64)


def hsv_rgba(t: np.ndarray) -> np.ndarray:
    """Vectorized matplotlib-'hsv'-like colormap: t in [0,1] -> (N,4) RGBA (S=V=1)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    h = t * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [1, 1 - f, 0, 0, f, 1])
    g = np.select(conds, [f, 1, 1, 1 - f, 0, 0])
    b = np.select(conds, [0, 0, f, 1, 1, 1 - f])
    out = np.ones((t.shape[0], 4), dtype=np.float64)
    out[:, 0], out[:, 1], out[:, 2] = r, g, b
    return out


# ---------------------------------------------------------------------------
# LiDAR backend: mujoco-uni post1 batched multi_ray on a single live MjData.
# ---------------------------------------------------------------------------
class BatchRayLidar:
    """Wrap ``BatchEnvPool.multi_ray`` (nbatch=1) as a LiDAR over a live MjData."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        site_name: str = "lidar",
        bodyexclude_name: str = "base",
        geomgroup: np.ndarray | None = None,
        cutoff: float = 100.0,
        nthread: int = 1,
    ):
        self.model = mj_model
        self.site_id = mj_model.site(site_name).id
        self.bodyexclude = int(mj_model.body(bodyexclude_name).id)
        self.geomgroup = geomgroup
        self.cutoff = float(cutoff)
        self.pool = BatchEnvPool(mj_model, nbatch=1, nthread=nthread)
        self._state = np.zeros(self.pool.nstate, dtype=np.float64)

    def trace(self, mj_data: mujoco.MjData, theta: np.ndarray, phi: np.ndarray):
        """Return (dist[nray], origin[3], world_dirs[nray,3]) for the current pose."""
        mujoco.mj_getState(
            self.model, mj_data, self._state, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        origin = mj_data.site_xpos[self.site_id].astype(np.float64)
        site_rot = mj_data.site_xmat[self.site_id].reshape(3, 3)
        world_dirs = angles_to_local_dirs(theta, phi) @ site_rot.T

        dist, _geomid, _normal = self.pool.multi_ray(
            self._state[None],
            origin,
            world_dirs,
            geomgroup=self.geomgroup,
            bodyexclude=self.bodyexclude,
            cutoff=self.cutoff,
        )
        return dist[0], origin, world_dirs


# ---------------------------------------------------------------------------
# ONNX walking controller (verbatim port of MuJoCo-LiDAR examples/unitree_go2.py).
# ---------------------------------------------------------------------------
class OnnxController:
    """ONNX controller for the Go-2 robot."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        stand: bool = False,
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._counter = 0
        self._n_substeps = n_substeps
        self.stand = stand

    def get_obs(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
        linvel = mj_data.sensor("local_linvel").data
        gyro = mj_data.sensor("gyro").data
        imu_xmat = mj_data.site_xmat[mj_model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = mj_data.qpos[7 : 7 + _JOINT_NUM] - self._default_angles
        joint_velocities = mj_data.qvel[6 : 6 + _JOINT_NUM]

        command = np.zeros(3, dtype=np.float32)
        if not self.stand:
            if mj_data.time % 20.0 < 5.0:
                command[0] = 1.0  # forward
            elif 5.0 < mj_data.time % 20.0 < 10.0:
                command[1] = -1.0
            elif 10.0 < mj_data.time % 20.0 < 15.0:
                command[0] = -1.0
            else:
                command[1] = 1.0
        obs = np.hstack(
            [linvel, gyro, gravity, joint_angles, joint_velocities, self._last_action, command]
        )
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_pred = self._policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Go2 LiDAR demo using mujoco-uni post1 BatchEnvPool.multi_ray"
    )
    parser.add_argument("--lidar", choices=["airy", "mid360"], default="mid360")
    parser.add_argument("--stand", action="store_true", help="hold still instead of cruising")
    parser.add_argument("--cutoff", type=float, default=100.0, help="max ray distance (m)")
    args = parser.parse_args()

    scene_path = _ASSETS / "models" / "scene_go2.xml"
    policy_path = _ASSETS / "onnx" / "go2_policy.onnx"
    mid360_npy = _ASSETS / "scan_mode" / "mid360.npy"
    for p in (scene_path, policy_path):
        if not p.is_file():
            raise SystemExit(f"Missing bundled demo asset: {p}")

    mj_model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    ctrl_dt = 0.02
    lidar_dt = 1.0 / 10.0
    mj_model.opt.timestep = 0.004

    policy = OnnxController(
        policy_path=policy_path.as_posix(),
        default_angles=np.array(mj_model.key_qpos[0, 7 : 7 + _JOINT_NUM]),
        n_substeps=int(round(ctrl_dt / mj_model.opt.timestep)),
        action_scale=0.5,
        stand=args.stand,
    )

    # Scan pattern (mid360 rotates frame-by-frame; airy is static).
    if args.lidar == "mid360":
        if not mid360_npy.is_file():
            raise SystemExit(f"Missing mid360 pattern: {mid360_npy}")
        livox = LivoxGenerator(mid360_npy)
        rays_theta, rays_phi = livox.sample_ray_angles()
        dynamic_lidar = True
    else:
        livox = None
        rays_theta, rays_phi = generate_airy96()
        dynamic_lidar = False
    rays_theta = np.ascontiguousarray(rays_theta, dtype=np.float64)
    rays_phi = np.ascontiguousarray(rays_phi, dtype=np.float64)

    # geomgroup: keep groups 0..2 (robot + scene), drop 3+ (matches the warp demo).
    geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.uint8)
    geomgroup[3:] = 0
    lidar = BatchRayLidar(
        mj_model, site_name="lidar", bodyexclude_name="base", geomgroup=geomgroup, cutoff=args.cutoff
    )

    n_rays = rays_theta.shape[0]

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.user_scn.ngeom = n_rays
        for i in range(n_rays):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8]),
            )
        print("Starting simulation (LiDAR backend: mujoco-uni post1 multi_ray)...")
        print("Number of rays:", n_rays)

        _last_time = 1e6
        n_substeps = int(round(lidar_dt / mj_model.opt.timestep))
        _start_time = time.time()
        _counter = 0
        while viewer.is_running():
            if mj_data.time < _last_time:
                _counter = 0
                _start_time = time.time()
            _last_time = mj_data.time

            mujoco.mj_step(mj_model, mj_data)
            policy.get_control(mj_model, mj_data)

            _counter += 1
            if _counter % n_substeps == 0:
                if dynamic_lidar:
                    rays_theta, rays_phi = livox.sample_ray_angles()
                    rays_theta = np.ascontiguousarray(rays_theta, dtype=np.float64)
                    rays_phi = np.ascontiguousarray(rays_phi, dtype=np.float64)

                dist, origin, world_dirs = lidar.trace(mj_data, rays_theta, rays_phi)
                hit = dist >= 0.0
                world_points = origin[None] + dist[:, None] * world_dirs

                if hit.any():
                    z = world_points[:, 2]
                    z_min, z_max = float(z[hit].min()), float(z[hit].max())
                    rng = z_max - z_min
                    z_norm = (z_max - z) / rng if rng > 0 else np.zeros_like(z)
                    colors = hsv_rgba(z_norm)  # (n_rays, 4)
                    for i in range(n_rays):
                        geom = viewer.user_scn.geoms[i]
                        if hit[i]:
                            geom.pos[:] = world_points[i]
                            geom.rgba[:] = colors[i]
                        else:
                            geom.rgba[3] = 0.0  # hide missed rays

            viewer.sync()
            run_time = time.time() - _start_time
            if run_time < mj_data.time:
                time.sleep(mj_data.time - run_time)

    print("Done.")


if __name__ == "__main__":
    main()
