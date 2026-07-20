"""Generate the GH-specific G1 robot MJCF (``g1_gh/robot.xml``) from the shared
``g1/g1.xml`` via MjSpec. Reproducible cold-path asset build.

Derives (WITHOUT modifying the shared ``g1/g1.xml``):
  - 3 mimic welded marker bodies (USD-extracted offsets; massless, no collision)
  - per-joint armature + zero joint friction (humanoid.py)
  - full self-collision disable (collision geoms ``conaffinity=0``)
  - full-precision humanoid.py PD gains on the position actuators
  - per-actuator ``actfrc_<actuator>`` sensors for the Phase-1 backend contract

Run:  uv run python src/unilab/assets/robots/g1_gh/build_robot_xml.py
"""

from __future__ import annotations

import os

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_G1 = os.path.normpath(os.path.join(_HERE, "..", "g1", "g1.xml"))
_OUT = os.path.join(_HERE, "robot.xml")

# mimic markers: name -> (parent body, localPos0 in parent frame)  [USD 5.1]
MIMIC = {
    "head_mimic": ("torso_link", (0.0100035, 0.0, 0.41)),
    "left_hand_mimic": ("left_wrist_yaw_link", (0.115996, 0.0, 0.0)),
    "right_hand_mimic": ("right_wrist_yaw_link", (0.115996, 0.0, 0.0)),
}
MIMIC_MASS = 0.005
MIMIC_INERTIA = (1e-9, 1e-9, 1e-9)  # USD diagonalInertia=0; welded -> merges into parent


def joint_params(j: str) -> tuple[float, float, float, float]:
    """(kp, kv, armature, effort) per joint, transcribed from humanoid.py."""
    if "hip_roll" in j or "knee" in j:
        return (99.09842777666113, 6.3088018534966395, 0.025101925, 139.0)
    if "hip_yaw" in j or "hip_pitch" in j or "waist_yaw" in j:
        return (40.17923847137318, 2.5578897650279457, 0.010177520, 88.0)
    if "waist_roll" in j or "waist_pitch" in j or "ankle" in j:
        return (28.50124619574858, 1.814445686584846, 0.00721945, 50.0)
    if "wrist_pitch" in j or "wrist_yaw" in j:
        return (16.77832748089279, 1.06814150219, 0.00425, 5.0)
    # shoulder_pitch/roll/yaw, elbow, wrist_roll
    return (14.25062309787429, 0.907222843292423, 0.003609725, 25.0)


def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(_G1)
    spec.modelname = "g1_29dof_gh"
    spec.meshdir = "../g1/assets"  # reuse shared meshes; resolves relative to robot.xml

    # per-joint armature + zero friction (skip the free joint)
    for j in spec.joints:
        if j.name and j.name != "floating_base_joint":
            _, _, arm, _ = joint_params(j.name)
            j.armature = arm
            j.frictionloss = 0.0

    # full-precision PD gains on the position actuators (bias = -kp*q - kv*qvel)
    for a in spec.actuators:
        kp, kv, _, fr = joint_params(a.name)
        gp = list(a.gainprm)
        gp[0] = kp
        a.gainprm = gp
        bp = list(a.biasprm)
        bp[1] = -kp
        bp[2] = -kv
        a.biasprm = bp
        a.forcerange = [-fr, fr]

    # full self-collision disable: every collidable geom conaffinity=0
    # (robot-robot never collides; ground contact via ground.conaffinity=1 stays)
    for g in spec.geoms:
        if int(g.contype) == 1:
            g.conaffinity = 0

    # 3 mimic welded marker bodies (no joint -> welded; no geom -> no collision)
    for name, (parent, pos) in MIMIC.items():
        b = spec.body(parent).add_body()
        b.name = name
        b.pos = list(pos)
        b.quat = [1.0, 0.0, 0.0, 0.0]
        b.mass = MIMIC_MASS
        b.inertia = list(MIMIC_INERTIA)
        b.explicitinertial = True  # force <inertial> into to_xml (else marker is dropped)

    # per-actuator effort sensors (Phase-1 contract: actfrc_<actuator>)
    for a in list(spec.actuators):
        s = spec.add_sensor()
        s.name = f"actfrc_{a.name}"
        s.type = mujoco.mjtSensor.mjSENS_ACTUATORFRC
        s.objtype = mujoco.mjtObj.mjOBJ_ACTUATOR
        s.objname = a.name

    spec.compile()  # validate before writing
    return spec


def main() -> None:
    spec = build()
    with open(_OUT, "w") as f:
        f.write(spec.to_xml())
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
