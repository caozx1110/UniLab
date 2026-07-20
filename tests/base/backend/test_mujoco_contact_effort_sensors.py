"""Real-MuJoCo analytical tests for GH-migration backend sensor reads.

- 1.4 actuator effort (post-saturation) via ``actuatorfrc`` sensors.
- 1.3 per-body NET EXTERNAL contact force + contact state via ``contact``
  (netforce) sensors, validated against analytical values (a resting body's
  contact force must equal m*g and oppose gravity), NOT parent-child wrench.

Sensors follow the naming convention the GH MJCF (Phase 2) will author:
``actfrc_<actuator>``, ``netcontact_<body>``, ``contactfound_<body>``.
"""

from __future__ import annotations

import numpy as np

from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg


def _backend(tmp_path, xml: str, base_name, num_envs: int = 2):
    p = tmp_path / "model.xml"
    p.write_text(xml)
    bk = MuJoCoBackend(SceneCfg(model_file=str(p)), num_envs, 0.005, base_name=base_name)
    bk.materialize()
    return bk


# --------------------------------------------------------------------------- #
# 1.4 actuator effort readback                                                #
# --------------------------------------------------------------------------- #

_ACT_XML = """<mujoco><option gravity="0 0 0"/>
<worldbody><body name="link"><joint name="j" type="hinge" axis="0 0 1"/>
<geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.1"/></body></worldbody>
<actuator><position name="a" joint="j" kp="100" forcerange="-1 1"/></actuator>
<sensor><actuatorfrc name="actfrc_a" actuator="a"/></sensor></mujoco>"""


def test_actuator_effort_is_post_saturation(tmp_path) -> None:
    bk = _backend(tmp_path, _ACT_XML, base_name=None)
    for _ in range(10):
        bk.step(np.full((2, 1), 10.0, dtype=np.float32), nsteps=1)  # saturating target

    eff = bk.get_actuator_effort()

    assert eff.shape == (2, 1)
    np.testing.assert_allclose(np.abs(eff), 1.0, atol=1e-3)  # clamped to forcerange
    assert not np.allclose(eff, 10.0)  # actual effort != commanded target


def test_actuator_effort_sign_follows_target(tmp_path) -> None:
    bk = _backend(tmp_path, _ACT_XML, base_name=None)
    for _ in range(10):
        bk.step(np.full((2, 1), -10.0, dtype=np.float32), nsteps=1)

    eff = bk.get_actuator_effort()

    np.testing.assert_allclose(eff, -1.0, atol=1e-3)  # negative saturation


# --------------------------------------------------------------------------- #
# 1.3 net external contact force + contact state (analytical, R5)             #
# --------------------------------------------------------------------------- #

_M, _G = 2.0, 9.81


def _box_xml(m: float = _M, g: float = _G, z0: float = 0.11, friction: float = 1.0) -> str:
    fr = f"{friction} 0.005 0.0001"
    return f"""<mujoco><option gravity="0 0 -{g}"/>
<worldbody><geom name="floor" type="plane" size="5 5 0.1" friction="{fr}"/>
<body name="box" pos="0 0 {z0}"><freejoint/>
<geom name="boxg" type="box" size="0.1 0.1 0.1" mass="{m}" friction="{fr}"/></body></worldbody>
<sensor><contact name="netcontact_box" body1="box" data="force" reduce="netforce"/>
<contact name="contactfound_box" body1="box" data="found"/></sensor></mujoco>"""


def test_net_contact_force_matches_analytical_mg(tmp_path) -> None:
    # Scenario 1 (vertical): a resting body's net contact force == m*g, opposing
    # gravity (+z). Locks the vertical sign/frame.
    bk = _backend(tmp_path, _box_xml(), base_name="box")
    for _ in range(3000):
        bk.step(np.zeros((2, 0), dtype=np.float32), nsteps=1)

    bid = bk.get_body_ids(["box"])
    f = bk.get_body_net_contact_force_w(bid)

    assert f.shape == (2, 1, 3)
    np.testing.assert_allclose(f[:, 0, 2], _M * _G, atol=0.2)  # +z, force ON the body
    np.testing.assert_allclose(f[:, 0, :2], 0.0, atol=0.2)  # ~no horizontal in equilibrium
    assert bk.get_body_contact_state(bid).all()


def test_contact_state_false_in_air(tmp_path) -> None:
    bk = _backend(tmp_path, _box_xml(z0=1.0), base_name="box")
    bid = bk.get_body_ids(["box"])
    for _ in range(10):  # still falling, no contact yet
        bk.step(np.zeros((2, 0), dtype=np.float32), nsteps=1)

    assert not bk.get_body_contact_state(bid).any()
    np.testing.assert_allclose(bk.get_body_net_contact_force_w(bid), 0.0, atol=1e-3)


def test_net_contact_force_horizontal_sign(tmp_path) -> None:
    # Scenario 2 (horizontal): push the resting box with +Fx. In equilibrium the
    # net contact force ON the box = -(gravity + applied) = (-Fx, 0, +m*g).
    # Locks the horizontal sign (not just the vertical case).
    bk = _backend(tmp_path, _box_xml(), base_name="box")
    bid = bk.get_body_ids(["box"])
    fx = 3.0
    push = np.tile(np.array([fx, 0.0, 0.0]), (2, 1, 1))  # (2,1,3)
    for _ in range(3000):
        bk.apply_body_force(bid, push)  # re-stage each step (cleared after step)
        bk.step(np.zeros((2, 0), dtype=np.float32), nsteps=1)

    f = bk.get_body_net_contact_force_w(bid)
    np.testing.assert_allclose(f[:, 0, 0], -fx, atol=0.5)  # horizontal sign locked
    np.testing.assert_allclose(f[:, 0, 2], _M * _G, atol=0.5)
