"""Tests for the GH admittance mass-chain (Phase 5).

AdmittanceMassChain ports GH admittance.py: semi-implicit Euler at physics_dt,
integrated 4x per control step (0.005 x 4), with mass/damping and norm-clamped
acc/vel. clamp_norm scales a vector to a max norm.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.admittance import AdmittanceMassChain, clamp_norm


def test_clamp_norm_scales_only_when_exceeding() -> None:
    x = np.array([[3.0, 4.0, 0.0]])  # norm 5
    np.testing.assert_allclose(clamp_norm(x, 5.0), x)  # <= max unchanged
    np.testing.assert_allclose(np.linalg.norm(clamp_norm(x, 2.0), axis=-1), 2.0)  # capped


def test_clamp_norm_broadcast_max() -> None:
    x = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    max_arr = np.array([[2.0], [5.0]])
    out = clamp_norm(x, max_arr)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), [2.0, 5.0])


def test_admittance_single_step_golden() -> None:
    a = AdmittanceMassChain(num_envs=1, num_points=1, dt=0.005, mass=0.1,
                            damping=2.0, vel_clip=4.0, acc_clip=1000.0)
    fd = np.array([[[[1.0, 0.0, 0.0]]]])  # (H=1, N=1, M=1, 3)
    fe = np.zeros_like(fd)
    a.step(fd, fe)
    # F_damp=0; a=1/0.1=10; v=10*0.005=0.05; x=0.05*0.005=0.00025
    np.testing.assert_allclose(a.v[0, 0, 0], [0.05, 0, 0], atol=1e-9)
    np.testing.assert_allclose(a.x[0, 0, 0], [0.00025, 0, 0], atol=1e-9)


def test_admittance_four_step_matches_reference_loop() -> None:
    a = AdmittanceMassChain(1, 1, 0.005, mass=0.1, damping=2.0, vel_clip=4.0, acc_clip=1000.0)
    fd = np.array([[[[1.0, 0.0, 0.0]]]])
    fe = np.zeros_like(fd)
    x = np.zeros(3)
    v = np.zeros(3)
    for _ in range(4):  # independent reference
        f_damp = -2.0 * v
        f_total = np.array([1.0, 0.0, 0.0]) + f_damp
        acc = f_total / 0.1
        v = v + acc * 0.005
        x = x + v * 0.005
    for _ in range(4):
        a.step(fd, fe)
    np.testing.assert_allclose(a.x[0, 0, 0], x, atol=1e-9)
    np.testing.assert_allclose(a.v[0, 0, 0], v, atol=1e-9)


def test_admittance_vel_and_acc_clip() -> None:
    a = AdmittanceMassChain(1, 1, 0.005, mass=0.1, damping=2.0, vel_clip=4.0, acc_clip=1000.0)
    fd = np.array([[[[1e6, 0.0, 0.0]]]])  # acc clamps to 1000 -> v=5 -> vel clamps to 4
    a.step(fd, np.zeros_like(fd))
    np.testing.assert_allclose(np.linalg.norm(a.v[0, 0, 0]), 4.0, atol=1e-6)


def test_admittance_reset_sets_x0_v0() -> None:
    a = AdmittanceMassChain(2, 1, 0.005)
    x0 = np.array([[[1.0, 2.0, 3.0]]])
    v0 = np.array([[[0.1, 0.0, 0.0]]])
    a.reset(np.array([0]), x0, v0)
    np.testing.assert_allclose(a.x[0, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(a.v[0, 0, 0], [0.1, 0, 0])
