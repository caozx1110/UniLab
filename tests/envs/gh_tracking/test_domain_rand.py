"""Tests for GH domain randomization (Phase 7): sampling semantics.

Startup-once material/COM (sampled once per env, fixed across resets), per-reset
motor params, and the batch-shared per-joint zero offset. The live write into the
UniLab DR system (kp/kd/armature/com/friction) and the PhysX->MuJoCo friction
approximation are deferred; here we verify the GH sampling lifecycle + ranges.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.gh_tracking.domain_rand import GHDomainRand, material_to_geom_friction_sliding


def _dr(n: int = 4) -> GHDomainRand:
    return GHDomainRand(num_envs=n, num_joints=29, num_com_bodies=2, seed=0)


def test_startup_com_sampled_once_and_fixed() -> None:
    dr = _dr()
    c1 = dr.startup_com().copy()
    c2 = dr.startup_com().copy()
    np.testing.assert_array_equal(c1, c2)  # cached: fixed across calls (i.e. across resets)
    assert c1.shape == (4, 2, 3)
    assert (np.abs(c1) <= 0.02 + 1e-9).all()  # com_range [-0.02, 0.02]


def test_startup_material_sampled_once_and_fixed_with_ranges() -> None:
    dr = _dr()
    m1 = dr.startup_material()
    m2 = dr.startup_material()
    np.testing.assert_array_equal(m1["static"], m2["static"])  # cached
    assert ((m1["static"] >= 0.3) & (m1["static"] <= 1.6)).all()
    assert ((m1["dynamic_frac"] >= 0.75) & (m1["dynamic_frac"] <= 1.0)).all()
    assert ((m1["restitution"] >= 0.0) & (m1["restitution"] <= 0.2)).all()


def test_joint_offset_is_batch_shared_per_joint_draw() -> None:
    dr = _dr()
    off = dr.sample_joint_offset(np.arange(4))  # (4, 29)
    assert off.shape == (4, 29)
    # all envs in the reset batch share the SAME per-joint draw (source-compatible)
    np.testing.assert_array_equal(off[0], off[1])
    np.testing.assert_array_equal(off[0], off[2])
    np.testing.assert_array_equal(off[0], off[3])
    assert (np.abs(off) <= 0.01 + 1e-9).all()  # [-0.01, 0.01]


def test_motor_params_per_reset_ranges() -> None:
    dr = _dr()
    m = dr.sample_motor(np.arange(4))  # per-env per-joint scales
    assert m["stiffness_scale"].shape == (4, 29)
    assert ((m["stiffness_scale"] >= 0.9) & (m["stiffness_scale"] <= 1.1)).all()
    assert ((m["damping_scale"] >= 0.9) & (m["damping_scale"] <= 1.1)).all()
    assert ((m["armature_scale"] >= 0.75) & (m["armature_scale"] <= 1.25)).all()


def test_motor_resample_differs_across_resets() -> None:
    dr = _dr()
    a = dr.sample_motor(np.arange(4))["stiffness_scale"]
    b = dr.sample_motor(np.arange(4))["stiffness_scale"]
    assert not np.allclose(a, b)  # per-reset resample (not cached like startup)


def test_friction_sliding_only_approximation() -> None:
    static = np.array([[0.8], [1.2]])
    sliding = material_to_geom_friction_sliding(static)
    np.testing.assert_array_equal(sliding, static)  # only the sliding coefficient is produced
    # restitution / dynamic have no MuJoCo geom_friction slot -> approximation (not returned here)
