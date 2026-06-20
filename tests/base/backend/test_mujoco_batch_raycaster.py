from __future__ import annotations

import numpy as np

from unilab.base.backend.mujoco.backend import MuJoCoBackend


def test_mujoco_raycaster_calls_batch_multi_ray() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.calls = []

        def multi_ray(self, initial_state, pnt, vec, **kwargs):
            self.calls.append((initial_state, pnt, vec, kwargs))
            return (
                np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64),
                np.asarray([[4, 5], [-1, -1]], dtype=np.int32),
                np.asarray(
                    [
                        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float64,
                ),
            )

    backend = object.__new__(MuJoCoBackend)
    backend._pool = FakePool()
    backend._physics_state = np.zeros((2, 5), dtype=np.float32)
    backend._num_envs = 2
    backend._np_dtype = np.float32
    backend._base_body_id = 7
    backend.add_body_sensors = False
    backend.get_base_pos = lambda: np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    backend.get_base_quat = lambda: np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    raycaster = backend.create_raycaster(
        frame_body_id=7,
        directions=np.asarray([[0.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float64),
        geomgroup=[0, 2],
        return_normal=True,
        cutoff=5.0,
    )

    result = raycaster.cast()

    assert result.distances.shape == (2, 2)
    assert result.geom_ids.shape == (2, 2)
    assert result.normals is not None
    assert result.normals.shape == (2, 2, 3)
    call_state, pnt, vec, kwargs = backend._pool.calls[0]
    np.testing.assert_array_equal(call_state, backend._physics_state)
    np.testing.assert_array_equal(
        pnt,
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
        ],
    )
    np.testing.assert_allclose(np.linalg.norm(vec, axis=2), 1.0)
    np.testing.assert_array_equal(kwargs["geomgroup"], np.asarray([1, 0, 1, 0, 0, 0], dtype=np.uint8))
    assert kwargs["bodyexclude"] == 7
    assert kwargs["return_normal"] is True
    assert kwargs["cutoff"] == 5.0


def test_mujoco_raycaster_validates_multi_ray_shapes() -> None:
    class BadPool:
        def multi_ray(self, *args, **kwargs):
            del args, kwargs
            return np.zeros((1, 2)), np.zeros((1, 2), dtype=np.int32), None

    backend = object.__new__(MuJoCoBackend)
    backend._pool = BadPool()
    backend._physics_state = np.zeros((2, 5), dtype=np.float32)
    backend._num_envs = 2
    backend._np_dtype = np.float32
    backend._base_body_id = 7
    backend.add_body_sensors = False
    backend.get_base_pos = lambda: np.zeros((2, 3), dtype=np.float64)
    backend.get_base_quat = lambda: np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]]), (2, 1))

    raycaster = backend.create_raycaster(
        frame_body_id=7,
        directions=np.asarray([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]], dtype=np.float64),
    )

    try:
        raycaster.cast()
    except ValueError as exc:
        assert "invalid shapes" in str(exc)
    else:
        raise AssertionError("expected shape validation to fail")


def test_mujoco_raycaster_accepts_per_ray_origin_offsets() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.pnt = None
            self.vec = None

        def multi_ray(self, initial_state, pnt, vec, **kwargs):
            del initial_state, kwargs
            self.pnt = pnt
            self.vec = vec
            return np.ones((2, 2)), np.zeros((2, 2), dtype=np.int32), None

    backend = object.__new__(MuJoCoBackend)
    backend._pool = FakePool()
    backend._physics_state = np.zeros((2, 5), dtype=np.float32)
    backend._num_envs = 2
    backend._np_dtype = np.float32
    backend._base_body_id = 7
    backend.add_body_sensors = False
    backend.get_base_pos = lambda: np.asarray([[10.0, 0.0, 1.0], [20.0, 0.0, 2.0]])
    backend.get_base_quat = lambda: np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]]), (2, 1))

    raycaster = backend.create_raycaster(
        frame_body_id=7,
        directions=np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        origin_offsets=np.asarray([[-0.8, -0.5, 0.0], [0.8, 0.5, 0.0]], dtype=np.float64),
    )

    raycaster.cast()

    np.testing.assert_allclose(
        backend._pool.pnt,
        [
            [[9.2, -0.5, 1.0], [10.8, 0.5, 1.0]],
            [[19.2, -0.5, 2.0], [20.8, 0.5, 2.0]],
        ],
    )
    np.testing.assert_allclose(
        backend._pool.vec,
        np.tile([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]], (2, 1, 1)),
    )
