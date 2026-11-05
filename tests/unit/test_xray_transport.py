import numpy as np

from plugins.virtRTG.xray.xrayTransport import XRayTransportModel


class _PreparedDirectSource:
    projection_label = "prepared"

    def uses_direct_integral(self):
        return True

    def resolve_physics_model(self, physics_model):
        return physics_model

    def ray_integral_world(self, **kwargs):
        assert kwargs["ray_origins"].shape == (2, 3)
        assert kwargs["hit_ray_indices"].tolist() == [1, 3]
        return np.array([1.25, 2.5], dtype=np.float64), 7


def test_transport_integrates_only_prepared_hit_rays_and_returns_flat_maps():
    source = _PreparedDirectSource()
    model = XRayTransportModel([source])
    ray_origins = np.arange(12, dtype=np.float32).reshape(4, 3)
    ray_directions = np.ones((4, 3), dtype=np.float32)
    t_starts = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    t_ends = np.array([0.0, 3.0, 0.0, 4.0], dtype=np.float32)
    hit_mask = np.array([False, True, False, True])

    result = model.integrate(
        geometry=object(),
        physics_model=object(),
        reference_transform=np.eye(4, dtype=np.float32),
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        t_starts=t_starts,
        t_ends=t_ends,
        hit_mask=hit_mask,
        hit_indices=np.array([1, 3]),
        detector_shape_hw=(2, 2),
        step_mm=0.5,
    )

    assert result.total_primary_integral.dtype == np.float32
    assert result.total_primary_integral.tolist() == [0.0, 1.25, 0.0, 2.5]
    assert np.array_equal(
        result.per_source_primary_integrals[0], result.total_primary_integral,
    )
    assert result.total_work_count == 7
    assert result.traced_pixels == 2
    assert result.per_source_stats[0]["label"] == "_PreparedDirectSource"
