# -*- coding: utf-8 -*-
"""Unit tests for lightweight mesh and export helpers in `xraySource`."""

from __future__ import annotations

import numpy as np

from plugins.virtRTG.xray.xraySource import (
	_build_triangle_bvh,
	_one_ray_per_triangle_hit,
	_ray_triangle_hit_distances,
	_ray_triangle_intersections_bvh,
	_rays_single_triangle_hit_distances,
	normalize_projection_to_uint8,
	normalize_projection_to_uint16,
)


def _triangle_at_z(z_value):
	"""Return one right triangle lying in a plane of constant `z`."""
	return np.array(
		[
			[0.0, 0.0, z_value],
			[1.0, 0.0, z_value],
			[0.0, 1.0, z_value],
		],
		dtype=np.float32,
	)


def test_projection_normalization_helpers_cover_fixed_and_empty_ranges():
	"""Normalize projection images into integer outputs for non-empty and empty inputs."""
	image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
	empty = np.array([[np.nan, np.nan]], dtype=np.float32)

	image_u8 = normalize_projection_to_uint8(image, fixed_range=(0.0, 3.0), invert=True)
	image_u16 = normalize_projection_to_uint16(image, fixed_range=(0.0, 3.0), invert=False)
	empty_u16 = normalize_projection_to_uint16(empty)

	assert image_u8.dtype == np.uint8
	assert image_u8.tolist() == [[255, 170], [85, 0]]
	assert image_u16.dtype == np.uint16
	assert image_u16[0, 0] == 0
	assert image_u16[1, 1] == 65535
	assert np.array_equal(empty_u16, np.zeros_like(empty_u16))


def test_triangle_hit_helpers_return_expected_distances_for_simple_geometry():
	"""Report one hit distance for direct, paired and many-rays helper variants."""
	hit_triangle = _triangle_at_z(5.0)
	miss_triangle = _triangle_at_z(8.0) + np.array([2.0, 0.0, 0.0], dtype=np.float32)
	triangles = np.stack([hit_triangle, miss_triangle], axis=0)

	t_values = _ray_triangle_hit_distances(
		ray_origin=[0.25, 0.25, 0.0],
		ray_direction=[0.0, 0.0, 1.0],
		triangles_world=triangles,
	)
	per_triangle_hits = _one_ray_per_triangle_hit(
		ray_origins=np.array([[0.25, 0.25, 0.0], [0.25, 0.25, 0.0]], dtype=np.float32),
		ray_directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
		triangles_world=triangles,
	)
	per_ray_hits = _rays_single_triangle_hit_distances(
		ray_origins=np.array([[0.25, 0.25, 0.0], [1.25, 1.25, 0.0]], dtype=np.float32),
		ray_directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
		triangle_world=hit_triangle,
	)

	assert np.allclose(t_values, [5.0])
	assert np.isclose(per_triangle_hits[0], 5.0)
	assert np.isnan(per_triangle_hits[1])
	assert np.isclose(per_ray_hits[0], 5.0)
	assert np.isnan(per_ray_hits[1])


def test_bvh_intersections_match_direct_triangle_hits():
	"""Traverse one triangle BVH and recover the same sorted hit distances."""
	triangles = np.stack([_triangle_at_z(5.0), _triangle_at_z(8.0)], axis=0)
	bvh_nodes = _build_triangle_bvh(triangles_world=triangles, max_leaf_size=1)

	direct_hits = np.sort(_ray_triangle_hit_distances(
		ray_origin=[0.25, 0.25, 0.0],
		ray_direction=[0.0, 0.0, 1.0],
		triangles_world=triangles,
	))
	bvh_hits = _ray_triangle_intersections_bvh(
		ray_origin=[0.25, 0.25, 0.0],
		ray_direction=[0.0, 0.0, 1.0],
		triangles_world=triangles,
		bvh_nodes=bvh_nodes,
	)

	assert len(bvh_nodes) >= 1
	assert np.allclose(bvh_hits, direct_hits)
