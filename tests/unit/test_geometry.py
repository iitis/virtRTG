# -*- coding: utf-8 -*-
"""Unit tests for X-ray geometry helpers and low-level clipping functions."""

from __future__ import annotations

import numpy as np

from plugins.virtRTG.xray.xrayProjection import (
	XRayProjectionGeometry,
	_ray_box_intersections_vectorized,
	_ray_planar_slab_intersections_vectorized,
)


def test_from_detector_pose_preserves_requested_detector_center():
	"""Build geometry from pose parameters and recover the same detector centre."""
	geometry = XRayProjectionGeometry.from_detector_pose(
		detector_center_ref=[10.0, 20.0, 30.0],
		detector_normal_ref=[0.0, 0.0, -1.0],
		detector_up_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[5, 7],
		detector_pixel_size_mm=[0.5, 0.25],
		source_position_ref=[10.0, 20.0, -100.0],
		step_mm=0.75,
	)

	assert np.allclose(geometry.detector_center_ref_point(), [10.0, 20.0, 30.0])
	assert np.allclose(geometry.detector_pixel_size_mm_uv(), [0.5, 0.25])
	assert geometry.detector_shape_hw == [5, 7]
	assert geometry.is_cone_beam() is True
	assert geometry.is_parallel_beam() is False


def test_depth_window_limits_are_sorted_and_validation_accepts_planar_mode():
	"""Sort depth-window limits and validate a consistent planar depth window."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, 0.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[16, 16],
		step_mm=1.0,
		ray_direction_ref=[0.0, 0.0, 1.0],
		depth_window_mode="planar",
		depth_window_mm=[30.0, 10.0],
		depth_window_axis_ref=[0.0, 0.0, 1.0],
	)

	assert geometry.depth_window_limits_mm() == (10.0, 30.0)
	geometry.validate()


def test_ray_box_intersections_vectorized_reports_hit_and_miss_rays():
	"""Compute ray/AABB intersections for one ray that hits and one that misses."""
	ray_origins = np.array(
		[
			[0.0, 0.0, -5.0],
			[5.0, 5.0, -5.0],
		],
		dtype=np.float32,
	)
	ray_directions = np.array(
		[
			[0.0, 0.0, 1.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)
	t_start, t_end, hit_mask = _ray_box_intersections_vectorized(
		ray_origins=ray_origins,
		ray_directions=ray_directions,
		box_min=[-1.0, -1.0, -1.0],
		box_max=[1.0, 1.0, 1.0],
	)

	assert np.isclose(t_start[0], 4.0)
	assert np.isclose(t_end[0], 6.0)
	assert bool(hit_mask[0]) is True
	assert bool(hit_mask[1]) is False


def test_ray_planar_slab_intersections_vectorized_clips_to_slab_interval():
	"""Clip rays against one planar slab aligned with the Z axis."""
	ray_origins = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
	ray_directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
	t_start, t_end, hit_mask = _ray_planar_slab_intersections_vectorized(
		ray_origins=ray_origins,
		ray_directions=ray_directions,
		slab_origin=[0.0, 0.0, 0.0],
		slab_axis=[0.0, 0.0, 1.0],
		slab_min=2.0,
		slab_max=4.0,
	)

	assert np.allclose(t_start, [2.0])
	assert np.allclose(t_end, [4.0])
	assert np.array_equal(hit_mask, [True])
