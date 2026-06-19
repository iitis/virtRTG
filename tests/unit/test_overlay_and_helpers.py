# -*- coding: utf-8 -*-
"""Unit tests for detector overlay projection helpers and transform utilities."""

from __future__ import annotations

import numpy as np
import pytest

from plugins.virtRTG.xray.xrayAnnotationOverlay import (
	AnnotationPathProjector,
	AnnotationPointProjector,
	XRayAnnotationProjectionContext,
	build_overlay_projection_set,
)
from plugins.virtRTG.xray.xrayHelpers import (
	_normalize_vector,
	_transform_direction,
	_transform_point,
)


class _FakeColor:
	"""Provide the minimal RGBA API used by overlay style extraction."""

	def __init__(self, red, green, blue, alpha):
		"""Store one deterministic RGBA tuple."""
		self._rgba = (red, green, blue, alpha)

	def red(self):
		"""Return the red channel."""
		return self._rgba[0]

	def green(self):
		"""Return the green channel."""
		return self._rgba[1]

	def blue(self):
		"""Return the blue channel."""
		return self._rgba[2]

	def alpha(self):
		"""Return the alpha channel."""
		return self._rgba[3]


class _FakeGeometry:
	"""Expose the minimal geometry API used by overlay projection."""

	def __init__(self, source_position_ref=None, ray_direction_ref=None):
		"""Store one simple detector plane and beam definition."""
		self.detector_origin_ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		self.detector_u_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
		self.detector_v_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		self.detector_shape_hw = [5, 5]
		self.source_position_ref = source_position_ref
		self.ray_direction_ref = ray_direction_ref

	def detector_normal_ref_vector(self):
		"""Return one detector normal pointing along positive Z."""
		return np.array([0.0, 0.0, 1.0], dtype=np.float32)

	def is_cone_beam(self):
		"""Return whether the geometry uses one point source."""
		return self.source_position_ref is not None


class _FakeAnnotationBase:
	"""Provide the minimal scene-object API shared by test annotations."""

	def __init__(self, label, transform=None, visible=True, checked=False):
		"""Store common scene-object state for overlay tests."""
		self.label = label
		self.visible = visible
		self.checked = checked
		self._transform = np.eye(4, dtype=np.float32) if transform is None else np.asarray(transform, dtype=np.float32)

	def getGlobalTransformation(self):
		"""Return one homogeneous transform used by the overlay context."""
		return self._transform

	def getColor(self):
		"""Return one default annotation color."""
		return _FakeColor(10, 20, 30, 40)

	def getSelColor(self):
		"""Return one selected annotation color."""
		return _FakeColor(100, 110, 120, 130)


class _FakePoint(_FakeAnnotationBase):
	"""Represent one point annotation compatible with the point projector."""

	def __init__(self, point_xyz, vector_xyz=None, show_vector=False, **kwargs):
		"""Store one local point and optional vector payload."""
		super().__init__(**kwargs)
		self._point_xyz = np.asarray(point_xyz, dtype=np.float32)
		self._vector_xyz = None if vector_xyz is None else tuple(vector_xyz)
		self.m_showVector = bool(show_vector)

	def getPoint(self):
		"""Return the local point position."""
		return self._point_xyz

	def getVector(self):
		"""Return the optional local vector payload."""
		return self._vector_xyz


class _FakePath(_FakeAnnotationBase):
	"""Represent one polyline annotation compatible with the path projector."""

	def __init__(self, points_xyz, width=1.0, **kwargs):
		"""Store one list of local polyline vertices."""
		super().__init__(**kwargs)
		self.m_points = [np.asarray(point_xyz, dtype=np.float32) for point_xyz in points_xyz]
		self.m_width = float(width)


def test_normalize_vector_and_transform_helpers_apply_expected_geometry():
	"""Normalize vectors and apply point and direction transforms independently."""
	normalized = _normalize_vector([0.0, 3.0, 4.0])
	transform = np.array(
		[
			[1.0, 0.0, 0.0, 10.0],
			[0.0, 1.0, 0.0, 20.0],
			[0.0, 0.0, 1.0, 30.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)

	assert np.allclose(normalized, [0.0, 0.6, 0.8])
	assert np.allclose(_transform_point(transform, [1.0, 2.0, 3.0]), [11.0, 22.0, 33.0])
	assert np.allclose(_transform_direction(transform, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


def test_normalize_vector_rejects_zero_length_input():
	"""Reject one degenerate vector with zero Euclidean norm."""
	with pytest.raises(ValueError):
		_normalize_vector([0.0, 0.0, 0.0])


def test_projection_context_object_to_reference_uses_inverse_reference_transform():
	"""Map one object-local point through object and reference transforms."""
	scene_transform = np.array(
		[
			[1.0, 0.0, 0.0, 5.0],
			[0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)
	reference_transform = np.array(
		[
			[1.0, 0.0, 0.0, 2.0],
			[0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)
	context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 1.0]),
		reference_transform=reference_transform,
	)
	scene_object = _FakePoint(label="p", point_xyz=[0.0, 0.0, 0.0], transform=scene_transform)

	point_ref = context.object_point_to_reference(scene_object, [1.0, 0.0, 0.0])

	assert np.allclose(point_ref, [4.0, 0.0, 0.0])


def test_projection_context_reports_parallel_beam_and_cone_beam_edge_statuses():
	"""Report deterministic statuses for projected, invalid and degenerate rays."""
	parallel_context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 1.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)

	projected = parallel_context.project_reference_point([1.0, 2.0, -5.0])
	behind = parallel_context.project_reference_point([1.0, 2.0, 5.0])
	out_of_bounds = parallel_context.project_reference_point([9.0, 2.0, -5.0])

	invalid_direction_context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 0.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)
	invalid_direction = invalid_direction_context.project_reference_point([0.0, 0.0, -1.0])

	parallel_to_detector_context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[1.0, 0.0, 0.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)
	parallel_to_detector = parallel_to_detector_context.project_reference_point([0.0, 0.0, -1.0])

	cone_context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(source_position_ref=[0.0, 0.0, -5.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)
	at_source = cone_context.project_reference_point([0.0, 0.0, -5.0])

	assert projected["status"] == "projected"
	assert projected["detector_pixel_uv"] == (1.0, 2.0)
	assert bool(projected["in_bounds"]) is True
	assert behind["status"] == "behind_ray_origin"
	assert bool(behind["visible"]) is False
	assert out_of_bounds["status"] == "projected"
	assert bool(out_of_bounds["in_bounds"]) is False
	assert invalid_direction["status"] == "invalid_direction"
	assert parallel_to_detector["status"] == "parallel_to_detector"
	assert at_source["status"] == "at_source"


def test_overlay_projectors_build_cross_and_polyline_items():
	"""Project one point and one path into overlay items with detector metadata."""
	context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 1.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)
	point_projector = AnnotationPointProjector()
	point_projector.scene_type = _FakePoint
	path_projector = AnnotationPathProjector()
	path_projector.scene_type = _FakePath

	point = _FakePoint(
		label="point-a",
		point_xyz=[1.0, 2.0, -5.0],
		vector_xyz=[0.0, 1.0, 0.0],
		show_vector=True,
		checked=True,
	)
	path = _FakePath(
		label="path-a",
		points_xyz=[[0.0, 0.0, -5.0], [2.0, 1.0, -5.0]],
		width=2.4,
	)
	hidden = _FakePoint(label="hidden", point_xyz=[0.0, 0.0, -5.0], visible=False)

	overlay_set = build_overlay_projection_set(
		descendants=[point, path, hidden],
		context=context,
		projectors=[point_projector, path_projector],
	)

	assert overlay_set.detector_shape_hw == (5, 5)
	assert len(overlay_set.items) == 2

	cross_item = overlay_set.items[0]
	polyline_item = overlay_set.items[1]

	assert cross_item.kind == "AnnotationPoint"
	assert cross_item.label == "point-a"
	assert cross_item.pixel_uv == (1.0, 2.0)
	assert cross_item.style.color_rgba == (100, 110, 120, 130)
	assert cross_item.metadata["status"] == "projected"
	assert cross_item.metadata["show_vector"] is True
	assert cross_item.metadata["vector"] == (0.0, 1.0, 0.0)

	assert polyline_item.kind == "AnnotationPath"
	assert polyline_item.visible is True
	assert polyline_item.in_bounds is True
	assert polyline_item.pixel_uvs == [(0.0, 0.0), (2.0, 1.0)]
	assert polyline_item.style.line_width_px == 2
	assert polyline_item.metadata["point_count"] == 2


def test_overlay_context_can_suffix_labels_and_override_object_transform():
	"""Allow frame-specific label suffixes and explicit reference-space transforms."""
	context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 1.0]),
		reference_transform=np.eye(4, dtype=np.float32),
		object_transform_resolver=lambda _scene_object: np.array(
			[
				[1.0, 0.0, 0.0, 3.0],
				[0.0, 1.0, 0.0, 4.0],
				[0.0, 0.0, 1.0, -2.0],
				[0.0, 0.0, 0.0, 1.0],
			],
			dtype=np.float32,
		),
		overlay_label_suffix=" [frame 007]",
	)
	projector = AnnotationPointProjector()
	projector.scene_type = _FakePoint
	point = _FakePoint(
		label="point-a",
		point_xyz=[0.0, 0.0, 0.0],
	)

	item = projector.project(point, context)[0]

	assert item.label == "point-a [frame 007]"
	assert item.pixel_uv == (3.0, 4.0)


def test_annotation_path_projector_hides_polylines_shorter_than_two_points():
	"""Mark one projected path as invisible when fewer than two points survive."""
	context = XRayAnnotationProjectionContext(
		geometry=_FakeGeometry(ray_direction_ref=[0.0, 0.0, 1.0]),
		reference_transform=np.eye(4, dtype=np.float32),
	)
	projector = AnnotationPathProjector()
	projector.scene_type = _FakePath
	path = _FakePath(label="short", points_xyz=[[1.0, 1.0, -2.0]], width=1.0)

	item = projector.project(path, context)[0]

	assert item.kind == "AnnotationPath"
	assert item.pixel_uvs == [(1.0, 1.0)]
	assert item.visible is False
	assert item.in_bounds is True
