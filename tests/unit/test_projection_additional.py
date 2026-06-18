# -*- coding: utf-8 -*-
"""Additional unit tests for projection configuration, physics and scene flow."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from plugins.virtRTG.xray.xrayPresentation import RawPresentationModel
from plugins.virtRTG.xray.xrayProjection import (
	XRayPhysicsModel,
	XRayProjectionConfig,
	XRayProjectionGeometry,
	XRayProjectionQualityProfile,
	XRayScene,
)
from plugins.virtRTG.virtualXRay import VirtualXRay


class _ConstantBoxSource:
	"""Return one constant attenuation inside one axis-aligned scene box."""

	def __init__(self, attenuation_value=0.5):
		"""Store one constant attenuation coefficient."""
		self.attenuation_value = float(attenuation_value)

	def bounds_world(self):
		"""Return one simple world-space AABB crossed by every test ray."""
		return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 2.0], dtype=np.float32)

	def sample_attenuation_world(self, points_world, physics_model):
		"""Return one constant attenuation for each sampled point."""
		return np.full(points_world.shape[0], self.attenuation_value, dtype=np.float32)

	def uses_direct_integral(self):
		"""Opt into slab marching instead of the direct-integral path."""
		return False

	def resolve_physics_model(self, physics_model):
		"""Keep the shared physics model unchanged for this simple helper source."""
		return physics_model


class _ScalarDrivenBoxSource:
	"""Map one constant scalar value through the effective per-source physics model."""

	def __init__(self, scalar_value=0.0, response_mode_override=None, bone_threshold_override=None):
		"""Store one scalar test value and optional source-local response overrides."""
		self.scalar_value = float(scalar_value)
		self.response_mode_override = response_mode_override
		self.bone_threshold_override = bone_threshold_override

	def bounds_world(self):
		"""Return one simple world-space AABB crossed by every test ray."""
		return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 2.0], dtype=np.float32)

	def sample_attenuation_world(self, points_world, physics_model):
		"""Return attenuation derived from one scalar and the effective physics model."""
		mu_value = float(np.asarray(
			physics_model.scalar_to_mu(np.array([self.scalar_value], dtype=np.float32)),
			dtype=np.float32,
		)[0])
		return np.full(points_world.shape[0], mu_value, dtype=np.float32)

	def uses_direct_integral(self):
		"""Opt into slab marching instead of the direct-integral path."""
		return False

	def resolve_physics_model(self, physics_model):
		"""Optionally override the shared material response for this one source."""
		if self.response_mode_override is None:
			return physics_model
		override_kwargs = {
			"material_response_mode": str(self.response_mode_override),
		}
		if self.bone_threshold_override is not None:
			override_kwargs["bone_threshold_hu"] = float(self.bone_threshold_override)
		return replace(physics_model, **override_kwargs)


def test_quality_profile_downsamples_detector_and_updates_step():
	"""Apply one draft profile to geometry and scale detector sampling consistently."""
	geometry = XRayProjectionGeometry.from_detector_pose(
		detector_center_ref=[0.0, 0.0, 0.0],
		detector_normal_ref=[0.0, 0.0, -1.0],
		detector_up_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[9, 7],
		detector_pixel_size_mm=[0.5, 0.25],
		source_position_ref=[0.0, 0.0, -100.0],
		step_mm=1.0,
	)

	quality_geometry = geometry.with_quality_profile(XRayProjectionQualityProfile.draft())

	assert quality_geometry.detector_shape_hw == [5, 4]
	assert np.allclose(quality_geometry.detector_u_ref, np.asarray(geometry.detector_u_ref, dtype=np.float32) * 2.0)
	assert np.allclose(quality_geometry.detector_v_ref, np.asarray(geometry.detector_v_ref, dtype=np.float32) * 2.0)
	assert quality_geometry.step_mm == 2.0


def test_integral_output_mode_applies_distance_gain_as_log_offset():
	"""Shift integral-mode output by the logarithm of the cone-beam distance gain."""
	model = XRayPhysicsModel(
		output_mode="integral",
		source_distance_falloff_mode="inverse_square",
		source_distance_reference_mm=1000.0,
		source_distance_power=2.0,
	)

	image = model.integral_to_image(
		line_integral=np.array([2.0], dtype=np.float32),
		source_to_detector_distance_mm=np.array([2000.0], dtype=np.float32),
		projection_mode="cone",
	)

	assert np.allclose(image, [2.0 - np.log(0.25)], atol=1e-6)


def test_source_distance_gain_uses_median_reference_when_not_configured():
	"""Infer the reference cone-beam distance from the finite-distance median."""
	model = XRayPhysicsModel(
		source_distance_falloff_mode="inverse_square",
		source_distance_reference_mm=None,
		source_distance_power=2.0,
	)

	gain = model.source_distance_gain(
		source_to_detector_distance_mm=np.array([1000.0, 2000.0, 4000.0], dtype=np.float32),
		projection_mode="cone",
	)

	assert np.allclose(gain, [4.0, 1.0, 0.25], atol=1e-6)


def test_piecewise_and_threshold_physics_models_return_non_negative_mu_values():
	"""Keep attenuation non-negative across piecewise and threshold response modes."""
	values = np.array([-1000.0, 0.0, 500.0, 1500.0], dtype=np.float32)
	bone_model = XRayPhysicsModel(material_response_mode="piecewise_bone")
	threshold_model = XRayPhysicsModel(
		material_response_mode="bone_threshold",
		bone_threshold_hu=400.0,
		bone_threshold_softness=50.0,
	)

	bone_mu = bone_model.scalar_to_mu(values)
	threshold_mu = threshold_model.scalar_to_mu(values)

	assert np.all(bone_mu >= 0.0)
	assert np.all(threshold_mu >= 0.0)
	assert bone_mu[-1] > bone_mu[1]
	assert threshold_mu[-1] > threshold_mu[1]


def test_scene_render_returns_presented_image_and_projection_stats():
	"""Render one constant source through the full scene/configuration API."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, -1.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[2, 2],
		ray_direction_ref=[0.0, 0.0, 1.0],
		step_mm=1.0,
	)
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=XRayPhysicsModel(output_mode="integral"),
		presentation_model=RawPresentationModel(),
	)
	scene = XRayScene.from_sample_sources([_ConstantBoxSource(attenuation_value=0.5)])

	image, stats = scene.render(config=config, return_stats=True)

	assert image.shape == (2, 2)
	assert np.allclose(image, np.full((2, 2), 1.5, dtype=np.float32))
	assert stats.traced_pixels == 4
	assert stats.total_pixels == 4
	assert stats.total_sample_count == 12
	assert stats.projection_mode == "parallel"
	assert "Detector:" in stats.format_report()


def test_scene_project_capture_exposes_additive_per_source_line_integrals():
	"""Return total and per-source detector maps so object contributions can be exported separately."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, -1.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[1, 1],
		ray_direction_ref=[0.0, 0.0, 1.0],
		step_mm=1.0,
	)
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=XRayPhysicsModel(output_mode="integral"),
	)
	scene = XRayScene.from_sample_sources([
		_ConstantBoxSource(attenuation_value=0.5),
		_ConstantBoxSource(attenuation_value=0.25),
	])

	capture = scene.project_capture(config=config, return_stats=True)

	assert capture.stats is not None
	assert capture.stats.source_count == 2
	assert np.allclose(capture.line_integral_image, [[2.25]], atol=1e-6)
	assert np.allclose(capture.detector_image, [[2.25]], atol=1e-6)
	assert len(capture.source_projections) == 2
	assert np.allclose(capture.source_projections[0].line_integral_image, [[1.5]], atol=1e-6)
	assert np.allclose(capture.source_projections[1].line_integral_image, [[0.75]], atol=1e-6)
	assert np.allclose(capture.source_projections[0].detector_image, [[1.5]], atol=1e-6)
	assert np.allclose(capture.source_projections[1].detector_image, [[0.75]], atol=1e-6)
	assert np.allclose(
		capture.source_projections[0].line_integral_image + capture.source_projections[1].line_integral_image,
		capture.line_integral_image,
		atol=1e-6,
	)


def test_scene_project_respects_ray_depth_window_limits():
	"""Clip slab marching to the configured ray-parameter interval."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, -1.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[1, 1],
		ray_direction_ref=[0.0, 0.0, 1.0],
		step_mm=1.0,
		depth_window_mode="ray",
		depth_window_mm=[1.0, 1.1],
	)
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=XRayPhysicsModel(output_mode="integral"),
	)
	scene = XRayScene.from_sample_sources([_ConstantBoxSource(attenuation_value=0.5)])

	image = scene.project(config=config, return_stats=False)

	assert np.allclose(image, [[0.5]], atol=1e-6)


def test_virtual_xray_cached_projection_roundtrip_supports_npy_and_text(tmp_path):
	"""Persist cached detector arrays as `.npy` and text files without presentation side effects."""
	virtual_xray = VirtualXRay()
	virtual_xray.projection_mode = "parallel"
	expected = np.array([[1.0, 2.5], [3.25, 4.5]], dtype=np.float32)
	virtual_xray.last_raw_projection = expected.copy()
	virtual_xray.last_line_integral_projection = expected.copy()

	raw_npy_path = tmp_path / "raw_projection.npy"
	line_txt_path = tmp_path / "line_integral.txt"

	virtual_xray.export_cached_projection(raw_npy_path, stage="raw")
	virtual_xray.export_cached_projection(line_txt_path, stage="line_integral")

	virtual_xray.last_raw_projection = None
	virtual_xray.last_line_integral_projection = None

	raw_loaded = virtual_xray.import_cached_projection(raw_npy_path, stage="raw")
	assert np.allclose(raw_loaded, expected, atol=1e-6)
	assert np.allclose(virtual_xray.last_raw_projection, expected, atol=1e-6)

	line_loaded = virtual_xray.import_cached_projection(line_txt_path, stage="line_integral")
	assert np.allclose(line_loaded, expected, atol=1e-6)
	assert np.allclose(virtual_xray.last_line_integral_projection, expected, atol=1e-6)
	assert np.allclose(virtual_xray.last_raw_projection, expected, atol=1e-6)


def test_scene_sources_inherit_global_material_response_by_default():
	"""Use the shared material response when a source does not define a local override."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, -1.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[1, 1],
		ray_direction_ref=[0.0, 0.0, 1.0],
		step_mm=1.0,
	)
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=XRayPhysicsModel(output_mode="integral", material_response_mode="piecewise_bone"),
	)
	scene = XRayScene.from_sample_sources([_ScalarDrivenBoxSource(scalar_value=600.0)])

	image = scene.project(config=config, return_stats=False)
	expected_mu = float(np.asarray(
		config.physics_model.scalar_to_mu(np.array([600.0], dtype=np.float32)),
		dtype=np.float32,
	)[0])

	assert np.allclose(image, [[expected_mu * 3.0]], atol=1e-6)


def test_scene_sources_can_override_material_response_per_source():
	"""Allow one source to replace the shared material response with a local override."""
	geometry = XRayProjectionGeometry(
		detector_origin_ref=[0.0, 0.0, -1.0],
		detector_u_ref=[1.0, 0.0, 0.0],
		detector_v_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[1, 1],
		ray_direction_ref=[0.0, 0.0, 1.0],
		step_mm=1.0,
	)
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=XRayPhysicsModel(
			output_mode="integral",
			material_response_mode="linear",
			bone_threshold_hu=900.0,
			bone_threshold_softness=10.0,
		),
	)
	scene = XRayScene.from_sample_sources([
		_ScalarDrivenBoxSource(
			scalar_value=600.0,
			response_mode_override="bone_threshold",
			bone_threshold_override=500.0,
		)
	])

	image = scene.project(config=config, return_stats=False)
	override_model = replace(
		config.physics_model,
		material_response_mode="bone_threshold",
		bone_threshold_hu=500.0,
	)
	expected_mu = float(np.asarray(
		override_model.scalar_to_mu(np.array([600.0], dtype=np.float32)),
		dtype=np.float32,
	)[0])

	assert np.allclose(image, [[expected_mu * 3.0]], atol=1e-6)
