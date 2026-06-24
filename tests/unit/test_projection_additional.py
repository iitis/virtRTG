# -*- coding: utf-8 -*-
"""Additional unit tests for projection configuration, physics and scene flow."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dpVision import Mesh, Motion, Transform, Volumetric

from plugins.virtRTG.xray.xrayPresentation import RawPresentationModel
from plugins.virtRTG.xray.xrayProjection import (
	XRayPhysicsModel,
	XRayProjectionConfig,
	XRayProjectionGeometry,
	XRayProjectionQualityProfile,
	XRayScene,
)
from plugins.virtRTG.xray.xrayAnnotationOverlay import (
	XRayOverlayCross,
	XRayOverlayProjectionSet,
	XRayOverlayStyle,
)
from plugins.virtRTG.detectorImage import DetectorImage
from plugins.virtRTG.virtualXRay import VirtualXRay


def _translated_transform(tx=0.0, ty=0.0, tz=0.0):
	"""Build one simple translation transform for scene-assembly tests."""
	transform = Transform()
	transform.setTranslation(tx, ty, tz)
	return transform


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


def test_virtual_xray_cached_projection_npz_roundtrip_restores_overlay_metadata(tmp_path):
	"""Persist one projection package as `.npz` together with projected overlays and metadata."""
	virtual_xray = VirtualXRay()
	expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
	virtual_xray.last_raw_projection = expected.copy()
	virtual_xray.last_projected_annotations = XRayOverlayProjectionSet(
		detector_shape_hw=(2, 2),
		items=[
			XRayOverlayCross(
				kind="AnnotationPoint",
				label="A",
				pixel_uv=(0.5, 1.5),
				style=XRayOverlayStyle(color_rgba=(255, 0, 0, 255), line_width_px=2, marker_size_px=7),
				metadata={"status": "projected", "source_point_ref": [1.0, 2.0, 3.0]},
			)
		],
	)

	export_path = tmp_path / "projection_bundle.npz"
	virtual_xray.export_cached_projection(export_path, stage="raw")

	virtual_xray.last_raw_projection = None
	virtual_xray.last_projected_annotations = None

	loaded = virtual_xray.import_cached_projection(export_path, stage="line_integral")

	assert np.allclose(loaded, expected, atol=1e-6)
	assert np.allclose(virtual_xray.last_raw_projection, expected, atol=1e-6)
	assert virtual_xray.last_projected_annotations is not None
	assert virtual_xray.last_projected_annotations.detector_shape_hw == (2, 2)
	assert len(virtual_xray.last_projected_annotations.items) == 1
	item = virtual_xray.last_projected_annotations.items[0]
	assert item.label == "A"
	assert item.kind == "AnnotationPoint"
	assert item.pixel_uv == (0.5, 1.5)
	assert item.metadata["status"] == "projected"


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


def test_detector_image_projection_package_roundtrip_preserves_layers_and_metadata(tmp_path):
	"""Keep full projection-package context when importing from VirtualXRay and exporting again."""
	virtual_xray = VirtualXRay()
	virtual_xray.label = "vx-package"
	virtual_xray.last_raw_projection = np.array([[1.0, 2.0]], dtype=np.float32)
	virtual_xray.last_line_integral_projection = np.array([[3.0, 4.0]], dtype=np.float32)
	virtual_xray.last_source_projections = [
		type("SourceProjection", (), {
			"source_index": 0,
			"label": "Bone",
			"source_type": "mesh",
			"detector_image": np.array([[5.0, 6.0]], dtype=np.float32),
			"line_integral_image": np.array([[7.0, 8.0]], dtype=np.float32),
		})()
	]
	virtual_xray.last_projected_annotations = XRayOverlayProjectionSet(
		detector_shape_hw=(1, 2),
		items=[
			XRayOverlayCross(
				kind="AnnotationPoint",
				label="P1",
				pixel_uv=(0.0, 0.0),
				style=XRayOverlayStyle(color_rgba=(255, 0, 0, 255), line_width_px=1, marker_size_px=5),
			)
		],
	)
	virtual_xray.set_detector_image_defaults({"mode": "film", "gamma": 1.5})

	detector_image = DetectorImage()
	detector_image.sync_from_virtual_xray(virtual_xray, auto_window=False)

	layer_keys = [layer_key for layer_key, _layer_label in detector_image.package_layer_choices()]
	assert "composited_raw" in layer_keys
	assert "composited_line_integral" in layer_keys
	assert "source_000_raw" in layer_keys
	assert "source_000_line_integral" in layer_keys
	assert detector_image.package_metadata["simulation_context"]["detector_image_defaults"]["mode"] == "film"

	export_path = tmp_path / "detector_package.npz"
	detector_image.set_active_layer("source_000_line_integral", auto_window=False)
	detector_image.export_array(export_path)
	with np.load(export_path, allow_pickle=False) as archive:
		assert "metadata_json" in archive
		assert "image" not in archive
		assert any(name.startswith("image_") for name in archive.files)

	imported = DetectorImage()
	imported.import_array(export_path, auto_window=False)

	assert imported.active_layer_key == "source_000_line_integral"
	assert np.allclose(imported.raw_array, [[7.0, 8.0]], atol=1e-6)
	assert imported.package_metadata["simulation_context"]["source_virtual_xray_label"] == "vx-package"
	assert imported.overlay_projection_set is not None
	assert len(imported.overlay_projection_set.items) == 1

	imported.set_active_layer("composited_raw", auto_window=False)
	assert np.allclose(imported.raw_array, [[1.0, 2.0]], atol=1e-6)


def test_detector_image_defaults_roundtrip_preserves_log_and_clahe_settings(tmp_path):
	"""Keep extended presentation settings when exporting and importing one detector package."""
	detector_image = DetectorImage()
	detector_image.label = "detector-defaults"
	detector_image.setArray(np.array([[0.0, 10.0, 1000.0]], dtype=np.float32), auto_window=False)
	detector_image.apply_presentation_defaults({
		"mode": "digital",
		"gamma": 1.3,
		"contrast": 1.1,
		"input_transform": "log1p",
		"local_enhancement": "clahe",
		"clahe_clip_limit": 3.0,
		"clahe_tile_grid_size": 12,
		"robust_low_percentile": 1.5,
	})

	export_path = tmp_path / "detector_defaults_roundtrip.npz"
	detector_image.export_array(export_path)

	imported = DetectorImage()
	imported.import_array(export_path, auto_window=False)

	assert imported.presentation_input_transform == "log1p"
	assert imported.presentation_local_enhancement == "clahe"
	assert imported.presentation_clahe_clip_limit == 3.0
	assert imported.presentation_clahe_tile_grid_size == 12
	assert imported.presentation_robust_low_percentile == 1.5


def test_detector_image_defaults_roundtrip_preserves_exposure_fusion_settings(tmp_path):
	"""Keep optional exposure-fusion presentation settings in the detector package."""
	detector_image = DetectorImage()
	detector_image.label = "detector-fusion-defaults"
	detector_image.setArray(np.array([[0.0, 1.0, 2.0, 50.0]], dtype=np.float32), auto_window=False)
	detector_image.apply_presentation_defaults({
		"mode": "digital",
		"exposure_fusion_enabled": True,
		"exposure_fusion_profile": "soft_tissue",
		"exposure_fusion_strength": 0.72,
	})

	export_path = tmp_path / "detector_fusion_defaults_roundtrip.npz"
	detector_image.export_array(export_path)

	imported = DetectorImage()
	imported.import_array(export_path, auto_window=False)

	assert imported.presentation_exposure_fusion_enabled is True
	assert imported.presentation_exposure_fusion_profile == "soft_tissue"
	assert np.isclose(imported.presentation_exposure_fusion_strength, 0.72)


def test_detector_image_optional_exposure_fusion_changes_presented_array():
	"""Blend several presentation variants only when the optional fusion toggle is enabled."""
	detector_image = DetectorImage()
	detector_image.setArray(np.array([[0.0, 4.0, 8.0, 32.0, 128.0]], dtype=np.float32), auto_window=False)
	detector_image.presentation_mode = "digital"
	detector_image.presentation_invert = False
	detector_image.presentation_gamma = 1.0
	detector_image.presentation_contrast = 1.0
	detector_image.presentation_window_center = 16.0
	detector_image.presentation_window_width = 32.0

	base = detector_image.get_presented_array()
	detector_image.presentation_exposure_fusion_enabled = True
	detector_image.presentation_exposure_fusion_profile = "balanced"
	detector_image.presentation_exposure_fusion_strength = 1.0
	fused = detector_image.get_presented_array()

	assert fused.dtype == np.float32
	assert fused.shape == base.shape
	assert float(np.min(fused)) >= 0.0
	assert float(np.max(fused)) <= 1.0
	assert not np.allclose(fused, base)


def test_detector_image_exposure_fusion_profiles_produce_distinct_results():
	"""Use profile selection to change the presentation character of exposure fusion."""
	detector_image = DetectorImage()
	detector_image.setArray(np.array([[0.0, 3.0, 9.0, 24.0, 80.0, 200.0]], dtype=np.float32), auto_window=False)
	detector_image.presentation_mode = "digital"
	detector_image.presentation_invert = False
	detector_image.presentation_gamma = 1.0
	detector_image.presentation_contrast = 1.0
	detector_image.presentation_window_center = 20.0
	detector_image.presentation_window_width = 40.0
	detector_image.presentation_exposure_fusion_enabled = True
	detector_image.presentation_exposure_fusion_strength = 1.0

	detector_image.presentation_exposure_fusion_profile = "soft_tissue"
	soft_tissue = detector_image.get_presented_array()
	detector_image.presentation_exposure_fusion_profile = "bone_preserving"
	bone_preserving = detector_image.get_presented_array()

	assert soft_tissue.dtype == np.float32
	assert bone_preserving.dtype == np.float32
	assert soft_tissue.shape == bone_preserving.shape
	assert not np.allclose(soft_tissue, bone_preserving)


def test_detector_image_auto_window_uses_two_sided_robust_percentiles():
	"""Use both lower and upper robust percentiles when computing one automatic display window."""
	detector_image = DetectorImage()
	detector_image.setArray(np.array([[0.0, 1.0, 2.0, 100.0]], dtype=np.float32), auto_window=False)
	detector_image.presentation_robust_low_percentile = 25.0
	detector_image.presentation_robust_percentile = 75.0

	vmin, vmax = detector_image.auto_window_range()

	assert np.isclose(vmin, 0.75)
	assert np.isclose(vmax, 26.5)


def test_detector_image_transfer_curve_preview_data_uses_presented_input_histogram():
	"""Build one normalized histogram from the pre-curve presented image and expose curve points."""
	detector_image = DetectorImage()
	detector_image.setArray(np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32), auto_window=False)
	detector_image.presentation_mode = "raw"
	detector_image.presentation_window_center = 1.5
	detector_image.presentation_window_width = 3.0
	detector_image.set_transfer_points_pct([(0.0, 10.0), (50.0, 90.0), (100.0, 100.0)])

	preview = detector_image.transfer_curve_preview_data(histogram_bins=4)

	assert preview["histogram"].shape == (8,)
	assert np.isclose(float(np.max(preview["histogram"])), 1.0)
	assert np.allclose(preview["curve_x"], [0.0, 0.5, 1.0], atol=1e-6)
	assert np.allclose(preview["curve_y"], [0.1, 0.9, 1.0], atol=1e-6)


def test_detector_image_transfer_curve_preview_data_ignores_zero_background_when_possible():
	"""Skip zero-valued background samples so the preview histogram shows useful occupied bins."""
	detector_image = DetectorImage()
	detector_image.setArray(np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32), auto_window=False)
	detector_image.presentation_mode = "raw"
	detector_image.presentation_window_center = 1.5
	detector_image.presentation_window_width = 3.0

	preview = detector_image.transfer_curve_preview_data(histogram_bins=8, ignore_zero_values=True)

	assert preview["histogram"].shape == (8,)
	assert np.isclose(preview["histogram"][0], 0.0)
	assert np.count_nonzero(preview["histogram"]) >= 2


def test_virtual_xray_scene_sources_use_active_motion_frame_by_default():
	"""Use only the active `Motion` frame unless the expansion mode requests all frames."""
	virtual_xray = VirtualXRay()
	parent_transform = _translated_transform(10.0, 0.0, 0.0)
	virtual_xray.addChild(parent_transform)

	motion = Motion([
		Motion.FrameVal(0, _translated_transform(1.0, 2.0, 3.0)),
		Motion.FrameVal(0, _translated_transform(100.0, 200.0, 300.0)),
	], parent=parent_transform)
	parent_transform.addChild(motion)
	motion.setKey(1)

	child_transform = _translated_transform(0.5, 0.0, 0.0)
	motion.addChild(child_transform)

	mesh = Mesh(parent=child_transform)
	child_transform.addChild(mesh)
	mesh.addVertex(0.0, 0.0, 0.0)
	mesh.addVertex(1.0, 0.0, 0.0)
	mesh.addVertex(0.0, 1.0, 0.0)
	mesh.addFace([0, 1, 2])

	volume = Volumetric(parent=child_transform)
	child_transform.addChild(volume)
	volume.m_volume = np.zeros((1, 1, 1), dtype=np.float32)
	volume.shape = volume.m_volume.shape
	volume.metadata = []

	sources = virtual_xray.scene_sources()
	mesh_sources = [source for source in sources if hasattr(source, "mesh") and source.mesh is mesh]
	volume_sources = [source for source in sources if hasattr(source, "volumetric") and source.volumetric is volume]

	assert len(mesh_sources) == 1
	assert len(volume_sources) == 1
	expected = np.eye(4, dtype=np.float32)
	expected[:3, 3] = [110.5, 200.0, 300.0]
	assert np.allclose(mesh_sources[0].global_transform, expected, atol=1e-6)
	assert np.allclose(volume_sources[0].global_transform, expected, atol=1e-6)


def test_virtual_xray_scene_sources_expand_all_frames_when_requested():
	"""Expand one `Motion` subtree into separate source instances for all frames."""
	virtual_xray = VirtualXRay()
	virtual_xray.motion_frame_mode = "all"
	parent_transform = _translated_transform(10.0, 0.0, 0.0)
	virtual_xray.addChild(parent_transform)

	motion = Motion([
		Motion.FrameVal(0, _translated_transform(1.0, 2.0, 3.0)),
		Motion.FrameVal(0, _translated_transform(100.0, 200.0, 300.0)),
	], parent=parent_transform)
	parent_transform.addChild(motion)
	motion.setKey(1)

	child_transform = _translated_transform(0.5, 0.0, 0.0)
	motion.addChild(child_transform)

	mesh = Mesh(parent=child_transform)
	child_transform.addChild(mesh)
	mesh.addVertex(0.0, 0.0, 0.0)
	mesh.addVertex(1.0, 0.0, 0.0)
	mesh.addVertex(0.0, 1.0, 0.0)
	mesh.addFace([0, 1, 2])

	sources = virtual_xray.scene_sources()
	mesh_sources = [source for source in sources if hasattr(source, "mesh") and source.mesh is mesh]

	assert len(mesh_sources) == 2
	assert [source.projection_label for source in mesh_sources] == [
		"Mesh [frame 000]",
		"Mesh [frame 001]",
	]

	expected0 = np.eye(4, dtype=np.float32)
	expected0[:3, 3] = [11.5, 2.0, 3.0]
	expected1 = np.eye(4, dtype=np.float32)
	expected1[:3, 3] = [110.5, 200.0, 300.0]
	assert np.allclose(mesh_sources[0].global_transform, expected0, atol=1e-6)
	assert np.allclose(mesh_sources[1].global_transform, expected1, atol=1e-6)


def test_virtual_xray_scene_source_groups_expand_single_motion_into_all_frames():
	"""Build one source group per frame when the subtree contains a single `Motion`."""
	virtual_xray = VirtualXRay()
	parent_transform = _translated_transform(10.0, 0.0, 0.0)
	virtual_xray.addChild(parent_transform)

	motion = Motion([
		Motion.FrameVal(0, _translated_transform(1.0, 2.0, 3.0)),
		Motion.FrameVal(0, _translated_transform(4.0, 5.0, 6.0)),
	], parent=parent_transform)
	parent_transform.addChild(motion)
	motion.setKey(1)

	child_transform = _translated_transform(0.5, 0.0, 0.0)
	motion.addChild(child_transform)

	mesh = Mesh(parent=child_transform)
	child_transform.addChild(mesh)
	mesh.addVertex(0.0, 0.0, 0.0)
	mesh.addVertex(1.0, 0.0, 0.0)
	mesh.addVertex(0.0, 1.0, 0.0)
	mesh.addFace([0, 1, 2])

	groups = virtual_xray.scene_source_groups()

	assert [group["frame_index"] for group in groups] == [0, 1]
	assert [group["label"] for group in groups] == ["frame_000", "frame_001"]

	frame0_mesh_source = next(source for source in groups[0]["sources"] if hasattr(source, "mesh") and source.mesh is mesh)
	frame1_mesh_source = next(source for source in groups[1]["sources"] if hasattr(source, "mesh") and source.mesh is mesh)

	expected0 = np.eye(4, dtype=np.float32)
	expected0[:3, 3] = [11.5, 2.0, 3.0]
	expected1 = np.eye(4, dtype=np.float32)
	expected1[:3, 3] = [14.5, 5.0, 6.0]

	assert np.allclose(frame0_mesh_source.global_transform, expected0, atol=1e-6)
	assert np.allclose(frame1_mesh_source.global_transform, expected1, atol=1e-6)
