# -*- coding: utf-8 -*-
"""Unit tests for scalar preprocessing and X-ray physics response helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from plugins.virtRTG.xray.xrayProjection import XRayPhysicsModel, XRayScalarPreprocessor
from plugins.virtRTG.xray.xraySource import (
	MeshXRaySource,
	VolumetricXRaySource,
	XRayMaterialResponseConfig,
	get_xray_material_response_config,
	set_xray_material_response_config,
)


def test_scalar_preprocessor_percentile_rescale_maps_values_into_output_range():
	"""Rescale scalar values into one configured pseudo-HU output interval."""
	preprocessor = XRayScalarPreprocessor(
		mode="percentile_rescale",
		input_low_percentile=0.0,
		input_high_percentile=100.0,
		output_low_value=-1000.0,
		output_high_value=1000.0,
	)
	volume = np.array([0.0, 5.0, 10.0], dtype=np.float32)
	stats = preprocessor.estimate_volume_stats(volume)
	rescaled = preprocessor.apply(np.array([0.0, 5.0, 10.0], dtype=np.float32), stats)

	assert np.allclose(rescaled, [-1000.0, 0.0, 1000.0])


def test_scalar_to_mu_linear_mode_clamps_negative_relative_density():
	"""Map HU-like scalar values to non-negative attenuation in linear mode."""
	model = XRayPhysicsModel(mu_air=0.0, mu_water=0.02, hounsfield_air=-1000.0)
	values = np.array([-1500.0, -1000.0, 0.0], dtype=np.float32)
	mu = model.scalar_to_mu(values)

	assert np.isclose(mu[0], 0.0)
	assert np.isclose(mu[1], 0.0)
	assert np.isclose(mu[2], 0.02)


def test_integral_to_image_intensity_mode_applies_beer_lambert_response():
	"""Convert line integrals to detector intensity with exponential attenuation."""
	model = XRayPhysicsModel(output_mode="intensity", intensity_floor=0.0)
	intensity = model.integral_to_image(np.array([0.0, np.log(2.0)], dtype=np.float32))

	assert np.allclose(intensity, [1.0, 0.5], atol=1e-6)


def test_source_distance_gain_uses_inverse_power_falloff():
	"""Apply inverse-square-like gain relative to one reference source distance."""
	model = XRayPhysicsModel(
		source_distance_falloff_mode="inverse_square",
		source_distance_reference_mm=1000.0,
		source_distance_power=2.0,
	)
	gain = model.source_distance_gain(np.array([1000.0, 2000.0], dtype=np.float32), projection_mode="cone")

	assert np.allclose(gain, [1.0, 0.25], atol=1e-6)


def test_volumetric_source_local_material_override_can_replace_window_and_response():
	"""Allow one volumetric source to override the shared material interpretation."""
	source = VolumetricXRaySource.__new__(VolumetricXRaySource)
	source.material_response_config = XRayMaterialResponseConfig(
		enabled=True,
		mode="bone_threshold",
		bone_threshold_hu=450.0,
		bone_threshold_softness=35.0,
		window_center=700.0,
		window_width=300.0,
		window_mode="sigmoid",
		window_softness=20.0,
	)

	global_model = XRayPhysicsModel(
		material_response_mode="linear",
		bone_threshold_hu=900.0,
		bone_threshold_softness=120.0,
		material_window_center=None,
		material_window_width=None,
		material_window_mode="hard",
		material_window_softness=150.0,
	)

	effective_model = source.resolve_physics_model(global_model)

	assert effective_model.material_response_mode == "bone_threshold"
	assert effective_model.bone_threshold_hu == 450.0
	assert effective_model.bone_threshold_softness == 35.0
	assert effective_model.material_window_center == 700.0
	assert effective_model.material_window_width == 300.0
	assert effective_model.material_window_mode == "sigmoid"
	assert effective_model.material_window_softness == 20.0


def test_mesh_source_without_override_keeps_shared_material_model():
	"""Leave the shared acquisition physics unchanged when no mesh override is enabled."""
	source = MeshXRaySource.__new__(MeshXRaySource)
	source.material_response_config = XRayMaterialResponseConfig(enabled=False)

	global_model = XRayPhysicsModel(
		material_response_mode="piecewise_soft_tissue",
		material_window_center=250.0,
		material_window_width=800.0,
		material_window_mode="linear",
		material_window_softness=60.0,
	)

	effective_model = source.resolve_physics_model(global_model)

	assert effective_model is global_model


def test_material_response_config_roundtrip_syncs_with_legacy_source_fields():
	"""Store and retrieve the new per-source config object without a legacy mirror."""
	source = SimpleNamespace()
	config = XRayMaterialResponseConfig(
		enabled=True,
		mode="piecewise_bone",
		bone_threshold_hu=500.0,
		bone_threshold_softness=45.0,
		window_center=350.0,
		window_width=900.0,
		window_mode="linear",
		window_softness=30.0,
	)

	set_xray_material_response_config(source, config)
	roundtrip = get_xray_material_response_config(source)

	assert roundtrip == config.normalized()
	assert source.xray_material_response_config == config.normalized()
