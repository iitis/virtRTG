# -*- coding: utf-8 -*-
"""Unit tests for scalar preprocessing and X-ray physics response helpers."""

from __future__ import annotations

import numpy as np

from plugins.virtRTG.xray.xrayProjection import XRayPhysicsModel, XRayScalarPreprocessor


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

