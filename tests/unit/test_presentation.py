# -*- coding: utf-8 -*-
"""Unit tests for presentation-layer image normalization models."""

from __future__ import annotations

import numpy as np

from plugins.virtRTG.xray.xrayPresentation import (
	DigitalRadiographyPresentationModel,
	FilmLikePresentationModel,
	RawPresentationModel,
	fuse_exposure_stack,
)


def test_raw_presentation_model_returns_float_copy(sample_projection_image):
	"""Return a float32 copy so presentation does not alias the input array."""
	model = RawPresentationModel()
	result = model.apply(sample_projection_image)

	assert result.dtype == np.float32
	assert np.array_equal(result, sample_projection_image)
	assert result is not sample_projection_image


def test_film_like_presentation_model_normalizes_into_unit_interval(sample_projection_image):
	"""Normalize film-like output to the inclusive range `[0, 1]`."""
	model = FilmLikePresentationModel(invert=True, gamma=1.4, contrast=1.0, fixed_range=(0.0, 5.0))
	result = model.apply(sample_projection_image)

	assert result.dtype == np.float32
	assert float(np.min(result)) >= 0.0
	assert float(np.max(result)) <= 1.0
	assert result[0, 0] > result[1, 2]


def test_digital_radiography_windowing_respects_explicit_window(sample_projection_image):
	"""Apply one explicit centre-width window before digital presentation mapping."""
	model = DigitalRadiographyPresentationModel(
		window_center=2.5,
		window_width=5.0,
		invert=False,
		gamma=1.0,
		contrast=1.0,
	)
	result = model.apply(sample_projection_image)

	assert result.dtype == np.float32
	assert np.isclose(result[0, 0], 0.0)
	assert np.isclose(result[1, 2], 1.0)


def test_digital_radiography_log1p_compresses_high_dynamic_range():
	"""Compress extreme DRR-like values before normalization when `log1p` is enabled."""
	image = np.array([[0.0, 10.0, 1000.0]], dtype=np.float32)
	linear = DigitalRadiographyPresentationModel(
		invert=False,
		gamma=1.0,
		contrast=1.0,
		input_transform="linear",
	).apply(image)
	logged = DigitalRadiographyPresentationModel(
		invert=False,
		gamma=1.0,
		contrast=1.0,
		input_transform="log1p",
	).apply(image)

	assert linear.dtype == np.float32
	assert logged.dtype == np.float32
	assert logged[0, 1] > linear[0, 1]


def test_digital_radiography_clahe_increases_local_contrast():
	"""Increase local contrast on a low-contrast gradient when CLAHE is enabled."""
	image = np.tile(np.linspace(100.0, 120.0, 32, dtype=np.float32), (32, 1))
	base = DigitalRadiographyPresentationModel(
		invert=False,
		gamma=1.0,
		contrast=1.0,
		local_enhancement="off",
	).apply(image)
	enhanced = DigitalRadiographyPresentationModel(
		invert=False,
		gamma=1.0,
		contrast=1.0,
		local_enhancement="clahe",
		clahe_clip_limit=2.0,
		clahe_tile_grid_size=8,
	).apply(image)

	assert enhanced.dtype == np.float32
	assert float(np.max(np.abs(np.diff(enhanced, axis=1)))) >= float(np.max(np.abs(np.diff(base, axis=1))))


def test_exposure_fusion_stack_returns_unit_interval_blend():
	"""Blend multiple display-ready presentation variants into one bounded image."""
	under = np.array([[0.10, 0.25, 0.35]], dtype=np.float32)
	base = np.array([[0.20, 0.50, 0.80]], dtype=np.float32)
	over = np.array([[0.40, 0.70, 0.95]], dtype=np.float32)

	result = fuse_exposure_stack(np.stack([under, base, over], axis=0), strength=0.85)

	assert result.dtype == np.float32
	assert result.shape == base.shape
	assert float(np.min(result)) >= 0.0
	assert float(np.max(result)) <= 1.0
	assert not np.allclose(result, base)
