# -*- coding: utf-8 -*-
"""Unit tests for presentation-layer image normalization models."""

from __future__ import annotations

import numpy as np

from plugins.virtRTG.xray.xrayPresentation import (
	DigitalRadiographyPresentationModel,
	FilmLikePresentationModel,
	RawPresentationModel,
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

