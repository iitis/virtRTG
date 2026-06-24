from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import logging
import os
from time import perf_counter
from typing import Iterable, Sequence

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class XRayPresentationModel(ABC):
	"""Transform raw projection output into a presentation-ready image."""

	@abstractmethod
	def apply(self, image):
		"""Return a presentation image derived from the raw projection result."""


def fuse_exposure_stack(images, strength=1.0, sigma=0.28, contrast_boost=2.0):
	"""Blend multiple normalized presentation variants into one exposure-fused image.

	The fusion is intentionally lightweight and presentation-only. It combines
	well-exposedness with a small local-contrast term so the output can improve
	readability without changing the raw projection physics.
	"""
	stack = np.asarray(images, dtype=np.float32)
	if stack.ndim != 3 or stack.shape[0] <= 0:
		raise ValueError("Exposure fusion expects a stack shaped as (N, H, W).")
	stack = np.clip(stack, 0.0, 1.0)
	if stack.shape[0] == 1:
		return stack[0].astype(np.float32, copy=True)

	sigma = max(1e-3, float(sigma))
	contrast_boost = max(0.0, float(contrast_boost))
	strength = min(1.0, max(0.0, float(strength)))
	base = stack[min(stack.shape[0] // 2, stack.shape[0] - 1)]

	exposure_weight = np.exp(-0.5 * np.square((stack - 0.5) / sigma)).astype(np.float32, copy=False)
	contrast_weight = np.empty_like(stack, dtype=np.float32)
	for index, image in enumerate(stack):
		laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
		contrast_weight[index] = np.abs(laplacian).astype(np.float32, copy=False)
	if contrast_boost > 0.0:
		contrast_weight = np.power(contrast_weight + 1e-6, contrast_boost).astype(np.float32, copy=False)
	weights = exposure_weight * (0.10 + contrast_weight)
	weights += 1e-6
	weights_sum = np.sum(weights, axis=0, keepdims=True)
	fused = np.sum(stack * weights, axis=0) / np.maximum(weights_sum[0], 1e-6)
	if strength <= 0.0:
		return base.astype(np.float32, copy=True)
	return np.clip(base * (1.0 - strength) + fused * strength, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_range(image, vmin, vmax):
	"""Return one clipped `[0, 1]` normalization for the provided numeric range."""
	return np.clip((image - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)


def _apply_input_transform(image, transform_name):
	"""Apply one global input-domain transform before range normalization."""
	image = np.asarray(image, dtype=np.float32)
	transform = str(transform_name or "linear").strip().lower()
	if transform == "log1p":
		finite_values = image[np.isfinite(image)]
		if finite_values.size == 0:
			return np.zeros(image.shape, dtype=np.float32)
		offset = float(-np.min(finite_values)) if float(np.min(finite_values)) < 0.0 else 0.0
		return np.log1p(np.clip(image + offset, 0.0, None)).astype(np.float32, copy=False)
	return image.astype(np.float32, copy=False)


def _apply_local_enhancement(normalized, local_enhancement, clahe_clip_limit, clahe_tile_grid_size):
	"""Apply one optional local contrast enhancement to a normalized float image."""
	mode = str(local_enhancement or "off").strip().lower()
	if mode != "clahe":
		return np.asarray(normalized, dtype=np.float32)
	image_u8 = np.clip(np.asarray(normalized, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
	tile_size = max(1, int(clahe_tile_grid_size))
	clahe = cv2.createCLAHE(
		clipLimit=max(0.01, float(clahe_clip_limit)),
		tileGridSize=(tile_size, tile_size),
	)
	return (clahe.apply(image_u8).astype(np.float32) / 255.0).astype(np.float32, copy=False)



@dataclass
class RawPresentationModel(XRayPresentationModel):
	"""Return the raw projection output without any presentation processing."""

	def apply(self, image):
		"""Return a float32 copy of the acquisition image."""
		return np.asarray(image, dtype=np.float32).copy()

@dataclass
class FilmLikePresentationModel(XRayPresentationModel):
	"""Apply a simple inverted, non-linear film-like tone curve."""

	robust_percentile: float = 99.5
	robust_low_percentile: float = 0.5
	gamma: float = 1.4
	contrast: float = 1.0
	invert: bool = True
	fixed_range: Sequence[float] | None = None
	input_transform: str = "linear"
	local_enhancement: str = "off"
	clahe_clip_limit: float = 2.0
	clahe_tile_grid_size: int = 8

	def apply(self, image):
		"""Return a film-like normalized float image in the range `[0, 1]`."""
		image = _apply_input_transform(image, self.input_transform)
		finite_values = image[np.isfinite(image)]
		if finite_values.size == 0:
			return np.zeros(image.shape, dtype=np.float32)

		if self.fixed_range is not None:
			vmin = float(self.fixed_range[0])
			vmax = float(self.fixed_range[1])
		else:
			vmin = float(np.percentile(finite_values, float(self.robust_low_percentile)))
			vmax = float(np.percentile(finite_values, float(self.robust_percentile)))
			if vmax <= vmin:
				vmin = float(np.min(finite_values))
			if vmax <= vmin:
				vmax = float(np.max(finite_values))
			if vmax <= vmin:
				vmax = vmin + 1.0

		normalized = _normalize_range(image, vmin, vmax)
		normalized = _apply_local_enhancement(
			normalized,
			self.local_enhancement,
			self.clahe_clip_limit,
			self.clahe_tile_grid_size,
		)
		if self.invert:
			normalized = 1.0 - normalized
		normalized = np.clip(0.5 + (normalized - 0.5) * float(self.contrast), 0.0, 1.0)
		gamma = max(1e-6, float(self.gamma))
		return np.power(normalized, 1.0 / gamma).astype(np.float32, copy=False)


@dataclass
class DigitalRadiographyPresentationModel(XRayPresentationModel):
	"""Apply a simple digital-radiography style windowing and tone mapping."""

	window_center: float | None = None
	window_width: float | None = None
	robust_percentile: float = 99.5
	robust_low_percentile: float = 0.5
	invert: bool = True
	gamma: float = 1.0
	contrast: float = 1.0
	input_transform: str = "linear"
	local_enhancement: str = "off"
	clahe_clip_limit: float = 2.0
	clahe_tile_grid_size: int = 8

	def apply(self, image):
		"""Return a digital-radiography style normalized float image in the range `[0, 1]`."""
		
		if (image is None) or (not np.any(np.isfinite(image))):
			_log.warning("Input image is empty or contains no finite values. Returning zero image.")
			return np.zeros(image.shape, dtype=np.float32)
		
		image = _apply_input_transform(image, self.input_transform)
		finite_values = image[np.isfinite(image)]
		if finite_values.size == 0:
			return np.zeros(image.shape, dtype=np.float32)

		if self.window_center is not None and self.window_width is not None and float(self.window_width) > 0.0:
			vmin = float(self.window_center) - float(self.window_width) / 2.0
			vmax = float(self.window_center) + float(self.window_width) / 2.0
		else:
			vmin = float(np.percentile(finite_values, float(self.robust_low_percentile)))
			vmax = float(np.percentile(finite_values, float(self.robust_percentile)))
			if vmax <= vmin:
				vmin = float(np.min(finite_values))
			if vmax <= vmin:
				vmax = float(np.max(finite_values))
			if vmax <= vmin:
				vmax = vmin + 1.0

		normalized = _normalize_range(image, vmin, vmax)
		normalized = _apply_local_enhancement(
			normalized,
			self.local_enhancement,
			self.clahe_clip_limit,
			self.clahe_tile_grid_size,
		)
		if self.invert:
			normalized = 1.0 - normalized
		normalized = np.clip(0.5 + (normalized - 0.5) * float(self.contrast), 0.0, 1.0)
		gamma = max(1e-6, float(self.gamma))
		return np.power(normalized, 1.0 / gamma).astype(np.float32, copy=False)
