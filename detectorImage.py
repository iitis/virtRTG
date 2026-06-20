# -*- coding: utf-8 -*-
"""Plugin-local 2D detector image object with presentation and overlay support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QColor, QImage, QPainter, QPen

from dpVision import Object

from .xray.xrayAnnotationOverlay import XRayOverlayCross, XRayOverlayPolyline
from .xray.xrayAnnotationOverlay import (
	overlay_projection_set_from_payload,
	overlay_projection_set_to_payload,
)
from .xray.xrayPresentation import (
	DigitalRadiographyPresentationModel,
	FilmLikePresentationModel,
	RawPresentationModel,
)


class DetectorImage(Object):
	"""Represent one flat detector-space float image stored in the scene tree."""

	PACKAGE_SCHEMA = "virtRTG-detector-package"
	PACKAGE_VERSION = 1

	def __init__(self, parent=None, array=None, source_stage="raw"):
		"""Initialize one hidden-in-3D detector image object."""
		super().__init__(parent)
		self.visible = False
		self.source_stage = str(source_stage)
		self.source_virtual_xray_label = ""
		self.raw_array = None
		self.presentation_mode = "digital"
		self.presentation_invert = False
		self.presentation_gamma = 1.0
		self.presentation_contrast = 1.0
		self.presentation_input_transform = "linear"
		self.presentation_local_enhancement = "off"
		self.presentation_clahe_clip_limit = 2.0
		self.presentation_clahe_tile_grid_size = 8
		self.presentation_robust_low_percentile = 0.5
		self.presentation_robust_percentile = 99.5
		self.presentation_window_center = None
		self.presentation_window_width = None
		self.presentation_overlay_annotations = False
		self.presentation_overlay_labels = False
		self.presentation_overlay_cross_size_px = 6
		self.overlay_projection_set = None
		self.transfer_points_pct = [(0.0, 0.0), (100.0, 100.0)]
		self.display_only_window_range = False
		self.package_images = {}
		self.package_layers = []
		self.package_metadata = {}
		self.active_layer_key = None
		if array is not None:
			self.setArray(array)

	@property
	def window_center(self):
		"""Backward-compatible alias for the presentation window center."""
		return self.presentation_window_center

	@window_center.setter
	def window_center(self, value):
		"""Backward-compatible alias for the presentation window center."""
		self.presentation_window_center = None if value is None else float(value)

	@property
	def window_width(self):
		"""Backward-compatible alias for the presentation window width."""
		return self.presentation_window_width

	@window_width.setter
	def window_width(self, value):
		"""Backward-compatible alias for the presentation window width."""
		if value is None:
			self.presentation_window_width = None
			return
		width = float(value)
		self.presentation_window_width = width if width > 0.0 else 0.0

	@property
	def display_invert(self):
		"""Backward-compatible alias for the presentation invert flag."""
		return self.presentation_invert

	@display_invert.setter
	def display_invert(self, value):
		"""Backward-compatible alias for the presentation invert flag."""
		self.presentation_invert = bool(value)

	@property
	def display_gamma(self):
		"""Backward-compatible alias for the presentation gamma."""
		return self.presentation_gamma

	@display_gamma.setter
	def display_gamma(self, value):
		"""Backward-compatible alias for the presentation gamma."""
		self.presentation_gamma = max(0.05, float(value))

	@property
	def display_contrast(self):
		"""Backward-compatible alias for the presentation contrast."""
		return self.presentation_contrast

	@display_contrast.setter
	def display_contrast(self, value):
		"""Backward-compatible alias for the presentation contrast."""
		self.presentation_contrast = max(0.05, float(value))

	@property
	def display_robust_percentile(self):
		"""Backward-compatible alias for the presentation robust percentile."""
		return self.presentation_robust_percentile

	@display_robust_percentile.setter
	def display_robust_percentile(self, value):
		"""Backward-compatible alias for the presentation robust percentile."""
		self.presentation_robust_percentile = min(100.0, max(50.0, float(value)))

	def renderSelf(self):
		"""Skip 3D rendering; this object is intended for the scene tree and 2D viewers only."""
		return

	def getLocalBB(self):
		"""Return an empty bounding box because the object is not rendered in 3D."""
		return False, None, None

	def info(self):
		"""Return a concise detector image summary."""
		if self.raw_array is None:
			return "DetectorImage: empty"
		height, width = self.shape_hw()
		stats = self.data_stats()
		return (
			f"DetectorImage: {width} x {height} px, stage={self.source_stage}, "
			f"mode={self.presentation_mode}, min={stats['min']:.4g}, max={stats['max']:.4g}"
		)

	def shape_hw(self):
		"""Return the detector shape as `(height, width)`."""
		if self.raw_array is None:
			return 0, 0
		return int(self.raw_array.shape[0]), int(self.raw_array.shape[1])

	def data_stats(self):
		"""Return finite-value statistics used by the viewer and property panel."""
		if self.raw_array is None:
			return {"min": 0.0, "max": 1.0, "mean": 0.0}
		finite_values = np.asarray(self.raw_array, dtype=np.float32)
		finite_values = finite_values[np.isfinite(finite_values)]
		if finite_values.size == 0:
			return {"min": 0.0, "max": 1.0, "mean": 0.0}
		return {
			"min": float(np.min(finite_values)),
			"max": float(np.max(finite_values)),
			"mean": float(np.mean(finite_values)),
		}

	def auto_window_range(self, robust_percentile=None, robust_low_percentile=None):
		"""Return one automatic display range based on finite values."""
		if self.raw_array is None:
			return 0.0, 1.0
		finite_values = np.asarray(self.raw_array, dtype=np.float32)
		finite_values = finite_values[np.isfinite(finite_values)]
		if finite_values.size == 0:
			return 0.0, 1.0
		high_percentile = float(
			self.presentation_robust_percentile if robust_percentile is None else robust_percentile
		)
		low_percentile = float(
			self.presentation_robust_low_percentile
			if robust_low_percentile is None else robust_low_percentile
		)
		high_percentile = min(100.0, max(50.0, high_percentile))
		low_percentile = min(high_percentile - 1e-6, max(0.0, low_percentile))
		vmin = float(np.percentile(finite_values, low_percentile))
		vmax = float(np.percentile(finite_values, high_percentile))
		if vmax <= vmin:
			vmin = float(np.min(finite_values))
		if vmax <= vmin:
			vmax = float(np.max(finite_values))
		if vmax <= vmin:
			vmax = vmin + 1.0
		return vmin, vmax

	def reset_window_to_full_range(self):
		"""Set the presentation window to the full finite range of the current array."""
		stats = self.data_stats()
		vmin = stats["min"]
		vmax = stats["max"]
		if vmax <= vmin:
			vmax = vmin + 1.0
		self.presentation_window_center = (vmin + vmax) / 2.0
		self.presentation_window_width = vmax - vmin

	def auto_window(self, robust_percentile=None, robust_low_percentile=None):
		"""Set the presentation window using the robust automatic range."""
		vmin, vmax = self.auto_window_range(
			robust_percentile=robust_percentile,
			robust_low_percentile=robust_low_percentile,
		)
		self.presentation_window_center = (vmin + vmax) / 2.0
		self.presentation_window_width = max(vmax - vmin, 1e-6)

	def setArray(self, array, source_stage=None, auto_window=True):
		"""Replace the stored detector array and optionally reset the display window."""
		array = np.asarray(array, dtype=np.float32)
		if array.ndim != 2:
			raise ValueError("DetectorImage expects a 2D float32 array.")
		self.raw_array = np.array(array, dtype=np.float32, copy=True)
		if source_stage is not None:
			self.source_stage = str(source_stage)
		if auto_window:
			self.auto_window()

	def _normalize_package_layer(self, layer):
		"""Normalize one package-layer mapping used for multi-image detector bundles."""
		layer = {} if layer is None else dict(layer)
		key = str(layer.get("key", "")).strip()
		if key == "":
			raise ValueError("Package layer is missing a non-empty key.")
		return {
			"key": key,
			"label": str(layer.get("label", key)),
			"stage": str(layer.get("stage", "raw")),
			"role": str(layer.get("role", "derived")),
			"source_index": None if layer.get("source_index", None) is None else int(layer.get("source_index")),
			"source_label": str(layer.get("source_label", "")),
			"source_type": str(layer.get("source_type", "")),
		}

	def clear_projection_package(self):
		"""Forget all optional multi-image package data and keep only the active array."""
		self.package_images = {}
		self.package_layers = []
		self.package_metadata = {}
		self.active_layer_key = None

	def set_projection_package(self, package_images, package_layers, metadata=None, active_layer_key=None, auto_window=False):
		"""Replace the optional projection package stored by this detector image."""
		normalized_images = {}
		for key, array in dict(package_images or {}).items():
			array = np.asarray(array, dtype=np.float32)
			if array.ndim != 2:
				raise ValueError("Projection package images must be 2D float32 arrays.")
			normalized_images[str(key)] = np.array(array, dtype=np.float32, copy=True)

		normalized_layers = []
		for layer in list(package_layers or []):
			normalized_layer = self._normalize_package_layer(layer)
			if normalized_layer["key"] not in normalized_images:
				continue
			normalized_layers.append(normalized_layer)

		if not normalized_layers and normalized_images:
			first_key = next(iter(normalized_images.keys()))
			normalized_layers.append(self._normalize_package_layer({
				"key": first_key,
				"label": first_key,
				"stage": self.source_stage,
				"role": "derived",
			}))

		self.package_images = normalized_images
		self.package_layers = normalized_layers
		self.package_metadata = {} if metadata is None else dict(metadata)
		self.active_layer_key = None if active_layer_key is None else str(active_layer_key)
		if self.active_layer_key not in self.package_images and normalized_layers:
			self.active_layer_key = normalized_layers[0]["key"]
		if self.active_layer_key in self.package_images:
			self.set_active_layer(self.active_layer_key, auto_window=auto_window)

	def package_layer_choices(self):
		"""Return a compact list of `(key, label)` pairs for UI selectors."""
		return [(layer["key"], layer["label"]) for layer in self.package_layers]

	def active_layer_info(self):
		"""Return metadata for the currently active package layer, or `None`."""
		active_key = None if self.active_layer_key is None else str(self.active_layer_key)
		for layer in self.package_layers:
			if layer["key"] == active_key:
				return dict(layer)
		return None

	def set_active_layer(self, layer_key, auto_window=False):
		"""Switch the displayed array to one package layer if it exists."""
		layer_key = str(layer_key)
		array = self.package_images.get(layer_key, None)
		if array is None:
			raise KeyError(f"Unknown detector package layer: {layer_key}")
		layer_info = next((layer for layer in self.package_layers if layer["key"] == layer_key), None)
		self.active_layer_key = layer_key
		self.setArray(
			array,
			source_stage=(self.source_stage if layer_info is None else layer_info["stage"]),
			auto_window=auto_window,
		)

	def _sync_active_layer_into_package(self):
		"""Mirror the current active array back into the in-memory package image map."""
		if self.raw_array is None or self.active_layer_key is None:
			return
		if self.active_layer_key not in self.package_images:
			return
		self.package_images[self.active_layer_key] = np.array(self.raw_array, dtype=np.float32, copy=True)

	def build_presentation_model(self):
		"""Build the selected presentation model for display-ready output."""
		mode = str(self.presentation_mode).lower()
		if mode == "raw":
			return RawPresentationModel()
		if mode == "film":
			return FilmLikePresentationModel(
				robust_low_percentile=self.presentation_robust_low_percentile,
				robust_percentile=self.presentation_robust_percentile,
				gamma=self.presentation_gamma,
				contrast=self.presentation_contrast,
				invert=self.presentation_invert,
				input_transform=self.presentation_input_transform,
				local_enhancement=self.presentation_local_enhancement,
				clahe_clip_limit=self.presentation_clahe_clip_limit,
				clahe_tile_grid_size=self.presentation_clahe_tile_grid_size,
			)
		return DigitalRadiographyPresentationModel(
			window_center=self.presentation_window_center,
			window_width=self.presentation_window_width,
			robust_low_percentile=self.presentation_robust_low_percentile,
			robust_percentile=self.presentation_robust_percentile,
			invert=self.presentation_invert,
			gamma=self.presentation_gamma,
			contrast=self.presentation_contrast,
			input_transform=self.presentation_input_transform,
			local_enhancement=self.presentation_local_enhancement,
			clahe_clip_limit=self.presentation_clahe_clip_limit,
			clahe_tile_grid_size=self.presentation_clahe_tile_grid_size,
		)

	def effective_window(self):
		"""Return the active `(center, width)` pair used by the display logic."""
		if (
			self.presentation_window_center is not None
			and self.presentation_window_width is not None
			and float(self.presentation_window_width) > 0.0
		):
			return float(self.presentation_window_center), max(float(self.presentation_window_width), 1e-6)
		vmin, vmax = self.auto_window_range()
		return (vmin + vmax) / 2.0, max(vmax - vmin, 1e-6)

	def _window_mask(self, image):
		"""Return the in-window mask used by optional out-of-window suppression."""
		center, width = self.effective_window()
		lower = center - width / 2.0
		upper = center + width / 2.0
		image = np.asarray(image, dtype=np.float32)
		return (image >= lower) & (image <= upper)

	def _normalize_raw_window(self, image):
		"""Map one raw float detector image into `[0, 1]` using the active window."""
		center, width = self.effective_window()
		lower = center - width / 2.0
		upper = center + width / 2.0
		return np.clip((image - lower) / max(upper - lower, 1e-6), 0.0, 1.0).astype(np.float32, copy=False)

	def _apply_raw_tone_controls(self, normalized):
		"""Apply only the minimal raw-view controls kept for detector inspection."""
		if bool(self.presentation_invert):
			normalized = 1.0 - normalized
		return normalized.astype(np.float32, copy=False)

	def get_presented_array(self):
		"""Return the presentation-stage image before the optional transfer curve."""
		if self.raw_array is None:
			return np.zeros((1, 1), dtype=np.float32)
		image = np.asarray(self.raw_array, dtype=np.float32)
		mode = str(self.presentation_mode).lower()
		if mode == "raw":
			return self._apply_raw_tone_controls(self._normalize_raw_window(image))
		if mode == "film":
			return FilmLikePresentationModel(
				robust_low_percentile=self.presentation_robust_low_percentile,
				robust_percentile=self.presentation_robust_percentile,
				gamma=self.presentation_gamma,
				contrast=self.presentation_contrast,
				invert=self.presentation_invert,
				input_transform=self.presentation_input_transform,
				local_enhancement=self.presentation_local_enhancement,
				clahe_clip_limit=self.presentation_clahe_clip_limit,
				clahe_tile_grid_size=self.presentation_clahe_tile_grid_size,
			).apply(image)
		return self.build_presentation_model().apply(image)

	def set_transfer_points_pct(self, points):
		"""Store one sorted transfer curve in percent coordinates."""
		if points is None:
			points = [(0.0, 0.0), (100.0, 100.0)]
		sanitized = []
		for point in points:
			if point is None or len(point) != 2:
				continue
			x_value = min(100.0, max(0.0, float(point[0])))
			y_value = min(100.0, max(0.0, float(point[1])))
			sanitized.append((x_value, y_value))
		if len(sanitized) < 2:
			sanitized = [(0.0, 0.0), (100.0, 100.0)]
		sanitized.sort(key=lambda item: (item[0], item[1]))
		merged = []
		for x_value, y_value in sanitized:
			if merged and abs(x_value - merged[-1][0]) <= 1e-6:
				merged[-1] = (x_value, y_value)
			else:
				merged.append((x_value, y_value))
		if len(merged) < 2:
			merged = [(0.0, 0.0), (100.0, 100.0)]
		self.transfer_points_pct = merged

	def transfer_points_summary(self):
		"""Return one compact human-readable summary of the transfer curve."""
		return " | ".join(f"{x_value:.1f}->{y_value:.1f}" for x_value, y_value in self.transfer_points_pct)

	def transfer_curve_preview_data(self, histogram_bins=64, ignore_zero_values=True):
		"""Return normalized histogram and control points for transfer-curve preview widgets."""
		bin_count = max(8, int(histogram_bins))
		presented = np.asarray(self.get_presented_array(), dtype=np.float32)
		finite_values = presented[np.isfinite(presented)]
		if finite_values.size == 0:
			histogram = np.zeros(bin_count, dtype=np.float32)
		else:
			finite_values = np.clip(finite_values, 0.0, 1.0)
			if bool(ignore_zero_values):
				non_zero_values = finite_values[finite_values > 0.0]
				if non_zero_values.size > 0:
					finite_values = non_zero_values
			histogram, _edges = np.histogram(finite_values, bins=bin_count, range=(0.0, 1.0))
			histogram = np.asarray(histogram, dtype=np.float32)
			histogram_max = float(np.max(histogram))
			if histogram_max > 0.0:
				histogram = histogram / histogram_max
		x_values = np.asarray([point[0] for point in self.transfer_points_pct], dtype=np.float32) / 100.0
		y_values = np.asarray([point[1] for point in self.transfer_points_pct], dtype=np.float32) / 100.0
		return {
			"histogram": histogram,
			"curve_x": np.clip(x_values, 0.0, 1.0),
			"curve_y": np.clip(y_values, 0.0, 1.0),
		}

	def get_display_array(self):
		"""Return the current detector image mapped into the `[0, 1]` display range."""
		if self.raw_array is None:
			return np.zeros((1, 1), dtype=np.float32)
		normalized = np.asarray(self.get_presented_array(), dtype=np.float32)
		x_values = np.asarray([point[0] for point in self.transfer_points_pct], dtype=np.float32) / 100.0
		y_values = np.asarray([point[1] for point in self.transfer_points_pct], dtype=np.float32) / 100.0
		if len(x_values) >= 2:
			normalized = np.interp(
				normalized,
				x_values,
				y_values,
				left=y_values[0],
				right=y_values[-1],
			).astype(np.float32, copy=False)
		if bool(self.display_only_window_range):
			normalized = np.where(self._window_mask(self.raw_array), normalized, 0.0).astype(np.float32, copy=False)
		return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

	def set_projected_annotations(self, overlay_projection_set):
		"""Store detector-space overlay primitives projected from the source scene."""
		self.overlay_projection_set = overlay_projection_set

	def apply_presentation_defaults(self, defaults):
		"""Apply one normalized presentation-default mapping to this detector image."""
		defaults = {} if defaults is None else dict(defaults)
		self.presentation_mode = str(defaults.get("mode", self.presentation_mode))
		self.presentation_invert = bool(defaults.get("invert", self.presentation_invert))
		self.presentation_gamma = max(0.05, float(defaults.get("gamma", self.presentation_gamma)))
		self.presentation_contrast = max(0.05, float(defaults.get("contrast", self.presentation_contrast)))
		self.presentation_input_transform = str(
			defaults.get("input_transform", self.presentation_input_transform)
		).lower()
		self.presentation_local_enhancement = str(
			defaults.get("local_enhancement", self.presentation_local_enhancement)
		).lower()
		self.presentation_clahe_clip_limit = max(
			0.01,
			float(defaults.get("clahe_clip_limit", self.presentation_clahe_clip_limit)),
		)
		self.presentation_clahe_tile_grid_size = max(
			1,
			int(defaults.get("clahe_tile_grid_size", self.presentation_clahe_tile_grid_size)),
		)
		self.presentation_robust_low_percentile = min(
			99.999,
			max(0.0, float(defaults.get("robust_low_percentile", self.presentation_robust_low_percentile))),
		)
		self.presentation_robust_percentile = min(
			100.0,
			max(
				self.presentation_robust_low_percentile + 1e-6,
				float(defaults.get("robust_percentile", self.presentation_robust_percentile)),
			),
		)
		self.presentation_window_center = defaults.get("window_center", self.presentation_window_center)
		self.presentation_window_width = defaults.get("window_width", self.presentation_window_width)
		self.presentation_overlay_annotations = bool(
			defaults.get("overlay_annotations", self.presentation_overlay_annotations)
		)
		self.presentation_overlay_labels = bool(
			defaults.get("overlay_labels", self.presentation_overlay_labels)
		)
		self.presentation_overlay_cross_size_px = max(
			1,
			int(defaults.get("overlay_cross_size_px", self.presentation_overlay_cross_size_px)),
		)

	def _npz_array_ref_payload(self, archive_key, array):
		"""Return one JSON-ready reference to an array stored as a separate NPZ entry."""
		array = np.asarray(array, dtype=np.float32)
		return {
			"storage": "npz_entry",
			"archive_key": str(archive_key),
			"dtype": "float32",
			"shape": [int(array.shape[0]), int(array.shape[1])],
		}

	def _build_projection_package_npz_payload(self):
		"""Return `(metadata, arrays)` for the externalized NPZ detector package format."""
		self._sync_active_layer_into_package()
		if self.raw_array is None:
			raise ValueError("No detector array is available to export.")
		package_images = dict(self.package_images)
		package_layers = list(self.package_layers)
		if not package_images:
			fallback_key = "composited_raw"
			package_images[fallback_key] = np.array(self.raw_array, dtype=np.float32, copy=True)
			package_layers = [self._normalize_package_layer({
				"key": fallback_key,
				"label": "Composited raw",
				"stage": self.source_stage,
				"role": "composited",
			})]
			if self.active_layer_key is None:
				self.active_layer_key = fallback_key
		active_layer_key = self.active_layer_key if self.active_layer_key in package_images else package_layers[0]["key"]
		arrays = {}
		image_refs = {}
		for layer_index, layer in enumerate(package_layers):
			layer_key = layer["key"]
			array = package_images.get(layer_key, None)
			if array is None:
				continue
			archive_key = f"image_{layer_index:03d}_{layer_key}"
			arrays[archive_key] = np.asarray(array, dtype=np.float32)
			image_refs[layer_key] = self._npz_array_ref_payload(archive_key, array)
		metadata = {
			"schema": self.PACKAGE_SCHEMA,
			"version": self.PACKAGE_VERSION,
			"active_layer_key": str(active_layer_key),
			"image_object": {
				"source_stage": str(self.source_stage),
				"source_virtual_xray_label": str(self.source_virtual_xray_label),
				"presentation": self._package_presentation_payload(),
			},
			"simulation_context": dict(self.package_metadata.get("simulation_context", {})),
			"annotations": {
				"projected_annotations": overlay_projection_set_to_payload(self.overlay_projection_set),
			},
			"layers": [dict(layer) for layer in package_layers],
			"images": image_refs,
		}
		return metadata, arrays

	def _package_presentation_payload(self):
		"""Return the current presentation state as a JSON-ready mapping."""
		return {
			"mode": str(self.presentation_mode),
			"invert": bool(self.presentation_invert),
			"gamma": float(self.presentation_gamma),
			"contrast": float(self.presentation_contrast),
			"input_transform": str(self.presentation_input_transform),
			"local_enhancement": str(self.presentation_local_enhancement),
			"clahe_clip_limit": float(self.presentation_clahe_clip_limit),
			"clahe_tile_grid_size": int(self.presentation_clahe_tile_grid_size),
			"robust_low_percentile": float(self.presentation_robust_low_percentile),
			"robust_percentile": float(self.presentation_robust_percentile),
			"window_center": self.presentation_window_center,
			"window_width": self.presentation_window_width,
			"overlay_annotations": bool(self.presentation_overlay_annotations),
			"overlay_labels": bool(self.presentation_overlay_labels),
			"overlay_cross_size_px": int(self.presentation_overlay_cross_size_px),
			"display_only_window_range": bool(self.display_only_window_range),
			"transfer_points_pct": [
				[float(x_value), float(y_value)]
				for x_value, y_value in self.transfer_points_pct
			],
		}

	def _apply_projection_package_payload(self, payload, auto_window=False, archive=None):
		"""Restore this object from one complete detector package payload."""
		image_object = dict(payload.get("image_object", {}))
		self.source_stage = str(image_object.get("source_stage", self.source_stage))
		self.source_virtual_xray_label = str(
			image_object.get("source_virtual_xray_label", self.source_virtual_xray_label)
		)
		presentation = dict(image_object.get("presentation", {}))
		self.apply_presentation_defaults(presentation)
		self.display_only_window_range = bool(
			presentation.get("display_only_window_range", self.display_only_window_range)
		)
		self.set_transfer_points_pct(presentation.get("transfer_points_pct", self.transfer_points_pct))
		self.overlay_projection_set = overlay_projection_set_from_payload(
			dict(payload.get("annotations", {})).get("projected_annotations", None)
		)
		package_images = {}
		for key, array_payload in dict(payload.get("images", {})).items():
			storage = str(array_payload.get("storage", "")).strip().lower()
			if storage != "npz_entry":
				raise ValueError("Detector package images must be stored as separate NPZ entries.")
			if archive is None:
				raise ValueError("Detector package references NPZ entries, but no archive was provided.")
			archive_key = str(array_payload.get("archive_key", "")).strip()
			if archive_key == "":
				raise ValueError("Detector package image reference is missing archive_key.")
			if archive_key not in archive:
				raise ValueError(f"Detector package is missing array entry: {archive_key}")
			package_images[key] = np.asarray(archive[archive_key], dtype=np.float32)
		self.set_projection_package(
			package_images=package_images,
			package_layers=list(payload.get("layers", [])),
			metadata={"simulation_context": dict(payload.get("simulation_context", {}))},
			active_layer_key=payload.get("active_layer_key", None),
			auto_window=auto_window,
		)

	def sync_from_virtual_xray(self, virtual_xray, auto_window=False):
		"""Copy raw data, presentation settings, and overlays from one `VirtualXRay` object."""
		self.source_virtual_xray_label = str(getattr(virtual_xray, "label", ""))
		self.label = f"{self.source_virtual_xray_label}_projection"
		defaults_getter = getattr(virtual_xray, "get_detector_image_defaults", None)
		defaults = defaults_getter() if callable(defaults_getter) else getattr(virtual_xray, "detector_image_defaults", None)
		self.apply_presentation_defaults(defaults)
		self.set_projected_annotations(getattr(virtual_xray, "last_projected_annotations", None))
		package_images = {}
		package_layers = []
		if getattr(virtual_xray, "last_raw_projection", None) is not None:
			package_images["composited_raw"] = np.asarray(
				getattr(virtual_xray, "last_raw_projection"),
				dtype=np.float32,
			)
			package_layers.append({
				"key": "composited_raw",
				"label": "Composited raw",
				"stage": "raw",
				"role": "composited",
			})
		if getattr(virtual_xray, "last_line_integral_projection", None) is not None:
			package_images["composited_line_integral"] = np.asarray(
				getattr(virtual_xray, "last_line_integral_projection"),
				dtype=np.float32,
			)
			package_layers.append({
				"key": "composited_line_integral",
				"label": "Composited line integral",
				"stage": "line_integral",
				"role": "composited",
			})
		for source_projection in list(getattr(virtual_xray, "last_source_projections", [])):
			source_prefix = f"source_{int(source_projection.source_index):03d}"
			source_label = str(source_projection.label)
			package_images[f"{source_prefix}_raw"] = np.asarray(source_projection.detector_image, dtype=np.float32)
			package_layers.append({
				"key": f"{source_prefix}_raw",
				"label": f"{source_label} raw",
				"stage": "raw",
				"role": "per_source",
				"source_index": int(source_projection.source_index),
				"source_label": source_label,
				"source_type": str(source_projection.source_type),
			})
			package_images[f"{source_prefix}_line_integral"] = np.asarray(
				source_projection.line_integral_image,
				dtype=np.float32,
			)
			package_layers.append({
				"key": f"{source_prefix}_line_integral",
				"label": f"{source_label} line integral",
				"stage": "line_integral",
				"role": "per_source",
				"source_index": int(source_projection.source_index),
				"source_label": source_label,
				"source_type": str(source_projection.source_type),
			})
		should_auto_window = bool(
			auto_window
			and (
				self.presentation_window_center is None
				or self.presentation_window_width is None
				or float(self.presentation_window_width) <= 0.0
			)
		)
		self.set_projection_package(
			package_images=package_images,
			package_layers=package_layers,
			metadata={
				"simulation_context": {
					"source_virtual_xray_label": str(getattr(virtual_xray, "label", "")),
					"detector_image_defaults": {} if defaults is None else dict(defaults),
					"geometry_snapshot": {
						"projection_mode": str(getattr(virtual_xray, "projection_mode", "cone")),
						"detector_shape_hw": [
							int(getattr(virtual_xray, "detector_shape_hw", [0, 0])[0]),
							int(getattr(virtual_xray, "detector_shape_hw", [0, 0])[1]),
						],
					},
				},
			},
			active_layer_key="composited_raw" if "composited_raw" in package_images else (package_layers[0]["key"] if package_layers else None),
			auto_window=should_auto_window,
		)

	def _paint_projected_overlays(self, qimage):
		"""Paint generic projected overlay primitives on the final display image."""
		if qimage is None or not bool(self.presentation_overlay_annotations):
			return
		annotation_set = self.overlay_projection_set
		if annotation_set is None or not getattr(annotation_set, "items", None):
			return
		painter = QPainter(qimage)
		try:
			painter.setRenderHint(QPainter.Antialiasing, True)
			painter.setRenderHint(QPainter.TextAntialiasing, True)
			for item in annotation_set.items:
				if not item.visible:
					continue
				self._paint_overlay_item(painter, qimage, item)
		finally:
			painter.end()

	def _paint_overlay_item(self, painter, qimage, item):
		"""Paint one generic overlay item and its optional label."""
		if isinstance(item, XRayOverlayCross):
			self._paint_overlay_cross(painter, qimage, item)
			self._paint_overlay_label(
				painter,
				qimage,
				item,
				item.pixel_uv,
				self.presentation_overlay_cross_size_px,
			)
			return
		if isinstance(item, XRayOverlayPolyline):
			self._paint_overlay_polyline(painter, qimage, item)
			anchor_uv = item.pixel_uvs[0] if item.pixel_uvs else None
			marker_size_px = item.style.marker_size_px if item.style is not None else self.presentation_overlay_cross_size_px
			self._paint_overlay_label(painter, qimage, item, anchor_uv, marker_size_px)

	def _paint_overlay_cross(self, painter, qimage, item: XRayOverlayCross):
		"""Paint one cross overlay item on the projection image."""
		if item.pixel_uv is None or not item.in_bounds:
			return
		style = item.style
		color = QColor(*(style.color_rgba if style is not None else (255, 0, 0, 255)))
		size_px = max(
			1,
			int(self.presentation_overlay_cross_size_px if self.presentation_overlay_cross_size_px else style.marker_size_px if style is not None else 6),
		)
		line_width_px = max(1, int(style.line_width_px if style is not None else 1))
		x_coord = int(round(item.pixel_uv[0]))
		y_coord = qimage.height() - 1 - int(round(item.pixel_uv[1]))
		painter.setPen(QPen(color, line_width_px))
		painter.drawLine(x_coord - size_px, y_coord, x_coord + size_px, y_coord)
		painter.drawLine(x_coord, y_coord - size_px, x_coord, y_coord + size_px)

	def _paint_overlay_polyline(self, painter, qimage, item: XRayOverlayPolyline):
		"""Paint one detector-space polyline overlay."""
		if not item.pixel_uvs:
			return
		style = item.style
		color = QColor(*(style.color_rgba if style is not None else (255, 0, 0, 255)))
		line_width_px = max(1, int(style.line_width_px if style is not None else 1))
		painter.setPen(QPen(color, line_width_px))
		points_xy = [
			(int(round(pixel_uv[0])), qimage.height() - 1 - int(round(pixel_uv[1])))
			for pixel_uv in item.pixel_uvs
		]
		for point_a, point_b in zip(points_xy, points_xy[1:]):
			painter.drawLine(point_a[0], point_a[1], point_b[0], point_b[1])
		if item.closed and len(points_xy) > 2:
			painter.drawLine(points_xy[-1][0], points_xy[-1][1], points_xy[0][0], points_xy[0][1])

	def _paint_overlay_label(self, painter, qimage, item, anchor_uv, marker_size_px):
		"""Paint one optional overlay label next to the supplied anchor point."""
		if not bool(self.presentation_overlay_labels):
			return
		if anchor_uv is None or not item.in_bounds:
			return
		label_text = str(getattr(item, "label", "")).strip()
		if not label_text:
			return
		style = getattr(item, "style", None)
		color_rgba = style.color_rgba if style is not None else (255, 0, 0, 255)
		text_color = QColor(*color_rgba)
		x_coord = int(round(anchor_uv[0])) + max(4, int(marker_size_px)) + 2
		y_coord = qimage.height() - 1 - int(round(anchor_uv[1])) - 4
		painter.setPen(QPen(QColor(0, 0, 0, 220)))
		for dx_offset, dy_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
			painter.drawText(x_coord + dx_offset, y_coord + dy_offset, label_text)
		painter.setPen(QPen(text_color))
		painter.drawText(x_coord, y_coord, label_text)

	def toQImage(self):
		"""Return one grayscale Qt image created from the current display settings."""
		display = np.ascontiguousarray(np.flipud(self.get_display_array()))
		image_u8 = np.round(display * 255.0).astype(np.uint8, copy=False)
		height, width = image_u8.shape
		grayscale_image = QImage(
			image_u8.data,
			width,
			height,
			image_u8.strides[0],
			QImage.Format_Grayscale8,
		).copy()
		qimage = grayscale_image.convertToFormat(QImage.Format_ARGB32)
		self._paint_projected_overlays(qimage)
		return qimage

	def save_png(self, path):
		"""Save the current displayed image to PNG."""
		path = Path(path)
		if not self.toQImage().save(str(path), "PNG"):
			raise IOError(f"Failed to save detector image PNG to: {path}")
		return path

	def import_array(self, path, auto_window=False):
		"""Load one detector image package or raw array into this object."""
		path = Path(path)
		suffix = path.suffix.lower()
		metadata = None
		if suffix == ".npy":
			array = np.load(path)
		elif suffix == ".npz":
			with np.load(path, allow_pickle=False) as archive:
				if "metadata_json" in archive:
					metadata_raw = archive["metadata_json"]
					metadata = json.loads(str(metadata_raw.tolist() if hasattr(metadata_raw, "tolist") else metadata_raw))
					if isinstance(metadata, dict) and str(metadata.get("schema", "")).strip() == self.PACKAGE_SCHEMA:
						self._apply_projection_package_payload(metadata, auto_window=auto_window, archive=archive)
						return path
				if "image" in archive:
					array = archive["image"]
				elif len(archive.files) == 1:
					array = archive[archive.files[0]]
				else:
					raise ValueError("DetectorImage NPZ must contain an 'image' array.")
		elif suffix in {".txt", ".csv", ".tsv"}:
			delimiter = "," if suffix == ".csv" else None
			if suffix == ".tsv":
				delimiter = "\t"
			array = np.loadtxt(path, delimiter=delimiter)
		else:
			raise ValueError("Supported detector array formats are: .npz, .npy, .txt, .csv, .tsv.")
		array = np.asarray(array, dtype=np.float32)
		if array.ndim != 2:
			raise ValueError("Imported detector arrays must be 2D.")
		self.clear_projection_package()
		self.setArray(array, auto_window=auto_window)
		return path

	def export_array(self, path):
		"""Save the raw detector array as `.npy`, `.npz`, or text."""
		path = Path(path)
		if self.raw_array is None:
			raise ValueError("No detector array is available to export.")
		suffix = path.suffix.lower()
		if suffix == ".npy":
			np.save(path, self.raw_array)
			return path
		if suffix == ".npz":
			metadata, arrays = self._build_projection_package_npz_payload()
			archive_payload = {
				"metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
			}
			for archive_key, array in arrays.items():
				archive_payload[archive_key] = np.asarray(array, dtype=np.float32)
			np.savez_compressed(path, **archive_payload)
			return path
		delimiter = "," if suffix == ".csv" else None
		if suffix == ".tsv":
			delimiter = "\t"
		if suffix in {".txt", ".csv", ".tsv"}:
			save_kwargs = {"fmt": "%.9g"}
			if delimiter is not None:
				save_kwargs["delimiter"] = delimiter
			np.savetxt(path, self.raw_array, **save_kwargs)
			return path
		raise ValueError("Supported detector array formats are: .npy, .npz, .txt, .csv, .tsv.")
