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
		self.presentation_robust_percentile = 99.5
		self.presentation_window_center = None
		self.presentation_window_width = None
		self.presentation_overlay_annotations = False
		self.presentation_overlay_labels = False
		self.presentation_overlay_cross_size_px = 6
		self.overlay_projection_set = None
		self.transfer_points_pct = [(0.0, 0.0), (100.0, 100.0)]
		self.display_only_window_range = False
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

	def auto_window_range(self, robust_percentile=None):
		"""Return one automatic display range based on finite values."""
		if self.raw_array is None:
			return 0.0, 1.0
		finite_values = np.asarray(self.raw_array, dtype=np.float32)
		finite_values = finite_values[np.isfinite(finite_values)]
		if finite_values.size == 0:
			return 0.0, 1.0
		percentile = float(
			self.presentation_robust_percentile if robust_percentile is None else robust_percentile
		)
		percentile = min(100.0, max(50.0, percentile))
		vmin = float(np.min(finite_values))
		vmax = float(np.percentile(finite_values, percentile))
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

	def auto_window(self, robust_percentile=None):
		"""Set the presentation window using the robust automatic range."""
		vmin, vmax = self.auto_window_range(robust_percentile=robust_percentile)
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

	def build_presentation_model(self):
		"""Build the selected presentation model for display-ready output."""
		mode = str(self.presentation_mode).lower()
		if mode == "raw":
			return RawPresentationModel()
		if mode == "film":
			return FilmLikePresentationModel(
				robust_percentile=self.presentation_robust_percentile,
				gamma=self.presentation_gamma,
				contrast=self.presentation_contrast,
				invert=self.presentation_invert,
			)
		return DigitalRadiographyPresentationModel(
			window_center=self.presentation_window_center,
			window_width=self.presentation_window_width,
			robust_percentile=self.presentation_robust_percentile,
			invert=self.presentation_invert,
			gamma=self.presentation_gamma,
			contrast=self.presentation_contrast,
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
				robust_percentile=self.presentation_robust_percentile,
				gamma=self.presentation_gamma,
				contrast=self.presentation_contrast,
				invert=self.presentation_invert,
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

	def sync_from_virtual_xray(self, virtual_xray, auto_window=False):
		"""Copy raw data, presentation settings, and overlays from one `VirtualXRay` object."""
		self.source_virtual_xray_label = str(getattr(virtual_xray, "label", ""))
		self.label = f"{self.source_virtual_xray_label}_projection"
		self.presentation_mode = str(getattr(virtual_xray, "presentation_mode", self.presentation_mode))
		self.presentation_invert = bool(getattr(virtual_xray, "presentation_invert", self.presentation_invert))
		self.presentation_gamma = max(0.05, float(getattr(virtual_xray, "presentation_gamma", self.presentation_gamma)))
		self.presentation_contrast = max(0.05, float(getattr(virtual_xray, "presentation_contrast", self.presentation_contrast)))
		self.presentation_robust_percentile = min(
			100.0,
			max(50.0, float(getattr(virtual_xray, "presentation_robust_percentile", self.presentation_robust_percentile))),
		)
		self.presentation_window_center = getattr(virtual_xray, "presentation_window_center", None)
		self.presentation_window_width = getattr(virtual_xray, "presentation_window_width", None)
		self.presentation_overlay_annotations = bool(
			getattr(virtual_xray, "presentation_overlay_annotations", self.presentation_overlay_annotations)
		)
		self.presentation_overlay_labels = bool(
			getattr(virtual_xray, "presentation_overlay_labels", self.presentation_overlay_labels)
		)
		self.presentation_overlay_cross_size_px = max(
			1,
			int(getattr(virtual_xray, "presentation_overlay_cross_size_px", self.presentation_overlay_cross_size_px)),
		)
		self.set_projected_annotations(getattr(virtual_xray, "last_projected_annotations", None))
		if getattr(virtual_xray, "last_raw_projection", None) is not None:
			should_auto_window = bool(
				auto_window
				and (
					self.presentation_window_center is None
					or self.presentation_window_width is None
					or float(self.presentation_window_width) <= 0.0
				)
			)
			self.setArray(
				getattr(virtual_xray, "last_raw_projection"),
				source_stage="raw",
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
		qimage = QImage(
			image_u8.data,
			width,
			height,
			image_u8.strides[0],
			QImage.Format_Grayscale8,
		).copy()
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
				if "image" in archive:
					array = archive["image"]
				elif len(archive.files) == 1:
					array = archive[archive.files[0]]
				else:
					raise ValueError("DetectorImage NPZ must contain an 'image' array.")
				if "metadata_json" in archive:
					metadata_raw = archive["metadata_json"]
					metadata = json.loads(str(metadata_raw.tolist() if hasattr(metadata_raw, "tolist") else metadata_raw))
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
		self.setArray(array, auto_window=auto_window)
		if not isinstance(metadata, dict):
			return path
		self.source_stage = str(metadata.get("source_stage", self.source_stage))
		self.source_virtual_xray_label = str(metadata.get("source_virtual_xray_label", self.source_virtual_xray_label))
		presentation = dict(metadata.get("presentation", {}))
		self.presentation_mode = str(presentation.get("mode", self.presentation_mode))
		self.presentation_invert = bool(presentation.get("invert", self.presentation_invert))
		self.presentation_gamma = max(0.05, float(presentation.get("gamma", self.presentation_gamma)))
		self.presentation_contrast = max(0.05, float(presentation.get("contrast", self.presentation_contrast)))
		self.presentation_robust_percentile = min(
			100.0,
			max(50.0, float(presentation.get("robust_percentile", self.presentation_robust_percentile))),
		)
		self.presentation_window_center = presentation.get("window_center", self.presentation_window_center)
		self.presentation_window_width = presentation.get("window_width", self.presentation_window_width)
		self.presentation_overlay_annotations = bool(
			presentation.get("overlay_annotations", self.presentation_overlay_annotations)
		)
		self.presentation_overlay_labels = bool(
			presentation.get("overlay_labels", self.presentation_overlay_labels)
		)
		self.presentation_overlay_cross_size_px = max(
			1,
			int(presentation.get("overlay_cross_size_px", self.presentation_overlay_cross_size_px)),
		)
		self.display_only_window_range = bool(
			presentation.get("display_only_window_range", self.display_only_window_range)
		)
		self.set_transfer_points_pct(presentation.get("transfer_points_pct", self.transfer_points_pct))
		self.overlay_projection_set = overlay_projection_set_from_payload(
			metadata.get("projected_annotations", None)
		)
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
			metadata = {
				"schema": "virtRTG-detector-image-export",
				"version": 1,
				"source_stage": str(self.source_stage),
				"source_virtual_xray_label": str(self.source_virtual_xray_label),
				"presentation": {
					"mode": str(self.presentation_mode),
					"invert": bool(self.presentation_invert),
					"gamma": float(self.presentation_gamma),
					"contrast": float(self.presentation_contrast),
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
				},
				"projected_annotations": overlay_projection_set_to_payload(self.overlay_projection_set),
			}
			np.savez_compressed(
				path,
				image=np.asarray(self.raw_array, dtype=np.float32),
				metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
			)
			return path
		delimiter = "," if suffix == ".csv" else None
		if suffix == ".tsv":
			delimiter = "\t"
		if suffix in {".txt", ".csv", ".tsv"}:
			np.savetxt(path, self.raw_array, fmt="%.9g", delimiter=delimiter)
			return path
		raise ValueError("Supported detector array formats are: .npy, .npz, .txt, .csv, .tsv.")
