# -*- coding: utf-8 -*-
"""Plugin-local 2D detector image object with interactive window/level support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtGui import QImage

from dpVision import Object


class DetectorImage(Object):
	"""Represent one flat detector-space float image stored in the scene tree."""

	def __init__(self, parent=None, array=None, source_stage="raw"):
		"""Initialize one hidden-in-3D detector image object."""
		super().__init__(parent)
		self.visible = False
		self.source_stage = str(source_stage)
		self.source_virtual_xray_label = ""
		self.raw_array = None
		self.window_center = None
		self.window_width = None
		self.display_only_window_range = False
		self.display_invert = False
		self.display_gamma = 1.0
		self.display_contrast = 1.0
		self.display_robust_percentile = 99.5
		if array is not None:
			self.setArray(array)

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
			f"min={stats['min']:.4g}, max={stats['max']:.4g}"
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
		percentile = float(self.display_robust_percentile if robust_percentile is None else robust_percentile)
		percentile = min(100.0, max(50.0, percentile))
		vmin = float(np.min(finite_values))
		vmax = float(np.percentile(finite_values, percentile))
		if vmax <= vmin:
			vmax = float(np.max(finite_values))
		if vmax <= vmin:
			vmax = vmin + 1.0
		return vmin, vmax

	def reset_window_to_full_range(self):
		"""Set the display window to the full finite range of the current array."""
		stats = self.data_stats()
		vmin = stats["min"]
		vmax = stats["max"]
		if vmax <= vmin:
			vmax = vmin + 1.0
		self.window_center = (vmin + vmax) / 2.0
		self.window_width = vmax - vmin

	def auto_window(self, robust_percentile=None):
		"""Set the display window using the robust automatic range."""
		vmin, vmax = self.auto_window_range(robust_percentile=robust_percentile)
		self.window_center = (vmin + vmax) / 2.0
		self.window_width = max(vmax - vmin, 1e-6)

	def setArray(self, array, source_stage=None, auto_window=True):
		"""Replace the stored detector array and optionally reset the display window."""
		array = np.asarray(array, dtype=np.float32)
		if array.ndim != 2:
			raise ValueError("DetectorImage expects a 2D float32 array.")
		self.raw_array = np.array(array, dtype=np.float32, copy=True)
		if source_stage is not None:
			self.source_stage = str(source_stage)
		if auto_window or self.window_center is None or self.window_width is None:
			self.auto_window()

	def effective_window(self):
		"""Return the active `(center, width)` pair."""
		if self.window_center is not None and self.window_width is not None and float(self.window_width) > 0.0:
			return float(self.window_center), max(float(self.window_width), 1e-6)
		vmin, vmax = self.auto_window_range()
		return (vmin + vmax) / 2.0, max(vmax - vmin, 1e-6)

	def get_display_array(self):
		"""Return the current detector image mapped into the `[0, 1]` display range."""
		if self.raw_array is None:
			return np.zeros((1, 1), dtype=np.float32)
		image = np.asarray(self.raw_array, dtype=np.float32)
		center, width = self.effective_window()
		lower = center - width / 2.0
		upper = center + width / 2.0
		in_window_mask = (image >= lower) & (image <= upper)
		normalized = np.clip((image - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
		contrast = max(0.05, float(self.display_contrast))
		gamma = max(0.05, float(self.display_gamma))
		normalized = np.clip(0.5 + (normalized - 0.5) * contrast, 0.0, 1.0)
		normalized = np.power(normalized, 1.0 / gamma).astype(np.float32, copy=False)
		if bool(self.display_invert):
			normalized = 1.0 - normalized
		if bool(self.display_only_window_range):
			normalized = np.where(in_window_mask, normalized, 0.0).astype(np.float32, copy=False)
		return normalized.astype(np.float32, copy=False)

	def toQImage(self):
		"""Return one grayscale Qt image created from the current display settings."""
		display = np.ascontiguousarray(np.flipud(self.get_display_array()))
		image_u8 = np.round(display * 255.0).astype(np.uint8, copy=False)
		height, width = image_u8.shape
		return QImage(
			image_u8.data,
			width,
			height,
			image_u8.strides[0],
			QImage.Format_Grayscale8,
		).copy()

	def save_png(self, path):
		"""Save the current displayed image to PNG."""
		path = Path(path)
		if not self.toQImage().save(str(path), "PNG"):
			raise IOError(f"Failed to save detector image PNG to: {path}")
		return path

	def export_array(self, path):
		"""Save the raw detector array as `.npy` or text."""
		path = Path(path)
		if self.raw_array is None:
			raise ValueError("No detector array is available to export.")
		suffix = path.suffix.lower()
		if suffix == ".npy":
			np.save(path, self.raw_array)
			return path
		delimiter = "," if suffix == ".csv" else None
		if suffix == ".tsv":
			delimiter = "\t"
		if suffix in {".txt", ".csv", ".tsv"}:
			np.savetxt(path, self.raw_array, fmt="%.9g", delimiter=delimiter)
			return path
		raise ValueError("Supported detector array formats are: .npy, .txt, .csv, .tsv.")
