# -*- coding: utf-8 -*-
"""Plugin-local 2D viewer for `DetectorImage` objects with interactive window/level."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap

from dpVision import AP


class DetectorImageViewer(QWidget):
	"""Show one detector image and allow interactive zoom and window/level changes."""

	def __init__(self, detector_image=None, parent=None):
		"""Create one scrollable image viewer."""
		super().__init__(parent)
		self._zoom = 1.0
		self._pixmap = QPixmap()
		self._detector_image = None
		self._drag_origin = None
		self._drag_window = None
		self._drag_mode = None

		self._label = QLabel()
		self._label.setAlignment(Qt.AlignCenter)
		self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self._label.setMinimumSize(1, 1)
		self._label.setMouseTracking(True)

		self._scroll = QScrollArea()
		self._scroll.setWidget(self._label)
		self._scroll.setWidgetResizable(False)
		self._scroll.setAlignment(Qt.AlignCenter)
		self._scroll.viewport().setMouseTracking(True)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._scroll)

		self._label.installEventFilter(self)
		self._scroll.viewport().installEventFilter(self)

		if detector_image is not None:
			self.setImage(detector_image)

	def setImage(self, detector_image):
		"""Bind the viewer to one detector image object and refresh its pixmap."""
		self._detector_image = detector_image
		self._zoom = 1.0
		self.refresh_from_object()

	def refresh_from_object(self):
		"""Rebuild the displayed pixmap from the currently bound detector object."""
		if self._detector_image is None:
			return
		self._pixmap = QPixmap.fromImage(self._detector_image.toQImage())
		self._updateDisplay()

	def _updateDisplay(self):
		if self._pixmap.isNull():
			return
		width = max(1, int(self._pixmap.width() * self._zoom))
		height = max(1, int(self._pixmap.height() * self._zoom))
		scaled = self._pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		self._label.setPixmap(scaled)
		self._label.resize(scaled.size())

	def wheelEvent(self, event):
		"""Zoom with the mouse wheel."""
		delta = event.angleDelta().y()
		factor = 1.15 if delta > 0 else 1.0 / 1.15
		self._zoom = max(0.05, min(self._zoom * factor, 32.0))
		self._updateDisplay()

	def eventFilter(self, watched, event):
		"""Handle drag-based window/level interaction on the viewer label and viewport."""
		if self._detector_image is None:
			return super().eventFilter(watched, event)

		if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
			self._drag_origin = event.globalPos()
			self._drag_window = self._detector_image.effective_window()
			self._drag_mode = None
			return True

		if event.type() == QEvent.MouseMove and self._drag_origin is not None and (event.buttons() & Qt.LeftButton):
			self._update_window_from_drag(event.globalPos())
			return True

		if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
			if self._drag_origin is not None:
				self._update_window_from_drag(event.globalPos())
			self._drag_origin = None
			self._drag_window = None
			self._drag_mode = None
			return True

		return super().eventFilter(watched, event)

	def _update_window_from_drag(self, global_pos: QPoint):
		"""Map drag deltas to either window center or width using a dominant-axis lock."""
		if self._detector_image is None or self._drag_origin is None or self._drag_window is None:
			return
		stats = self._detector_image.data_stats()
		data_span = max(stats["max"] - stats["min"], 1e-6)
		delta = global_pos - self._drag_origin
		delta_x = float(delta.x())
		delta_y = float(delta.y())
		if self._drag_mode is None:
			lock_threshold_px = 6.0
			if abs(delta_x) >= lock_threshold_px or abs(delta_y) >= lock_threshold_px:
				self._drag_mode = "center" if abs(delta_x) >= abs(delta_y) else "width"
			else:
				return
		center0, width0 = self._drag_window
		center = center0
		width = width0
		if self._drag_mode == "center":
			center = center0 + delta_x * data_span / 300.0
		else:
			width = max(1e-6, width0 + delta_y * data_span / 300.0)
		self._detector_image.window_center = float(center)
		self._detector_image.window_width = float(width)
		self.refresh_from_object()
		AP.updateProperties()


class DetectorImageViewerChild(QWidget):
	"""MDI child wrapper exposing the detector image object through `m_widget`."""

	def __init__(self, detector_image, parent=None):
		"""Embed one detector-image viewer in a widget compatible with the MDI area."""
		super().__init__(parent)
		self.m_widget = detector_image

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)

		self._viewer = DetectorImageViewer(detector_image, self)
		layout.addWidget(self._viewer)
