# -*- coding: utf-8 -*-
"""Reusable transfer-curve editor dialog with optional preview histogram and point dragging."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
	QCheckBox,
	QDialog,
	QDoubleSpinBox,
	QFrame,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QSizePolicy,
	QVBoxLayout,
	QWidget,
)


class _CurvePointRow(QFrame):
	"""One editable transfer-curve row with active-state highlighting."""

	def __init__(self, point_index, point, x_range, y_range, on_activate, on_value_changed, parent=None):
		"""Build one row containing input and output spin boxes."""
		super().__init__(parent)
		self.point_index = int(point_index)
		self._on_activate = on_activate
		self._on_value_changed = on_value_changed
		self.setFrameShape(QFrame.StyledPanel)
		layout = QHBoxLayout(self)
		layout.setContentsMargins(6, 4, 6, 4)
		layout.setSpacing(4)

		self.inputSpin = QDoubleSpinBox()
		self.inputSpin.setRange(float(x_range[0]), float(x_range[1]))
		self.inputSpin.setDecimals(4)
		self.inputSpin.setSingleStep(max(0.0001, (float(x_range[1]) - float(x_range[0])) / 100.0))
		self.inputSpin.setValue(float(point[0]))

		self.outputSpin = QDoubleSpinBox()
		self.outputSpin.setRange(float(y_range[0]), float(y_range[1]))
		self.outputSpin.setDecimals(4)
		self.outputSpin.setSingleStep(max(0.0001, (float(y_range[1]) - float(y_range[0])) / 100.0))
		self.outputSpin.setValue(float(point[1]))

		layout.addWidget(self.inputSpin)
		layout.addWidget(QLabel("->"))
		layout.addWidget(self.outputSpin)

		self.inputSpin.valueChanged.connect(self._emit_value_changed)
		self.outputSpin.valueChanged.connect(self._emit_value_changed)
		self.inputSpin.installEventFilter(self)
		self.outputSpin.installEventFilter(self)
		self.set_active(False)

	def values(self):
		"""Return the current `(x, y)` pair."""
		return float(self.inputSpin.value()), float(self.outputSpin.value())

	def set_active(self, active):
		"""Update the visual highlight of the active row."""
		self.setStyleSheet(
			"QFrame { border: 1px solid #4a90e2; background: rgba(74, 144, 226, 28); }" if active else ""
		)

	def mousePressEvent(self, event):
		"""Activate the row when the user clicks anywhere inside it."""
		self._on_activate(self.point_index)
		super().mousePressEvent(event)

	def eventFilter(self, watched, event):
		"""Activate the row when one of its editors gains focus."""
		if event.type() == event.FocusIn:
			self._on_activate(self.point_index)
		return super().eventFilter(watched, event)

	def _emit_value_changed(self):
		"""Forward editor changes to the dialog controller."""
		self._on_activate(self.point_index)
		self._on_value_changed()


class _TransferCurvePreviewWidget(QWidget):
	"""Paint one compact preview of the curve, with optional histogram."""

	def __init__(self, dialog, parent=None):
		"""Bind the preview widget to one dialog controller."""
		super().__init__(parent)
		self.dialog = dialog
		self._drag_point_index = None
		self._drag_point_radius_px = 9.0
		self.setMinimumSize(240, 220)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setMouseTracking(True)

	def sizeHint(self):
		"""Return a stable default size for dock- and dialog-friendly layouts."""
		return QSize(280, 260)

	def _plot_rect(self):
		"""Return the inner plot rectangle used by the preview painter."""
		margins = (16.0, 12.0, 14.0, 22.0)
		return QRectF(
			margins[0],
			margins[1],
			max(40.0, float(self.width()) - margins[0] - margins[2]),
			max(40.0, float(self.height()) - margins[1] - margins[3]),
		)

	def _map_plot_point(self, plot_rect, x_norm, y_norm):
		"""Map one normalized `(x, y)` point into widget coordinates."""
		return QPointF(
			plot_rect.left() + float(x_norm) * plot_rect.width(),
			plot_rect.bottom() - float(y_norm) * plot_rect.height(),
		)

	def _unmap_plot_point(self, plot_rect, point):
		"""Map one widget position back into normalized plot coordinates."""
		x_value = (float(point.x()) - plot_rect.left()) / max(plot_rect.width(), 1e-6)
		y_value = (plot_rect.bottom() - float(point.y())) / max(plot_rect.height(), 1e-6)
		return np.clip(x_value, 0.0, 1.0), np.clip(y_value, 0.0, 1.0)

	def _point_value_to_norm(self, x_value, y_value):
		"""Map one curve point from value coordinates into normalized plot coordinates."""
		x_min, x_max = self.dialog.x_range
		y_min, y_max = self.dialog.y_range
		x_norm = (float(x_value) - x_min) / max(x_max - x_min, 1e-6)
		y_norm = (float(y_value) - y_min) / max(y_max - y_min, 1e-6)
		return np.clip(x_norm, 0.0, 1.0), np.clip(y_norm, 0.0, 1.0)

	def _norm_to_point_value(self, x_norm, y_norm):
		"""Map one normalized point into the configured value ranges."""
		x_min, x_max = self.dialog.x_range
		y_min, y_max = self.dialog.y_range
		x_value = x_min + float(x_norm) * (x_max - x_min)
		y_value = y_min + float(y_norm) * (y_max - y_min)
		return x_value, y_value

	def _curve_points_screen(self):
		"""Return the current curve points mapped into screen coordinates."""
		plot_rect = self._plot_rect()
		screen_points = []
		for x_value, y_value in self.dialog.points:
			x_norm, y_norm = self._point_value_to_norm(x_value, y_value)
			screen_points.append(self._map_plot_point(plot_rect, x_norm, y_norm))
		return plot_rect, screen_points

	def _point_hit_index(self, position):
		"""Return the index of the point under the cursor, or `None`."""
		_plot_rect, screen_points = self._curve_points_screen()
		best_index = None
		best_distance_sq = self._drag_point_radius_px * self._drag_point_radius_px
		for point_index, point in enumerate(screen_points):
			dx_value = float(position.x()) - float(point.x())
			dy_value = float(position.y()) - float(point.y())
			distance_sq = dx_value * dx_value + dy_value * dy_value
			if distance_sq <= best_distance_sq:
				best_index = point_index
				best_distance_sq = distance_sq
		return best_index

	def _update_cursor(self, position):
		"""Show a pointing cursor when the mouse is over one draggable point."""
		if self._drag_point_index is not None or self._point_hit_index(position) is not None:
			self.setCursor(Qt.PointingHandCursor)
		else:
			self.unsetCursor()

	def _drag_active_point(self, position):
		"""Move the active control point while respecting the neighboring points."""
		if self._drag_point_index is None:
			return
		plot_rect = self._plot_rect()
		x_norm, y_norm = self._unmap_plot_point(plot_rect, position)
		x_value, y_value = self._norm_to_point_value(x_norm, y_norm)
		point_index = int(self._drag_point_index)
		x_min, x_max = self.dialog.x_range
		y_min, y_max = self.dialog.y_range
		x_margin = max(1e-6, (x_max - x_min) * 1e-4)
		if point_index > 0:
			left_limit = float(self.dialog.points[point_index - 1][0]) + x_margin
		else:
			left_limit = x_min
		if point_index < len(self.dialog.points) - 1:
			right_limit = float(self.dialog.points[point_index + 1][0]) - x_margin
		else:
			right_limit = x_max
		if point_index == 0 and self.dialog.lock_endpoints_x[0]:
			x_value = float(self.dialog.points[0][0])
		elif point_index == len(self.dialog.points) - 1 and self.dialog.lock_endpoints_x[1]:
			x_value = float(self.dialog.points[-1][0])
		else:
			x_value = float(np.clip(x_value, left_limit, right_limit))
		y_value = float(np.clip(y_value, y_min, y_max))
		self.dialog.update_point_from_preview(point_index, x_value, y_value)

	def mousePressEvent(self, event):
		"""Start dragging when the user presses one of the curve control points."""
		if event.button() == Qt.LeftButton:
			point_index = self._point_hit_index(event.localPos())
			if point_index is not None:
				self._drag_point_index = int(point_index)
				self.dialog._set_active_index(point_index)
				self._drag_active_point(event.localPos())
				event.accept()
				return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		"""Drag the selected point or update the hover cursor."""
		if self._drag_point_index is not None and bool(event.buttons() & Qt.LeftButton):
			self._drag_active_point(event.localPos())
			event.accept()
			return
		self._update_cursor(event.localPos())
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		"""Stop dragging after the left mouse button is released."""
		if event.button() == Qt.LeftButton and self._drag_point_index is not None:
			self._drag_point_index = None
			self._update_cursor(event.localPos())
			event.accept()
			return
		super().mouseReleaseEvent(event)

	def leaveEvent(self, event):
		"""Restore the normal cursor when the pointer leaves the preview."""
		if self._drag_point_index is None:
			self.unsetCursor()
		super().leaveEvent(event)

	def paintEvent(self, event):
		"""Render the curve preview and the optional histogram."""
		super().paintEvent(event)
		painter = QPainter(self)
		try:
			painter.setRenderHint(QPainter.Antialiasing, True)
			painter.fillRect(self.rect(), self.palette().window())

			plot_rect = self._plot_rect()
			painter.fillRect(plot_rect, QColor(248, 248, 248))
			painter.setPen(QPen(QColor(210, 210, 210), 1))
			for fraction in (0.25, 0.5, 0.75):
				x_coord = plot_rect.left() + fraction * plot_rect.width()
				y_coord = plot_rect.top() + fraction * plot_rect.height()
				painter.drawLine(QPointF(x_coord, plot_rect.top()), QPointF(x_coord, plot_rect.bottom()))
				painter.drawLine(QPointF(plot_rect.left(), y_coord), QPointF(plot_rect.right(), y_coord))
			painter.setPen(QPen(QColor(160, 160, 160), 1))
			painter.drawRect(plot_rect)

			preview_data = self.dialog.preview_data()
			histogram = np.asarray(preview_data.get("histogram", []), dtype=np.float32)
			if histogram.size > 0:
				painter.save()
				painter.setRenderHint(QPainter.Antialiasing, False)
				painter.setPen(Qt.NoPen)
				painter.setBrush(QColor(150, 150, 150, 90))
				bin_edges = np.linspace(plot_rect.left(), plot_rect.right(), histogram.size + 1, dtype=np.float32)
				for bar_index, value in enumerate(histogram):
					bar_height = float(np.clip(value, 0.0, 1.0)) * plot_rect.height()
					left = float(bin_edges[bar_index])
					right = float(bin_edges[bar_index + 1])
					painter.drawRect(QRectF(left, plot_rect.bottom() - bar_height, max(1.0, right - left), bar_height))
				painter.restore()

			painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
			painter.drawLine(self._map_plot_point(plot_rect, 0.0, 0.0), self._map_plot_point(plot_rect, 1.0, 1.0))

			if len(self.dialog.points) >= 2:
				first_x, first_y = self._point_value_to_norm(*self.dialog.points[0])
				path = QPainterPath(self._map_plot_point(plot_rect, first_x, first_y))
				for x_value, y_value in self.dialog.points[1:]:
					x_norm, y_norm = self._point_value_to_norm(x_value, y_value)
					path.lineTo(self._map_plot_point(plot_rect, x_norm, y_norm))
				painter.setPen(QPen(QColor(44, 112, 201), 2))
				painter.setBrush(Qt.NoBrush)
				painter.drawPath(path)

			active_index = int(getattr(self.dialog, "active_index", -1))
			for point_index, (x_value, y_value) in enumerate(self.dialog.points):
				x_norm, y_norm = self._point_value_to_norm(x_value, y_value)
				point = self._map_plot_point(plot_rect, x_norm, y_norm)
				if point_index == active_index:
					painter.setPen(QPen(QColor(44, 112, 201), 2))
					painter.setBrush(QColor(255, 255, 255))
					painter.drawEllipse(point, 5.5, 5.5)
				painter.setPen(QPen(QColor(44, 112, 201), 1))
				painter.setBrush(QColor(44, 112, 201))
				painter.drawEllipse(point, 3.0, 3.0)

			painter.setPen(QPen(QColor(90, 90, 90), 1))
			painter.drawText(
				QRectF(plot_rect.left(), plot_rect.bottom() + 4.0, plot_rect.width(), 16.0),
				f"{self.dialog.x_label} {self.dialog.x_range[0]:.6g} ... {self.dialog.x_range[1]:.6g}",
			)
			painter.save()
			painter.translate(4.0, plot_rect.bottom())
			painter.rotate(-90.0)
			painter.drawText(
				QRectF(0.0, 0.0, plot_rect.height(), 16.0),
				f"{self.dialog.y_label} {self.dialog.y_range[0]:.6g} ... {self.dialog.y_range[1]:.6g}",
			)
			painter.restore()
		finally:
			painter.end()


class TransferCurveDialog(QDialog):
	"""Edit one piecewise-linear curve with optional histogram preview."""

	def __init__(
		self,
		curve_owner,
		on_curve_changed=None,
		parent=None,
		*,
		title=None,
		points_getter=None,
		points_setter=None,
		preview_provider=None,
		scope_text_provider=None,
		x_label="input",
		y_label="output",
		x_range=(0.0, 100.0),
		y_range=(0.0, 100.0),
		show_histogram=True,
		ignore_zero_values_default=True,
		lock_endpoints_x=(True, True),
	):
		"""Build the dialog using either a detector-like owner or explicit callbacks."""
		super().__init__(parent)
		self.curve_owner = curve_owner
		self.on_curve_changed = on_curve_changed
		self.points_getter = points_getter or self._default_points_getter
		self.points_setter = points_setter or self._default_points_setter
		self.preview_provider = preview_provider or self._default_preview_provider
		self.scope_text_provider = scope_text_provider or self._default_scope_text_provider
		self.x_label = str(x_label)
		self.y_label = str(y_label)
		self.x_range = (float(x_range[0]), float(x_range[1]))
		self.y_range = (float(y_range[0]), float(y_range[1]))
		self.show_histogram = bool(show_histogram)
		self.lock_endpoints_x = (bool(lock_endpoints_x[0]), bool(lock_endpoints_x[1]))
		self.points = list(self.points_getter())
		self.active_index = 0
		self._rebuilding = False
		window_title = str(title) if title is not None else f"Transfer Curve: {getattr(curve_owner, 'label', 'Curve')}"
		self.setWindowTitle(window_title)
		self.resize(680, 380)

		layout = QVBoxLayout(self)
		self.scopeLabel = QLabel("")
		self.scopeLabel.setWordWrap(True)
		layout.addWidget(self.scopeLabel)

		self.ignoreZerosCheck = QCheckBox("Ignore zeros in histogram")
		self.ignoreZerosCheck.setChecked(bool(ignore_zero_values_default))
		self.ignoreZerosCheck.setVisible(self.show_histogram)
		layout.addWidget(self.ignoreZerosCheck)

		body = QWidget()
		body_layout = QHBoxLayout(body)
		body_layout.setContentsMargins(0, 0, 0, 0)
		body_layout.setSpacing(12)
		layout.addWidget(body, 1)

		self.rowsHost = QWidget()
		self.rowsLayout = QVBoxLayout(self.rowsHost)
		self.rowsLayout.setContentsMargins(0, 0, 0, 0)
		self.rowsLayout.setSpacing(4)
		body_layout.addWidget(self.rowsHost, 0)

		preview_host = QWidget()
		preview_layout = QVBoxLayout(preview_host)
		preview_layout.setContentsMargins(0, 0, 0, 0)
		preview_layout.setSpacing(4)
		self.previewTitleLabel = QLabel("Curve preview" if not self.show_histogram else "Curve preview with input histogram")
		self.previewTitleLabel.setWordWrap(True)
		self.previewWidget = _TransferCurvePreviewWidget(self, parent=preview_host)
		preview_layout.addWidget(self.previewTitleLabel)
		preview_layout.addWidget(self.previewWidget, 1)
		body_layout.addWidget(preview_host, 1)

		button_row = QWidget()
		button_layout = QHBoxLayout(button_row)
		button_layout.setContentsMargins(0, 0, 0, 0)
		self.addButton = QPushButton("Add")
		self.removeButton = QPushButton("Remove")
		self.closeButton = QPushButton("Close")
		button_layout.addWidget(self.addButton)
		button_layout.addWidget(self.removeButton)
		button_layout.addStretch(1)
		button_layout.addWidget(self.closeButton)
		layout.addWidget(button_row)

		self.addButton.clicked.connect(self.on_add_point)
		self.removeButton.clicked.connect(self.on_remove_point)
		self.closeButton.clicked.connect(self.accept)
		self.ignoreZerosCheck.toggled.connect(self.previewWidget.update)
		self._rebuild_rows()

	def _default_points_getter(self):
		"""Return points from a detector-like owner using the existing attribute name."""
		return list(getattr(self.curve_owner, "transfer_points_pct", [(self.x_range[0], self.y_range[0]), (self.x_range[1], self.y_range[1])]))

	def _default_points_setter(self, points):
		"""Store points into a detector-like owner using the existing mutator method."""
		self.curve_owner.set_transfer_points_pct(points)

	def _default_preview_provider(self, *, ignore_zero_values):
		"""Return histogram preview data from a detector-like owner when available."""
		if not self.show_histogram or not hasattr(self.curve_owner, "transfer_curve_preview_data"):
			return {}
		return dict(self.curve_owner.transfer_curve_preview_data(
			histogram_bins=100,
			ignore_zero_values=bool(ignore_zero_values),
		))

	def _default_scope_text_provider(self):
		"""Return the default detector-specific scope text when supported."""
		if not hasattr(self.curve_owner, "window_width") or not hasattr(self.curve_owner, "effective_window"):
			return ""
		if self.curve_owner.window_width is None or float(self.curve_owner.window_width) <= 0.0:
			return "Scope: full data range (window width = 0)"
		center, width = self.curve_owner.effective_window()
		return f"Scope: active window, C={center:.6g}, W={width:.6g}"

	def preview_data(self):
		"""Return preview data from the configured provider."""
		return dict(self.preview_provider(ignore_zero_values=self.ignoreZerosCheck.isChecked()))

	def _rebuild_rows(self):
		"""Rebuild the editable row list from the current point array."""
		self._rebuilding = True
		self.scopeLabel.setText(str(self.scope_text_provider() or ""))
		self.scopeLabel.setVisible(bool(self.scopeLabel.text()))
		while self.rowsLayout.count():
			item = self.rowsLayout.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
		self.active_index = min(max(0, self.active_index), len(self.points) - 1)
		for point_index, point in enumerate(self.points):
			row = _CurvePointRow(
				point_index=point_index,
				point=point,
				x_range=self.x_range,
				y_range=self.y_range,
				on_activate=self._set_active_index,
				on_value_changed=self._on_rows_changed,
				parent=self.rowsHost,
			)
			row.set_active(point_index == self.active_index)
			self.rowsLayout.addWidget(row)
		self.rowsLayout.addStretch(1)
		self.removeButton.setEnabled(len(self.points) > 2)
		self.addButton.setEnabled(self.active_index < len(self.points) - 1)
		self._rebuilding = False
		self.previewWidget.update()

	def _set_active_index(self, point_index):
		"""Mark one row as active and refresh row highlighting."""
		self.active_index = int(point_index)
		for row_index in range(self.rowsLayout.count()):
			widget = self.rowsLayout.itemAt(row_index).widget()
			if isinstance(widget, _CurvePointRow):
				widget.set_active(widget.point_index == self.active_index)
		self.addButton.setEnabled(self.active_index < len(self.points) - 1)
		self.removeButton.setEnabled(len(self.points) > 2)
		self.previewWidget.update()

	def _row_widgets(self):
		"""Yield the currently instantiated editable curve-point rows."""
		for row_index in range(self.rowsLayout.count()):
			widget = self.rowsLayout.itemAt(row_index).widget()
			if isinstance(widget, _CurvePointRow):
				yield widget

	def _sync_points_from_rows(self):
		"""Read the current row editor values into the local point list."""
		self.points = [widget.values() for widget in self._row_widgets()]

	def _sync_rows_to_points(self):
		"""Push the local point list into the visible row editors without rebuilding them."""
		for widget, point in zip(self._row_widgets(), self.points):
			widget.inputSpin.blockSignals(True)
			widget.outputSpin.blockSignals(True)
			widget.inputSpin.setValue(float(point[0]))
			widget.outputSpin.setValue(float(point[1]))
			widget.inputSpin.blockSignals(False)
			widget.outputSpin.blockSignals(False)

	def _apply_points(self):
		"""Persist the current point list and refresh dependent UI."""
		self.points_setter(self.points)
		self.points = list(self.points_getter())
		self.previewWidget.update()
		if callable(self.on_curve_changed):
			self.on_curve_changed()

	def _on_rows_changed(self):
		"""Store edited point values after any spin-box change."""
		if self._rebuilding:
			return
		self._sync_points_from_rows()
		self._apply_points()
		self._rebuild_rows()

	def update_point_from_preview(self, point_index, x_value, y_value):
		"""Apply one interactive point move from the preview canvas."""
		if point_index < 0 or point_index >= len(self.points):
			return
		self.points[point_index] = (
			float(np.clip(x_value, self.x_range[0], self.x_range[1])),
			float(np.clip(y_value, self.y_range[0], self.y_range[1])),
		)
		self._set_active_index(point_index)
		self._sync_rows_to_points()
		self._apply_points()
		self._sync_rows_to_points()

	@pyqtSlot()
	def on_add_point(self):
		"""Insert one point midway between the active point and the next point."""
		if self.active_index >= len(self.points) - 1:
			return
		self._sync_points_from_rows()
		point_a = self.points[self.active_index]
		point_b = self.points[self.active_index + 1]
		self.points.insert(
			self.active_index + 1,
			(
				(float(point_a[0]) + float(point_b[0])) / 2.0,
				(float(point_a[1]) + float(point_b[1])) / 2.0,
			),
		)
		self.active_index += 1
		self._apply_points()
		self._rebuild_rows()

	@pyqtSlot()
	def on_remove_point(self):
		"""Remove the currently active point unless only the edge points remain."""
		if len(self.points) <= 2:
			return
		self._sync_points_from_rows()
		self.points.pop(self.active_index)
		self.active_index = min(self.active_index, len(self.points) - 1)
		self._apply_points()
		self._rebuild_rows()
