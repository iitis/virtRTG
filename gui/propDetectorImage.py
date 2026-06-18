# -*- coding: utf-8 -*-
"""Property panel for the plugin-local `DetectorImage` scene object."""

from __future__ import annotations

import weakref

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDialog,
	QDoubleSpinBox,
	QFileDialog,
	QFrame,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QPushButton,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from dpVision import AP
from dpVision.gui.propBaseObject import PropBaseObject
from dpVision.gui.propWidget import PropWidget

from ..detectorImage import DetectorImage
from .detectorImageViewer import DetectorImageViewerChild


class _CurvePointRow(QFrame):
	"""One editable transfer-curve row with active-state highlighting."""

	def __init__(self, point_index, point, on_activate, on_value_changed, parent=None):
		"""Build one row containing input and output percentage spin boxes."""
		super().__init__(parent)
		self.point_index = int(point_index)
		self._on_activate = on_activate
		self._on_value_changed = on_value_changed
		self.setFrameShape(QFrame.StyledPanel)
		layout = QHBoxLayout(self)
		layout.setContentsMargins(6, 4, 6, 4)
		layout.setSpacing(4)

		self.inputSpin = QDoubleSpinBox()
		self.inputSpin.setRange(0.0, 100.0)
		self.inputSpin.setDecimals(2)
		self.inputSpin.setSingleStep(1.0)
		self.inputSpin.setValue(float(point[0]))

		self.outputSpin = QDoubleSpinBox()
		self.outputSpin.setRange(0.0, 100.0)
		self.outputSpin.setDecimals(2)
		self.outputSpin.setSingleStep(1.0)
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
		"""Return the current `(input_pct, output_pct)` pair."""
		return float(self.inputSpin.value()), float(self.outputSpin.value())

	def set_active(self, active):
		"""Update the visual highlight of the active row."""
		if active:
			self.setStyleSheet("QFrame { border: 1px solid #4a90e2; background: rgba(74, 144, 226, 28); }")
		else:
			self.setStyleSheet("")

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


class TransferCurveDialog(QDialog):
	"""Simple transfer-curve editor using percent input/output control points."""

	def __init__(self, detector_image, on_curve_changed=None, parent=None):
		"""Build the dialog and initialize it from the detector image object."""
		super().__init__(parent)
		self.detector_image = detector_image
		self.on_curve_changed = on_curve_changed
		self.points = list(getattr(detector_image, "transfer_points_pct", [(0.0, 0.0), (100.0, 100.0)]))
		self.active_index = 0
		self._rebuilding = False
		self.setWindowTitle(f"Transfer Curve: {detector_image.label}")
		self.resize(320, 360)

		layout = QVBoxLayout(self)
		self.scopeLabel = QLabel("")
		self.scopeLabel.setWordWrap(True)
		layout.addWidget(self.scopeLabel)

		self.rowsHost = QWidget()
		self.rowsLayout = QVBoxLayout(self.rowsHost)
		self.rowsLayout.setContentsMargins(0, 0, 0, 0)
		self.rowsLayout.setSpacing(4)
		layout.addWidget(self.rowsHost)

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
		self._rebuild_rows()

	def _current_scope_text(self):
		"""Return a short note describing whether the curve uses the window or full range."""
		if self.detector_image.window_width is None or float(self.detector_image.window_width) <= 0.0:
			return "Scope: full data range (window width = 0)"
		center, width = self.detector_image.effective_window()
		return f"Scope: active window, C={center:.6g}, W={width:.6g}"

	def _rebuild_rows(self):
		"""Rebuild the editable row list from the current point array."""
		self._rebuilding = True
		self.scopeLabel.setText(self._current_scope_text())
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

	def _set_active_index(self, point_index):
		"""Mark one row as active and refresh row highlighting."""
		self.active_index = int(point_index)
		for row_index in range(self.rowsLayout.count()):
			widget = self.rowsLayout.itemAt(row_index).widget()
			if isinstance(widget, _CurvePointRow):
				widget.set_active(widget.point_index == self.active_index)
		self.addButton.setEnabled(self.active_index < len(self.points) - 1)
		self.removeButton.setEnabled(len(self.points) > 2)

	def _sync_points_from_rows(self):
		"""Read the current row editor values into the local point list."""
		points = []
		for row_index in range(self.rowsLayout.count()):
			widget = self.rowsLayout.itemAt(row_index).widget()
			if isinstance(widget, _CurvePointRow):
				points.append(widget.values())
		self.points = points

	def _apply_points_to_object(self):
		"""Persist the current point list and refresh dependent UI."""
		self.detector_image.set_transfer_points_pct(self.points)
		self.points = list(self.detector_image.transfer_points_pct)
		if callable(self.on_curve_changed):
			self.on_curve_changed()

	def _on_rows_changed(self):
		"""Store edited point values after any spin-box change."""
		if self._rebuilding:
			return
		self._sync_points_from_rows()
		self._apply_points_to_object()
		self._rebuild_rows()

	@pyqtSlot()
	def on_add_point(self):
		"""Insert one point midway between the active point and the next point."""
		if self.active_index >= len(self.points) - 1:
			return
		self._sync_points_from_rows()
		point_a = self.points[self.active_index]
		point_b = self.points[self.active_index + 1]
		new_point = (
			(float(point_a[0]) + float(point_b[0])) / 2.0,
			(float(point_a[1]) + float(point_b[1])) / 2.0,
		)
		self.points.insert(self.active_index + 1, new_point)
		self.active_index += 1
		self._apply_points_to_object()
		self._rebuild_rows()

	@pyqtSlot()
	def on_remove_point(self):
		"""Remove the currently active point unless only the edge points remain."""
		if len(self.points) <= 2:
			return
		self._sync_points_from_rows()
		self.points.pop(self.active_index)
		self.active_index = min(self.active_index, len(self.points) - 1)
		self._apply_points_to_object()
		self._rebuild_rows()


class PropDetectorImage(PropWidget):
	"""Edit display settings and viewer actions for one detector image object."""

	def __init__(self, _obj: DetectorImage, parent=None):
		"""Build the detector-image property editor."""
		super().__init__(parent)
		self.obj_ref = weakref.ref(_obj)
		self._setup_ui()

	def _setup_ui(self):
		"""Create one compact manual property panel."""
		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)

		info_group = QGroupBox("Detector Image")
		info_form = QFormLayout(info_group)
		self.infoLabel = QLabel("")
		self.windowLabel = QLabel("")
		self.rangeLabel = QLabel("")
		self.modeLabel = QLabel("")
		self.transferLabel = QLabel("")
		self.transferLabel.setWordWrap(True)
		info_form.addRow("Info:", self.infoLabel)
		info_form.addRow("Mode:", self.modeLabel)
		info_form.addRow("Window:", self.windowLabel)
		info_form.addRow("Range:", self.rangeLabel)
		info_form.addRow("Curve:", self.transferLabel)
		layout.addWidget(info_group)

		preview_group = QGroupBox("Preview")
		preview_layout = QVBoxLayout(preview_group)
		self.thumbnailLabel = QLabel("brak podglądu")
		self.thumbnailLabel.setAlignment(Qt.AlignCenter)
		self.thumbnailLabel.setMinimumHeight(140)
		preview_layout.addWidget(self.thumbnailLabel)
		layout.addWidget(preview_group)

		display_group = QGroupBox("Display")
		display_form = QFormLayout(display_group)
		self.modeCombo = QComboBox()
		self.modeCombo.addItems(["raw", "digital", "film"])
		self.windowCenterSpin = QDoubleSpinBox()
		self.windowCenterSpin.setRange(-1e12, 1e12)
		self.windowCenterSpin.setDecimals(6)
		self.windowCenterSpin.setSingleStep(1.0)
		self.windowWidthSpin = QDoubleSpinBox()
		self.windowWidthSpin.setRange(0.0, 1e12)
		self.windowWidthSpin.setDecimals(6)
		self.windowWidthSpin.setSingleStep(1.0)
		self.gammaSpin = QDoubleSpinBox()
		self.gammaSpin.setRange(0.05, 10.0)
		self.gammaSpin.setDecimals(3)
		self.gammaSpin.setSingleStep(0.05)
		self.contrastSpin = QDoubleSpinBox()
		self.contrastSpin.setRange(0.05, 10.0)
		self.contrastSpin.setDecimals(3)
		self.contrastSpin.setSingleStep(0.05)
		self.robustPercentileSpin = QDoubleSpinBox()
		self.robustPercentileSpin.setRange(50.0, 100.0)
		self.robustPercentileSpin.setDecimals(2)
		self.robustPercentileSpin.setSingleStep(0.1)
		self.overlayAnnotationsCheck = QCheckBox("Show projected annotations")
		self.overlayLabelsCheck = QCheckBox("Show annotation labels")
		self.overlayCrossSizeSpin = QSpinBox()
		self.overlayCrossSizeSpin.setRange(1, 256)
		self.onlyWindowRangeCheck = QCheckBox("Only window range")
		self.invertCheck = QCheckBox("Invert")
		display_form.addRow("Mode:", self.modeCombo)
		display_form.addRow("Window center:", self.windowCenterSpin)
		display_form.addRow("Window width:", self.windowWidthSpin)
		display_form.addRow("Gamma:", self.gammaSpin)
		display_form.addRow("Contrast:", self.contrastSpin)
		display_form.addRow("Robust [%]:", self.robustPercentileSpin)
		display_form.addRow("Cross size [px]:", self.overlayCrossSizeSpin)
		display_form.addRow("", self.overlayAnnotationsCheck)
		display_form.addRow("", self.overlayLabelsCheck)
		display_form.addRow("", self.onlyWindowRangeCheck)
		display_form.addRow("", self.invertCheck)
		layout.addWidget(display_group)

		button_row = QWidget()
		button_layout = QHBoxLayout(button_row)
		button_layout.setContentsMargins(0, 0, 0, 0)
		self.autoWindowButton = QPushButton("Auto window")
		self.fullRangeButton = QPushButton("Full range")
		self.showWindowButton = QPushButton("Show 2D")
		self.editCurveButton = QPushButton("Edit curve")
		self.savePngButton = QPushButton("Save PNG")
		self.importArrayButton = QPushButton("Import array")
		self.exportArrayButton = QPushButton("Export array")
		button_layout.addWidget(self.autoWindowButton)
		button_layout.addWidget(self.fullRangeButton)
		button_layout.addWidget(self.showWindowButton)
		button_layout.addWidget(self.editCurveButton)
		button_layout.addWidget(self.savePngButton)
		button_layout.addWidget(self.importArrayButton)
		button_layout.addWidget(self.exportArrayButton)
		layout.addWidget(button_row)
		layout.addStretch(1)

		self.modeCombo.currentTextChanged.connect(self.on_display_changed)
		self.windowCenterSpin.valueChanged.connect(self.on_display_changed)
		self.windowWidthSpin.valueChanged.connect(self.on_display_changed)
		self.gammaSpin.valueChanged.connect(self.on_display_changed)
		self.contrastSpin.valueChanged.connect(self.on_display_changed)
		self.robustPercentileSpin.valueChanged.connect(self.on_display_changed)
		self.overlayAnnotationsCheck.toggled.connect(self.on_display_changed)
		self.overlayLabelsCheck.toggled.connect(self.on_display_changed)
		self.overlayCrossSizeSpin.valueChanged.connect(self.on_display_changed)
		self.onlyWindowRangeCheck.toggled.connect(self.on_display_changed)
		self.invertCheck.toggled.connect(self.on_display_changed)
		self.autoWindowButton.clicked.connect(self.on_auto_window)
		self.fullRangeButton.clicked.connect(self.on_full_range)
		self.showWindowButton.clicked.connect(self.on_show_window)
		self.editCurveButton.clicked.connect(self.on_edit_curve)
		self.savePngButton.clicked.connect(self.on_save_png)
		self.importArrayButton.clicked.connect(self.on_import_array)
		self.exportArrayButton.clicked.connect(self.on_export_array)

	@staticmethod
	def create(m, parent=0):
		"""Build the combined base-object and detector-image property panel."""
		return PropWidget.build([PropDetectorImage(m), PropBaseObject(m)], parent)

	def blockAll(self, blocked):
		"""Block or unblock signals on the interactive controls of this panel."""
		for widget in (
			self.modeCombo,
			self.windowCenterSpin,
			self.windowWidthSpin,
			self.gammaSpin,
			self.contrastSpin,
			self.robustPercentileSpin,
			self.overlayAnnotationsCheck,
			self.overlayLabelsCheck,
			self.overlayCrossSizeSpin,
			self.onlyWindowRangeCheck,
			self.invertCheck,
		):
			widget.blockSignals(blocked)

	def _update_display_visibility(self, obj):
		"""Enable only controls that affect the currently selected presentation mode."""
		mode = str(obj.presentation_mode).lower()
		is_raw = mode == "raw"
		is_digital = mode == "digital"
		is_film = mode == "film"
		self.windowCenterSpin.setEnabled(is_raw or is_digital)
		self.windowWidthSpin.setEnabled(is_raw or is_digital)
		self.fullRangeButton.setEnabled(is_raw or is_digital)
		self.onlyWindowRangeCheck.setEnabled(is_raw or is_digital)
		self.invertCheck.setEnabled(True)
		self.gammaSpin.setEnabled(not is_raw)
		self.contrastSpin.setEnabled(not is_raw)
		self.robustPercentileSpin.setEnabled(is_digital or is_film)
		self.autoWindowButton.setEnabled(True)
		overlay_enabled = bool(getattr(obj, "presentation_overlay_annotations", False))
		self.overlayLabelsCheck.setEnabled(overlay_enabled)
		self.overlayCrossSizeSpin.setEnabled(overlay_enabled)

	def updateProperties(self):
		"""Synchronize widget values with the current detector image object."""
		obj = self.obj_ref()
		if obj is None:
			return
		self.blockAll(True)
		stats = obj.data_stats()
		height, width = obj.shape_hw()
		center, window_width = obj.effective_window()
		stored_center = 0.0 if obj.presentation_window_center is None else float(obj.presentation_window_center)
		stored_width = 0.0 if obj.presentation_window_width is None else float(obj.presentation_window_width)
		self.infoLabel.setText(f"{width} x {height} px, stage={obj.source_stage}")
		self.modeLabel.setText(str(obj.presentation_mode))
		if stored_width <= 0.0:
			self.windowLabel.setText(f"full range -> effective C={center:.6g}, W={window_width:.6g}")
		else:
			self.windowLabel.setText(f"C={center:.6g}, W={window_width:.6g}")
		self.rangeLabel.setText(f"{stats['min']:.6g} .. {stats['max']:.6g}")
		self.transferLabel.setText(obj.transfer_points_summary())
		self.modeCombo.setCurrentText(str(obj.presentation_mode))
		self.windowCenterSpin.setValue(stored_center)
		self.windowWidthSpin.setValue(max(0.0, stored_width))
		self.gammaSpin.setValue(float(obj.presentation_gamma))
		self.contrastSpin.setValue(float(obj.presentation_contrast))
		self.robustPercentileSpin.setValue(float(obj.presentation_robust_percentile))
		self.overlayAnnotationsCheck.setChecked(bool(getattr(obj, "presentation_overlay_annotations", False)))
		self.overlayLabelsCheck.setChecked(bool(getattr(obj, "presentation_overlay_labels", False)))
		self.overlayCrossSizeSpin.setValue(int(getattr(obj, "presentation_overlay_cross_size_px", 6)))
		self.onlyWindowRangeCheck.setChecked(bool(getattr(obj, "display_only_window_range", False)))
		self.invertCheck.setChecked(bool(obj.presentation_invert))
		pixmap = QPixmap.fromImage(obj.toQImage())
		if pixmap.isNull():
			self.thumbnailLabel.setText("brak podglądu")
		else:
			thumb = pixmap.scaled(
				self.thumbnailLabel.width() or 240,
				self.thumbnailLabel.height() or 160,
				Qt.KeepAspectRatio,
				Qt.SmoothTransformation,
			)
			self.thumbnailLabel.setPixmap(thumb)
		self._update_display_visibility(obj)
		self.blockAll(False)

	def _refresh_viewers(self, obj):
		"""Refresh open detector-image viewers bound to the same object."""
		mdi_area = getattr(AP.mainWin, "mdiArea", None)
		if mdi_area is None:
			return
		for sub_window in mdi_area.subWindowList():
			widget = sub_window.widget()
			if getattr(widget, "m_widget", None) is obj and hasattr(widget, "_viewer"):
				widget._viewer.refresh_from_object()

	@pyqtSlot()
	def on_display_changed(self):
		"""Store the current display settings and refresh the preview and open viewers."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.presentation_mode = str(self.modeCombo.currentText())
		obj.presentation_window_center = float(self.windowCenterSpin.value())
		window_width = float(self.windowWidthSpin.value())
		obj.presentation_window_width = window_width if window_width > 0.0 else 0.0
		obj.presentation_gamma = max(0.05, float(self.gammaSpin.value()))
		obj.presentation_contrast = max(0.05, float(self.contrastSpin.value()))
		obj.presentation_robust_percentile = min(100.0, max(50.0, float(self.robustPercentileSpin.value())))
		obj.presentation_overlay_annotations = bool(self.overlayAnnotationsCheck.isChecked())
		obj.presentation_overlay_labels = bool(self.overlayLabelsCheck.isChecked())
		obj.presentation_overlay_cross_size_px = max(1, int(self.overlayCrossSizeSpin.value()))
		obj.display_only_window_range = bool(self.onlyWindowRangeCheck.isChecked())
		obj.presentation_invert = bool(self.invertCheck.isChecked())
		self.updateProperties()
		self._refresh_viewers(obj)
		AP.updateAllViews()

	@pyqtSlot()
	def on_auto_window(self):
		"""Set a robust automatic display window."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.auto_window(robust_percentile=float(self.robustPercentileSpin.value()))
		self.updateProperties()
		self._refresh_viewers(obj)

	@pyqtSlot()
	def on_full_range(self):
		"""Reset the display window to the full finite data range."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.reset_window_to_full_range()
		self.updateProperties()
		self._refresh_viewers(obj)

	@pyqtSlot()
	def on_show_window(self):
		"""Open one 2D detector-image viewer window."""
		obj = self.obj_ref()
		if obj is None:
			return
		child = DetectorImageViewerChild(obj, AP.mainWin.mdiArea)
		child.setMinimumSize(400, 300)
		sub_window = AP.mainWin.mdiArea.addSubWindow(child)
		sub_window.setWindowTitle(f"Detector Image: {obj.label}")
		sub_window.showNormal()
		sub_window.update()

	@pyqtSlot()
	def on_edit_curve(self):
		"""Open the simple piecewise transfer-curve editor dialog."""
		obj = self.obj_ref()
		if obj is None:
			return
		dialog = TransferCurveDialog(
			obj,
			on_curve_changed=lambda: (self._refresh_viewers(obj), AP.updateProperties()),
			parent=self,
		)
		dialog.exec_()

	@pyqtSlot()
	def on_save_png(self):
		"""Save the currently displayed detector image view as PNG."""
		obj = self.obj_ref()
		if obj is None:
			return
		path, _ = QFileDialog.getSaveFileName(self, "Save detector image", "", "PNG (*.png)")
		if not path:
			return
		if not path.lower().endswith(".png"):
			path += ".png"
		try:
			obj.save_png(path)
		except Exception as exc:
			QMessageBox.critical(self, "Save detector image", str(exc))

	@pyqtSlot()
	def on_import_array(self):
		"""Import one detector array or detector-image package into this object."""
		obj = self.obj_ref()
		if obj is None:
			return
		path, _ = QFileDialog.getOpenFileName(
			self,
			"Import detector array",
			"",
			"Detector array (*.npz *.npy *.txt *.csv *.tsv);;Projection package (*.npz);;NumPy (*.npy);;Text (*.txt);;CSV (*.csv);;TSV (*.tsv)",
		)
		if not path:
			return
		try:
			obj.import_array(path, auto_window=False)
		except Exception as exc:
			QMessageBox.critical(self, "Import detector array", str(exc))
			return
		self.updateProperties()
		self._refresh_viewers(obj)
		AP.mainWin.dock["workspace"].refreshAll()
		AP.updateAllViews()

	@pyqtSlot()
	def on_export_array(self):
		"""Export the raw detector array as NumPy or text."""
		obj = self.obj_ref()
		if obj is None:
			return
		path, _ = QFileDialog.getSaveFileName(
			self,
			"Export detector array",
			f"{obj.label}.npz",
			"Detector array (*.npz *.npy *.txt *.csv *.tsv);;Projection package (*.npz);;NumPy (*.npy);;Text (*.txt);;CSV (*.csv);;TSV (*.tsv)",
		)
		if not path:
			return
		try:
			obj.export_array(path)
		except Exception as exc:
			QMessageBox.critical(self, "Export detector array", str(exc))
