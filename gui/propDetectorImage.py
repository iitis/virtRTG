# -*- coding: utf-8 -*-
"""Property panel for the plugin-local `DetectorImage` scene object."""

from __future__ import annotations

import weakref

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
	QCheckBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from dpVision import AP
from dpVision.gui.propBaseObject import PropBaseObject
from dpVision.gui.propWidget import PropWidget

from ..detectorImage import DetectorImage
from .detectorImageViewer import DetectorImageViewerChild


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
		info_form.addRow("Info:", self.infoLabel)
		info_form.addRow("Window:", self.windowLabel)
		info_form.addRow("Range:", self.rangeLabel)
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
		self.windowCenterSpin = QDoubleSpinBox()
		self.windowCenterSpin.setRange(-1e12, 1e12)
		self.windowCenterSpin.setDecimals(6)
		self.windowCenterSpin.setSingleStep(1.0)
		self.windowWidthSpin = QDoubleSpinBox()
		self.windowWidthSpin.setRange(1e-6, 1e12)
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
		self.onlyWindowRangeCheck = QCheckBox("Only window range")
		self.invertCheck = QCheckBox("Invert")
		display_form.addRow("Window center:", self.windowCenterSpin)
		display_form.addRow("Window width:", self.windowWidthSpin)
		display_form.addRow("Gamma:", self.gammaSpin)
		display_form.addRow("Contrast:", self.contrastSpin)
		display_form.addRow("Robust [%]:", self.robustPercentileSpin)
		display_form.addRow("", self.onlyWindowRangeCheck)
		display_form.addRow("", self.invertCheck)
		layout.addWidget(display_group)

		button_row = QWidget()
		button_layout = QHBoxLayout(button_row)
		button_layout.setContentsMargins(0, 0, 0, 0)
		self.autoWindowButton = QPushButton("Auto window")
		self.fullRangeButton = QPushButton("Full range")
		self.showWindowButton = QPushButton("Show 2D")
		self.savePngButton = QPushButton("Save PNG")
		self.exportArrayButton = QPushButton("Export array")
		button_layout.addWidget(self.autoWindowButton)
		button_layout.addWidget(self.fullRangeButton)
		button_layout.addWidget(self.showWindowButton)
		button_layout.addWidget(self.savePngButton)
		button_layout.addWidget(self.exportArrayButton)
		layout.addWidget(button_row)
		layout.addStretch(1)

		self.windowCenterSpin.valueChanged.connect(self.on_display_changed)
		self.windowWidthSpin.valueChanged.connect(self.on_display_changed)
		self.gammaSpin.valueChanged.connect(self.on_display_changed)
		self.contrastSpin.valueChanged.connect(self.on_display_changed)
		self.robustPercentileSpin.valueChanged.connect(self.on_display_changed)
		self.onlyWindowRangeCheck.toggled.connect(self.on_display_changed)
		self.invertCheck.toggled.connect(self.on_display_changed)
		self.autoWindowButton.clicked.connect(self.on_auto_window)
		self.fullRangeButton.clicked.connect(self.on_full_range)
		self.showWindowButton.clicked.connect(self.on_show_window)
		self.savePngButton.clicked.connect(self.on_save_png)
		self.exportArrayButton.clicked.connect(self.on_export_array)

	@staticmethod
	def create(m, parent=0):
		"""Build the combined base-object and detector-image property panel."""
		return PropWidget.build([PropDetectorImage(m), PropBaseObject(m)], parent)

	def blockAll(self, blocked):
		"""Block or unblock signals on the interactive controls of this panel."""
		for widget in (
			self.windowCenterSpin,
			self.windowWidthSpin,
			self.gammaSpin,
			self.contrastSpin,
			self.robustPercentileSpin,
			self.onlyWindowRangeCheck,
			self.invertCheck,
		):
			widget.blockSignals(blocked)

	def updateProperties(self):
		"""Synchronize widget values with the current detector image object."""
		obj = self.obj_ref()
		if obj is None:
			return
		self.blockAll(True)
		stats = obj.data_stats()
		height, width = obj.shape_hw()
		center, window_width = obj.effective_window()
		self.infoLabel.setText(f"{width} x {height} px, stage={obj.source_stage}")
		self.windowLabel.setText(f"C={center:.6g}, W={window_width:.6g}")
		self.rangeLabel.setText(f"{stats['min']:.6g} .. {stats['max']:.6g}")
		self.windowCenterSpin.setValue(center)
		self.windowWidthSpin.setValue(window_width)
		self.gammaSpin.setValue(float(obj.display_gamma))
		self.contrastSpin.setValue(float(obj.display_contrast))
		self.robustPercentileSpin.setValue(float(obj.display_robust_percentile))
		self.onlyWindowRangeCheck.setChecked(bool(getattr(obj, "display_only_window_range", False)))
		self.invertCheck.setChecked(bool(obj.display_invert))
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
		obj.window_center = float(self.windowCenterSpin.value())
		obj.window_width = max(1e-6, float(self.windowWidthSpin.value()))
		obj.display_gamma = max(0.05, float(self.gammaSpin.value()))
		obj.display_contrast = max(0.05, float(self.contrastSpin.value()))
		obj.display_robust_percentile = min(100.0, max(50.0, float(self.robustPercentileSpin.value())))
		obj.display_only_window_range = bool(self.onlyWindowRangeCheck.isChecked())
		obj.display_invert = bool(self.invertCheck.isChecked())
		self._refresh_viewers(obj)
		AP.updateProperties()

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
	def on_export_array(self):
		"""Export the raw detector array as NumPy or text."""
		obj = self.obj_ref()
		if obj is None:
			return
		path, _ = QFileDialog.getSaveFileName(
			self,
			"Export detector array",
			f"{obj.label}.npy",
			"Detector array (*.npy *.txt *.csv *.tsv);;NumPy (*.npy);;Text (*.txt);;CSV (*.csv);;TSV (*.tsv)",
		)
		if not path:
			return
		try:
			obj.export_array(path)
		except Exception as exc:
			QMessageBox.critical(self, "Export detector array", str(exc))
