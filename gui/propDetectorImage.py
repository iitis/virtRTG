# -*- coding: utf-8 -*-
"""Property panel for the plugin-local `DetectorImage` scene object."""

from __future__ import annotations

import weakref

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
	QCheckBox,
	QColorDialog,
	QComboBox,
	QDialog,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMessageBox,
	QPushButton,
	QSizePolicy,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from dpVision import AP
from dpVision.gui.propBaseObject import PropBaseObject
from dpVision.gui.propWidget import PropWidget

from ..detectorImage import DetectorImage
from ..xray.xrayAnnotationOverlay import (
	XRayOverlayCross,
	XRayOverlayProjectionSet,
	XRayOverlayStyle,
)
from .detectorImageViewer import DetectorImageViewerChild
from .flowPanel import FlowPanelMixin
from .transferCurveDialog import TransferCurveDialog


class OverlayEditorDialog(QDialog):
	"""Simple editor for point overlays stored in one `DetectorImage` object."""

	def __init__(self, detector_image, on_overlay_changed=None, parent=None):
		"""Build the dialog and initialize it from detector-space point overlays."""
		super().__init__(parent)
		self.detector_image = detector_image
		self.on_overlay_changed = on_overlay_changed
		self._rebuilding = False
		self._point_items = []
		self._current_color = QColor(255, 0, 0, 255)
		self.setWindowTitle(f"Overlay Editor: {detector_image.label}")
		self.resize(360, 260)

		layout = QVBoxLayout(self)

		self.infoLabel = QLabel("Only point overlays are editable in this dialog.")
		self.infoLabel.setWordWrap(True)
		layout.addWidget(self.infoLabel)

		form = QFormLayout()
		self.pointCombo = QComboBox()
		self.labelEdit = QLineEdit()
		self.uSpin = QDoubleSpinBox()
		self.vSpin = QDoubleSpinBox()
		for spin in (self.uSpin, self.vSpin):
			spin.setRange(-1e6, 1e6)
			spin.setDecimals(3)
			spin.setSingleStep(1.0)
		self.colorButton = QPushButton("Select color")
		self.colorPreview = QLabel("")
		self.colorPreview.setFixedHeight(20)
		color_widget = QWidget()
		color_layout = QHBoxLayout(color_widget)
		color_layout.setContentsMargins(0, 0, 0, 0)
		color_layout.addWidget(self.colorButton)
		color_layout.addWidget(self.colorPreview)
		form.addRow("Point:", self.pointCombo)
		form.addRow("Label:", self.labelEdit)
		form.addRow("U [px]:", self.uSpin)
		form.addRow("V [px]:", self.vSpin)
		form.addRow("Color:", color_widget)
		layout.addLayout(form)

		button_row = QWidget()
		button_layout = QHBoxLayout(button_row)
		button_layout.setContentsMargins(0, 0, 0, 0)
		self.addButton = QPushButton("Add point")
		self.removeButton = QPushButton("Remove point")
		self.closeButton = QPushButton("Close")
		button_layout.addWidget(self.addButton)
		button_layout.addWidget(self.removeButton)
		button_layout.addStretch(1)
		button_layout.addWidget(self.closeButton)
		layout.addWidget(button_row)

		self.pointCombo.currentIndexChanged.connect(self._on_point_selected)
		self.labelEdit.textEdited.connect(self._on_value_changed)
		self.uSpin.valueChanged.connect(self._on_value_changed)
		self.vSpin.valueChanged.connect(self._on_value_changed)
		self.colorButton.clicked.connect(self._on_select_color)
		self.addButton.clicked.connect(self._on_add_point)
		self.removeButton.clicked.connect(self._on_remove_point)
		self.closeButton.clicked.connect(self.accept)
		self._rebuild()

	def _ensure_projection_set(self):
		"""Create an empty overlay set if the image does not have one yet."""
		if self.detector_image.overlay_projection_set is not None:
			return self.detector_image.overlay_projection_set
		height, width = self.detector_image.shape_hw()
		self.detector_image.overlay_projection_set = XRayOverlayProjectionSet(
			detector_shape_hw=(max(1, height), max(1, width)),
			items=[],
		)
		return self.detector_image.overlay_projection_set

	def _point_label(self, item_index, item):
		"""Return a compact user-facing label for one point overlay item."""
		label = str(getattr(item, "label", "")).strip()
		if label == "":
			label = f"Point {item_index + 1}"
		return f"{item_index + 1}: {label}"

	def _current_item(self):
		"""Return the currently selected point overlay item, or `None`."""
		index = int(self.pointCombo.currentIndex())
		if index < 0 or index >= len(self._point_items):
			return None
		return self._point_items[index]

	def _set_color_preview(self, color):
		"""Update the small swatch showing the current point color."""
		self._current_color = QColor(color)
		self.colorPreview.setStyleSheet(
			f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});"
			"border: 1px solid #666;"
		)

	def _rebuild(self):
		"""Refresh the point list and the editor fields from the detector image state."""
		self._rebuilding = True
		projection_set = self._ensure_projection_set()
		current_index = max(0, int(self.pointCombo.currentIndex()))
		self._point_items = [item for item in projection_set.items if isinstance(item, XRayOverlayCross)]
		self.pointCombo.clear()
		for item_index, item in enumerate(self._point_items):
			self.pointCombo.addItem(self._point_label(item_index, item))
		has_points = len(self._point_items) > 0
		self.removeButton.setEnabled(has_points)
		for widget in (self.pointCombo, self.labelEdit, self.uSpin, self.vSpin, self.colorButton):
			widget.setEnabled(has_points)
		if not has_points:
			self.labelEdit.setText("")
			self.uSpin.setValue(0.0)
			self.vSpin.setValue(0.0)
			self._set_color_preview(QColor(255, 0, 0, 255))
			self._rebuilding = False
			return
		self.pointCombo.setCurrentIndex(min(current_index, len(self._point_items) - 1))
		self._load_current_item()
		self._rebuilding = False

	def _load_current_item(self):
		"""Load the selected point overlay into the editor fields."""
		item = self._current_item()
		if item is None:
			return
		self.labelEdit.setText(str(getattr(item, "label", "")))
		pixel_uv = item.pixel_uv if item.pixel_uv is not None else (0.0, 0.0)
		self.uSpin.setValue(float(pixel_uv[0]))
		self.vSpin.setValue(float(pixel_uv[1]))
		style = item.style if item.style is not None else XRayOverlayStyle((255, 0, 0, 255))
		self._set_color_preview(QColor(*style.color_rgba))

	def _notify_change(self):
		"""Refresh dependent UI after editing overlays."""
		if callable(self.on_overlay_changed):
			self.on_overlay_changed()

	def _update_item_bounds(self, item):
		"""Update the item's in-bounds flag using the current detector dimensions."""
		height, width = self.detector_image.shape_hw()
		if item.pixel_uv is None:
			item.in_bounds = False
			return
		item.in_bounds = 0.0 <= float(item.pixel_uv[0]) <= max(width - 1, 0) and 0.0 <= float(item.pixel_uv[1]) <= max(height - 1, 0)

	@pyqtSlot()
	def _on_point_selected(self):
		"""Refresh editor fields when the user selects a different point."""
		if self._rebuilding:
			return
		self._load_current_item()

	@pyqtSlot()
	def _on_value_changed(self):
		"""Store current editor values in the selected point overlay item."""
		if self._rebuilding:
			return
		item = self._current_item()
		if item is None:
			return
		item.label = str(self.labelEdit.text())
		item.pixel_uv = (float(self.uSpin.value()), float(self.vSpin.value()))
		style = item.style if item.style is not None else XRayOverlayStyle((255, 0, 0, 255))
		style.color_rgba = (
			self._current_color.red(),
			self._current_color.green(),
			self._current_color.blue(),
			self._current_color.alpha(),
		)
		item.style = style
		self._update_item_bounds(item)
		current_index = self.pointCombo.currentIndex()
		self.pointCombo.setItemText(current_index, self._point_label(current_index, item))
		self._notify_change()

	@pyqtSlot()
	def _on_select_color(self):
		"""Open a color picker for the currently selected point overlay."""
		item = self._current_item()
		if item is None:
			return
		color = QColorDialog.getColor(
			self._current_color,
			self,
			"Select overlay color",
			options=QColorDialog.ShowAlphaChannel | QColorDialog.DontUseNativeDialog,
		)
		if not color.isValid():
			return
		self._set_color_preview(color)
		self._on_value_changed()

	@pyqtSlot()
	def _on_add_point(self):
		"""Append one new point overlay near the detector centre."""
		projection_set = self._ensure_projection_set()
		height, width = self.detector_image.shape_hw()
		new_item = XRayOverlayCross(
			kind="AnnotationPoint",
			label=f"Point {len(self._point_items) + 1}",
			pixel_uv=(max((width - 1) / 2.0, 0.0), max((height - 1) / 2.0, 0.0)),
			style=XRayOverlayStyle(
				color_rgba=(255, 0, 0, 255),
				line_width_px=1,
				marker_size_px=max(1, int(self.detector_image.presentation_overlay_cross_size_px)),
			),
			visible=True,
			in_bounds=True,
			metadata={"status": "manual"},
		)
		projection_set.items.append(new_item)
		self._rebuild()
		self.pointCombo.setCurrentIndex(len(self._point_items) - 1)
		self._notify_change()

	@pyqtSlot()
	def _on_remove_point(self):
		"""Remove the currently selected point overlay."""
		item = self._current_item()
		if item is None:
			return
		projection_set = self._ensure_projection_set()
		projection_set.items = [candidate for candidate in projection_set.items if candidate is not item]
		self._rebuild()
		self._notify_change()


class PropDetectorImage(FlowPanelMixin, PropWidget):
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
		layout.setAlignment(Qt.AlignTop)

		info_group, info_layout = self._create_flow_group("Detector Image")
		self.infoLabel = QLabel("")
		self.windowLabel = QLabel("")
		self.windowLabel.setWordWrap(True)
		self.rangeLabel = QLabel("")
		self.modeLabel = QLabel("")
		self.layerLabel = QLabel("")
		self.transferLabel = QLabel("")
		self.transferLabel.setWordWrap(True)
		self.transferLabel.setMaximumWidth(320)
		self._add_flow_control(info_layout, "Info", self.infoLabel)
		self._add_flow_control(info_layout, "Layer", self.layerLabel)
		self._add_flow_control(info_layout, "Mode", self.modeLabel)
		self._add_flow_control(info_layout, "Window", self.windowLabel)
		self._add_flow_control(info_layout, "Range", self.rangeLabel)
		self._add_flow_control(info_layout, "Curve", self.transferLabel)
		layout.addWidget(info_group)

		preview_group = QGroupBox("Preview")
		preview_layout = QVBoxLayout(preview_group)
		self.thumbnailLabel = QLabel("brak podglądu")
		self.thumbnailLabel.setAlignment(Qt.AlignCenter)
		self.thumbnailLabel.setMinimumHeight(140)
		preview_layout.addWidget(self.thumbnailLabel)
		layout.addWidget(preview_group)

		display_group, display_layout = self._create_flow_group("Display")
		self.layerCombo = QComboBox()
		self._set_compact_field(self.layerCombo)
		self.modeCombo = QComboBox()
		self.modeCombo.addItems(["raw", "digital", "film"])
		self._set_compact_field(self.modeCombo)
		self.windowCenterSpin = QDoubleSpinBox()
		self.windowCenterSpin.setRange(-1e12, 1e12)
		self.windowCenterSpin.setDecimals(6)
		self.windowCenterSpin.setSingleStep(1.0)
		self._set_compact_field(self.windowCenterSpin)
		self.windowWidthSpin = QDoubleSpinBox()
		self.windowWidthSpin.setRange(0.0, 1e12)
		self.windowWidthSpin.setDecimals(6)
		self.windowWidthSpin.setSingleStep(1.0)
		self._set_compact_field(self.windowWidthSpin)
		self.gammaSpin = QDoubleSpinBox()
		self.gammaSpin.setRange(0.05, 10.0)
		self.gammaSpin.setDecimals(3)
		self.gammaSpin.setSingleStep(0.05)
		self._set_compact_field(self.gammaSpin)
		self.contrastSpin = QDoubleSpinBox()
		self.contrastSpin.setRange(0.05, 10.0)
		self.contrastSpin.setDecimals(3)
		self.contrastSpin.setSingleStep(0.05)
		self._set_compact_field(self.contrastSpin)
		self.inputTransformCombo = QComboBox()
		self.inputTransformCombo.addItems(["linear", "log1p"])
		self._set_compact_field(self.inputTransformCombo)
		self.localEnhancementCombo = QComboBox()
		self.localEnhancementCombo.addItems(["off", "clahe"])
		self._set_compact_field(self.localEnhancementCombo)
		self.claheClipLimitSpin = QDoubleSpinBox()
		self.claheClipLimitSpin.setRange(0.01, 100.0)
		self.claheClipLimitSpin.setDecimals(3)
		self.claheClipLimitSpin.setSingleStep(0.1)
		self._set_compact_field(self.claheClipLimitSpin)
		self.claheTileGridSpin = QSpinBox()
		self.claheTileGridSpin.setRange(1, 64)
		self._set_compact_field(self.claheTileGridSpin)
		self.robustPercentileLowSpin = QDoubleSpinBox()
		self.robustPercentileLowSpin.setRange(0.0, 99.99)
		self.robustPercentileLowSpin.setDecimals(2)
		self.robustPercentileLowSpin.setSingleStep(0.1)
		self._set_compact_field(self.robustPercentileLowSpin)
		self.robustPercentileSpin = QDoubleSpinBox()
		self.robustPercentileSpin.setRange(50.0, 100.0)
		self.robustPercentileSpin.setDecimals(2)
		self.robustPercentileSpin.setSingleStep(0.1)
		self._set_compact_field(self.robustPercentileSpin)
		self.robustPercentileWidget = QWidget()
		self._set_compact_field(self.robustPercentileWidget)
		robust_percentile_layout = QHBoxLayout(self.robustPercentileWidget)
		robust_percentile_layout.setContentsMargins(0, 0, 0, 0)
		robust_percentile_layout.setSpacing(4)
		robust_percentile_layout.addWidget(QLabel("Low"))
		robust_percentile_layout.addWidget(self.robustPercentileLowSpin)
		robust_percentile_layout.addWidget(QLabel("High"))
		robust_percentile_layout.addWidget(self.robustPercentileSpin)
		self.overlayAnnotationsCheck = QCheckBox("Show projected annotations")
		self.overlayLabelsCheck = QCheckBox("Show annotation labels")
		self.overlayCrossSizeSpin = QSpinBox()
		self.overlayCrossSizeSpin.setRange(1, 256)
		self._set_compact_field(self.overlayCrossSizeSpin)
		self.onlyWindowRangeCheck = QCheckBox("Only window range")
		self.invertCheck = QCheckBox("Invert")
		self._add_flow_control(display_layout, "Layer", self.layerCombo)
		self._add_flow_control(display_layout, "Mode", self.modeCombo)
		self._add_flow_control(display_layout, "Window center", self.windowCenterSpin)
		self._add_flow_control(display_layout, "Window width", self.windowWidthSpin)
		self._add_flow_control(display_layout, "Gamma", self.gammaSpin)
		self._add_flow_control(display_layout, "Contrast", self.contrastSpin)
		self._add_flow_control(display_layout, "Input transform", self.inputTransformCombo)
		self._add_flow_control(display_layout, "Local enhancement", self.localEnhancementCombo)
		self._add_flow_control(display_layout, "CLAHE clip", self.claheClipLimitSpin)
		self._add_flow_control(display_layout, "CLAHE tile", self.claheTileGridSpin)
		self._add_flow_control(display_layout, "Robust [%]", self.robustPercentileWidget)
		self._add_flow_control(display_layout, "Cross size [px]", self.overlayCrossSizeSpin)
		self._add_flow_control(display_layout, "", self.overlayAnnotationsCheck)
		self._add_flow_control(display_layout, "", self.overlayLabelsCheck)
		self._add_flow_control(display_layout, "", self.onlyWindowRangeCheck)
		self._add_flow_control(display_layout, "", self.invertCheck)
		layout.addWidget(display_group)

		actions_group, actions_layout = self._create_flow_group("Actions")
		self.autoWindowButton = QPushButton("Auto window")
		self.fullRangeButton = QPushButton("Full range")
		self.showWindowButton = QPushButton("Show 2D")
		self.editCurveButton = QPushButton("Edit curve")
		self.editOverlaysButton = QPushButton("Edit overlays")
		self.savePngButton = QPushButton("Save PNG")
		self.importArrayButton = QPushButton("Import array")
		self.exportArrayButton = QPushButton("Export array")
		for button in (
			self.autoWindowButton,
			self.fullRangeButton,
			self.showWindowButton,
			self.editCurveButton,
			self.editOverlaysButton,
			self.savePngButton,
			self.importArrayButton,
			self.exportArrayButton,
		):
			button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self._add_flow_control(actions_layout, "", self.autoWindowButton)
		self._add_flow_control(actions_layout, "", self.fullRangeButton)
		self._add_flow_control(actions_layout, "", self.showWindowButton)
		self._add_flow_control(actions_layout, "", self.editCurveButton)
		self._add_flow_control(actions_layout, "", self.editOverlaysButton)
		self._add_flow_control(actions_layout, "", self.savePngButton)
		self._add_flow_control(actions_layout, "", self.importArrayButton)
		self._add_flow_control(actions_layout, "", self.exportArrayButton)
		layout.addWidget(actions_group)
		layout.addStretch(1)

		self.layerCombo.currentIndexChanged.connect(self.on_layer_changed)
		self.modeCombo.currentTextChanged.connect(self.on_display_changed)
		self.windowCenterSpin.valueChanged.connect(self.on_display_changed)
		self.windowWidthSpin.valueChanged.connect(self.on_display_changed)
		self.gammaSpin.valueChanged.connect(self.on_display_changed)
		self.contrastSpin.valueChanged.connect(self.on_display_changed)
		self.inputTransformCombo.currentTextChanged.connect(self.on_display_changed)
		self.localEnhancementCombo.currentTextChanged.connect(self.on_display_changed)
		self.claheClipLimitSpin.valueChanged.connect(self.on_display_changed)
		self.claheTileGridSpin.valueChanged.connect(self.on_display_changed)
		self.robustPercentileLowSpin.valueChanged.connect(self.on_display_changed)
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
		self.editOverlaysButton.clicked.connect(self.on_edit_overlays)
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
			self.layerCombo,
			self.modeCombo,
			self.windowCenterSpin,
			self.windowWidthSpin,
			self.gammaSpin,
			self.contrastSpin,
			self.inputTransformCombo,
			self.localEnhancementCombo,
			self.claheClipLimitSpin,
			self.claheTileGridSpin,
			self.robustPercentileLowSpin,
			self.robustPercentileSpin,
			self.overlayAnnotationsCheck,
			self.overlayLabelsCheck,
			self.overlayCrossSizeSpin,
			self.onlyWindowRangeCheck,
			self.invertCheck,
		):
			widget.blockSignals(blocked)

	def _configure_form_layout(self, layout):
		"""Keep form rows compact instead of stretching editor fields across the dock."""
		layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
		layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
		layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

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
		self.inputTransformCombo.setEnabled(not is_raw)
		self.localEnhancementCombo.setEnabled(not is_raw)
		clahe_enabled = (not is_raw) and str(getattr(obj, "presentation_local_enhancement", "off")).lower() == "clahe"
		self.claheClipLimitSpin.setEnabled(clahe_enabled)
		self.claheTileGridSpin.setEnabled(clahe_enabled)
		self.robustPercentileLowSpin.setEnabled(is_digital or is_film)
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
		active_layer = obj.active_layer_info()
		center, window_width = obj.effective_window()
		stored_center = 0.0 if obj.presentation_window_center is None else float(obj.presentation_window_center)
		stored_width = 0.0 if obj.presentation_window_width is None else float(obj.presentation_window_width)
		self.infoLabel.setText(f"{width} x {height} px, stage={obj.source_stage}")
		self.layerLabel.setText("-" if active_layer is None else str(active_layer.get("label", active_layer.get("key", "-"))))
		self.modeLabel.setText(str(obj.presentation_mode))
		if stored_width <= 0.0:
			self.windowLabel.setText(f"full range -> effective C={center:.6g}, W={window_width:.6g}")
		else:
			self.windowLabel.setText(f"C={center:.6g}, W={window_width:.6g}")
		self.rangeLabel.setText(f"{stats['min']:.6g} .. {stats['max']:.6g}")
		self.transferLabel.setText(obj.transfer_points_summary())
		self.layerCombo.clear()
		layer_choices = obj.package_layer_choices()
		if not layer_choices:
			layer_choices = [("current", "Current array")]
		for layer_key, layer_label in layer_choices:
			self.layerCombo.addItem(layer_label, layer_key)
		if obj.active_layer_key is not None:
			layer_index = self.layerCombo.findData(obj.active_layer_key)
			if layer_index >= 0:
				self.layerCombo.setCurrentIndex(layer_index)
		self.modeCombo.setCurrentText(str(obj.presentation_mode))
		self.windowCenterSpin.setValue(stored_center)
		self.windowWidthSpin.setValue(max(0.0, stored_width))
		self.gammaSpin.setValue(float(obj.presentation_gamma))
		self.contrastSpin.setValue(float(obj.presentation_contrast))
		self.inputTransformCombo.setCurrentText(str(getattr(obj, "presentation_input_transform", "linear")))
		self.localEnhancementCombo.setCurrentText(str(getattr(obj, "presentation_local_enhancement", "off")))
		self.claheClipLimitSpin.setValue(float(getattr(obj, "presentation_clahe_clip_limit", 2.0)))
		self.claheTileGridSpin.setValue(int(getattr(obj, "presentation_clahe_tile_grid_size", 8)))
		self.robustPercentileLowSpin.setValue(float(getattr(obj, "presentation_robust_low_percentile", 0.5)))
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

	@pyqtSlot()
	def on_layer_changed(self):
		"""Switch the active detector package layer when the user selects a different one."""
		obj = self.obj_ref()
		if obj is None:
			return
		layer_key = self.layerCombo.currentData()
		if layer_key in {None, "current"}:
			return
		try:
			obj.set_active_layer(str(layer_key), auto_window=False)
		except Exception as exc:
			QMessageBox.critical(self, "Switch detector layer", str(exc))
			return
		self.updateProperties()
		self._refresh_viewers(obj)
		AP.updateAllViews()

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
		obj.presentation_input_transform = str(self.inputTransformCombo.currentText()).lower()
		obj.presentation_local_enhancement = str(self.localEnhancementCombo.currentText()).lower()
		obj.presentation_clahe_clip_limit = max(0.01, float(self.claheClipLimitSpin.value()))
		obj.presentation_clahe_tile_grid_size = max(1, int(self.claheTileGridSpin.value()))
		obj.presentation_robust_low_percentile = min(
			99.999,
			max(0.0, float(self.robustPercentileLowSpin.value())),
		)
		obj.presentation_robust_percentile = min(100.0, max(50.0, float(self.robustPercentileSpin.value())))
		if obj.presentation_robust_percentile <= obj.presentation_robust_low_percentile:
			obj.presentation_robust_percentile = min(100.0, obj.presentation_robust_low_percentile + 0.01)
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
		obj.auto_window(
			robust_percentile=float(self.robustPercentileSpin.value()),
			robust_low_percentile=float(self.robustPercentileLowSpin.value()),
		)
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
	def on_edit_overlays(self):
		"""Open a dialog for editing detector overlay points."""
		obj = self.obj_ref()
		if obj is None:
			return
		dialog = OverlayEditorDialog(
			obj,
			on_overlay_changed=lambda: (self.updateProperties(), self._refresh_viewers(obj), AP.updateAllViews()),
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
