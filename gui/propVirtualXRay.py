# -*- coding: utf-8 -*-
"""Property panel for the `VirtualXRay` scene object."""

from __future__ import annotations

from time import perf_counter
import logging
_log = logging.getLogger(__name__)

import weakref

import numpy as np
from PyQt5.QtCore import Qt, QEventLoop, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
	QApplication,
	QCheckBox,
	QComboBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QSpinBox,
	QTabWidget,
	QVBoxLayout,
	QWidget,
	QDoubleSpinBox,
	QSizePolicy,
)

from dpVision import AP, Image, Mesh, Volumetric
from dpVision.gui.multiSpinBox import MultiSpinBox
from dpVision.gui.propBaseObject import PropBaseObject
from dpVision.gui.propWidget import PropWidget
from ..virtualXRay import VirtualXRay
from ..xray.xrayAnnotationOverlay import XRayOverlayCross, XRayOverlayPolyline
from ..xray.xraySource import get_xray_material_response_config, set_xray_material_response_config
from ..xray.xraySource import normalize_projection_to_uint8, ensure_xray_source_config

class _CollapsibleGroup(QWidget):
	"""Simple collapsible section: a toggle button + a hidden/shown body widget."""

	def __init__(self, title, collapsed=True, parent=None):
		"""Create one collapsible group with a body that participates in relayout."""
		super().__init__(parent)
		self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		outer = QVBoxLayout(self)
		outer.setContentsMargins(0, 2, 0, 2)
		outer.setSpacing(0)
		outer.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		self._title = title
		self._btn = QPushButton()
		self._btn.setCheckable(True)
		self._btn.setChecked(not collapsed)
		self._btn.setFlat(True)
		self._btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self._btn.setStyleSheet(
			"QPushButton { text-align: left; padding: 3px 6px; font-weight: bold; }"
		)
		self._update_text(not collapsed)
		self._btn.setMaximumWidth(self._btn.sizeHint().width())
		outer.addWidget(self._btn)

		self._body = QWidget()
		self._body.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		self._body.setVisible(not collapsed)
		outer.addWidget(self._body)

		self._btn.toggled.connect(self._on_toggle)

	def _update_text(self, expanded):
		self._btn.setText(("▼  " if expanded else "▶  ") + self._title)
		self._btn.setMaximumWidth(self._btn.sizeHint().width())

	def _on_toggle(self, checked):
		"""Show or hide the body and notify parent layouts about the new size hint."""
		self._update_text(checked)
		self._body.setVisible(checked)
		self.updateGeometry()
		parent = self.parentWidget()
		while parent is not None:
			parent.updateGeometry()
			parent = parent.parentWidget()

	def body(self):
		"""Return the inner widget to which a content layout should be assigned."""
		return self._body


class PropVirtualXRay(PropWidget):
	"""Edit basic source and detector parameters of one `VirtualXRay` scene node."""

	def __init__(self, _obj: VirtualXRay, parent=None):
		"""Build the property editor widgets and bind them to the provided scene object."""
		super().__init__(parent)
		self.obj_ref = weakref.ref(_obj)
		self._setup_ui()
		self._connect_signals()

	def _setup_ui(self):
		"""Create the full property form for detector, source and sampling parameters."""
		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		self.tabs = QTabWidget()
		self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		# layout.addWidget(self.tabs)

		# ── Geometry tab: Scene + Detector + Source + Sampling + Advanced source ──
		geomTab = QWidget()
		geomLayout = QVBoxLayout(geomTab)
		geomLayout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		sceneGroup = QGroupBox("Scene")
		self._set_compact_group(sceneGroup)
		scene_layout = QFormLayout(sceneGroup)
		self._configure_form_layout(scene_layout)
		self.volumesLabel = QLabel("-")
		self.geometryPresetWidget = QWidget()
		self._set_compact_field(self.geometryPresetWidget)
		geometry_preset_layout = QHBoxLayout(self.geometryPresetWidget)
		geometry_preset_layout.setContentsMargins(0, 0, 0, 0)
		geometry_preset_layout.setSpacing(4)
		self.geometryPresetCombo = QComboBox()
		self.geometryPresetCombo.addItems(VirtualXRay.geometry_preset_names())
		self._set_compact_field(self.geometryPresetCombo)
		self.applyGeometryPresetButton = QPushButton("Apply")
		self.applyGeometryPresetButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		geometry_preset_layout.addWidget(self.geometryPresetCombo)
		geometry_preset_layout.addWidget(self.applyGeometryPresetButton)
		self.modeCombo = QComboBox()
		self.modeCombo.addItems(["cone", "parallel"])
		self._set_compact_field(self.modeCombo)
		scene_layout.addRow("Sources:", self.volumesLabel)
		scene_layout.addRow("Preset:", self.geometryPresetWidget)
		scene_layout.addRow("Mode:", self.modeCombo)
		geomLayout.addWidget(sceneGroup)

		detectorGroup = _CollapsibleGroup("Detector", collapsed=True)
		detector_layout = QFormLayout(detectorGroup.body())
		self._configure_form_layout(detector_layout)
		self.detectorCenterSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.detectorNormalSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.detectorUpSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.detectorPixelSizeSpin = MultiSpinBox(2, labels=("U", "V"))
		for widget in (
			self.detectorCenterSpin,
			self.detectorNormalSpin,
			self.detectorUpSpin,
			self.detectorPixelSizeSpin,
		):
			self._set_compact_field(widget)
		self.detectorShapeWidget = QWidget()
		self._set_compact_field(self.detectorShapeWidget)
		detector_shape_layout = QHBoxLayout(self.detectorShapeWidget)
		detector_shape_layout.setContentsMargins(0, 0, 0, 0)
		detector_shape_layout.setSpacing(4)
		self.detectorHeightSpin = QSpinBox()
		self.detectorWidthSpin = QSpinBox()
		for spin in (self.detectorHeightSpin, self.detectorWidthSpin):
			spin.setRange(1, 8192)
			self._set_compact_field(spin)
		detector_shape_layout.addWidget(QLabel("H"))
		detector_shape_layout.addWidget(self.detectorHeightSpin)
		detector_shape_layout.addWidget(QLabel("W"))
		detector_shape_layout.addWidget(self.detectorWidthSpin)
		detector_layout.addRow("Center [mm]:", self.detectorCenterSpin)
		detector_layout.addRow("Normal:", self.detectorNormalSpin)
		detector_layout.addRow("Up:", self.detectorUpSpin)
		detector_layout.addRow("Pixel size [mm]:", self.detectorPixelSizeSpin)
		detector_layout.addRow("Shape [px]:", self.detectorShapeWidget)
		geomLayout.addWidget(detectorGroup)

		sourceGroup = _CollapsibleGroup("Source", collapsed=True)
		source_layout = QFormLayout(sourceGroup.body())
		self._configure_form_layout(source_layout)
		self.sourcePositionSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.rayDirectionSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self._set_compact_field(self.sourcePositionSpin)
		self._set_compact_field(self.rayDirectionSpin)
		source_layout.addRow("Position [mm]:", self.sourcePositionSpin)
		source_layout.addRow("Direction:", self.rayDirectionSpin)
		geomLayout.addWidget(sourceGroup)

		samplingGroup = QGroupBox("Sampling")
		self._set_compact_group(samplingGroup)
		sampling_layout = QFormLayout(samplingGroup)
		self._configure_form_layout(sampling_layout)
		self.stepSpin = QDoubleSpinBox()
		self.stepSpin.setRange(0.01, 50.0)
		self.stepSpin.setDecimals(3)
		self.stepSpin.setSingleStep(0.1)
		self._set_compact_field(self.stepSpin)
		self.qualityCombo = QComboBox()
		self.qualityCombo.addItems(["draft", "normal", "high", "custom"])
		self._set_compact_field(self.qualityCombo)
		sampling_layout.addRow("Step [mm]:", self.stepSpin)
		sampling_layout.addRow("Quality:", self.qualityCombo)
		geomLayout.addWidget(samplingGroup)

		depthWindowGroup = _CollapsibleGroup("Depth window", collapsed=True)
		depth_window_layout = QFormLayout(depthWindowGroup.body())
		self._configure_form_layout(depth_window_layout)
		self.depthWindowModeCombo = QComboBox()
		self.depthWindowModeCombo.addItems(["off", "ray", "planar_auto", "planar_custom"])
		self._set_compact_field(self.depthWindowModeCombo)
		self.depthWindowRangeSpin = MultiSpinBox(2, labels=("From", "To"))
		self._set_compact_field(self.depthWindowRangeSpin)
		self.depthWindowOriginSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.depthWindowAxisSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self._set_compact_field(self.depthWindowOriginSpin)
		self._set_compact_field(self.depthWindowAxisSpin)
		self.depthWindowAxisSpin.setToolTip("Custom planar mode uses this axis. Auto planar mode follows the current projection axis.")
		self.depthWindowToolsWidget = QWidget()
		self._set_compact_field(self.depthWindowToolsWidget)
		depth_tools_layout = QHBoxLayout(self.depthWindowToolsWidget)
		depth_tools_layout.setContentsMargins(0, 0, 0, 0)
		depth_tools_layout.setSpacing(4)
		self.depthAlignAxisButton = QPushButton("Align axis")
		self.depthAlignOriginButton = QPushButton("Origin = detector")
		self.depthAlignAxisButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self.depthAlignOriginButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		depth_tools_layout.addWidget(self.depthAlignAxisButton)
		depth_tools_layout.addWidget(self.depthAlignOriginButton)
		depth_window_layout.addRow("Mode:", self.depthWindowModeCombo)
		depth_window_layout.addRow("Range [mm]:", self.depthWindowRangeSpin)
		depth_window_layout.addRow("Origin [mm]:", self.depthWindowOriginSpin)
		depth_window_layout.addRow("Axis:", self.depthWindowAxisSpin)
		depth_window_layout.addRow("", self.depthWindowToolsWidget)
		geomLayout.addWidget(depthWindowGroup)

		self.geometryAdvancedCheck = QCheckBox("Show advanced")
		geomLayout.addWidget(self.geometryAdvancedCheck)

		self.advancedSourceGroup = QGroupBox("Advanced")
		self._set_compact_group(self.advancedSourceGroup)
		advanced_source_layout = QFormLayout(self.advancedSourceGroup)
		self._configure_form_layout(advanced_source_layout)
		self.sourceInterpolationCombo = QComboBox()
		self.sourceInterpolationCombo.addItems(["nearest", "linear", "cubic"])
		self._set_compact_field(self.sourceInterpolationCombo)
		self.sourcePreprocessModeCombo = QComboBox()
		self.sourcePreprocessModeCombo.addItems(["none", "percentile_rescale"])
		self._set_compact_field(self.sourcePreprocessModeCombo)
		self.sourcePreprocessLowPercentileSpin = QDoubleSpinBox()
		self.sourcePreprocessLowPercentileSpin.setRange(0.0, 100.0)
		self.sourcePreprocessLowPercentileSpin.setDecimals(2)
		self.sourcePreprocessLowPercentileSpin.setSingleStep(0.1)
		self._set_compact_field(self.sourcePreprocessLowPercentileSpin)
		self.sourcePreprocessHighPercentileSpin = QDoubleSpinBox()
		self.sourcePreprocessHighPercentileSpin.setRange(0.0, 100.0)
		self.sourcePreprocessHighPercentileSpin.setDecimals(2)
		self.sourcePreprocessHighPercentileSpin.setSingleStep(0.1)
		self._set_compact_field(self.sourcePreprocessHighPercentileSpin)
		self.sourcePreprocessOutputLowSpin = QDoubleSpinBox()
		self.sourcePreprocessOutputLowSpin.setRange(-1e6, 1e6)
		self.sourcePreprocessOutputLowSpin.setDecimals(3)
		self.sourcePreprocessOutputLowSpin.setSingleStep(1.0)
		self._set_compact_field(self.sourcePreprocessOutputLowSpin)
		self.sourcePreprocessOutputHighSpin = QDoubleSpinBox()
		self.sourcePreprocessOutputHighSpin.setRange(-1e6, 1e6)
		self.sourcePreprocessOutputHighSpin.setDecimals(3)
		self.sourcePreprocessOutputHighSpin.setSingleStep(1.0)
		self._set_compact_field(self.sourcePreprocessOutputHighSpin)
		self.sourceUseFillValueCheck = QCheckBox("Use explicit fill value")
		self.sourceFillValueSpin = QDoubleSpinBox()
		self.sourceFillValueSpin.setRange(-1e9, 1e9)
		self.sourceFillValueSpin.setDecimals(3)
		self.sourceFillValueSpin.setSingleStep(1.0)
		self._set_compact_field(self.sourceFillValueSpin)
		advanced_source_layout.addRow("Interpolation:", self.sourceInterpolationCombo)
		advanced_source_layout.addRow("Preprocess:", self.sourcePreprocessModeCombo)
		advanced_source_layout.addRow("Input low [%]:", self.sourcePreprocessLowPercentileSpin)
		advanced_source_layout.addRow("Input high [%]:", self.sourcePreprocessHighPercentileSpin)
		advanced_source_layout.addRow("Output low:", self.sourcePreprocessOutputLowSpin)
		advanced_source_layout.addRow("Output high:", self.sourcePreprocessOutputHighSpin)
		advanced_source_layout.addRow("", self.sourceUseFillValueCheck)
		advanced_source_layout.addRow("Fill value:", self.sourceFillValueSpin)
		self.advancedSourceGroup.setVisible(False)
		geomLayout.addWidget(self.advancedSourceGroup)

		geomLayout.addStretch(1)
		self.tabs.addTab(geomTab, "Geometry")

		# ── Physics tab: Material filter + Advanced physics ────────────
		physTab = QWidget()
		physLayout = QVBoxLayout(physTab)
		physLayout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		physicsGroup = QGroupBox("Material filter")
		self._set_compact_group(physicsGroup)
		physics_layout = QFormLayout(physicsGroup)
		self._configure_form_layout(physics_layout)
		self.physicsMaterialWindowCenterSpin = QDoubleSpinBox()
		self.physicsMaterialWindowCenterSpin.setRange(-1e6, 1e6)
		self.physicsMaterialWindowCenterSpin.setDecimals(3)
		self.physicsMaterialWindowCenterSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsMaterialWindowCenterSpin)
		self.physicsMaterialResponseModeCombo = QComboBox()
		self.physicsMaterialResponseModeCombo.addItems(["linear", "piecewise_bone", "piecewise_soft_tissue", "bone_threshold"])
		self._set_compact_field(self.physicsMaterialResponseModeCombo)
		self.physicsBoneThresholdSpin = QDoubleSpinBox()
		self.physicsBoneThresholdSpin.setRange(-1e6, 1e6)
		self.physicsBoneThresholdSpin.setDecimals(3)
		self.physicsBoneThresholdSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsBoneThresholdSpin)
		self.physicsBoneThresholdSoftnessSpin = QDoubleSpinBox()
		self.physicsBoneThresholdSoftnessSpin.setRange(0.0, 1e6)
		self.physicsBoneThresholdSoftnessSpin.setDecimals(3)
		self.physicsBoneThresholdSoftnessSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsBoneThresholdSoftnessSpin)
		self.physicsMaterialWindowWidthSpin = QDoubleSpinBox()
		self.physicsMaterialWindowWidthSpin.setRange(0.0, 1e6)
		self.physicsMaterialWindowWidthSpin.setDecimals(3)
		self.physicsMaterialWindowWidthSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsMaterialWindowWidthSpin)
		self.physicsMaterialWindowModeCombo = QComboBox()
		self.physicsMaterialWindowModeCombo.addItems(["hard", "linear", "sigmoid"])
		self._set_compact_field(self.physicsMaterialWindowModeCombo)
		self.physicsMaterialWindowSoftnessSpin = QDoubleSpinBox()
		self.physicsMaterialWindowSoftnessSpin.setRange(0.0, 1e6)
		self.physicsMaterialWindowSoftnessSpin.setDecimals(3)
		self.physicsMaterialWindowSoftnessSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsMaterialWindowSoftnessSpin)
		self.physicsAutoBoneButton = QPushButton("Auto bone threshold")
		self.physicsAutoBoneButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		physics_layout.addRow("Response:", self.physicsMaterialResponseModeCombo)
		physics_layout.addRow("Bone threshold [HU]:", self.physicsBoneThresholdSpin)
		physics_layout.addRow("Bone softness [HU]:", self.physicsBoneThresholdSoftnessSpin)
		physics_layout.addRow("Center [HU]:", self.physicsMaterialWindowCenterSpin)
		physics_layout.addRow("Width [HU]:", self.physicsMaterialWindowWidthSpin)
		physics_layout.addRow("Mode:", self.physicsMaterialWindowModeCombo)
		physics_layout.addRow("Softness [HU]:", self.physicsMaterialWindowSoftnessSpin)
		physics_layout.addRow("", self.physicsAutoBoneButton)
		physLayout.addWidget(physicsGroup)

		self.physicsAdvancedCheck = QCheckBox("Show advanced")
		physLayout.addWidget(self.physicsAdvancedCheck)

		self.advancedPhysicsGroup = QGroupBox("Advanced")
		self._set_compact_group(self.advancedPhysicsGroup)
		advanced_physics_layout = QFormLayout(self.advancedPhysicsGroup)
		self._configure_form_layout(advanced_physics_layout)
		self.physicsMuAirSpin = QDoubleSpinBox()
		self.physicsMuAirSpin.setRange(-1e6, 1e6)
		self.physicsMuAirSpin.setDecimals(6)
		self.physicsMuAirSpin.setSingleStep(0.001)
		self._set_compact_field(self.physicsMuAirSpin)
		self.physicsMuWaterSpin = QDoubleSpinBox()
		self.physicsMuWaterSpin.setRange(-1e6, 1e6)
		self.physicsMuWaterSpin.setDecimals(6)
		self.physicsMuWaterSpin.setSingleStep(0.001)
		self._set_compact_field(self.physicsMuWaterSpin)
		self.physicsHounsfieldAirSpin = QDoubleSpinBox()
		self.physicsHounsfieldAirSpin.setRange(-1e6, 1e6)
		self.physicsHounsfieldAirSpin.setDecimals(3)
		self.physicsHounsfieldAirSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsHounsfieldAirSpin)
		self.physicsAttenuationScaleSpin = QDoubleSpinBox()
		self.physicsAttenuationScaleSpin.setRange(0.0, 1e6)
		self.physicsAttenuationScaleSpin.setDecimals(6)
		self.physicsAttenuationScaleSpin.setSingleStep(0.01)
		self._set_compact_field(self.physicsAttenuationScaleSpin)
		self.physicsSourceEnergySpin = QDoubleSpinBox()
		self.physicsSourceEnergySpin.setRange(1.0, 1000.0)
		self.physicsSourceEnergySpin.setDecimals(3)
		self.physicsSourceEnergySpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsSourceEnergySpin)
		self.physicsReferenceEnergySpin = QDoubleSpinBox()
		self.physicsReferenceEnergySpin.setRange(1.0, 1000.0)
		self.physicsReferenceEnergySpin.setDecimals(3)
		self.physicsReferenceEnergySpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsReferenceEnergySpin)
		self.physicsEnergyExponentSpin = QDoubleSpinBox()
		self.physicsEnergyExponentSpin.setRange(0.0, 10.0)
		self.physicsEnergyExponentSpin.setDecimals(3)
		self.physicsEnergyExponentSpin.setSingleStep(0.1)
		self._set_compact_field(self.physicsEnergyExponentSpin)
		self.physicsOutputModeCombo = QComboBox()
		self.physicsOutputModeCombo.addItems(["integral", "intensity"])
		self._set_compact_field(self.physicsOutputModeCombo)
		self.physicsIntensityFloorSpin = QDoubleSpinBox()
		self.physicsIntensityFloorSpin.setRange(0.0, 1e6)
		self.physicsIntensityFloorSpin.setDecimals(6)
		self.physicsIntensityFloorSpin.setSingleStep(0.001)
		self._set_compact_field(self.physicsIntensityFloorSpin)
		self.physicsDistanceFalloffModeCombo = QComboBox()
		self.physicsDistanceFalloffModeCombo.addItems(["none", "inverse_square"])
		self._set_compact_field(self.physicsDistanceFalloffModeCombo)
		self.physicsDistanceReferenceSpin = QDoubleSpinBox()
		self.physicsDistanceReferenceSpin.setRange(0.0, 1e6)
		self.physicsDistanceReferenceSpin.setDecimals(3)
		self.physicsDistanceReferenceSpin.setSingleStep(1.0)
		self._set_compact_field(self.physicsDistanceReferenceSpin)
		self.physicsDistancePowerSpin = QDoubleSpinBox()
		self.physicsDistancePowerSpin.setRange(0.0, 10.0)
		self.physicsDistancePowerSpin.setDecimals(3)
		self.physicsDistancePowerSpin.setSingleStep(0.1)
		self._set_compact_field(self.physicsDistancePowerSpin)
		advanced_physics_layout.addRow("mu_air:", self.physicsMuAirSpin)
		advanced_physics_layout.addRow("mu_water:", self.physicsMuWaterSpin)
		advanced_physics_layout.addRow("hounsfield_air:", self.physicsHounsfieldAirSpin)
		advanced_physics_layout.addRow("attenuation_scale:", self.physicsAttenuationScaleSpin)
		advanced_physics_layout.addRow("source_energy_kev:", self.physicsSourceEnergySpin)
		advanced_physics_layout.addRow("reference_energy_kev:", self.physicsReferenceEnergySpin)
		advanced_physics_layout.addRow("energy_exponent:", self.physicsEnergyExponentSpin)
		advanced_physics_layout.addRow("output_mode:", self.physicsOutputModeCombo)
		advanced_physics_layout.addRow("intensity_floor:", self.physicsIntensityFloorSpin)
		advanced_physics_layout.addRow("distance_falloff:", self.physicsDistanceFalloffModeCombo)
		advanced_physics_layout.addRow("distance_ref [mm]:", self.physicsDistanceReferenceSpin)
		advanced_physics_layout.addRow("distance_power:", self.physicsDistancePowerSpin)
		self.advancedPhysicsGroup.setVisible(False)
		physLayout.addWidget(self.advancedPhysicsGroup)

		physLayout.addStretch(1)
		self.tabs.addTab(physTab, "Physics")

		# ── Presentation tab ───────────────────────────────────────────
		presentationTab = QWidget()
		presentationTabLayout = QVBoxLayout(presentationTab)
		presentationTabLayout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		presentationGroup = QGroupBox("Presentation")
		self._set_compact_group(presentationGroup)
		presentation_layout = QFormLayout(presentationGroup)
		self._configure_form_layout(presentation_layout)
		self.presentationPresetWidget = QWidget()
		self._set_compact_field(self.presentationPresetWidget)
		presentation_preset_layout = QHBoxLayout(self.presentationPresetWidget)
		presentation_preset_layout.setContentsMargins(0, 0, 0, 0)
		presentation_preset_layout.setSpacing(4)
		self.presentationPresetCombo = QComboBox()
		self.presentationPresetCombo.addItems(VirtualXRay.presentation_preset_names())
		self._set_compact_field(self.presentationPresetCombo)
		self.applyPresentationPresetButton = QPushButton("Apply")
		self.applyPresentationPresetButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		presentation_preset_layout.addWidget(self.presentationPresetCombo)
		presentation_preset_layout.addWidget(self.applyPresentationPresetButton)
		self.presentationModeCombo = QComboBox()
		self.presentationModeCombo.addItems(["digital", "film", "raw"])
		self._set_compact_field(self.presentationModeCombo)
		self.presentationInvertCheck = QCheckBox("Invert")
		self.presentationGammaSpin = QDoubleSpinBox()
		self.presentationGammaSpin.setRange(0.05, 10.0)
		self.presentationGammaSpin.setDecimals(3)
		self.presentationGammaSpin.setSingleStep(0.05)
		self._set_compact_field(self.presentationGammaSpin)
		self.presentationContrastSpin = QDoubleSpinBox()
		self.presentationContrastSpin.setRange(0.05, 10.0)
		self.presentationContrastSpin.setDecimals(3)
		self.presentationContrastSpin.setSingleStep(0.05)
		self._set_compact_field(self.presentationContrastSpin)
		self.presentationPercentileSpin = QDoubleSpinBox()
		self.presentationPercentileSpin.setRange(50.0, 100.0)
		self.presentationPercentileSpin.setDecimals(2)
		self.presentationPercentileSpin.setSingleStep(0.1)
		self._set_compact_field(self.presentationPercentileSpin)
		self.presentationWindowCenterSpin = QDoubleSpinBox()
		self.presentationWindowCenterSpin.setRange(-1e6, 1e6)
		self.presentationWindowCenterSpin.setDecimals(3)
		self.presentationWindowCenterSpin.setSingleStep(0.1)
		self._set_compact_field(self.presentationWindowCenterSpin)
		self.presentationWindowWidthSpin = QDoubleSpinBox()
		self.presentationWindowWidthSpin.setRange(0.0, 1e6)
		self.presentationWindowWidthSpin.setDecimals(3)
		self.presentationWindowWidthSpin.setSingleStep(0.1)
		self._set_compact_field(self.presentationWindowWidthSpin)
		self.presentationOverlayAnnotationsCheck = QCheckBox("Overlay projected annotations")
		self.presentationOverlayLabelsCheck = QCheckBox("Show labels")
		self.presentationOverlayCrossSizeSpin = QSpinBox()
		self.presentationOverlayCrossSizeSpin.setRange(1, 64)
		self._set_compact_field(self.presentationOverlayCrossSizeSpin)
		presentation_layout.addRow("Preset:", self.presentationPresetWidget)
		presentation_layout.addRow("Mode:", self.presentationModeCombo)
		presentation_layout.addRow("", self.presentationInvertCheck)
		presentation_layout.addRow("Gamma:", self.presentationGammaSpin)
		presentation_layout.addRow("Contrast:", self.presentationContrastSpin)
		presentation_layout.addRow("Robust [%]:", self.presentationPercentileSpin)
		presentation_layout.addRow("Window center:", self.presentationWindowCenterSpin)
		presentation_layout.addRow("Window width:", self.presentationWindowWidthSpin)
		presentation_layout.addRow("", self.presentationOverlayAnnotationsCheck)
		presentation_layout.addRow("", self.presentationOverlayLabelsCheck)
		presentation_layout.addRow("Cross size [px]:", self.presentationOverlayCrossSizeSpin)
		presentationTabLayout.addWidget(presentationGroup)
		presentationTabLayout.addStretch(1)
		self.tabs.addTab(presentationTab, "Presentation")

		# ── Run tab ────────────────────────────────────────────────────
		runTab = QWidget()
		runTabLayout = QVBoxLayout(runTab)
		runTabLayout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		actionsWidget = QWidget()
		actionsWidget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		actions_layout = QHBoxLayout(actionsWidget)
		actions_layout.setContentsMargins(0, 0, 0, 0)
		actions_layout.setSpacing(4)
		self.refreshButton = QPushButton("Refresh")
		self.runSimulationButton = QPushButton("Run Simulation")
		self.updateDisplayButton = QPushButton("Update display")
		self.updateDisplayButton.setEnabled(False)
		self.renderInfoLabel = QLabel("")
		self.renderInfoLabel.setWordWrap(True)
		self.renderInfoLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
		actions_layout.addWidget(self.runSimulationButton)
		actions_layout.addWidget(self.updateDisplayButton)
		actions_layout.addWidget(self.refreshButton)
		runTabLayout.addWidget(actionsWidget)
		self.progressBar = QProgressBar()
		self.progressBar.setRange(0, 100)
		self.progressBar.setValue(0)
		self.progressBar.setVisible(False)
		runTabLayout.addWidget(self.progressBar)
		runTabLayout.addWidget(self.renderInfoLabel)
		runTabLayout.addStretch(1)
		# self.tabs.addTab(runTab, "Run")

		layout.addWidget(runTab)
		layout.addWidget(self.tabs)

		self._build_sources_tab()

		layout.addStretch(1)

	# ── Sources tab ──────────────────────────────────────────────────────────

	def _build_sources_tab(self):
		"""Build the Sources tab container — content is filled dynamically on each refresh."""
		from PyQt5.QtWidgets import QStackedWidget
		sourcesTab = QWidget()
		sourcesTabLayout = QVBoxLayout(sourcesTab)
		sourcesTabLayout.setContentsMargins(4, 4, 4, 4)
		sourcesTabLayout.setSpacing(4)
		sourcesTabLayout.setAlignment(Qt.AlignTop)

		self._sourcesCombo = QComboBox()
		self._sourcesCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		sourcesTabLayout.addWidget(self._sourcesCombo)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self._sourcesStack = QStackedWidget()
		scroll.setWidget(self._sourcesStack)
		sourcesTabLayout.addWidget(scroll)

		self._sourcesCombo.currentIndexChanged.connect(self._sourcesStack.setCurrentIndex)

		self._sources_tab_index = self.tabs.insertTab(
			# self.tabs.count() - 1, sourcesTab, "Sources"
			0, sourcesTab, "Sources"
		)

	def _rebuild_sources_tab(self, obj: VirtualXRay):
		"""Rebuild per-source controls to match the current scene tree."""
		from PyQt5.QtWidgets import QStackedWidget
		all_sources = obj.collect_volumetrics() + obj.collect_meshes()

		# Build new labels list to compare with current combo state.
		new_labels = [
			f"[{'Vol' if isinstance(s, Volumetric) else 'Mesh'}]  {s.label}"
			for s in all_sources
		]
		old_labels = [self._sourcesCombo.itemText(i) for i in range(self._sourcesCombo.count())]

		# Preserve selected index across refreshes when the source list didn't change.
		prev_index = self._sourcesCombo.currentIndex()

		# Always rebuild to pick up any property value changes on the source objects.
		self._sourcesCombo.blockSignals(True)

		# Remove old stack pages.
		while self._sourcesStack.count():
			w = self._sourcesStack.widget(0)
			self._sourcesStack.removeWidget(w)
			w.deleteLater()
		self._sourcesCombo.clear()

		if not all_sources:
			placeholder = QWidget()
			pl_layout = QVBoxLayout(placeholder)
			lbl = QLabel("No Mesh or Volumetric objects in this VirtualXRay subtree.")
			lbl.setWordWrap(True)
			pl_layout.addWidget(lbl)
			self._sourcesStack.addWidget(placeholder)
			self._sourcesCombo.addItem("—")
			self._sourcesCombo.blockSignals(False)
			return

		for label, src in zip(new_labels, all_sources):
			self._sourcesCombo.addItem(label)
			self._sourcesStack.addWidget(self._make_source_group(src))

		# Restore previous selection when possible.
		if old_labels == new_labels and 0 <= prev_index < len(all_sources):
			self._sourcesCombo.setCurrentIndex(prev_index)
			self._sourcesStack.setCurrentIndex(prev_index)
		else:
			self._sourcesCombo.setCurrentIndex(0)
			self._sourcesStack.setCurrentIndex(0)

		self._sourcesCombo.blockSignals(False)

	def _make_source_group(self, source_obj):
		"""Return a QGroupBox with xray controls for one Mesh or Volumetric source object."""
		src_ref = weakref.ref(source_obj)
		src_type = "Vol" if isinstance(source_obj, Volumetric) else "Mesh"
		material_config = get_xray_material_response_config(source_obj)
		group = QGroupBox(f"[{src_type}]  {source_obj.label}")
		group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		layout = QVBoxLayout(group)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.setSpacing(6)

		def _section(title):
			box = QGroupBox(title)
			self._set_compact_group(box)
			form = QFormLayout(box)
			self._configure_form_layout(form)
			form.setContentsMargins(6, 6, 6, 6)
			layout.addWidget(box)
			return form

		source_form = _section("Source")
		material_form = _section("Material Response")
		type_form = _section("Volume Sampling" if isinstance(source_obj, Volumetric) else "Mesh Model")

		enabled_check = QCheckBox("Enabled")
		enabled_check.setChecked(bool(source_obj.xray_source_enabled))
		source_form.addRow("", enabled_check)

		scale_spin = QDoubleSpinBox()
		scale_spin.setRange(-1e3, 1e3)
		scale_spin.setDecimals(6)
		scale_spin.setSingleStep(0.05)
		scale_spin.setValue(float(source_obj.xray_scalar_scale))
		source_form.addRow("Scalar scale:", scale_spin)

		bias_spin = QDoubleSpinBox()
		bias_spin.setRange(-1e6, 1e6)
		bias_spin.setDecimals(3)
		bias_spin.setSingleStep(10.0)
		bias_spin.setValue(float(source_obj.xray_scalar_bias))
		source_form.addRow("Scalar bias:", bias_spin)

		atten_spin = QDoubleSpinBox()
		atten_spin.setRange(0.0, 1e6)
		atten_spin.setDecimals(6)
		atten_spin.setSingleStep(0.05)
		atten_spin.setValue(float(source_obj.xray_attenuation_multiplier))
		source_form.addRow("Attenuation x:", atten_spin)

		material_override_check = QCheckBox("Use individual material response")
		material_override_check.setChecked(bool(material_config.enabled))
		material_form.addRow("", material_override_check)

		material_response_combo = QComboBox()
		material_response_combo.addItems(["linear", "piecewise_bone", "piecewise_soft_tissue", "bone_threshold"])
		material_response_combo.setCurrentText(str(material_config.mode))
		material_form.addRow("Response:", material_response_combo)

		material_threshold_spin = QDoubleSpinBox()
		material_threshold_spin.setRange(-1e6, 1e6)
		material_threshold_spin.setDecimals(3)
		material_threshold_spin.setSingleStep(10.0)
		material_threshold_spin.setValue(
			0.0 if material_config.bone_threshold_hu is None
			else float(material_config.bone_threshold_hu)
		)
		material_form.addRow("Bone threshold:", material_threshold_spin)

		material_softness_spin = QDoubleSpinBox()
		material_softness_spin.setRange(0.0, 1e6)
		material_softness_spin.setDecimals(3)
		material_softness_spin.setSingleStep(10.0)
		material_softness_spin.setValue(float(material_config.bone_threshold_softness))
		material_form.addRow("Threshold soft.:", material_softness_spin)

		material_window_center_spin = QDoubleSpinBox()
		material_window_center_spin.setRange(-1e6, 1e6)
		material_window_center_spin.setDecimals(3)
		material_window_center_spin.setSingleStep(10.0)
		material_window_center_spin.setValue(
			0.0 if material_config.window_center is None
			else float(material_config.window_center)
		)
		material_form.addRow("Window center:", material_window_center_spin)

		material_window_width_spin = QDoubleSpinBox()
		material_window_width_spin.setRange(0.0, 1e6)
		material_window_width_spin.setDecimals(3)
		material_window_width_spin.setSingleStep(10.0)
		material_window_width_spin.setValue(
			0.0 if material_config.window_width is None
			else float(material_config.window_width)
		)
		material_form.addRow("Window width:", material_window_width_spin)

		material_window_mode_combo = QComboBox()
		material_window_mode_combo.addItems(["hard", "linear", "sigmoid"])
		material_window_mode_combo.setCurrentText(str(material_config.window_mode))
		material_form.addRow("Window mode:", material_window_mode_combo)

		material_window_softness_spin = QDoubleSpinBox()
		material_window_softness_spin.setRange(0.0, 1e6)
		material_window_softness_spin.setDecimals(3)
		material_window_softness_spin.setSingleStep(10.0)
		material_window_softness_spin.setValue(float(material_config.window_softness))
		material_form.addRow("Window soft.:", material_window_softness_spin)

		def _update_material_override_widgets():
			override_enabled = bool(material_override_check.isChecked())
			uses_bone_threshold = str(material_response_combo.currentText()).lower() == "bone_threshold"
			width_enabled = float(material_window_width_spin.value()) > 0.0
			material_response_combo.setEnabled(override_enabled)
			material_threshold_spin.setEnabled(override_enabled and uses_bone_threshold)
			material_softness_spin.setEnabled(override_enabled and uses_bone_threshold)
			material_window_center_spin.setEnabled(override_enabled)
			material_window_width_spin.setEnabled(override_enabled)
			material_window_mode_combo.setEnabled(override_enabled and width_enabled)
			window_mode = str(material_window_mode_combo.currentText()).lower()
			material_window_softness_spin.setEnabled(
				override_enabled and width_enabled and window_mode in {"linear", "sigmoid"}
			)

		_update_material_override_widgets()

		if isinstance(source_obj, Volumetric):
			interp_combo = QComboBox()
			interp_combo.addItems(["default", "nearest", "linear", "cubic"])
			interp_combo.setCurrentText(str(source_obj.xray_interpolation_override))
			type_form.addRow("Interpolation:", interp_combo)

			backend_combo = QComboBox()
			backend_combo.addItems(["sampling", "siddon"])
			backend_combo.setToolTip(
				"sampling – uniform ray-marching (step_mm)\n"
				"siddon  – exact voxel traversal (chord-length, step_mm independent)"
			)
			backend_combo.setCurrentText(str(getattr(source_obj, "xray_volume_backend", "sampling")))
			type_form.addRow("Backend:", backend_combo)

			fill_check = QCheckBox("Use explicit fill value")
			fill_check.setChecked(bool(source_obj.xray_fill_value_override_enabled))
			type_form.addRow("", fill_check)

			fill_spin = QDoubleSpinBox()
			fill_spin.setRange(-1e9, 1e9)
			fill_spin.setDecimals(3)
			fill_spin.setSingleStep(10.0)
			fill_spin.setValue(float(source_obj.xray_fill_value_override))
			fill_spin.setEnabled(bool(source_obj.xray_fill_value_override_enabled))
			type_form.addRow("Fill value:", fill_spin)

			def _on_vol_changed(
				_ref=src_ref, _en=enabled_check, _sc=scale_spin, _bi=bias_spin,
				_at=atten_spin, _mo=material_override_check, _mm=material_response_combo,
				_mt=material_threshold_spin, _ms=material_softness_spin,
				_mwc=material_window_center_spin, _mww=material_window_width_spin,
				_mwm=material_window_mode_combo, _mws=material_window_softness_spin,
				_ic=interp_combo, _bc=backend_combo, _fc=fill_check, _fs=fill_spin
			):
				s = _ref()
				if s is None:
					return
				s.xray_source_enabled = bool(_en.isChecked())
				s.xray_scalar_scale = float(_sc.value())
				s.xray_scalar_bias = float(_bi.value())
				s.xray_attenuation_multiplier = max(0.0, float(_at.value()))
				window_width = max(0.0, float(_mww.value()))
				material_cfg = get_xray_material_response_config(s)
				material_cfg.enabled = bool(_mo.isChecked())
				material_cfg.mode = str(_mm.currentText()).lower()
				material_cfg.bone_threshold_hu = (
					float(_mt.value()) if material_cfg.mode == "bone_threshold" else None
				)
				material_cfg.bone_threshold_softness = max(0.0, float(_ms.value()))
				material_cfg.window_center = float(_mwc.value()) if window_width > 0.0 else None
				material_cfg.window_width = window_width if window_width > 0.0 else None
				material_cfg.window_mode = str(_mwm.currentText()).lower()
				material_cfg.window_softness = max(0.0, float(_mws.value()))
				set_xray_material_response_config(s, material_cfg)
				s.xray_interpolation_override = str(_ic.currentText()).lower()
				s.xray_volume_backend = str(_bc.currentText()).lower()
				s.xray_fill_value_override_enabled = bool(_fc.isChecked())
				s.xray_fill_value_override = float(_fs.value())
				_update_material_override_widgets()
				_fs.setEnabled(s.xray_fill_value_override_enabled)
				AP.updateAllViews()

			for w in (
				enabled_check, scale_spin, bias_spin, atten_spin,
				material_override_check, material_response_combo, material_threshold_spin,
				material_softness_spin, material_window_center_spin, material_window_width_spin,
				material_window_mode_combo, material_window_softness_spin,
				interp_combo, backend_combo, fill_check, fill_spin
			):
				if hasattr(w, "toggled"):
					w.toggled.connect(lambda *_: _on_vol_changed())
				elif hasattr(w, "currentTextChanged"):
					w.currentTextChanged.connect(lambda *_: _on_vol_changed())
				else:
					w.valueChanged.connect(lambda *_: _on_vol_changed())

		else:  # Mesh
			backend_combo = QComboBox()
			backend_combo.addItems(["analytic_bvh", "projected_intersection_list"])
			backend_combo.setCurrentText(str(getattr(source_obj, "xray_mesh_backend", "analytic_bvh")))
			type_form.addRow("Backend:", backend_combo)

			mode_combo = QComboBox()
			mode_combo.addItems(["solid", "shell"])
			mode_combo.setCurrentText(str(source_obj.xray_mesh_mode))
			type_form.addRow("Mode:", mode_combo)

			scalar_val_spin = QDoubleSpinBox()
			scalar_val_spin.setRange(-1e6, 1e6)
			scalar_val_spin.setDecimals(3)
			scalar_val_spin.setSingleStep(10.0)
			scalar_val_spin.setValue(float(source_obj.xray_mesh_scalar_value))
			type_form.addRow("Scalar value:", scalar_val_spin)

			shell_spin = QDoubleSpinBox()
			shell_spin.setRange(0.001, 1e6)
			shell_spin.setDecimals(3)
			shell_spin.setSingleStep(0.1)
			shell_spin.setValue(float(source_obj.xray_mesh_shell_thickness_mm))
			shell_spin.setEnabled(str(source_obj.xray_mesh_mode).lower() == "shell")
			type_form.addRow("Shell [mm]:", shell_spin)

			def _on_mesh_changed(
				_ref=src_ref, _en=enabled_check, _sc=scale_spin, _bi=bias_spin,
				_at=atten_spin, _mo=material_override_check, _mm=material_response_combo,
				_mt=material_threshold_spin, _ms=material_softness_spin,
				_mwc=material_window_center_spin, _mww=material_window_width_spin,
				_mwm=material_window_mode_combo, _mws=material_window_softness_spin,
				_bc=backend_combo, _mc=mode_combo, _sv=scalar_val_spin, _sh=shell_spin
			):
				s = _ref()
				if s is None:
					return
				s.xray_source_enabled = bool(_en.isChecked())
				s.xray_mesh_backend = str(_bc.currentText()).lower()
				s.xray_mesh_mode = str(_mc.currentText()).lower()
				s.xray_mesh_scalar_value = float(_sv.value())
				s.xray_mesh_shell_thickness_mm = max(1e-4, float(_sh.value()))
				s.xray_scalar_scale = float(_sc.value())
				s.xray_scalar_bias = float(_bi.value())
				s.xray_attenuation_multiplier = max(0.0, float(_at.value()))
				window_width = max(0.0, float(_mww.value()))
				material_cfg = get_xray_material_response_config(s)
				material_cfg.enabled = bool(_mo.isChecked())
				material_cfg.mode = str(_mm.currentText()).lower()
				material_cfg.bone_threshold_hu = (
					float(_mt.value()) if material_cfg.mode == "bone_threshold" else None
				)
				material_cfg.bone_threshold_softness = max(0.0, float(_ms.value()))
				material_cfg.window_center = float(_mwc.value()) if window_width > 0.0 else None
				material_cfg.window_width = window_width if window_width > 0.0 else None
				material_cfg.window_mode = str(_mwm.currentText()).lower()
				material_cfg.window_softness = max(0.0, float(_mws.value()))
				set_xray_material_response_config(s, material_cfg)
				_update_material_override_widgets()
				_sh.setEnabled(s.xray_mesh_mode == "shell")
				AP.updateAllViews()

			for w in (
				enabled_check, scale_spin, bias_spin, atten_spin,
				material_override_check, material_response_combo, material_threshold_spin,
				material_softness_spin, material_window_center_spin, material_window_width_spin,
				material_window_mode_combo, material_window_softness_spin,
				backend_combo, mode_combo, scalar_val_spin, shell_spin
			):
				if hasattr(w, "toggled"):
					w.toggled.connect(lambda *_: _on_mesh_changed())
				elif hasattr(w, "currentTextChanged"):
					w.currentTextChanged.connect(lambda *_: _on_mesh_changed())
				else:
					w.valueChanged.connect(lambda *_: _on_mesh_changed())

		layout.addStretch(1)
		return group

	# ─────────────────────────────────────────────────────────────────────────

	def _configure_form_layout(self, layout):
		"""Keep form labels and fields compact instead of stretching across the dock."""
		layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
		layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
		layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

	def _set_compact_field(self, widget):
		"""Prefer size-hint width for editor widgets used inside form layouts."""
		widget.setSizePolicy(QSizePolicy.Maximum, widget.sizePolicy().verticalPolicy())

	def _set_compact_group(self, group):
		"""Keep group boxes content-sized instead of letting them dictate full tab width."""
		group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

	def _connect_signals(self):
		"""Connect all editor widgets to their slots."""
		self.applyGeometryPresetButton.clicked.connect(self.on_apply_geometry_preset)
		self.modeCombo.currentTextChanged.connect(self.on_mode_changed)
		self.detectorCenterSpin.valueChanged.connect(self.on_detector_center_changed)
		self.detectorNormalSpin.valueChanged.connect(self.on_detector_normal_changed)
		self.detectorUpSpin.valueChanged.connect(self.on_detector_up_changed)
		self.detectorPixelSizeSpin.valueChanged.connect(self.on_detector_pixel_size_changed)
		self.detectorHeightSpin.valueChanged.connect(self.on_detector_shape_changed)
		self.detectorWidthSpin.valueChanged.connect(self.on_detector_shape_changed)
		self.sourcePositionSpin.valueChanged.connect(self.on_source_position_changed)
		self.rayDirectionSpin.valueChanged.connect(self.on_ray_direction_changed)
		self.stepSpin.valueChanged.connect(self.on_step_changed)
		self.qualityCombo.currentTextChanged.connect(self.on_quality_changed)
		self.depthWindowModeCombo.currentTextChanged.connect(self.on_depth_window_mode_changed)
		self.depthWindowRangeSpin.valueChanged.connect(self.on_depth_window_range_changed)
		self.depthWindowOriginSpin.valueChanged.connect(self.on_depth_window_origin_changed)
		self.depthWindowAxisSpin.valueChanged.connect(self.on_depth_window_axis_changed)
		self.depthAlignAxisButton.clicked.connect(self.on_depth_align_axis)
		self.depthAlignOriginButton.clicked.connect(self.on_depth_align_origin)
		self.physicsMaterialResponseModeCombo.currentTextChanged.connect(self.on_physics_material_response_changed)
		self.physicsBoneThresholdSpin.valueChanged.connect(self.on_physics_bone_threshold_changed)
		self.physicsBoneThresholdSoftnessSpin.valueChanged.connect(self.on_physics_bone_threshold_softness_changed)
		self.physicsMaterialWindowCenterSpin.valueChanged.connect(self.on_physics_material_window_changed)
		self.physicsMaterialWindowWidthSpin.valueChanged.connect(self.on_physics_material_window_changed)
		self.physicsMaterialWindowModeCombo.currentTextChanged.connect(self.on_physics_material_window_mode_changed)
		self.physicsMaterialWindowSoftnessSpin.valueChanged.connect(self.on_physics_material_window_softness_changed)
		self.physicsAutoBoneButton.clicked.connect(self.on_auto_bone_from_scene)
		self.presentationModeCombo.currentTextChanged.connect(self.on_presentation_mode_changed)
		self.presentationInvertCheck.toggled.connect(self.on_presentation_invert_changed)
		self.presentationGammaSpin.valueChanged.connect(self.on_presentation_gamma_changed)
		self.presentationContrastSpin.valueChanged.connect(self.on_presentation_contrast_changed)
		self.presentationPercentileSpin.valueChanged.connect(self.on_presentation_percentile_changed)
		self.presentationWindowCenterSpin.valueChanged.connect(self.on_presentation_window_changed)
		self.presentationWindowWidthSpin.valueChanged.connect(self.on_presentation_window_changed)
		self.presentationOverlayAnnotationsCheck.toggled.connect(self.on_presentation_overlay_annotations_changed)
		self.presentationOverlayLabelsCheck.toggled.connect(self.on_presentation_overlay_labels_changed)
		self.presentationOverlayCrossSizeSpin.valueChanged.connect(self.on_presentation_overlay_cross_size_changed)
		self.applyPresentationPresetButton.clicked.connect(self.on_apply_presentation_preset)
		self.physicsMuAirSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsMuWaterSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsHounsfieldAirSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsAttenuationScaleSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsSourceEnergySpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsReferenceEnergySpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsEnergyExponentSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsOutputModeCombo.currentTextChanged.connect(self.on_advanced_physics_changed)
		self.physicsIntensityFloorSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsDistanceFalloffModeCombo.currentTextChanged.connect(self.on_advanced_physics_changed)
		self.physicsDistanceReferenceSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.physicsDistancePowerSpin.valueChanged.connect(self.on_advanced_physics_changed)
		self.sourceInterpolationCombo.currentTextChanged.connect(self.on_advanced_source_changed)
		self.sourcePreprocessModeCombo.currentTextChanged.connect(self.on_advanced_source_changed)
		self.sourcePreprocessLowPercentileSpin.valueChanged.connect(self.on_advanced_source_changed)
		self.sourcePreprocessHighPercentileSpin.valueChanged.connect(self.on_advanced_source_changed)
		self.sourcePreprocessOutputLowSpin.valueChanged.connect(self.on_advanced_source_changed)
		self.sourcePreprocessOutputHighSpin.valueChanged.connect(self.on_advanced_source_changed)
		self.sourceUseFillValueCheck.toggled.connect(self.on_advanced_source_changed)
		self.sourceFillValueSpin.valueChanged.connect(self.on_advanced_source_changed)
		self.geometryAdvancedCheck.toggled.connect(lambda checked: self.advancedSourceGroup.setVisible(checked))
		self.physicsAdvancedCheck.toggled.connect(lambda checked: self.advancedPhysicsGroup.setVisible(checked))
		self.refreshButton.clicked.connect(self.on_refresh_requested)
		self.runSimulationButton.clicked.connect(self.on_run_simulation)
		self.updateDisplayButton.clicked.connect(self.on_update_display)

	@staticmethod
	def create(m, parent=0):
		"""Build the combined base-object and VirtualXRay property panel."""
		return PropWidget.build([PropVirtualXRay(m), PropBaseObject(m)], parent)

	def blockAll(self, b):
		"""Block or unblock signals for all editable widgets in this panel."""
		for widget in (
			self.modeCombo,
			self.geometryPresetCombo,
			self.detectorCenterSpin,
			self.detectorNormalSpin,
			self.detectorUpSpin,
			self.detectorPixelSizeSpin,
			self.detectorHeightSpin,
			self.detectorWidthSpin,
			self.sourcePositionSpin,
			self.rayDirectionSpin,
			self.stepSpin,
			self.qualityCombo,
			self.depthWindowModeCombo,
			self.depthWindowRangeSpin,
			self.depthWindowOriginSpin,
			self.depthWindowAxisSpin,
			self.physicsMaterialResponseModeCombo,
			self.physicsBoneThresholdSpin,
			self.physicsBoneThresholdSoftnessSpin,
			self.physicsMaterialWindowCenterSpin,
			self.physicsMaterialWindowWidthSpin,
			self.physicsMaterialWindowModeCombo,
			self.physicsMaterialWindowSoftnessSpin,
			self.presentationModeCombo,
			self.presentationPresetCombo,
			self.presentationInvertCheck,
			self.presentationGammaSpin,
			self.presentationContrastSpin,
			self.presentationPercentileSpin,
			self.presentationWindowCenterSpin,
			self.presentationWindowWidthSpin,
			self.presentationOverlayAnnotationsCheck,
			self.presentationOverlayLabelsCheck,
			self.presentationOverlayCrossSizeSpin,
			self.physicsMuAirSpin,
			self.physicsMuWaterSpin,
			self.physicsHounsfieldAirSpin,
			self.physicsAttenuationScaleSpin,
			self.physicsSourceEnergySpin,
			self.physicsReferenceEnergySpin,
			self.physicsEnergyExponentSpin,
			self.physicsOutputModeCombo,
			self.physicsIntensityFloorSpin,
			self.physicsDistanceFalloffModeCombo,
			self.physicsDistanceReferenceSpin,
			self.physicsDistancePowerSpin,
			self.sourceInterpolationCombo,
			self.sourcePreprocessModeCombo,
			self.sourcePreprocessLowPercentileSpin,
			self.sourcePreprocessHighPercentileSpin,
			self.sourcePreprocessOutputLowSpin,
			self.sourcePreprocessOutputHighSpin,
			self.sourceUseFillValueCheck,
			self.sourceFillValueSpin,
		):
			widget.blockSignals(b)

	def _update_mode_visibility(self, obj: VirtualXRay):
		"""Enable either the point-source editor or the parallel-ray editor based on the current mode."""
		is_cone = str(obj.projection_mode).lower() == "cone"
		self.sourcePositionSpin.setEnabled(is_cone)
		self.rayDirectionSpin.setEnabled(not is_cone)

	def _update_depth_window_visibility(self, obj: VirtualXRay):
		"""Enable only controls relevant to the currently selected depth-window mode."""
		depth_mode = str(getattr(obj, "depth_window_mode", "off")).strip().lower()
		enabled = depth_mode not in {"", "none", "off"}
		is_planar = depth_mode in {"planar", "planar_auto", "planar_custom"}
		is_custom_planar = depth_mode == "planar_custom"
		self.depthWindowRangeSpin.setEnabled(enabled)
		self.depthWindowOriginSpin.setEnabled(is_planar)
		self.depthWindowAxisSpin.setEnabled(is_custom_planar)
		self.depthAlignAxisButton.setEnabled(is_planar)
		self.depthAlignOriginButton.setEnabled(is_planar)

	def _update_presentation_visibility(self, obj: VirtualXRay):
		"""Enable only presentation controls relevant to the selected display mode."""
		mode = str(obj.presentation_mode).lower()
		is_raw = mode == "raw"
		is_digital = mode == "digital"
		self.presentationInvertCheck.setEnabled(not is_raw)
		self.presentationGammaSpin.setEnabled(not is_raw)
		self.presentationContrastSpin.setEnabled(not is_raw)
		self.presentationPercentileSpin.setEnabled(not is_raw)
		self.presentationWindowCenterSpin.setEnabled(is_digital)
		self.presentationWindowWidthSpin.setEnabled(is_digital)
		overlay_enabled = bool(getattr(obj, "presentation_overlay_annotations", False))
		self.presentationOverlayLabelsCheck.setEnabled(overlay_enabled)
		self.presentationOverlayCrossSizeSpin.setEnabled(overlay_enabled)

	def _update_physics_visibility(self, obj: VirtualXRay):
		"""Enable only physics controls relevant to the selected material window mode."""
		response_mode = str(obj.physics_material_response_mode).lower()
		uses_bone_threshold = response_mode == "bone_threshold"
		width_enabled = obj.physics_material_window_width is not None and float(obj.physics_material_window_width) > 0.0
		mode = str(obj.physics_material_window_mode).lower()
		distance_enabled = str(getattr(obj, "physics_source_distance_falloff_mode", "none")).lower() != "none"
		self.physicsBoneThresholdSpin.setEnabled(uses_bone_threshold)
		self.physicsBoneThresholdSoftnessSpin.setEnabled(uses_bone_threshold)
		self.physicsMaterialWindowModeCombo.setEnabled(width_enabled)
		self.physicsMaterialWindowSoftnessSpin.setEnabled(width_enabled and mode in {"linear", "sigmoid"})
		self.physicsIntensityFloorSpin.setEnabled(str(obj.physics_output_mode).lower() == "intensity")
		self.physicsDistanceReferenceSpin.setEnabled(distance_enabled)
		self.physicsDistancePowerSpin.setEnabled(distance_enabled)

	def _update_advanced_source_visibility(self, obj: VirtualXRay):
		"""Enable explicit source fill value only when that override is active."""
		preprocess_enabled = str(obj.source_preprocess_mode).lower() != "none"
		self.sourcePreprocessLowPercentileSpin.setEnabled(preprocess_enabled)
		self.sourcePreprocessHighPercentileSpin.setEnabled(preprocess_enabled)
		self.sourcePreprocessOutputLowSpin.setEnabled(preprocess_enabled)
		self.sourcePreprocessOutputHighSpin.setEnabled(preprocess_enabled)
		self.sourceFillValueSpin.setEnabled(obj.source_fill_value is not None)

	def updateProperties(self):
		"""Synchronize widget values with the current state of the bound VirtualXRay object."""
		obj = self.obj_ref()
		if obj is None:
			return
		self.blockAll(True)
		self.modeCombo.setCurrentText(str(obj.projection_mode))
		self.detectorCenterSpin.setValue(obj.detector_center_ref)
		self.detectorNormalSpin.setValue(obj.detector_normal_ref)
		self.detectorUpSpin.setValue(obj.detector_up_ref)
		self.detectorPixelSizeSpin.setValue(obj.detector_pixel_size_mm)
		self.detectorHeightSpin.setValue(int(obj.detector_shape_hw[0]))
		self.detectorWidthSpin.setValue(int(obj.detector_shape_hw[1]))
		self.sourcePositionSpin.setValue(obj.source_position_ref)
		self.rayDirectionSpin.setValue(obj.ray_direction_ref)
		self.stepSpin.setValue(float(obj.step_mm))
		self.qualityCombo.setCurrentText(str(obj.quality_profile_name))
		depth_mode = str(getattr(obj, "depth_window_mode", "off"))
		self.depthWindowModeCombo.setCurrentText(depth_mode)
		self.depthWindowRangeSpin.setValue(tuple(getattr(obj, "depth_window_mm", [0.0, 0.0])))
		self.depthWindowOriginSpin.setValue(getattr(obj, "depth_window_origin_ref", np.array([0.0, 0.0, 0.0], dtype=np.float32)))
		depth_mode_norm = depth_mode.strip().lower()
		if depth_mode_norm in {"planar", "planar_auto"} and hasattr(obj, "_projection_axis_ref"):
			self.depthWindowAxisSpin.setValue(obj._projection_axis_ref())
		else:
			self.depthWindowAxisSpin.setValue(getattr(obj, "depth_window_axis_ref", np.array([0.0, 0.0, 1.0], dtype=np.float32)))
		self.physicsMaterialResponseModeCombo.setCurrentText(str(obj.physics_material_response_mode))
		self.physicsBoneThresholdSpin.setValue(0.0 if obj.physics_bone_threshold_hu is None else float(obj.physics_bone_threshold_hu))
		self.physicsBoneThresholdSoftnessSpin.setValue(float(obj.physics_bone_threshold_softness))
		self.physicsMaterialWindowCenterSpin.setValue(0.0 if obj.physics_material_window_center is None else float(obj.physics_material_window_center))
		self.physicsMaterialWindowWidthSpin.setValue(0.0 if obj.physics_material_window_width is None else float(obj.physics_material_window_width))
		self.physicsMaterialWindowModeCombo.setCurrentText(str(obj.physics_material_window_mode))
		self.physicsMaterialWindowSoftnessSpin.setValue(float(obj.physics_material_window_softness))
		self.presentationModeCombo.setCurrentText(str(obj.presentation_mode))
		self.presentationInvertCheck.setChecked(bool(obj.presentation_invert))
		self.presentationGammaSpin.setValue(float(obj.presentation_gamma))
		self.presentationContrastSpin.setValue(float(obj.presentation_contrast))
		self.presentationPercentileSpin.setValue(float(obj.presentation_robust_percentile))
		self.presentationWindowCenterSpin.setValue(0.0 if obj.presentation_window_center is None else float(obj.presentation_window_center))
		self.presentationWindowWidthSpin.setValue(0.0 if obj.presentation_window_width is None else float(obj.presentation_window_width))
		self.presentationOverlayAnnotationsCheck.setChecked(bool(getattr(obj, "presentation_overlay_annotations", False)))
		self.presentationOverlayLabelsCheck.setChecked(bool(getattr(obj, "presentation_overlay_labels", False)))
		self.presentationOverlayCrossSizeSpin.setValue(int(getattr(obj, "presentation_overlay_cross_size_px", 6)))
		self.physicsMuAirSpin.setValue(float(obj.physics_mu_air))
		self.physicsMuWaterSpin.setValue(float(obj.physics_mu_water))
		self.physicsHounsfieldAirSpin.setValue(float(obj.physics_hounsfield_air))
		self.physicsAttenuationScaleSpin.setValue(float(obj.physics_attenuation_scale))
		self.physicsSourceEnergySpin.setValue(float(getattr(obj, "physics_source_energy_kev", 70.0)))
		self.physicsReferenceEnergySpin.setValue(float(getattr(obj, "physics_reference_energy_kev", 70.0)))
		self.physicsEnergyExponentSpin.setValue(float(getattr(obj, "physics_attenuation_energy_exponent", 2.0)))
		self.physicsOutputModeCombo.setCurrentText(str(obj.physics_output_mode))
		self.physicsIntensityFloorSpin.setValue(float(obj.physics_intensity_floor))
		self.physicsDistanceFalloffModeCombo.setCurrentText(str(getattr(obj, "physics_source_distance_falloff_mode", "none")))
		self.physicsDistanceReferenceSpin.setValue(0.0 if getattr(obj, "physics_source_distance_reference_mm", None) is None else float(obj.physics_source_distance_reference_mm))
		self.physicsDistancePowerSpin.setValue(float(getattr(obj, "physics_source_distance_power", 2.0)))
		self.sourceInterpolationCombo.setCurrentText(str(obj.source_interpolation))
		self.sourcePreprocessModeCombo.setCurrentText(str(obj.source_preprocess_mode))
		self.sourcePreprocessLowPercentileSpin.setValue(float(obj.source_preprocess_low_percentile))
		self.sourcePreprocessHighPercentileSpin.setValue(float(obj.source_preprocess_high_percentile))
		self.sourcePreprocessOutputLowSpin.setValue(float(obj.source_preprocess_output_low))
		self.sourcePreprocessOutputHighSpin.setValue(float(obj.source_preprocess_output_high))
		self.sourceUseFillValueCheck.setChecked(obj.source_fill_value is not None)
		self.sourceFillValueSpin.setValue(0.0 if obj.source_fill_value is None else float(obj.source_fill_value))
		self.volumesLabel.setText(str(len(obj.collect_xray_objects())))
		self.renderInfoLabel.setText(obj.info())
		self.updateDisplayButton.setEnabled(obj.last_raw_projection is not None)
		self._update_mode_visibility(obj)
		self._update_depth_window_visibility(obj)
		self._update_physics_visibility(obj)
		self._update_advanced_source_visibility(obj)
		self._update_presentation_visibility(obj)
		self._rebuild_sources_tab(obj)
		self.blockAll(False)

	def _after_change(self, obj: VirtualXRay):
		"""Refresh dependent state after changing one property."""
		obj.invalidate_bb()
		self.updateProperties()
		AP.mainWin.dock["workspace"].refreshAll()
		AP.updateAllViews()

	@pyqtSlot(str)
	def on_mode_changed(self, mode):
		"""Switch between cone-beam and parallel-beam geometry editing."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.projection_mode = str(mode).lower()
		self._after_change(obj)

	@pyqtSlot()
	def on_apply_geometry_preset(self):
		"""Apply one predefined geometry preset and refresh the 3D gizmo immediately."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.apply_geometry_preset(self.geometryPresetCombo.currentText())
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_detector_center_changed(self, values):
		"""Store the detector center in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.detector_center_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_detector_normal_changed(self, values):
		"""Store the detector normal vector in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.detector_normal_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_detector_up_changed(self, values):
		"""Store the detector up vector in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.detector_up_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_detector_pixel_size_changed(self, values):
		"""Store detector pixel pitch along the local detector axes."""
		obj = self.obj_ref()
		obj.detector_pixel_size_mm = [max(1e-4, float(values[0])), max(1e-4, float(values[1]))]
		self._after_change(obj)

	@pyqtSlot(int)
	def on_detector_shape_changed(self, _value):
		"""Store detector raster size in pixels."""
		obj = self.obj_ref()
		obj.detector_shape_hw = [int(self.detectorHeightSpin.value()), int(self.detectorWidthSpin.value())]
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_source_position_changed(self, values):
		"""Store point-source position in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.source_position_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_ray_direction_changed(self, values):
		"""Store parallel-ray direction in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.ray_direction_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(float)
	def on_step_changed(self, value):
		"""Store the ray-marching integration step in millimeters."""
		obj = self.obj_ref()
		obj.step_mm = max(0.01, float(value))
		self._after_change(obj)

	@pyqtSlot(str)
	def on_quality_changed(self, value):
		"""Store the currently selected quality preset name."""
		obj = self.obj_ref()
		obj.quality_profile_name = str(value)
		self._after_change(obj)

	@pyqtSlot(str)
	def on_depth_window_mode_changed(self, value):
		"""Store the optional projection depth-window mode."""
		obj = self.obj_ref()
		obj.depth_window_mode = str(value).strip().lower()
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_depth_window_range_changed(self, values):
		"""Store the depth-window limits in millimetres."""
		obj = self.obj_ref()
		obj.depth_window_mm = [float(values[0]), float(values[1])]
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_depth_window_origin_changed(self, values):
		"""Store the planar depth-window origin in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.depth_window_origin_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(tuple)
	def on_depth_window_axis_changed(self, values):
		"""Store the planar depth-window axis in the local X-ray reference frame."""
		obj = self.obj_ref()
		obj.depth_window_axis_ref = np.asarray(values, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot()
	def on_depth_align_axis(self):
		"""Align the custom planar axis with the current projection axis."""
		obj = self.obj_ref()
		if obj is None or not hasattr(obj, "_projection_axis_ref"):
			return
		obj.depth_window_axis_ref = np.asarray(obj._projection_axis_ref(), dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot()
	def on_depth_align_origin(self):
		"""Set the planar depth-window origin to the current detector centre."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.depth_window_origin_ref = np.asarray(obj.detector_center_ref, dtype=np.float32)
		self._after_change(obj)

	@pyqtSlot(str)
	def on_physics_material_response_changed(self, value):
		"""Store the base HU-to-attenuation response model."""
		obj = self.obj_ref()
		obj.physics_material_response_mode = str(value)
		self._after_change(obj)

	@pyqtSlot(float)
	def on_physics_bone_threshold_changed(self, value):
		"""Store the HU threshold used by the threshold-driven bone response."""
		obj = self.obj_ref()
		obj.physics_bone_threshold_hu = float(value)
		self._after_change(obj)

	@pyqtSlot(float)
	def on_physics_bone_threshold_softness_changed(self, value):
		"""Store the HU softness used by the threshold-driven bone response."""
		obj = self.obj_ref()
		obj.physics_bone_threshold_softness = max(0.0, float(value))
		self._after_change(obj)

	@pyqtSlot()
	def on_physics_material_window_changed(self):
		"""Store an optional HU window applied before attenuation integration."""
		obj = self.obj_ref()
		center = float(self.physicsMaterialWindowCenterSpin.value())
		width = float(self.physicsMaterialWindowWidthSpin.value())
		obj.physics_material_window_center = center if width > 0.0 else None
		obj.physics_material_window_width = width if width > 0.0 else None
		self._after_change(obj)

	@pyqtSlot(str)
	def on_physics_material_window_mode_changed(self, value):
		"""Store the material-window weighting mode used before attenuation integration."""
		obj = self.obj_ref()
		obj.physics_material_window_mode = str(value)
		self._after_change(obj)

	@pyqtSlot(float)
	def on_physics_material_window_softness_changed(self, value):
		"""Store the transition softness used by non-binary material window modes."""
		obj = self.obj_ref()
		obj.physics_material_window_softness = max(0.0, float(value))
		self._after_change(obj)

	@pyqtSlot()
	def on_auto_bone_from_scene(self):
		"""Estimate a bone HU threshold from the current X-ray scene and apply it."""
		obj = self.obj_ref()
		if obj is None:
			return
		try:
			estimate = obj.apply_estimated_bone_threshold()
		except Exception as exc:
			QMessageBox.critical(self, "Auto bone estimation error", str(exc))
			return
		self._after_change(obj)
		self.renderInfoLabel.setText(
			f"{obj.info()}\nAuto bone threshold: T={estimate['threshold']:.0f} HU"
		)

	@pyqtSlot(str)
	def on_presentation_mode_changed(self, value):
		"""Store the currently selected presentation mode."""
		obj = self.obj_ref()
		obj.presentation_mode = str(value)
		self._after_change(obj)

	@pyqtSlot(bool)
	def on_presentation_invert_changed(self, value):
		"""Store the inversion state of the presentation model."""
		obj = self.obj_ref()
		obj.presentation_invert = bool(value)
		self._after_change(obj)

	@pyqtSlot(float)
	def on_presentation_gamma_changed(self, value):
		"""Store the gamma applied by the presentation model."""
		obj = self.obj_ref()
		obj.presentation_gamma = max(0.05, float(value))
		self._after_change(obj)

	@pyqtSlot(float)
	def on_presentation_contrast_changed(self, value):
		"""Store the contrast applied by the presentation model."""
		obj = self.obj_ref()
		obj.presentation_contrast = max(0.05, float(value))
		self._after_change(obj)

	@pyqtSlot(float)
	def on_presentation_percentile_changed(self, value):
		"""Store the robust percentile used by film-like and digital presentation."""
		obj = self.obj_ref()
		obj.presentation_robust_percentile = min(100.0, max(50.0, float(value)))
		self._after_change(obj)

	@pyqtSlot()
	def on_presentation_window_changed(self):
		"""Store optional digital-radiography window center and width."""
		obj = self.obj_ref()
		center = float(self.presentationWindowCenterSpin.value())
		width = float(self.presentationWindowWidthSpin.value())
		obj.presentation_window_center = center if width > 0.0 else None
		obj.presentation_window_width = width if width > 0.0 else None
		self._after_change(obj)

	@pyqtSlot(bool)
	def on_presentation_overlay_annotations_changed(self, value):
		"""Store whether projected annotations should be overlaid on the display image."""
		obj = self.obj_ref()
		obj.presentation_overlay_annotations = bool(value)
		self._after_change(obj)

	@pyqtSlot(bool)
	def on_presentation_overlay_labels_changed(self, value):
		"""Store whether projected annotation labels should be painted on the display image."""
		obj = self.obj_ref()
		obj.presentation_overlay_labels = bool(value)
		self._after_change(obj)

	@pyqtSlot(int)
	def on_presentation_overlay_cross_size_changed(self, value):
		"""Store the marker size used when projected annotations are overlaid on the display image."""
		obj = self.obj_ref()
		obj.presentation_overlay_cross_size_px = max(1, int(value))
		self._after_change(obj)

	@pyqtSlot()
	def on_apply_presentation_preset(self):
		"""Apply one predefined presentation preset for quick visual comparison."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.apply_presentation_preset(self.presentationPresetCombo.currentText())
		self._after_change(obj)

	@pyqtSlot()
	def on_refresh_requested(self):
		"""Refresh the source count and textual scene summary."""
		obj = self.obj_ref()
		self.renderInfoLabel.setText(obj.info())
		self.volumesLabel.setText(str(len(obj.collect_xray_objects())))

	@pyqtSlot()
	def on_advanced_physics_changed(self):
		"""Store lower-level physics parameters exposed for quick backend testing."""
		obj = self.obj_ref()
		obj.physics_mu_air = float(self.physicsMuAirSpin.value())
		obj.physics_mu_water = float(self.physicsMuWaterSpin.value())
		obj.physics_hounsfield_air = float(self.physicsHounsfieldAirSpin.value())
		obj.physics_attenuation_scale = float(self.physicsAttenuationScaleSpin.value())
		obj.physics_source_energy_kev = max(1e-6, float(self.physicsSourceEnergySpin.value()))
		obj.physics_reference_energy_kev = max(1e-6, float(self.physicsReferenceEnergySpin.value()))
		obj.physics_attenuation_energy_exponent = max(0.0, float(self.physicsEnergyExponentSpin.value()))
		obj.physics_output_mode = str(self.physicsOutputModeCombo.currentText())
		obj.physics_intensity_floor = float(self.physicsIntensityFloorSpin.value())
		obj.physics_source_distance_falloff_mode = str(self.physicsDistanceFalloffModeCombo.currentText())
		distance_reference = float(self.physicsDistanceReferenceSpin.value())
		obj.physics_source_distance_reference_mm = distance_reference if distance_reference > 0.0 else None
		obj.physics_source_distance_power = max(0.0, float(self.physicsDistancePowerSpin.value()))
		self._after_change(obj)

	@pyqtSlot()
	def on_advanced_source_changed(self):
		"""Store lower-level source sampling parameters exposed for quick backend testing."""
		obj = self.obj_ref()
		obj.source_interpolation = str(self.sourceInterpolationCombo.currentText())
		obj.source_preprocess_mode = str(self.sourcePreprocessModeCombo.currentText())
		obj.source_preprocess_low_percentile = float(self.sourcePreprocessLowPercentileSpin.value())
		obj.source_preprocess_high_percentile = float(self.sourcePreprocessHighPercentileSpin.value())
		obj.source_preprocess_output_low = float(self.sourcePreprocessOutputLowSpin.value())
		obj.source_preprocess_output_high = float(self.sourcePreprocessOutputHighSpin.value())
		obj.source_fill_value = float(self.sourceFillValueSpin.value()) if self.sourceUseFillValueCheck.isChecked() else None
		self._after_change(obj)

	@pyqtSlot()
	def _display_image_array(self, obj, display_image):
		"""Convert a float display image to uint8 and either create or update one workspace image object."""
		mode = str(obj.presentation_mode).lower()
		if mode == "raw":
			image_u8 = normalize_projection_to_uint8(
				display_image,
				robust_percentile=float(obj.presentation_robust_percentile),
				invert=False,
			)
		else:
			image_u8 = normalize_projection_to_uint8(
				display_image,
				fixed_range=(0.0, 1.0),
				invert=False,
			)
		overlay_enabled = bool(getattr(obj, "presentation_overlay_annotations", False))
		image_u8 = np.ascontiguousarray(np.flipud(image_u8))
		height, width = image_u8.shape[:2]
		if image_u8.ndim == 2 and not overlay_enabled:
			qimage = QImage(
				image_u8.data,
				width,
				height,
				image_u8.strides[0],
				QImage.Format_Grayscale8,
			).copy()
		else:
			if image_u8.ndim == 2:
				image_u8 = np.ascontiguousarray(np.repeat(image_u8[:, :, None], 3, axis=2))
			qimage = QImage(
				image_u8.data,
				width,
				height,
				image_u8.strides[0],
				QImage.Format_RGB888,
			).copy()
		self._paint_projected_overlays(qimage, obj)
		image_obj = getattr(obj, "last_projection_image", None)
		if isinstance(image_obj, Image) and self._is_image_in_workspace(image_obj):
			image_obj.setImage(qimage)
			image_obj.label = f"{obj.label}_projection"
			AP.mainWin.dock["workspace"].refreshAll()
			self._refresh_image_viewers(image_obj)
			return

		image_obj = Image()
		image_obj.setImage(qimage)
		image_obj.label = f"{obj.label}_projection"
		obj.last_projection_image = image_obj
		AP.addObject(image_obj)
		self._refresh_image_viewers(image_obj)

	def _is_image_in_workspace(self, image_obj):
		"""Return True if image_obj is still reachable in the workspace tree."""
		if image_obj.parent is not None:
			return True
		return any(item is image_obj for item in AP.mainWin.workspace.m_data)

	def _paint_projected_overlays(self, qimage, obj):
		"""Paint generic projected overlay primitives on the final display image."""
		if qimage is None or not bool(getattr(obj, "presentation_overlay_annotations", False)):
			return

		annotation_set = getattr(obj, "last_projected_annotations", None)
		if annotation_set is None or not getattr(annotation_set, "items", None):
			return

		painter = QPainter(qimage)
		try:
			painter.setRenderHint(QPainter.Antialiasing, True)
			painter.setRenderHint(QPainter.TextAntialiasing, True)
			for item in annotation_set.items:
				if not item.visible:
					continue
				self._paint_overlay_item(painter, qimage, obj, item)
		finally:
			painter.end()

	def _paint_overlay_item(self, painter, qimage, obj, item):
		"""Paint one generic overlay item and its optional label."""
		if isinstance(item, XRayOverlayCross):
			self._paint_overlay_cross(painter, qimage, item)
			self._paint_overlay_label(painter, qimage, obj, item, item.pixel_uv, item.style.marker_size_px if item.style else 6)
			return
		if isinstance(item, XRayOverlayPolyline):
			self._paint_overlay_polyline(painter, qimage, item)
			anchor_uv = item.pixel_uvs[0] if item.pixel_uvs else None
			self._paint_overlay_label(painter, qimage, obj, item, anchor_uv, item.style.marker_size_px if item.style else 6)

	def _paint_overlay_cross(self, painter, qimage, item: XRayOverlayCross):
		"""Paint one cross overlay item on the projection image."""
		if item.pixel_uv is None or not item.in_bounds:
			return
		style = item.style
		color = QColor(*(style.color_rgba if style is not None else (255, 0, 0, 255)))
		size_px = max(1, int(style.marker_size_px if style is not None else 6))
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

	def _paint_overlay_label(self, painter, qimage, obj, item, anchor_uv, marker_size_px):
		"""Paint one optional overlay label next to the supplied anchor point."""
		if not bool(getattr(obj, "presentation_overlay_labels", False)):
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

	def _freeze_gl_viewers(self):
		"""Temporarily disable GL viewer updates while image objects are inserted into the workspace."""
		gl_viewers = AP.mainWin.allGLViewers()
		for viewer in gl_viewers:
			viewer.setUpdatesEnabled(False)
		return gl_viewers

	def _refresh_image_viewers(self, image_obj):
		"""Refresh auxiliary 2D viewers and property panels that may cache their own pixmaps."""
		mdi_area = getattr(AP.mainWin, "mdiArea", None)
		if mdi_area is not None:
			for sub_window in mdi_area.subWindowList():
				widget = sub_window.widget()
				if getattr(widget, "m_widget", None) is image_obj and hasattr(widget, "_viewer"):
					widget._viewer.setImage(image_obj)

		AP.updateProperties()

	def on_run_simulation(self):
		"""Run one X-ray projection, cache the raw result and insert the display image into the workspace."""
		obj = self.obj_ref()
		if obj is None:
			return

		self.progressBar.setValue(0)
		self.progressBar.setVisible(True)
		self.runSimulationButton.setEnabled(False)

		# Zamroź viewery GL przed processEvents — renderowanie Image przez glTexImage2D
		# poza normalnym cyklem paintGL powoduje crash przy drugiej symulacji.
		gl_viewers = self._freeze_gl_viewers()

		QApplication.processEvents()  # odmaluj pasek przed startem blokującego obliczenia

		def _on_progress(fraction):
			self.progressBar.setValue(int(fraction * 100))
			QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

		_t_integ = perf_counter()
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			try:
				_, stats = obj.project_and_cache(return_stats=True, progress_callback=_on_progress)
				display_image = obj.apply_presentation()
			except Exception as exc:
				QMessageBox.critical(self, "Simulation error", str(exc))
				return
			self._display_image_array(obj, display_image)
			self.updateDisplayButton.setEnabled(True)
			projected_annotations = getattr(getattr(obj, "last_projected_annotations", None), "items", [])
			projected_on_detector = len([item for item in projected_annotations if item.in_bounds])
			self.renderInfoLabel.setText(
				f"{stats.elapsed_seconds:.2f}s, traced={stats.traced_pixels}, "
				f"avgS={stats.average_samples_per_traced_pixel:.1f}, ann={projected_on_detector}"
			)
		finally:
			QApplication.restoreOverrideCursor()
			_t_integ_end = perf_counter()
			_log.info(
				"Simulation done in %.3f s",
				_t_integ_end - _t_integ,
			)
			for v in gl_viewers:
				v.setUpdatesEnabled(True)
			AP.updateAllViews()
			self.progressBar.setVisible(False)
			self.runSimulationButton.setEnabled(True)


	def on_update_display(self):
		"""Re-apply the current presentation model to the cached raw projection without re-projecting."""
		obj = self.obj_ref()
		if obj is None:
			return
		self.updateDisplayButton.setEnabled(False)
		gl_viewers = self._freeze_gl_viewers()
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			try:
				display_image = obj.apply_presentation()
			except Exception as exc:
				QMessageBox.critical(self, "Update display error", str(exc))
				return
			if display_image is None:
				return
			self._display_image_array(obj, display_image)
		finally:
			QApplication.restoreOverrideCursor()
			for viewer in gl_viewers:
				viewer.setUpdatesEnabled(True)
			self.updateDisplayButton.setEnabled(obj.last_raw_projection is not None)
			AP.updateAllViews()
