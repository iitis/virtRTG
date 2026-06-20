# -*- coding: utf-8 -*-
"""Property panel for the `VirtualXRay` scene object."""

from __future__ import annotations

import os
from time import perf_counter
import logging

from dpVision.gui.flowLayout import FlowLayout
_log = logging.getLogger(__name__)

import weakref

import numpy as np
from PyQt5.QtCore import Qt, QEventLoop, pyqtSlot
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
	QFileDialog,
	QSizePolicy,
)

from dpVision import AP, Mesh, Volumetric
from dpVision.gui.multiSpinBox import MultiSpinBox
from dpVision.gui.propBaseObject import PropBaseObject
from dpVision.gui.propWidget import PropWidget
from ..virtualXRay import VirtualXRay
from ..detectorImage import DetectorImage
from .detectorImageViewer import DetectorImageViewerChild
from ..xray.xraySource import get_xray_material_response_config, set_xray_material_response_config
from ..xray.xraySource import ensure_xray_source_config

class _CollapsibleGroup(QWidget):
	"""Simple collapsible section: a toggle button + a hidden/shown body widget."""

	def __init__(self, title, collapsed=True, parent=None):
		"""Create one collapsible group with a body that participates in relayout."""
		super().__init__(parent)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
		self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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


class _FlowGroupBox(QGroupBox):
	"""Group box that forwards height-for-width to its wrapping child layout."""

	def hasHeightForWidth(self):
		"""Report height-for-width when the installed layout supports it."""
		layout = self.layout()
		return bool(layout is not None and layout.hasHeightForWidth())

	def heightForWidth(self, width):
		"""Return the group height needed for the given content width."""
		layout = self.layout()
		if layout is None or not layout.hasHeightForWidth():
			return super().heightForWidth(width)
		return layout.totalHeightForWidth(max(0, width))

	def minimumSizeHint(self):
		"""Keep the minimum size consistent with the flow layout's wrapped content."""
		layout = self.layout()
		if layout is None:
			return super().minimumSizeHint()
		return layout.minimumSize()

	def sizeHint(self):
		"""Prefer the current width and let the height follow the wrapped content."""
		size_hint = super().sizeHint()
		if self.hasHeightForWidth():
			size_hint.setHeight(self.heightForWidth(size_hint.width()))
		return size_hint


class PropVirtualXRay(PropWidget):
	"""Edit basic source and detector parameters of one `VirtualXRay` scene node."""

	def __init__(self, _obj: VirtualXRay, parent=None):
		"""Build the property editor widgets and bind them to the provided scene object."""
		super().__init__(parent)
		self.obj_ref = weakref.ref(_obj)
		self._setup_ui()
		self._connect_signals()


	def _create_geom_tab(self): 
		# ── Geometry tab: Scene + Detector + Source + Sampling + Advanced source ──
		geomTab = QWidget()
		geomLayout = QVBoxLayout(geomTab)
		geomLayout.setAlignment(Qt.AlignTop)

		sceneGroup, scene_layout = self._create_flow_group("Scene")
		self.volumesLabel = QLabel("-")
		self._set_compact_field(self.volumesLabel)
		self.geometryPresetWidget = QWidget()
		self._set_compact_field(self.geometryPresetWidget)
		geometry_preset_layout = QHBoxLayout(self.geometryPresetWidget)
		geometry_preset_layout.setContentsMargins(0, 0, 0, 0)
		geometry_preset_layout.setSpacing(4)
		self.geometryPresetCombo = QComboBox()
		self.geometryPresetCombo.addItems(VirtualXRay.geometry_preset_names())
		self._set_compact_field(self.geometryPresetCombo)
		self.geometryPresetCombo.setMinimumContentsLength(8)
		self.geometryPresetCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
		self.applyGeometryPresetButton = QPushButton("Use")
		self.applyGeometryPresetButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self.applyGeometryPresetButton.setToolTip("Apply the selected geometry preset.")
		geometry_preset_layout.addWidget(self.geometryPresetCombo)
		geometry_preset_layout.addWidget(self.applyGeometryPresetButton)
		self.modeCombo = QComboBox()
		self.modeCombo.addItems(["cone", "parallel"])
		self._set_compact_field(self.modeCombo)
		self._add_flow_control(scene_layout, "Objects", self.volumesLabel)
		self._add_flow_control(scene_layout, "Preset", self.geometryPresetWidget)
		self._add_flow_control(scene_layout, "Mode", self.modeCombo)
		geomLayout.addWidget(sceneGroup)

		detectorGroup, detector_layout = self._create_collapsible_flow_group("Detector", collapsed=True)
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
		self._add_flow_control(detector_layout, "Center [mm]", self.detectorCenterSpin)
		self._add_flow_control(detector_layout, "Normal", self.detectorNormalSpin)
		self._add_flow_control(detector_layout, "Up", self.detectorUpSpin)
		self._add_flow_control(detector_layout, "Pixel size [mm]", self.detectorPixelSizeSpin)
		self._add_flow_control(detector_layout, "Shape [px]", self.detectorShapeWidget)
		geomLayout.addWidget(detectorGroup)

		sourceGroup, source_layout = self._create_collapsible_flow_group("Source", collapsed=True)
		self.sourcePositionSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self.rayDirectionSpin = MultiSpinBox(3, labels=("X", "Y", "Z"))
		self._set_compact_field(self.sourcePositionSpin)
		self._set_compact_field(self.rayDirectionSpin)
		self._add_flow_control(source_layout, "Position [mm]", self.sourcePositionSpin)
		self._add_flow_control(source_layout, "Direction", self.rayDirectionSpin)
		geomLayout.addWidget(sourceGroup)

		samplingGroup, sampling_layout = self._create_flow_group("Sampling")
		self.stepSpin = QDoubleSpinBox()
		self.stepSpin.setRange(0.01, 50.0)
		self.stepSpin.setDecimals(3)
		self.stepSpin.setSingleStep(0.1)
		self._set_compact_field(self.stepSpin)
		self.qualityCombo = QComboBox()
		self.qualityCombo.addItems(["draft", "normal", "high", "custom"])
		self._set_compact_field(self.qualityCombo)
		self._add_flow_control(sampling_layout, "Step [mm]", self.stepSpin)
		self._add_flow_control(sampling_layout, "Quality", self.qualityCombo)
		geomLayout.addWidget(samplingGroup)

		depthWindowGroup, depth_window_layout = self._create_collapsible_flow_group("Depth window", collapsed=True)
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
		self._add_flow_control(depth_window_layout, "Mode", self.depthWindowModeCombo)
		self._add_flow_control(depth_window_layout, "Range [mm]", self.depthWindowRangeSpin)
		self._add_flow_control(depth_window_layout, "Origin [mm]", self.depthWindowOriginSpin)
		self._add_flow_control(depth_window_layout, "Axis", self.depthWindowAxisSpin)
		self._add_flow_control(depth_window_layout, "Tools", self.depthWindowToolsWidget)
		geomLayout.addWidget(depthWindowGroup)

		self.geometryAdvancedCheck = QCheckBox("Show advanced")
		geomLayout.addWidget(self.geometryAdvancedCheck)

		self.advancedSourceGroup, advanced_source_layout = self._create_flow_group("Advanced")
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
		self._add_flow_control(advanced_source_layout, "Interpolation", self.sourceInterpolationCombo)
		self._add_flow_control(advanced_source_layout, "Preprocess", self.sourcePreprocessModeCombo)
		self._add_flow_control(advanced_source_layout, "Input low [%]", self.sourcePreprocessLowPercentileSpin)
		self._add_flow_control(advanced_source_layout, "Input high [%]", self.sourcePreprocessHighPercentileSpin)
		self._add_flow_control(advanced_source_layout, "Output low", self.sourcePreprocessOutputLowSpin)
		self._add_flow_control(advanced_source_layout, "Output high", self.sourcePreprocessOutputHighSpin)
		self._add_flow_control(advanced_source_layout, "", self.sourceUseFillValueCheck)
		self._add_flow_control(advanced_source_layout, "Fill value", self.sourceFillValueSpin)
		self.advancedSourceGroup.setVisible(False)
		geomLayout.addWidget(self.advancedSourceGroup)

		geomLayout.addStretch(1)
		return geomTab
	
	def _create_phys_tab(self):

		def _create_physicsGroup(self):
			physicsGroup, physics_layout = self._create_flow_group("Material filter")
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
			self.physicsAutoBoneButton.setToolTip("Automatically set the bone threshold based on the current volumes and source energy.")
			self.physicsAutoBoneButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
			self._add_flow_control(physics_layout, "Response", self.physicsMaterialResponseModeCombo)
			self._add_flow_control(physics_layout, "Bone thresh.", self.physicsBoneThresholdSpin)
			self._add_flow_control(physics_layout, "Bone softn.", self.physicsBoneThresholdSoftnessSpin)
			self._add_flow_control(physics_layout, "Center [HU]", self.physicsMaterialWindowCenterSpin)
			self._add_flow_control(physics_layout, "Width [HU]", self.physicsMaterialWindowWidthSpin)
			self._add_flow_control(physics_layout, "Mode", self.physicsMaterialWindowModeCombo)
			self._add_flow_control(physics_layout, "Softness [HU]", self.physicsMaterialWindowSoftnessSpin)
			self._add_flow_control(physics_layout, "", self.physicsAutoBoneButton)
			return physicsGroup
	
		def _create_advancedPhysicsGroup(self):
			advancedPhysicsGroup = _FlowGroupBox("Advanced")
			advancedPhysicsGroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
			advanced_physics_layout = FlowLayout(advancedPhysicsGroup)
			advanced_physics_layout.setContentsMargins(6, 6, 6, 6)
			advanced_physics_layout.setSpacing(4)
			advanced_physics_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
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
			advanced_physics_layout.addWidget(self._vcontrol("mu_air", self.physicsMuAirSpin))
			advanced_physics_layout.addWidget(self._vcontrol("mu_water", self.physicsMuWaterSpin))
			advanced_physics_layout.addWidget(self._vcontrol("hounsfield_air", self.physicsHounsfieldAirSpin))
			advanced_physics_layout.addWidget(self._vcontrol("attenuation_scale", self.physicsAttenuationScaleSpin))
			advanced_physics_layout.addWidget(self._vcontrol("source_energy_kev", self.physicsSourceEnergySpin))
			advanced_physics_layout.addWidget(self._vcontrol("reference_energy_kev", self.physicsReferenceEnergySpin))
			advanced_physics_layout.addWidget(self._vcontrol("energy_exponent", self.physicsEnergyExponentSpin))
			advanced_physics_layout.addWidget(self._vcontrol("output_mode", self.physicsOutputModeCombo))
			advanced_physics_layout.addWidget(self._vcontrol("intensity_floor", self.physicsIntensityFloorSpin))
			advanced_physics_layout.addWidget(self._vcontrol("distance_falloff", self.physicsDistanceFalloffModeCombo))
			advanced_physics_layout.addWidget(self._vcontrol("distance_ref [mm]", self.physicsDistanceReferenceSpin))
			advanced_physics_layout.addWidget(self._vcontrol("distance_power", self.physicsDistancePowerSpin))
			return advancedPhysicsGroup
		
		# ── Physics tab: Material filter + Advanced physics ────────────
		physTab = QWidget()
		physLayout = QVBoxLayout(physTab)
		physLayout.setAlignment(Qt.AlignTop)

		physicsGroup = _create_physicsGroup(self)
		physLayout.addWidget(physicsGroup)

		self.physicsAdvancedCheck = QCheckBox("Show advanced")
		physLayout.addWidget(self.physicsAdvancedCheck)

		self.advancedPhysicsGroup = _create_advancedPhysicsGroup(self)
		self.advancedPhysicsGroup.setVisible(False)
		physLayout.setAlignment(self.advancedPhysicsGroup, Qt.AlignTop)
		physLayout.addWidget(self.advancedPhysicsGroup)

		physLayout.addStretch(1)
		return physTab
	

	def _create_run_tab(self):
		# ── Run tab ────────────────────────────────────────────────────
		runTab = QWidget()
		runTabLayout = QVBoxLayout(runTab)
		runTabLayout.setAlignment(Qt.AlignTop)
		runGroup, runGroupLayout = self._create_flow_group("Run")
		motionFrameWidget = QWidget()
		motionFrameWidget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		motionFrameLayout = QHBoxLayout(motionFrameWidget)
		motionFrameLayout.setContentsMargins(0, 0, 0, 0)
		motionFrameLayout.setSpacing(4)
		self.motionFrameModeCombo = QComboBox()
		self.motionFrameModeCombo.addItem("Active frame", "active")
		self.motionFrameModeCombo.addItem("All frames", "all")
		self._set_compact_field(self.motionFrameModeCombo)
		motionFrameLayout.addWidget(QLabel("Motion frames:"))
		motionFrameLayout.addWidget(self.motionFrameModeCombo)
		self._add_flow_control(runGroupLayout, "Motion frames", motionFrameWidget)
		self.motionFrameInfoLabel = QLabel("For scenes without Motion this setting has no effect.")
		self.motionFrameInfoLabel.setWordWrap(True)
		self.motionFrameInfoLabel.setMaximumWidth(320)

		self.refreshButton = QPushButton("Refresh")
		self.runSimulationButton = QPushButton("Run simulation")
		self.updateDisplayButton = QPushButton("Update view")
		self.openDetectorImageButton = QPushButton("Open detector")
		self.cacheStageCombo = QComboBox()
		self.cacheStageCombo.addItem("Detector", "raw")
		self.cacheStageCombo.addItem("Integral", "line_integral")
		self._set_compact_field(self.cacheStageCombo)
		self.cacheStageCombo.setMinimumContentsLength(8)
		self.cacheStageCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
		self.cacheFormatCombo = QComboBox()
		self.cacheFormatCombo.addItem("Package", ".npz")
		self.cacheFormatCombo.addItem("NumPy", ".npy")
		self.cacheFormatCombo.addItem("Text", ".txt")
		self.cacheFormatCombo.addItem("CSV", ".csv")
		self.cacheFormatCombo.addItem("TSV", ".tsv")
		self._set_compact_field(self.cacheFormatCombo)
		self.cacheFormatCombo.setMinimumContentsLength(7)
		self.cacheFormatCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
		self.cacheFormatCombo.setToolTip(
			"Package (*.npz), NumPy (*.npy), Text (*.txt), CSV (*.csv), or TSV (*.tsv)."
		)
		self.exportProjectionButton = QPushButton("Export proj.")
		self.importProjectionButton = QPushButton("Import proj.")
		self.exportSourcesButton = QPushButton("Export sources")
		for button in (
			self.refreshButton,
			self.runSimulationButton,
			self.updateDisplayButton,
			self.openDetectorImageButton,
			self.exportProjectionButton,
			self.importProjectionButton,
			self.exportSourcesButton,
		):
			button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self.updateDisplayButton.setEnabled(False)
		self.openDetectorImageButton.setEnabled(False)
		self.renderInfoLabel = QLabel("")
		self.renderInfoLabel.setWordWrap(True)
		self.renderInfoLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
		self._add_flow_control(runGroupLayout, "", self.runSimulationButton)
		self._add_flow_control(runGroupLayout, "", self.updateDisplayButton)
		self._add_flow_control(runGroupLayout, "", self.openDetectorImageButton)
		self._add_flow_control(runGroupLayout, "", self.refreshButton)
		self.presentationInfoLabel = QLabel(
			"Presentation controls are handled by the generated DetectorImage object."
		)
		self.presentationInfoLabel.setWordWrap(True)
		self.presentationInfoLabel.setMaximumWidth(320)
		self.presentationInfoLabel.setToolTip(
			"Display parameters are edited on the generated DetectorImage object."
		)
		self._add_flow_control(runGroupLayout, "", self._wrap_flow_widget(self.motionFrameInfoLabel))
		self._add_flow_control(runGroupLayout, "", self._wrap_flow_widget(self.presentationInfoLabel))
		runTabLayout.addWidget(runGroup)
		cacheGroup, cacheGroupLayout = self._create_flow_group("Cache")
		self.cacheSelectorsWidget = QWidget()
		self._set_compact_field(self.cacheSelectorsWidget)
		cacheSelectorsLayout = FlowLayout(self.cacheSelectorsWidget, spacing=4)
		cacheSelectorsLayout.setContentsMargins(0, 0, 0, 0)
		cacheSelectorsLayout.addWidget(self.cacheStageCombo)
		cacheSelectorsLayout.addWidget(self.cacheFormatCombo)
		self.cacheActionsWidget = QWidget()
		self._set_compact_field(self.cacheActionsWidget)
		cacheActionsLayout = FlowLayout(self.cacheActionsWidget, spacing=4)
		cacheActionsLayout.setContentsMargins(0, 0, 0, 0)
		cacheActionsLayout.addWidget(self.exportProjectionButton)
		cacheActionsLayout.addWidget(self.importProjectionButton)
		cacheActionsLayout.addWidget(self.exportSourcesButton)
		self._add_flow_control(cacheGroupLayout, "Data", self.cacheSelectorsWidget)
		self._add_flow_control(cacheGroupLayout, "Actions", self.cacheActionsWidget)
		runTabLayout.addWidget(cacheGroup)
		self.progressBar = QProgressBar()
		self.progressBar.setRange(0, 100)
		self.progressBar.setValue(0)
		self.progressBar.setVisible(False)
		runTabLayout.addWidget(self.progressBar)
		runTabLayout.addWidget(self.renderInfoLabel)
		runTabLayout.addStretch(1)
		return runTab

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
		geomTab = self._create_geom_tab()
		self.tabs.addTab(geomTab, "Geometry")

		# ── Physics tab: Material filter + Advanced physics ────────────
		physTab = self._create_phys_tab()
		self.tabs.addTab(physTab, "Physics")

		runTab = self._create_run_tab()
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
		self._sourcesCombo.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self._sourcesCombo.setMinimumContentsLength(16)
		self._sourcesCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
		sourcesTabLayout.addWidget(self._sourcesCombo)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self._sourcesStack = QStackedWidget()
		scroll.setWidget(self._sourcesStack)
		sourcesTabLayout.addWidget(scroll)

		self._sourcesCombo.currentIndexChanged.connect(self._sourcesStack.setCurrentIndex)

		self._sources_tab_index = self.tabs.insertTab(0, sourcesTab, "Sources")

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
			self._sourcesCombo.addItem("-")
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
			box, form = self._create_flow_group(title)
			layout.addWidget(box)
			return form

		source_form = _section("Source")
		material_form = _section("Material Response")
		type_form = _section("Volume Sampling" if isinstance(source_obj, Volumetric) else "Mesh Model")

		enabled_check = QCheckBox("Enabled")
		enabled_check.setChecked(bool(source_obj.xray_source_enabled))
		self._add_flow_control(source_form, "", enabled_check)

		scale_spin = QDoubleSpinBox()
		scale_spin.setRange(-1e3, 1e3)
		scale_spin.setDecimals(6)
		scale_spin.setSingleStep(0.05)
		scale_spin.setValue(float(source_obj.xray_scalar_scale))
		self._set_compact_field(scale_spin)
		self._add_flow_control(source_form, "Scalar scale", scale_spin)

		bias_spin = QDoubleSpinBox()
		bias_spin.setRange(-1e6, 1e6)
		bias_spin.setDecimals(3)
		bias_spin.setSingleStep(10.0)
		bias_spin.setValue(float(source_obj.xray_scalar_bias))
		self._set_compact_field(bias_spin)
		self._add_flow_control(source_form, "Scalar bias", bias_spin)

		atten_spin = QDoubleSpinBox()
		atten_spin.setRange(0.0, 1e6)
		atten_spin.setDecimals(6)
		atten_spin.setSingleStep(0.05)
		atten_spin.setValue(float(source_obj.xray_attenuation_multiplier))
		self._set_compact_field(atten_spin)
		self._add_flow_control(source_form, "Attenuation x", atten_spin)

		material_override_check = QCheckBox("Use individual material response")
		material_override_check.setChecked(bool(material_config.enabled))
		self._add_flow_control(material_form, "", material_override_check)

		material_response_combo = QComboBox()
		material_response_combo.addItems(["linear", "piecewise_bone", "piecewise_soft_tissue", "bone_threshold"])
		material_response_combo.setCurrentText(str(material_config.mode))
		self._set_compact_field(material_response_combo)
		self._add_flow_control(material_form, "Response", material_response_combo)

		material_threshold_spin = QDoubleSpinBox()
		material_threshold_spin.setRange(-1e6, 1e6)
		material_threshold_spin.setDecimals(3)
		material_threshold_spin.setSingleStep(10.0)
		material_threshold_spin.setValue(
			0.0 if material_config.bone_threshold_hu is None
			else float(material_config.bone_threshold_hu)
		)
		self._set_compact_field(material_threshold_spin)
		self._add_flow_control(material_form, "Bone threshold", material_threshold_spin)

		material_softness_spin = QDoubleSpinBox()
		material_softness_spin.setRange(0.0, 1e6)
		material_softness_spin.setDecimals(3)
		material_softness_spin.setSingleStep(10.0)
		material_softness_spin.setValue(float(material_config.bone_threshold_softness))
		self._set_compact_field(material_softness_spin)
		self._add_flow_control(material_form, "Threshold soft.", material_softness_spin)

		material_window_center_spin = QDoubleSpinBox()
		material_window_center_spin.setRange(-1e6, 1e6)
		material_window_center_spin.setDecimals(3)
		material_window_center_spin.setSingleStep(10.0)
		material_window_center_spin.setValue(
			0.0 if material_config.window_center is None
			else float(material_config.window_center)
		)
		self._set_compact_field(material_window_center_spin)
		self._add_flow_control(material_form, "Window center", material_window_center_spin)

		material_window_width_spin = QDoubleSpinBox()
		material_window_width_spin.setRange(0.0, 1e6)
		material_window_width_spin.setDecimals(3)
		material_window_width_spin.setSingleStep(10.0)
		material_window_width_spin.setValue(
			0.0 if material_config.window_width is None
			else float(material_config.window_width)
		)
		self._set_compact_field(material_window_width_spin)
		self._add_flow_control(material_form, "Window width", material_window_width_spin)

		material_window_mode_combo = QComboBox()
		material_window_mode_combo.addItems(["hard", "linear", "sigmoid"])
		material_window_mode_combo.setCurrentText(str(material_config.window_mode))
		self._set_compact_field(material_window_mode_combo)
		self._add_flow_control(material_form, "Window mode", material_window_mode_combo)

		material_window_softness_spin = QDoubleSpinBox()
		material_window_softness_spin.setRange(0.0, 1e6)
		material_window_softness_spin.setDecimals(3)
		material_window_softness_spin.setSingleStep(10.0)
		material_window_softness_spin.setValue(float(material_config.window_softness))
		self._set_compact_field(material_window_softness_spin)
		self._add_flow_control(material_form, "Window soft.", material_window_softness_spin)

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
			self._set_compact_field(interp_combo)
			self._add_flow_control(type_form, "Interpolation", interp_combo)

			backend_combo = QComboBox()
			backend_combo.addItems(["sampling", "siddon"])
			backend_combo.setToolTip(
				"sampling – uniform ray-marching (step_mm)\n"
				"siddon  – exact voxel traversal (chord-length, step_mm independent)"
			)
			backend_combo.setCurrentText(str(getattr(source_obj, "xray_volume_backend", "sampling")))
			self._set_compact_field(backend_combo)
			self._add_flow_control(type_form, "Backend", backend_combo)

			fill_check = QCheckBox("Use explicit fill value")
			fill_check.setChecked(bool(source_obj.xray_fill_value_override_enabled))
			self._add_flow_control(type_form, "", fill_check)

			fill_spin = QDoubleSpinBox()
			fill_spin.setRange(-1e9, 1e9)
			fill_spin.setDecimals(3)
			fill_spin.setSingleStep(10.0)
			fill_spin.setValue(float(source_obj.xray_fill_value_override))
			fill_spin.setEnabled(bool(source_obj.xray_fill_value_override_enabled))
			self._set_compact_field(fill_spin)
			self._add_flow_control(type_form, "Fill value", fill_spin)

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
			self._set_compact_field(backend_combo)
			self._add_flow_control(type_form, "Backend", backend_combo)

			mode_combo = QComboBox()
			mode_combo.addItems(["solid", "shell"])
			mode_combo.setCurrentText(str(source_obj.xray_mesh_mode))
			self._set_compact_field(mode_combo)
			self._add_flow_control(type_form, "Mode", mode_combo)

			scalar_val_spin = QDoubleSpinBox()
			scalar_val_spin.setRange(-1e6, 1e6)
			scalar_val_spin.setDecimals(3)
			scalar_val_spin.setSingleStep(10.0)
			scalar_val_spin.setValue(float(source_obj.xray_mesh_scalar_value))
			self._set_compact_field(scalar_val_spin)
			self._add_flow_control(type_form, "Scalar value", scalar_val_spin)

			shell_spin = QDoubleSpinBox()
			shell_spin.setRange(0.001, 1e6)
			shell_spin.setDecimals(3)
			shell_spin.setSingleStep(0.1)
			shell_spin.setValue(float(source_obj.xray_mesh_shell_thickness_mm))
			shell_spin.setEnabled(str(source_obj.xray_mesh_mode).lower() == "shell")
			self._set_compact_field(shell_spin)
			self._add_flow_control(type_form, "Shell [mm]", shell_spin)

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

	def _create_flow_group(self, title):
		"""Create one width-aware group box with wrapping controls."""
		group = _FlowGroupBox(title)
		group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		layout = FlowLayout(group)
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		return group, layout

	def _create_collapsible_flow_group(self, title, collapsed=True):
		"""Create one collapsible section whose body uses a wrapping flow layout."""
		group = _CollapsibleGroup(title, collapsed=collapsed)
		layout = FlowLayout(group.body())
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		return group, layout

	def _wrap_flow_widget(self, control):
		"""Wrap one standalone control so it behaves like a flow item."""
		widget = QWidget()
		widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout = QVBoxLayout(widget)
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.addWidget(control)
		return widget

	def _vcontrol(self, label_text, control):
		"""Return one labeled control block suitable for flow-based groups."""
		widget = QWidget()
		widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout = QVBoxLayout()
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		widget.setLayout(layout)
		label = QLabel(f"{label_text}:")
		label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout.addWidget(label)
		layout.addWidget(control)
		return widget

	def _add_flow_control(self, layout, label_text, control):
		"""Add one wrapped control to a flow layout, with or without label."""
		if label_text:
			layout.addWidget(self._vcontrol(label_text, control))
		else:
			layout.addWidget(self._wrap_flow_widget(control))

	def _add_labeled_control(self, layout, label_text, control):
		"""Add one compact 'label over control' block to a vertical layout."""
		label = QLabel(f"{label_text}:")
		label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout.addWidget(label)
		layout.addWidget(control)

	def _set_compact_field(self, widget):
		"""Prefer size-hint width for editor widgets used inside form layouts."""
		widget.setSizePolicy(QSizePolicy.Maximum, widget.sizePolicy().verticalPolicy())

	def _set_compact_group(self, group):
		"""Keep group boxes content-sized instead of letting them dictate full tab width."""
		group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

	def _width_debug_label(self, widget):
		"""Return one compact widget label for width-constraint diagnostics."""
		text = ""
		for attr_name in ("text", "windowTitle", "title", "currentText"):
			attr = getattr(widget, attr_name, None)
			if callable(attr):
				try:
					text = str(attr() or "").strip()
				except Exception:
					text = ""
			if text:
				break
		text = text.replace("\n", " ")
		if len(text) > 40:
			text = text[:37] + "..."
		return f"{widget.__class__.__name__}('{text}')" if text else widget.__class__.__name__

	def _set_render_info_text(self, text):
		"""Keep status text readable without letting it dominate the dock width."""
		full_text = str(text)
		short_text = full_text
		if len(short_text) > 72:
			short_text = short_text[:69].rstrip() + "..."
		self.renderInfoLabel.setText(short_text)
		self.renderInfoLabel.setToolTip(full_text)

	def _maybe_report_width_constraints(self):
		"""Log the widest size hints in this panel when width debugging is enabled."""
		debug_flag = os.environ.get("DPVISION_DEBUG_PROP_WIDTHS", "").strip().lower()
		if debug_flag not in {"1", "true", "yes", "on"}:
			return
		candidates = []
		for widget in [self] + self.findChildren(QWidget):
			if not widget.isVisible():
				continue
			size_hint_width = widget.sizeHint().width()
			minimum_hint_width = widget.minimumSizeHint().width()
			minimum_width = widget.minimumWidth()
			policy = widget.sizePolicy().horizontalPolicy()
			candidates.append({
				"widget": widget,
				"label": self._width_debug_label(widget),
				"size_hint": int(size_hint_width),
				"minimum_hint": int(minimum_hint_width),
				"minimum_width": int(minimum_width),
				"policy": int(policy),
			})
		candidates.sort(
			key=lambda item: (
				item["minimum_hint"],
				item["size_hint"],
				item["minimum_width"],
			),
			reverse=True,
		)
		_log.warning("PropVirtualXRay width audit: top contributors")
		for item in candidates[:20]:
			_log.warning(
				"  minHint=%s sizeHint=%s minWidth=%s policy=%s %s",
				item["minimum_hint"],
				item["size_hint"],
				item["minimum_width"],
				item["policy"],
				item["label"],
			)

	def _set_advanced_physics_visible(self, visible):
		"""Toggle advanced physics and refresh wrapped geometry in the dock."""
		self.advancedPhysicsGroup.setVisible(visible)
		self.advancedPhysicsGroup.updateGeometry()
		self.tabs.updateGeometry()
		self.updateGeometry()

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
		self.motionFrameModeCombo.currentIndexChanged.connect(self.on_motion_frame_mode_changed)
		self.geometryAdvancedCheck.toggled.connect(lambda checked: self.advancedSourceGroup.setVisible(checked))
		self.physicsAdvancedCheck.toggled.connect(self._set_advanced_physics_visible)
		self.refreshButton.clicked.connect(self.on_refresh_requested)
		self.runSimulationButton.clicked.connect(self.on_run_simulation)
		self.updateDisplayButton.clicked.connect(self.on_update_display)
		self.openDetectorImageButton.clicked.connect(self.on_open_detector_image)
		self.exportProjectionButton.clicked.connect(self.on_export_projection_requested)
		self.importProjectionButton.clicked.connect(self.on_import_projection_requested)
		self.exportSourcesButton.clicked.connect(self.on_export_source_projections_requested)

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
			self.motionFrameModeCombo,
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
		motion_frame_mode = str(getattr(obj, "motion_frame_mode", "active")).strip().lower()
		motion_frame_mode_index = self.motionFrameModeCombo.findData(motion_frame_mode)
		self.motionFrameModeCombo.setCurrentIndex(0 if motion_frame_mode_index < 0 else motion_frame_mode_index)
		self.volumesLabel.setText(str(len(obj.collect_xray_objects())))
		self._set_render_info_text(obj.info())
		self.updateDisplayButton.setEnabled(obj.last_raw_projection is not None)
		self.openDetectorImageButton.setEnabled(isinstance(getattr(obj, "last_projection_image", None), DetectorImage))
		self.exportProjectionButton.setEnabled(obj.last_raw_projection is not None or getattr(obj, "last_line_integral_projection", None) is not None)
		self.exportSourcesButton.setEnabled(bool(getattr(obj, "last_source_projections", [])))
		self._update_mode_visibility(obj)
		self._update_depth_window_visibility(obj)
		self._update_physics_visibility(obj)
		self._update_advanced_source_visibility(obj)
		self._rebuild_sources_tab(obj)
		self.blockAll(False)
		self._maybe_report_width_constraints()

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
		self._set_render_info_text(
			f"{obj.info()}\nAuto bone threshold: T={estimate['threshold']:.0f} HU"
		)


	@pyqtSlot()
	def on_refresh_requested(self):
		"""Refresh the source count and textual scene summary."""
		obj = self.obj_ref()
		self._set_render_info_text(obj.info())
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

	@pyqtSlot(int)
	def on_motion_frame_mode_changed(self, _index):
		"""Store whether projection should use one active frame or all motion frames."""
		obj = self.obj_ref()
		if obj is None:
			return
		obj.motion_frame_mode = str(self.motionFrameModeCombo.currentData() or "active")
		self._after_change(obj)

	def _display_image_array(self, obj, auto_window=False):
		"""Create or update one plugin-local detector image object from the cached raw projection."""
		if obj.last_raw_projection is None:
			return
		image_obj = getattr(obj, "last_projection_image", None)
		if isinstance(image_obj, DetectorImage) and self._is_image_in_workspace(image_obj):
			image_obj.sync_from_virtual_xray(obj, auto_window=auto_window)
			image_obj.label = f"{obj.label}_projection"
			image_obj.source_virtual_xray_label = str(obj.label)
			AP.mainWin.dock["workspace"].refreshAll()
			self._refresh_image_viewers(image_obj)
			return

		image_obj = DetectorImage()
		image_obj.sync_from_virtual_xray(obj, auto_window=auto_window)
		obj.last_projection_image = image_obj
		AP.addObject(image_obj)
		self._refresh_image_viewers(image_obj)

	def _is_image_in_workspace(self, image_obj):
		"""Return True if image_obj is still reachable in the workspace tree."""
		if image_obj.parent is not None:
			return True
		return any(item is image_obj for item in AP.mainWin.workspace.m_data)

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

	@pyqtSlot()
	def on_open_detector_image(self):
		"""Open the current detector image in a dedicated 2D viewer window."""
		obj = self.obj_ref()
		if obj is None:
			return
		image_obj = getattr(obj, "last_projection_image", None)
		if not isinstance(image_obj, DetectorImage):
			QMessageBox.information(self, "Detector Image", "No detector image is available yet.")
			return
		child = DetectorImageViewerChild(image_obj, AP.mainWin.mdiArea)
		child.setMinimumSize(400, 300)
		sub_window = AP.mainWin.mdiArea.addSubWindow(child)
		sub_window.setWindowTitle(f"Detector Image: {image_obj.label}")
		sub_window.showNormal()
		sub_window.update()

	def _selected_cache_stage(self):
		"""Return the currently selected cache stage identifier."""
		stage = self.cacheStageCombo.currentData()
		return "raw" if stage is None else str(stage)

	def _selected_cache_format(self):
		"""Return the currently selected cache export file suffix."""
		file_format = self.cacheFormatCombo.currentData()
		return ".npy" if file_format is None else str(file_format)

	def _cache_stage_label(self, stage):
		"""Return a user-facing label for one cache stage identifier."""
		return "line integral" if str(stage).lower() in {"line", "line_integral", "integral"} else "raw detector"

	def _cache_dialog_filter(self):
		"""Return one file-dialog filter for supported projection cache formats."""
		return "Projection cache (*.npz *.npy *.txt *.csv *.tsv);;Projection package (*.npz);;NumPy (*.npy);;Text (*.txt);;CSV (*.csv);;TSV (*.tsv)"

	def on_run_simulation(self):
		"""Run one X-ray projection, cache the raw result and insert the display image into the workspace."""
		obj = self.obj_ref()
		if obj is None:
			return

		self.progressBar.setValue(0)
		self.progressBar.setVisible(True)
		self.runSimulationButton.setEnabled(False)

		# Zamroź viewery GL przed processEvents, żeby uniknąć odświeżania sceny
		# podczas blokującego przebiegu symulacji.
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
			except Exception as exc:
				QMessageBox.critical(self, "Simulation error", str(exc))
				return
			self._display_image_array(obj, auto_window=True)
			self.updateDisplayButton.setEnabled(True)
			self.openDetectorImageButton.setEnabled(isinstance(getattr(obj, "last_projection_image", None), DetectorImage))
			self.exportProjectionButton.setEnabled(True)
			self.exportSourcesButton.setEnabled(bool(getattr(obj, "last_source_projections", [])))
			projected_annotations = getattr(getattr(obj, "last_projected_annotations", None), "items", [])
			projected_on_detector = len([item for item in projected_annotations if item.in_bounds])
			self._set_render_info_text(
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
				if obj.last_raw_projection is None:
					return
			except Exception as exc:
				QMessageBox.critical(self, "Update display error", str(exc))
				return
			self._display_image_array(obj, auto_window=False)
		finally:
			QApplication.restoreOverrideCursor()
			for viewer in gl_viewers:
				viewer.setUpdatesEnabled(True)
			self.updateDisplayButton.setEnabled(obj.last_raw_projection is not None)
			self.openDetectorImageButton.setEnabled(isinstance(getattr(obj, "last_projection_image", None), DetectorImage))
			self.exportProjectionButton.setEnabled(obj.last_raw_projection is not None or getattr(obj, "last_line_integral_projection", None) is not None)
			self.exportSourcesButton.setEnabled(bool(getattr(obj, "last_source_projections", [])))
			AP.updateAllViews()

	@pyqtSlot()
	def on_export_projection_requested(self):
		"""Export the currently selected cached projection stage to a user-chosen file."""
		obj = self.obj_ref()
		if obj is None:
			return
		stage = self._selected_cache_stage()
		default_suffix = self._selected_cache_format()
		default_path = f"{obj.label}_{stage}{default_suffix}"
		file_path, _selected_filter = QFileDialog.getSaveFileName(
			self,
			f"Export {self._cache_stage_label(stage)} projection",
			default_path,
			self._cache_dialog_filter(),
		)
		if not file_path:
			return
		try:
			obj.export_cached_projection(file_path, stage=stage)
		except Exception as exc:
			QMessageBox.critical(self, "Projection export error", str(exc))
			return
		self._set_render_info_text(
			f"{obj.info()}\nExported {self._cache_stage_label(stage)} projection to:\n{file_path}"
		)

	@pyqtSlot()
	def on_import_projection_requested(self):
		"""Import one cached projection file and refresh the displayed image."""
		obj = self.obj_ref()
		if obj is None:
			return
		stage = self._selected_cache_stage()
		file_path, _selected_filter = QFileDialog.getOpenFileName(
			self,
			f"Import {self._cache_stage_label(stage)} projection",
			"",
			self._cache_dialog_filter(),
		)
		if not file_path:
			return
		gl_viewers = self._freeze_gl_viewers()
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			try:
				obj.import_cached_projection(file_path, stage=stage)
			except Exception as exc:
				QMessageBox.critical(self, "Projection import error", str(exc))
				return
			if obj.last_raw_projection is not None:
				self._display_image_array(obj, auto_window=True)
			self.updateDisplayButton.setEnabled(obj.last_raw_projection is not None)
			self.openDetectorImageButton.setEnabled(isinstance(getattr(obj, "last_projection_image", None), DetectorImage))
			self.exportProjectionButton.setEnabled(obj.last_raw_projection is not None or getattr(obj, "last_line_integral_projection", None) is not None)
			self.exportSourcesButton.setEnabled(bool(getattr(obj, "last_source_projections", [])))
			self._set_render_info_text(
				f"{obj.info()}\nImported {self._cache_stage_label(stage)} projection from:\n{file_path}"
			)
		finally:
			QApplication.restoreOverrideCursor()
			for viewer in gl_viewers:
				viewer.setUpdatesEnabled(True)
			AP.updateAllViews()

	@pyqtSlot()
	def on_export_source_projections_requested(self):
		"""Export cached per-source projection contributions into a chosen directory."""
		obj = self.obj_ref()
		if obj is None:
			return
		stage = self._selected_cache_stage()
		if not getattr(obj, "last_source_projections", []):
			QMessageBox.warning(self, "Per-source export", "No cached per-source projections are available.")
			return
		directory = QFileDialog.getExistingDirectory(
			self,
			f"Export {self._cache_stage_label(stage)} per-source projections",
			"",
		)
		if not directory:
			return
		try:
			exported_paths = obj.export_cached_source_projections(
				directory,
				stage=stage,
				file_format=self._selected_cache_format(),
			)
		except Exception as exc:
			QMessageBox.critical(self, "Per-source export error", str(exc))
			return
		self._set_render_info_text(
			f"{obj.info()}\nExported {len(exported_paths)} per-source {self._cache_stage_label(stage)} projections to:\n{directory}"
		)
