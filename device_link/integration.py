"""GUI-thread bridge between accepted device state and one selected VirtualXRay."""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from .device_adapter import DeviceState
from .virtualrtg_geometry import map_device_geometry
from ..xray.xrayProjection import RectangularBeamGeometry


_BUILD_GEOMETRY_BACKFILLS = (
	"depth_window_mode", "depth_window_mm", "depth_window_origin_ref",
	"depth_window_axis_ref", "motion_frame_mode",
)
# VirtualXRay.build_geometry() returns a new geometry value and does not cache
# one on the scene object; its only current persistent side effect is backfill.


def _snapshot_build_geometry_backfills(virtual_xray):
	"""Capture the legacy defaults that build_geometry may add before it fails."""
	return {
		name: (hasattr(virtual_xray, name), getattr(virtual_xray, name, None))
		for name in _BUILD_GEOMETRY_BACKFILLS
	}


def _restore_build_geometry_backfills(virtual_xray, snapshot):
	"""Undo only defaults introduced by a failed geometry rebuild."""
	for name, (existed, value) in snapshot.items():
		if existed:
			setattr(virtual_xray, name, value)
		elif hasattr(virtual_xray, name):
			delattr(virtual_xray, name)


class DeviceLinkSignals(QObject):
	"""Signals emitted by the transport thread and consumed by the GUI thread."""
	geometry_received = pyqtSignal(object)
	status_changed = pyqtSignal(str, str)


class DeviceLinkIntegration(QObject):
	"""Apply accepted geometry only to the uniquely selected VirtualXRay object."""

	def __init__(self, plugin, parent=None):
		super().__init__(parent)
		self.plugin = plugin
		self.active = True
		self.signals = DeviceLinkSignals(self)
		self.signals.geometry_received.connect(self._apply_geometry, Qt.QueuedConnection)
		self.signals.status_changed.connect(plugin._on_device_server_status, Qt.QueuedConnection)

	def disconnect(self):
		self.active = False
		for signal, slot in ((self.signals.geometry_received, self._apply_geometry),
							 (self.signals.status_changed, self.plugin._on_device_server_status)):
			try:
				signal.disconnect(slot)
			except (TypeError, RuntimeError):
				pass

	@pyqtSlot(object)
	def _apply_geometry(self, payload):
		if not self.active:
			return
		try:
			state = DeviceState.from_dict(payload)
			geometry = map_device_geometry(state)
			if not geometry.get("geometry_consistent"):
				self.plugin._on_device_server_status("invalid-geometry", "Accepted state retained; geometry is inconsistent")
				return
			vectors = [geometry[key] for key in (
				"source_position_ref", "detector_center_ref", "detector_normal_ref", "detector_up_ref")]
			if any(len(vector) != 3 or not all(math.isfinite(float(value)) for value in vector) for vector in vectors):
				raise ValueError("geometry contains non-finite or malformed vectors")
		except (KeyError, TypeError, ValueError) as exc:
			self.plugin._on_device_server_status("invalid-geometry", f"Accepted state retained; geometry was not applied: {exc}")
			return

		virtual_xray = self.plugin.selected_virtual_xray()
		if virtual_xray is None:
			self.plugin._on_device_server_status("no-target", "Accepted device state retained; select one VirtualXRay object")
			return

		attributes = ("source_position_ref", "detector_center_ref", "detector_normal_ref", "detector_up_ref")
		previous = {name: getattr(virtual_xray, name).copy() for name in attributes}
		previous_beam = getattr(virtual_xray, "directional_beam_geometry", None)
		backfills = _snapshot_build_geometry_backfills(virtual_xray)
		try:
			for name in attributes:
				setattr(virtual_xray, name, np.asarray(geometry[name], dtype=np.float32))
			virtual_xray.directional_beam_geometry = RectangularBeamGeometry(
				geometry["beam_axis_ref"], geometry["beam_horizontal_ref"], geometry["beam_vertical_ref"],
				geometry["field_width_cm"], geometry["field_height_cm"], geometry["field_reference_distance_mm"])
			virtual_xray.build_geometry()
		except Exception as exc:
			for name, value in previous.items():
				setattr(virtual_xray, name, value)
			virtual_xray.directional_beam_geometry = previous_beam
			_restore_build_geometry_backfills(virtual_xray, backfills)
			self.plugin._on_device_server_status("invalid-geometry", f"Accepted state retained; VirtualXRay geometry rejected: {exc}")
			return
		self.plugin.refresh_virtual_xray(virtual_xray)
		self.plugin._on_device_server_status("geometry-applied", f"Updated {virtual_xray.label or 'VirtualXRay'}; no projection requested")
