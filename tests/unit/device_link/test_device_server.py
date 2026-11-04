"""Transport lifecycle and plugin bridge regressions for device link 0.4.0."""
from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import threading
import time

import pytest
from PyQt5.QtWidgets import QApplication

from plugins.virtRTG.device_link.integration import DeviceLinkIntegration
from plugins.virtRTG.device_link.server import DeviceServer


def free_port():
	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[1]


def wait_for(predicate, timeout=3):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		if predicate():
			return True
		time.sleep(0.01)
	return False


@pytest.fixture
def qapp():
	os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
	app = QApplication.instance() or QApplication([])
	return app


def wait_qt(app, predicate, timeout=2):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		app.processEvents()
		if predicate():
			return True
		time.sleep(0.005)
	return False


def state_message():
	poses = {
		"tube": ((0, 2350, 0), (0, 0, 0)),
		"detector": ((0, 770, 0), (0, 0, 0)),
		"phantom": ((0, 1100, 0), (0, 0, 0)),
		"table": ((0, 835, 0), (0, 0, 0)),
	}
	return {"protocol": "virtualrtg.device", "version": 1, "type": "state.update", "seq": 1,
		"device_id": "test", "timestamp": "2026-11-04T00:00:00Z", "revision": 1,
		"device": {"poses": {key: {"position_mm": list(p), "rotation_deg": list(r), "rotation_order": "XYZ"}
											 for key, (p, r) in poses.items()},
				"generator": {"kvp": 80, "ma": 200, "exposure_ms": 20, "mas": 4},
				"collimator": {"width_cm": 24, "height_cm": 30, "light_enabled": True}}}


def test_server_disabled_until_started_and_repeated_start_stop_releases_port():
	port = free_port()
	server = DeviceServer(lambda _state: None, lambda _state, _detail: None, port=port)
	assert server.state == "stopped"
	for _ in range(2):
		assert server.start()
		assert wait_for(lambda: server.state == "running")
		assert not server.start()
		server.stop()
		server.stop()
		assert wait_for(lambda: server.state == "stopped")
		with socket.socket() as sock:
			sock.bind(("127.0.0.1", port))


def test_port_conflict_is_reported_without_raising_to_caller():
	with socket.socket() as blocker:
		blocker.bind(("127.0.0.1", 0))
		blocker.listen()
		port = blocker.getsockname()[1]
		statuses = []
		server = DeviceServer(lambda _state: None, lambda state, detail: statuses.append((state, detail)), port=port)
		assert server.start()
		assert wait_for(lambda: server.state == "error")
		assert any(state == "error" for state, _ in statuses)
		server.stop()


def test_stop_requested_during_startup_finishes_cleanly():
	server = DeviceServer(lambda _state: None, lambda *_args: None, port=free_port())
	assert server.start()
	server.stop()
	assert wait_for(lambda: server.state == "stopped")


def test_application_about_to_quit_stops_the_listener(qapp):
	server = DeviceServer(lambda _state: None, lambda *_args: None, port=free_port())
	qapp.aboutToQuit.connect(server.stop)
	server.start()
	assert wait_for(lambda: server.state == "running")
	qapp.aboutToQuit.emit()
	assert wait_for(lambda: server.state == "stopped")
	qapp.aboutToQuit.disconnect(server.stop)


def test_stop_closes_an_active_websocket_connection():
	websockets = pytest.importorskip("websockets.asyncio.client")
	from websockets.exceptions import ConnectionClosed
	port = free_port()
	server = DeviceServer(lambda _state: None, lambda *_args: None, port=port)
	server.start()
	assert wait_for(lambda: server.state == "running")

	async def connect_and_stop():
		connection = await websockets.connect(f"ws://127.0.0.1:{port}")
		server.stop()
		with pytest.raises(ConnectionClosed):
			await asyncio.wait_for(connection.recv(), 2)
		await connection.close()

	asyncio.run(connect_and_stop())
	assert server.state == "stopped"


def test_slow_worker_keeps_stopping_state_and_blocks_restart_until_listener_closes():
	websockets = pytest.importorskip("websockets.asyncio.client")
	from websockets.exceptions import ConnectionClosed
	port = free_port()
	entered = threading.Event()
	release = threading.Event()
	def delay_geometry(_state):
		entered.set()
		release.wait(3)
	server = DeviceServer(delay_geometry, lambda *_args: None, port=port)
	server.start()
	assert wait_for(lambda: server.state == "running")

	async def connect_and_check():
		connection = await websockets.connect(f"ws://127.0.0.1:{port}")
		await connection.send(json.dumps(state_message()))
		assert json.loads(await connection.recv())["status"] == "accepted"
		assert entered.wait(2)
		server.stop(timeout=0.01)
		assert server.state == "stopping"
		assert not server.start()
		with socket.socket() as sock:
			with pytest.raises(OSError):
				sock.bind(("127.0.0.1", port))
		release.set()
		with pytest.raises(ConnectionClosed):
			await asyncio.wait_for(connection.recv(), 2)
		await connection.close()

	try:
		asyncio.run(connect_and_check())
		assert wait_for(lambda: server.state == "stopped")
		with socket.socket() as sock:
			sock.bind(("127.0.0.1", port))
	finally:
		release.set()
		server.stop()


def test_websocket_acceptance_reaches_integration_callback():
	websockets = pytest.importorskip("websockets.asyncio.client")
	port = free_port()
	received = []
	event = threading.Event()
	server = DeviceServer(lambda state: (received.append(state), event.set()), lambda *_args: None, port=port)
	server.start()
	assert wait_for(lambda: server.state == "running")

	async def exchange():
		async with websockets.connect(f"ws://127.0.0.1:{port}") as connection:
			await connection.send(json.dumps(state_message()))
			return json.loads(await connection.recv())

	try:
		response = asyncio.run(exchange())
		assert response["status"] == "accepted"
		assert response["detail"].endswith("no image generated")
		assert event.wait(2)
		assert received[0]["revision"] == 1
	finally:
		server.stop()


class FakeVirtualXRay:
	def __init__(self):
		import numpy as np
		self.source_position_ref = np.array([1, 2, 3], dtype=np.float32)
		self.detector_center_ref = np.array([4, 5, 6], dtype=np.float32)
		self.detector_normal_ref = np.array([0, 0, -1], dtype=np.float32)
		self.detector_up_ref = np.array([0, 1, 0], dtype=np.float32)
		self.global_transform = "unchanged"
		self.anatomy_transform = "unchanged"
		self.builds = 0
		self.label = "selected"

	def build_geometry(self):
		self.builds += 1


class FakePlugin:
	def __init__(self, selected):
		self.selected = selected
		self.status = []
		self.refreshes = []

	def selected_virtual_xray(self):
		return self.selected

	def refresh_virtual_xray(self, obj):
		self.refreshes.append(obj)

	def _on_device_server_status(self, state, detail):
		self.status.append((state, detail))


def test_qt_bridge_applies_only_geometry_and_missing_target_is_nonfatal(qapp):
	obj = FakeVirtualXRay()
	plugin = FakePlugin(obj)
	bridge = DeviceLinkIntegration(plugin)
	before_global, before_anatomy = obj.global_transform, obj.anatomy_transform
	bridge.signals.geometry_received.emit({"poses": {
		"tube": {"position_m": [0, 2.35, 0], "rotation_rad": [0, 0, 0]},
		"detector": {"position_m": [0, .77, 0], "rotation_rad": [0, 0, 0]},
		"phantom": {"position_m": [0, 1.1, 0], "rotation_rad": [0, 0, 0]},
		"table": {"position_m": [0, .835, 0], "rotation_rad": [0, 0, 0]},
	}})
	assert wait_qt(qapp, lambda: bool(plugin.refreshes))
	assert obj.builds == 1
	assert plugin.refreshes == [obj]
	assert obj.global_transform == before_global
	assert obj.anatomy_transform == before_anatomy
	assert plugin.status[-1][0] == "geometry-applied"

	plugin.selected = None
	bridge.signals.geometry_received.emit({"poses": {
		"tube": {"position_m": [0, 2.35, 0], "rotation_rad": [0, 0, 0]},
		"detector": {"position_m": [0, .77, 0], "rotation_rad": [0, 0, 0]},
		"phantom": {"position_m": [0, 1.1, 0], "rotation_rad": [0, 0, 0]},
		"table": {"position_m": [0, .835, 0], "rotation_rad": [0, 0, 0]},
	}})
	assert wait_qt(qapp, lambda: plugin.status[-1][0] == "no-target")
	assert len(plugin.refreshes) == 1
	bridge.disconnect()


def test_invalid_geometry_is_not_applied(qapp):
	obj = FakeVirtualXRay()
	plugin = FakePlugin(obj)
	bridge = DeviceLinkIntegration(plugin)
	before = obj.source_position_ref.copy()
	payload = state_message()["device"]
	payload["poses"]["tube"]["position_mm"] = [0, 1240, 0]
	from plugins.virtRTG.device_link.device_adapter import DeviceAdapter
	adapter = DeviceAdapter()
	adapter.handle({key: value for key, value in state_message().items() if key != "device"} | {"type": "state.update", "revision": 1, "device": payload})
	bridge.signals.geometry_received.emit(adapter.state.to_dict())
	assert wait_qt(qapp, lambda: bool(plugin.status))
	assert plugin.status[-1][0] == "invalid-geometry"
	assert obj.source_position_ref.tolist() == before.tolist()
	assert not plugin.refreshes
	bridge.disconnect()


def test_selected_virtual_xray_and_anatomical_child_transforms_are_preserved(qapp):
	from dpVision import Transform
	from plugins.virtRTG.virtualXRay import VirtualXRay

	virtual_xray = VirtualXRay()
	anatomy = Transform()
	anatomy.translate(13, -7, 29)
	virtual_xray.addChild(anatomy)
	plugin = FakePlugin(virtual_xray)
	bridge = DeviceLinkIntegration(plugin)
	global_before = virtual_xray.getGlobalTransformation().copy()
	anatomy_before = anatomy.getGlobalTransformation().copy()
	bridge.signals.geometry_received.emit({"poses": {
		"tube": {"position_m": [0, 2.35, 0], "rotation_rad": [0, 0, 0]},
		"detector": {"position_m": [0, .77, 0], "rotation_rad": [0, 0, 0]},
		"phantom": {"position_m": [0, 1.1, 0], "rotation_rad": [0, 0, 0]},
		"table": {"position_m": [0, .835, 0], "rotation_rad": [0, 0, 0]},
	}})
	assert wait_qt(qapp, lambda: bool(plugin.refreshes))
	assert plugin.refreshes == [virtual_xray]
	assert np_allclose(virtual_xray.getGlobalTransformation(), global_before)
	assert np_allclose(anatomy.getGlobalTransformation(), anatomy_before)
	assert virtual_xray.last_raw_projection is None
	bridge.disconnect()


def test_failed_geometry_rebuild_restores_inputs_and_build_backfills(qapp, monkeypatch):
	from plugins.virtRTG.virtualXRay import VirtualXRay
	from plugins.virtRTG.xray.xrayProjection import XRayProjectionGeometry

	virtual_xray = VirtualXRay()
	virtual_xray.depth_window_mode = "invalid-test-mode"
	for name in ("depth_window_mm", "depth_window_origin_ref", "depth_window_axis_ref", "motion_frame_mode"):
		delattr(virtual_xray, name)
	plugin = FakePlugin(virtual_xray)
	bridge = DeviceLinkIntegration(plugin)
	attrs = ("source_position_ref", "detector_center_ref", "detector_normal_ref", "detector_up_ref")
	before = {name: getattr(virtual_xray, name).copy() for name in attrs}
	monkeypatch.setattr(XRayProjectionGeometry, "from_detector_pose",
		lambda **_kwargs: (_ for _ in ()).throw(ValueError("rebuild failed after defaults were backfilled")))
	bridge.signals.geometry_received.emit({"poses": {
		"tube": {"position_m": [0, 2.35, 0], "rotation_rad": [0, 0, 0]},
		"detector": {"position_m": [0, .77, 0], "rotation_rad": [0, 0, 0]},
		"phantom": {"position_m": [0, 1.1, 0], "rotation_rad": [0, 0, 0]},
		"table": {"position_m": [0, .835, 0], "rotation_rad": [0, 0, 0]},
	}})
	assert wait_qt(qapp, lambda: bool(plugin.status))
	assert plugin.status[-1][0] == "invalid-geometry"
	for name, value in before.items():
		assert np_allclose(getattr(virtual_xray, name), value)
	assert virtual_xray.depth_window_mode == "invalid-test-mode"
	for name in ("depth_window_mm", "depth_window_origin_ref", "depth_window_axis_ref", "motion_frame_mode"):
		assert not hasattr(virtual_xray, name)
	assert not plugin.refreshes
	bridge.disconnect()


def test_directional_beam_device_update_rotation_rollback_and_no_projection(qapp, monkeypatch):
	from plugins.virtRTG.device_link.device_adapter import DeviceState
	from plugins.virtRTG.virtualXRay import VirtualXRay
	from plugins.virtRTG.xray.xrayProjection import XRayProjectionGeometry

	virtual_xray = VirtualXRay()
	plugin = FakePlugin(virtual_xray)
	bridge = DeviceLinkIntegration(plugin)
	projection_calls = []
	monkeypatch.setattr(virtual_xray, "project_and_cache", lambda *args, **kwargs: projection_calls.append((args, kwargs)))

	state = DeviceState()
	bridge.signals.geometry_received.emit(state.to_dict())
	assert wait_qt(qapp, lambda: len(plugin.refreshes) == 1)
	initial_beam = virtual_xray.directional_beam_geometry
	assert initial_beam is not None
	assert initial_beam.beam_axis_ref == (0.0, -1.0, 0.0)
	assert initial_beam.field_width_cm == state.collimator.width_cm
	assert initial_beam.field_height_cm == state.collimator.height_cm
	initial_detector = tuple(getattr(virtual_xray, name).copy() for name in (
		"detector_center_ref", "detector_normal_ref", "detector_up_ref"))

	state.poses["tube"].rotation_rad = (0.0, 0.0, math.pi / 2.0)
	bridge.signals.geometry_received.emit(state.to_dict())
	assert wait_qt(qapp, lambda: len(plugin.refreshes) == 2)
	rotated_beam = virtual_xray.directional_beam_geometry
	assert rotated_beam.beam_axis_ref[0] == pytest.approx(1.0)
	assert rotated_beam.beam_axis_ref[1] == pytest.approx(0.0, abs=1e-7)
	for name, expected in zip(("detector_center_ref", "detector_normal_ref", "detector_up_ref"), initial_detector):
		assert np_allclose(getattr(virtual_xray, name), expected)
	assert virtual_xray.last_raw_projection is None
	assert not projection_calls

	attrs = ("source_position_ref", "detector_center_ref", "detector_normal_ref", "detector_up_ref")
	before_failure = {name: getattr(virtual_xray, name).copy() for name in attrs}
	beam_before_failure = virtual_xray.directional_beam_geometry
	refresh_count = len(plugin.refreshes)
	monkeypatch.setattr(XRayProjectionGeometry, "from_detector_pose",
		lambda **_kwargs: (_ for _ in ()).throw(ValueError("rebuild failed")))
	state.poses["tube"].rotation_rad = (0.0, 0.0, math.pi)
	bridge.signals.geometry_received.emit(state.to_dict())
	assert wait_qt(qapp, lambda: plugin.status[-1][0] == "invalid-geometry")
	for name, expected in before_failure.items():
		assert np_allclose(getattr(virtual_xray, name), expected)
	assert virtual_xray.directional_beam_geometry is beam_before_failure
	assert len(plugin.refreshes) == refresh_count
	assert virtual_xray.last_raw_projection is None
	assert not projection_calls
	bridge.disconnect()


def test_server_and_geometry_statuses_are_independent(qapp):
	from PyQt5.QtWidgets import QAction
	from plugins.virtRTG.pluginMain import VirtualRTG

	plugin = VirtualRTG.__new__(VirtualRTG)
	plugin._device_server_status_action = QAction("Server status: disabled")
	plugin._device_geometry_status_action = QAction("Device geometry: waiting")
	plugin._on_device_server_status("running", "ws://127.0.0.1:8765")
	running = plugin._device_server_status_action.text()
	plugin._on_device_server_status("geometry-applied", "Updated VirtualXRay")
	assert plugin._device_server_status_action.text() == running
	assert "geometry applied" in plugin._device_geometry_status_action.text()
	plugin._on_device_server_status("stopped")
	assert plugin._device_server_status_action.text() == "Server status: disabled"
	assert "geometry applied" in plugin._device_geometry_status_action.text()


def np_allclose(left, right):
	import numpy as np
	return bool(np.allclose(left, right))
