"""WebSocket transport and lifecycle for the browser device adapter."""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from .device_adapter import DeviceAdapter, ProtocolError
from .device_adapter import LOGGER as ADAPTER_LOGGER

LOGGER = logging.getLogger("virtualrtg.plugin")


class DeviceServer:
	"""Own one asyncio WebSocket listener on a dedicated Python thread."""

	def __init__(self, on_geometry, on_status, host="127.0.0.1", port=8765):
		self.host = host
		self.port = int(port)
		self.on_geometry = on_geometry
		self.on_status = on_status
		self._lock = threading.RLock()
		self._thread = None
		self._loop = None
		self._stop_event = None
		self._server = None
		self._state = "stopped"
		self._stop_requested = False

	@property
	def state(self):
		with self._lock:
			return self._state

	def _set_state(self, state, detail=""):
		with self._lock:
			if state == "running" and (self._stop_requested or self._state == "stopping"):
				return
			self._state = state
			self.on_status(state, detail)

	def start(self):
		with self._lock:
			if (self._state in {"starting", "running", "stopping"}
					or (self._thread is not None and self._thread.is_alive())):
				return False
			self._state = "starting"
			self._stop_requested = False
			self._thread = threading.Thread(target=self._thread_main, name="virtRTG-device-server", daemon=True)
			self.on_status("starting", f"ws://{self.host}:{self.port}")
			self._thread.start()
		return True

	def stop(self, timeout=3.0):
		stopped = False
		with self._lock:
			thread = self._thread
			loop, event = self._loop, self._stop_event
			if thread is None or not thread.is_alive():
				self._state = "stopped"
				self.on_status("stopped", "")
				stopped = True
			else:
				if self._state != "stopping":
					self._state = "stopping"
					self._stop_requested = True
					if loop is not None and event is not None:
						loop.call_soon_threadsafe(event.set)
				self.on_status("stopping", "Closing device connections")
		if stopped:
			return
		if thread is not threading.current_thread():
			thread.join(timeout)
			with self._lock:
				if self._thread is thread and thread.is_alive():
					LOGGER.warning("WebSocket server shutdown exceeded %.1f seconds", timeout)
					self.on_status("stopping", "Shutdown is still pending; the listener may remain active")

	def _thread_main(self):
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		with self._lock:
			self._loop = loop
		try:
			loop.run_until_complete(self._serve())
		except Exception as exc:
			LOGGER.exception("Device server stopped unexpectedly")
			self._set_state("error", str(exc))
		finally:
			if self._server is not None:
				self._server.close(close_connections=True)
				try:
					loop.run_until_complete(self._server.wait_closed())
				except Exception as exc:
					LOGGER.exception("Device listener failed to close cleanly")
					self._set_state("error", f"Listener shutdown failed: {exc}")
			pending = asyncio.all_tasks(loop)
			for task in pending:
				task.cancel()
			if pending:
				loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
			loop.close()
			with self._lock:
				self._loop = self._stop_event = self._server = None
				self._thread = None
				if self._state != "error":
					self._state = "stopped"
					self.on_status("stopped", "")

	async def _serve(self):
		from websockets.asyncio.server import serve

		self._stop_event = asyncio.Event()
		with self._lock:
			if self._stop_requested:
				self._stop_event.set()
		self._server = await serve(self._handle_connection, self.host, self.port, close_timeout=1)
		if self._stop_event.is_set():
			self._server.close(close_connections=True)
			await self._server.wait_closed()
			return
		self._set_state("running", f"ws://{self.host}:{self.port}")
		await self._stop_event.wait()
		self._server.close(close_connections=True)
		await self._server.wait_closed()

	async def _handle_connection(self, websocket):
		adapter = DeviceAdapter()
		LOGGER.info("Browser device connected: %s", websocket.remote_address)
		try:
			async for raw in websocket:
				message = await _receive_message(websocket, adapter, raw)
				if message is not None:
					self.on_geometry(adapter.state.to_dict())
		finally:
			LOGGER.info("Browser device disconnected: %s", websocket.remote_address)


async def _receive_message(websocket, adapter, raw, verbose=False):
	"""Validate one incoming frame, acknowledge it, and report accepted state changes."""
	message = None
	try:
		message = json.loads(raw)
		if verbose:
			ADAPTER_LOGGER.debug("Incoming JSON: %s", json.dumps(message, ensure_ascii=True, sort_keys=True))
		response = adapter.handle(message)
	except (json.JSONDecodeError, ProtocolError, TypeError, ValueError) as exc:
		kind = message.get("type", "?") if isinstance(message, dict) else "?"
		seq = message.get("seq", "?") if isinstance(message, dict) else "?"
		ADAPTER_LOGGER.warning("Rejected type=%s seq=%s reason=%s", kind, seq, exc)
		response = {"protocol": "virtualrtg.device", "version": 1, "type": "error",
			"status": "rejected", "message": str(exc)}
	await websocket.send(json.dumps(response, allow_nan=False))
	if (isinstance(message, dict) and message.get("type") in {"device.pose", "state.update"}
			and response.get("status") == "accepted"):
		return message
	return None


async def websocket_handler(websocket, adapter: DeviceAdapter, verbose=False, on_geometry=None):
	"""Handle protocol-v1 frames for one connection using its isolated adapter state."""
	adapter.verbose = verbose
	try:
		async for raw in websocket:
			message = await _receive_message(websocket, adapter, raw, verbose)
			if message is not None and on_geometry is not None:
				on_geometry(adapter.state.to_dict())
	finally:
		LOGGER.info("Browser device disconnected: %s", getattr(websocket, "remote_address", None))


async def serve(host="127.0.0.1", port=8765, verbose=False):
	"""Run the standalone transport lifecycle used by adapter regression checks."""
	from websockets.asyncio.server import serve as ws_serve
	async with ws_serve(lambda ws: websocket_handler(ws, DeviceAdapter(), verbose), host, port):
		try:
			await asyncio.Future()
		except asyncio.CancelledError:
			pass
