# virtRTG Device Simulator Integration 0.4.0

## Files changed

- `device_link/device_adapter.py`: transferred protocol-v1 validation, state dataclasses, acknowledgement handling, projection references, and diagnostics from `D:/current/virtualrtg-console/device_adapter.py` (accepted 0.3.1).
- `device_link/virtualrtg_geometry.py`: transferred the T_RS mapping and geometry consistency report from `D:/current/virtualrtg-console/virtualrtg_geometry.py` (accepted 0.3.1).
- `device_link/server.py`: WebSocket protocol transport, connection state, threaded asyncio listener, and start/stop lifecycle.
- `device_link/integration.py`: queued Qt bridge and VirtualXRay geometry application.
- `device_link/__init__.py`: internal package marker.
- `pluginMain.py`: server menu actions/status, selected-object lookup, application-exit/unload cleanup, and view refresh.
- `requirements.txt`: added `websockets>=14.0` as an optional plugin dependency.
- `tests/unit/device_link/test_device_adapter.py` and `test_virtualrtg_geometry.py`: transferred 0.3.1 regression tests and changed their imports to the plugin package; patched the geometry diagnostic mock target to its new package path.
- `tests/unit/device_link/test_device_server.py`: server lifecycle, protocol-to-integration, Qt dispatch, geometry safety, and scene preservation regressions.

## Pre-acceptance corrections

- The plugin menu now has independent disabled status rows for server lifecycle and the last geometry synchronization result. Geometry status cannot replace running, stopping, stopped, or bind-error status.
- `VirtualXRay.build_geometry()` was inspected. It returns a newly built `XRayProjectionGeometry` and does not store a derived geometry cache. Its persistent side effect is adding legacy depth-window defaults. If building fails, the integration restores the four incoming geometry fields and removes any defaults introduced during that failed attempt; existing defaults remain intact.
- Server shutdown remains `stopping` until the listener worker actually exits. A timeout reports that shutdown is still pending, and start checks both lifecycle state and worker liveness. Cleanup closes and awaits any remaining WebSocket listener before the thread exits, including exceptional cleanup paths.

## Migration changes

Adapter and mapper behavior was transferred without changing protocol or numerical geometry rules. The mapper imports the adapter with a package-relative import, and optional verbose diagnostics now import the mapper the same way. WebSocket handler and standalone lifecycle functions were placed in `server.py` to keep protocol/state, mapping, transport, Qt, and scene integration in separate modules. The adapter's diagnostic logger retains its original name for the copied logging expectations.

The source repository's `mock_server.py` and browser app remain in `virtualrtg-console`; the plugin has no runtime import or path dependency on that checkout. The browser continues to connect to `ws://127.0.0.1:8765`.

## Threading and lifecycle

The server is disabled until **Start Device Server** is selected. An asyncio WebSocket server runs on one daemon Python thread. It creates a fresh `DeviceAdapter` for each connection. The listener binds to `127.0.0.1:8765`; repeated starts are ignored while active. Stop closes connected clients and the listener; repeated stops are harmless. If shutdown exceeds its wait timeout, status remains stopping and restart stays blocked until the listener thread ends. Bind errors become an error status instead of escaping into the GUI. The menu shows server lifecycle separately from geometry synchronization.

The server emits accepted geometry snapshots through a Qt signal explicitly queued to the GUI thread. On unload, the plugin marks the bridge inactive, disconnects its signals, and stops the server. `QApplication.aboutToQuit` also stops it if plugin unload was not called.

## Active object and geometry safety

Every accepted `state.update` or `device.pose` is mapped through the transferred `map_device_geometry()`. The target is the uniquely selected `VirtualXRay` object at delivery time. With no unique selected VirtualXRay, the adapter state remains accepted and the UI status explains that an object must be selected.

The mapper preserves browser XYZ Euler composition, mm/degrees to m/radians adapter conversion, mm reference geometry, explicit detector-normal reversal, and independent T_RS/T_WR frames. The separately mapped phantom pose is not applied. Before changing the four reference geometry fields, the bridge rejects inconsistent/non-finite mappings. It then builds the object's existing geometry to validate the candidate; on failure it restores the previous values and any defaults backfilled during the failed build. `build_geometry()` returns a fresh geometry object and does not persist derived geometry. Successful changes call `AP.updateAllViews()`. The object's global transform and anatomical children are not modified.

Acquisition prepare/request/cancel validation and protocol acknowledgements remain in the adapter. An acknowledgement explicitly states that no image was generated. No projection is run from geometry updates or acquisition requests.

## Verification

- `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest plugins/virtRTG/tests -q`: **90 passed, 5 skipped, 10 subtests passed**.
- `python -m compileall -q plugins/virtRTG/device_link plugins/virtRTG/pluginMain.py`: passed.
- `git diff --check` in the virtRTG plugin repository: passed.
- Offscreen host-window smoke test loaded the plugin, confirmed the server initially stopped, sent a protocol-v1 state update over WebSocket, and received an accepted acknowledgement. It confirmed geometry status did not overwrite running server status, no projection cache was created, Stop completed, and `aboutToQuit` cleanup stopped a restarted listener.
- A separate Qt test applies geometry to a real `VirtualXRay` instance and confirms its global transformation and an anatomical child's global transformation are unchanged.
- Regression tests cover a rebuild exception after legacy defaults are backfilled, a connected client, a worker blocked past the stop timeout, blocked restart while that worker/listener is active, and port release after completion.

## Manual verification

1. Load the Virtual RTG plugin and confirm **Server status: disabled** is visible and disabled as a status item.
2. Create a VirtualXRay object and select exactly that object in the workspace.
3. Start the device server. Confirm the menu reports it running at `127.0.0.1:8765`.
4. Open the browser simulator and connect it to `ws://127.0.0.1:8765`. Move the tube or detector and inspect the VirtualXRay gizmo geometry.
5. Confirm no projection is generated. Stop the server, verify the browser disconnects, and start it again to confirm the port was released.
6. Unload the plugin while the server is running and repeat with application exit; both paths should close the connection and release the port.

## Known limitations

- One uniquely selected VirtualXRay is supported. There is no multi-device routing or persistent cross-connection device state.
- The server binds only to loopback and uses the fixed default port 8765.
- Device acquisition remains diagnostic and does not run a projection engine.
- The headless end-to-end smoke supplied the selected VirtualXRay through the plugin's selection lookup. It did not exercise a real browser or mouse-driven workspace selection; the full workspace selection workflow remains a manual check.
