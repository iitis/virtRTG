# virtRTG Communication Compatibility Audit

Date: 2026-11-06  
Scope: inspection only; no production code changed.

## Protocol and communication path

The available interface is the plugin's browser/device-link integration, not a hardware protocol. It is a local WebSocket listener (`ws://127.0.0.1:8765` by default) carrying JSON frames. `device_link/server.py` owns the asyncio listener/thread and per-connection `DeviceAdapter`; `_receive_message()` parses each frame, acknowledges it, and emits accepted `device.pose` / `state.update` state to the plugin. `DeviceLinkIntegration` receives it through queued Qt signals, maps geometry, validates/rebuilds a selected `VirtualXRay`, refreshes the view and reports status. It deliberately does not invoke projection.

Envelope: `protocol="virtualrtg.device"`, `version=1`, strictly increasing integer `seq`, nonempty `device_id`, string `timestamp`, and `type`. Supported types: `device.hello`, `state.update`, `device.pose`, `procedure.select`, `acquisition.prepare`, `acquisition.request`, `acquisition.cancel`. Success is a JSON `response` with `in_reply_to`, `status="accepted"`, timestamp and detail explicitly saying no image was generated. Rejections return `type="error"`, `status="rejected"`, and a message; invalid frames are not applied. Each connection has isolated state.

For `state.update`, all four poses (`tube`, `detector`, `phantom`, `table`) are required, in Three.js world mm and XYZ Euler degrees (`rotation_order="XYZ"`); adapter state uses meters and radians. Generator fields: `kvp` (default 80), `ma` (200), `exposure_ms` (20), `mas` (4). Collimator: `width_cm` (24), `height_cm` (30), `light_enabled` (true). Bounds and numeric validation are enforced in `device_adapter.py`; booleans are rejected as numbers. Acquisition requests require `request_id`, matching `scene_revision`, a prepared state, and finite `kvp`, `ma`, `exposure_ms`, `mas`; the adapter records lifecycle state but generates no image. See also the external simulator's [`README.md`](../../../virtualrtg-console/README.md) and [`protocol-notes.md`](../../../virtualrtg-console/docs/protocol-notes.md), which describe this as an exploratory/mock protocol, not a frozen hardware specification.

Geometry mapping uses tube focal offset `(0,-0.43,0)` m, detector active-center offset `(0,0.04,0)` m, positions transformed by `T_RS` (mm translation), directions by its rotation, with no `T_WR`. The detector normal reversal preserves the plugin's pixel basis. Phantom pose is reported separately. The bridge applies source position, detector center/normal/up and directional rectangular field dimensions/orientation, then resets the collimation reference. Generator/exposure values and acquisition commands are not applied to a projection.

## Baseline and changes

Comparison baseline: `4c514ac806e425233e536a073a0c50e63ccc810a` (`docs(virtRTG): accept 0.6.0 architecture design`), immediately before accepted 0.6.1 commit `8b7ff41`.

No files under `device_link/` changed between that baseline and current `647e709` (0.6.4). Commits 0.6.1–0.6.4 changed the projection model/config/projector: immutable tube/collimator snapshots (0.6.1), transport extraction (0.6.2), identity detector component (0.6.3), and projector delegation to supplied tube/collimator components (0.6.4). No protocol version, command, field name/type/unit, default, adapter validation, response schema, geometry mapper, or status path changed.

Static runtime trace for a full `state.update`: server accepts and acknowledges frame; adapter stores all device fields; callback sends the state to the Qt bridge; mapper derives source/detector/beam geometry and carries collimator width/height; bridge updates the selected scene object's effective source/detector fields and directional beam, validates `build_geometry()`, rolls back on failure, and refreshes without projection. A later manual projection builds fresh config snapshots. In 0.6.4, the projector uses a supplied cone tube's focal position for ray origin and the supplied collimator's exposure mask; missing components are adapted from legacy geometry. This retains the former geometry-driven projection values for existing geometry-only callers. The 0.6.1 handoff records tube energy as informational, not an intensity control.

## Findings

- **Verified unchanged behavior:** Communication interfaces and wire formats are unchanged since baseline. No direct protocol compatibility defect found.
- **Verified unchanged behavior / scope limit:** Accepted generator/exposure and acquisition request data are stored/acknowledged, but do not control projection or produce a radiograph. This predates 0.6.1 and is explicit in both simulator documentation and adapter response text; it is not a regression from component refactoring. No correction is indicated for the documented mock contract. If an external consumer assumes `acquisition.request` triggers projection, that expectation is not satisfied; smallest future correction would be to define and implement that lifecycle explicitly, rather than alter protocol fields implicitly.
- **Verified unchanged behavior:** Component extraction preserves the effective projection call inputs by static trace and existing numerical regression coverage; DeviceLink does not call this path automatically.
- **Not verifiable with available evidence:** Live browser-to-plugin interoperability/end-to-end simulator behavior was not exercised. The external simulator is documented as exploratory, and no running simulator/device was available. Numerical projection regressions do not establish this interoperability.

## Tests

Executed with the workspace `.venv`:

`python -m pytest tests/unit/device_link tests/unit/test_xray_components.py tests/unit/test_xray_transport.py -q`  
Result: **41 passed, 10 subtests passed** (11.59 s).

An initial attempt with system Python failed during collection due to missing PyQt5 and pydicom; rerunning under the workspace environment succeeded.

## Verdict

**Not fully verified.** Static evidence supports protocol-v1 compatibility and preserved geometry-to-projection configuration behavior through 0.6.4. The focused local suite passes. End-to-end interoperability with the external browser simulator or hardware remains unverified; this repository's interface is a documented mock/device-link integration, not a hardware protocol implementation.