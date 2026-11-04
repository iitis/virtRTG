# virtRTG Baseline

Status: version 0.5.0 - Directional X-ray Beam Geometry is accepted. Full interactive GUI verification remains outstanding.

## Accepted 0.4.0 architecture

- The browser Device Adapter was migrated from `virtualrtg-console` version 0.3.1 into `device_link/` in the virtRTG plugin. The plugin does not depend at runtime on that separate checkout.
- Protocol v1 validation and per-connection `DeviceState` remain the source of accepted device state.
- WebSocket transport runs an asyncio server on a dedicated thread. The Device Server is disabled by default and controlled by the virtRTG plugin menu. It listens on `127.0.0.1:8765`.
- The menu has separate, non-interactive status indicators for server lifecycle and geometry synchronization.
- Plugin unload and application shutdown initiate server cleanup. Shutdown remains in progress until the worker and listener finish closing.
- Accepted geometry updates reach the GUI through queued Qt signals.
- At update time, the uniquely selected `VirtualXRay` receives source position, detector center, detector normal, and detector up vectors. Invalid geometry is not applied.
- Mapping retains the accepted browser XYZ Euler and unit conventions, maps through `T_RS` into the VirtualXRay reference frame, keeps `T_WR` independent, and explicitly reverses the detector normal to preserve the detector pixel basis.
- Applying device geometry does not change the `VirtualXRay` global placement or anatomical child transforms. It does not automatically run a projection.

## Accepted 0.5.0 directional beam geometry

- `VirtualXRay.directional_beam_geometry` is optional and defaults to `None` for legacy scenes. Device updates populate it with a validated `RectangularBeamGeometry` containing the beam axis, horizontal and vertical transverse axes, field width and height, and field reference distance.
- Current demonstrator tube construction geometry is centralized in the mapper: focal spot offset `(0,-0.43,0)` meters, emission axis `(0,-1,0)`, beam horizontal `(1,0,0)`, and beam vertical `(0,0,1)`, all in tube-local coordinates.
- The rectangular field dimensions use centimeters and refer to a plane 1000 mm from the focal spot. The field diverges from the ideal point source; containment predicates are available but are not connected to projection generation.
- Device pose mapping applies the accepted Three.js Euler XYZ rotation to local positions and directions. The full rigid `T_RS` transform maps positions from station frame S to VirtualXRay reference frame R, while only its rotational part maps directions. `T_WR` is not applied. Beam direction comes from the tube and is not inferred from the detector.
- The GUI-thread Device Adapter applies the beam alongside source and detector geometry. If geometry rebuilding fails, it restores the previous beam and source/detector fields. Protocol v1 and the existing detector normal reversal remain unchanged.
- Legacy projection behavior is preserved: `build_geometry()` and all projection algorithms ignore the optional beam value, and device updates do not trigger projection.

## Current limitations

- Only one uniquely selected `VirtualXRay` can receive an update. There is no persistent session-to-object binding, local editing lock during remote control, or multi-device session routing.
- Phantom pose is received and mapped separately but is not applied to anatomy. Table pose is received but not applied.
- Acquisition requests are validated and acknowledged without generating images.
- The browser does not stream pose updates during gizmo dragging; it sends the pose when the drag ends.
- Directional beam geometry is not yet connected to the projection engine. Generated pixels are not masked or otherwise restricted by the rectangular field.
- Device parameters and geometry synchronization are not bidirectional.
- Full browser interaction and workspace-selection behavior still require manual verification.

## Verification record

Version 0.5.0 verification passed: **93 passed, 5 skipped, 10 subtests passed** in the virtRTG unit suite. The Qt integration regression exercises actual `VirtualXRay` and `DeviceLinkIntegration` updates, tube rotation, preserved detector mapping, rollback after a failed geometry rebuild, and no automatic projection. Ruff F401 findings in `virtualXRay.py` and `xray/xrayProjection.py` were confirmed unchanged from `HEAD`; device-link and test modules passed F401 checks. `git diff --check` passed.

The earlier 0.4.0 offscreen host smoke exercised plugin loading, a protocol-v1 WebSocket update, queued geometry delivery, separate status indicators, stop, and application-shutdown cleanup. A Qt test also confirmed that the `VirtualXRay` global transform and an anatomical child's transform remain unchanged.

These checks do not confirm full interactive GUI behavior. In particular, a real browser session and mouse-driven workspace selection have not yet been verified.

## Planned work, not implemented

The following are possible future directions. They are not requirements or claims about version 0.4.0:

- Persistent ownership of a `VirtualXRay` by a remote-control session.
- Restricting local edits to device parameters while remote control owns them.
- Physical beam collimation and an exposure mask.
- Correct handling of unexposed detector pixels in raw and intensity outputs.
- Registration between the simulator phantom and anatomy.
- Potential multi-session educational operation through one server with independent `VirtualXRay` sessions.
