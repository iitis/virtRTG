# virtRTG Baseline

Status: version 0.4.0 is accepted and committed as the current implementation baseline. Full interactive GUI verification remains outstanding.

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

## Current limitations

- Only one uniquely selected `VirtualXRay` can receive an update. There is no persistent session-to-object binding, local editing lock during remote control, or multi-device session routing.
- Phantom pose is received and mapped separately but is not applied to anatomy. Table pose is received but not applied.
- Acquisition requests are validated and acknowledged without generating images.
- The browser does not stream pose updates during gizmo dragging; it sends the pose when the drag ends.
- The projection engine has no directional beam model or physical collimation. Its current cone-beam source is an ideal point source, with no independently represented source orientation.
- Device parameters and geometry synchronization are not bidirectional.
- Full browser interaction and workspace-selection behavior still require manual verification.

## Verification record

Automated and offscreen verification passed: **90 passed, 5 skipped, 10 subtests passed**. The offscreen host smoke exercised plugin loading, a protocol-v1 WebSocket update, queued geometry delivery, separate status indicators, stop, and application-shutdown cleanup. A Qt test also confirmed that the `VirtualXRay` global transform and an anatomical child's transform remain unchanged.

These checks do not confirm full interactive GUI behavior. In particular, a real browser session and mouse-driven workspace selection have not yet been verified.

## Planned work, not implemented

The following are possible future directions. They are not requirements or claims about version 0.4.0:

- Persistent ownership of a `VirtualXRay` by a remote-control session.
- Restricting local edits to device parameters while remote control owns them.
- A directional X-ray source model with independent beam orientation.
- Physical beam collimation and an exposure mask.
- Correct handling of unexposed detector pixels in raw and intensity outputs.
- Registration between the simulator phantom and anatomy.
- Potential multi-session educational operation through one server with independent `VirtualXRay` sessions.
