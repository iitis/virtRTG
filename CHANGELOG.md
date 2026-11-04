# Changelog

## 0.5.0

- Added optional `RectangularBeamGeometry` to represent the tube-derived beam axis, rectangular field orientation, collimator width and height, and 1000 mm field reference distance.
- Mapped beam positions and directions from browser station frame S into the `VirtualXRay` reference frame R using the accepted rigid-transform conventions.
- Populated and validated beam geometry through the existing Device Adapter path, including rollback on failed geometry rebuilds.
- Preserved protocol v1 and existing projection behavior; the beam geometry does not mask or suppress generated projection rays.

## 0.4.0

- Added an optional browser Device Server, controlled from the virtRTG plugin menu and disabled by default.
- Integrated the protocol-v1 Device Adapter and geometry mapper into the plugin.
- Added synchronization of source and detector geometry to the uniquely selected `VirtualXRay`, with geometry validation and rollback on rebuild failure.
- Routed accepted geometry updates to the scene through queued Qt signals.
- Added automatic server cleanup on plugin unload and application shutdown.
- Added separate menu status indicators for server lifecycle and geometry synchronization.
