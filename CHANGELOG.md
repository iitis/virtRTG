# Changelog

## 0.4.0

- Added an optional browser Device Server, controlled from the virtRTG plugin menu and disabled by default.
- Integrated the protocol-v1 Device Adapter and geometry mapper into the plugin.
- Added synchronization of source and detector geometry to the uniquely selected `VirtualXRay`, with geometry validation and rollback on rebuild failure.
- Routed accepted geometry updates to the scene through queued Qt signals.
- Added automatic server cleanup on plugin unload and application shutdown.
- Added separate menu status indicators for server lifecycle and geometry synchronization.
