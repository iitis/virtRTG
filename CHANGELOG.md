# Changelog

## 0.6.4

- Integrated `XRayTube` as the N=1 cone-beam focal source and `XRayCollimator`
  as the owner of the vectorized ideal rectangular exposure mask.
- Preserved geometry-only projector compatibility and existing projection
  results; verified all 79 frozen reference arrays exactly.

## 0.6.3

- Added an ideal `XRayDetector` response component for total intensity-mode
  output, preserving the existing signal array without copying.
- Kept integral outputs and per-source diagnostics outside detector response;
  projection geometry, legacy physics conversion, capture behavior, GUI, and
  serialization remain unchanged.
- Verified exact equivalence across all 79 frozen original and supplementary
  reference arrays; no reference data was regenerated.

## 0.6.2

- Extracted prepared-ray primary attenuation integration into `XRayTransportModel`, preserving the existing sampling, Siddon, and mesh integration backends and legacy projection outputs.
- Verified exact equivalence across all 79 frozen original and supplementary reference arrays; no reference data was regenerated.

## 0.6.1

- Added transient `FocalSpotSamples`, `XRayTube`, and `XRayCollimator` configuration snapshots while keeping `VirtualXRay` as the geometry and serialization authority.
- Adapted the ideal angular collimator snapshot to the existing projection geometry without changing ray generation, exposure-mask, or projection kernels.
- Added regression coverage for snapshot consistency, point-source configuration, collimator-only roll, and parallel-mode compatibility.

## 0.6.0

- Completed and accepted the physical-model architecture audit and staged
  0.6.x migration design, including explicit boundaries for acquisition
  configuration, transport, detector response, diagnostics, and presentation.
- Documented preservation requirements for existing research workflows and legacy `integral`/`intensity` output semantics.
- Documentation/design-only release; no new simulation functionality or physical-model implementation was introduced.

## 0.5.2

- Added standalone rectangular collimation controls for field size, reference distance, horizontal and vertical beam steering, and beam roll.
- Added explicit alignment of the beam frame to the detector, with zero-relative-angle reset.
- Kept angle edits stable across repeated changes, projection, ordinary GUI refreshes, and properties-panel recreation; external geometry replacement and scene loading rebase the controls explicitly.
- Preserved the effective rolled and steered beam frame in scenes while keeping relative angle controls session-only.
- Manually verified standalone collimation controls, H/V steering, beam roll, and projection behavior.

## 0.5.1

- Added opt-in physical rectangular collimation for cone-beam projections, controlled by `VirtualXRay.physical_collimation_enabled`; directional beam geometry alone remains descriptive unless enabled.
- Generated an acquisition-specific exposure mask from the independent directional beam frame and skip attenuation integration for rays outside the field.
- Preserved unexposed-pixel primary-signal semantics through physics conversion, projection caching, DetectorImage presentation, and the active NPZ package workflow.
- Persisted directional beam geometry and collimation mode in scene files, preserving compatibility with legacy scenes and NPZ packages without masks.
- Kept parallel-beam projection unrestricted; enabled physical collimation in parallel mode is rejected explicitly.

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
