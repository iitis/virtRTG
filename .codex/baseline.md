# virtRTG Baseline

Status: the version 0.6.0 architecture design and the narrowly scoped 0.6.1
tube/collimator configuration implementation are accepted. Version 0.6.1 adds
configuration snapshots while preserving the existing projection kernels,
scene serialization, GUI, and simulator protocol. Full manual
browser-to-plugin GUI validation remains outstanding.

## Accepted 0.6.1 tube and collimator configuration

- `VirtualXRay` remains the authority for editable and serialized settings.
  `build_projection_config()` creates one `XRayTube` snapshot and one
  `XRayCollimator` snapshot. The exact collimator snapshot is adapted into
  `XRayProjectionGeometry` and retained on `XRayProjectionConfig`; public
  `build_geometry()` creates and passes a fresh snapshot.
- `FocalSpotSamples` validates finite `(N,3)` positions in millimeters,
  finite normalized nonnegative `(N,)` weights, and an explicit coordinate
  frame. The current `XRayTube` snapshot contains the existing single point
  source and scalar energy context. It is informational: projection ray
  generation does not consume the tube model.
- Independent mechanical tube orientation is not represented. The effective
  `directional_beam_geometry` belongs to the collimation/projection field and
  changes under H/V/Roll. No tube orientation is inferred from that field;
  `XRayTube.tube_to_reference_rotation` remains unset.
- `XRayCollimator` represents the existing ideal angular rectangular field,
  not a positioned aperture. Its optional direction-based batch helper assumes
  rays from a shared focal point and is not used in the projection hot path.
  The projector retains the existing vectorized point-source exposure mask.
- No N>1 transport, physical blades, spectra, scatter, penumbra, detector
  response, or transport extraction was implemented. Parallel mode continues
  through its existing compatibility path with no tube snapshot.
- Verification: the plugin suite passed **117 tests**, with **5 skipped** and
  **10 subtests passed**. Focused geometry, projection/collimation, physics,
  scene-format, device-link and presentation coverage passed (**102 passed,
  10 subtests**). Compile and `git diff --check` passed.
- Post-change 512x512 benchmark medians were 21.8 us for `build_geometry()`,
  60.4 us for `build_projection_config()`, and 22.85 ms for vector ray setup
  plus the rectangular mask. The mask occupies 262,144 bytes; the retained
  array proxy is about 18.1 MB. No pre-refactoring performance measurement was
  captured, so no before/after comparison is available.

## Accepted 0.6.0 physical-model architecture design

- The accepted design reference is
  [virtRTG-0.6.0-architecture.md](design/virtRTG-0.6.0-architecture.md). It
  defines a long-term direction for representing real X-ray imaging processes
  at selectable physical and numerical approximation levels. It does not
  claim that current VirtualRTG implements complete X-ray physics and does not
  add simulation functionality.
- Keep five responsibilities distinct without requiring five class
  hierarchies: physical models represent processes; experimental configuration
  selects conditions, restrictions and deliberate modifications; numerical
  methods calculate the selected model; diagnostic/research outputs expose
  intermediate or analytical results; presentation visualizes results.
- `VirtualXRay` remains the scene-integrated acquisition coordinator and
  settings/persistence boundary. `XRayTube` owns emission;
  `XRayCollimator` owns beam restriction/transmission; material-response
  models own scalar/material-to-attenuation mapping; `XRayTransportModel` owns
  transport and integration strategies; `XRayDetector` owns detector geometry
  and physical response; `DetectorImage` remains a separate presentation
  scene object. These boundaries do not imply scene nodes or serialized model
  instances.
- Keep primary line-integral diagnostics distinct from incident radiation,
  detector response, recorded signal and analytical outputs. Preserve legacy
  `integral` and `intensity` arithmetic and meanings through compatibility
  behavior; neither output implies calibrated detector counts. Per-source
  attenuation maps remain diagnostics.
- Preserve current research capabilities on the same acquisition/transport
  path: ray/planar depth-window clipping of the integration domain; anatomical
  source inclusion selection; per-object attenuation multipliers and
  material-response overrides; per-source line-integral diagnostics;
  independently processed mesh and volume sources with their supported
  backends; motion-frame expansion semantics; global material curves/windows;
  and repeated cached presentation through independent `DetectorImage`
  objects. Depth clipping is not collimation, and
  `motion_frame_mode="all"` is not time-weighted exposure.
- The accepted migration sequence is 0.6.1 tube/collimator separation with
  unchanged geometry, effective beam and device-link mapping; 0.6.2 transport
  extraction preserving clipping, source selection, per-source diagnostics
  and numerical backends; 0.6.3 detector separation preserving legacy output
  modes and distinguishing analytical results from recorded signals; 0.6.4
  capture/cache review preserving repeated presentation without transport
  reruns; and 0.6.5 integration validation including depth-clipped and
  per-source diagnostic regressions plus the accepted simulator/device-link
  workflow.
- Protocol v1, scene serialization, GUI behavior and projection geometry
  conventions remain unchanged during design acceptance. No physical-model
  classes or 0.6.1 implementation are included in 0.6.0.

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
- At the 0.5.0 baseline, projection algorithms did not use this optional beam value. Version 0.5.1 connects it to optional cone-beam collimation as described below. Device updates still do not trigger projection.

## Accepted 0.5.1 physical beam collimation

- Cone-beam projection optionally restricts primary exposure to the forward rectangular divergent field described by `RectangularBeamGeometry`. Beam axis, horizontal and vertical axes, field width and height, and reference distance remain independent of detector orientation; beam direction is never inferred from the detector.
- Collimation is explicitly enabled by `VirtualXRay.physical_collimation_enabled`, which defaults to `False`. A directional beam value alone does not enable it. The VirtualXRay properties panel exposes the explicit mode. Enabled collimation requires beam geometry and cone-beam projection; parallel-beam collimation raises a clear error.
- The projector computes a vectorized detector exposure mask from focal spot rays and the beam frame in VirtualXRay reference coordinates. It intersects exposure with scene-intersection and depth-window masks before attenuation integration, skipping work for unexposed rays. Existing attenuation integration for active rays is unchanged.
- Physical intensity conversion applies attenuation and configured geometric gain to exposed rays. Unexposed primary signal is zero, including after intensity-floor handling. Exposed rays that miss anatomy retain unattenuated primary intensity. Integral values outside the field are zero placeholders and are not measurements; the independent exposure mask records validity. No scatter or detector dark-current signal is modeled.
- The exposure mask is retained on the projection capture and `VirtualXRay.last_exposure_mask` cache, then copied into `DetectorImage`. DetectorImage reapplies the mask after presentation mapping so unexposed pixels remain distinguishable from exposed zero-attenuation pixels. Per-source outputs share the acquisition mask.
- Scene geometry serialization includes the optional directional beam axes and field parameters. Loading reconstructs them through `RectangularBeamGeometry`; older scenes without the field keep no beam geometry and collimation disabled. The active DetectorImage NPZ package serializes its mask and accepts legacy packages without a mask, while rejecting malformed declared masks. NPY and text array formats cannot preserve mask metadata.
- Device-state changes do not automatically run projections, and an acquired exposure mask remains associated with its acquisition if device geometry later changes.

## Accepted 0.5.2 standalone collimation controls

- The VirtualXRay properties panel provides a collapsible Collimation section for explicit physical-collimation activation, field width and height, reference distance, horizontal and vertical steering, beam roll, and explicit alignment to the detector. Alignment does not enable collimation or run a projection.
- Beam orientation edits use the existing `RectangularBeamGeometry`. Horizontal steering is applied about the reference vertical axis, vertical steering about the negative horizontal axis of the already-steered frame, then roll about the resulting forward beam axis using the right-hand rule. The frame remains orthonormal and right-handed. All three rotations are recomputed from one stable reference geometry; roll changes only the field horizontal and vertical axes.
- Horizontal, vertical, and roll values are relative session controls. Alignment resets them to zero and establishes a new reference. Dimension edits preserve the orientation reference; physical-collimation toggles and ordinary GUI refreshes do not rebase it. Projection, panel recreation, and selection changes preserve model-owned edit state. Successful external device geometry replacement explicitly rebases the controls; failed updates preserve the previous state.
- The effective `directional_beam_geometry` is the geometry used by projection and the only beam orientation persisted. Scene loading restores that effective frame and rebases session controls to zero around it. Relative control values and the session reference are not serialized separately. Protocol v1 is unchanged.
- Manual verification covered standalone rectangular collimation controls, H/V steering, beam roll, and projection behavior. Full manual browser-to-plugin GUI validation remains outstanding.

## Current limitations

- Only one uniquely selected `VirtualXRay` can receive an update. There is no persistent session-to-object binding, local editing lock during remote control, or multi-device session routing.
- Phantom pose is received and mapped separately but is not applied to anatomy. Table pose is received but not applied.
- Acquisition requests are validated and acknowledged without generating images.
- The browser does not stream pose updates during gizmo dragging; it sends the pose when the drag ends.
- Physical collimation is an ideal hard-edged primary-radiation model. Scatter, dark current, electronic noise, focal-spot penumbra, heel effect, and spectral emission are not modeled. Parallel-beam collimation is unsupported.
- Full manual browser-to-plugin GUI validation remains outstanding; automated Qt integration tests cover device updates, rollback, and the no-automatic-projection behavior.
- `XRayPhysicsModel` modularization into separate tube, collimator, and detector classes is deferred; `XRayTube`, `XRayCollimator`, and `XRayDetector` are not part of version 0.5.2.
- Device parameters and geometry synchronization are not bidirectional.
- Full browser interaction and workspace-selection behavior still require manual verification.

## Verification record

Version 0.5.0 verification passed: **93 passed, 5 skipped, 10 subtests passed** in the virtRTG unit suite. The Qt integration regression exercises actual `VirtualXRay` and `DeviceLinkIntegration` updates, tube rotation, preserved detector mapping, rollback after a failed geometry rebuild, and no automatic projection. Ruff F401 findings in `virtualXRay.py` and `xray/xrayProjection.py` were confirmed unchanged from `HEAD`; device-link and test modules passed F401 checks. `git diff --check` passed.

The earlier 0.4.0 offscreen host smoke exercised plugin loading, a protocol-v1 WebSocket update, queued geometry delivery, separate status indicators, stop, and application-shutdown cleanup. A Qt test also confirmed that the `VirtualXRay` global transform and an anatomical child's transform remain unchanged.

These checks do not confirm full interactive GUI behavior. In particular, a real browser session and mouse-driven workspace selection have not yet been verified.

## Planned work, not implemented

The following are possible future directions and are not part of the accepted 0.5.1 implementation:

- Persistent ownership of a `VirtualXRay` by a remote-control session.
- Restricting local edits to device parameters while remote control owns them.
- Registration between the simulator phantom and anatomy.
- Potential multi-session educational operation through one server with independent `VirtualXRay` sessions.
- Further physical detector and source modeling, including the deferred model modularization.
