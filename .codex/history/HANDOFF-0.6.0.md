# virtRTG 0.6.0 Architecture Audit Handoff

Status: final architecture design accepted as the baseline for planned 0.6.x
refactoring. This acceptance closes the design stage only; implementation is
not included in 0.6.0.

## Final research-capability consolidation

The architecture report now states the long-term goal as selectable
representations of X-ray imaging processes, without claiming that current
VirtualRTG implements complete X-ray physics. It separates physical models,
experimental configuration, numerical methods, diagnostic/research outputs,
and presentation as responsibilities rather than required class hierarchies.
Research workflows remain on the existing acquisition/transport path.

Verified implemented capabilities and their locations are inventoried in the
architecture report: ray/planar depth-window clipping restricts attenuation
integration (`xrayProjection.py` 1017-1044); enabled flags select volume and
mesh material sources (`virtualXRay.py` 464-491, 533-561); volume/mesh
backends and per-source material response settings are preserved in source
adapters (`xraySource.py` 640-747, 990-1090, 1729-1757); total and per-source
line-integral maps are captured and exportable (`xrayProjection.py` 797-804,
1193-1237; `virtualXRay.py` 1230-1244); motion `all` expands frames as
co-present source instances (`virtualXRay.py` 571-599); and display refresh
reuses cached projection without transport (`propVirtualXRay.py` 1744-1761).
Global material windows/curves and energy scaling remain heuristic
configuration (`xrayProjection.py` 429-507, 526-606).

The report clarifies that depth clipping is not collimation; anatomical source
selection is not a hardware control; per-source maps are diagnostics, not
independently measurable or generally additive detector signals; and numerical
backends calculate the selected attenuation model using different methods.
Legacy `integral` remains L, optionally adjusted by `-log(distance_gain)`;
`intensity` remains `exp(-L)` with optional gain/floor and exposure mask
(`xrayProjection.py` 607-630). Neither should be silently presented as
calibrated detector counts. Existing capture fields are sufficient for current
diagnostics. If new result types create ambiguity, the proposed smallest
addition is a compact semantic descriptor, not duplicate image caches.

No newly identified blocker changes the component contracts or stage sequence.
The A5 register entry protects research configuration and diagnostic
compatibility during transport extraction. 0.6.1 should keep those settings
outside tube/collimator device models. 0.6.2 acceptance now includes depth
clipping, source selection, per-source integrals, material overrides and all
supported backend paths. 0.6.3 retains exact legacy output arithmetic and may
add semantic metadata only if needed. 0.6.4 records selected research settings
only as justified metadata, without promising full provenance or model replay.
0.6.5 requires depth-clipped and per-source diagnostic regression cases and
the existing simulator/device-link workflow.

Items deferred: N>1 focal-spot execution, spectra and energy-dependent
materials, scatter algorithms, physical detector response, time-weighted
exposure scheduling, and replay/comparison from cached captures. A future
non-primary incident contribution may need a distinct computational strategy;
the current straight-ray transport does not implement it.
`motion_frame_mode="all"` remains co-present frame expansion. No additional
class hierarchy, research projector, scene node, serialization format or
protocol change is proposed.

The only deferred semantics question is whether a semantic tag is needed
immediately or can wait until 0.6.3 introduces multiple output meanings.
Current output fields and compatibility behavior are clear; this does not block
0.6.1 implementation review. The accepted design is the baseline for that
review.

## Findings

- `VirtualXRay` is both the scene-integrated configuration owner and acquisition
  coordinator. It gathers descendant volume/mesh sources, handles relative and
  motion transforms, constructs projection configuration, and owns runtime
  capture caches.
- `XRayProjector` currently combines vectorized detector ray construction,
  ideal rectangular exposure masking, AABB/depth clipping, per-source direct
  integration dispatch, batched slab sampling, line-integral summation and
  simplified detector-value conversion.
- Active volume routes are default batched point sampling ("Joseph-type") and
  optional vectorized Siddon. Mesh projection uses analytic BVH direct
  integration by default; projected intersection-list is experimental.
- Current `XRayPhysicsModel` maps scalar data to heuristic attenuation and also
  performs simplified Beer-Lambert/distance/floor conversion. There is not yet
  a separately modeled detector response, calibrated output, energy spectrum,
  scatter, or detector spatial/noise response.
- Existing acquisition/presentation separation works: the capture/cache keeps
  line-integral and recorded maps plus mask/per-source data; the separate
  `DetectorImage` scene object owns presentation/package state. "Update display"
  reuses those results without recomputing transport.
- Device-link protocol v1 maps simulator tube pose to fixed-offset focal spot
  and tube-aligned beam frame, collimator dimensions to an ideal rectangle at
  1000 mm, and detector pose to a handedness-adjusted detector basis. It does
  not transmit independent collimator orientation.
- Recommended roadmap: characterize first; then tube/collimator adapters,
  transport extraction, detector response separation, capture/cache review,
  and end-to-end integration. Do not add an acquisition orchestrator or
  source-assembly class unless new independent lifecycle/pose requirements
  justify them.

## Extended focal-spot readiness revision

The design now treats a weighted set of focal-spot samples as the tube emission
contract: `positions_mm` has shape `(N,3)` and `weights` has shape `(N,)`, with
tube-local positions transformed once into VirtualXRay reference frame R and
finite nonnegative weights normalized to sum to one. No regular-grid layout is
assumed. Current N=1 is explicitly the legacy point source with weight 1 and
must not allocate repeated origins, sample indices, or float transmission
buffers.

Collimation is specified per ray with independent origins and directions. Its
general result can be binary acceptance or fractional transmission, with
different values for different focal samples. The current vectorized
`bool[H,W]` rectangular exposure mask remains the exact N=1 optimized path and
compatibility result.

Ray batches retain detector-pixel and focal-sample mapping and can be chunked,
avoiding an all-samples-by-all-pixels allocation. Transport results retain
sample identity; they cannot require a single aggregated line-integral image
for N>1. Physical accumulation follows per-sample attenuation and collimator
transmission:

```text
I(pixel) = sum_i I0_i(pixel) * T_i(pixel) * exp(-L_i(pixel))
```

The design explicitly rejects averaging line integrals before Beer-Lambert
conversion. Current N=1 captured line integrals and existing `integral` and
`intensity` output semantics remain unchanged. Independent collimator pose can
later be represented by a named relative transform; protocol v1 continues to
derive the collimator frame from tube orientation.

The revision identifies required contract compatibility now, later features
that fit those contracts, and speculative physics excluded from current work.
It updates 0.6.1 and 0.6.2 acceptance criteria without adding a milestone.
Extended-source execution, extra ray tracing, spectra, blade physics, penumbra,
serialization changes, GUI changes, and protocol changes remain out of scope.

## Final assumptions and extensibility audit

The source audit found three additional contract-level A risks and one
confirmed focal-spot risk. The revised report adds a concise, source-cited
"Physical and numerical simplifications - extensibility register" with runtime,
memory and complexity estimates:

- **A1, focal spot/collimator origins (0.6.1-0.6.2):** current projection
  geometry holds one source position and builds all pixel rays from it; device
  mapping derives a tube-aligned field. The proposed weighted-sample,
  independent-origin, per-ray acceptance contracts address this. N=1 stays on
  the existing vector and bool-mask path.
- **A2, scalar energy context (0.6.1 and 0.6.3):** existing energy is a scalar
  attenuation heuristic. Reserve optional scalar energy/context and fluence
  weights per batched evaluation/contribution so emission, material mapping,
  collimation and detector response can later evaluate spectral batches without
  mandatory energy arrays. Omitted context has no current-path cost.
- **A3, line-integral-only result risk (0.6.2-0.6.3):** retain primary
  per-sample line integrals as diagnostics and allow detector-plane incident
  contributions that are not line integrals. This leaves a boundary for future
  scatter without implying that current straight-ray transport implements it.
- **A4, source gain mixed into detector conversion (0.6.1 and 0.6.3):** current
  inverse-square gain is applied inside `XRayPhysicsModel.integral_to_image()`.
  Separate emitted fluence/geometric gain from detector recording while
  preserving both existing output modes through a compatibility adapter.

The register also documents replaceable B simplifications: no heel-effect or
absolute tube model; ideal binary rectangular collimation without physical
aperture/thickness; heuristic scalar-to-attenuation mappings and stepped/Siddon
voxel assumptions; constant solid/shell mesh materials; ideal detector pixels
with no blur/noise/saturation; and current all-motion-frame expansion semantics.
Category C intentionally retains scatter algorithms, capture sufficiency for
alternate physical-model comparisons, and per-source image additivity as
documented limitations pending a concrete requirement.

Important verified source locations include `xray/xrayProjection.py` lines
162-169 and 963-1004 (single-source ray and detector mask), 429-507 (scalar
energy heuristic), 584-630 (distance gain and signal conversion), and 797-804,
1184-1240 (capture/result semantics); `xray/xraySource.py` lines 702-727,
729-747 and 986-1015 (volume/material and mesh assumptions); `virtualXRay.py`
lines 571-599 (motion frame expansion); and `device_link/virtualrtg_geometry.py`
lines 18-25 and 117-159 (demonstrator offsets and tube-aligned field).

The architecture diagram and contracts now separate primary attenuation
diagnostics, incident detector-plane contributions, recorded signal and
presentation. The duplicate Recorded signal row was removed. The capture
contract does not promise alternative physical-model results from cached data;
re-running transport may be needed. Multiple DetectorImage presentations still
reuse the current acquisition cache.

Assumptions intentionally retained: monoenergetic simplified primary
attenuation; straight rays and static geometry within one projection call;
current point-source numerical outputs; optional inverse-square compatibility
behavior; no scatter, spectra, blade physics or detector response effects; and
no protocol v1, GUI, serialization or scene-graph changes.

Unresolved design questions: what material and spectrum model should supply
energy-dependent coefficients; what specific scatter strategy/accuracy is
required; how future exposure schedules weight motion states; and whether
physical-model comparison should rerun from scene sources or retain costly
intermediate material data. None is required to accept the current component
boundaries.

## Created files

- [virtRTG-0.6.0-architecture.md](../design/virtRTG-0.6.0-architecture.md)
  is the finalized and accepted reference containing the source-path audit,
  component proposal, contracts, frames, cache
  guidance, device compatibility, focused benchmark protocol, staged roadmap,
  risks, recommended first implementation step, and the extended focal-spot
  contract plus final assumptions/extensibility revision.
- [HANDOFF-0.6.0.md](HANDOFF-0.6.0.md) records the findings and deliverables.

The design report includes file and line references for verified implementation
claims, distinguishes current behavior from proposals, and maps each category-A
risk to a migration stage and acceptance criterion. The accepted 0.6.0 closure
updates only this handoff, `.codex/baseline.md`, and `CHANGELOG.md` alongside
this finalized design reference. No production Python, tests, benchmark, GUI,
serialization, protocol, simulator behavior, or release metadata was changed.
No tests were run because this closure contains documentation only.
