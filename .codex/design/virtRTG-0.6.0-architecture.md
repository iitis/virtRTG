# virtRTG 0.6.0 Physical Model Architecture Audit

Status: design proposal; implementation is not authorized by this document.

Scope: audit the checked-out `plugins/virtRTG` implementation and define a
minimal, compatible direction for separating physical-model responsibilities.
Source references below point to the inspected checkout and should be rechecked
before implementation if those files have changed.

## 1. Executive summary

### Overarching architectural principle

VirtualRTG should be capable over time of representing the processes involved
in real X-ray imaging at selectable physical and numerical approximation
levels. This is a development direction, not a claim about the current
simulation. Research workflows use the same acquisition and transport path:
they configure inputs or integration domains and request diagnostic outputs,
rather than introducing a parallel research projector.

Keep five responsibilities conceptually distinct: physical models (represented
processes), experimental configuration (conditions, restrictions and
deliberate modifications), numerical methods (calculation of the selected
model), diagnostic/research outputs (intermediate or analytical results), and
presentation (visualization). These are responsibility labels, not a mandate
for five class hierarchies.

The plugin already has useful separations for projection geometry, attenuation
sources, transport orchestration, presentation, and a `DetectorImage` scene
object. The actual computation is less modular than these type names suggest:
`XRayProjector` owns ray generation, rectangular exposure masking, scene-bound
clipping, depth clipping, dispatch to direct source integrators or batched
sampling, and conversion of integrals to image values. `XRayPhysicsModel` both
maps scalar values to attenuation and converts the integrated result into a
simple detector-space value. `VirtualXRay` is the scene object and the current
acquisition coordinator; it also builds configuration and owns runtime caches.

Recommended direction:

1. Keep `VirtualXRay` as the scene-integrated acquisition coordinator and
   source of persisted/user-edited settings. Do not add an acquisition-system
   class at this stage.
2. Treat `XRayProjectionGeometry` as the existing acquisition-geometry
   contract. Tube emission must support weighted focal-spot samples from the
   beginning while preserving the current N=1 point-source fast path.
   Collimation must accept independent ray origins and keep the existing
   detector mask as its optimized point-source specialization.
3. Extract transport behind one coarse, vectorized `project(capture request)`
   operation. Preserve NumPy ray batches and the existing source-level direct
   integral hook. Support chunked sample/pixel batches without materializing
   their full Cartesian product. Never call a Python model once per ray.
4. Keep primary per-sample line integrals as transport diagnostics. Build
   incident radiation separately so additional detector-plane contributions
   need not be encoded as line integrals. Accumulate transmitted primary
   intensity after each sample's transport and transmission; do not average
   line integrals before Beer-Lambert conversion. Preserve legacy N=1 modes.
5. Separate detector response from presentation only after defining the
   existing `integral_to_image` behavior as a compatibility detector-response
   implementation. Do not claim current values are physically calibrated
   incident radiation or detector counts.
6. Preserve the current capture-to-`DetectorImage` flow. A second incident
   radiation cache is not justified by current single-energy calculations.

The safest release order differs slightly from the candidate sequence: first
characterize and freeze existing contracts, then introduce tube/collimator
configuration facades, then transport extraction, then detector response, and
only then review capture/caching. This avoids simultaneously moving the
ray-generation boundary and changing signal semantics.

## 2. Current architecture and execution-path map

### GUI acquisition path (verified)

`PropVirtualXRay.on_run_simulation()` disables the run button, freezes GL viewer
updates, pumps Qt events, then calls `VirtualXRay.project_and_cache()`; after it
returns, the panel creates or synchronizes a `DetectorImage`. The separate
"Update display" handler reads the existing raw cache and synchronizes the same
image object without a projection call. See
[`propVirtualXRay.py`](../../gui/propVirtualXRay.py#L1670) and
[`propVirtualXRay.py`](../../gui/propVirtualXRay.py#L1744).

```text
GUI Run Simulation
  -> VirtualXRay.project_and_cache
     -> project_capture
        -> build_scene / scene_sources
        -> build_projection_config
           -> build_geometry + build_physics_model + quality profile
        -> XRayScene.project_capture -> XRayProjector.project_capture/project
           -> detector pixels and (H*W,3) ray arrays
           -> ideal rectangular exposure mask
           -> vectorized AABB/depth clipping
           -> direct source integration (Siddon or mesh) and/or slab sampling
           -> integral_to_image; capture line integral, detector image,
              per-source maps, statistics, exposure mask
     -> cache arrays/mask; project scene annotation overlays
  -> sync data and presentation state to separate DetectorImage scene object

GUI Update display
  -> synchronize cached arrays/settings to existing DetectorImage; no transport
```

`VirtualXRay` traverses descendants, excludes nested `VirtualXRay` sub-scenes,
converts mesh/volume objects into `XRaySampleSource` adapters, computes each
source transform relative to its own reference frame, and groups motion frames
where requested. The scene and projection configuration are created at the
`VirtualXRay` boundary, not in the GUI. See
[`virtualXRay.py`](../../virtualXRay.py#L416) (relative transforms),
[`virtualXRay.py`](../../virtualXRay.py#L464) (descendants),
[`virtualXRay.py`](../../virtualXRay.py#L628) (scene assembly), and
[`virtualXRay.py`](../../virtualXRay.py#L882) (configuration).

### Active and non-primary integration paths (verified)

- **Volume sampling (default):** `VolumetricXRaySource` samples scalar values
  at batched world points, maps them to attenuation, and the projector
  integrates with a single Python loop over depth positions. Each iteration
  evaluates all active rays and loops over sources. This is Joseph-type
  ray-marching behavior; it is not a separate Joseph class. See
  [`xraySource.py`](../../xray/xraySource.py#L724) and
  [`xrayProjection.py`](../../xray/xrayProjection.py#L1046).
- **Volume Siddon:** the selectable `siddon` backend computes exact voxel chord
  lengths in batched NumPy chunks, reports a direct integral, and bypasses the
  projector's depth-step loop for that source. `step_mm` does not determine
  its chord integration. See
  [`xraySource.py`](../../xray/xraySource.py#L729) and
  [`xraySource.py`](../../xray/xraySource.py#L937).
- **Mesh analytic BVH:** the normal mesh path uses direct per-ray triangle
  intersections accelerated by a BVH, with simplified constant-material
  `solid` or `shell` semantics. The mesh adapter always selects the direct
  integral route. See [`xraySource.py`](../../xray/xraySource.py#L986) and
  [`xraySource.py`](../../xray/xraySource.py#L1729).
- **Projected mesh intersection list:** an experimental direct integration
  backend builds/caches projected per-pixel intersection stacks. It remains
  selectable in source configuration and should be retained as an explicit
  backend, not described as the default. See
  [`xraySource.py`](../../xray/xraySource.py#L1115) and
  [`xraySource.py`](../../xray/xraySource.py#L1756).
- No executable function/class named `Joseph` or `Siddon` exists as a
  projector-level strategy. "Joseph-type" describes the default point-sampling
  path; Siddon is a volume-source integration backend. The benchmark module
  compares sampling and Siddon but is not the GUI acquisition path.

`XRayProjectionGeometry` constructs pixel-axis vectors from detector pose,
shape and pixel spacing; stores cone-source or parallel-direction parameters,
optional depth window, beam geometry and collimation toggle. `RectangularBeamGeometry`
is a validated immutable value object for an ideal rectangular divergent field;
its own `contains_point` helper is not the hot mask path. The projector creates
the detector sample grid and mask in vectorized NumPy operations. See
[`xrayProjection.py`](../../xray/xrayProjection.py#L153),
[`xrayProjection.py`](../../xray/xrayProjection.py#L356), and
[`xrayProjection.py`](../../xray/xrayProjection.py#L949).

### Serialization and integration (verified)

The plugin-specific XML scene format persists `VirtualXRay` geometry,
directional beam, physics/source defaults and image defaults. A distinct
`DetectorImage` node persists image/presentation/package data. Scene export
explicitly skips DetectorImage while collecting the VirtualXRay's projection
input descendants, and the loader instantiates it as a separate scene object.
ATMDL exports VirtualXRay settings and children; DetectorImage is skipped by
the current ATMDL export hook. No tube, collimator, transport, or detector
physical-model objects are currently serialized. See
[`sceneFormat.py`](../../sceneFormat.py#L326),
[`sceneFormat.py`](../../sceneFormat.py#L357),
[`sceneFormat.py`](../../sceneFormat.py#L430),
[`sceneFormat.py`](../../sceneFormat.py#L521), and
[`sceneFormat.py`](../../sceneFormat.py#L1074).

The plugin registers its property panel and detector-image panel with the
existing application/plugin system. The active workflow remains plugin-local;
the core supplies scene objects, transforms, workspace, and GUI infrastructure.

### Verified research-capability inventory

The following are current behaviors, not proposed physics. Their
configuration belongs with the existing acquisition request/source adapters;
their calculation remains in the shared projector/transport path. Moving
transport must preserve them. Any later extraction should add focused
compatibility coverage for depth clipping, source selection/maps, both output
modes, backend dispatch, motion-frame selection, and cached presentation.

| Capability and status | Verified semantics and location | Interpretation and owner | Migration preservation |
|---|---|---|---|
| Depth window: implemented | `VirtualXRay` stores off/ray/planar limits and passes active values to `XRayProjectionGeometry`; projector intersects ray parameter intervals or a planar slab with source/AABB intervals before integration (`virtualXRay.py` 658-683; `xrayProjection.py` 287-353, 1017-1044). | Restricts the attenuation integration domain. It is an experimental acquisition/transport setting, not collimation or a hardware aperture. | Transport configuration and interval clipping must survive 0.6.2. Add a depth-clipped projection regression case in 0.6.5. |
| Anatomical inclusion: implemented | Descendant collection filters volume and mesh sources by `xray_source_enabled`; disabled items do not become `XRaySampleSource` adapters (`virtualXRay.py` 464-491, 533-561). | Selects which modeled materials participate. It does not remove scene objects or represent an acquisition device control. Owner: acquisition/source selection. | Preserve enabled/disabled selection and mixed-source behavior in 0.6.2; cover by a source-selection regression. |
| Per-object and per-source attenuation: implemented | Volume/mesh adapters apply `attenuation_multiplier`, optional material-response overrides, and source-local material mapping; projector accumulates separate source `L` maps and converts each map to its own output (`xraySource.py` 640-727, 990-1033, 1733-1754; `xrayProjection.py` 1049-1109, 1193-1216). | Per-source maps are computational diagnostics; independent nonlinear conversions are not generally additive detector measurements. Owner: source/material adapter and transport diagnostics. | Retain source identity, integrals, override settings and diagnostics through transport extraction. Add per-source diagnostic-output coverage by 0.6.5. |
| Diagnostic line-integral maps: implemented | Capture contains total `line_integral_image`; each source entry contains its own line-integral and converted image, and `VirtualXRay` caches/exports them (`xrayProjection.py` 785-804, 1193-1237; `virtualXRay.py` 1026, 1230-1244). | Analytical outputs, distinct from incident radiation and recorded detector signal. Owner: transport result/capture. | Keep existing fields and export/import compatibility. No separate research engine or required extra cache. |
| Output modes: implemented | `integral_to_image()` returns `L` (or `L-log(gain)` with distance falloff) in `integral` mode; `intensity` returns `exp(-L)`, optionally times gain and floored. Both apply exposure masking (`xrayProjection.py` 607-630). | Legacy compatibility outputs; `integral` is not a calibrated detector measurement. `intensity` is simplified monoenergetic signal conversion, not a modeled physical detector response. Owner: compatibility response/output semantics. | Preserve exact arithmetic, masks, defaults and serialized settings in 0.6.3; do not relabel old values as calibrated counts. |
| Independent mesh/volume processing and backends: implemented | Volumes use sampled interpolation or batched Siddon; meshes use direct analytic BVH or experimental projected-intersection-list integration (`xraySource.py` 702-747, 937-962, 986-1015, 1729-1757). | Different numerical methods/source representations calculate selected attenuation; projected-list remains experimental. Mesh and volume can coexist in one scene. Owner: source adapters and transport numerical strategy. | Retain each selectable backend and source-local dispatch. Geometry/ray batches should remain model-independent and reusable; backend-specific data can require distinct strategies. |
| Motion frame handling: implemented | `active` uses the active frame; `all` expands frame transforms into source instances within one projection (`virtualXRay.py` 513-599). | Current `all` semantics treat frames as co-present attenuation sources, not time-weighted exposure. Owner: experimental source configuration. | Preserve current expansion in 0.6.2; do not describe or convert it as temporal integration. |
| Repeated cached presentation: implemented | Run computes/caches projection; update-display synchronizes cached results to an independent `DetectorImage` without transport (`propVirtualXRay.py` 1670-1688, 1744-1761; `virtualXRay.py` 1010-1030). | Presentation reuses acquisition; it is not a new exposure. Owner: capture/cache and `DetectorImage` presentation boundary. | Preserve independent image objects and no-rerun behavior in 0.6.4. |
| Material and attenuation configuration: implemented | Global `XRayPhysicsModel` has heuristic scalar-to-mu curves/windows and scalar energy scaling; source overrides and multipliers are copied into source adapters (`xrayProjection.py` 429-507, 526-606; `virtualXRay.py` 533-561; `xraySource.py` 640-727, 990-1090). | Deliberate material-model configuration, not calibrated composition data. Owner: material-response model, with per-object overrides at source adapters. | Preserve global defaults and local overrides. Keep material response distinct from emission and detector response. |

These orthogonal interventions belong in the existing `VirtualXRay`
acquisition/configuration and transport request boundaries. Do not put depth
windows or anatomy selection into tube, collimator, or detector models. Custom
material settings belong to material response/source adapters. Numerical
backends remain selectable strategies for the same requested attenuation
calculation; non-primary contributions may need a distinct computational
strategy and are not made available by the current straight-ray integrators.

## 3. Responsibility inventory

| Existing element | Actual current responsibility | Boundary assessment |
|---|---|---|
| `VirtualXRay` | Scene node; settings owner; descendant collection; relative and motion transforms; geometry/config/source construction; capture/cache ownership; annotation projection; export/import helpers | Too many coordination/runtime responsibilities for a pure physical component, but it is the right existing orchestrator and persistence boundary. Avoid scene-model proliferation. |
| `XRayProjectionGeometry` | Detector pixel lattice and pose; cone/parallel selection by populated geometry fields; depth window; beam/mask settings | Good geometry/request value. Keep as compatibility adapter until new models can produce it. |
| `RectangularBeamGeometry` | Ideal field frame/size validation and simple membership helpers | Good immutable collimator description; geometry-only, not blade transport. |
| `XRayPhysicsModel` | Scalar-to-linear-attenuation map; material curves/window; energy scaling heuristic; Beer-Lambert conversion; optional inverse-square gain/floor | Split conceptually into material/attenuation mapping and detector conversion. Current energy support is a scalar heuristic, not polyenergetic transport. |
| `XRaySampleSource` adapters | Source bounds, world sampling and optional direct integration; volume preprocessing/material overrides; mesh solid/shell and BVH/intersection behavior | These are scene-material adapters consumed by transport; they are not tube/source emission models. Existing abstract method dispatch is per source/batch, not per ray. |
| `XRayScene` | Holds sample-source adapters and creates a projector for capture | Thin container. No extra scene graph node required. |
| `XRayProjector` | Builds rays; masks and clips them; dispatches direct/marched sources; combines line integrals; applies signal conversion; returns capture | This is current transport plus orchestration and parts of source/detector responsibilities. Extract behind batch API. |
| `XRayProjectionCapture` | Mutable dataclass holding line-integral image, detector image, per-source output maps, optional stats and exposure mask | Useful boundary but not immutable and contains computed signal, not full acquisition metadata. Review in later cache stage. |
| `XRayPresentationModel` / `DetectorImage` | Presentation mapping, windows/transfer settings, overlays, multi-layer packages and viewing/export state | Presentation is properly separate from expensive projection. `DetectorImage` is independently persisted/scene-managed. |
| Device-link mapping | Protocol-state mapping into VirtualXRay-local focal spot/detector pose and effective rectangular beam; GUI-thread applies it | Preserve the accepted adapter; pass values into future component configuration without changing protocol v1. |

## 4. Proposed component architecture

The components below are ordinary Python objects, instantiated or configured
by `VirtualXRay` (or a small factory called by it). They do not inherit from
scene `Object`, do not become workspace children, and do not acquire extra
scene graph identities. `DetectorImage` remains the separate scene object.

```text
VirtualXRay (existing scene node / configuration + coordinator)
  |-- TubeModel (focal samples, weights, direction/frame, simple emission)
  |-- CollimatorModel (per-ray field acceptance/transmission)
  |-- XRayProjectionGeometry (compatibility acquisition lattice and frames)
  |-- XRayScene (volume/mesh material adapters)
  `-- XRayTransportModel.project(batch request)
       -> TransportResult: per-sample primary L + separate diagnostics
       -> Incident accumulator: I0 * T * exp(-L), plus optional additional
          detector-plane radiation contributions (not represented as line L)
       -> DetectorResponseModel.record(incident contributions, geometry)
       -> AcquisitionCapture (incident/recorded data + L diagnostics + metadata)
       -> DetectorImage presentation/view (existing scene object)
```

### XRayTube

Use a lightweight ordinary Python configuration/model object, not an
inheritance family. Define its emission contract as `FocalSpotSamples`:

```text
positions_mm: float32[N, 3]
weights:      float32[N]
```

Positions are tube-local when produced by the tube model. The geometry builder
applies the tube-to-reference transform once and returns focal positions in R.
Existing entry points that already supply an R-frame source position adapt it
as a one-element sample. The frame must be explicit in the contract. Weights
are finite, nonnegative relative emission weights normalized to sum to one.
No regular-grid assumption is permitted. A future direction-dependent source
intensity can be evaluated per sample-ray direction and multiplied by the
sample weight without changing the tube/transport boundary. Do not add
spectral arrays or bins.

The current point source is N=1 with its legacy focal position and weight 1.
Keep an explicit N=1 branch that reuses the existing `(H*W,3)` ray arrays and
point-source collimation mask; do not tile origins, weights, directions, or
transmission. A model may produce samples on demand, but eager one-dimensional
arrays suffice now. Chunk production can be added later without changing array
meaning or frame.

### XRayCollimator

Use a lightweight model/value object for the existing ideal divergent rectangle:
axis, horizontal, vertical, full width and height at a reference distance,
plus `enabled`. Its fundamental operation accepts rays with independent
origins and directions. Results are per-ray and support binary acceptance or
fractional transmission. Multiple focal samples may have different results.

Keep the current `bool[H,W]` exposure mask as the optimized point-source
implementation and compatibility output. For N=1, preserve the existing
vectorized rectangle predicate exactly; do not convert it to float
transmission or duplicate it. An extended spot evaluates the same field for
each sample's rays. A future blade model may return `float32` transmission in
`[0,1]`; no blade physics or penumbra is required now. The contract may return
the shared detector mask for the proven N=1 case or chunk-local per-ray arrays
for general samples.

### XRaySourceAssembly

Do **not** add a permanent assembly class initially. `VirtualXRay` already
defines the local reference frame and stores all device parameters; the beam
reference and effective geometry are already synchronized by explicit methods.
An assembly value would duplicate ownership and persistence. If composition
logic becomes nontrivial, introduce a small immutable pose/config value that
combines tube pose and collimator-local pose while translating to the current
geometry. A true class is justified only when there are independently mounted
components whose relative transforms are persisted and edited. Device v1 has
no independent collimator orientation.

### XRayTransportModel

Separate traversal/integration from geometry setup and final signal conversion.
Its public entry point operates on batches and an `XRayScene`; it may dispatch
once per source/backend. It owns AABB and depth clipping and integration
policy. Ray origins and directions are independent `float32` arrays of shape
`(R,3)`. No shared-origin assumption is allowed. Preserve a stable mapping
from each ray to focal sample and detector pixel. Generate rays in bounded
sample/pixel chunks rather than allocating the full
`(N_samples * H * W,3)` Cartesian product. Existing N=1 uses the current
single `(H*W,3)` setup and integration fast path.

For migration, transport can delegate to existing sampling, Siddon and mesh
functions. Preserve per-sample primary line-integral diagnostics (or a
chunk-equivalent retaining sample identity), but do not define line integrals
as the universal transport output. The transport-to-response boundary must
also permit detector-plane incident contributions that are not derived from
straight-ray attenuation, without requiring a scatter implementation now. N=1
retains the current `float32[H,W]` line-integral result and legacy aliases. Ray
parameters and integral semantics remain in mm and `mu * mm`. Source adapters
receive vectorized rays per source/chunk; no Python dispatch per ray.

Ray construction is geometry-facing: detector pixel centers and ray origins /
directions describe acquisition geometry. Keep one builder near
`XRayProjectionGeometry`/a pure helper called by transport, rather than
duplicating it in tube, collimator and detector models. It combines tube-local
sample positions transformed to R with detector pixel centers, then applies
the reference-to-world transform once. The point-source builder may remain
unchanged; multi-sample requests produce bounded chunks of independent rays.

### XRayDetector

Use an ordinary configurable Python object, no abstract subclass hierarchy for
one current implementation. Separate geometry (active plane, pixel pitch,
shape and pixel centers) from response (conversion of incident signal into a
recorded signal), even if one object initially owns both settings. The existing
pixel lattice remains `XRayProjectionGeometry` through the first migration.
There is no current scintillator, charge collection, detector MTF, noise, lag,
or energy response model. Name current conversion "simple monochromatic
response" or equivalent, and make its identity/conventions explicit.

### XRayAcquisitionSystem

Not justified currently. `VirtualXRay.project_capture()` and
`project_and_cache()` already orchestrate a single acquisition and its cache;
`XRayScene` and `XRayProjector` perform backend work. Another acquisition
object would mostly forward calls and duplicate settings ownership. Reconsider
only if multi-exposure scheduling, asynchronous acquisition lifecycle,
calibration, or several synchronized hardware components become a real use
case.

## 5. Explicit interfaces and data contracts

These contracts are proposed migration targets, not current classes unless
explicitly identified. Preserve tuple/array compatibility during initial
extraction; avoid a generic universal radiation framework. The sample axis is
present in the contract from the start, but N=1 remains a specialized
representation and execution path rather than being expanded into redundant
arrays.

| Contract | Producer -> consumer | Fields, units, shape and frame | Ownership/performance |
|---|---|---|---|
| Emission / `FocalSpotSamples` | Tube -> geometry builder and incident accumulator | Tube-local `positions_mm: float32[N,3]`; `weights: float32[N]`, finite/nonnegative and normalized to sum 1. Builder transforms positions once to R. Optional scalar energy/context and scalar batch fluence weight may accompany an evaluation; direction-dependent intensity is evaluated per sample-ray. | Owned or read-only arrays; no grid requirement. N=1 is legacy focal position plus weight 1 and must not be tiled. No spectrum array; on-demand sample generation may be added later. |
| Ray batch | Geometry builder -> collimator and transport | Independent `origins_mm: float32[R,3]`, normalized `directions: float32[R,3]`, `t_start_mm/t_end_mm: float32[R]`, valid flags, detector flat indices, and sample indices (or a known N=1 sentinel). Directions are in W for source adapters. | Chunk-local arrays preserve pixel/sample ordering. No per-ray Python objects/calls. N=1 reuses current `(H*W,3)` buffers without sample-index allocation where practical. |
| Collimator result | Collimator -> transport/accumulator | Per-ray binary acceptance `bool[R]` or fractional transmission `float32[R]` in `[0,1]`, aligned to ray/sample/pixel order; evaluation may receive optional scalar energy context. N=1 may use current shared `exposure_mask: bool[H,W]`. Extended samples can differ. | Vectorized and chunk-local for general rays. N=1 preserves exact mask path without float conversion or duplicate arrays; energy omission is zero-cost. |
| Primary transport diagnostics | Transport -> capture/diagnostics | Per-sample `line_integral_i: float32[H,W]`, or chunk-equivalent retaining sample identity; N=1 keeps existing `float32[H,W]` map. Values are `mu * mm`; optional source maps and stats. Not fluence and not a universal transport output. | Do not require one aggregated L image for N>1. Arrays are call-owned; current capture is mutable pending cache review. |
| Incident contribution | Emission/collimator/transport -> detector response | Current primary contribution can be one monoenergetic `float32[H,W]` map after per-ray `I0*T*exp(-L)`. Later detector response may accept distinct additive detector-plane contributions, including scatter, optionally tagged by scalar energy/context. | Keep identity until energy-sensitive response is applied. No mandatory energy axis or universal radiation hierarchy. N=1 may reuse current detector output buffer. |
| Recorded signal | Detector response -> acquisition capture | `float32[H,W]`; numeric meaning documented by response mode (current `integral` or `intensity`); detector shape and exposure state. Response evaluation may receive scalar energy/context per contribution, not an energy cube. Current value is pre-display, not calibrated digitizer units. | One output array. Keep display/window/gamma/CLAHE out of this result. |
| Acquisition capture | Acquisition coordinator -> cache and `DetectorImage` | Current capture stores line-integral and recorded maps, per-source maps, mask and stats. Proposed snapshot may additionally retain incident contributions and geometry/model/acquisition metadata. Arrays are detector-shaped unless retained as per-sample contributions; no required time/energy axis. | Snapshot ownership should be explicit; current capture is mutable and lacks complete geometry/model metadata. Do not add copies or serialization fields without cache review. |
| Presentation | `DetectorImage` -> viewer/export | Takes one selected capture layer and presentation config (window, invert, gamma, transfer curve, enhancement and overlays); returns display array. Display array is not written back as transport or recorded signal. | A presentation can be recomputed from retained capture. Existing separate DetectorImage object and cached raw projection already support multiple presentations. |

One energy scalar in `XRayPhysicsModel` currently scales heuristic `mu`; it does
not create an energy spectrum. The line integral/Beer-Lambert simplification
remains sufficient for current modes. Keep spectral samples out of current
arrays, but permit energy as scalar evaluation context on an emission,
material-response, collimator-transmission, or detector-response call. Future
spectral work can evaluate energy batches and weighted detector-plane
contributions without imposing an `[E,H,W]` representation on current callers.
The existing `source_energy_kev` does not provide this behavior by itself.

### Physical accumulation semantics

For each ray, transport returns integrated attenuation `L_i(pixel)`.
Collimator transmission `T_i(pixel)` is separate, and emitted fluence
`I0_i(pixel)` contains the normalized focal-sample weight and any future
direction-dependent intensity. For monoenergetic extended-source transport,
the ideal incident signal is conceptually:

```text
I(pixel) = sum_i I0_i(pixel) * T_i(pixel) * exp(-L_i(pixel))
```

Accumulate focal-sample contributions after their individual transport and
collimator evaluation, at the physical detector-response/accumulator boundary.
Do not average `L_i` before Beer-Lambert conversion. Transport output must
retain sample identity, through per-sample maps or a chunked equivalent, until
this accumulation. A single summed line-integral image is insufficient for
N>1. If an effective integral is needed later, define it only as
`-log(I/I0_total)` where `I0_total` is positive, label it an effective
equivalent integral, and never equate it to mean sample attenuation.

For current N=1, the captured line-integral image remains `L_1`. The existing
`integral` output mode remains its current converted value, including the
optional `-log(distance_gain)` adjustment when enabled; `intensity` remains the
current Beer-Lambert conversion, optional gain, and floor. Contract extraction
must preserve both outputs exactly.

Use existing capture fields for the established diagnostic maps. If output
semantics become ambiguous across new response models, add a small semantic
descriptor to the capture/output metadata (for example, legacy integral,
simplified intensity, or recorded response). Do not duplicate arrays or force
analytical outputs through detector response. Keep `DetectorImage` as
presentation; a displayed image does not itself establish physical recording.

### Physical and numerical simplifications - extensibility register

This register distinguishes **verified current behavior** from **proposed
contract support**. Priority A marks a contract boundary to protect during
0.6.x; it does not authorize implementing the future physics. B and C items can
remain replaceable-model limitations or documented future work.

| ID / component | Current assumption and source | Consequence; plausible extension | Existing support; contract change and minimum preparation | Approximate cost; priority / stage |
|---|---|---|---|---|
| A1 Tube, collimator, transport | Current geometry contains one `source_position_ref`; ray setup fills all origins from that point and the mask is detector-wide (`xrayProjection.py` 162-169, 963-1004). Device-link derives a tube-aligned field and has no independent collimator pose (`virtualrtg_geometry.py` 117-124, 141-159). | Extended focal spot creates different origins and aperture acceptance per pixel/sample. | Revised `FocalSpotSamples`, independent-origin batches, per-ray collimator result and N=1 fast path address this. 0.6.1/0.6.2 must enforce frame and pixel/sample mapping. | N=1 no extra arrays/dispatch; N>1 cost scales with rays and chunk-bounded memory. A / 0.6.1-0.6.2. |
| A2 Energy context across components | `source_energy_kev` is one scalar and `attenuation_energy_scale()` applies `(reference/source)**exponent`; `scalar_to_mu()` uses heuristic response (`xrayProjection.py` 429-444, 484-507). The mask predicate has no energy input (`xrayProjection.py` 983-1004). | Spectral emission and energy-dependent `mu`, collimator transmission or detector response would otherwise require signature changes or energy assumptions inside one component. | Permit optional scalar `energy_kev`/energy context per batch evaluation and tag incident contributions with scalar energy/context. Later spectral quadrature can use repeated/chunked calls; no spectral arrays now. Enforce across 0.6.1 and 0.6.3. | No cost when omitted; future runtime scales with sampled energies, with chunk-bounded temporary memory. A / 0.6.1, 0.6.3. |
| A3 Diagnostics vs physical incident radiation | `XRayProjectionCapture` centers on line-integral and detector-image arrays; projector converts summed `projection_flat` through `integral_to_image()` (`xrayProjection.py` 797-804, 1184-1240). | Scatter or another non-straight-ray contribution cannot be represented by `L` alone; requiring all output to be `L` would give false semantics. | Keep primary per-sample `L` as a diagnostic and separately represent detector-plane incident contributions consumed by detector response. Allow an additional contribution not represented by line integral. No Monte Carlo framework. Define at 0.6.2 and consume at 0.6.3. | No added current-path cost beyond existing outputs; future scatter adds its own computation/storage. A / 0.6.2-0.6.3. |
| A4 Emission/geometric gain vs detector response | Inverse-square gain lives on `XRayPhysicsModel` and is folded into `integral_to_image()`; integral mode changes `L` by `-log(gain)` (`xrayProjection.py` 584-630). | Tube fluence/spreading and detector recording are coupled, complicating source or detector replacement. | Move source intensity/geometric gain conceptually into incident-radiation construction, then apply detector response. Retain an adapter reproducing existing `integral` and `intensity` exactly. Reuse current vector distance/gain data; no extra N=1 cache. Enforce at 0.6.1/0.6.3. | No material cost if vector reused; low refactoring complexity. A / 0.6.1, 0.6.3. |
| A5 Research configuration and diagnostic compatibility | `VirtualXRay` applies source-enable selection and depth windows before integration; source adapters carry numerical backend and material overrides; capture exposes total and per-source L maps (`virtualXRay.py` 464-491, 533-561, 658-683; `xrayProjection.py` 1017-1109, 1193-1237). | Treating these as device state, dropping them during transport extraction, or presenting source maps as independent measurements would break supported research workflows. | Keep depth limits and source selection in acquisition/transport configuration, backend and attenuation overrides in material/source inputs, and per-source maps as explicitly diagnostic outputs. Add focused regression cases in 0.6.2/0.6.5. | Configuration metadata is small; current maps already incur per-source image storage. A / 0.6.2-0.6.5. |
| B1 Tube emission | One point source; no angular intensity/anode heel model; device offsets are demonstrator constants (`virtualrtg_geometry.py` 18-25, 117-120). | Finite spot, direction-dependent output, heel effect and calibrated absolute intensity are absent. | Weighted samples and direction-dependent intensity slot are proposed; keep detailed emission inside replaceable TubeModel. | Spot samples increase ray work; compact sample metadata has low memory cost. B. |
| B2 Collimator | Ideal rectangular field with binary exposure; no blade position/thickness/transmission (`xrayProjection.py` 983-1004; `RectangularBeamGeometry` 356-413). | Real aperture transmission can depend on energy/geometry; finite spot can cause geometric penumbra. | Per-ray binary/fractional contract is already proposed. Blade physics and penumbra remain replaceable-model work. | N=1 unchanged; multi-origin work scales with rays; blade intersections add moderate complexity. B. |
| B3 Materials and primary transport | Volumes interpolate scalar voxels with fill/preprocessing and heuristic HU-like mapping (`xraySource.py` 702-727; `xrayProjection.py` 491-520). Sampling is stepped; Siddon integrates piecewise voxel attenuation (`xraySource.py` 729-747, 937-962). | Quantitative composition, tabulated energy coefficients and alternative accuracy are absent. | Preserve `XRaySampleSource`, source-local material overrides and direct-integral hook. New material maps belong behind source/material interfaces; a new integrator can be a backend without changing orchestration. | Sampling scales with rays x steps; Siddon with crossed voxels; model cost varies. B. |
| B4 Mesh representation | Mesh means constant-attenuation closed solid or thin shell (`xraySource.py` 986-1015); mesh uses direct integration (`xraySource.py` 1729-1757). | Heterogeneous mesh materials, partial volume and other topology policies are not represented. | Existing adapter boundary permits other representations; retain analytic BVH and experimental projected-list paths and per-object overrides. New mesh semantics may extend adapter input, not scene orchestration. | Material maps add memory/lookups; geometry work moderate/high. B. |
| B5 Detector response | Detector samples ideal pixel centers; signal is `L` or `exp(-L)` plus optional gain/floor (`xrayProjection.py` 607-630, 949-979). No blur, noise, saturation or temporal lag is modeled. | Energy response, finite pixel aperture, MTF and readout effects are absent. | Keep incident contribution separate from detector response; add effects inside replaceable detector model. Pixel subsampling can use ray batches; A2 provides scalar energy context. | Subsampling multiplies ray work; image effects scale with pixel count. B. |
| B6 Time and motion | One projection call uses one geometry config. `motion_frame_mode="all"` expands each frame into source instances in one scene (`virtualXRay.py` 571-599), rather than modeling time-weighted exposure. | Temporal poses/exposure durations are not simulated; current all-frame option combines frame sources as co-present attenuation. | Future scheduling may issue multiple state requests and accumulate detector-plane signal. No time axis or scheduler is required in current array contract; do not reinterpret existing `all` behavior. | Runtime scales with states; accumulation can stay one detector map. B. |
| C1 Scatter and secondary radiation implementation | Rays are straight and integration uses `origin + t*direction` samples (`xrayProjection.py` 1046-1165); no scatter transport exists. | Scatter needs an additional computational strategy and is not provided by the current integrators. | A3 prevents the contract from falsely requiring line integral, but does not make current transport implement scatter. Select a strategy only for a concrete accuracy/performance requirement. | Potentially high runtime/memory and method-specific complexity. C for implementation; A3 is boundary preparation. |
| C2 Capture provenance and model comparison | Capture is mutable and lacks full geometry/model snapshot (`xrayProjection.py` 797-804); DetectorImage metadata contains only limited geometry context (`detectorImage.py` 798-882); scene format persists settings separately (`sceneFormat.py` 357-427). | Exact reproduction or alternate material-model comparison may require rerunning transport; the cached arrays are not sufficient inputs to reconstruct every scene. | 0.6.4 already reviews snapshot ownership/metadata. Do not promise multiple physical-model evaluations from one capture; multiple display presentations remain supported without transport recomputation. | Metadata is small; retaining scalar/material intermediates can be large and is unjustified now. C / 0.6.4 review only. |
| C3 Per-source maps | Each `XRaySourceProjection` has per-source `L` and independently converted image (`xrayProjection.py` 785-794, 1193-1216). | Per-source intensity maps are diagnostic and do not sum to the composite when response is nonlinear. | Clarify diagnostic meaning; no contract change unless source-resolved measurements become a requirement. | Existing two maps/source already cost memory; clarification costs none. C. |

### Contract amendments justified by the register

Three additions to the revised focal-spot contracts are justified by concrete
current boundaries:

1. Allow optional scalar energy context on batched component evaluation and
   incident contributions. This prevents energy from being hard-coded in the
   projector without introducing energy arrays. Zero cost when omitted;
   enforce across 0.6.1 and 0.6.3.
2. Separate primary per-sample line-integral diagnostics from detector-plane
   incident contributions. This permits later non-line-integral contributions
   such as scatter without claiming the present engine computes them. No
   current output change; define at 0.6.2 and consume at 0.6.3.
3. Separate emitted fluence/geometric gain from detector recording while
   preserving current conversion as a compatibility adapter. Reuse existing
   vectorized gain data with no N=1 cache; enforce at 0.6.1/0.6.3.

Energy and acquisition state are scalar context per batch/contribution. Future
energy or time sequences can be evaluated and accumulated without mandatory
`[E,H,W]` or `[T,H,W]` arrays. This avoids premature generalized radiation
types and scheduling frameworks.

## 6. Coordinate-system and transformation rules

- **R (reference):** `VirtualXRay` local frame; geometry fields named `_ref`
  use millimeters. The VirtualXRay global transform maps R to world W.
- **W (world):** source adapters expose world-space bounds, point sampling and
  direct integrals. `VirtualXRay` computes descendant-to-R relative transforms
  before building those adapters. Current projector accepts an explicit
  `reference_transform`; production config supplies identity after the
  referenced geometry/source transforms are already expressed consistently.
- **Detector basis:** `u = normalize(cross(up, normal))`; `v = projected
  normalized up`; detector pixel steps include physical pixel pitch (mm).
  Pixel centers begin from detector origin and advance by `col*u_step +
  row*v_step`; preserve current handedness and center/origin convention.
- **Beam basis:** current `RectangularBeamGeometry` requires orthogonal unit
  vectors and `horizontal x vertical = axis`; field dimensions are full widths
  in cm at a reference distance in mm. The detector exposure predicate scales
  half-width by `5 * forward / reference_distance_mm`.
- **Units:** spatial position, pixel pitch, step and ray parameter are mm;
  field widths are cm; vector directions are dimensionless; energy settings
  are keV; scalar material input follows source scalar/HU-like conventions;
  attenuation coefficients and physical calibration are heuristic, so avoid
  claiming clinically calibrated units. Integral is `mu * mm` under current
  numerical conventions.
- **T_RS / T_WR:** device-link mapper applies `T_RS` station-to-reference
  transform, including mm translation, to its derived device poses. It returns
  R-frame positions/directions and explicitly does not apply `T_WR`; the core
  VirtualXRay object applies its global scene transform through existing
  infrastructure. Preserve this separation.
- **Tube/collimator arrangement:** tube pose maps tube-local focal sample
  positions and emission directions into R. Collimator pose is currently
  assumed aligned to the tube-derived beam frame, and its field dimensions are
  evaluated at the reference distance. A future independent collimator roll
  can use a named `T_tubeCollimator` (or direct `T_Rcollimator`) transform for
  its aperture frame. This is pose composition, not a current need for an
  `XRaySourceAssembly` class.

Future assembly composition must name transform direction in code/docs (for
example `T_Rtube`, `T_tubeCollimator`) and multiply homogeneous transforms in
one documented order. Do not infer emission direction from detector normal or
vice versa.

## 7. Acquisition, presentation and caching strategy

The separation is already functionally present: `project_and_cache()` retains
line-integral, raw detector, per-source, mask and overlay data; `DetectorImage`
copies the raw/package data and has independent display settings; GUI "Update
display" calls `_display_image_array` without projection. The DetectorImage
instance is retained/reused when still in the workspace. See
[`virtualXRay.py`](../../virtualXRay.py#L1019),
[`detectorImage.py`](../../detectorImage.py#L798), and
[`propVirtualXRay.py`](../../gui/propVirtualXRay.py#L1670).

Current acquisition capture is not immutable and the metadata is split across
mutable `VirtualXRay` attributes plus `XRayProjectionCapture`. Do not conflate
that shortcoming with a need for another cache. At 0.6.4, assess a coherent
capture/snapshot type and ownership, while keeping compatibility properties on
VirtualXRay if useful. Any schema/serialization change requires a separate
approved migration; this audit does not authorize it.

Caching incident radiation separately from recorded signal has measurable
value only when changing detector-response parameters can avoid costly work and
the response cannot be recomputed cheaply from the retained line integral.
For current monoenergetic data, `exp(-L)` and optional distance gain are
elementwise; a float32 incident array costs `4*H*W` bytes, duplicates the
recorded map by that amount, and gives little benefit over retaining `L`.
For a spectrum with E bins, a full incident cube costs `4*E*H*W` bytes and
may become prohibitive; benchmark whether recomputing response from retained
per-energy integrals beats that memory. Future energy-sensitive transport may
make it valuable, but it is not a present requirement.

Per-source image packages already duplicate per-source raw and integral maps;
record their memory in any later benchmark. Overlay primitives remain separate
from raw detector pixels and presentation composition.

## 8. Simulator and device-link compatibility

The accepted mapper converts simulator meter/radian poses to millimeter
positions and directions in VirtualXRay-local R. The tube pose maps a fixed
local focal-spot offset `(0,-0.43,0) m` and emission axis `(0,-1,0)` through
tube orientation; it also derives horizontal/vertical beam axes. Detector pose
maps a fixed active-center offset `(0,0.04,0) m` and its normal/up basis. The
plugin basis explicitly flips browser detector normal to satisfy the plugin's
pixel handedness convention. Collimator width/height map to full field size at
the current 1000 mm reference distance. See
[`virtualrtg_geometry.py`](../../device_link/virtualrtg_geometry.py#L18) and
[`virtualrtg_geometry.py`](../../device_link/virtualrtg_geometry.py#L100).

Integration applies only source position, detector center/normal/up and one
directional rectangular beam to the selected VirtualXRay, validates by calling
`build_geometry()`, rolls back on failure, refreshes, and explicitly does not
request a projection. See
[`integration.py`](../../device_link/integration.py#L65).

Current protocol v1 supplies tube, detector and collimator dimensions, but no
independently transmitted collimator orientation. Therefore the mapper assumes
collimator axes follow the tube frame; manual VirtualXRay adjustments can steer
and roll an effective beam, but those edits are not independently represented
in the incoming device state. New Tube/Collimator objects must accept this
derived relationship as the v1 default, with no protocol or browser changes.
Do not report the simulator's educational geometry as a validated hardware
specification.

## 9. Performance considerations and minimal benchmark protocol

Expensive work is dominated by volume interpolation/scalar-to-attenuation over
many active ray samples; Siddon voxel traversal; mesh intersection/BVH traversal
or projected intersection-stack construction; plus temporary NumPy arrays for
detector pixels, rays, masks, active indices and source outputs. Current vector
work is batched across rays. Python loops are over depth steps, sources,
motion groups and some mesh BVH/direct integration operations; the default
volume route does not call Python once per ray. Keep one component call per
batch and direct integration dispatch per source, not callbacks/objects per ray.

Minimal before/after comparison, once implementation is separately authorized:

1. Save one deterministic small volume-only scene (sampling) and one mixed
   volume + mesh scene; also include a Siddon volume case. Fix geometry, frame,
   mask, quality, interpolation, physics, and output mode.
2. Warm up once, then run each current/refactored case at least five times in
   the same process; report median wall time and range, using existing phase
   stats where available.
3. Record peak process memory with a practical process RSS high-water measure
   (or a consistently sampled RSS proxy) and report detector/source array byte
   totals. Note allocator/platform caveats.
4. Compare line-integral and recorded arrays by max absolute error, MAE and
   relative MAE with declared tolerances; require shape/dtype equality.
5. Compare boolean exposure masks exactly, including off-axis and rolled field
   fixtures. Preserve fully unexposed pixels and direct backend behavior.
6. Repeatedly change DetectorImage presentation and call Update display;
   verify no transport invocation and unchanged cached acquisition arrays.

This is a focused regression benchmark, not a broad parameter campaign. A
component split can add array copies, validation, Python calls per source,
temporary allocations, and separate image conversions; check each explicitly.

## 10. Migration roadmap

The user's candidate versions are retained as milestones, with a 0.6.0
characterization prerequisite and adjusted order where separation depends on
stable semantics. These are proposed scopes, not permission to implement.

### 0.6.1 -- Freeze current contracts; tube and collimator separation

- **Scope:** introduce only the tube-emission and collimator contracts plus
  point-source adapters into existing `XRayProjectionGeometry` and mask
  inputs. Define `FocalSpotSamples` as `(N,3)` positions plus normalized `(N,)`
  weights and define collimator evaluation for independent ray origins. Do not
  implement N>1 sampling, extra tracing, blades, or penumbra in this milestone.
  Preserve the existing N=1 geometry builder and rectangular mask fast path.
- **Likely files:** `xray/xrayProjection.py`, `virtualXRay.py`, focused
  geometry/collimation tests, design docs; only touch `device_link` if adapting
  model constructors is unavoidable (mapping output/protocol stays identical).
- **Contracts:** explicit tube-local sample positions transformed once to R;
  normalized sample weights; arbitrary independent origins at the collimator
  boundary; legacy shared `bool[H,W]` mask for N=1. Frames/units are R, mm
  positions and cm field dimensions. Arrays have documented ownership and are
  not expanded/tiled for N=1. Permit optional scalar energy context and scalar
  fluence normalization at model boundaries, with no spectral arrays and no
  behavior when those values are omitted.
- **Compatibility:** keep scene payload schema, `build_geometry()` behavior,
  `RectangularBeamGeometry` compatibility and `project_capture()` call path.
- **Numerical checks:** existing geometry and collimation results; exact mask
  equality for aligned, steered, rolled, partial/no-overlap cases; source and
  detector rays unchanged. Verify N=1 energy-context omission preserves exact
  output and optional context is passed through without changing mask values.
- **Performance:** compare ray setup/mask timing and allocation count proxy;
  N=1 must allocate no sample-expanded ray or float transmission buffers and
  retain the current one-mask cost.
- **Acceptance:** byte/elementwise-identical mask and matching geometry/rays;
  settings round-trip unchanged; no GUI/device protocol changes. Contracts can
  represent arbitrary sample arrays and
  sample-dependent mask/transmission without later changing field/frame/shape
  semantics; current N=1 results and speed remain unchanged. Also reserve
  scalar per-evaluation energy context across emission/material/collimator
  settings. Depth windows, source selection, material overrides and numerical
  method choices remain acquisition/source configuration, not tube or
  collimator state. No N>1 physics or spectral execution is required for
  acceptance.
- **Exclude:** finite-spot execution, additional ray tracing, blade models,
  independent device collimator pose input, serialization changes, core scene
  nodes, protocol changes.

### 0.6.2 -- Transport model extraction

- **Scope:** move current projector work behind one ordinary
  `XRayTransportModel` batch operation with independent origins and directions.
  Preserve existing implementations and source adapter hook. Keep primary
  line-integral diagnostics distinct from physical detector-plane incident
  contributions. The result boundary must permit contributions that are not
  line integrals, while current execution remains primary-only. Retain
  focal-sample identity (per-sample maps or chunk-equivalent) although only N=1
  is executed in this milestone. No scatter or N>1 tracing is added.
- **Likely files:** `xray/xrayProjection.py`, `xray/xraySource.py`,
  `virtualXRay.py`, projection tests and benchmark helper.
- **Contracts:** chunk-capable ray batch with detector-pixel/sample mapping,
  `XRayScene` material sources, primary L diagnostics, separate incident
  contribution channel and optional scalar energy context per batch. The N=1 fast
  path reuses current `(H*W,3)` arrays. Chunking must not allocate all sample
  by detector rays at once; dispatch remains per source/chunk, never per ray.
  Preserve ray/planar depth-window clipping as an integration-domain
  restriction, source enable selection, per-source integrals, material
  overrides and both volume and mesh backend choices. Geometry batches remain
  reusable across methods where their contracts match; backend-specific
  traversal may still require a distinct computational strategy.
- **Compatibility:** old `XRayScene.project*`, `XRayProjector` imports and
  project wrappers should remain facades during migration; no per-ray dispatch.
- **Numerical checks:** sampling, Siddon, analytic BVH and experimental
  projected-list outputs; mixed-source summation; clipping and ray/planar depth
  windows; enabled/disabled source selection; global and per-source material
  response settings; per-source integral maps.
- **Performance:** focused protocol above; especially peak arrays and Python
  call overhead versus existing projector.
- **Acceptance:** N=1 line integrals and masks match current outputs within
  established tolerance; all sample identity and chunk mapping contracts are
  explicit and adequate for later multi-origin rays; preserve backend choices
  and point-source performance. Verify a detector-plane contribution can be
  represented independently of L in the contract, and that omitted energy
  context preserves the existing N=1 result. No N>1 tracing is needed yet.
  Include a depth-clipped projection and per-source diagnostic-output
  regression cases by integration acceptance.
- **Exclude:** N>1/focal-spot execution, spectral integration, scatter
  implementation, detector response conversion, source model redesign,
  behavior changes to mesh uncertainty handling.

### 0.6.3 -- Detector geometry/response separation

- **Scope:** define `XRayDetector` settings/response and move
  `integral_to_image` semantics into a compatibility response implementation.
  Detector geometry initially adapts the existing detector pixel lattice;
  transport consumes ray batch from shared geometry construction.
- **Likely files:** `xray/xrayProjection.py`, `virtualXRay.py`,
  `detectorImage.py` only if naming/metadata must be clarified, physics tests.
- **Contracts:** detector response consumes incident detector-plane
  contributions, optional scalar energy/context tags and geometry. It may not
  require L as the only input. Current N=1 compatibility adapter derives its
  incident value from L/emission/transmission and preserves output modes;
  primary L remains a diagnostic. Apply transmission once. No extended-source
  execution or scatter implementation is required in 0.6.3.
- **Compatibility:** exact existing `integral` and `intensity` outputs including
  inverse-square/floor semantics; public config settings remain readable and
  serializable as currently.
- **Numerical checks:** Beer-Lambert mode, integral mode, distance gain,
  exposure gating and masks; energy context omitted reproduces the exact
  existing values. Confirm a detector-plane additive contribution can reach
  the response boundary without being encoded as L; no presentation changes to
  stored raw values.
- **Performance:** response time and extra-array memory; no extra response map
  unless it is retained for a demonstrated use.
- **Acceptance:** same output within tolerance; "raw" documented as pre-display
  recorded image under selected simplified response, not calibrated counts.
  Emission/geometric gain and detector recording have distinct inputs even when
  the legacy adapter keeps their current arithmetic. Preserve existing output
  names and arithmetic; use current line-integral/per-source fields for
  diagnostics and add only a compact semantic descriptor if new outputs
  otherwise become ambiguous.
- **Exclude:** real detector material/charge physics, MTF, noise, lag, spectra,
  changes to scene format.

### 0.6.4 -- Capture and cache ownership review

- **Scope:** inspect whether one coherent acquisition capture is useful; define
  immutable/snapshot ownership and metadata completeness. Preserve existing
  aliases and `DetectorImage` separation. Add no intermediate incident cache
  unless benchmark establishes value.
- **Likely files:** `xray/xrayProjection.py`, `virtualXRay.py`,
  `detectorImage.py`, capture/package tests; `sceneFormat.py` only under a
  separately reviewed backward-compatible serialization plan.
- **Contracts:** capture holds raw transport result, recorded output, mask,
  per-source maps, stats and geometry/model metadata; arrays copied once or
  frozen at ownership boundary. Record depth restriction, selected source
  identities and backend/material settings only to the extent justified for
  identifying a deliberate research intervention; do not promise full scene
  provenance or alternate-model replay.
- **Compatibility:** `last_raw_projection`, `last_line_integral_projection`,
  per-source fields, export/import, scene load and repeated presentations.
- **Numerical checks:** capture roundtrip and unchanged outputs; display update
  must not invoke transport; mask and metadata correspondence.
- **Performance:** capture copy count, retained bytes, peak RSS. Incident cache
  only if response recomputation is materially costly or required by future
  energy model.
- **Acceptance:** one unambiguous acquisition snapshot, no stale aliases,
  presentation reuse and documented memory increase/decrease.
- **Exclude:** file-format/schema revision without separate decision, GUI
  redesign, spectral cache, annotations in raw buffers.

### 0.6.5 -- Integration and validation

- **Scope:** remove only redundant compatibility glue proven unnecessary;
  update technical docs and verify full plugin integrations across GUI, scene
  load/save, device mapping, mesh/volume backends and detector presentation.
- **Likely files:** integration points and plugin docs/tests, only after stages
  0.6.1-0.6.4 are accepted.
- **Contracts:** end-to-end capture snapshot and v1 device geometry mapping;
  no extra scene objects.
- **Compatibility:** protocol v1; scene XML and ATMDL; DetectorImage remains a
  separate scene object and multiple presentations remain possible.
- **Numerical checks:** end-to-end fixtures from each backend and exact exposure
  masks; presentation-only update/no recomputation.
- **Performance:** one small benchmark suite as described above.
- **Acceptance:** all prior contracts, no unresolved numerical/performance
  regression, docs and tests agree with active path. Include a depth-clipped
  projection and per-source diagnostic-output regression case, alongside a
  normal physically motivated acquisition and the accepted simulator/device
  link workflow.
- **Exclude:** browser simulator/protocol changes, calibration claims, broad
  benchmarking, release closure unless separately accepted.

## 11. Risks, unresolved questions and alternatives

### Minimality levels

- **A. Required contract compatibility now:** `FocalSpotSamples` shape/frame
  and normalized-weight rules; independent ray origins/directions; per-ray
  collimator acceptance/transmission semantics; sample-to-pixel identity;
  chunkable transport results that retain sample identity; N=1 compatibility
  branches and exact current boolean-mask behavior; scalar energy/context
  pass-through; distinct primary line-integral diagnostics and detector-plane
  incident contributions; separation of emission/gain from detector recording.
- **B. Add later without changing those contracts:** N>1 focal-spot sampling,
  on-demand sample generation, direction-dependent intensity evaluation,
  sampled spectral batches with scalar energy/fluence weights, fractional
  transmission, independent collimator pose/roll, temporal state scheduling,
  and alternate chunk sizes. These need execution and validation, but not a
  new component boundary.
- **C. Speculative and excluded from current design decisions:** regular source
  grids, mandatory spectral arrays, focal-spot blur/penumbra implementation,
  blade material physics, scatter algorithm/Monte Carlo, detector MTF/noise/lag,
  GPU kernels, and a universal radiation abstraction. The minimal incident
  contribution boundary is A; advanced calculation methods remain C. Add only
  when a concrete requirement justifies them.

The main N=1 overhead risk is wrapping the old source position in arrays,
repeating it H*W times, allocating sample indices, converting the boolean mask
to float, or routing every ray through Python objects. Avoid all five: represent
the single source with singleton metadata and take the existing ray and mask
path. General N>1 allocation is bounded by chunk size; an implementation must
report peak temporary memory and preserve pixel/sample mapping.

For the existing point source, one detector pixel still maps to one ray, so
the ray count is H*W. Keep the current flat pixel ordering and vectorized
`(H*W,3)` origin/direction construction, one `bool[H,W]` mask, and direct
per-source dispatch. No extra source dimension or source-sample Python loop is
needed when N=1.

- **"Source" ambiguity:** `XRaySampleSource` means an anatomical/material
  source, while `XRayTube` is radiation emission. New names must distinguish
  these roles.
- **Physics-model boundary:** attenuation coefficients depend on scalar
  material maps; present signal conversion sits in that same class. Decide
  whether attenuation mapping remains a `MaterialResponse` owned by sources
  or a `XRayPhysicsModel` sub-object. Prefer an ordinary composition object,
  not strategy inheritance, and retain per-source overrides.
- **Geometric coupling:** collimator exposure is currently a detector mask
  built from rays from one point source. Moving mask creation must preserve
  clipping and projected-field convention exactly.
- **Ray coordinate contract:** source adapters consume W-space rays while
  settings are R-space. Make transformation application single-owner to avoid
  double transforms, especially with nested transforms and `Motion` frames.
- **Value naming:** "raw projection" means selected detector response before
  display, and line integral is separately cached. New metadata should remove
  ambiguity without silently changing existing export meanings.
- **Capture immutability:** freezing arrays may break code that currently
  mutates cache references. First audit writers and aliases; copy only at the
  capture/cache boundary.
- **Serializer burden:** geometry and physics settings currently live on
  VirtualXRay. Persisting models directly could change schemas and risk
  defaults/older scenes. First adapters should remain transient and serialize
  the existing settings.
- **Assembly object alternative:** a formal assembly is useful only once tube
  and collimator have independently meaningful local transforms. Until then,
  explicit configuration relationships are smaller and safer.
- **Acquisition-system alternative:** a dedicated orchestrator could later own
  asynchronous lifecycle and multi-exposure sequences. At present it duplicates
  VirtualXRay and should not be added for symmetry.
- **Benchmark reproducibility:** process RSS includes Qt/allocator state and
  repeated runs may reuse BVH/stack caches. Define cold versus warm results and
  report both only when cache construction is material.
- **Energy contract detail:** defer normalization of a real spectrum, material
  coefficient tables, and energy-dependent response curves until one supported
  physical model is selected. The current proposal reserves scalar context and
  contribution identity only.
- **Non-primary contribution method:** the interface permits an additional
  detector-plane contribution, but how that contribution is computed and
  validated remains open. Do not infer that the present straight-ray transport
  can produce scatter.
- **Temporal exposure semantics:** a future acquisition schedule needs a rule
  for exposure weighting, motion state and detector persistence. Existing
  `motion_frame_mode="all"` expands frames into co-present source adapters and
  must not be reinterpreted as temporal integration.
- **Model-comparison provenance:** decide later whether a capture should retain
  enough scene/material input to compare alternative transport/attenuation
  models, or whether rerunning the acquisition is acceptable. Current raw and
  line-integral caches do not guarantee this capability.

## 12. Recommended first implementation step

After explicit architecture acceptance, begin with a narrowly scoped
0.6.1-compatible characterization patch: codify current geometry, mask, ray,
and signal behavior with deterministic tests and a small fixed benchmark case.
Then implement only lightweight tube/collimator value/config objects that adapt
to existing `XRayProjectionGeometry`; do not move the ray kernel in the same
change. This provides a numerical baseline and keeps protocol, scene graph,
serialization, GUI, and detector presentation unchanged.

No production code was changed as part of this design audit.
