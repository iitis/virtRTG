# virtRTG 0.6.2 Accepted Transport Extraction Handoff

Status: accepted implementation. `XRayProjector` delegates prepared-ray
attenuation integration to the independent `XRayTransportModel`; the frozen
references, numerical comparisons, and final plugin suite are recorded below.

## Accepted implementation recovery

The nested plugin checkout is at accepted 0.6.1 commit
`8b7ff41883b329cfdb11afb1243c1ce34e59a81e`. At that commit,
`xray/xrayProjection.py` defines `XRayProjector` as the implementation owner;
there is no `XRayTransportModel` rename or compatibility subclass in the
checked-in source. The accepted 0.6.1 implementation was therefore already
present and was left unchanged. No production projection code was edited.

## Accepted extraction boundary

Keep `XRayProjector.project()` responsible for detector pixel-grid construction,
the current N=1 ray construction, optimized rectangular exposure masking,
scene/AABB and optional depth clipping, `integral_to_image()` conversion,
capture assembly, and statistics presentation. VirtualXRay scene traversal
and source selection also stay outside.

The extracted operation is the existing section beginning with
direct-versus-marched source dispatch and ending after total/per-source
attenuation accumulation and primary integration diagnostics. Clipping remains
in the projector. Inputs are
the already prepared `ray_origins[H*W,3]` and normalized
`ray_directions[H*W,3]` in world millimeters; clipped `t_starts`/`t_ends` in
millimeters; the ascending flattened C-order hit pixel indices; the detector
shape; `step_mm`; the geometry/reference transform needed by direct adapters;
the shared physics model; and selected `XRaySampleSource` adapters. Rays with
false exposure or no valid clipped intersection are omitted from integration.
Source adapters return line integrals in attenuation-coefficient times mm;
sampled sources accumulate `mu * step_mm`. The result contains flattened
float32 total and per-source detector maps plus work/per-source integration
diagnostics. Hit-mask and hit-index consistency is currently guaranteed by the
projector caller. This operation does not own detector pixel creation, ray
generation, clipping, final signal conversion, exposure-mask generation,
capture/presentation, or scene traversal.

## Frozen baseline artifacts

The baseline was generated before any production transport edit from commit
`8b7ff41883b329cfdb11afb1243c1ce34e59a81e` using
`benchmarks/0.6.2/run_baseline.py` and the repository `.venv` Python 3.11.9.
The nested worktree already contained the staged 0.6.2 handoff and the new
benchmark directory during execution; `baseline.json` records this exact
worktree state. The baseline artifacts are frozen and must not be regenerated
or overwritten during extraction.

The 12 compact, deterministic 5x5 detector cases cover cone sampling, cone
Siddon, analytic mesh BVH, experimental projected-intersection-list mesh,
mixed volume/mesh, two-source maps, ray and planar depth restrictions,
collimation enabled and disabled, integral and intensity output, and parallel
sampling. The runner constructs real `VolumetricXRaySource` and `MeshXRaySource`
instances, records adapter/backend identity, checks actual integration work,
checks exact exposure masks, and rejects non-finite/missing results. It does
not infer backend execution from the skipped source skeleton tests.

`baseline_arrays.npz` contains 64 arrays: total integrals, raw detector outputs,
exposure masks, and per-source integrals/outputs for the cases that return
sources. Shapes and dtypes, exact masks, source labels/types, workload counts,
phase timings, and per-source integration statistics are recorded in
`baseline.json`. The generated NPZ inventory was re-opened and validated for
finite values and matching array count.

Tolerances were fixed before any extraction: exact mask equality; exact shape
and dtype equality; float attenuation integrals and per-source maps use
`atol=1e-6`, `rtol=1e-6`; raw output uses the same tolerance; relative error
comparisons use denominator `max(abs(reference), 1e-7)` and are reported
alongside absolute errors. Integer counts, backend names, source count, and
projection mode require exact equality. Floating integration timings are
descriptive and are not acceptance-equality fields.

### Performance baseline

Each benchmark uses two warm-ups and five measured calls. Wall-time medians and
observed min/max spread are in `baseline.json`; phase medians distinguish
direct integration or slab marching from total projection. `tracemalloc` peak
is recorded only as a Python-allocation proxy and explicitly does not capture
all NumPy native allocations or process peak RSS. Retained capture-array bytes
are separately recorded; no peak-memory claim is made.

| Case | Median | Observed range | Integration phase median | Python allocation proxy | Retained capture arrays |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume sampling | 6.694 ms | 6.475-8.004 ms | marching 5.349 ms | 15,041 B | 225 B |
| Volume Siddon | 2.786 ms | 2.754-3.957 ms | direct 1.610 ms | 61,421 B | 225 B |
| Mixed volume and analytic mesh | 47.451 ms | 46.646-49.553 ms | direct 40.474 ms; marching 5.603 ms | 21,091 B | 225 B |

Environment: Windows 10, x64, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1,
OpenCV 4.13.0, pydicom 3.0.2, 10 logical CPUs (Intel Family 6 Model 198).
The scenes use a 5x5 detector, 0.5 mm step, 5x5x5 scalar-zero volume, and a
closed 12-triangle cube. Mesh integration uses solid mode. Each iteration
constructs new source adapters outside the timer, so BVH/stack caches are cold
at the start of every measured projection. Warm-ups apply to the runtime and
operating environment, not those source caches.
The environment had no `psutil`, so physical RAM and process RSS were not
measured.

SHA-256: `baseline.json`
`E4044FC47A8B18D8D8275B198D4BE560B168260BA1912419CA61E2B6B8F1D062`;
`baseline_arrays.npz`
`C86967088175DB26D4AADE1AA65FC7B070E5D91D07A64997C44D0512E3B30C4A`.
These digests are also recorded in `benchmarks/0.6.2/baseline_hashes.json` and
verified by supplement, after, and compare modes.

## Final baseline validation and runner protection

The runner requires an explicit `--mode`. `--mode baseline
--confirm-baseline-write` is refused if either original baseline file exists.
For initial creation in a clean reference directory, it also compares
`xray/xrayProjection.py` and `xray/xraySource.py` byte-for-byte against accepted
commit `8b7ff41883b329cfdb11afb1243c1ce34e59a81e`; baseline creation is refused
after those production sources change. The original files' SHA-256 values
above were checked before and after validation and still match.

The original `cone_depth_planar` and `parallel_volume_sampling` references are
intentionally unchanged even though they trace zero rays. The planar case put
its slab at world z=9..13 while the volume occupies z=-2..2. The parallel case
started at detector z=8 and pointed +Z away from the volume. Corrected cases
are frozen separately in `supplemental_arrays.npz` and `supplemental.json`:

| Supplementary case | Traced pixels | Work count | Integral sum |
| --- | ---: | ---: | ---: |
| Planar depth, origin z=-8 and limits 7..9 mm | 25 | 101 | 1.010 |
| Parallel sampling, detector rays directed -Z | 25 | 225 | 2.250 |
| Ray depth, limits 9..13 mm | 25 | 75 | 0.750 |

Every corrected case asserts nonzero traced pixels, integration work, and
attenuation. Both depth cases also assert that the per-pixel integrals do not
exceed unrestricted cone integrals and that total attenuation is reduced. The
original frozen cases remain available for exact historical comparison.

The bounded larger performance reference is in `large_performance.json`. It
uses a 32x32x16 volume (32x32x16 mm), a closed 12-face mesh scaled to 16 mm
edges, and 0.5 mm steps. Each measurement has two warm-ups and five samples;
source adapters are recreated before each timed projection, so BVH caches are
cold. JSON records environment, ray counts, integration work, phase medians,
array sizes, and the allocation proxy.

| Case | Detector | Median | Observed range | Traced rays | Work |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume sampling | 64x64 | 35.906 ms | 34.355-41.116 ms | 4,096 | 117,049 |
| Volume Siddon | 32x32 | 6.525 ms | 6.303-7.726 ms | 1,024 | 1,024 |
| Mixed sampling and mesh | 32x32 | 1.701 s | 1.629-1.711 s | 1,024 | 35,468 |

The mixed case is substantially slower because the analytic mesh backend
visits the triangle BVH for each ray. These timings are reference measurements,
not an acceptable-throughput claim. Process peak RSS was not measured.

### Runner usage

From the repository root:

```powershell
.venv\Scripts\python.exe plugins\virtRTG\.codex\benchmarks\0.6.2\run_baseline.py --mode supplement
.venv\Scripts\python.exe plugins\virtRTG\.codex\benchmarks\0.6.2\run_baseline.py --mode after
.venv\Scripts\python.exe plugins\virtRTG\.codex\benchmarks\0.6.2\run_baseline.py --mode compare
```

`supplement` creates the corrected coverage and larger performance references
without changing the original baseline files. `--refresh-supplement` can
deliberately recreate only those supplementary files and is refused if
production sources differ from accepted 0.6.1. `after` (alias `refactored`)
writes after outputs and a comparison; it requires the supplement and verifies
the original file hashes first. `compare` compares existing artifacts without
projecting again. Case and array inventories must match exactly: the original
inventory plus the declared `supplemental__` prefix. Shapes and dtypes and
masks compare exactly. Floating maps use the fixed absolute/relative tolerances.
Non-timing stats, backend identities, and source ordering compare exactly.
Timing samples compare as distributions with medians, spreads, and ratios, not
as deterministic numerical values.

After mode was run against unchanged accepted production code to validate the
infrastructure. All 64 original and 15 supplementary arrays passed with zero
error. This self-comparison is not evidence of transport extraction or
post-refactoring equivalence.

## Verification and remaining work

Baseline runner completed successfully: 12/12 cases, 64 arrays, all required
backends present and finite. Actual Siddon work and both standalone mesh
backends' integration statistics were checked. The detailed measured samples
and environment are the machine-readable record in
`benchmarks/0.6.2/baseline.json`.

The full plugin test suite passed: **117 passed, 5 skipped**. The five skips
are the pre-existing empty `test_source_skeleton.py` placeholders and are not
backend evidence. Supplementary validation, after-mode self-comparison, runner
compilation, baseline overwrite refusal, and `git diff --check` passed.

Known coverage limits: scenes are synthetic; there is no claim about clinical
data, large-volume scaling, peak RSS, or GUI behavior. Parallel mode is covered
for sampling, not each direct backend. The original frozen planar and parallel
cases remain ineffective by design; corrected supplementary cases now cover
those behaviors. These references describe the accepted single-focal-source
implementation only.

The 0.6.2 implementation and comparison results are recorded in the accepted
extraction section below. No additional transport work is part of this
handoff.

## Accepted 0.6.2 minimal transport extraction

The extraction described here is accepted. The earlier sections preserve the
baseline preparation record; this section records the accepted implementation
and supersedes historical statements that extraction remained future work.

### Frozen reference verification and artifact inventory

Before production edits, the nested plugin was at accepted 0.6.1 commit
`8b7ff41883b329cfdb11afb1243c1ce34e59a81e`. The normalized Git diff for
`xray/xrayProjection.py` and `xray/xraySource.py` against that commit was empty.
The baseline file hashes matched the accepted constants in the benchmark
runner. Original and supplementary arrays and the larger performance reference
were present. The frozen references were not regenerated or modified.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `baseline.json` | 31,557 | `E4044FC47A8B18D8D8275B198D4BE560B168260BA1912419CA61E2B6B8F1D062` |
| `baseline_arrays.npz` | 17,054 | `C86967088175DB26D4AADE1AA65FC7B070E5D91D07A64997C44D0512E3B30C4A` |
| `supplemental.json` | 11,031 | `7C5FFC5415D38F19712B0A1344B6F94BDA21E2B065DF49A205482DD58A7B0ED5` |
| `supplemental_arrays.npz` | 4,059 | `309CE01A5780C6692935D4377E4D7EB4218291C33FC1A2A334E489FD99460BC7` |
| `large_performance.json` | 4,453 | `A89C3F11B8EB1D0B243056A8FBAA6A054CC02F0F769CDCDAAF59B81D00EF1B28` |

The runner now compares `baseline_hashes.json` to its hard-coded authoritative
hash mapping and always verifies files against those hard-coded values. Editing
the manifest alone cannot change the accepted hashes. The former `after.json`,
`after_arrays.npz`, and `comparison.json` self-validation outputs were replaced
by the post-refactoring run; they are not used as evidence below.

### Files changed and responsibility boundary

- `xray/xrayTransport.py` adds `XRayTransportModel` and
  `XRayTransportResult`. The model performs the existing source classification,
  direct integral dispatch, sampled depth loop, attenuation accumulation,
  work counts, and direct/marched source diagnostics.
- `xray/xrayProjection.py` delegates integration to the model. It continues to
  own detector pixel centers, the current N=1 cone/parallel rays, optimized
  rectangular exposure mask, scene AABB and depth clipping, compatibility
  `integral_to_image` conversions, per-source output maps, capture assembly,
  and projection statistics.
- `tests/unit/test_xray_transport.py` covers the prepared-hit-ray contract and
  flat result maps at the new component boundary.
- `.codex/benchmarks/0.6.2/run_baseline.py` hardens frozen hash validation.
- `.codex/benchmarks/0.6.2/after.json`, `after_arrays.npz`, and
  `comparison.json` now contain the new run's reports.
- This handoff records the boundary, artifact inventory, and verification.

The transport input is one full detector-flattened C-order batch of world-space
ray origins and normalized directions, clipped start/end parameters in mm,
the existing hit mask and ascending hit indices, detector shape, step size,
selected source adapters, physics model, geometry, and reference transform.
Direct adapters receive only hit rows; sampled sources retain the original
batched depth-step loop and do not receive sample-expanded rays. Origins remain
independent in the interface, while projection generation retains the current
N=1 cone and parallel paths and existing coordinate frames.

The result contains a full flattened float32 total primary attenuation array,
one flattened float32 primary contribution array per source, work count, traced
pixel count, phase timings, and source diagnostics. Full detector-flat arrays
retain the prior zero-for-unhit layout and avoid an additional scatter/copy in
the projector. Primary attenuation remains distinct from incident radiation;
no placeholder scatter, spectrum, detector, or focal-sample arrays were added.

The source dispatch and numerical operations were moved without rewriting the
sampling, Siddon, analytic mesh BVH, or projected-intersection-list backends.
Direct sources remain accumulated before sampled sources, preserving the prior
group ordering, with per-source maps retaining scene source order. Source
material overrides and physics resolution still occur at their prior points.
Experimental/research restrictions and diagnostics remain on the same shared
path: depth clipping and source selection stay in the existing scene/projector
flow, projected-intersection-list remains experimental, and no research-only
projection engine or new backend restriction was introduced.

The new component is independent computational code, not a scene node. No
`VirtualXRay`, `XRayScene`, `DetectorImage`, GUI, serialization, or device-link
protocol changes were needed. The stage API is `integrate(prepared batch)`
rather than a whole acquisition `project()` request, matching the intentionally
narrow 0.6.2 boundary. Detector-response extraction remains for 0.6.3.

### Numerical regression and plugin tests

`run_baseline.py --mode after` completed against the original and supplementary
frozen references. All **64 original** and **15 supplementary** arrays passed.
Across both groups, maximum absolute error, MAE, and relative error were all
zero. Exposure masks, shapes, dtypes, source labels/order/backend identities,
non-timing statistics, and work counts matched exactly. The effective planar
depth, ray-depth, and parallel integration cases in the supplementary set all
ran with their expected nonzero work. The historical ineffective original
planar and parallel cases remain unchanged.

The full plugin suite passed: **118 passed, 5 skipped**. The skips are the
existing empty source-skeleton placeholders.

### Performance and memory comparison

Values below are medians and observed min-max spreads over five measurements;
they do not establish statistical significance. Timings are milliseconds.
Allocation is the Python `tracemalloc` proxy, not total NumPy memory or RSS.
Retained output bytes are measured capture arrays.

| Benchmark | Before median (spread) | After median (spread) | Ratio | Integration phase before -> after | Python allocation before -> after | Retained bytes before -> after |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Sampling, 5x5 | 6.694 (6.475-8.004) | 4.770 (4.658-5.807) | 0.713 | marching 5.349 -> 3.550 ms | 15,041 -> 15,261 B | 225 -> 225 |
| Siddon, 5x5 | 2.786 (2.754-3.957) | 2.635 (2.592-3.786) | 0.946 | direct 1.610 -> 1.522 ms | 61,421 -> 61,485 B | 225 -> 225 |
| Mixed sampling/mesh, 5x5 | 47.451 (46.646-49.553) | 45.778 (44.995-46.715) | 0.965 | direct 40.474 -> 40.040 ms; marching 5.603 -> 3.663 ms | 21,091 -> 21,227 B | 225 -> 225 |
| Sampling, 64x64 | 35.906 (34.355-41.116) | 26.418 (24.583-28.015) | 0.736 | marching 34.192 -> 24.865 ms | 690,436 -> 690,340 B | 36,864 -> 36,864 |
| Siddon, 32x32 | 6.525 (6.303-7.726) | 6.578 (6.195-7.757) | 1.008 | direct 5.181 -> 5.274 ms | 8,084,198 -> 8,084,206 B | 9,216 -> 9,216 |
| Mixed sampling/mesh, 32x32 | 1,700.811 (1,629.454-1,711.323) | 1,753.094 (1,677.034-1,843.006) | 1.031 | direct 1,675.309 -> 1,732.598 ms; marching 24.792 -> 19.143 ms | 189,717 -> 189,725 B | 9,216 -> 9,216 |

Sampling medians are about 26-29% lower in both sizes. Inspection shows the
depth-step arithmetic and loop were carried over unchanged; the Siddon direct
phase and analytic mesh phase do not show the same reduction. The timing shift
is therefore recorded as observed and unexplained, not attributed to an
optimization. The mixed 32x32 total is 3.1% higher, mainly in direct mesh
timing; its before/after spreads overlap. No kernel optimization or array-copy
change was made, and five repetitions are insufficient for significance
claims.

The sampled integration API takes both `hit_mask` and `hit_indices`; their
consistency is currently guaranteed by `XRayProjector`, which derives both
from the same clipped ray mask. The focused transport unit test exercises
direct prepared-ray dispatch. A standalone sampled-path transport unit test is
optional future coverage; sampled integration is exercised by the frozen
end-to-end projection comparisons. The observed sampling timing reductions
were not causally established and are not claimed as an optimization.

### Remaining limitations and architecture notes

N>1 focal-spot transport, independent focal-sample weights, spectra, scatter,
physical blade transport, and detector response remain out of scope. Projection
generation and clipping remain in `XRayProjector`; transport receives prepared
and clipped rays. The transport result contains primary attenuation integrals,
not a complete incident-radiation model. The current N=1 integration boundary
can receive independent ray origins, but focal-sample expansion is not
implemented. The output remains the legacy integral/intensity compatibility
value, not calibrated incident radiation or detector counts. Synthetic
benchmark scenes do not establish clinical behavior, process peak RSS, or GUI
behavior.

The only material deviation from the broader 0.6.0 architecture proposal is
that the extracted operation accepts already prepared/clipped ray batches and
returns primary integrals, rather than owning a complete transport,
incident-radiation, and capture request. This is the narrower staged boundary
accepted for 0.6.2; detector response and acquisition-contract work remain
staged for later versions.
