# virtRTG 0.6.2 Recovery and Transport Baseline Handoff

Status: intermediate numerical and performance baseline preparation. This is
not a completed 0.6.2 implementation, and no transport extraction has started.

## Accepted implementation recovery

The nested plugin checkout is at accepted 0.6.1 commit
`8b7ff41883b329cfdb11afb1243c1ce34e59a81e`. At that commit,
`xray/xrayProjection.py` defines `XRayProjector` as the implementation owner;
there is no `XRayTransportModel` rename or compatibility subclass in the
checked-in source. The accepted 0.6.1 implementation was therefore already
present and was left unchanged. No production projection code was edited.

## Minimal future extraction boundary

Keep `XRayProjector.project()` responsible for detector pixel-grid construction,
the current N=1 ray construction, optimized rectangular exposure masking,
scene/AABB and optional depth clipping, `integral_to_image()` conversion,
capture assembly, and statistics presentation. VirtualXRay scene traversal
and source selection also stay outside.

The smallest coherent future operation is the existing section beginning with
direct-versus-marched source dispatch and ending after total/per-source
attenuation accumulation and primary integration diagnostics. Initially keep
clipping in the projector to avoid widening the extraction patch. Inputs are
the already prepared `ray_origins[H*W,3]` and normalized
`ray_directions[H*W,3]` in world millimeters; clipped `t_starts`/`t_ends` in
millimeters; the ascending flattened C-order hit pixel indices; the detector
shape; `step_mm`; the geometry/reference transform needed by direct adapters;
the shared physics model; and selected `XRaySampleSource` adapters. Rays with
false exposure or no valid clipped intersection are omitted from integration.
Source adapters return line integrals in attenuation-coefficient times mm;
sampled sources accumulate `mu * step_mm`. The output is one float32 total
integral per hit ray, one float32 contribution per source per hit ray, and
work/per-source integration diagnostics. The caller scatters those values into
row-major detector maps. This operation does not own detector pixel creation,
final signal conversion, exposure-mask generation, capture/presentation, or
scene traversal.

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

Recommendation for the later extraction: make a narrow orchestration-only
change around the identified integration block, keep source adapters and all
current numerical kernels intact, then run the exact saved runner against the
new code and compare with these immutable arrays and tolerances before
considering any further structural work.
