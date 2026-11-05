# virtRTG 0.6.3 Ideal X-Ray Detector Handoff

Status: accepted and closed. The 0.6.3 implementation passed review and the
release documentation and verification record are finalized.

## Files changed

- `xray/xrayDetector.py`: new lightweight computational detector component.
- `xray/xrayProjection.py`: default detector construction, intensity-only
  response call, and clarification of the legacy signal's physical meaning.
- `tests/unit/test_projection_additional.py`: detector identity and projector
  integration coverage.
- `.codex/history/HANDOFF-0.6.3.md`: this handoff.
- `.codex/baseline.md`: accepted architecture and detector contract.
- `CHANGELOG.md`: concise 0.6.3 entry.

## Detector contract and pipeline position

`XRayDetector.record(incident_signal)` returns the exact input object. It does
not copy, convert, or modify values, shape, or dtype. The implementation adds
no noise, blur, gain, nonlinear response, or quantization. It is a plain
computational object, not a scene node.

`XRayProjector` retains ray generation, collimation, clipping, and calls the
existing `XRayPhysicsModel.integral_to_image()` after transport integration.
When `output_mode` is `intensity`, the resulting simplified incident signal is
passed once through `XRayDetector.record()` before capture assembly. Integral
mode remains the existing analytical line-integral output and does not pass
through the detector. Per-source conversion outputs remain unchanged and are
not separately sent through detector response; they remain diagnostics.

The current intensity is a simplified compatibility signal derived from
attenuation integration. It is not calibrated photon fluence or detector-count
data. `XRayPhysicsModel` continues to own the existing conversion behavior,
including legacy distance gain and output-mode compatibility. Its
`integral_to_image()` implementation was not moved or rewritten.

`VirtualXRay` and its existing projection geometry remain authoritative for
detector pose and pixel layout. `XRayDetector` is geometry-independent because
identity response requires no geometry; no second mutable position/orientation,
scene node, GUI setting, or serialization field was added. `DetectorImage`
remains the existing scene object for cached output and presentation. Capture
fields, cache behavior, and presentation flow are unchanged.

## Verification

- Focused projection tests: **31 passed**.
- Full plugin suite: **120 passed, 5 skipped, 10 subtests passed**. The five
  skips are existing source-skeleton placeholders.
- Frozen 0.6.2 runner, `--mode after`: all **64 original** and **15
  supplementary** arrays passed (79 total); maximum absolute and relative
  differences were zero. Shapes, dtypes, masks, and diagnostic ordering passed.
  The runner verified the frozen baseline hashes before comparison.
- `git diff --check`: passed.

The frozen numerical arrays, their manifests, and original 0.6.2 post-extraction
benchmark reports were preserved. The 0.6.3 performance observations above
remain recorded in this handoff. The original baseline continues to
cover cone and parallel projection, sampling and Siddon volume integration,
both mesh backends, mixed sources, per-source diagnostics, collimation, depth
restrictions, and both output modes.

## Performance observations

The existing benchmark runner used two warm-ups and five timed repetitions.
Compared with its recorded 0.6.2 timing samples, this run measured medians of
5.206 ms for 5x5 volume sampling (6.694 ms reference), 2.985 ms for 5x5
Siddon (2.786 ms), 46.561 ms for 5x5 mixed volume/mesh (47.451 ms), 25.492 ms
for 64x64 sampling (35.906 ms), 6.555 ms for 32x32 Siddon (6.525 ms), and
1.735 s for 32x32 mixed sampling/mesh (1.701 s). These are noisy environment
measurements, not evidence of a speedup or regression; the short Siddon and
mixed-case differences are small and the component performs no array
allocation. The physics-conversion phase was approximately 32-42 microseconds
in these runs. Python `tracemalloc` is only an allocation proxy and does not
include all NumPy native allocations or process RSS.

## Limitations and future work

This is only an architectural boundary for later detector-response work. It
does not model calibrated exposure, energy spectra, focal spot sampling above
N=1, scatter, detector noise, point-spread/MTF, saturation, or ADC behavior.
Adding such behavior will require an explicit incident-signal contract and
separate verification; none is implied by the identity implementation.

No GUI, scene serialization, device-link, or release metadata changes were
made. Baseline and changelog documentation were updated for accepted closure.
