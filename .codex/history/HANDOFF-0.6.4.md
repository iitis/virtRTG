# virtRTG 0.6.4 Tube and Collimator Runtime Integration Handoff

Status: **accepted implementation**. This is a responsibility extraction for
the existing N=1 point source and ideal rectangular field; it does not add
projection physics.

## Runtime call path

Previously, `VirtualXRay.build_projection_config()` made fresh `XRayTube` and
`XRayCollimator` snapshots, but `XRayScene.project_capture()` forwarded only
`config.effective_geometry()`, physics, and the reference transform. The
projector therefore read `geometry.source_position_ref` to set cone-ray origins
and independently calculated the rectangular field mask from the geometry's
beam fields.

Now, `XRayScene.project_capture()` still creates the projector and calls its
capture path. `XRayProjector.project_capture()` forwards both snapshots along
with the effective geometry. The projector resolves a tube and collimator
before preparing rays: a supplied component is used directly; a missing one
is adapted from legacy geometry. It builds detector pixels and ray directions,
uses `XRayTube.focal_position_mm()` for the N=1 cone origin, and delegates the
full bool exposure mask to `XRayCollimator.exposure_mask()`. The existing AABB
intersection, exposure intersection, depth clipping, transport call, output
conversion, and capture assembly follow unchanged.

## Responsibility boundaries

- `VirtualXRay` remains the authority for editable and serialized parameters.
  It builds fresh per-run component snapshots. No component settings are
  serialized separately.
- `XRayTube` owns the current focal-source representation and supplies the
  effective focal position in `virtual_xray_ref_mm`. The projector accepts only
  one sample in that frame, matching the current N=1 execution. Tube energy
  and orientation remain outside ray generation.
- `XRayCollimator` owns the ideal rectangular field's production mask. Its
  enabled path uses the former vectorized float32 operations, forward-positive
  test, and inclusive `<=` boundaries. Its disabled path returns a bool array
  from the requested detector shape. It does not construct detector pixels or
  handle AABB/depth clipping or transport.
- `XRayProjector` retains detector grid and ray-direction construction,
  transforms, scene intersection, depth clipping, transport coordination,
  projection conversion, detector response, and capture/stat assembly.
- `XRayTransportModel` and `XRayDetector` are unchanged.

## Compatibility and precedence

`XRayProjectionConfig.effective_geometry()` still applies the quality profile.
It is consumed by the projector, annotation projection, and VirtualXRay
projection helpers. Detector pose/layout, projection mode, and depth settings
continue to come from effective geometry. If a tube or collimator component is
supplied, its focal position or beam geometry/enabled flag respectively takes
precedence over the corresponding legacy geometry fields. Geometry continues
to select cone versus parallel mode. Thus a tube is used only for cone rays,
and enabled collimation on a parallel projection still raises the established
error.

Legacy geometry-only calls receive temporary equivalent components. Existing
positional arguments to `XRayProjector.project()` remain valid; optional
`tube_model` and `collimator_model` arguments were appended. A component with
an unsupported focal coordinate frame or sample count fails clearly rather
than being silently ignored. The collimator's existing
`transmission_for_directions()` helper remains available; it uses float64
direction arithmetic, while the production mask retains float32 detector
coordinate arithmetic. Near a field edge, roundoff can therefore differ. The
production mask remains authoritative, and no new discrepancy was observed in
the frozen regression cases.

## Files and methods changed

- `xray/xrayComponents.py`: added `XRayTube.focal_position_mm()` and
  vectorized `XRayCollimator.exposure_mask()`.
- `xray/xrayProjection.py`: added geometry-to-component compatibility
  resolution; forwarded config components; sourced cone focal origin from the
  tube; delegated enabled and disabled masks to the collimator.
- `tests/unit/test_projection_additional.py`: added runtime delegation,
  component precedence, geometry-only compatibility, and N=1 source tests.
- `.codex/history/HANDOFF-0.6.4.md`: this handoff.

## Numerical and test verification

The 0.6.2 runner verified the frozen hashes and compared all 64 original arrays
and all 15 supplemental arrays. Every array passed with exact equality,
including masks, shapes and dtypes. Source ordering and all non-timing case
diagnostics also matched. The verified original SHA-256 values remain:

- `baseline.json`: `E4044FC47A8B18D8D8275B198D4BE560B168260BA1912419CA61E2B6B8F1D062`
- `baseline_arrays.npz`: `C86967088175DB26D4AADE1AA65FC7B070E5D91D07A64997C44D0512E3B30C4A`

The full plugin suite passed: **123 passed, 5 skipped, 10 subtests passed**.
The focused projection and component set passed: **40 passed**. Python
compilation and nested repository `git diff --check` passed.

## Performance observations

The 0.6.2 frozen runner uses two warm-ups and five measured iterations. Its
historical `after.json` and `comparison.json` files were restored after the
0.6.4 comparison. The 0.6.4 run measured the following representative medians:

| Case | Accepted 0.6.2 median | 0.6.4 median | 0.6.4 observed range |
| --- | ---: | ---: | ---: |
| 5x5 volume sampling | 6.694 ms | 4.818 ms | 4.730-4.951 ms |
| 5x5 volume Siddon | 2.786 ms | 2.743 ms | 2.721-2.983 ms |
| 5x5 mixed volume/mesh | 47.451 ms | 47.514 ms | 45.075-48.291 ms |

These short runs are environment-sensitive and do not establish an optimization
or meaningful regression. The disabled mask path does not allocate detector
coordinate grids; the enabled path retains the existing vectorized grid/mask
work.

## Limitations

Only the current single point source and ideal angular rectangular field are
integrated. Multiple focal samples, energy-dependent emission, positioned
apertures/blades, parallel collimation, scatter, and other physics extensions
remain unsupported. The existing helper and production mask may differ at
float-roundoff boundaries as described above. Projection cache freshness and
configuration provenance remain outside this stage. No GUI, scene
serialization, transport, detector, or projection physics changes were made.
This handoff closes the accepted 0.6.4 stage and does not start another stage.
