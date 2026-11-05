# virtRTG 0.6.1 Implementation Handoff

Status: accepted implementation stage; release documentation and verification
are complete.

## Scope

This unaccepted implementation adds lightweight tube-emission and collimator
configuration snapshots. Projection algorithms, scene serialization, the
browser simulator, protocol v1, detector presentation, and release metadata
remain unchanged.

## Files changed

- `xray/xrayComponents.py`: `FocalSpotSamples`, `XRayTube`, and
  `XRayCollimator` immutable configuration objects.
- `xray/xrayProjection.py`: typed optional component references in
  `XRayProjectionConfig`.
- `virtualXRay.py`: fresh snapshots from authoritative VirtualXRay settings;
  one collimator snapshot per projection config is adapted to the existing
  projection geometry and reused on the config.
- `tests/unit/test_xray_components.py`: focal sample validation, angular-field
  semantics, snapshot refresh, collimator-only roll, tube-orientation
  limitation, parallel compatibility, rotation validation, and supplied
  snapshot consistency.

## Geometry, models, and snapshots

`VirtualXRay` remains the sole owner of editable and serialized settings.
`XRayProjectionGeometry` remains the calculation contract consumed by the
projector. For this version, `XRayCollimator` contributes only the existing
`RectangularBeamGeometry` and enabled flag to that contract. The projector's
optimized vectorized detector exposure mask still reads those geometry fields
and produces the same `bool[H,W]` result. Neither component model is called per
ray or per depth sample.

`XRayTube` contains a one-sample `FocalSpotSamples` snapshot of
`source_position_ref` in millimeters and the explicit `virtual_xray_ref_mm`
frame, plus optional scalar energy context copied from the legacy physics
setting. This snapshot is informational in 0.6.1: it does not generate or
change projection rays, and its energy field does not alter current intensity
semantics. Parallel mode continues to set `tube_model=None`.

There is no tube orientation setting independent of beam/collimation state in
the current persisted VirtualXRay model. `directional_beam_geometry` is the
effective field orientation used for collimation and projection; it changes
with H/V/Roll controls. The session collimation reference is an editing
reference that can be reset by alignment or device updates, not a stable,
independent tube pose. Therefore `XRayTube.tube_to_reference_rotation` is
explicitly `None`; no tube pose is inferred from steering, roll, or alignment.
If a rotation is supplied to the model in a future use, it must be a finite,
orthonormal, right-handed 3x3 rotation.

`build_projection_config()` creates one tube snapshot and one collimator
snapshot, then passes that exact collimator to `_build_geometry()` and stores
the same object in `config.collimator_model`. `_build_geometry()` now uses its
argument directly and does not reconstruct from mutable scene fields. A
separate call to public `build_geometry()` creates a fresh collimator snapshot
and passes it to `_build_geometry()`. This keeps direct field edits reflected
without persistent aliases or a second mutable authority. A focused regression
supplies a snapshot that disagrees with current mutable settings and fails if
the adapter attempts to rebuild it. `XRayProjectionConfig` uses `TYPE_CHECKING`
imports and postponed annotations for typed optional model fields, avoiding a
runtime import cycle.

## Collimator origin semantics

The existing `RectangularBeamGeometry` specifies an ideal angular rectangular
field: axis, transverse axes, field dimensions, and reference distance. It
contains no positioned aperture plane or blade geometry. Accordingly, the
batch helper is now explicitly `transmission_for_directions()`/
`accept_directions()` and evaluates rays as directions from a shared focal
point. It does not accept arbitrary origins and does not claim to calculate
physically correct acceptance for different focal-spot positions. That
requires a positioned aperture model and is deferred; no mechanical blades or
finite-aperture physics are implemented.

The direction helper is not used by the projector. The existing optimized
rectangular exposure-mask calculation remains the point-source implementation
and is covered by existing exact expected-mask regressions.

## Verification

The final focused set covering component contracts, geometry, projection and
collimation, physics/output modes, scene format, device-link, and presentation
passed: **102 passed, 10 subtests passed**. The full plugin test suite passed:
**117 passed, 5 skipped, 10 subtests passed**. The suite includes point-source
detector geometry, rectangular mask cases, H/V/Roll and alignment,
device-link mapping/rollback, scene save/load, parallel compatibility,
integral/intensity modes, depth-window clipping, research/source diagnostics,
cached DetectorImage presentation, and presentation-model coverage. The new
focused tests ensure collimator roll changes the effective field without
creating a tube rotation, reject invalid tube rotation matrices, and preserve
identity of the supplied collimator snapshot during geometry adaptation.

`py_compile` and nested repository `git diff --check` passed. Ruff is not
installed in the project virtual environment, so lint was not run. The five
skips are reported by the existing plugin suite; there were no test failures.

## Performance sample

No pre-refactoring timing baseline was captured. The measurements below are a
repeatable post-change sample only; they do not imply a before/after speedup or
performance equivalence.

Environment: repository `.venv`, 512x512 detector, cone source, enabled
20x20 cm field at 1000 mm, 300 measured iterations for configuration builders
and 30 for vector setup/mask. Each figure is the median wall time:

| Operation | Median |
| --- | ---: |
| `build_geometry()` | 21.8 us |
| `build_projection_config()` | 60.4 us |
| Vector ray setup plus rectangular mask | 22.85 ms |

The vector benchmark follows the current NumPy detector-grid, ray-normalization,
and angular mask operations; the projector kernel itself is unchanged. The
mask output is a 512x512 bool array (262,144 bytes). A practical retained-array
proxy for that setup is about 69 bytes per pixel (about 18.1 MB at 512x512),
counting the grids, centers, rays, normalized directions, mask intermediates,
and output mask; this is not a peak-RSS measurement and excludes transient
NumPy expression temporaries. The component snapshots are constructed once per
configuration build and add no H*W or per-depth Python objects.

## Intentional simplifications and next-stage limits

- Only N=1 point-source projection exists; no extended focal-spot projection,
  spectra, scatter, physical blades, penumbra, detector response, or transport
  extraction was added.
- The current collimator is an ideal angular field. A future physically
  positioned aperture must define its position/frame and calculate acceptance
  for independent origins before claiming focal-sample-specific transmission.
- Tube orientation remains unavailable until the application has an
  authoritative tube pose independent of effective collimator orientation.
- 0.6.2 must keep the component snapshots separate from transport execution
  until a batch-level request/result contract can preserve N=1 ray/mask and
  legacy output behavior without per-ray dispatch.
- The accepted baseline and changelog now record the 0.6.1 implementation.
- No scene schema, GUI, or simulator protocol changes were made. This closes
  only the accepted 0.6.1 implementation stage; it does not begin 0.6.2.
