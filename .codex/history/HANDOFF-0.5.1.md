# virtRTG 0.5.1 Handoff: Physical Beam Collimation

Status: **Accepted**.

## Active data path

The active projection path is `VirtualXRay.project_and_cache()` to
`VirtualXRay.project_capture()`, `XRayScene.project_capture()`, and
`XRayProjector.project()`. A capture carries the acquisition exposure mask with
its combined and per-source projections. `project_and_cache()` stores that mask
as `last_exposure_mask`; `DetectorImage.sync_from_virtual_xray()` copies it for
display/export. Device-state updates still only update geometry and do not run a
projection.

## Activation and mask computation

`physical_collimation_enabled` explicitly opts into collimation and defaults to
false, including on legacy objects. A `directional_beam_geometry` value alone
does not enable it. The VirtualXRay properties panel exposes one checkbox.
When enabled, cone-beam projection requires directional beam geometry; parallel
projection raises a clear `ValueError`. No beam direction is inferred from the
detector.

Detector sample locations and the accepted beam frame are evaluated in the
VirtualXRay reference frame. The projector builds a vectorized forward-ray and
rectangular half-width/half-height test from focal spot to each detector pixel.
The reference transform used by ray integration is not applied to this test a
second time. The resulting exposure mask is intersected with scene AABB and
depth-window hits before direct, Joseph, Siddon, or mesh attenuation work.

## Signal and presentation semantics

Exposed rays that miss anatomy retain zero attenuation and therefore receive
the configured unattenuated primary response. Unexposed pixels are explicitly
zero after intensity conversion, including after application of the intensity
floor. Integral-mode values outside the field are set to zero but remain
unmeasured; `exposure_mask` carries validity separately. Per-source outputs use
the same acquisition exposure mask, so a source with no anatomical contribution
is not interpreted as unexposed.

`DetectorImage` retains the mask and applies it after presentation and transfer
curve mapping. Its active NPZ package stores the mask as a separate array and
older packages without a mask remain readable. `.npy`, text, CSV, and TSV arrays
cannot represent mask metadata; exporting those formats loses the distinction
between unexposed pixels and zero-valued pixels. The older simple VirtualXRay
array NPZ helper is not the active package path; the current package route is
through `DetectorImage`.

## Compatibility and cache

Unrestricted capture behavior remains unchanged: its mask is all true and the
existing physics conversion is unchanged. Acquisition masks are copied into the
projection cache and exported package, so later device geometry changes do not
recompute an earlier mask. Existing package reads without a mask initialize no
presentation mask. Collimation mode is persisted in the VirtualXRay scene
geometry payload. The complete optional `directional_beam_geometry` mapping is serialized with its axes, field dimensions, and reference distance. Loading reconstructs it through `RectangularBeamGeometry`, so constructor validation applies. Legacy scenes without that mapping retain `None` beam geometry and default collimation off.

## Files changed

- `xray/xrayProjection.py`
- `virtualXRay.py`
- `detectorImage.py`
- `sceneFormat.py`
- `gui/propVirtualXRay.py`
- `tests/unit/test_projection_additional.py`
- `.codex/history/HANDOFF-0.5.1.md`

## Verification

Focused regressions cover GUI mode refresh, scene round-trip with collimation enabled, legacy scene loading, analytic rectangular exposure, reversed beam direction, explicit enablement, skipped unexposed intersections, presentation masking, fixed source/detector beam rotation, and NPZ mask validation for legacy, valid, missing, wrong-shape, and invalid-value packages.

Complete virtRTG unit suite: **100 passed, 5 skipped, 10 subtests passed**. Python compilation passed. Ruff ran on the changed Python modules and reported 40 existing findings; comparing against the staged pre-correction files found no newly introduced Ruff diagnostics. `git diff --check` passed for source, tests, and handoff; the generated unified diff was excluded because its embedded patch lines naturally contain source indentation whitespace. No dependency was installed for this correction round.

Full manual browser-to-plugin GUI validation remains pending. The automated Qt integration tests do not replace a real browser session and interactive workspace selection.

## Legacy API candidates

No historical method is asserted to be unused in this change. The standalone
`VirtualXRay._save_projection_array()` and `_load_projection_array()` remain
unchanged; the active rich NPZ package workflow is implemented by
`DetectorImage`.

## Limitations

The model is ideal point-source primary radiation with hard rectangular edges.
It does not model scatter, dark current, detector noise, focal-spot penumbra,
heel effect, or spectral emission. Parallel-beam collimation is unsupported.
