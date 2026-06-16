# virtRTG Agent Instructions

This directory contains the virtual X-ray / projection simulation plugin for
pyDpVision.

Read the root `AGENTS.md` first, then read the local
`.context/current-state.md` before making non-trivial changes in this plugin.

## Scope

Keep projection-specific logic inside `plugins/virtRTG/`.

The core `dpVision/` package should provide shared scene, GUI, object,
transformation, parser, and rendering infrastructure. Do not move plugin-specific
physics, projection, presentation, or benchmark logic into the core unless the
task explicitly asks for such refactoring.

Keep virtRTG documentation local to this plugin or to the standalone virtRTG
repository if the plugin is split out. The main pyDpVision documentation should
only describe the plugin mechanism and generic integration points.

## Current Architecture

The main scene object is `VirtualXRay` in `virtualXRay.py`.

The projection backend lives primarily in `xray/xrayProjection.py`.

The GUI property panel lives in `gui/propVirtualXRay.py`.

Projected annotation overlay logic should stay separated in
`xray/xrayAnnotationOverlay.py` or equivalent overlay-specific helpers.

Geometry presets should be loaded from:

```text
presets/xray_geometry_presets.json
```

Hard-coded geometry presets should be fallback only.

## VirtualXRay Rules

`VirtualXRay` works in its own local coordinate system. Child source objects are
transformed relative to the `VirtualXRay` object.

Use the explicit field:

```text
projection_mode = "cone" | "parallel"
```

Do not infer the projection mode from `source_position_ref is None`.

`build_geometry()` should pass only the active parameter:

- `source_position_ref` for cone-beam projection.
- `ray_direction_ref` for parallel projection.

## Projection Cache

Preserve the separation between expensive projection and presentation:

- `project_and_cache(...)` computes and stores the raw projection.
- `apply_presentation()` uses the cached raw projection.
- `Update display` should not repeat ray marching.

Projection image updates should update the existing `Image` object in place when
possible. Keep OpenGL texture allocation lazy and inside controlled render paths.

## Physics and Presentation

Material windows operate before projection at the `scalar -> mu` stage.

Presentation settings operate after projection and should not change the raw
projection cache.

Do not mix projected annotations into the raw projection buffer. Composite them
at presentation or export time.

## Mesh Sources

Mesh projection currently supports simplified `solid` and `shell` modes.

The BVH backend is the main acceleration path. The `projected_intersection_list`
backend is experimental and may include diagnostic exports.

In `solid` mode, isolated odd intersections should be treated as uncertain, not
converted into shell-like fallback contributions.

## Tests

When modifying projection geometry, physics, presentation, source handling, or
annotation overlays, add or update focused tests in `plugins/virtRTG/tests/` when
practical.

For risky rendering or GUI changes, prefer a small headless backend test plus a
manual GUI verification note.

## Documentation

Keep `README.md`, `docs/THIRD_PARTY_ATTRIBUTION.md`, `docs/xray_projection.tex`,
and article-oriented documentation consistent with meaningful backend changes.

Mention algorithmic references where relevant, especially Beer-Lambert
attenuation, Siddon-style traversal, and Moller-Trumbore ray-triangle
intersection.
