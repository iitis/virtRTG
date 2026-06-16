# virtRTG Current State

This file summarizes the current implementation state of the virtual X-ray
plugin. It is context, not a changelog. Prefer this file over older historical
notes when there is a conflict.

## Location and Responsibility

The virtual X-ray plugin lives in:

```text
plugins/virtRTG/
```

The core `dpVision/` package should provide scene objects, transformations,
volumetric data, mesh data, GUI integration points, and reusable infrastructure.
Projection-specific logic should stay in `plugins/virtRTG/`.

## Main Components

- `pluginMain.py`: plugin integration and property-panel registration.
- `virtualXRay.py`: scene object representing source, detector, projection
  settings, cache, and scene integration.
- `benchmark.py`: benchmark helpers.
- `gui/propVirtualXRay.py`: property panel for the `VirtualXRay` object.
- `xray/xrayProjection.py`: backend projection architecture.
- `xray/xrayAnnotationOverlay.py`: projected annotation overlay helpers.
- `presets/xray_geometry_presets.json`: geometry presets.
- `docs/`: technical and article-style documentation.
- `tests/`: plugin-local pytest suite.

## Backend Architecture

The projection backend is separated from `Volumetric`.

Important backend classes include:

- `XRayProjectionGeometry`
- `XRayPhysicsModel`
- `XRayScalarPreprocessor`
- `XRaySampleSource`
- `VolumetricXRaySource`
- `MeshXRaySource`
- `XRayProjector`
- `XRayProjectionQualityProfile`
- `XRayProjectionConfig`
- `XRayScene`
- `XRayPresentationModel`
- `RawPresentationModel`
- `FilmLikePresentationModel`
- `DigitalRadiographyPresentationModel`

## VirtualXRay Object

`VirtualXRay` is a normal scene `Object`. It visualizes a source and detector
gizmo, gathers child `Volumetric` and `Mesh` objects, builds an `XRayScene`, and
generates projections from the current scene hierarchy.

`VirtualXRay` works in its own local coordinate system. Child volumes and meshes
are transformed into the `VirtualXRay` coordinate system using relative
transformations, not by assuming absolute global coordinates.

`VirtualXRay` has an explicit:

```text
projection_mode = "cone" | "parallel"
```

Do not infer projection mode from `source_position_ref is None`.

`build_geometry()` should pass only the active geometry parameter:

- `source_position_ref` for `cone`
- `ray_direction_ref` for `parallel`

## Projection and Presentation Cache

`VirtualXRay` stores:

- `last_raw_projection`: raw float32 projection cache.
- `last_projection_image`: existing workspace `Image` updated in place.
- `last_projected_annotations`: projected annotation metadata.

`project_and_cache(...)` performs the expensive projection and updates the raw
cache.

`apply_presentation()` applies the current presentation model to the cached raw
projection without repeating ray marching.

The GUI should use this separation so changing contrast, gamma, windowing, or
overlays does not require a full projection recomputation.

## GUI Panel

The `VirtualXRay` property panel is organized into tabs:

- `Geometry`
- `Physics`
- `Presentation`
- `Run`

`Geometry` and `Physics` may contain `Show advanced` sections for low-level
backend parameters.

The `Run` tab contains:

- `Refresh`
- `Run Simulation`
- `Update display`

`Run Simulation` computes a new projection and updates the cache.

`Update display` applies presentation to the cached projection only.

When updating projection images, preserve the existing pattern that updates the
existing `Image` object in place and lets `Image` allocate OpenGL textures lazily
inside controlled render paths.

## Sources

`VirtualXRay` gathers child `Volumetric` and `Mesh` objects.

Volumetric sources support interpolation, fill values, scalar scaling, scalar
bias, attenuation multipliers, and preprocessing.

Mesh sources support simplified X-ray modes:

- `solid`: closed mesh interpreted as filled material.
- `shell`: mesh interpreted as a thin shell with configurable thickness.

Per-object X-ray settings may exist on `Mesh` and `Volumetric` objects with
names such as `xray_enabled`, `xray_scalar_scale`, `xray_scalar_bias`, and
`xray_attenuation_multiplier`.

Property panels for `Mesh` and `Volumetric` may expose working `XRay Source`
sections for per-object settings.

## Mesh Projection Backends

`MeshXRaySource` has an analytic BVH-based backend and an experimental
`projected_intersection_list` backend.

The BVH backend builds and caches a local AABB tree and traverses it per ray.

The projected-intersection backend projects triangles to the detector and builds
compressed per-pixel intersection stacks. It is experimental and includes
diagnostic export options.

In `solid` mode, do not convert an isolated odd intersection into a shell-like
fallback. Such pixels should be treated as uncertain instead of producing local
bright artifacts.

## Physics and Preprocessing

`XRayPhysicsModel` includes parameters such as:

- `mu_air`
- `mu_water`
- `hounsfield_air`
- `attenuation_scale`
- `output_mode`
- `intensity_floor`
- `material_response_mode`
- `bone_threshold_hu`
- `bone_threshold_softness`
- `material_window_center`
- `material_window_width`
- `material_window_mode`
- `material_window_softness`

Material response modes include:

- `linear`
- `piecewise_bone`
- `piecewise_soft_tissue`
- `bone_threshold`

Material windows operate on the `scalar -> mu` stage, before projection. Window
modes include `hard`, `linear`, and `sigmoid`.

`XRayScalarPreprocessor` can perform percentile-based rescaling before
`scalar -> mu` mapping.

## Geometry Features

`XRayProjectionGeometry` supports optional depth windows:

- `off`
- `ray`
- `planar_auto`
- `planar_custom`

Depth windows restrict the integration range along rays or inside a planar slab.

The gizmo should visualize active depth windows when possible.

Geometry presets are loaded from:

```text
plugins/virtRTG/presets/xray_geometry_presets.json
```

Hard-coded presets in `virtualXRay.py` should be treated only as fallback.

## Annotation Overlays

Projected annotations are not mixed into the raw projection buffer. They are
projected to detector-space metadata and composited at presentation/export time.

`AnnotationPoint` currently produces colored crosses and optional labels.

Overlay logic is being separated into:

```text
plugins/virtRTG/xray/xrayAnnotationOverlay.py
```

Keep this separation when extending support for more annotation types.

## Documentation and Attribution

`plugins/virtRTG/README.md` is maintained as a public-style README for a possible
standalone plugin repository.

`plugins/virtRTG/docs/THIRD_PARTY_ATTRIBUTION.md` records algorithmic references.
Known algorithms and conventions to cite include Beer-Lambert attenuation,
Siddon-style traversal, and Moller-Trumbore ray-triangle intersection.

`docs/xray_projection.tex` and `docs/main.tex` may contain article-style
technical descriptions of the current backend and should be kept consistent with
implementation changes.

## Testing and Debugging

`plugins/virtRTG/tests/` contains plugin-local pytest tests for backend geometry,
physics, and presentation.

`dp_testy.py` contains development helpers for synthetic X-ray scenes, demo
DICOMs, headless projection runs, mesh topology reports, and mesh cleanup.

Treat `dp_testy.py` as a development/testing helper rather than production API.

