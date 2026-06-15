# virtRTG test scaffold

This directory contains a plugin-local `pytest` scaffold for the `xray/`
backend. The goal is to keep backend validation close to the plugin so the same
tests can move with `virtRTG` if it is later extracted into a standalone
repository. The focus is on deterministic validation of the transformation-driven
projection backend rather than GUI integration testing.

## Current contents

- `conftest.py`
  - shared import-path bootstrap and tiny deterministic fixtures,
- `unit/test_geometry.py`
  - lightweight geometry and clipping tests,
- `unit/test_physics_model.py`
  - scalar preprocessing and detector-response tests,
- `unit/test_presentation.py`
  - presentation-layer normalization tests,
- `unit/test_overlay_and_helpers.py`
  - detector overlay projection, annotation metadata and transform helpers,
- `unit/test_projection_additional.py`
  - quality profiles, physics-model branches and lightweight scene/projector flow,
- `unit/test_source_helpers.py`
  - projection export normalization plus triangle/BVH helper coverage,
- `unit/test_source_skeleton.py`
  - planned source-backend tests kept as an explicit scaffold.

## Recommended next tests

1. Add a tiny synthetic `Volumetric` stub or fixture and cover:
   - `sample_scalar_world()`,
   - `sample_attenuation_world()`,
   - `_siddon_integral_vectorized()`.
2. Add one tiny synthetic `Mesh` fixture and cover:
   - `_ray_triangle_hit_distances()`,
   - `_rays_single_triangle_hit_distances()`,
   - BVH traversal parity against the direct variant.
3. Add one end-to-end backend test that builds:
   - `XRayProjectionGeometry`,
   - `XRayPhysicsModel`,
   - one synthetic source,
   - `XRayScene.project()`.

## Notes

- The current scaffold intentionally avoids Qt GUI and OpenGL rendering and
focuses on deterministic backend behaviour.
- The skipped skeleton file marks the highest-value next steps without making
  the suite fail before source fixtures are ready.
