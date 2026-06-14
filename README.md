# virtRTG

`virtRTG` is a `pyDpVision` plugin for synthetic X-ray projection from 3D scene
data. It is intended primarily for research, prototyping, and technical
experimentation around virtual radiography, especially for volumetric medical
data and mixed volume/mesh scenes.

## Status

The plugin is usable as an experimental RTG workflow inside `pyDpVision`, but it
should not be treated as a clinical, regulatory, or production-grade radiography
simulator.

What is already strong:

- flexible detector and source geometry,
- support for cone-beam and parallel-beam projection,
- hybrid scene handling for volumes and meshes,
- multiple presentation modes for the final image,
- plugin-local numeric tests for key backend components,
- explicit attribution notes for major algorithmic references.

What is still evolving:

- the physics model is simplified and partly heuristic,
- some mesh backends are still exploratory,
- GUI and OpenGL paths are not the main target of automated testing,
- standalone packaging outside `pyDpVision` has not been completed yet.

## Features

### Projection geometry

- detector geometry built from center, normal, and up vectors,
- configurable detector resolution and pixel pitch,
- cone-beam and parallel-beam projection modes,
- optional depth-window clipping,
- reusable quality profiles for draft and higher-quality runs,
- JSON-based geometry presets.

### Source models

- volumetric X-ray source with interpolated sampling,
- optional Siddon-style exact voxel traversal for volume integration,
- mesh X-ray source with analytic ray-triangle intersection,
- projected mesh intersection backend for detector-space experiments,
- per-source scaling and attenuation controls.

### Physics and presentation

- CT-like scalar-to-attenuation mapping,
- Beer-Lambert style intensity conversion,
- optional heuristic energy and distance falloff terms,
- raw, digital radiography, and film-like presentation models,
- robust percentile normalization, contrast, gamma, inversion, and windowing.

### Workflow integration

- `VirtualXRay` scene object integrated into the `pyDpVision` object tree,
- Qt property panel for interactive setup and simulation,
- projected annotation overlays,
- PNG, TIFF, and DICOM export helpers,
- synthetic demos and benchmark helpers.

## How it is used

Within `pyDpVision`, a typical workflow looks like this:

1. Load or create one or more `Volumetric` or `Mesh` objects.
2. Create a `VirtualXRay` object from the plugin menu.
3. Choose a preset or define detector/source geometry manually.
4. Tune sampling, source handling, physics, and presentation parameters.
5. Run the projection.
6. Inspect the result in the workspace or export it to file.

In practice, the central user-facing object is `VirtualXRay`, while the heavy
computation lives in the backend modules under `xray/`.

## Quick start inside pyDpVision

The plugin is loaded by the host application during startup.

Run the application from the project root:

```powershell
python main.py
```

Then, inside the application:

1. Open or import scene data.
2. Go to the plugin menu and create a new RTG object.
3. Select the created `VirtualXRay` object in the workspace.
4. Use the property panel to configure geometry and run the simulation.

## Architecture overview

The plugin is split into a small number of responsibility layers.

### Integration layer

- `pluginMain.py`
  - plugin registration, menu integration, property-panel hookup,
- `virtualXRay.py`
  - scene-tree object coordinating geometry, source discovery, backend
    configuration, caching, and visual integration with the host scene.

### GUI layer

- `gui/propVirtualXRay.py`
  - interactive Qt property panel for configuring and running the RTG workflow.

### Backend layer

- `xray/xrayProjection.py`
  - geometry, clipping, physics, projector logic, scene-level projection API,
- `xray/xraySource.py`
  - volumetric and mesh source backends,
- `xray/xrayPresentation.py`
  - final image presentation models,
- `xray/xrayAnnotationOverlay.py`
  - annotation projection and 2D overlay primitives,
- `xray/xrayHelpers.py`
  - shared transform and math helpers.

### Support files

- `presets/xray_geometry_presets.json`
  - editable geometry presets,
- `docs/THIRD_PARTY_ATTRIBUTION.md`
  - algorithmic attribution notes,
- `benchmark.py`
  - demo data generation and performance helpers.

## Repository layout

```text
plugins/virtRTG/
  pluginMain.py
  virtualXRay.py
  benchmark.py
  gui/
    propVirtualXRay.py
  xray/
    xrayProjection.py
    xraySource.py
    xrayPresentation.py
    xrayAnnotationOverlay.py
    xrayHelpers.py
  presets/
    xray_geometry_presets.json
  docs/
    THIRD_PARTY_ATTRIBUTION.md
  tests/
    README.md
```

## Tests

The plugin has a local `pytest` scaffold in `tests/`, focused on the pure numeric
backend under `xray/`.

Currently covered:

- geometry helpers,
- clipping helpers,
- scalar preprocessing and physics response,
- presentation models,
- additional backend-oriented helper coverage as the suite grows.

Current test philosophy:

- prioritize deterministic numeric behavior,
- avoid coupling basic unit tests to Qt and OpenGL,
- keep backend tests close to the plugin so they can move with it if it becomes
  a standalone repository.

See [tests/README.md](C:/praca/pyDpVision/plugins/virtRTG/tests/README.md:1)
for details on the current scaffold and next planned cases.

## Limitations

This plugin is intentionally pragmatic, not physically complete.

- The attenuation model is simplified and should be treated as approximate.
- Material-response modes are useful for experimentation, not validated imaging physics.
- Presentation models are image-generation layers, not calibrated detector models.
- Some advanced mesh workflows remain sensitive to source topology and scene setup.
- Host-side integration still assumes the `pyDpVision` application lifecycle.

## Documentation

- attribution and algorithm notes:
  [docs/THIRD_PARTY_ATTRIBUTION.md](C:/praca/pyDpVision/plugins/virtRTG/docs/THIRD_PARTY_ATTRIBUTION.md:1)
- test scaffold notes:
  [tests/README.md](C:/praca/pyDpVision/plugins/virtRTG/tests/README.md:1)

## Roadmap

Reasonable next steps for the plugin itself:

- strengthen source-backend tests with synthetic `Volumetric` and `Mesh` fixtures,
- add more lightweight end-to-end backend projection tests,
- reduce host-specific assumptions around plugin loading and global state,
- document a backend-only usage path outside the GUI,
- separate experimental backends more clearly from the default workflow,
- improve standalone readiness if the plugin is moved into its own repository.
