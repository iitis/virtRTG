# virtRTG 0.7.0 Browser Device Link Projection Integration Handoff

Status: **accepted for closure by user; end-to-end verification remains partial**.

## Scope

This stage reconnects the protocol-v1 browser Device Link to local projection
execution on the current `xray_refactor` branch. It also applies the simulator
phantom pose to anatomy under a selected `VirtualXRay` and disables the
extended source-to-detector helper line in the scene gizmo. The version is
0.7.0 because this integration milestone is distinct from the 0.6.x projection
component architecture series.

## Runtime behavior

- Accepted `device.pose` and `state.update` messages continue to map source,
  detector, and beam geometry to the uniquely selected `VirtualXRay`.
- The mapped phantom pose is represented by a dedicated `Transform` child of
  `VirtualXRay`. Mesh and volumetric anatomy roots are placed under this frame,
  retaining their existing local transforms. Subsequent updates change the
  frame rather than accumulating transforms.
- Accepted `acquisition.request` messages are acknowledged by the server and
  forwarded from its WebSocket thread to the Qt GUI thread. The selected
  `VirtualXRay` runs `project_and_cache()`, then the plugin creates or refreshes
  its associated workspace `DetectorImage`.
- Projection mode may be specified as `cone` or `parallel` in the request's
  optional `projection` parameter. The plugin enables physical collimation for
  the acquisition and restores the previous projection mode and collimation
  setting afterward.
- The result stays in the plugin. No image data is sent to the browser, and no
  protocol schema or version changed. `kvp`, `ma`, `exposure_ms`, and `mas` are
  validated but are not currently applied to projection physics. Table pose
  is not applied to the scene.
- The gizmo's extended source-to-detector helper line is disabled by commenting
  out its drawing commands; the remaining source, frustum, detector, and axes
  are unchanged.

## Files changed

- `device_link/server.py`: route accepted acquisition requests to a callback
  separately from geometry updates.
- `device_link/device_adapter.py`: describe the local-only acquisition result
  in the acknowledgement detail.
- `device_link/integration.py`: apply the phantom transform frame and execute
  accepted acquisitions on the GUI thread.
- `pluginMain.py`: connect the acquisition callback and create or update the
  resulting `DetectorImage` in the workspace.
- `virtualXRay.py`: comment out the extended gizmo helper-line drawing block.
- `README.md`, `CHANGELOG.md`, `.codex/baseline.md`, and this handoff: document
  the 0.7.0 behavior, limitations, and verification state.

## Verification and limitations

- `git diff --check`: passed after implementation and documentation changes.
- No automated tests, compilation checks, or manual end-to-end browser-to-plugin
  run by Codex were performed on `xray_refactor` for this stage.
- The user manually checked the 0.7.0 behavior in the GUI, including the
  helper-line display change, phantom positioning, and acquisition projection.
- The prior `device_link` patch had focused tests, but those results do not
  establish this branch's runtime behavior and are not counted as 0.7.0
  verification.
- There is no image return path to the browser. Generator settings are not
  connected to projection physics, table pose is ignored, and the link still
  requires exactly one selected `VirtualXRay`.

## Review state

The implementation and documentation are present on local branch
`xray_refactor`. The configured remote has no matching review branch, and this
handoff does not claim a push, merge, or remote publication.
