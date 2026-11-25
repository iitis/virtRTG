# virtRTG Agent Instructions

This directory contains the virtual X-ray / projection simulation plugin for
pyDpVision.

Read `../../AGENTS.md` for shared pyDpVision engineering and repository-isolation
rules, then read `.context/current-state.md` before making non-trivial changes.
This file owns the virtRTG repository's branch, remote, artifact, and publication
settings; do not inherit those settings from the host repository.

## Scope

Keep projection-specific logic inside `plugins/virtRTG/`.

The core `dpVision/` package should provide shared scene, GUI, object,
transformation, parser, and rendering infrastructure. Do not move plugin-specific
physics, projection, presentation, or benchmark logic into the core unless the
task explicitly asks for such refactoring.

Keep virtRTG documentation local to this plugin or to the standalone virtRTG
repository if the plugin is split out. The main pyDpVision documentation should
only describe the plugin mechanism and generic integration points.


## Project workflow configuration

Use the `project-stage-workflow` skill for versioned implementation, review corrections and accepted-stage closure. The shared skill is located at `../../.agents/skills/project-stage-workflow/SKILL.md`, relative to this plugin's repository root. If automatic discovery is unavailable, read that file directly. If it is unavailable, follow the local rules below and report its absence.

The following settings apply only to this nested plugin repository, not the host or other plugins:

- GitHub repository: `iitis/virtRTG`
- Git remote: `origin` (verify it targets the repository above)
- Review branch: `codex`
- Accepted branch: `dpwork01`
- Accepted PR merge method: squash
- Commit convention from the next stage: `<version> - <short lowercase description>` for both review commits and accepted squash-merge commits. Do not rewrite historical commit messages.
- Automatic intended review commit and normal review push: authorized for versioned implementation and correction tasks unless explicitly disabled by the user
- Accepted baseline: `.codex/baseline.md`
- Changelog: `CHANGELOG.md`
- Active HANDOFF: `.codex/tasks/HANDOFF-<version>.md` (create `.codex/tasks/` when starting a stage if it does not exist)
- HANDOFF template: `.codex/templates/HANDOFF-template.md` when present; otherwise use the latest archived HANDOFF as a structural reference and replace its stage-specific content
- Archived HANDOFF directory: `.codex/history/`
- Complete experiment results: `experiment-results/` (excluded from Git tracking via .gitignore)
- Temporary archive directory: `D:/current/.TEMP/`
- Artifact delivery destination: `Z:/Mój dysk/CodexResults/virtRTG/` (verify availability before each delivery)

Resolve repository-relative paths from this plugin's root. Verify the actual repository, remote and branch before any Git write. Do not use host Git settings for plugin operations.

The artifact destination is stored as UTF-8 in this file. In Windows PowerShell, use `Get-Content -Encoding UTF8` when reading this file so `ó` is not decoded through the active console code page. To avoid passing that character through a shell or console encoding when accessing the destination, construct the path with its Unicode code point and use literal-path parameters:

```powershell
$destination = Join-Path 'Z:\' ('M' + [char]0x00F3 + 'j dysk\CodexResults\virtRTG')
if (-not (Test-Path -LiteralPath $destination)) {
    throw "Artifact destination is unavailable: $destination"
}
```

Check the destination once before delivery. If it is unavailable, stop and report it; do not retry with mojibake or alternate spellings. Copy with `Copy-Item -LiteralPath` and verify destination byte count and SHA-256 against the source in the same PowerShell invocation. Garbled console rendering alone does not prove that a copy failed; inspect the filesystem result before considering any retry.

## Stage workflow and authorization

- Read the accepted baseline, active task and relevant code before changes. Implement only the active bounded task; report scope conflicts.
- Work on the review branch; the accepted branch contains accepted versions only. Preserve host and unrelated changes.
- Run focused and full plugin tests, compilation/import checks, unused-import checks and root/nested diff checks as applicable. For the full plugin suite, use the host's configured Python environment. From the host directory the current command is `.venv/Scripts/python.exe -m pytest -q plugins/virtRTG/tests`. Distinguish real GUI/OpenGL checks from headless tests.
- Maintain a self-contained tracked HANDOFF, including results, limitations, unresolved issues, manual inspection instructions and artifact locations.
- Keep complete experiment outputs in `experiment-results/`, ignored by this repository's `.gitignore`. Generate review archives only when required or justified by new materials. Create them in the configured temporary directory outside Git, use revision-specific names, validate integrity, and record byte count and SHA-256. Copy validated archives to the configured delivery destination when available and verify copied byte count and SHA-256. Report failed delivery separately from successful local creation; a local copy does not confirm remote synchronization. Preserve original results and keep generated archives and large outputs out of Git.
- After verification, commit and push intended review changes according to the authorization above. Report verified remote SHA and worktree status. Stop for review. A review commit never constitutes acceptance.
- Only after explicit user acceptance of the exact reviewed stage, complete the entire accepted-stage closure automatically. This repository explicitly authorizes the agent to create the PR, squash-merge it and perform the narrow post-squash synchronization below without further user instructions. Verify that the accepted branch still matches the stage's accepted starting point, the reviewed remote head has not changed unexpectedly, and the worktree/index contain no unrelated changes. Finalize and move the tracked HANDOFF to history, update the baseline and changelog, verify closure changes, then commit and push the closure on `codex` using the configured version message. Create a PR from `codex` to `dpwork01`, or reuse the existing PR for this exact stage. Verify the PR base, head, exact remote closure SHA, intended diff, mergeability and required checks/reviews. Squash-merge only that verified PR, using the exact configured version commit message and an expected-head SHA guard; if the head cannot be safely guarded, stop and report the blocker. Never bypass branch protection, required checks or reviews, force-merge, or directly commit to the accepted branch. Confirm the PR is merged and verify the actual accepted commit and tree before synchronizing. If any prerequisite fails or authorization is ambiguous, stop and report the exact blocker. One accepted-version commit per stage is required. Explicit acceptance of a stage is sufficient authorization for this entire closure; a review-ready report, test success or an implementation request alone is not acceptance.
- After a confirmed squash merge, automatically perform post-squash synchronization of the review branch as part of accepted-stage closure, but only after verifying the PR is merged, the accepted and review trees are identical, no extra commits or local changes would be lost, and the remote review head has not changed. Fast-forward the local accepted branch to its remote counterpart. Synchronize the local review branch to the accepted commit and push using an explicit `--force-with-lease` against the independently verified prior remote review SHA. Never force-push the accepted branch. This narrow permission does not authorize other history rewrites or discarding unrelated work. If prerequisites are not met, stop and report the blocker.
- Report the accepted version, PR URL/number, actual squash-merge SHA, verified local and remote branch heads, worktree/index status, and any verification or synchronization blockers. Stop after closure; do not start a new stage automatically.
- Before next-stage work, verify accepted remote state and reconcile local branches safely. Do not advance to another stage unless requested.



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
