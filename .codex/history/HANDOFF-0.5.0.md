# virtRTG 0.5.0 Handoff: Directional X-ray Beam Geometry

Status: **Accepted**.

## Implementation summary

Added an optional rectangular divergent beam value object and populated it from protocol-v1 device state. Beam geometry is descriptive only and does not affect projection pixels.

## Files changed

- `xray/xrayProjection.py`: added `RectangularBeamGeometry`, frame and dimension validation, and point/ray containment predicates.
- `virtualXRay.py`: added the optional `directional_beam_geometry` property, defaulting to `None` for legacy scenes.
- `device_link/virtualrtg_geometry.py`: maps the tube focal spot and beam frame from the accepted tube-local construction values; includes collimator field dimensions and 1000 mm reference distance.
- `device_link/integration.py`: applies the beam geometry with existing source/detector geometry and restores it on failed geometry rebuild.
- `tests/unit/device_link/test_virtualrtg_geometry.py`: added rotation, translation, detector independence, rigid mapping, beam boundary, and invalid-frame coverage.
- `tests/unit/device_link/test_device_server.py`: added a Qt integration regression using the actual `VirtualXRay` and `DeviceLinkIntegration` path.

## Beam frame and coordinate transformations

Beam axis, horizontal axis, and vertical axis start from tube-local `(0,-1,0)`, `(1,0,0)`, and `(0,0,1)`. They are rotated using the accepted Three.js XYZ convention, then transformed with only the rotational part of `T_RS`. The focal spot starts at tube-local `(0,-0.43,0)` meters and uses the complete `T_RS` transform after conversion to millimeters. `T_WR` is not applied. The detector normal reversal and detector pose mapping remain unchanged. Beam axis comes from the tube and is not inferred from the detector.

The field dimensions remain centimeters and refer to the plane 1000 mm from the focal spot. `RectangularBeamGeometry` stores unit orthogonal right-handed axes and positive finite field dimensions. Its point predicate uses millimeter coordinates and the specified divergent half-width/half-height inequalities; its ray predicate tests a focal-spot ray direction.

## Adapter integration and compatibility

Protocol `virtualrtg.device` version 1 is unchanged. The adapter uses the existing tube pose and collimator dimensions; no browser files or messages changed. Existing scenes start with `directional_beam_geometry = None` and projection geometry construction ignores this optional field. Device updates continue not to trigger projections. Integration rollback restores both the previous source/detector vectors and prior beam value if rebuilding fails. VirtualXRay global placement and anatomy are not modified.

## Verification

- Focused device-link and geometry regression tests: **37 passed, 10 subtests passed**.
- Final complete virtRTG unit suite: **93 passed, 5 skipped, 10 subtests passed**.
- The Qt integration regression verifies initial beam population, tube rotation with unchanged detector mapping, rollback of beam and source/detector fields after a failed rebuild, and no automatic projection call.
- Ruff F401 findings in `virtualXRay.py` and `xray/xrayProjection.py` were confirmed unchanged from `HEAD`; the changed device-link and test modules pass F401 checks.
- `git diff --check`: passed.

The suite includes actual VirtualXRay and projection geometry coverage alongside mapper tests.

## Limitations

Directional beam geometry does not yet restrict the generated X-ray projection. No exposure mask, ray suppression, detector border, or projection algorithm changes were introduced. Browser/profile configuration remains fixed at the centralized mapping conventions described above; no profile negotiation or device configuration protocol exists.
