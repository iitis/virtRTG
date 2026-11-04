"""Map accepted browser device poses into VirtualXRay reference geometry.

Browser positions are stored in meters and XYZ Euler angles in radians. The
returned projection geometry uses millimeters in the VirtualXRay local frame.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

from .device_adapter import DeviceState, _add, _norm, _rotate_xyz, _sub


IDENTITY_4 = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

# Current demonstrator construction geometry; these can be supplied by a
# future device profile without changing the mapping mathematics.
TUBE_FOCAL_SPOT_OFFSET_M = (0.0, -0.43, 0.0)
TUBE_EMISSION_AXIS = (0.0, -1.0, 0.0)
TUBE_BEAM_HORIZONTAL = (1.0, 0.0, 0.0)
TUBE_BEAM_VERTICAL = (0.0, 0.0, 1.0)
DETECTOR_ACTIVE_CENTER_OFFSET_M = (0.0, 0.04, 0.0)
COLLIMATION_REFERENCE_DISTANCE_MM = 1000.0


def _validate_rigid(transform: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Validate and normalize a rigid T_RS matrix whose translation is in mm."""
    if isinstance(transform, (str, bytes)):
        raise ValueError("T_RS must be a 4x4 numeric matrix")
    try:
        raw_rows = tuple(transform)
    except (TypeError, ValueError) as exc:
        raise ValueError("T_RS must be an iterable 4x4 numeric matrix") from exc
    if len(raw_rows) != 4:
        raise ValueError("T_RS must be a 4x4 matrix")
    rows = []
    for index, row in enumerate(raw_rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"T_RS row {index} must contain four numeric values")
        try:
            values = tuple(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T_RS row {index} must contain four numeric values") from exc
        if len(values) != 4:
            raise ValueError(f"T_RS row {index} must contain four values")
        converted = []
        for column, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"T_RS[{index}][{column}] must be a finite real number")
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"T_RS[{index}][{column}] must be a finite real number") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"T_RS[{index}][{column}] must be a finite real number")
            converted.append(numeric)
        rows.append(tuple(converted))
    rows = tuple(rows)
    if any(not math.isfinite(v) for row in rows for v in row):
        raise ValueError("T_RS must contain finite values")
    if any(abs(rows[3][i] - (1.0 if i == 3 else 0.0)) > 1e-9 for i in range(4)):
        raise ValueError("T_RS must be a rigid homogeneous transform")
    axes = [tuple(rows[r][c] for r in range(3)) for c in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-7:
                raise ValueError("T_RS rotation must be orthonormal")
    det = (axes[0][0] * (axes[1][1]*axes[2][2]-axes[1][2]*axes[2][1])
           - axes[1][0] * (axes[0][1]*axes[2][2]-axes[0][2]*axes[2][1])
           + axes[2][0] * (axes[0][1]*axes[1][2]-axes[0][2]*axes[1][1]))
    if abs(det - 1.0) > 1e-7:
        raise ValueError("T_RS must have a proper rotation (determinant +1)")
    return rows


def _rotate_mm(transform, vector):
    return tuple(sum(transform[r][c] * vector[c] for c in range(3)) for r in range(3))


def _point_mm(transform, point_m):
    point = tuple(v * 1000.0 for v in point_m)
    rotated = _rotate_mm(transform, point)
    return tuple(rotated[i] + transform[i][3] for i in range(3))


def _unit(v):
    length = _norm(v)
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return tuple(x / length for x in v)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def map_device_geometry(state: DeviceState, t_rs: Sequence[Sequence[float]] = IDENTITY_4) -> dict[str, Any]:
    """Return projection references in R and phantom pose separately.

    T_RS maps station coordinates S into VirtualXRay local reference R. Its
    translation is millimeters; T_WR is intentionally not part of this API.
    """
    transform = _validate_rigid(t_rs)
    tube, detector = state.poses["tube"], state.poses["detector"]

    def point_for(pose, offset_m):
        local = _rotate_xyz(offset_m, pose.rotation_rad)
        return _point_mm(transform, _add(pose.position_m, local))

    def direction_for(pose, axis):
        in_s = _rotate_xyz(axis, pose.rotation_rad)
        return _unit(_rotate_mm(transform, in_s))

    source = point_for(tube, TUBE_FOCAL_SPOT_OFFSET_M)
    beam_axis = direction_for(tube, TUBE_EMISSION_AXIS)
    beam_horizontal = direction_for(tube, TUBE_BEAM_HORIZONTAL)
    beam_vertical = direction_for(tube, TUBE_BEAM_VERTICAL)
    center = point_for(detector, DETECTOR_ACTIVE_CENTER_OFFSET_M)
    browser_horizontal = direction_for(detector, (1.0, 0.0, 0.0))
    browser_up = direction_for(detector, (0.0, 0.0, 1.0))
    browser_normal = direction_for(detector, (0.0, 1.0, 0.0))

    # VirtualXRay defines u = cross(up, normal). The browser's visible basis
    # has cross(+X, +Z) = -Y, while VirtualXRay requires u x v = normal.
    # Keep browser up and horizontal pixel directions; explicitly map the
    # supplied VirtualXRay normal opposite the browser detector-facing normal.
    plugin_normal = tuple(-v for v in browser_normal)
    plugin_horizontal = _unit(_cross(browser_up, plugin_normal))
    source_to_detector = _sub(center, source)
    sid = _norm(source_to_detector)
    # Coincident reference points have no defined ray. Preserve axes and
    # positions for diagnostics, and report the geometry as inconsistent.
    alignment = None
    if sid > 1e-12:
        toward_detector = tuple(component / sid for component in source_to_detector)
        alignment = sum(plugin_normal[i] * toward_detector[i] for i in range(3))
    phantom = state.poses["phantom"]
    return {
        "source_position_ref": source,
        "beam_axis_ref": beam_axis,
        "beam_horizontal_ref": beam_horizontal,
        "beam_vertical_ref": beam_vertical,
        "field_width_cm": state.collimator.width_cm,
        "field_height_cm": state.collimator.height_cm,
        "field_reference_distance_mm": COLLIMATION_REFERENCE_DISTANCE_MM,
        "detector_center_ref": center,
        "detector_normal_ref": plugin_normal,
        "detector_up_ref": browser_up,
        "detector_horizontal_ref": plugin_horizontal,
        "browser_detector_normal_ref": browser_normal,
        "browser_detector_horizontal_ref": browser_horizontal,
        "source_to_detector_distance_mm": sid,
        "units": "mm",
        "coordinate_frame": "R (VirtualXRay local reference)",
        "station_to_reference": transform,
        "global_reference_transform_applied": False,
        "detector_normal_ray_alignment": alignment,
        "orientation_consistent": (all(abs(plugin_horizontal[i] - browser_horizontal[i]) < 1e-7 for i in range(3))
                                   and all(abs(a - b) < 1e-7 for a, b in zip(_cross(browser_up, plugin_normal), plugin_horizontal))),
        "geometry_consistent": sid > 1e-8 and alignment is not None and alignment > 0.0,
        "phantom_pose_ref": {
            "position_ref": _point_mm(transform, phantom.position_m),
            "rotation_axes_ref": tuple(direction_for(phantom, axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        },
    }


def plugin_geometry_arguments(geometry: dict[str, Any]) -> dict[str, Any]:
    """Return the pose arguments accepted by XRayProjectionGeometry factory."""
    return {"source_position_ref": geometry["source_position_ref"],
            "detector_center_ref": geometry["detector_center_ref"],
            "detector_normal_ref": geometry["detector_normal_ref"],
            "detector_up_ref": geometry["detector_up_ref"]}
