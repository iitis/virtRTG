"""Protocol-v1 device state and projection reference geometry.

Positions enter as Three.js world coordinates in millimeters; rotations are
intrinsic Euler XYZ values in degrees, matching THREE.Euler order "XYZ".
Python stores meters and radians, with right-handed X lateral, Y up, Z depth.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any


OBJECTS = {"tube", "detector", "phantom", "table"}
LOGGER = logging.getLogger("virtualrtg.adapter")
DEFAULTS = {
    "tube": ((0, 2350, 0), (0, 0, 0)),
    "detector": ((0, 770, 0), (0, 0, 0)),
    "phantom": ((0, 1100, 0), (0, 0, 0)),
    "table": ((0, 835, 0), (0, 0, 0)),
}


class ProtocolError(ValueError):
    """A malformed or stale protocol message."""


@dataclass
class Pose:
    position_m: tuple[float, float, float]
    rotation_rad: tuple[float, float, float]


@dataclass
class Generator:
    kvp: float = 80
    ma: float = 200
    exposure_ms: float = 20
    mas: float = 4


@dataclass
class Collimator:
    width_cm: float = 24
    height_cm: float = 30
    light_enabled: bool = True


@dataclass
class Acquisition:
    status: str = "idle"
    request_id: str | None = None
    prepared_revision: int | None = None


@dataclass
class DeviceState:
    device_id: str = "browser-simulator-01"
    session_id: str | None = None
    last_timestamp: str | None = None
    capabilities: list[str] = field(default_factory=list)
    transport: str | None = None
    poses: dict[str, Pose] = field(default_factory=lambda: {
        key: Pose(tuple(v / 1000 for v in p), tuple(math.radians(x) for x in r))
        for key, (p, r) in DEFAULTS.items()
    })
    generator: Generator = field(default_factory=Generator)
    collimator: Collimator = field(default_factory=Collimator)
    procedure: str = "table"
    revision: int = 0
    last_sequence: int = 0
    acquisition: Acquisition = field(default_factory=Acquisition)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceState":
        state = cls()
        for key in ("device_id", "session_id", "last_timestamp", "capabilities", "transport", "procedure", "revision", "last_sequence"):
            if key in data:
                setattr(state, key, data[key])
        state.poses = {k: Pose(tuple(v["position_m"]), tuple(v["rotation_rad"])) for k, v in data.get("poses", {}).items()}
        state.generator = Generator(**data.get("generator", {}))
        state.collimator = Collimator(**data.get("collimator", {}))
        state.acquisition = Acquisition(**data.get("acquisition", {}))
        return state


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ProtocolError(f"{label} must be an array of {length} finite numbers")
    return tuple(float(v) for v in value)


def _pose(value: Any) -> Pose:
    if not isinstance(value, dict) or value.get("rotation_order") != "XYZ":
        raise ProtocolError("pose must declare rotation_order XYZ")
    p = _finite_vector(value.get("position_mm"), 3, "position_mm")
    r = _finite_vector(value.get("rotation_deg"), 3, "rotation_deg")
    return Pose(tuple(v / 1000 for v in p), tuple(math.radians(v) for v in r))


def _number(mapping: dict, key: str, minimum: float, maximum: float) -> float:
    v = mapping.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not minimum <= v <= maximum:
        raise ProtocolError(f"{key} must be finite and in [{minimum}, {maximum}]")
    return float(v)


def _rotate_xyz(v: tuple[float, float, float], angles: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply Three.js Euler XYZ quaternion-equivalent rotation to a local vector."""
    x, y, z = v
    a, b, c = angles
    # THREE.Euler XYZ matrix is R_x * R_y * R_z; act on vectors right to left.
    x, y, z = (math.cos(c)*x-math.sin(c)*y, math.sin(c)*x+math.cos(c)*y, z)
    x, y, z = (math.cos(b)*x+math.sin(b)*z, y, -math.sin(b)*x+math.cos(b)*z)
    return (x, math.cos(a)*y-math.sin(a)*z, math.sin(a)*y+math.cos(a)*z)


def _add(a, b): return tuple(x+y for x, y in zip(a, b))
def _sub(a, b): return tuple(x-y for x, y in zip(a, b))
def _norm(v): return math.sqrt(sum(x*x for x in v))


def projection_geometry(state: DeviceState) -> dict[str, Any]:
    """Derive demo source/detector references from accepted object transforms."""
    tube, detector = state.poses["tube"], state.poses["detector"]
    source = _add(tube.position_m, _rotate_xyz((0, -0.43, 0), tube.rotation_rad))
    center = _add(detector.position_m, _rotate_xyz((0, 0.04, 0), detector.rotation_rad))
    horizontal = _rotate_xyz((1, 0, 0), detector.rotation_rad)
    vertical = _rotate_xyz((0, 0, 1), detector.rotation_rad)
    normal = _rotate_xyz((0, 1, 0), detector.rotation_rad)
    return {"tube_focal_point_m": source, "detector_active_center_m": center,
            "detector_horizontal": horizontal, "detector_vertical": vertical,
            "detector_normal": normal, "source_to_detector_m": _norm(_sub(center, source))}


class DeviceAdapter:
    """Validates protocol v1 and atomically applies accepted messages."""
    def __init__(self, state: DeviceState | None = None, verbose: bool = False):
        self.state = state or DeviceState()
        self.verbose = verbose

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        kind = message.get("type") if isinstance(message, dict) else None
        seq = message.get("seq") if isinstance(message, dict) else None
        identifiers = {key: message[key] for key in ("device_id", "object", "request_id", "preset") if isinstance(message, dict) and key in message}
        LOGGER.info("Incoming type=%s seq=%s%s", kind or "?", seq if seq is not None else "?",
                    " " + " ".join(f"{key}={value}" for key, value in identifiers.items()) if identifiers else "")
        if not isinstance(message, dict) or message.get("protocol") != "virtualrtg.device" or message.get("version") != 1:
            raise ProtocolError("expected virtualrtg.device protocol version 1")
        seq = message.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= self.state.last_sequence:
            raise ProtocolError("seq must be a strictly increasing positive integer")
        if not isinstance(message.get("device_id"), str) or not message["device_id"] or not isinstance(message.get("timestamp"), str):
            raise ProtocolError("device_id and timestamp are required strings")
        kind, payload = message.get("type"), message
        if kind not in {"device.hello", "state.update", "device.pose", "procedure.select", "acquisition.prepare", "acquisition.request", "acquisition.cancel"}:
            raise ProtocolError(f"unsupported message type: {kind}")

        # Validate and stage all changes before mutating live state.
        changes: dict[str, Any] = {}
        if kind == "device.hello":
            caps = payload.get("capabilities", [])
            if not isinstance(caps, list) or any(not isinstance(x, str) for x in caps): raise ProtocolError("capabilities must be strings")
            changes.update(capabilities=caps[:], transport=payload.get("transport"))
        elif kind == "state.update":
            rev = payload.get("revision")
            if isinstance(rev, bool) or not isinstance(rev, int) or rev <= self.state.revision: raise ProtocolError("revision must increase")
            dev = payload.get("device")
            if not isinstance(dev, dict): raise ProtocolError("device object required")
            poses = dev.get("poses")
            if not isinstance(poses, dict) or set(poses) != OBJECTS: raise ProtocolError("poses must contain tube, detector, phantom, table")
            changes["poses"] = {k: _pose(v) for k, v in poses.items()}
            gen, col = dev.get("generator"), dev.get("collimator")
            if not isinstance(gen, dict) or not isinstance(col, dict): raise ProtocolError("generator and collimator objects required")
            changes["generator"] = Generator(_number(gen,"kvp",1,500), _number(gen,"ma",0.1,2000), _number(gen,"exposure_ms",0.1,100000), _number(gen,"mas",0,100000))
            light=col.get("light_enabled")
            if not isinstance(light,bool): raise ProtocolError("light_enabled must be boolean")
            changes["collimator"] = Collimator(_number(col,"width_cm",0.1,200),_number(col,"height_cm",0.1,200),light)
            changes["revision"] = rev
        elif kind == "device.pose":
            obj = payload.get("object")
            if obj not in OBJECTS: raise ProtocolError("unknown object identifier")
            rev = payload.get("revision")
            if isinstance(rev,bool) or not isinstance(rev,int) or rev <= self.state.revision: raise ProtocolError("revision must increase")
            changes.update(pose_update=(obj,_pose(payload.get("pose"))), revision=rev)
        elif kind == "procedure.select":
            preset=payload.get("preset")
            if preset not in {"table","lateral","upright"}: raise ProtocolError("unknown demonstration preset")
            changes["procedure"]=preset
        elif kind == "acquisition.prepare":
            rev=payload.get("revision")
            if isinstance(rev,bool) or not isinstance(rev,int) or rev != self.state.revision: raise ProtocolError("prepare revision must match accepted state")
            changes["acquisition"]=Acquisition("ready", None, rev)
        elif kind == "acquisition.request":
            rid=payload.get("request_id"); rev=payload.get("scene_revision"); params=payload.get("parameters")
            if not isinstance(rid,str) or not rid or rev != self.state.revision or not isinstance(params,dict): raise ProtocolError("request id, current scene_revision and parameters required")
            for k,lo,hi in (("kvp",1,500),("ma",.1,2000),("exposure_ms",.1,100000),("mas",0,100000)):_number(params,k,lo,hi)
            if self.state.acquisition.status != "ready": raise ProtocolError("acquisition must be prepared")
            changes["acquisition"]=Acquisition("requested",rid,self.state.revision)
        elif kind == "acquisition.cancel":
            changes["acquisition"]=Acquisition("cancelled")

        s=self.state
        previous_poses = s.poses.copy()
        previous_generator = s.generator
        previous_collimator = s.collimator
        previous_procedure = s.procedure
        previous_acquisition = s.acquisition
        if "pose_update" in changes:
            obj, pose=changes.pop("pose_update"); s.poses[obj]=pose
        for k,v in changes.items(): setattr(s,k,v)
        s.device_id=message["device_id"]; s.last_timestamp=message["timestamp"]; s.last_sequence=seq
        LOGGER.info("Accepted type=%s seq=%s", kind, seq)
        if s.generator != previous_generator:
            LOGGER.info("Generator: %.1f kVp, %.1f mA, %.1f ms, %.2f mAs", s.generator.kvp, s.generator.ma, s.generator.exposure_ms, s.generator.mas)
        if s.collimator != previous_collimator:
            LOGGER.info("Collimator: %.1f x %.1f cm, light=%s", s.collimator.width_cm, s.collimator.height_cm, "on" if s.collimator.light_enabled else "off")
        if s.procedure != previous_procedure:
            LOGGER.info("Procedure selected: %s", s.procedure)
        if s.acquisition != previous_acquisition:
            LOGGER.info("Acquisition: %s%s", s.acquisition.status, f" request_id={s.acquisition.request_id}" if s.acquisition.request_id else "")
        if s.poses != previous_poses:
            changed = [key for key in OBJECTS if s.poses[key] != previous_poses[key]]
            for object_id in changed:
                pose = s.poses[object_id]
                pos = tuple(round(v * 1000, 1) for v in pose.position_m)
                rot = tuple(round(math.degrees(v), 2) for v in pose.rotation_rad)
                LOGGER.info("Pose %s: position_mm=%s rotation_deg_XYZ=%s", object_id, pos, rot)
            try:
                _log_geometry(s, detailed=self.verbose)
            except Exception as exc:
                # Diagnostics run after state acceptance and must never turn
                # an accepted protocol message into a failed acknowledgement.
                LOGGER.warning("Geometry diagnostics failed after state acceptance: %s", exc)
        return {"protocol":"virtualrtg.device","version":1,"type":"response","in_reply_to":seq,"status":"accepted","timestamp":message["timestamp"],"detail":"protocol acknowledgement; no image generated"}

    def snapshot(self) -> dict[str, Any]:
        return {"device":self.state.to_dict(),"projection_geometry":projection_geometry(self.state),"image_generated":False}


def _log_geometry(state: DeviceState, detailed: bool = False) -> None:
    geometry = projection_geometry(state)
    source = tuple(round(v * 1000, 1) for v in geometry["tube_focal_point_m"])
    center = tuple(round(v * 1000, 1) for v in geometry["detector_active_center_m"])
    LOGGER.info("Projection geometry: source_mm=%s detector_center_mm=%s SID_mm=%.1f",
                source, center, geometry["source_to_detector_m"] * 1000)
    if detailed:
        for key in ("detector_horizontal", "detector_vertical", "detector_normal"):
            LOGGER.debug("  %s=%s", key, tuple(round(v, 6) for v in geometry[key]))
        # Keep the richer R-frame mapping opt-in and confined to pose changes.
        from .virtualrtg_geometry import map_device_geometry
        mapped = map_device_geometry(state)
        LOGGER.debug("VirtualXRay mapped geometry (mm, frame=R): source=%s detector=%s SID=%.2f",
                     tuple(round(v, 3) for v in mapped["source_position_ref"]),
                     tuple(round(v, 3) for v in mapped["detector_center_ref"]),
                     mapped["source_to_detector_distance_mm"])
        for key in ("detector_normal_ref", "detector_up_ref", "detector_horizontal_ref"):
            LOGGER.debug("  %s=%s", key, tuple(round(v, 6) for v in mapped[key]))
        LOGGER.debug("  T_RS=%s T_WR applied=%s orientation_consistent=%s geometry_consistent=%s normal_ray_dot=%s",
                     mapped["station_to_reference"], mapped["global_reference_transform_applied"],
                     mapped["orientation_consistent"], mapped["geometry_consistent"],
                     ("undefined" if mapped["detector_normal_ray_alignment"] is None else
                      f"{mapped['detector_normal_ray_alignment']:.6f}"))

