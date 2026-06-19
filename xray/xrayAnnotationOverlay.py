"""Project scene annotations into detector-space overlay primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Iterable

import numpy as np

from dpVision import AnnotationPath, AnnotationPoint

_log = logging.getLogger(__name__)

@dataclass
class XRayOverlayStyle:
	"""Describe one detector-space overlay style independent of scene object types."""

	color_rgba: tuple[int, int, int, int]
	line_width_px: int = 1
	marker_size_px: int = 6


@dataclass
class XRayOverlayItem:
	"""Base detector-space overlay item produced by annotation projectors."""

	kind: str
	label: str = ""
	visible: bool = True
	in_bounds: bool = True
	metadata: dict = field(default_factory=dict)


@dataclass
class XRayOverlayCross(XRayOverlayItem):
	"""Represent one cross marker centered at detector pixel coordinates."""

	pixel_uv: tuple[float, float] | None = None
	style: XRayOverlayStyle | None = None


@dataclass
class XRayOverlayPolyline(XRayOverlayItem):
	"""Represent one polyline overlay in detector pixel coordinates."""

	pixel_uvs: list[tuple[float, float]] = field(default_factory=list)
	style: XRayOverlayStyle | None = None
	closed: bool = False


@dataclass
class XRayOverlayProjectionSet:
	"""Cache all detector-space overlay items for one projected image."""

	detector_shape_hw: tuple[int, int]
	items: list[XRayOverlayItem] = field(default_factory=list)


@dataclass
class XRayAnnotationProjectionContext:
	"""Bundle projection geometry and transforms needed by annotation projectors."""

	geometry: object
	reference_transform: np.ndarray
	object_transform_resolver: object | None = None
	overlay_label_suffix: str = ""

	def object_point_to_reference(self, scene_object, point_xyz):
		"""Return one local object-space point expressed in the VirtualXRay reference frame."""
		local_point = np.asarray(point_xyz, dtype=np.float32)
		local_point_h = np.append(local_point, 1.0).astype(np.float32)
		if callable(self.object_transform_resolver):
			object_transform = np.asarray(self.object_transform_resolver(scene_object), dtype=np.float32)
			return (object_transform @ local_point_h)[:3].astype(np.float32)
		global_transform = np.asarray(scene_object.getGlobalTransformation(), dtype=np.float32)
		global_point_h = global_transform @ local_point_h
		reference_transform_inv = np.linalg.inv(np.asarray(self.reference_transform, dtype=np.float32))
		return (reference_transform_inv @ global_point_h)[:3].astype(np.float32)

	def format_overlay_label(self, base_label):
		"""Return one overlay label optionally suffixed with the current frame marker."""
		return f"{base_label}{self.overlay_label_suffix}" if self.overlay_label_suffix else str(base_label)

	def project_reference_point(self, point_ref):
		"""Project one reference-space point onto the detector plane."""
		point_ref = np.asarray(point_ref, dtype=np.float32)
		detector_origin = np.asarray(self.geometry.detector_origin_ref, dtype=np.float32)
		detector_u = np.asarray(self.geometry.detector_u_ref, dtype=np.float32)
		detector_v = np.asarray(self.geometry.detector_v_ref, dtype=np.float32)
		detector_normal = np.asarray(self.geometry.detector_normal_ref_vector(), dtype=np.float32)
		epsilon = 1e-8

		if self.geometry.is_cone_beam():
			ray_origin = np.asarray(self.geometry.source_position_ref, dtype=np.float32)
			ray_direction = point_ref - ray_origin
			if float(np.linalg.norm(ray_direction)) <= epsilon:
				return {
					"status": "at_source",
					"detector_point_ref": None,
					"detector_pixel_uv": None,
					"visible": False,
					"in_bounds": False,
				}
		else:
			ray_origin = point_ref
			ray_direction = np.asarray(self.geometry.ray_direction_ref, dtype=np.float32)
			if float(np.linalg.norm(ray_direction)) <= epsilon:
				return {
					"status": "invalid_direction",
					"detector_point_ref": None,
					"detector_pixel_uv": None,
					"visible": False,
					"in_bounds": False,
				}

		denominator = float(np.dot(ray_direction, detector_normal))
		if abs(denominator) <= epsilon:
			return {
				"status": "parallel_to_detector",
				"detector_point_ref": None,
				"detector_pixel_uv": None,
				"visible": False,
				"in_bounds": False,
			}

		ray_parameter = float(np.dot(detector_origin - ray_origin, detector_normal) / denominator)
		if ray_parameter < 0.0:
			return {
				"status": "behind_ray_origin",
				"detector_point_ref": None,
				"detector_pixel_uv": None,
				"visible": False,
				"in_bounds": False,
			}

		detector_point_ref = ray_origin + ray_direction * ray_parameter
		detector_delta = detector_point_ref - detector_origin
		u_denominator = max(float(np.dot(detector_u, detector_u)), epsilon)
		v_denominator = max(float(np.dot(detector_v, detector_v)), epsilon)
		u_coord = float(np.dot(detector_delta, detector_u) / u_denominator)
		v_coord = float(np.dot(detector_delta, detector_v) / v_denominator)
		height = int(self.geometry.detector_shape_hw[0])
		width = int(self.geometry.detector_shape_hw[1])
		in_bounds = 0.0 <= u_coord <= float(width - 1) and 0.0 <= v_coord <= float(height - 1)
		return {
			"status": "projected",
			"detector_point_ref": detector_point_ref.astype(np.float32),
			"detector_pixel_uv": (u_coord, v_coord),
			"visible": True,
			"in_bounds": in_bounds,
		}

	def default_style_for(self, annotation_object, marker_size_px=6, line_width_px=1):
		"""Build one default overlay style from the active annotation color."""
		color = annotation_object.getSelColor() if annotation_object.checked else annotation_object.getColor()
		return XRayOverlayStyle(
			color_rgba=(color.red(), color.green(), color.blue(), color.alpha()),
			line_width_px=max(1, int(line_width_px)),
			marker_size_px=max(1, int(marker_size_px)),
		)


class BaseXRayAnnotationProjector:
	"""Abstract adapter turning one scene annotation into detector-space overlay items."""

	scene_type = object

	def project(self, scene_object, context: XRayAnnotationProjectionContext) -> list[XRayOverlayItem]:
		"""Return detector-space overlay items for one scene annotation."""
		raise NotImplementedError


class AnnotationPointProjector(BaseXRayAnnotationProjector):
	"""Project `AnnotationPoint` into one cross overlay with optional label metadata."""

	scene_type = AnnotationPoint

	def project(self, scene_object, context: XRayAnnotationProjectionContext) -> list[XRayOverlayItem]:
		point_ref = context.object_point_to_reference(scene_object, scene_object.getPoint())
		projection = context.project_reference_point(point_ref)
		return [
			XRayOverlayCross(
				kind="AnnotationPoint",
				label=context.format_overlay_label(scene_object.label),
				pixel_uv=projection["detector_pixel_uv"],
				style=context.default_style_for(scene_object),
				visible=bool(projection["visible"]),
				in_bounds=bool(projection["in_bounds"]),
				metadata={
					"status": str(projection["status"]),
					"source_point_ref": point_ref,
					"detector_point_ref": projection["detector_point_ref"],
					"show_vector": bool(getattr(scene_object, "m_showVector", False)),
					"vector": None if scene_object.getVector() is None else tuple(scene_object.getVector()),
				},
			)
		]


class AnnotationPathProjector(BaseXRayAnnotationProjector):
	"""Project `AnnotationPath` into one detector-space polyline overlay."""

	scene_type = AnnotationPath

	def project(self, scene_object, context: XRayAnnotationProjectionContext) -> list[XRayOverlayItem]:
		projected_points = []
		is_any_visible = False
		is_any_in_bounds = False
		for point_xyz in getattr(scene_object, "m_points", []):
			point_ref = context.object_point_to_reference(scene_object, point_xyz)
			projection = context.project_reference_point(point_ref)
			if projection["detector_pixel_uv"] is not None:
				projected_points.append(tuple(projection["detector_pixel_uv"]))
			is_any_visible = is_any_visible or bool(projection["visible"])
			is_any_in_bounds = is_any_in_bounds or bool(projection["in_bounds"])

		return [
			XRayOverlayPolyline(
				kind="AnnotationPath",
				label=context.format_overlay_label(scene_object.label),
				pixel_uvs=projected_points,
				style=context.default_style_for(
					scene_object,
					marker_size_px=6,
					line_width_px=max(1, int(round(getattr(scene_object, "m_width", 1.0)))),
				),
				visible=is_any_visible and len(projected_points) >= 2,
				in_bounds=is_any_in_bounds,
				metadata={"point_count": len(projected_points)},
			)
		]


DEFAULT_PROJECTORS = (
	AnnotationPointProjector(),
	AnnotationPathProjector(),
)


def _json_ready_overlay_value(value):
	"""Convert nested overlay metadata into a JSON-serializable structure."""
	if isinstance(value, np.ndarray):
		return _json_ready_overlay_value(value.tolist())
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		return {str(key): _json_ready_overlay_value(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_ready_overlay_value(item) for item in value]
	return value


def _overlay_style_to_payload(style: XRayOverlayStyle | None):
	"""Serialize one optional overlay style into a JSON-ready mapping."""
	if style is None:
		return None
	return {
		"color_rgba": [int(channel) for channel in style.color_rgba],
		"line_width_px": int(style.line_width_px),
		"marker_size_px": int(style.marker_size_px),
	}


def _overlay_style_from_payload(payload):
	"""Deserialize one optional overlay style payload."""
	if payload is None:
		return None
	return XRayOverlayStyle(
		color_rgba=tuple(int(channel) for channel in payload.get("color_rgba", [255, 0, 0, 255])),
		line_width_px=max(1, int(payload.get("line_width_px", 1))),
		marker_size_px=max(1, int(payload.get("marker_size_px", 6))),
	)


def overlay_projection_set_to_payload(projection_set: XRayOverlayProjectionSet | None):
	"""Serialize one projected overlay set into a JSON-ready mapping."""
	if projection_set is None:
		return None
	payload_items = []
	for item in getattr(projection_set, "items", []):
		item_payload = {
			"kind": str(getattr(item, "kind", "")),
			"label": str(getattr(item, "label", "")),
			"visible": bool(getattr(item, "visible", True)),
			"in_bounds": bool(getattr(item, "in_bounds", True)),
			"metadata": _json_ready_overlay_value(getattr(item, "metadata", {})),
			"style": _overlay_style_to_payload(getattr(item, "style", None)),
		}
		if isinstance(item, XRayOverlayCross):
			item_payload["type"] = "cross"
			item_payload["pixel_uv"] = None if item.pixel_uv is None else [float(item.pixel_uv[0]), float(item.pixel_uv[1])]
		elif isinstance(item, XRayOverlayPolyline):
			item_payload["type"] = "polyline"
			item_payload["pixel_uvs"] = [
				[float(pixel_uv[0]), float(pixel_uv[1])]
				for pixel_uv in item.pixel_uvs
			]
			item_payload["closed"] = bool(item.closed)
		else:
			item_payload["type"] = "item"
		payload_items.append(item_payload)
	return {
		"detector_shape_hw": [
			int(projection_set.detector_shape_hw[0]),
			int(projection_set.detector_shape_hw[1]),
		],
		"items": payload_items,
	}


def overlay_projection_set_from_payload(payload):
	"""Deserialize one projected overlay set from a JSON-ready mapping."""
	if not payload:
		return None
	items = []
	for item_payload in payload.get("items", []):
		common_kwargs = {
			"kind": str(item_payload.get("kind", "")),
			"label": str(item_payload.get("label", "")),
			"visible": bool(item_payload.get("visible", True)),
			"in_bounds": bool(item_payload.get("in_bounds", True)),
			"metadata": dict(item_payload.get("metadata", {})),
		}
		style = _overlay_style_from_payload(item_payload.get("style", None))
		item_type = str(item_payload.get("type", "item")).lower()
		if item_type == "cross":
			pixel_uv = item_payload.get("pixel_uv", None)
			items.append(XRayOverlayCross(
				pixel_uv=None if pixel_uv is None else (float(pixel_uv[0]), float(pixel_uv[1])),
				style=style,
				**common_kwargs,
			))
		elif item_type == "polyline":
			items.append(XRayOverlayPolyline(
				pixel_uvs=[
					(float(pixel_uv[0]), float(pixel_uv[1]))
					for pixel_uv in item_payload.get("pixel_uvs", [])
				],
				style=style,
				closed=bool(item_payload.get("closed", False)),
				**common_kwargs,
			))
		else:
			items.append(XRayOverlayItem(**common_kwargs))
	detector_shape_hw = payload.get("detector_shape_hw", [1, 1])
	return XRayOverlayProjectionSet(
		detector_shape_hw=(int(detector_shape_hw[0]), int(detector_shape_hw[1])),
		items=items,
	)


def build_overlay_projection_set(descendants: Iterable[object], context: XRayAnnotationProjectionContext, projectors=None):
	"""Project supported scene annotations into a detector-space overlay set."""
	projectors = tuple(DEFAULT_PROJECTORS if projectors is None else projectors)
	projected_items: list[XRayOverlayItem] = []
	for scene_object in descendants:
		if scene_object.visible:
			for projector in projectors:
				if isinstance(scene_object, projector.scene_type):
					projected_items.extend(projector.project(scene_object, context))
					break
		else:
			_log.debug(f"Skipping invisible object {scene_object.label} during overlay projection.")
	return XRayOverlayProjectionSet(
		detector_shape_hw=(int(context.geometry.detector_shape_hw[0]), int(context.geometry.detector_shape_hw[1])),
		items=projected_items,
	)
