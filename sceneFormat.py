# -*- coding: utf-8 -*-
"""Serialize one virtRTG scene subtree into an ATMDL-like XML document."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from dpVision import AP, Mesh, Transform, Volumetric
from dpVision.annotationPath import AnnotationPath
from dpVision.annotationPoint import AnnotationPoint
from dpVision.volumetric import SliceMetadata
from dpVision.parser import Parser

from .virtualXRay import VirtualXRay
from .xray.xraySource import (
	XRayMaterialResponseConfig,
	ensure_xray_source_config,
	get_xray_material_response_config,
	set_xray_material_response_config,
)


SCENE_FORMAT_NAME = "virtRTG-scene"
SCENE_FORMAT_VERSION = 2
ATMDL_VIRTUAL_XRAY_TOKENS = ("virtualXRay", "virtual_xray", "vxray")

_ATMDL_INTEGRATION_REGISTERED = False


def _vec_to_text(values):
	"""Return one compact numeric vector string."""
	arr = np.asarray(values, dtype=np.float64).reshape(-1)
	return " ".join(f"{value:.16g}" for value in arr.tolist())


def _matrix_to_text(matrix):
	"""Return one row-major homogeneous matrix string."""
	return _vec_to_text(np.asarray(matrix, dtype=np.float64).reshape(16))


def _json_text(payload):
	"""Return stable pretty JSON text for one payload mapping."""
	return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _json_compact_text(payload):
	"""Return one compact JSON string safe for single-line ATMDL payload storage."""
	return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _encode_payload64(payload):
	"""Return one URL-safe base64 string with compact JSON payload."""
	raw = _json_compact_text(payload).encode("utf-8")
	return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_payload64(payload64):
	"""Decode one URL-safe base64 JSON payload from ATMDL."""
	raw = base64.urlsafe_b64decode(str(payload64).encode("ascii"))
	return json.loads(raw.decode("utf-8"))


def _array_payload(array):
	"""Return one JSON-ready descriptor for a float32 detector array."""
	array = np.asarray(array, dtype=np.float32)
	if array.ndim != 2:
		raise ValueError("Projection cache arrays must be 2D.")
	return {
		"dtype": "float32",
		"shape": [int(array.shape[0]), int(array.shape[1])],
		"data_b64": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
	}


def _array_from_payload(payload):
	"""Decode one float32 detector array descriptor created by `_array_payload`."""
	if payload is None:
		return None
	shape = [int(v) for v in payload.get("shape", [])]
	if len(shape) != 2:
		raise ValueError("Projection cache payload is missing a valid 2D shape.")
	if str(payload.get("dtype", "float32")).lower() != "float32":
		raise ValueError("Projection cache payload currently supports only float32 arrays.")
	raw = base64.b64decode(str(payload.get("data_b64", "")).encode("ascii"))
	array = np.frombuffer(raw, dtype=np.float32)
	expected_size = int(shape[0]) * int(shape[1])
	if array.size != expected_size:
		raise ValueError("Projection cache payload size does not match its declared shape.")
	return array.reshape(shape).astype(np.float32, copy=False)


def _projection_cache_payload(virtual_xray):
	"""Return one JSON-ready payload for cached detector buffers."""
	cache_payload = virtual_xray.projection_cache_payload()
	if cache_payload is None:
		return None
	payload = {
		"schema": "virtRTG-projection-cache",
		"version": 1,
		"raw_projection": None,
		"line_integral_projection": None,
		"source_projections": [],
	}
	if cache_payload.get("raw_projection", None) is not None:
		payload["raw_projection"] = _array_payload(cache_payload["raw_projection"])
	if cache_payload.get("line_integral_projection", None) is not None:
		payload["line_integral_projection"] = _array_payload(cache_payload["line_integral_projection"])
	for source_payload in cache_payload.get("source_projections", []):
		payload["source_projections"].append({
			"source_index": int(source_payload.get("source_index", 0)),
			"label": str(source_payload.get("label", "")),
			"source_type": str(source_payload.get("source_type", "")),
			"raw_projection": (
				None if source_payload.get("raw_projection", None) is None
				else _array_payload(source_payload["raw_projection"])
			),
			"line_integral_projection": (
				None if source_payload.get("line_integral_projection", None) is None
				else _array_payload(source_payload["line_integral_projection"])
			),
		})
	return payload


def _apply_projection_cache_payload(virtual_xray, payload):
	"""Decode one cached projection payload and apply it to the object."""
	if not payload:
		return
	cache_payload = {
		"raw_projection": _array_from_payload(payload.get("raw_projection", None)),
		"line_integral_projection": _array_from_payload(payload.get("line_integral_projection", None)),
		"source_projections": [],
	}
	for source_payload in payload.get("source_projections", []):
		cache_payload["source_projections"].append({
			"source_index": int(source_payload.get("source_index", 0)),
			"label": str(source_payload.get("label", "")),
			"source_type": str(source_payload.get("source_type", "")),
			"raw_projection": _array_from_payload(source_payload.get("raw_projection", None)),
			"line_integral_projection": _array_from_payload(source_payload.get("line_integral_projection", None)),
		})
	virtual_xray.apply_projection_cache_payload(cache_payload)


def _atmdl_bool_text(value):
	"""Return one compact ATMDL-friendly boolean literal."""
	return "1" if bool(value) else "0"


def _atmdl_optional_scalar_text(value):
	"""Return one ATMDL-friendly scalar literal preserving explicit nulls."""
	if value is None:
		return "none"
	if isinstance(value, bool):
		return _atmdl_bool_text(value)
	return f"{float(value):.16g}" if isinstance(value, (int, float)) else str(value)


def _atmdl_read_optional_float(value, default=None):
	"""Parse one optional float literal used in text ATMDL plugin sections."""
	if value is None:
		return default
	text = str(value).strip().lower()
	if text in {"", "none", "null"}:
		return default
	return float(value)


def _atmdl_read_bool(value, default=False):
	"""Parse one ATMDL boolean literal."""
	if value is None:
		return bool(default)
	text = str(value).strip().lower()
	if text in {"1", "true", "yes", "on"}:
		return True
	if text in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _source_links_for_object(obj):
	"""Return best-effort source-file links for one scene object."""
	source_reference = Parser.get_source_reference(obj)
	if source_reference is not None:
		if source_reference["kind"] == "dicom_series":
			return {
				"kind": "dicom_series",
				"files": [str(Path(path)) for path in source_reference["files"]],
			}
		if source_reference["kind"] == "file":
			return {
				"kind": "file",
				"path": str(Path(source_reference["path"])),
			}
	return None


def _default_save_name_for_object(obj, default_ext):
	"""Return one default export file name for an object save dialog."""
	default_name = str(getattr(obj, "label", obj.__class__.__name__) or obj.__class__.__name__)
	if Path(default_name).suffix == "":
		default_name = default_name + default_ext
	return default_name


def _choose_save_path_for_object(obj, parent_widget=None):
	"""Ask the user where one source object should be saved."""
	save_parsers = Parser.getSaveParsers(obj)
	if not save_parsers:
		return None
	save_filters = Parser.getSaveExts(obj)
	default_ext = save_parsers[0].save_exts[0]
	file_name, selected_filter = QFileDialog.getSaveFileName(
		parent_widget or AP.mainWin,
		"Save Source Object",
		_default_save_name_for_object(obj, default_ext),
		save_filters,
	)
	if not file_name:
		return None
	file_path = Path(file_name)
	if file_path.suffix == "":
		filter_text = selected_filter if selected_filter else save_filters.split(";;")[0]
		filter_suffix = default_ext
		if "*." in filter_text:
			filter_suffix = "." + filter_text.split("*.")[1].split(")")[0].split()[0].strip(";")
		file_path = file_path.with_suffix(filter_suffix)
	return str(file_path)


def _ensure_source_links_for_object(obj, parent_widget=None, interactive=False):
	"""Ensure one exported source object has a reusable source link."""
	if _source_links_for_object(obj) is not None:
		return True
	if not interactive:
		return False

	save_parsers = Parser.getSaveParsers(obj)
	object_label = str(getattr(obj, "label", obj.__class__.__name__) or obj.__class__.__name__)
	if not save_parsers:
		QMessageBox.warning(
			parent_widget or AP.mainWin,
			"Export virtRTG Scene",
			f"Obiekt '{object_label}' nie ma powiązanej ścieżki źródłowej i nie obsługuje zapisu do pliku.",
		)
		return False

	answer = QMessageBox.question(
		parent_widget or AP.mainWin,
		"Export virtRTG Scene",
		f"Obiekt '{object_label}' nie ma powiązanej ścieżki źródłowej.\nCzy zapisać go teraz, aby dodać link do eksportowanej sceny?",
		QMessageBox.Yes | QMessageBox.No,
		QMessageBox.Yes,
	)
	if answer != QMessageBox.Yes:
		return False

	save_path = _choose_save_path_for_object(obj, parent_widget=parent_widget)
	if not save_path:
		return False
	if not Parser.save(obj, save_path):
		QMessageBox.warning(
			parent_widget or AP.mainWin,
			"Export virtRTG Scene",
			f"Nie udało się zapisać obiektu '{object_label}' do pliku:\n{save_path}",
		)
		return False
	save_warnings = Parser.consume_save_warnings(obj)
	if save_warnings:
		QMessageBox.warning(
			parent_widget or AP.mainWin,
			"Export virtRTG Scene",
			"\n\n".join(save_warnings),
		)
	return _source_links_for_object(obj) is not None


def _validate_scene_source_links(virtual_xray, parent_widget=None, interactive=False):
	"""Ensure all exported source objects can be referenced from the scene file."""
	missing_labels = []
	for obj in [virtual_xray, *_iter_export_nodes(virtual_xray)]:
		if not isinstance(obj, (Volumetric, Mesh)):
			continue
		if not _ensure_source_links_for_object(obj, parent_widget=parent_widget, interactive=interactive):
			missing_labels.append(str(getattr(obj, "label", obj.__class__.__name__)))
	if missing_labels:
		raise ValueError(
			"Brak ścieżek źródłowych dla obiektów: " + ", ".join(missing_labels)
		)


def _make_missing_volume_placeholder(files):
	"""Create one renderer-safe placeholder volumetric object preserving source links."""
	obj = Volumetric()
	obj.m_volume = [np.zeros((1, 1), dtype=np.float32)]
	obj.shape = (1, 1, 1)
	obj.metadata = [SliceMetadata()]
	obj.m_dicom_files = [str(Path(path)) for path in files]
	obj.m_min = 0.0
	obj.m_max = 0.0
	obj.m_minDisplWin = 0.0
	obj.m_maxDisplWin = 1.0
	obj.m_minSlice = 0
	obj.m_maxSlice = 0
	obj.m_minRow = 0
	obj.m_maxRow = 0
	obj.m_minColumn = 0
	obj.m_maxColumn = 0
	obj.visible = False
	obj._virt_rtg_missing_source_placeholder = True
	Parser.set_source_reference(obj, {"kind": "dicom_series", "files": files})
	return obj


def _source_config_payload(obj):
	"""Return one JSON-ready payload for a mesh or volumetric X-ray source."""
	payload = {
		"schema": "virtRTG-source-config",
		"version": 1,
		"enabled": bool(getattr(obj, "xray_source_enabled", True)),
		"scalar_scale": float(getattr(obj, "xray_scalar_scale", 1.0)),
		"scalar_bias": float(getattr(obj, "xray_scalar_bias", 0.0)),
		"attenuation_multiplier": float(getattr(obj, "xray_attenuation_multiplier", 1.0)),
		"material_response_config": get_xray_material_response_config(obj).to_mapping(),
	}
	source_links = _source_links_for_object(obj)
	if source_links is not None:
		payload["source_links"] = source_links

	if isinstance(obj, Volumetric):
		payload["source_type"] = "volumetric"
		payload["interpolation_override"] = str(getattr(obj, "xray_interpolation_override", "default")).lower()
		payload["fill_value_override_enabled"] = bool(getattr(obj, "xray_fill_value_override_enabled", False))
		payload["fill_value_override"] = float(getattr(obj, "xray_fill_value_override", 0.0))
		payload["volume_backend"] = str(getattr(obj, "xray_volume_backend", "sampling")).lower()
	else:
		payload["source_type"] = "mesh"
		payload["mesh_backend"] = str(getattr(obj, "xray_mesh_backend", "analytic_bvh")).lower()
		payload["mesh_mode"] = str(getattr(obj, "xray_mesh_mode", "solid")).lower()
		payload["mesh_scalar_value"] = float(getattr(obj, "xray_mesh_scalar_value", 1800.0))
		payload["mesh_shell_thickness_mm"] = float(getattr(obj, "xray_mesh_shell_thickness_mm", 1.0))
	return payload


def _annotation_payload(obj):
	"""Return one JSON-ready payload for supported annotation types."""
	if isinstance(obj, AnnotationPoint):
		return {
			"schema": "virtRTG-annotation-point",
			"version": 1,
			"point": list(np.asarray(obj.getPoint(), dtype=np.float64).reshape(3)),
			"vector": None if obj.getVector() is None else list(np.asarray(obj.getVector(), dtype=np.float64).reshape(3)),
			"show_vector": bool(getattr(obj, "m_showVector", False)),
		}
	if isinstance(obj, AnnotationPath):
		return {
			"schema": "virtRTG-annotation-path",
			"version": 1,
			"width": float(getattr(obj, "m_width", 1.0)),
			"points": [
				list(np.asarray(point, dtype=np.float64).reshape(3))
				for point in getattr(obj, "m_points", [])
			],
		}
	return None


def _virtual_xray_payload(virtual_xray):
	"""Return one JSON-ready payload for the VirtualXRay root node."""
	return {
		"schema": "virtRTG-virtual-xray",
		"version": 2,
		"geometry": {
			"projection_mode": str(virtual_xray.projection_mode),
			"detector_center_ref": list(np.asarray(virtual_xray.detector_center_ref, dtype=np.float64).reshape(3)),
			"detector_normal_ref": list(np.asarray(virtual_xray.detector_normal_ref, dtype=np.float64).reshape(3)),
			"detector_up_ref": list(np.asarray(virtual_xray.detector_up_ref, dtype=np.float64).reshape(3)),
			"detector_shape_hw": [int(virtual_xray.detector_shape_hw[0]), int(virtual_xray.detector_shape_hw[1])],
			"detector_pixel_size_mm": [float(virtual_xray.detector_pixel_size_mm[0]), float(virtual_xray.detector_pixel_size_mm[1])],
			"source_position_ref": list(np.asarray(virtual_xray.source_position_ref, dtype=np.float64).reshape(3)),
			"ray_direction_ref": list(np.asarray(virtual_xray.ray_direction_ref, dtype=np.float64).reshape(3)),
			"step_mm": float(virtual_xray.step_mm),
			"quality_profile_name": str(virtual_xray.quality_profile_name),
			"depth_window_mode": str(getattr(virtual_xray, "depth_window_mode", "off")),
			"depth_window_mm": [float(v) for v in getattr(virtual_xray, "depth_window_mm", [0.0, 0.0])],
			"depth_window_origin_ref": list(np.asarray(getattr(virtual_xray, "depth_window_origin_ref", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)),
			"depth_window_axis_ref": list(np.asarray(getattr(virtual_xray, "depth_window_axis_ref", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)),
		},
		"source_defaults": {
			"interpolation": str(virtual_xray.source_interpolation),
			"fill_value": virtual_xray.source_fill_value,
			"preprocess_mode": str(virtual_xray.source_preprocess_mode),
			"preprocess_low_percentile": float(virtual_xray.source_preprocess_low_percentile),
			"preprocess_high_percentile": float(virtual_xray.source_preprocess_high_percentile),
			"preprocess_output_low": float(virtual_xray.source_preprocess_output_low),
			"preprocess_output_high": float(virtual_xray.source_preprocess_output_high),
			"mesh_source_scalar_value": float(virtual_xray.mesh_source_scalar_value),
			"mesh_source_mode": str(virtual_xray.mesh_source_mode),
			"mesh_surface_thickness_mm": float(virtual_xray.mesh_surface_thickness_mm),
		},
		"physics": {
			"mu_air": float(virtual_xray.physics_mu_air),
			"mu_water": float(virtual_xray.physics_mu_water),
			"hounsfield_air": float(virtual_xray.physics_hounsfield_air),
			"attenuation_scale": float(virtual_xray.physics_attenuation_scale),
			"source_energy_kev": float(virtual_xray.physics_source_energy_kev),
			"reference_energy_kev": float(virtual_xray.physics_reference_energy_kev),
			"attenuation_energy_exponent": float(virtual_xray.physics_attenuation_energy_exponent),
			"output_mode": str(virtual_xray.physics_output_mode),
			"intensity_floor": float(virtual_xray.physics_intensity_floor),
			"source_distance_falloff_mode": str(virtual_xray.physics_source_distance_falloff_mode),
			"source_distance_reference_mm": virtual_xray.physics_source_distance_reference_mm,
			"source_distance_power": float(virtual_xray.physics_source_distance_power),
			"material_response_mode": str(virtual_xray.physics_material_response_mode),
			"bone_threshold_hu": virtual_xray.physics_bone_threshold_hu,
			"bone_threshold_softness": float(virtual_xray.physics_bone_threshold_softness),
			"material_window_center": virtual_xray.physics_material_window_center,
			"material_window_width": virtual_xray.physics_material_window_width,
			"material_window_mode": str(virtual_xray.physics_material_window_mode),
			"material_window_softness": float(virtual_xray.physics_material_window_softness),
		},
		"presentation": {
			"mode": str(virtual_xray.presentation_mode),
			"invert": bool(virtual_xray.presentation_invert),
			"gamma": float(virtual_xray.presentation_gamma),
			"contrast": float(virtual_xray.presentation_contrast),
			"robust_percentile": float(virtual_xray.presentation_robust_percentile),
			"window_center": virtual_xray.presentation_window_center,
			"window_width": virtual_xray.presentation_window_width,
			"overlay_annotations": bool(getattr(virtual_xray, "presentation_overlay_annotations", False)),
			"overlay_labels": bool(getattr(virtual_xray, "presentation_overlay_labels", False)),
			"overlay_cross_size_px": int(getattr(virtual_xray, "presentation_overlay_cross_size_px", 6)),
		},
		"projection_cache": _projection_cache_payload(virtual_xray),
	}


def _iter_export_nodes(virtual_xray, node=None):
	"""Yield exportable descendants while treating nested VirtualXRay nodes as separate scenes."""
	node = virtual_xray if node is None else node
	for child in node.children():
		yield child
		if child.__class__.__name__ == "VirtualXRay":
			continue
		yield from _iter_export_nodes(virtual_xray, child)


def _iter_export_child_objects(virtual_xray):
	"""Yield standard descendants of one VirtualXRay, skipping nested sub-scenes."""
	for child in virtual_xray.children():
		yield child
		if isinstance(child, VirtualXRay):
			continue
		yield from _iter_export_nodes(virtual_xray, child)


def _child_index_path(root, target):
	"""Return one stable child-index path from root to target within the exported subtree."""
	path = []
	current = target
	while current is not None and current is not root:
		parent = getattr(current, "parent", None)
		if parent is None:
			return None
		children = list(parent.children())
		try:
			index = children.index(current)
		except ValueError:
			return None
		path.append(index)
		current = parent
	return list(reversed(path)) if current is root else None


def _resolve_child_index_path(root, path):
	"""Resolve one child-index path against the already parsed subtree."""
	current = root
	for index in path:
		children = list(current.children())
		if index < 0 or index >= len(children):
			return None
		current = children[index]
	return current


def _virtual_xray_atmdl_payload(virtual_xray):
	"""Return one ATMDL payload containing VirtualXRay and per-source configs."""
	source_nodes = []
	for obj in _iter_export_child_objects(virtual_xray):
		if not isinstance(obj, (Volumetric, Mesh)):
			continue
		index_path = _child_index_path(virtual_xray, obj)
		if index_path is None:
			continue
		source_nodes.append(
			{
				"child_index_path": index_path,
				"payload": _source_config_payload(obj),
			}
		)
	return {
		"schema": "virtRTG-atmdl-virtual-xray",
		"version": 2,
		"virtual_xray": _virtual_xray_payload(virtual_xray),
		"source_nodes": source_nodes,
	}


def _apply_virtual_xray_atmdl_payload(virtual_xray, payload):
	"""Apply one ATMDL payload to the parsed VirtualXRay subtree."""
	virtual_xray_payload = payload.get("virtual_xray", {})
	if virtual_xray_payload:
		_apply_virtual_xray_payload(virtual_xray, virtual_xray_payload)
	for source_entry in payload.get("source_nodes", []):
		target = _resolve_child_index_path(
			virtual_xray,
			list(source_entry.get("child_index_path", [])),
		)
		source_payload = source_entry.get("payload", {})
		if target is None or not isinstance(target, (Volumetric, Mesh)):
			continue
		_apply_source_payload(target, source_payload)


def build_virtual_xray_scene_xml(virtual_xray):
	"""Return one XML tree describing the current virtRTG scene subtree."""
	root = ET.Element(
		"virtRTGScene",
		{
			"format": SCENE_FORMAT_NAME,
			"version": str(SCENE_FORMAT_VERSION),
		},
	)
	nodes_elem = ET.SubElement(root, "nodes")
	node_to_id = {virtual_xray: "node-0000"}
	export_nodes = [virtual_xray, *_iter_export_nodes(virtual_xray)]
	for idx, obj in enumerate(export_nodes):
		node_id = node_to_id.setdefault(obj, f"node-{idx:04d}")
		parent = getattr(obj, "parent", None)
		parent_id = ""
		if parent in node_to_id:
			parent_id = node_to_id[parent]
		elem = ET.SubElement(
			nodes_elem,
			"node",
			{
				"id": node_id,
				"parent": parent_id,
				"class": obj.__class__.__name__,
				"label": str(getattr(obj, "label", obj.__class__.__name__)),
				"visible": "1" if bool(getattr(obj, "visible", True)) else "0",
			},
		)
		description = str(getattr(obj, "description", "") or "")
		if description:
			ET.SubElement(elem, "description").text = description
		if isinstance(obj, Transform):
			ET.SubElement(elem, "matrix_row_major").text = _matrix_to_text(obj.toNumPy())
		if obj is virtual_xray:
			payload_elem = ET.SubElement(elem, "virtRTG")
			payload_elem.set("encoding", "json")
			payload_elem.text = _json_text(_virtual_xray_payload(virtual_xray))
		elif isinstance(obj, (Volumetric, Mesh)):
			payload_elem = ET.SubElement(elem, "virtRTGSource")
			payload_elem.set("encoding", "json")
			payload_elem.text = _json_text(_source_config_payload(obj))
		else:
			payload = _annotation_payload(obj)
			if payload is not None:
				payload_elem = ET.SubElement(elem, "payload")
				payload_elem.set("encoding", "json")
				payload_elem.text = _json_text(payload)
	return ET.ElementTree(root)


def save_virtual_xray_scene(virtual_xray, path, parent_widget=None, interactive=False):
	"""Write one simplified virtRTG scene description to an XML file."""
	_validate_scene_source_links(virtual_xray, parent_widget=parent_widget, interactive=interactive)
	tree = build_virtual_xray_scene_xml(virtual_xray)
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tree.write(path, encoding="utf-8", xml_declaration=True)
	return path


def _text_to_floats(text, expected_count=None):
	"""Parse one whitespace-separated list of floats."""
	normalized = str(text).replace(",", " ")
	values = [float(v) for v in normalized.split()]
	if expected_count is not None and len(values) != expected_count:
		raise ValueError(f"Expected {expected_count} values, got {len(values)}")
	return values


def _node_payload_json(elem, child_name):
	"""Parse one JSON payload stored in a named child element."""
	child = elem.find(child_name)
	if child is None or not (child.text or "").strip():
		return None
	return json.loads(child.text)


def _apply_virtual_xray_payload(virtual_xray, payload):
	"""Apply one serialized virtual X-ray payload to an existing object."""
	geometry = payload.get("geometry", {})
	virtual_xray.projection_mode = str(geometry.get("projection_mode", virtual_xray.projection_mode))
	virtual_xray.detector_center_ref = np.asarray(geometry.get("detector_center_ref", virtual_xray.detector_center_ref), dtype=np.float32)
	virtual_xray.detector_normal_ref = np.asarray(geometry.get("detector_normal_ref", virtual_xray.detector_normal_ref), dtype=np.float32)
	virtual_xray.detector_up_ref = np.asarray(geometry.get("detector_up_ref", virtual_xray.detector_up_ref), dtype=np.float32)
	virtual_xray.detector_shape_hw = [int(v) for v in geometry.get("detector_shape_hw", virtual_xray.detector_shape_hw)]
	virtual_xray.detector_pixel_size_mm = [float(v) for v in geometry.get("detector_pixel_size_mm", virtual_xray.detector_pixel_size_mm)]
	virtual_xray.source_position_ref = np.asarray(geometry.get("source_position_ref", virtual_xray.source_position_ref), dtype=np.float32)
	virtual_xray.ray_direction_ref = np.asarray(geometry.get("ray_direction_ref", virtual_xray.ray_direction_ref), dtype=np.float32)
	virtual_xray.step_mm = float(geometry.get("step_mm", virtual_xray.step_mm))
	virtual_xray.quality_profile_name = str(geometry.get("quality_profile_name", virtual_xray.quality_profile_name))
	virtual_xray.depth_window_mode = str(geometry.get("depth_window_mode", getattr(virtual_xray, "depth_window_mode", "off")))
	virtual_xray.depth_window_mm = [float(v) for v in geometry.get("depth_window_mm", getattr(virtual_xray, "depth_window_mm", [0.0, 0.0]))]
	virtual_xray.depth_window_origin_ref = np.asarray(geometry.get("depth_window_origin_ref", getattr(virtual_xray, "depth_window_origin_ref", [0.0, 0.0, 0.0])), dtype=np.float32)
	virtual_xray.depth_window_axis_ref = np.asarray(geometry.get("depth_window_axis_ref", getattr(virtual_xray, "depth_window_axis_ref", [0.0, 0.0, 1.0])), dtype=np.float32)

	source_defaults = payload.get("source_defaults", {})
	virtual_xray.source_interpolation = str(source_defaults.get("interpolation", virtual_xray.source_interpolation))
	virtual_xray.source_fill_value = source_defaults.get("fill_value", virtual_xray.source_fill_value)
	virtual_xray.source_preprocess_mode = str(source_defaults.get("preprocess_mode", virtual_xray.source_preprocess_mode))
	virtual_xray.source_preprocess_low_percentile = float(source_defaults.get("preprocess_low_percentile", virtual_xray.source_preprocess_low_percentile))
	virtual_xray.source_preprocess_high_percentile = float(source_defaults.get("preprocess_high_percentile", virtual_xray.source_preprocess_high_percentile))
	virtual_xray.source_preprocess_output_low = float(source_defaults.get("preprocess_output_low", virtual_xray.source_preprocess_output_low))
	virtual_xray.source_preprocess_output_high = float(source_defaults.get("preprocess_output_high", virtual_xray.source_preprocess_output_high))
	virtual_xray.mesh_source_scalar_value = float(source_defaults.get("mesh_source_scalar_value", virtual_xray.mesh_source_scalar_value))
	virtual_xray.mesh_source_mode = str(source_defaults.get("mesh_source_mode", virtual_xray.mesh_source_mode))
	virtual_xray.mesh_surface_thickness_mm = float(source_defaults.get("mesh_surface_thickness_mm", virtual_xray.mesh_surface_thickness_mm))

	physics = payload.get("physics", {})
	virtual_xray.physics_mu_air = float(physics.get("mu_air", virtual_xray.physics_mu_air))
	virtual_xray.physics_mu_water = float(physics.get("mu_water", virtual_xray.physics_mu_water))
	virtual_xray.physics_hounsfield_air = float(physics.get("hounsfield_air", virtual_xray.physics_hounsfield_air))
	virtual_xray.physics_attenuation_scale = float(physics.get("attenuation_scale", virtual_xray.physics_attenuation_scale))
	virtual_xray.physics_source_energy_kev = float(physics.get("source_energy_kev", virtual_xray.physics_source_energy_kev))
	virtual_xray.physics_reference_energy_kev = float(physics.get("reference_energy_kev", virtual_xray.physics_reference_energy_kev))
	virtual_xray.physics_attenuation_energy_exponent = float(physics.get("attenuation_energy_exponent", virtual_xray.physics_attenuation_energy_exponent))
	virtual_xray.physics_output_mode = str(physics.get("output_mode", virtual_xray.physics_output_mode))
	virtual_xray.physics_intensity_floor = float(physics.get("intensity_floor", virtual_xray.physics_intensity_floor))
	virtual_xray.physics_source_distance_falloff_mode = str(physics.get("source_distance_falloff_mode", virtual_xray.physics_source_distance_falloff_mode))
	virtual_xray.physics_source_distance_reference_mm = physics.get("source_distance_reference_mm", virtual_xray.physics_source_distance_reference_mm)
	virtual_xray.physics_source_distance_power = float(physics.get("source_distance_power", virtual_xray.physics_source_distance_power))
	virtual_xray.physics_material_response_mode = str(physics.get("material_response_mode", virtual_xray.physics_material_response_mode))
	virtual_xray.physics_bone_threshold_hu = physics.get("bone_threshold_hu", virtual_xray.physics_bone_threshold_hu)
	virtual_xray.physics_bone_threshold_softness = float(physics.get("bone_threshold_softness", virtual_xray.physics_bone_threshold_softness))
	virtual_xray.physics_material_window_center = physics.get("material_window_center", virtual_xray.physics_material_window_center)
	virtual_xray.physics_material_window_width = physics.get("material_window_width", virtual_xray.physics_material_window_width)
	virtual_xray.physics_material_window_mode = str(physics.get("material_window_mode", virtual_xray.physics_material_window_mode))
	virtual_xray.physics_material_window_softness = float(physics.get("material_window_softness", virtual_xray.physics_material_window_softness))

	presentation = payload.get("presentation", {})
	virtual_xray.presentation_mode = str(presentation.get("mode", virtual_xray.presentation_mode))
	virtual_xray.presentation_invert = bool(presentation.get("invert", virtual_xray.presentation_invert))
	virtual_xray.presentation_gamma = float(presentation.get("gamma", virtual_xray.presentation_gamma))
	virtual_xray.presentation_contrast = float(presentation.get("contrast", virtual_xray.presentation_contrast))
	virtual_xray.presentation_robust_percentile = float(presentation.get("robust_percentile", virtual_xray.presentation_robust_percentile))
	virtual_xray.presentation_window_center = presentation.get("window_center", virtual_xray.presentation_window_center)
	virtual_xray.presentation_window_width = presentation.get("window_width", virtual_xray.presentation_window_width)
	virtual_xray.presentation_overlay_annotations = bool(presentation.get("overlay_annotations", getattr(virtual_xray, "presentation_overlay_annotations", False)))
	virtual_xray.presentation_overlay_labels = bool(presentation.get("overlay_labels", getattr(virtual_xray, "presentation_overlay_labels", False)))
	virtual_xray.presentation_overlay_cross_size_px = int(presentation.get("overlay_cross_size_px", getattr(virtual_xray, "presentation_overlay_cross_size_px", 6)))
	_apply_projection_cache_payload(virtual_xray, payload.get("projection_cache", None))


def _apply_source_payload(obj, payload):
	"""Apply one serialized source payload to a mesh or volumetric object."""
	ensure_xray_source_config(obj)
	obj.xray_source_enabled = bool(payload.get("enabled", True))
	obj.xray_scalar_scale = float(payload.get("scalar_scale", 1.0))
	obj.xray_scalar_bias = float(payload.get("scalar_bias", 0.0))
	obj.xray_attenuation_multiplier = float(payload.get("attenuation_multiplier", 1.0))
	set_xray_material_response_config(
		obj,
		XRayMaterialResponseConfig.from_mapping(payload.get("material_response_config", {})),
	)
	if isinstance(obj, Volumetric):
		obj.xray_interpolation_override = str(payload.get("interpolation_override", "default")).lower()
		obj.xray_fill_value_override_enabled = bool(payload.get("fill_value_override_enabled", False))
		obj.xray_fill_value_override = float(payload.get("fill_value_override", 0.0))
		obj.xray_volume_backend = str(payload.get("volume_backend", "sampling")).lower()
	else:
		obj.xray_mesh_backend = str(payload.get("mesh_backend", "analytic_bvh")).lower()
		obj.xray_mesh_mode = str(payload.get("mesh_mode", "solid")).lower()
		obj.xray_mesh_scalar_value = float(payload.get("mesh_scalar_value", 1800.0))
		obj.xray_mesh_shell_thickness_mm = float(payload.get("mesh_shell_thickness_mm", 1.0))


def _write_virtual_xray_text_sections(writer_cls, stream, virtual_xray, indent):
	"""Write editable ATMDL sections mirroring the high-level VirtualXRay payload."""
	ind1 = writer_cls._ind(indent)
	ind2 = writer_cls._ind(indent + 1)

	stream.write(f"{ind1}geometry {{\n")
	stream.write(f'{ind2}projectionMode "{virtual_xray.projection_mode}"\n')
	stream.write(f"{ind2}detectorCenter [{','.join(f'{float(v):.16g}' for v in virtual_xray.detector_center_ref)}]\n")
	stream.write(f"{ind2}detectorNormal [{','.join(f'{float(v):.16g}' for v in virtual_xray.detector_normal_ref)}]\n")
	stream.write(f"{ind2}detectorUp [{','.join(f'{float(v):.16g}' for v in virtual_xray.detector_up_ref)}]\n")
	stream.write(f"{ind2}detectorShape [{int(virtual_xray.detector_shape_hw[0])},{int(virtual_xray.detector_shape_hw[1])}]\n")
	stream.write(f"{ind2}detectorPixelSize [{float(virtual_xray.detector_pixel_size_mm[0]):.16g},{float(virtual_xray.detector_pixel_size_mm[1]):.16g}]\n")
	stream.write(f"{ind2}sourcePosition [{','.join(f'{float(v):.16g}' for v in virtual_xray.source_position_ref)}]\n")
	stream.write(f"{ind2}rayDirection [{','.join(f'{float(v):.16g}' for v in virtual_xray.ray_direction_ref)}]\n")
	stream.write(f"{ind2}stepMM {float(virtual_xray.step_mm):.16g}\n")
	stream.write(f'{ind2}qualityProfile "{virtual_xray.quality_profile_name}"\n')
	stream.write(f'{ind2}depthWindowMode "{getattr(virtual_xray, "depth_window_mode", "off")}"\n')
	stream.write(f"{ind2}depthWindowMM [{float(getattr(virtual_xray, 'depth_window_mm', [0.0, 0.0])[0]):.16g},{float(getattr(virtual_xray, 'depth_window_mm', [0.0, 0.0])[1]):.16g}]\n")
	stream.write(f"{ind2}depthWindowOrigin [{','.join(f'{float(v):.16g}' for v in getattr(virtual_xray, 'depth_window_origin_ref', [0.0, 0.0, 0.0]))}]\n")
	stream.write(f"{ind2}depthWindowAxis [{','.join(f'{float(v):.16g}' for v in getattr(virtual_xray, 'depth_window_axis_ref', [0.0, 0.0, 1.0]))}]\n")
	stream.write(f"{ind1}}}\n")

	stream.write(f"{ind1}sourceDefaults {{\n")
	stream.write(f'{ind2}interpolation "{virtual_xray.source_interpolation}"\n')
	stream.write(f"{ind2}fillValue {_atmdl_optional_scalar_text(virtual_xray.source_fill_value)}\n")
	stream.write(f'{ind2}preprocessMode "{virtual_xray.source_preprocess_mode}"\n')
	stream.write(f"{ind2}preprocessLowPercentile {float(virtual_xray.source_preprocess_low_percentile):.16g}\n")
	stream.write(f"{ind2}preprocessHighPercentile {float(virtual_xray.source_preprocess_high_percentile):.16g}\n")
	stream.write(f"{ind2}preprocessOutputLow {float(virtual_xray.source_preprocess_output_low):.16g}\n")
	stream.write(f"{ind2}preprocessOutputHigh {float(virtual_xray.source_preprocess_output_high):.16g}\n")
	stream.write(f"{ind2}meshSourceScalarValue {float(virtual_xray.mesh_source_scalar_value):.16g}\n")
	stream.write(f'{ind2}meshSourceMode "{virtual_xray.mesh_source_mode}"\n')
	stream.write(f"{ind2}meshSurfaceThicknessMM {float(virtual_xray.mesh_surface_thickness_mm):.16g}\n")
	stream.write(f"{ind1}}}\n")

	stream.write(f"{ind1}physics {{\n")
	stream.write(f"{ind2}muAir {float(virtual_xray.physics_mu_air):.16g}\n")
	stream.write(f"{ind2}muWater {float(virtual_xray.physics_mu_water):.16g}\n")
	stream.write(f"{ind2}hounsfieldAir {float(virtual_xray.physics_hounsfield_air):.16g}\n")
	stream.write(f"{ind2}attenuationScale {float(virtual_xray.physics_attenuation_scale):.16g}\n")
	stream.write(f"{ind2}sourceEnergyKeV {float(virtual_xray.physics_source_energy_kev):.16g}\n")
	stream.write(f"{ind2}referenceEnergyKeV {float(virtual_xray.physics_reference_energy_kev):.16g}\n")
	stream.write(f"{ind2}attenuationEnergyExponent {float(virtual_xray.physics_attenuation_energy_exponent):.16g}\n")
	stream.write(f'{ind2}outputMode "{virtual_xray.physics_output_mode}"\n')
	stream.write(f"{ind2}intensityFloor {float(virtual_xray.physics_intensity_floor):.16g}\n")
	stream.write(f'{ind2}sourceDistanceFalloffMode "{virtual_xray.physics_source_distance_falloff_mode}"\n')
	stream.write(f"{ind2}sourceDistanceReferenceMM {_atmdl_optional_scalar_text(virtual_xray.physics_source_distance_reference_mm)}\n")
	stream.write(f"{ind2}sourceDistancePower {float(virtual_xray.physics_source_distance_power):.16g}\n")
	stream.write(f'{ind2}materialResponseMode "{virtual_xray.physics_material_response_mode}"\n')
	stream.write(f"{ind2}boneThresholdHU {_atmdl_optional_scalar_text(virtual_xray.physics_bone_threshold_hu)}\n")
	stream.write(f"{ind2}boneThresholdSoftness {float(virtual_xray.physics_bone_threshold_softness):.16g}\n")
	stream.write(f"{ind2}materialWindowCenter {_atmdl_optional_scalar_text(virtual_xray.physics_material_window_center)}\n")
	stream.write(f"{ind2}materialWindowWidth {_atmdl_optional_scalar_text(virtual_xray.physics_material_window_width)}\n")
	stream.write(f'{ind2}materialWindowMode "{virtual_xray.physics_material_window_mode}"\n')
	stream.write(f"{ind2}materialWindowSoftness {float(virtual_xray.physics_material_window_softness):.16g}\n")
	stream.write(f"{ind1}}}\n")

	stream.write(f"{ind1}presentation {{\n")
	stream.write(f'{ind2}mode "{virtual_xray.presentation_mode}"\n')
	stream.write(f"{ind2}invert {_atmdl_bool_text(virtual_xray.presentation_invert)}\n")
	stream.write(f"{ind2}gamma {float(virtual_xray.presentation_gamma):.16g}\n")
	stream.write(f"{ind2}contrast {float(virtual_xray.presentation_contrast):.16g}\n")
	stream.write(f"{ind2}robustPercentile {float(virtual_xray.presentation_robust_percentile):.16g}\n")
	stream.write(f"{ind2}windowCenter {_atmdl_optional_scalar_text(virtual_xray.presentation_window_center)}\n")
	stream.write(f"{ind2}windowWidth {_atmdl_optional_scalar_text(virtual_xray.presentation_window_width)}\n")
	stream.write(f"{ind2}overlayAnnotations {_atmdl_bool_text(getattr(virtual_xray, 'presentation_overlay_annotations', False))}\n")
	stream.write(f"{ind2}overlayLabels {_atmdl_bool_text(getattr(virtual_xray, 'presentation_overlay_labels', False))}\n")
	stream.write(f"{ind2}overlayCrossSizePx {int(getattr(virtual_xray, 'presentation_overlay_cross_size_px', 6))}\n")
	stream.write(f"{ind1}}}\n")


def _write_source_node_sections(writer_cls, stream, virtual_xray, indent):
	"""Write editable per-source ATMDL sections keyed by child-index path."""
	ind1 = writer_cls._ind(indent)
	ind2 = writer_cls._ind(indent + 1)
	ind3 = writer_cls._ind(indent + 2)
	for obj in _iter_export_child_objects(virtual_xray):
		if not isinstance(obj, (Volumetric, Mesh)):
			continue
		index_path = _child_index_path(virtual_xray, obj)
		if index_path is None:
			continue
		cfg = _source_config_payload(obj)
		stream.write(f"{ind1}sourceNode {{\n")
		stream.write(f"{ind2}childIndexPath [{','.join(str(int(v)) for v in index_path)}]\n")
		stream.write(f'{ind2}sourceType "{cfg.get("source_type", "")}"\n')
		stream.write(f"{ind2}enabled {_atmdl_bool_text(cfg.get('enabled', True))}\n")
		stream.write(f"{ind2}scalarScale {float(cfg.get('scalar_scale', 1.0)):.16g}\n")
		stream.write(f"{ind2}scalarBias {float(cfg.get('scalar_bias', 0.0)):.16g}\n")
		stream.write(f"{ind2}attenuationMultiplier {float(cfg.get('attenuation_multiplier', 1.0)):.16g}\n")
		stream.write(f"{ind2}materialResponse {{\n")
		mrc = cfg.get("material_response_config", {})
		stream.write(f"{ind3}enabled {_atmdl_bool_text(mrc.get('enabled', False))}\n")
		stream.write(f'{ind3}mode "{mrc.get("mode", "linear")}"\n')
		stream.write(f"{ind3}boneThresholdHU {_atmdl_optional_scalar_text(mrc.get('bone_threshold_hu', None))}\n")
		stream.write(f"{ind3}boneThresholdSoftness {_atmdl_optional_scalar_text(mrc.get('bone_threshold_softness', None))}\n")
		stream.write(f"{ind3}windowCenter {_atmdl_optional_scalar_text(mrc.get('window_center', None))}\n")
		stream.write(f"{ind3}windowWidth {_atmdl_optional_scalar_text(mrc.get('window_width', None))}\n")
		stream.write(f'{ind3}windowMode "{mrc.get("window_mode", "hard")}"\n')
		stream.write(f"{ind3}windowSoftness {_atmdl_optional_scalar_text(mrc.get('window_softness', None))}\n")
		stream.write(f"{ind2}}}\n")
		if cfg.get("source_type") == "volumetric":
			stream.write(f'{ind2}interpolationOverride "{cfg.get("interpolation_override", "default")}"\n')
			stream.write(f"{ind2}fillValueOverrideEnabled {_atmdl_bool_text(cfg.get('fill_value_override_enabled', False))}\n")
			stream.write(f"{ind2}fillValueOverride {float(cfg.get('fill_value_override', 0.0)):.16g}\n")
			stream.write(f'{ind2}volumeBackend "{cfg.get("volume_backend", "sampling")}"\n')
		else:
			stream.write(f'{ind2}meshBackend "{cfg.get("mesh_backend", "analytic_bvh")}"\n')
			stream.write(f'{ind2}meshMode "{cfg.get("mesh_mode", "solid")}"\n')
			stream.write(f"{ind2}meshScalarValue {float(cfg.get('mesh_scalar_value', 1800.0)):.16g}\n")
			stream.write(f"{ind2}meshShellThicknessMM {float(cfg.get('mesh_shell_thickness_mm', 1.0)):.16g}\n")
		stream.write(f"{ind1}}}\n")


def _instantiate_source_from_payload(payload):
	"""Create one mesh or volumetric object from source links when possible."""
	source_type = str(payload.get("source_type", "")).lower()
	source_links = payload.get("source_links", {}) or {}
	if source_type == "volumetric":
		files = source_links.get("files", [])
		if files:
			first_existing = next((path for path in files if Path(path).is_file()), None)
			if first_existing is not None:
				obj = Parser.load(first_existing)
				if obj is not None:
					Parser.set_source_reference(obj, {"kind": "dicom_series", "files": files})
					return obj
			return _make_missing_volume_placeholder(files)
		return _make_missing_volume_placeholder([])
	if source_type == "mesh":
		path = source_links.get("path", None)
		if isinstance(path, str) and Path(path).is_file():
			obj = Parser.load(path)
			if obj is not None:
				Parser.set_source_reference(obj, {"kind": "file", "path": path})
				return obj
		obj = Mesh()
		if isinstance(path, str) and path.strip():
			Parser.set_source_reference(obj, {"kind": "file", "path": path})
			obj.visible = False
			obj._virt_rtg_missing_source_placeholder = True
		return obj
	return None


def _parse_section_mapping(parser, stream):
	"""Parse one generic ATMDL subsection into a flat mapping."""
	slowo = parser.skip_comments(stream)
	if slowo is None:
		return None
	if slowo != "{":
		print("BLAD. Oczekiwano znaku { odczytano: " + str(slowo))
		return None

	result = {}
	matrix_keys = {
		"detectorCenter", "detectorNormal", "detectorUp", "detectorShape",
		"detectorPixelSize", "sourcePosition", "rayDirection", "depthWindowMM",
		"depthWindowOrigin", "depthWindowAxis", "childIndexPath",
	}
	string_keys = {
		"stepMM", "qualityProfile", "projectionMode", "depthWindowMode",
		"interpolation", "preprocessMode", "meshSourceMode", "outputMode",
		"sourceDistanceFalloffMode", "materialResponseMode", "materialWindowMode",
		"mode", "sourceType", "interpolationOverride", "volumeBackend",
		"meshBackend", "meshMode", "windowMode",
	}
	bool_keys = {
		"invert", "overlayAnnotations", "overlayLabels", "enabled",
		"fillValueOverrideEnabled",
	}
	scalar_keys = {
		"fillValue", "preprocessLowPercentile", "preprocessHighPercentile",
		"preprocessOutputLow", "preprocessOutputHigh", "meshSourceScalarValue",
		"meshSurfaceThicknessMM", "muAir", "muWater", "hounsfieldAir",
		"attenuationScale", "sourceEnergyKeV", "referenceEnergyKeV",
		"attenuationEnergyExponent", "intensityFloor", "sourceDistanceReferenceMM",
		"sourceDistancePower", "boneThresholdHU", "boneThresholdSoftness",
		"materialWindowCenter", "materialWindowWidth", "materialWindowSoftness",
		"gamma", "contrast", "robustPercentile", "windowCenter", "windowWidth",
		"overlayCrossSizePx", "scalarScale", "scalarBias", "attenuationMultiplier",
		"fillValueOverride", "meshScalarValue", "meshShellThicknessMM",
		"windowSoftness",
	}
	while slowo != "}":
		slowo = parser.skip_comments(stream)
		if slowo is None:
			return None
		if slowo == "}":
			break
		if slowo == "materialResponse":
			result["material_response"] = _parse_section_mapping(parser, stream)
			continue
		if slowo in matrix_keys:
			tekst = parser.parseType_matrix(stream)
			if tekst is not None:
				result[slowo] = tekst
			continue
		if slowo in string_keys:
			result[slowo] = parser.parseType_string(stream)
			continue
		if slowo in bool_keys:
			value, _ = parser.readWord(stream)
			result[slowo] = value
			continue
		if slowo in scalar_keys:
			value, _ = parser.readWord(stream)
			result[slowo] = value
			continue
		tekst = parser.parseType_matrix(stream)
		if tekst is not None:
			result[slowo] = tekst
		else:
			value, _ = parser.readWord(stream)
			result[slowo] = value

	return result


def _apply_virtual_xray_text_sections(virtual_xray, sections):
	"""Apply human-editable ATMDL sections to one VirtualXRay object."""
	geometry = sections.get("geometry", {})
	if geometry:
		if geometry.get("projectionMode") is not None:
			virtual_xray.projection_mode = str(geometry["projectionMode"])
		if geometry.get("detectorCenter") is not None:
			virtual_xray.detector_center_ref = np.asarray(_text_to_floats(geometry["detectorCenter"], 3), dtype=np.float32)
		if geometry.get("detectorNormal") is not None:
			virtual_xray.detector_normal_ref = np.asarray(_text_to_floats(geometry["detectorNormal"], 3), dtype=np.float32)
		if geometry.get("detectorUp") is not None:
			virtual_xray.detector_up_ref = np.asarray(_text_to_floats(geometry["detectorUp"], 3), dtype=np.float32)
		if geometry.get("detectorShape") is not None:
			virtual_xray.detector_shape_hw = [int(v) for v in _text_to_floats(geometry["detectorShape"], 2)]
		if geometry.get("detectorPixelSize") is not None:
			virtual_xray.detector_pixel_size_mm = _text_to_floats(geometry["detectorPixelSize"], 2)
		if geometry.get("sourcePosition") is not None:
			virtual_xray.source_position_ref = np.asarray(_text_to_floats(geometry["sourcePosition"], 3), dtype=np.float32)
		if geometry.get("rayDirection") is not None:
			virtual_xray.ray_direction_ref = np.asarray(_text_to_floats(geometry["rayDirection"], 3), dtype=np.float32)
		virtual_xray.step_mm = float(geometry.get("stepMM", virtual_xray.step_mm))
		if geometry.get("qualityProfile") is not None:
			virtual_xray.quality_profile_name = str(geometry["qualityProfile"])
		if geometry.get("depthWindowMode") is not None:
			virtual_xray.depth_window_mode = str(geometry["depthWindowMode"])
		if geometry.get("depthWindowMM") is not None:
			virtual_xray.depth_window_mm = _text_to_floats(geometry["depthWindowMM"], 2)
		if geometry.get("depthWindowOrigin") is not None:
			virtual_xray.depth_window_origin_ref = np.asarray(_text_to_floats(geometry["depthWindowOrigin"], 3), dtype=np.float32)
		if geometry.get("depthWindowAxis") is not None:
			virtual_xray.depth_window_axis_ref = np.asarray(_text_to_floats(geometry["depthWindowAxis"], 3), dtype=np.float32)

	source_defaults = sections.get("sourceDefaults", {})
	if source_defaults:
		if source_defaults.get("interpolation") is not None:
			virtual_xray.source_interpolation = str(source_defaults["interpolation"])
		virtual_xray.source_fill_value = _atmdl_read_optional_float(source_defaults.get("fillValue"), virtual_xray.source_fill_value)
		if source_defaults.get("preprocessMode") is not None:
			virtual_xray.source_preprocess_mode = str(source_defaults["preprocessMode"])
		virtual_xray.source_preprocess_low_percentile = float(source_defaults.get("preprocessLowPercentile", virtual_xray.source_preprocess_low_percentile))
		virtual_xray.source_preprocess_high_percentile = float(source_defaults.get("preprocessHighPercentile", virtual_xray.source_preprocess_high_percentile))
		virtual_xray.source_preprocess_output_low = float(source_defaults.get("preprocessOutputLow", virtual_xray.source_preprocess_output_low))
		virtual_xray.source_preprocess_output_high = float(source_defaults.get("preprocessOutputHigh", virtual_xray.source_preprocess_output_high))
		virtual_xray.mesh_source_scalar_value = float(source_defaults.get("meshSourceScalarValue", virtual_xray.mesh_source_scalar_value))
		if source_defaults.get("meshSourceMode") is not None:
			virtual_xray.mesh_source_mode = str(source_defaults["meshSourceMode"])
		virtual_xray.mesh_surface_thickness_mm = float(source_defaults.get("meshSurfaceThicknessMM", virtual_xray.mesh_surface_thickness_mm))

	physics = sections.get("physics", {})
	if physics:
		virtual_xray.physics_mu_air = float(physics.get("muAir", virtual_xray.physics_mu_air))
		virtual_xray.physics_mu_water = float(physics.get("muWater", virtual_xray.physics_mu_water))
		virtual_xray.physics_hounsfield_air = float(physics.get("hounsfieldAir", virtual_xray.physics_hounsfield_air))
		virtual_xray.physics_attenuation_scale = float(physics.get("attenuationScale", virtual_xray.physics_attenuation_scale))
		virtual_xray.physics_source_energy_kev = float(physics.get("sourceEnergyKeV", virtual_xray.physics_source_energy_kev))
		virtual_xray.physics_reference_energy_kev = float(physics.get("referenceEnergyKeV", virtual_xray.physics_reference_energy_kev))
		virtual_xray.physics_attenuation_energy_exponent = float(physics.get("attenuationEnergyExponent", virtual_xray.physics_attenuation_energy_exponent))
		if physics.get("outputMode") is not None:
			virtual_xray.physics_output_mode = str(physics["outputMode"])
		virtual_xray.physics_intensity_floor = float(physics.get("intensityFloor", virtual_xray.physics_intensity_floor))
		if physics.get("sourceDistanceFalloffMode") is not None:
			virtual_xray.physics_source_distance_falloff_mode = str(physics["sourceDistanceFalloffMode"])
		virtual_xray.physics_source_distance_reference_mm = _atmdl_read_optional_float(physics.get("sourceDistanceReferenceMM"), virtual_xray.physics_source_distance_reference_mm)
		virtual_xray.physics_source_distance_power = float(physics.get("sourceDistancePower", virtual_xray.physics_source_distance_power))
		if physics.get("materialResponseMode") is not None:
			virtual_xray.physics_material_response_mode = str(physics["materialResponseMode"])
		virtual_xray.physics_bone_threshold_hu = _atmdl_read_optional_float(physics.get("boneThresholdHU"), virtual_xray.physics_bone_threshold_hu)
		virtual_xray.physics_bone_threshold_softness = float(physics.get("boneThresholdSoftness", virtual_xray.physics_bone_threshold_softness))
		virtual_xray.physics_material_window_center = _atmdl_read_optional_float(physics.get("materialWindowCenter"), virtual_xray.physics_material_window_center)
		virtual_xray.physics_material_window_width = _atmdl_read_optional_float(physics.get("materialWindowWidth"), virtual_xray.physics_material_window_width)
		if physics.get("materialWindowMode") is not None:
			virtual_xray.physics_material_window_mode = str(physics["materialWindowMode"])
		virtual_xray.physics_material_window_softness = float(physics.get("materialWindowSoftness", virtual_xray.physics_material_window_softness))

	presentation = sections.get("presentation", {})
	if presentation:
		if presentation.get("mode") is not None:
			virtual_xray.presentation_mode = str(presentation["mode"])
		virtual_xray.presentation_invert = _atmdl_read_bool(presentation.get("invert"), virtual_xray.presentation_invert)
		virtual_xray.presentation_gamma = float(presentation.get("gamma", virtual_xray.presentation_gamma))
		virtual_xray.presentation_contrast = float(presentation.get("contrast", virtual_xray.presentation_contrast))
		virtual_xray.presentation_robust_percentile = float(presentation.get("robustPercentile", virtual_xray.presentation_robust_percentile))
		virtual_xray.presentation_window_center = _atmdl_read_optional_float(presentation.get("windowCenter"), virtual_xray.presentation_window_center)
		virtual_xray.presentation_window_width = _atmdl_read_optional_float(presentation.get("windowWidth"), virtual_xray.presentation_window_width)
		virtual_xray.presentation_overlay_annotations = _atmdl_read_bool(presentation.get("overlayAnnotations"), getattr(virtual_xray, "presentation_overlay_annotations", False))
		virtual_xray.presentation_overlay_labels = _atmdl_read_bool(presentation.get("overlayLabels"), getattr(virtual_xray, "presentation_overlay_labels", False))
		virtual_xray.presentation_overlay_cross_size_px = int(float(presentation.get("overlayCrossSizePx", getattr(virtual_xray, "presentation_overlay_cross_size_px", 6))))

	for source_section in sections.get("sourceNodes", []):
		path_text = source_section.get("childIndexPath", None)
		if path_text is None:
			continue
		target = _resolve_child_index_path(virtual_xray, [int(v) for v in _text_to_floats(path_text)])
		if target is None or not isinstance(target, (Volumetric, Mesh)):
			continue
		source_payload = {
			"enabled": _atmdl_read_bool(source_section.get("enabled"), True),
			"scalar_scale": float(source_section.get("scalarScale", 1.0)),
			"scalar_bias": float(source_section.get("scalarBias", 0.0)),
			"attenuation_multiplier": float(source_section.get("attenuationMultiplier", 1.0)),
			"source_type": str(source_section.get("sourceType", "volumetric" if isinstance(target, Volumetric) else "mesh")).lower(),
			"material_response_config": {
				"enabled": False,
				"mode": "linear",
			},
		}
		material_response = source_section.get("material_response", {})
		if material_response:
			source_payload["material_response_config"] = {
				"enabled": _atmdl_read_bool(material_response.get("enabled"), False),
				"mode": str(material_response.get("mode", "linear")),
				"bone_threshold_hu": _atmdl_read_optional_float(material_response.get("boneThresholdHU"), None),
				"bone_threshold_softness": _atmdl_read_optional_float(material_response.get("boneThresholdSoftness"), None),
				"window_center": _atmdl_read_optional_float(material_response.get("windowCenter"), None),
				"window_width": _atmdl_read_optional_float(material_response.get("windowWidth"), None),
				"window_mode": str(material_response.get("windowMode", "hard")),
				"window_softness": _atmdl_read_optional_float(material_response.get("windowSoftness"), None),
			}
		if isinstance(target, Volumetric):
			source_payload["interpolation_override"] = str(source_section.get("interpolationOverride", "default"))
			source_payload["fill_value_override_enabled"] = _atmdl_read_bool(source_section.get("fillValueOverrideEnabled"), False)
			source_payload["fill_value_override"] = float(source_section.get("fillValueOverride", 0.0))
			source_payload["volume_backend"] = str(source_section.get("volumeBackend", "sampling"))
		else:
			source_payload["mesh_backend"] = str(source_section.get("meshBackend", "analytic_bvh"))
			source_payload["mesh_mode"] = str(source_section.get("meshMode", "solid"))
			source_payload["mesh_scalar_value"] = float(source_section.get("meshScalarValue", 1800.0))
			source_payload["mesh_shell_thickness_mm"] = float(source_section.get("meshShellThicknessMM", 1.0))
		_apply_source_payload(target, source_payload)


def _instantiate_node_from_elem(elem, virtual_xray_root):
	"""Create one scene object matching the serialized node type."""
	class_name = elem.get("class", "")
	if class_name == "VirtualXRay":
		return virtual_xray_root
	if class_name == "Transform":
		obj = Transform()
		matrix_text = elem.findtext("matrix_row_major", "")
		if matrix_text.strip():
			obj.fromNumPy(np.asarray(_text_to_floats(matrix_text, expected_count=16), dtype=np.float64).reshape((4, 4)))
		return obj
	if class_name == "Volumetric":
		return _instantiate_source_from_payload(_node_payload_json(elem, "virtRTGSource") or {})
	if class_name == "Mesh":
		return _instantiate_source_from_payload(_node_payload_json(elem, "virtRTGSource") or {})
	if class_name == "AnnotationPoint":
		payload = _node_payload_json(elem, "payload") or {}
		obj = AnnotationPoint(point=payload.get("point", [0.0, 0.0, 0.0]), vector=payload.get("vector", None))
		obj.m_showVector = bool(payload.get("show_vector", obj.m_showVector))
		return obj
	if class_name == "AnnotationPath":
		payload = _node_payload_json(elem, "payload") or {}
		obj = AnnotationPath(points=payload.get("points", []))
		obj.m_width = float(payload.get("width", obj.m_width))
		return obj
	return None


def load_virtual_xray_scene(path, target_virtual_xray=None):
	"""Load one simplified virtRTG scene description into a VirtualXRay object."""
	tree = ET.parse(path)
	root = tree.getroot()
	if root.tag != "virtRTGScene":
		raise ValueError("Unsupported virtRTG scene root element.")
	virtual_xray = target_virtual_xray if target_virtual_xray is not None else VirtualXRay()
	nodes_elem = root.find("nodes")
	if nodes_elem is None:
		return virtual_xray
	node_objects = {}
	pending = []
	for elem in nodes_elem.findall("node"):
		obj = _instantiate_node_from_elem(elem, virtual_xray)
		if obj is None:
			continue
		obj.label = elem.get("label", getattr(obj, "label", obj.__class__.__name__))
		obj.description = elem.findtext("description", getattr(obj, "description", ""))
		if getattr(obj, "_virt_rtg_missing_source_placeholder", False):
			obj.visible = False
		else:
			obj.visible = elem.get("visible", "1") != "0"
		node_id = elem.get("id", "")
		node_objects[node_id] = obj
		parent_id = elem.get("parent", "")
		pending.append((obj, parent_id, elem))
	for obj, parent_id, elem in pending:
		if obj is virtual_xray:
			payload = _node_payload_json(elem, "virtRTG")
			if payload is not None:
				_apply_virtual_xray_payload(virtual_xray, payload)
			continue
		parent_obj = node_objects.get(parent_id, virtual_xray)
		parent_obj.addChild(obj)
		source_payload = _node_payload_json(elem, "virtRTGSource")
		if source_payload is not None and isinstance(obj, (Volumetric, Mesh)):
			_apply_source_payload(obj, source_payload)
	return virtual_xray


def _write_virtual_xray_atmdl(writer_cls, stream, obj, indent, context):
	"""Serialize one VirtualXRay subtree into an ATMDL plugin block."""
	from dpVision.parsers.parserATMDL import _atmdl_escape_string

	ind = writer_cls._ind(indent)
	stream.write(f"{ind}virtualXRay {{\n")
	writer_cls._write_common_props(stream, obj, indent + 1)
	if context.get("virt_rtg_write_base64", True):
		payload64 = _encode_payload64(_virtual_xray_atmdl_payload(obj))
		stream.write(f'{writer_cls._ind(indent + 1)}virtRTGConfig64 "{_atmdl_escape_string(payload64)}"\n')
	if context.get("virt_rtg_write_text_sections", True):
		_write_virtual_xray_text_sections(writer_cls, stream, obj, indent + 1)
		_write_source_node_sections(writer_cls, stream, obj, indent + 1)
	writer_cls._write_children(stream, obj, indent + 1, context)
	stream.write(f"{ind}}}\n")
	return True


def _parse_virtual_xray_atmdl(parser, stream, token):
	"""Parse one plugin-defined VirtualXRay ATMDL block with standard child objects."""
	slowo = parser.skip_comments(stream)

	if slowo is None:
		print("Osiagnieto koniec pliku podczas parsowania obiektu 'virtualXRay'")
		return None

	if slowo != "{":
		print("BLAD. Oczekiwano znaku { odczytano: " + slowo)
		return None

	opis = {}
	kids = []
	sections = {
		"sourceNodes": [],
	}

	while slowo != "}":
		slowo = parser.skip_comments(stream)

		if slowo is None:
			print("Osiagnieto koniec pliku podczas parsowania obiektu 'virtualXRay'")
			return None
		elif slowo == "}":
			print("Znaleziono klamre zamykajaca obiekt 'virtualXRay'")
		elif slowo in {"label", "descr", "virtRTGConfig64"}:
			tekst = parser.parseType_string(stream)
			if tekst:
				opis[slowo] = tekst
		elif slowo in {"geometry", "sourceDefaults", "physics", "presentation"}:
			parsed = _parse_section_mapping(parser, stream)
			if parsed is None:
				return None
			sections[slowo] = parsed
		elif slowo == "sourceNode":
			parsed = _parse_section_mapping(parser, stream)
			if parsed is None:
				return None
			sections["sourceNodes"].append(parsed)
		else:
			tmp = parser.parseObject(stream, slowo)
			if tmp is not None:
				kids.append(tmp)
			else:
				print("BLAD. Nierozpoznany symbol " + slowo)
				return None

	obj = VirtualXRay()
	if "label" in opis:
		obj.label = opis["label"]
	if "descr" in opis:
		obj.description = opis["descr"]
	for kid in kids:
		parser.add_kid(obj, kid)
	if "virtRTGConfig64" in opis:
		_apply_virtual_xray_atmdl_payload(obj, _decode_payload64(opis["virtRTGConfig64"]))
	_apply_virtual_xray_text_sections(obj, sections)
	return obj


def register_atmdl_integration():
	"""Register ATMDL read/write hooks for the VirtualXRay plugin object."""
	global _ATMDL_INTEGRATION_REGISTERED
	if _ATMDL_INTEGRATION_REGISTERED:
		return
	from dpVision.parsers.parserATMDL import ParserATMDL, WriterATMDL

	WriterATMDL.register_writer(
		lambda obj: isinstance(obj, VirtualXRay),
		_write_virtual_xray_atmdl,
	)
	ParserATMDL.register_object_reader(ATMDL_VIRTUAL_XRAY_TOKENS, _parse_virtual_xray_atmdl)
	_ATMDL_INTEGRATION_REGISTERED = True


def unregister_atmdl_integration():
	"""Remove previously registered ATMDL read/write hooks for VirtualXRay."""
	global _ATMDL_INTEGRATION_REGISTERED
	if not _ATMDL_INTEGRATION_REGISTERED:
		return
	from dpVision.parsers.parserATMDL import ParserATMDL, WriterATMDL

	WriterATMDL.unregister_writer(_write_virtual_xray_atmdl)
	ParserATMDL.unregister_object_reader(_parse_virtual_xray_atmdl)
	_ATMDL_INTEGRATION_REGISTERED = False
