# -*- coding: utf-8 -*-
"""Unit tests for the simplified virtRTG scene export format."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from dpVision import Transform, Volumetric
from dpVision.parser import Parser

from plugins.virtRTG.sceneFormat import build_virtual_xray_scene_xml, load_virtual_xray_scene, save_virtual_xray_scene
from plugins.virtRTG.virtualXRay import VirtualXRay
from plugins.virtRTG.xray.xraySource import (
	XRayMaterialResponseConfig,
	get_xray_material_response_config,
	set_xray_material_response_config,
)


def test_scene_export_uses_atmdl_like_nodes_with_json_payloads():
	"""Export one minimal scene subtree into XML nodes plus JSON plugin payloads."""
	virtual_xray = VirtualXRay()
	virtual_xray.label = "vxray-root"

	transform = Transform()
	transform.label = "jaw-transform"
	virtual_xray.addChild(transform)

	volume = Volumetric()
	volume.label = "ct-volume"
	volume.m_dicom_files = [r"d:\data\series\0001.dcm", r"d:\data\series\0002.dcm"]
	set_xray_material_response_config(
		volume,
		XRayMaterialResponseConfig(
			enabled=True,
			mode="bone_threshold",
			bone_threshold_hu=550.0,
			bone_threshold_softness=40.0,
			window_center=300.0,
			window_width=1200.0,
			window_mode="linear",
			window_softness=35.0,
		),
	)
	transform.addChild(volume)

	tree = build_virtual_xray_scene_xml(virtual_xray)
	root = tree.getroot()

	assert root.tag == "virtRTGScene"
	assert root.get("format") == "virtRTG-scene"

	nodes = root.find("nodes")
	assert nodes is not None
	node_elems = nodes.findall("node")
	assert len(node_elems) == 3

	root_node = node_elems[0]
	assert root_node.get("class") == "VirtualXRay"
	root_payload = json.loads(root_node.findtext("virtRTG"))
	assert root_payload["schema"] == "virtRTG-virtual-xray"
	assert root_payload["geometry"]["projection_mode"] == "cone"

	transform_node = node_elems[1]
	assert transform_node.get("class") == "Transform"
	assert transform_node.findtext("matrix_row_major") is not None

	volume_node = node_elems[2]
	assert volume_node.get("class") == "Volumetric"
	assert volume_node.get("parent") == transform_node.get("id")
	source_payload = json.loads(volume_node.findtext("virtRTGSource"))
	assert source_payload["schema"] == "virtRTG-source-config"
	assert source_payload["source_type"] == "volumetric"
	assert source_payload["material_response_config"]["mode"] == "bone_threshold"
	assert source_payload["source_links"]["kind"] == "dicom_series"
	assert len(source_payload["source_links"]["files"]) == 2


def test_scene_export_can_be_loaded_back_into_virtual_xray(tmp_path):
	"""Round-trip one simplified scene export into a fresh VirtualXRay object."""
	virtual_xray = VirtualXRay()
	virtual_xray.label = "vxray-roundtrip"
	virtual_xray.presentation_mode = "film"
	virtual_xray.physics_source_energy_kev = 85.0

	transform = Transform()
	transform.label = "source-transform"
	virtual_xray.addChild(transform)

	volume = Volumetric()
	volume.label = "imported-volume"
	volume.m_dicom_files = [str(tmp_path / "slice_0001.dcm")]
	set_xray_material_response_config(
		volume,
		XRayMaterialResponseConfig(
			enabled=True,
			mode="piecewise_soft_tissue",
			window_center=200.0,
			window_width=600.0,
			window_mode="sigmoid",
			window_softness=15.0,
		),
	)
	transform.addChild(volume)

	scene_path = tmp_path / "scene.vxrscene.xml"
	save_virtual_xray_scene(virtual_xray, scene_path)

	imported = load_virtual_xray_scene(scene_path)

	assert isinstance(imported, VirtualXRay)
	assert imported.presentation_mode == "film"
	assert imported.physics_source_energy_kev == 85.0
	assert len(imported.children()) == 1
	imported_transform = imported.children()[0]
	assert isinstance(imported_transform, Transform)
	assert len(imported_transform.children()) == 1
	imported_volume = imported_transform.children()[0]
	assert isinstance(imported_volume, Volumetric)
	imported_config = get_xray_material_response_config(imported_volume)
	assert imported_config.enabled is True
	assert imported_config.mode == "piecewise_soft_tissue"
	assert imported_config.window_width == 600.0


def test_scene_export_prefers_last_saved_source_path_over_original_dicom_series(tmp_path):
	"""Export should use the last saved file path when an object was saved after loading."""
	virtual_xray = VirtualXRay()
	volume = Volumetric()
	volume.label = "resampled-volume"
	volume.m_dicom_files = [r"d:\data\series\0001.dcm", r"d:\data\series\0002.dcm"]
	Parser.mark_saved_to_path(volume, tmp_path / "resampled_volume.nrrd")
	virtual_xray.addChild(volume)

	tree = build_virtual_xray_scene_xml(virtual_xray)
	source_payload = json.loads(tree.getroot().find("nodes").findall("node")[1].findtext("virtRTGSource"))

	assert source_payload["source_links"]["kind"] == "file"
	assert Path(source_payload["source_links"]["path"]) == tmp_path / "resampled_volume.nrrd"


def test_scene_import_creates_safe_hidden_placeholder_for_missing_volume_files(tmp_path):
	"""Import one missing volumetric source without creating an invalid render state."""
	scene_xml = tmp_path / "missing_volume.vxrscene.xml"
	scene_xml.write_text(
		"""<?xml version="1.0" encoding="utf-8"?>
<virtRTGScene format="virtRTG-scene" version="1">
  <nodes>
    <node id="node-0000" parent="" class="VirtualXRay" label="vx" visible="1">
      <virtRTG encoding="json">{"schema":"virtRTG-virtual-xray","version":1,"geometry":{},"source_defaults":{},"physics":{},"presentation":{}}</virtRTG>
    </node>
    <node id="node-0001" parent="node-0000" class="Volumetric" label="missing-ct" visible="1">
      <virtRTGSource encoding="json">{"schema":"virtRTG-source-config","version":1,"source_type":"volumetric","enabled":true,"material_response_config":{"enabled":false},"source_links":{"kind":"dicom_series","files":["Z:/does/not/exist/0001.dcm"]}}</virtRTGSource>
    </node>
  </nodes>
</virtRTGScene>
""",
		encoding="utf-8",
	)

	imported = load_virtual_xray_scene(scene_xml)
	imported_volume = imported.children()[0]

	assert isinstance(imported_volume, Volumetric)
	assert imported_volume.visible is False
	assert imported_volume.shape == (1, 1, 1)
	assert len(imported_volume.metadata) == 1
	assert Path(imported_volume.m_dicom_files[0]) == Path("Z:/does/not/exist/0001.dcm")
