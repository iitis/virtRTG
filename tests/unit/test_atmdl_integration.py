# -*- coding: utf-8 -*-
"""ATMDL integration tests for the virtRTG plugin object."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dpVision import Mesh, Parser, Transform
from dpVision.parsers.parserATMDL import ParserATMDL, WriterATMDL

from plugins.virtRTG.sceneFormat import register_atmdl_integration, unregister_atmdl_integration
from plugins.virtRTG.virtualXRay import VirtualXRay
from plugins.virtRTG.xray.xrayProjection import XRaySourceProjection


def _make_mesh(label="mesh-source"):
	"""Create one small triangle mesh suitable for OBJ round-trip tests."""
	mesh = Mesh.create(
		vertices=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 12.0, 0.0]],
		faces=[[0, 1, 2]],
	)
	mesh.label = label
	return mesh


def test_virtual_xray_atmdl_roundtrip_with_standard_mesh_child(tmp_path):
	"""ATMDL should load plugin-defined VirtualXRay plus standard child objects."""
	register_atmdl_integration()
	try:
		mesh = _make_mesh()
		mesh_path = tmp_path / "mesh_source.obj"
		assert Parser.save(mesh, str(mesh_path)) is True

		virtual_xray = VirtualXRay()
		virtual_xray.label = "vx-atmdl"
		virtual_xray.projection_mode = "parallel"
		virtual_xray.physics_source_energy_kev = 92.0
		virtual_xray.presentation_mode = "film"
		virtual_xray.last_raw_projection = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
		virtual_xray.last_line_integral_projection = np.array([[1.2, 1.4], [1.6, 1.8]], dtype=np.float32)
		virtual_xray.last_source_projections = [
			XRaySourceProjection(
				source_index=0,
				label="mesh-child",
				source_type="MeshXRaySource",
				line_integral_image=np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
				detector_image=np.array([[0.03, 0.04], [0.05, 0.06]], dtype=np.float32),
			),
		]

		frame = Transform()
		frame.label = "source-frame"
		virtual_xray.addChild(frame)

		loaded_mesh = Parser.load(str(mesh_path))
		assert loaded_mesh is not None
		loaded_mesh.label = "mesh-child"
		loaded_mesh.xray_source_enabled = True
		loaded_mesh.xray_scalar_scale = 2.5
		loaded_mesh.xray_scalar_bias = -120.0
		loaded_mesh.xray_attenuation_multiplier = 1.7
		loaded_mesh.xray_mesh_backend = "analytic_bvh"
		loaded_mesh.xray_mesh_mode = "shell"
		loaded_mesh.xray_mesh_scalar_value = 1337.0
		loaded_mesh.xray_mesh_shell_thickness_mm = 2.25
		frame.addChild(loaded_mesh)

		scene_path = tmp_path / "virtual_xray.atmdl"
		assert WriterATMDL.save(virtual_xray, str(scene_path)) is True
		text = Path(scene_path).read_text(encoding="utf-8")
		assert "virtRTGConfig64 " in text
		assert "geometry {" in text
		assert "physics {" in text
		assert "presentation {" in text
		assert "sourceNode {" in text

		imported = ParserATMDL.load(str(scene_path))
		assert imported is not None
		if imported.hasType("Transform"):
			assert len(imported.children()) == 1
			imported = imported.children()[0]
		assert imported.hasType("VirtualXRay")
		assert imported.label == "vx-atmdl"
		assert imported.projection_mode == "parallel"
		assert imported.physics_source_energy_kev == 92.0
		assert imported.presentation_mode == "film"
		assert np.allclose(imported.last_raw_projection, virtual_xray.last_raw_projection, atol=1e-6)
		assert np.allclose(imported.last_line_integral_projection, virtual_xray.last_line_integral_projection, atol=1e-6)
		assert len(imported.last_source_projections) == 1

		assert len(imported.children()) == 1
		imported_frame = imported.children()[0]
		assert imported_frame.hasType("Transform")
		assert imported_frame.label == "source-frame"

		assert len(imported_frame.children()) == 1
		imported_mesh = imported_frame.children()[0]
		assert imported_mesh.hasType("Mesh")
		assert imported_mesh.label == "mesh-child"
		assert imported_mesh.xray_source_enabled is True
		assert imported_mesh.xray_scalar_scale == 2.5
		assert imported_mesh.xray_scalar_bias == -120.0
		assert imported_mesh.xray_attenuation_multiplier == 1.7
		assert imported_mesh.xray_mesh_backend == "analytic_bvh"
		assert imported_mesh.xray_mesh_mode == "shell"
		assert imported_mesh.xray_mesh_scalar_value == 1337.0
		assert imported_mesh.xray_mesh_shell_thickness_mm == 2.25
		np.testing.assert_allclose(imported_mesh.m_vertices.shape, loaded_mesh.m_vertices.shape)
	finally:
		unregister_atmdl_integration()


def test_virtual_xray_atmdl_registration_affects_parser_capabilities():
	"""VirtualXRay should be saveable only when the plugin ATMDL hooks are registered."""
	virtual_xray = VirtualXRay()

	unregister_atmdl_integration()
	assert ParserATMDL.canSaveObject(virtual_xray) is False

	register_atmdl_integration()
	try:
		assert ParserATMDL.canSaveObject(virtual_xray) is True
	finally:
		unregister_atmdl_integration()


def test_virtual_xray_text_sections_override_stale_base64_payload(tmp_path):
	"""Editable text ATMDL sections should override older base64 payload values."""
	register_atmdl_integration()
	try:
		virtual_xray = VirtualXRay()
		virtual_xray.label = "vx-editable"
		virtual_xray.projection_mode = "cone"
		virtual_xray.physics_source_energy_kev = 70.0
		virtual_xray.presentation_gamma = 0.7

		scene_path = tmp_path / "virtual_xray_editable.atmdl"
		assert WriterATMDL.save(virtual_xray, str(scene_path)) is True

		text = Path(scene_path).read_text(encoding="utf-8")
		text = text.replace('projectionMode "cone"', 'projectionMode "parallel"')
		text = re.sub(r"sourceEnergyKeV 70(?:\.0+)?", "sourceEnergyKeV 123.5", text)
		text = re.sub(r"gamma 0\.7(?:0+)?", "gamma 1.9", text)
		Path(scene_path).write_text(text, encoding="utf-8", newline="\n")

		imported = ParserATMDL.load(str(scene_path))
		if imported.hasType("Transform"):
			imported = imported.children()[0]

		assert imported.hasType("VirtualXRay")
		assert imported.projection_mode == "parallel"
		assert imported.physics_source_energy_kev == 123.5
		assert imported.presentation_gamma == 1.9
	finally:
		unregister_atmdl_integration()
