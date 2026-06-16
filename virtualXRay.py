# -*- coding: utf-8 -*-
"""Scene-tree object that owns X-ray geometry and gathers descendant X-ray sources."""

from __future__ import annotations

import json
from pathlib import Path

import OpenGL.GL as gl
import numpy as np

from dpVision import Mesh, Object, Volumetric
from dpVision.marchingCubes import mc_estimate_threshold, mc_gradient

from .xray.xrayPresentation import (
	RawPresentationModel,
	FilmLikePresentationModel,
	DigitalRadiographyPresentationModel
)

from .xray.xraySource import (
	MeshXRaySource,
	VolumetricXRaySource,
	ensure_xray_source_config,
	get_xray_material_response_config,
)

from .xray.xrayProjection import (
	XRayPhysicsModel,
	XRayProjectionConfig,
	XRayProjectionGeometry,
	XRayProjectionQualityProfile,
	XRayScalarPreprocessor,
	XRayScene,
)
from .xray.xrayAnnotationOverlay import (
	XRayAnnotationProjectionContext,
	build_overlay_projection_set,
)


class VirtualXRay(Object):
	"""Represent one virtual X-ray setup integrated with the existing scene tree."""

	GEOMETRY_PRESET_FILE = Path(__file__).resolve().parent / "presets" / "xray_geometry_presets.json"

	DEFAULT_GEOMETRY_PRESETS = {
		"orthoralix": {
			"projection_mode": "cone",
			"detector_center_ref": [150.0, 0.0, 0.0],
			"detector_normal_ref": [-1.0, 0.0, 0.0],
			"detector_up_ref": [0.0, 0.0, 1.0],
			"detector_shape_hw": [800, 1000],
			"detector_pixel_size_mm": [0.30, 0.30],
			"source_position_ref": [-1350.0, 0.0, 0.0],
			"ray_direction_ref": [1.0, 0.0, 0.0],
			"step_mm": 0.5,
			"quality_profile_name": "normal",
		},
		"ceph_lateral": {
			"projection_mode": "cone",
			"detector_center_ref": [400.0, 0.0, 0.0],
			"detector_normal_ref": [-1.0, 0.0, 0.0],
			"detector_up_ref": [0.0, 0.0, 1.0],
			"detector_shape_hw": [800, 1000],
			"detector_pixel_size_mm": [0.30, 0.30],
			"source_position_ref": [-1500.0, 0.0, 0.0],
			"ray_direction_ref": [1.0, 0.0, 0.0],
			"step_mm": 1.0,
			"quality_profile_name": "normal",
		},
		"ceph_pa": {
			"projection_mode": "cone",
			"detector_center_ref": [0.0, 400.0, 0.0],
			"detector_normal_ref": [0.0, -1.0, 0.0],
			"detector_up_ref": [0.0, 0.0, 1.0],
			"detector_shape_hw": [800, 1000],
			"detector_pixel_size_mm": [0.30, 0.30],
			"source_position_ref": [0.0, -1500.0, 0.0],
			"ray_direction_ref": [0.0, 1.0, 0.0],
			"step_mm": 1.0,
			"quality_profile_name": "normal",
		},
		"skull_ap": {
			"projection_mode": "cone",
			"detector_center_ref": [0.0, 400.0, 0.0],
			"detector_normal_ref": [0.0, -1.0, 0.0],
			"detector_up_ref": [0.0, 0.0, 1.0],
			"detector_shape_hw": [900, 900],
			"detector_pixel_size_mm": [0.28, 0.28],
			"source_position_ref": [0.0, -1100.0, 0.0],
			"ray_direction_ref": [0.0, 1.0, 0.0],
			"step_mm": 0.9,
			"quality_profile_name": "normal",
		},
		"cone_closeup": {
			"projection_mode": "cone",
			"detector_center_ref": [0.0, 0.0, 180.0],
			"detector_normal_ref": [0.0, 0.0, -1.0],
			"detector_up_ref": [0.0, 1.0, 0.0],
			"detector_shape_hw": [768, 768],
			"detector_pixel_size_mm": [0.22, 0.22],
			"source_position_ref": [0.0, 0.0, -260.0],
			"ray_direction_ref": [0.0, 0.0, 1.0],
			"step_mm": 0.8,
			"quality_profile_name": "normal",
		},
	}

	PRESENTATION_PRESETS = {
		"default": {
			"presentation_mode": "digital",
			"presentation_invert": False,
			"presentation_gamma": 0.7,
			"presentation_contrast": 1.2,
			"presentation_robust_percentile": 99.5,
			"presentation_window_center": None,
			"presentation_window_width": None,
		},
		"balanced": {
			"presentation_mode": "digital",
			"presentation_invert": False,
			"presentation_gamma": 0.85,
			"presentation_contrast": 1.10,
			"presentation_robust_percentile": 99.2,
			"presentation_window_center": None,
			"presentation_window_width": None,
		},
		"bone_soft": {
			"presentation_mode": "digital",
			"presentation_invert": False,
			"presentation_gamma": 1.35,
			"presentation_contrast": 1.18,
			"presentation_robust_percentile": 98.8,
			"presentation_window_center": None,
			"presentation_window_width": None,
		},
		"bone_contrast": {
			"presentation_mode": "digital",
			"presentation_invert": False,
			"presentation_gamma": 1.55,
			"presentation_contrast": 1.40,
			"presentation_robust_percentile": 98.4,
			"presentation_window_center": None,
			"presentation_window_width": None,
		},
		"film_soft": {
			"presentation_mode": "film",
			"presentation_invert": False,
			"presentation_gamma": 1.60,
			"presentation_contrast": 1.10,
			"presentation_robust_percentile": 99.0,
			"presentation_window_center": None,
			"presentation_window_width": None,
		},
	}

	def __init__(self, parent=None):
		"""Initialize one X-ray scene node with default geometry, physics and presentation settings."""
		super().__init__(parent)
		self.label = "VirtualXRay"

		self.detector_center_ref = np.array([0.0, 0.0, 180.0], dtype=np.float32)
		self.detector_normal_ref = np.array([0.0, 0.0, -1.0], dtype=np.float32)
		self.detector_up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		self.detector_shape_hw = [512, 512]
		self.detector_pixel_size_mm = [0.4, 0.4]

		self.source_position_ref = np.array([0.0, 0.0, -220.0], dtype=np.float32)
		self.ray_direction_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
		self.projection_mode = "cone"
		self.step_mm = 1.0
		self.depth_window_mode = "off"
		self.depth_window_mm = [0.0, 0.0]
		self.depth_window_origin_ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		self.depth_window_axis_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
		self.last_raw_projection = None
		self.last_projected_annotations = None
		self.last_projection_image = None

		self.quality_profile_name = "normal"
		self.source_interpolation = "linear"
		self.source_fill_value = None
		self.source_preprocess_mode = "none"
		self.source_preprocess_low_percentile = 0.5
		self.source_preprocess_high_percentile = 99.5
		self.source_preprocess_output_low = -1000.0
		self.source_preprocess_output_high = 2500.0
		self.mesh_source_scalar_value = 1800.0
		self.mesh_source_mode = "solid"
		self.mesh_surface_thickness_mm = 1.0

		self.physics_mu_air = 0.0
		self.physics_mu_water = 0.02
		self.physics_hounsfield_air = -1000.0
		self.physics_attenuation_scale = 1.0
		self.physics_source_energy_kev = 70.0
		self.physics_reference_energy_kev = 70.0
		self.physics_attenuation_energy_exponent = 2.0
		self.physics_output_mode = "integral"
		self.physics_intensity_floor = 0.0
		self.physics_source_distance_falloff_mode = "none"
		self.physics_source_distance_reference_mm = None
		self.physics_source_distance_power = 2.0
		self.physics_material_response_mode = "linear"
		self.physics_bone_threshold_hu = None
		self.physics_bone_threshold_softness = 250.0
		self.physics_material_window_center = None
		self.physics_material_window_width = None
		self.physics_material_window_mode = "hard"
		self.physics_material_window_softness = 150.0

		self.presentation_mode = "digital"
		self.presentation_invert = False
		self.presentation_gamma = 0.7
		self.presentation_contrast = 1.2
		self.presentation_robust_percentile = 99.5
		self.presentation_window_center = None
		self.presentation_window_width = None
		self.presentation_overlay_annotations = False
		self.presentation_overlay_labels = False
		self.presentation_overlay_cross_size_px = 6

		self.detector_fill_color = (0.18, 0.55, 0.62)
		self.detector_edge_color = (0.42, 0.90, 0.95)
		self.detector_cross_color = (0.24, 0.72, 0.78)
		self.source_color = (1.00, 0.72, 0.22)
		self.link_color = (0.96, 0.78, 0.34)
		self.frustum_color = (0.86, 0.84, 0.52)
		self.depth_window_fill_color = (0.88, 0.42, 0.18)
		self.depth_window_edge_color = (1.00, 0.62, 0.28)
		self.depth_window_link_color = (0.98, 0.76, 0.44)
		self.axis_colors = (
			(0.92, 0.30, 0.30),
			(0.30, 0.82, 0.42),
			(0.34, 0.54, 0.95),
		)
		self.detector_fill_alpha = 0.20
		self.frustum_alpha = 0.28
		self.depth_window_fill_alpha = 0.30
		self.depth_window_link_alpha = 0.24
		self.source_gizmo_size_mm = 4.0
		self.axis_gizmo_length_mm = 18.0

	def reference_transform(self):
		"""Return the global transform of this scene node used as a local X-ray reference frame."""
		return np.asarray(self.getGlobalTransformation(), dtype=np.float32)

	def export_scene_description(self, path, parent_widget=None, interactive=False):
		"""Save one simplified, plugin-local scene description to an XML file."""
		from .sceneFormat import save_virtual_xray_scene
		return save_virtual_xray_scene(
			self,
			path,
			parent_widget=parent_widget,
			interactive=interactive,
		)

	def _ensure_depth_window_defaults(self):
		"""Backfill depth-window attributes for older serialized objects."""
		if not hasattr(self, "depth_window_mode"):
			self.depth_window_mode = "off"
		if not hasattr(self, "depth_window_mm"):
			self.depth_window_mm = [0.0, 0.0]
		if not hasattr(self, "depth_window_origin_ref"):
			self.depth_window_origin_ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		if not hasattr(self, "depth_window_axis_ref"):
			self.depth_window_axis_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)

	def _ensure_physics_defaults(self):
		"""Backfill newer physics attributes for older serialized objects."""
		if not hasattr(self, "physics_source_energy_kev"):
			self.physics_source_energy_kev = 70.0
		if not hasattr(self, "physics_reference_energy_kev"):
			self.physics_reference_energy_kev = 70.0
		if not hasattr(self, "physics_attenuation_energy_exponent"):
			self.physics_attenuation_energy_exponent = 2.0
		if not hasattr(self, "physics_source_distance_falloff_mode"):
			self.physics_source_distance_falloff_mode = "none"
		if not hasattr(self, "physics_source_distance_reference_mm"):
			self.physics_source_distance_reference_mm = None
		if not hasattr(self, "physics_source_distance_power"):
			self.physics_source_distance_power = 2.0

	def _ensure_annotation_projection_defaults(self):
		"""Backfill runtime annotation-projection attributes for older serialized objects."""
		if not hasattr(self, "last_projected_annotations"):
			self.last_projected_annotations = None
		if not hasattr(self, "presentation_overlay_annotations"):
			self.presentation_overlay_annotations = False
		if not hasattr(self, "presentation_overlay_labels"):
			self.presentation_overlay_labels = False
		if not hasattr(self, "presentation_overlay_cross_size_px"):
			self.presentation_overlay_cross_size_px = 6

	def child_transform_relative_to_self(self, child):
		"""Return one descendant transform expressed in the local frame of this X-ray object."""
		self_global = self.reference_transform()
		child_global = np.asarray(child.getGlobalTransformation(), dtype=np.float32)
		return np.linalg.inv(self_global) @ child_global

	def _iter_descendants(self, node=None):
		"""Yield descendants recursively while treating nested `VirtualXRay` nodes as separate sub-scenes."""
		node = self if node is None else node
		for child in node.children():
			yield child
			if isinstance(child, VirtualXRay):
				continue
			yield from self._iter_descendants(child)

	def collect_volumetrics(self):
		"""Return volumetric descendants that should participate in this X-ray scene."""
		return [ensure_xray_source_config(node) for node in self._iter_descendants() if isinstance(node, Volumetric)]

	def collect_meshes(self):
		"""Return mesh descendants that should participate in this X-ray scene."""
		return [ensure_xray_source_config(node) for node in self._iter_descendants() if isinstance(node, Mesh)]

	def collect_projectable_annotations(self):
		"""Return descendant scene annotations that may contribute detector overlays."""
		return [
			node for node in self._iter_descendants()
			if hasattr(node, "getColor") and hasattr(node, "getSelColor")
		]

	def collect_xray_objects(self):
		"""Return X-ray-capable descendants that are currently enabled."""
		candidates = self.collect_volumetrics() + self.collect_meshes()
		return [obj for obj in candidates if bool(getattr(obj, "xray_source_enabled", True))]

	def scene_sources(self):
		"""Build X-ray sample sources from descendant volumetrics and meshes."""
		scalar_preprocessor = self.build_scalar_preprocessor()
		sources = [
			VolumetricXRaySource(
				volumetric=vol,
				global_transform=self.child_transform_relative_to_self(vol),
				interpolation=(
					self.source_interpolation
					if str(getattr(vol, "xray_interpolation_override", "default")).lower() == "default"
					else str(vol.xray_interpolation_override).lower()
				),
				fill_value=(
					self.source_fill_value
					if not bool(getattr(vol, "xray_fill_value_override_enabled", False))
					else float(vol.xray_fill_value_override)
				),
				scalar_preprocessor=scalar_preprocessor,
				scalar_scale=float(getattr(vol, "xray_scalar_scale", 1.0)),
				scalar_bias=float(getattr(vol, "xray_scalar_bias", 0.0)),
				attenuation_multiplier=float(getattr(vol, "xray_attenuation_multiplier", 1.0)),
				material_response_config=get_xray_material_response_config(vol),
				volume_backend=str(getattr(vol, "xray_volume_backend", "sampling")).lower(),
			)
			for vol in self.collect_volumetrics()
			if bool(getattr(vol, "xray_source_enabled", True))
		]
		sources.extend(
			MeshXRaySource(
				mesh=mesh,
				global_transform=self.child_transform_relative_to_self(mesh),
				scalar_value=float(getattr(mesh, "xray_mesh_scalar_value", self.mesh_source_scalar_value)),
				backend=str(getattr(mesh, "xray_mesh_backend", "analytic_bvh")).lower(),
				mode=str(getattr(mesh, "xray_mesh_mode", self.mesh_source_mode)).lower(),
				shell_thickness_mm=float(getattr(mesh, "xray_mesh_shell_thickness_mm", self.mesh_surface_thickness_mm)),
				scalar_scale=float(getattr(mesh, "xray_scalar_scale", 1.0)),
				scalar_bias=float(getattr(mesh, "xray_scalar_bias", 0.0)),
				attenuation_multiplier=float(getattr(mesh, "xray_attenuation_multiplier", 1.0)),
				material_response_config=get_xray_material_response_config(mesh),
			)
			for mesh in self.collect_meshes()
			if bool(getattr(mesh, "xray_source_enabled", True))
		)
		return sources

	def build_scene(self):
		"""Return an `XRayScene` assembled from the current descendant X-ray sources."""
		return XRayScene.from_sample_sources(self.scene_sources())

	def quality_profile(self):
		"""Return the selected quality profile instance."""
		name = str(self.quality_profile_name).lower()
		if name == "draft":
			return XRayProjectionQualityProfile.draft()
		if name == "high":
			return XRayProjectionQualityProfile.high()
		if name == "custom":
			# preserve step_mm from geometry; only detector_downsample=1
			return XRayProjectionQualityProfile(name="custom", step_mm=None, detector_downsample=1)
		return XRayProjectionQualityProfile.normal()

	def build_geometry(self):
		"""Build the current projection geometry from intuitive detector pose parameters."""
		self._ensure_depth_window_defaults()
		is_cone = str(self.projection_mode).lower() == "cone"
		depth_mode = str(self.depth_window_mode).strip().lower()
		if depth_mode in {"", "none", "off"}:
			depth_mode = None
		geometry_depth_mode = depth_mode
		depth_axis_ref = None
		if depth_mode in {"planar", "planar_auto"}:
			geometry_depth_mode = "planar"
			depth_axis_ref = self._projection_axis_ref()
		elif depth_mode == "planar_custom":
			geometry_depth_mode = "planar"
			depth_axis_ref = np.asarray(self.depth_window_axis_ref, dtype=np.float32)
		return XRayProjectionGeometry.from_detector_pose(
			detector_center_ref=self.detector_center_ref,
			detector_normal_ref=self.detector_normal_ref,
			detector_up_ref=self.detector_up_ref,
			detector_shape_hw=self.detector_shape_hw,
			detector_pixel_size_mm=self.detector_pixel_size_mm,
			step_mm=self.step_mm,
			source_position_ref=self.source_position_ref if is_cone else None,
			ray_direction_ref=self.ray_direction_ref if not is_cone else None,
			depth_window_mode=geometry_depth_mode,
			depth_window_mm=self.depth_window_mm if geometry_depth_mode is not None else None,
			depth_window_origin_ref=self.depth_window_origin_ref if geometry_depth_mode == "planar" else None,
			depth_window_axis_ref=depth_axis_ref,
		)

	def build_scalar_preprocessor(self):
		"""Build the optional source scalar preprocessor used before attenuation mapping."""
		return XRayScalarPreprocessor(
			mode=self.source_preprocess_mode,
			input_low_percentile=self.source_preprocess_low_percentile,
			input_high_percentile=self.source_preprocess_high_percentile,
			output_low_value=self.source_preprocess_output_low,
			output_high_value=self.source_preprocess_output_high,
		)

	def build_physics_model(self):
		"""Build the physics model described by the current object state."""
		self._ensure_physics_defaults()
		return XRayPhysicsModel(
			mu_air=self.physics_mu_air,
			mu_water=self.physics_mu_water,
			hounsfield_air=self.physics_hounsfield_air,
			attenuation_scale=self.physics_attenuation_scale,
			source_energy_kev=self.physics_source_energy_kev,
			reference_energy_kev=self.physics_reference_energy_kev,
			attenuation_energy_exponent=self.physics_attenuation_energy_exponent,
			output_mode=self.physics_output_mode,
			intensity_floor=self.physics_intensity_floor,
			source_distance_falloff_mode=self.physics_source_distance_falloff_mode,
			source_distance_reference_mm=self.physics_source_distance_reference_mm,
			source_distance_power=self.physics_source_distance_power,
			material_response_mode=self.physics_material_response_mode,
			bone_threshold_hu=self.physics_bone_threshold_hu,
			bone_threshold_softness=self.physics_bone_threshold_softness,
			material_window_center=self.physics_material_window_center,
			material_window_width=self.physics_material_window_width,
			material_window_mode=self.physics_material_window_mode,
			material_window_softness=self.physics_material_window_softness,
		)

	def build_presentation_model(self):
		"""Build the selected presentation model for display-ready output."""
		mode = str(self.presentation_mode).lower()
		if mode == "raw":
			return RawPresentationModel()
		if mode == "film":
			return FilmLikePresentationModel(
				robust_percentile=self.presentation_robust_percentile,
				gamma=self.presentation_gamma,
				contrast=self.presentation_contrast,
				invert=self.presentation_invert,
			)
		return DigitalRadiographyPresentationModel(
			window_center=self.presentation_window_center,
			window_width=self.presentation_window_width,
			robust_percentile=self.presentation_robust_percentile,
			invert=self.presentation_invert,
			gamma=self.presentation_gamma,
			contrast=self.presentation_contrast,
		)

	@classmethod
	def presentation_preset_names(cls):
		"""Return presentation preset names exposed by the scene object."""
		return list(cls.PRESENTATION_PRESETS.keys())

	@classmethod
	def geometry_preset_names(cls):
		"""Return geometry preset names exposed by the scene object."""
		return list(cls.load_geometry_presets().keys())

	@classmethod
	def load_geometry_presets(cls):
		"""Load geometry presets from JSON and fall back to built-in defaults when needed."""
		preset_file = Path(cls.GEOMETRY_PRESET_FILE)
		if not preset_file.exists():
			print(f"Warning: Geometry preset file not found at {preset_file}. Using built-in defaults.")
			return dict(cls.DEFAULT_GEOMETRY_PRESETS)

		try:
			with preset_file.open("r", encoding="utf-8") as handle:
				payload = json.load(handle)
		except Exception:
			return dict(cls.DEFAULT_GEOMETRY_PRESETS)

		if not isinstance(payload, dict):
			return dict(cls.DEFAULT_GEOMETRY_PRESETS)

		presets = {}
		for preset_name, preset_definition in payload.items():
			if not isinstance(preset_definition, dict):
				continue
			presets[str(preset_name).lower()] = dict(preset_definition)

		if not presets:
			return dict(cls.DEFAULT_GEOMETRY_PRESETS)
		return presets

	def apply_presentation_preset(self, preset_name):
		"""Apply one predefined presentation preset to the current object state."""
		preset = self.PRESENTATION_PRESETS.get(str(preset_name).lower())
		if preset is None:
			raise KeyError(f"Unknown presentation preset: {preset_name}")
		for attr_name, attr_value in preset.items():
			setattr(self, attr_name, attr_value)

	def apply_geometry_preset(self, preset_name):
		"""Apply one predefined geometry preset to the current X-ray setup."""
		preset = self.load_geometry_presets().get(str(preset_name).lower())
		if preset is None:
			raise KeyError(f"Unknown geometry preset: {preset_name}")
		for attr_name, attr_value in preset.items():
			if attr_name.endswith("_ref"):
				setattr(self, attr_name, np.asarray(attr_value, dtype=np.float32))
			elif attr_name in {"detector_shape_hw"}:
				setattr(self, attr_name, [int(attr_value[0]), int(attr_value[1])])
			elif attr_name in {"detector_pixel_size_mm"}:
				setattr(self, attr_name, [float(attr_value[0]), float(attr_value[1])])
			elif attr_name == "step_mm":
				setattr(self, attr_name, float(attr_value))
			else:
				setattr(self, attr_name, attr_value)

	def build_projection_config(self):
		"""Return a complete projection configuration based on this scene object."""
		return XRayProjectionConfig(
			geometry=self.build_geometry(),
			physics_model=self.build_physics_model(),
			presentation_model=self.build_presentation_model(),
			reference_transform=np.eye(4, dtype=np.float32),
			quality_profile=self.quality_profile(),
		)

	def project_scene_annotations(self):
		"""Project descendant annotations into detector-space overlay primitives."""
		self._ensure_annotation_projection_defaults()
		config = self.build_projection_config()
		geometry = config.effective_geometry()
		context = XRayAnnotationProjectionContext(
			geometry=geometry,
			reference_transform=self.reference_transform(),
		)
		self.last_projected_annotations = build_overlay_projection_set(
			self.collect_projectable_annotations(),
			context=context,
		)
		return self.last_projected_annotations

	def estimate_bone_threshold(self, threshold_min=300.0, max_sample_voxels=4_000_000):
		"""Estimate one bone HU threshold from descendant volumetrics.

		This reuses the gradient-driven heuristic from `marchingCubes.py`, but
		stops at the threshold estimate instead of expanding it into a full
		projection window.
		"""
		estimates = []
		for volumetric in self.collect_volumetrics():
			volume = np.asarray(volumetric.m_volume, dtype=np.float32)
			if volume.ndim != 3 or volume.size == 0:
				continue

			downsample = 1
			if volume.size > max_sample_voxels:
				downsample = int(np.ceil((volume.size / float(max_sample_voxels)) ** (1.0 / 3.0)))
			volume_sample = volume[::downsample, ::downsample, ::downsample]

			_origin_world, _axes_world, spacing_xyz = volumetric.get_volume_geometry()
			px = max(1e-6, float(spacing_xyz[0]) * downsample)
			py = max(1e-6, float(spacing_xyz[1]) * downsample)
			pz = max(1e-6, float(spacing_xyz[2]) * downsample)
			gradient, _gx_full, _gy_full, _gz_full = mc_gradient(
				volume_sample,
				None,
				pz,
				py,
				px,
				sharpening=False,
			)
			threshold = mc_estimate_threshold(
				volume_sample,
				gradient,
				threshold=None,
				threshold_min=threshold_min,
			)
			estimates.append({
				"threshold": float(threshold),
				"voxel_count": int(volume_sample.size),
			})

		if not estimates:
			raise ValueError("Could not estimate one bone HU threshold from the current X-ray scene.")

		weights = np.asarray([item["voxel_count"] for item in estimates], dtype=np.float64)
		weights /= max(weights.sum(), 1.0)

		def _weighted_average(field_name):
			return float(sum(item[field_name] * weight for item, weight in zip(estimates, weights)))

		return {
			"threshold": _weighted_average("threshold"),
			"per_volume": estimates,
		}

	def apply_estimated_bone_threshold(self, threshold_min=300.0):
		"""Estimate and apply one bone HU threshold to the current scene."""
		estimate = self.estimate_bone_threshold(threshold_min=threshold_min)
		self.physics_material_response_mode = "bone_threshold"
		self.physics_bone_threshold_hu = float(estimate["threshold"])
		self.physics_material_window_center = None
		self.physics_material_window_width = None
		return estimate

	def project(self, return_stats=False):
		"""Project all descendant X-ray sources using the current setup state."""
		return self.build_scene().project(self.build_projection_config(), return_stats=return_stats)

	def render_projection(self, return_stats=False):
		"""Project the current scene and immediately apply the configured presentation model."""
		return self.build_scene().render(self.build_projection_config(), return_stats=return_stats)

	def project_and_cache(self, return_stats=False, progress_callback=None):
		"""Project the scene, store the raw result in `last_raw_projection`, and return it."""
		self._ensure_annotation_projection_defaults()
		if return_stats:
			raw, stats = self.build_scene().project(self.build_projection_config(), return_stats=True, progress_callback=progress_callback)
			self.last_raw_projection = np.asarray(raw, dtype=np.float32)
			self.project_scene_annotations()
			return self.last_raw_projection, stats
		raw = self.build_scene().project(self.build_projection_config(), return_stats=False, progress_callback=progress_callback)
		self.last_raw_projection = np.asarray(raw, dtype=np.float32)
		self.project_scene_annotations()
		return self.last_raw_projection

	def apply_presentation(self):
		"""Apply the current presentation model to `last_raw_projection` without re-projecting.

		Returns the display-ready float32 image, or ``None`` if no projection has been cached yet.
		"""
		if self.last_raw_projection is None:
			return None
		return self.build_presentation_model().apply(self.last_raw_projection)

	def detector_corners_ref(self):
		"""Return detector corners in local reference coordinates for gizmo drawing and bounding box computation."""
		geometry = self.build_geometry()
		height, width = int(geometry.detector_shape_hw[0]), int(geometry.detector_shape_hw[1])
		origin = np.asarray(geometry.detector_origin_ref, dtype=np.float32)
		u = np.asarray(geometry.detector_u_ref, dtype=np.float32)
		v = np.asarray(geometry.detector_v_ref, dtype=np.float32)
		return np.array([
			origin,
			origin + u * float(width - 1),
			origin + u * float(width - 1) + v * float(height - 1),
			origin + v * float(height - 1),
		], dtype=np.float32)

	def _projection_axis_ref(self):
		"""Return the current main projection axis in local reference coordinates."""
		if str(self.projection_mode).lower() == "cone":
			source = np.asarray(self.source_position_ref, dtype=np.float32)
			detector_center = np.asarray(self.detector_center_ref, dtype=np.float32)
			axis = detector_center - source
		else:
			axis = np.asarray(self.ray_direction_ref, dtype=np.float32)
		norm = float(np.linalg.norm(axis))
		if norm <= 1e-8:
			return np.array([0.0, 0.0, 1.0], dtype=np.float32)
		return axis / norm

	def _projection_axis_anchor_ref(self):
		"""Return one point lying on the current main projection axis."""
		if str(self.projection_mode).lower() == "cone":
			return np.asarray(self.detector_center_ref, dtype=np.float32)
		return np.asarray(self.detector_center_ref, dtype=np.float32)

	def _detector_axes_ref(self):
		"""Return detector pixel axes and centre in local reference coordinates."""
		geometry = self.build_geometry()
		center = np.asarray(geometry.detector_center_ref_point(), dtype=np.float32)
		u = np.asarray(geometry.detector_u_ref, dtype=np.float32)
		v = np.asarray(geometry.detector_v_ref, dtype=np.float32)
		height, width = int(geometry.detector_shape_hw[0]), int(geometry.detector_shape_hw[1])
		u_span = u * float(max(width - 1, 1))
		v_span = v * float(max(height - 1, 1))
		return center, u_span, v_span

	def _slab_basis_from_axis_ref(self, axis):
		"""Return two in-plane slab vectors orthogonal to the provided axis."""
		axis = np.asarray(axis, dtype=np.float32)
		axis_norm = float(np.linalg.norm(axis))
		if axis_norm <= 1e-8:
			return None, None
		axis = axis / axis_norm

		geometry = self.build_geometry()
		detector_up = np.asarray(geometry.detector_v_ref, dtype=np.float32)
		detector_up_norm = float(np.linalg.norm(detector_up))
		if detector_up_norm > 1e-8:
			detector_up = detector_up / detector_up_norm
		else:
			detector_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

		u_size = float(np.linalg.norm(np.asarray(geometry.detector_u_ref, dtype=np.float32)) * max(int(geometry.detector_shape_hw[1]) - 1, 1))
		v_size = float(np.linalg.norm(np.asarray(geometry.detector_v_ref, dtype=np.float32)) * max(int(geometry.detector_shape_hw[0]) - 1, 1))

		basis_v = detector_up - axis * float(np.dot(detector_up, axis))
		basis_v_norm = float(np.linalg.norm(basis_v))
		if basis_v_norm <= 1e-8:
			fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
			if abs(float(np.dot(fallback, axis))) > 0.9:
				fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
			basis_v = fallback - axis * float(np.dot(fallback, axis))
			basis_v_norm = float(np.linalg.norm(basis_v))
			if basis_v_norm <= 1e-8:
				return None, None
		basis_v = basis_v / basis_v_norm
		basis_u = np.cross(basis_v, axis)
		basis_u_norm = float(np.linalg.norm(basis_u))
		if basis_u_norm <= 1e-8:
			return None, None
		basis_u = basis_u / basis_u_norm
		return basis_u * u_size, basis_v * v_size

	def _depth_window_mode_normalized(self):
		"""Return the normalized depth-window mode used by the scene object."""
		self._ensure_depth_window_defaults()
		mode = str(self.depth_window_mode).strip().lower()
		return None if mode in {"", "none", "off"} else mode

	def _depth_window_visual_quads_ref(self):
		"""Return one list of quads visualizing the configured depth window in local coordinates."""
		mode = self._depth_window_mode_normalized()
		if mode is None:
			return []

		depth_start = float(self.depth_window_mm[0])
		depth_end = float(self.depth_window_mm[1])
		if depth_end < depth_start:
			depth_start, depth_end = depth_end, depth_start

		if mode in {"planar", "planar_auto", "planar_custom"}:
			origin = np.asarray(self.depth_window_origin_ref, dtype=np.float32)
			if mode in {"planar", "planar_auto"}:
				axis = self._projection_axis_ref()
				axis_anchor = self._projection_axis_anchor_ref()
			else:
				axis = np.asarray(self.depth_window_axis_ref, dtype=np.float32)
				axis_norm = float(np.linalg.norm(axis))
				if axis_norm <= 1e-8:
					return []
				axis = axis / axis_norm
				axis_anchor = origin
			u_span, v_span = self._slab_basis_from_axis_ref(axis)
			if u_span is None or v_span is None:
				return []

			def _planar_quad(offset_mm):
				offset_mm = float(offset_mm)
				axis_anchor_offset = float(np.dot(axis_anchor - origin, axis))
				plane_center = axis_anchor + axis * (offset_mm - axis_anchor_offset)
				return np.array([
					plane_center - 0.5 * u_span - 0.5 * v_span,
					plane_center + 0.5 * u_span - 0.5 * v_span,
					plane_center + 0.5 * u_span + 0.5 * v_span,
					plane_center - 0.5 * u_span + 0.5 * v_span,
				], dtype=np.float32)

			return [_planar_quad(depth_start), _planar_quad(depth_end)]

		corners = self.detector_corners_ref()
		if str(self.projection_mode).lower() == "cone":
			source = np.asarray(self.source_position_ref, dtype=np.float32)

			def _cone_quad(offset_mm):
				dirs = corners - source[np.newaxis, :]
				norms = np.linalg.norm(dirs, axis=1, keepdims=True)
				dirs = dirs / np.maximum(norms, 1e-8)
				return source[np.newaxis, :] + dirs * float(offset_mm)

			return [_cone_quad(depth_start), _cone_quad(depth_end)]

		ray_dir = np.asarray(self.ray_direction_ref, dtype=np.float32)
		norm = float(np.linalg.norm(ray_dir))
		if norm <= 1e-8:
			return []
		ray_dir = ray_dir / norm
		return [
			corners + ray_dir[np.newaxis, :] * depth_start,
			corners + ray_dir[np.newaxis, :] * depth_end,
		]

	def getLocalBB(self):
		"""Return a local bounding box covering the source and detector gizmos."""
		points = [self.detector_corners_ref()]
		if str(self.projection_mode).lower() == "cone":
			points.append(np.asarray(self.source_position_ref, dtype=np.float32)[None, :])
		for quad in self._depth_window_visual_quads_ref():
			points.append(np.asarray(quad, dtype=np.float32))
		all_points = np.vstack(points)
		return True, all_points.min(axis=0).tolist(), all_points.max(axis=0).tolist()

	def renderSelf(self):
		"""Render a lightweight source-detector gizmo directly in the current OpenGL model space."""
		corners = self.detector_corners_ref()
		center = corners.mean(axis=0)
		axis_length = float(self.axis_gizmo_length_mm)

		gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
		gl.glDisable(gl.GL_TEXTURE_2D)
		gl.glDisable(gl.GL_LIGHTING)
		gl.glDisable(gl.GL_COLOR_MATERIAL)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glLineWidth(2.0)

		gl.glColor4f(
			self.detector_fill_color[0],
			self.detector_fill_color[1],
			self.detector_fill_color[2],
			self.detector_fill_alpha,
		)
		gl.glBegin(gl.GL_QUADS)
		for corner in corners:
			gl.glVertex3f(*corner)
		gl.glEnd()

		gl.glColor3f(*self.detector_edge_color)
		gl.glBegin(gl.GL_LINE_LOOP)
		for corner in corners:
			gl.glVertex3f(*corner)
		gl.glEnd()

		gl.glColor3f(*self.detector_cross_color)
		gl.glBegin(gl.GL_LINES)
		gl.glVertex3f(*corners[0]); gl.glVertex3f(*corners[2])
		gl.glVertex3f(*corners[1]); gl.glVertex3f(*corners[3])
		gl.glEnd()

		gl.glLineWidth(1.5)
		gl.glBegin(gl.GL_LINES)
		for axis_idx, axis_color in enumerate(self.axis_colors):
			gl.glColor3f(*axis_color)
			axis_end = np.array(center, dtype=np.float32)
			axis_end[axis_idx] += axis_length
			gl.glVertex3f(*center)
			gl.glVertex3f(*axis_end)
		gl.glEnd()

		if self.projection_mode == "cone":
			source = np.asarray(self.source_position_ref, dtype=np.float32)
			size = float(self.source_gizmo_size_mm)

			gl.glLineWidth(1.0)
			gl.glColor4f(
				self.frustum_color[0],
				self.frustum_color[1],
				self.frustum_color[2],
				self.frustum_alpha,
			)
			gl.glBegin(gl.GL_LINES)
			for corner in corners:
				gl.glVertex3f(*source)
				gl.glVertex3f(*corner)
			gl.glEnd()

			gl.glLineWidth(1.5)
			gl.glColor3f(*self.detector_edge_color)
			gl.glBegin(gl.GL_LINES)
			gl.glVertex3f(*(source*1.2))
			gl.glVertex3f(*(-source*0.2))
			gl.glEnd()

			gl.glLineWidth(2.0)
			gl.glColor3f(*self.source_color)
			gl.glBegin(gl.GL_LINES)
			gl.glVertex3f(source[0] - size, source[1], source[2]); gl.glVertex3f(source[0] + size, source[1], source[2])
			gl.glVertex3f(source[0], source[1] - size, source[2]); gl.glVertex3f(source[0], source[1] + size, source[2])
			gl.glVertex3f(source[0], source[1], source[2] - size); gl.glVertex3f(source[0], source[1], source[2] + size)
			gl.glColor3f(*self.link_color)
			gl.glVertex3f(*source); gl.glVertex3f(*center)
			gl.glEnd()
		elif self.projection_mode == "parallel":
			ray_dir = np.asarray(self.ray_direction_ref, dtype=np.float32)
			norm = np.linalg.norm(ray_dir)
			if norm > 1e-8:
				ray_dir = ray_dir / norm
				arrow_length = max(20.0, 0.25 * np.linalg.norm(corners[2] - corners[0]))
				arrow_start = center - ray_dir * arrow_length
				gl.glColor3f(*self.link_color)
				gl.glBegin(gl.GL_LINES)
				gl.glVertex3f(*arrow_start); gl.glVertex3f(*center)
				gl.glEnd()

		depth_quads = self._depth_window_visual_quads_ref()
		if len(depth_quads) == 2:
			gl.glLineWidth(1.5)
			for quad in depth_quads:
				gl.glColor4f(
					self.depth_window_fill_color[0],
					self.depth_window_fill_color[1],
					self.depth_window_fill_color[2],
					self.depth_window_fill_alpha,
				)
				gl.glBegin(gl.GL_QUADS)
				for point in quad:
					gl.glVertex3f(*point)
				gl.glEnd()

				gl.glColor3f(*self.depth_window_edge_color)
				gl.glBegin(gl.GL_LINE_LOOP)
				for point in quad:
					gl.glVertex3f(*point)
				gl.glEnd()

			gl.glColor4f(
				self.depth_window_link_color[0],
				self.depth_window_link_color[1],
				self.depth_window_link_color[2],
				self.depth_window_link_alpha,
			)
			gl.glBegin(gl.GL_LINES)
			for point_a, point_b in zip(depth_quads[0], depth_quads[1]):
				gl.glVertex3f(*point_a)
				gl.glVertex3f(*point_b)
			gl.glEnd()

		gl.glPopAttrib()

	def info(self):
		"""Return a compact textual summary for debugging and quick inspection."""
		self._ensure_depth_window_defaults()
		self._ensure_annotation_projection_defaults()
		volumes = len([obj for obj in self.collect_volumetrics() if bool(getattr(obj, "xray_source_enabled", True))])
		meshes = len([obj for obj in self.collect_meshes() if bool(getattr(obj, "xray_source_enabled", True))])
		annotations = len(self.collect_projectable_annotations())
		depth_mode = str(self.depth_window_mode).strip().lower()
		if depth_mode in {"", "none", "off"}:
			depth_summary = "off"
		else:
			depth_summary = f"{depth_mode}:{float(self.depth_window_mm[0]):.1f}->{float(self.depth_window_mm[1]):.1f}"
		return (
			f"VirtualXRay(mode={self.projection_mode}, volumes={volumes}, meshes={meshes}, annotations={annotations}, "
			f"detector_shape={self.detector_shape_hw}, step_mm={self.step_mm}, depth={depth_summary})"
		)
