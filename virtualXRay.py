# -*- coding: utf-8 -*-
"""Scene-tree object that owns X-ray geometry and gathers descendant X-ray sources."""

from __future__ import annotations

import json
from pathlib import Path

import OpenGL.GL as gl
import numpy as np

from dpVision import Mesh, Motion, Object, Volumetric
from dpVision.marchingCubes import mc_estimate_threshold, mc_gradient

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
	XRaySourceProjection,
)
from .xray.xrayAnnotationOverlay import (
	XRayAnnotationProjectionContext,
	XRayOverlayProjectionSet,
	build_overlay_projection_set,
	overlay_projection_set_from_payload,
	overlay_projection_set_to_payload,
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

	DETECTOR_IMAGE_PRESETS = {
		"default": {
			"mode": "digital",
			"invert": False,
			"gamma": 0.7,
			"contrast": 1.2,
			"input_transform": "linear",
			"local_enhancement": "off",
			"clahe_clip_limit": 2.0,
			"clahe_tile_grid_size": 8,
			"robust_low_percentile": 0.5,
			"robust_percentile": 99.5,
			"window_center": None,
			"window_width": None,
			"overlay_annotations": False,
			"overlay_labels": False,
			"overlay_cross_size_px": 6,
		},
		"balanced": {
			"mode": "digital",
			"invert": False,
			"gamma": 0.85,
			"contrast": 1.10,
			"input_transform": "linear",
			"local_enhancement": "off",
			"clahe_clip_limit": 2.0,
			"clahe_tile_grid_size": 8,
			"robust_low_percentile": 0.5,
			"robust_percentile": 99.2,
			"window_center": None,
			"window_width": None,
			"overlay_annotations": False,
			"overlay_labels": False,
			"overlay_cross_size_px": 6,
		},
		"bone_soft": {
			"mode": "digital",
			"invert": False,
			"gamma": 1.35,
			"contrast": 1.18,
			"input_transform": "log1p",
			"local_enhancement": "off",
			"clahe_clip_limit": 2.0,
			"clahe_tile_grid_size": 8,
			"robust_low_percentile": 0.5,
			"robust_percentile": 98.8,
			"window_center": None,
			"window_width": None,
			"overlay_annotations": False,
			"overlay_labels": False,
			"overlay_cross_size_px": 6,
		},
		"bone_contrast": {
			"mode": "digital",
			"invert": False,
			"gamma": 1.55,
			"contrast": 1.40,
			"input_transform": "log1p",
			"local_enhancement": "clahe",
			"clahe_clip_limit": 2.5,
			"clahe_tile_grid_size": 8,
			"robust_low_percentile": 0.5,
			"robust_percentile": 98.4,
			"window_center": None,
			"window_width": None,
			"overlay_annotations": False,
			"overlay_labels": False,
			"overlay_cross_size_px": 6,
		},
		"film_soft": {
			"mode": "film",
			"invert": False,
			"gamma": 1.60,
			"contrast": 1.10,
			"input_transform": "log1p",
			"local_enhancement": "off",
			"clahe_clip_limit": 2.0,
			"clahe_tile_grid_size": 8,
			"robust_low_percentile": 0.5,
			"robust_percentile": 99.0,
			"window_center": None,
			"window_width": None,
			"overlay_annotations": False,
			"overlay_labels": False,
			"overlay_cross_size_px": 6,
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
		self.motion_frame_mode = "active"
		self.step_mm = 1.0
		self.depth_window_mode = "off"
		self.depth_window_mm = [0.0, 0.0]
		self.depth_window_origin_ref = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		self.depth_window_axis_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
		self.last_line_integral_projection = None
		self.last_raw_projection = None
		self.last_source_projections = []
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
		self.physics_material_response_curve_points = XRayPhysicsModel.default_material_response_curve_points()
		self.physics_material_window_center = None
		self.physics_material_window_width = None
		self.physics_material_window_mode = "hard"
		self.physics_material_window_softness = 150.0

		self.detector_image_defaults = self.default_detector_image_defaults()

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
		if not hasattr(self, "motion_frame_mode"):
			self.motion_frame_mode = "active"

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

	def _ensure_detector_image_defaults(self):
		"""Backfill detector-image presentation defaults for older serialized objects."""
		if not hasattr(self, "detector_image_defaults"):
			self.detector_image_defaults = self.default_detector_image_defaults()
			return
		self.detector_image_defaults = self.sanitize_detector_image_defaults(self.detector_image_defaults)

	def _ensure_projection_cache_defaults(self):
		"""Backfill runtime projection-cache attributes for older serialized objects."""
		if not hasattr(self, "last_line_integral_projection"):
			self.last_line_integral_projection = None
		if not hasattr(self, "last_raw_projection"):
			self.last_raw_projection = None
		if not hasattr(self, "last_source_projections"):
			self.last_source_projections = []

	def child_transform_relative_to_self(self, child):
		"""Return one descendant transform expressed in the local frame of this X-ray object."""
		return self.child_transform_relative_to_self_for_frame(child, motion_frame_index=None)

	def child_transform_relative_to_self_for_frame(self, child, motion_frame_index=None):
		"""Return one descendant transform expressed in the local frame of this X-ray object.

		This path-aware assembly handles `Motion` ancestors explicitly so the
		projection pipeline can either use the currently active animation key or
		build alternate source groups for selected frame indices.
		"""
		transform_relative = np.eye(4, dtype=np.float32)
		node = child
		while node is not None and node is not self:
			transform_relative = self._local_xray_transform_for_node(
				node,
				motion_frame_index=motion_frame_index,
			) @ transform_relative
			node = node.parent
		return transform_relative

	def _local_xray_transform_for_node(self, node, motion_frame_index=None):
		"""Return one node-local transform used during X-ray scene assembly."""
		if isinstance(node, Motion):
			frame_index = node.currentKey() if motion_frame_index is None else motion_frame_index
			return self._motion_frame_transform(node, frame_index=frame_index)
		matrix = getattr(node, "matrix", None)
		if matrix is None:
			return np.eye(4, dtype=np.float32)
		matrix = np.asarray(matrix, dtype=np.float32)
		if matrix.shape != (4, 4):
			return np.eye(4, dtype=np.float32)
		return matrix

	def _motion_frame_transform(self, motion, frame_index=0):
		"""Return the selected `Motion` frame transform as a homogeneous matrix."""
		if not isinstance(motion, Motion) or motion.size() <= 0:
			return np.eye(4, dtype=np.float32)
		frame = motion.frame(frame_index)
		frame_transform = getattr(frame, "transform", None)
		matrix = getattr(frame_transform, "matrix", None)
		if matrix is None:
			return np.eye(4, dtype=np.float32)
		matrix = np.asarray(matrix, dtype=np.float32)
		if matrix.shape != (4, 4):
			return np.eye(4, dtype=np.float32)
		return matrix

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

	def collect_motions(self):
		"""Return descendant `Motion` nodes participating in this X-ray scene."""
		return [node for node in self._iter_descendants() if isinstance(node, Motion)]

	def primary_motion(self):
		"""Return the one supported animated subtree root or `None` when absent."""
		motions = self.collect_motions()
		if not motions:
			return None
		if len(motions) > 1:
			raise ValueError("VirtualXRay currently supports frame grouping for only one Motion object.")
		return motions[0]

	def motion_frame_indices(self):
		"""Return available frame indices for the supported descendant `Motion`."""
		motion = self.primary_motion()
		if motion is None:
			return []
		return list(range(max(0, int(motion.size()))))

	def effective_motion_frame_mode(self):
		"""Return the normalized motion-frame expansion mode."""
		mode = str(getattr(self, "motion_frame_mode", "active")).strip().lower()
		if mode not in {"active", "all"}:
			mode = "active"
		return mode

	def _build_scene_sources_for_frame(self, motion_frame_index=None, source_label_suffix=""):
		"""Build X-ray sample sources for one selected animation frame."""
		scalar_preprocessor = self.build_scalar_preprocessor()
		sources = [
			VolumetricXRaySource(
				volumetric=vol,
				global_transform=self.child_transform_relative_to_self_for_frame(vol, motion_frame_index=motion_frame_index),
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
				global_transform=self.child_transform_relative_to_self_for_frame(mesh, motion_frame_index=motion_frame_index),
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
		label_suffix = str(source_label_suffix)
		if label_suffix:
			for source in sources:
				base_object = getattr(source, "mesh", None) or getattr(source, "volumetric", None)
				base_label = getattr(base_object, "label", type(source).__name__)
				source.projection_label = f"{base_label}{label_suffix}"
		return sources

	def scene_sources(self, motion_frame_index=None):
		"""Build X-ray sample sources from descendant volumetrics and meshes.

		When one `Motion` is present and no explicit frame is requested, all frames
		are expanded into separate source instances so one projection includes the
		full animated subtree.
		"""
		if motion_frame_index is not None:
			return self._build_scene_sources_for_frame(
				motion_frame_index=motion_frame_index,
				source_label_suffix=f" [frame {int(motion_frame_index):03d}]",
			)

		frame_indices = self.motion_frame_indices()
		if not frame_indices:
			return self._build_scene_sources_for_frame(motion_frame_index=None)
		if self.effective_motion_frame_mode() != "all":
			return self._build_scene_sources_for_frame(
				motion_frame_index=None,
				source_label_suffix="",
			)

		sources = []
		for frame_index in frame_indices:
			sources.extend(self._build_scene_sources_for_frame(
				motion_frame_index=frame_index,
				source_label_suffix=f" [frame {int(frame_index):03d}]",
			))
		return sources

	def scene_source_groups(self):
		"""Return per-frame source groups for the supported descendant `Motion`.

		Each group represents the same source subtree sampled at one animation
		frame. When no `Motion` exists, one default group is returned.
		"""
		frame_indices = self.motion_frame_indices()
		if not frame_indices:
			return [{
				"group_index": 0,
				"frame_index": None,
				"label": "current",
				"sources": self._build_scene_sources_for_frame(motion_frame_index=None),
			}]
		return [
			{
				"group_index": int(frame_index),
				"frame_index": int(frame_index),
				"label": f"frame_{int(frame_index):03d}",
				"sources": self._build_scene_sources_for_frame(
					motion_frame_index=frame_index,
					source_label_suffix=f" [frame {int(frame_index):03d}]",
				),
			}
			for frame_index in frame_indices
		]

	def build_scene(self, motion_frame_index=None):
		"""Return an `XRayScene` assembled from the current descendant X-ray sources."""
		return XRayScene.from_sample_sources(self.scene_sources(motion_frame_index=motion_frame_index))

	def build_scene_groups(self):
		"""Return per-frame `XRayScene` groups for the supported descendant `Motion`."""
		return [
			{
				"group_index": int(group["group_index"]),
				"frame_index": group["frame_index"],
				"label": str(group["label"]),
				"scene": XRayScene.from_sample_sources(group["sources"]),
			}
			for group in self.scene_source_groups()
		]

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
			material_response_curve_points=XRayPhysicsModel.sanitize_material_response_curve_points(
				self.physics_material_response_curve_points
			),
			material_window_center=self.physics_material_window_center,
			material_window_width=self.physics_material_window_width,
			material_window_mode=self.physics_material_window_mode,
			material_window_softness=self.physics_material_window_softness,
		)

	@classmethod
	def detector_image_preset_names(cls):
		"""Return detector-image preset names exposed by the scene object."""
		return list(cls.DETECTOR_IMAGE_PRESETS.keys())

	@classmethod
	def default_detector_image_defaults(cls):
		"""Return the default detector-image presentation mapping."""
		return cls.sanitize_detector_image_defaults(cls.DETECTOR_IMAGE_PRESETS["default"])

	@classmethod
	def sanitize_detector_image_defaults(cls, defaults):
		"""Normalize one detector-image presentation mapping."""
		base = dict(cls.DETECTOR_IMAGE_PRESETS["default"])
		if isinstance(defaults, dict):
			base.update(defaults)
		robust_low_percentile = min(99.999, max(0.0, float(base.get("robust_low_percentile", 0.5))))
		return {
			"mode": str(base.get("mode", "digital")).lower(),
			"invert": bool(base.get("invert", False)),
			"gamma": max(0.05, float(base.get("gamma", 0.7))),
			"contrast": max(0.05, float(base.get("contrast", 1.2))),
			"input_transform": str(base.get("input_transform", "linear")).lower(),
			"local_enhancement": str(base.get("local_enhancement", "off")).lower(),
			"clahe_clip_limit": max(0.01, float(base.get("clahe_clip_limit", 2.0))),
			"clahe_tile_grid_size": max(1, int(base.get("clahe_tile_grid_size", 8))),
			"robust_low_percentile": robust_low_percentile,
			"robust_percentile": min(
				100.0,
				max(robust_low_percentile + 1e-6, float(base.get("robust_percentile", 99.5))),
			),
			"window_center": base.get("window_center", None),
			"window_width": base.get("window_width", None),
			"overlay_annotations": bool(base.get("overlay_annotations", False)),
			"overlay_labels": bool(base.get("overlay_labels", False)),
			"overlay_cross_size_px": max(1, int(base.get("overlay_cross_size_px", 6))),
		}

	def get_detector_image_defaults(self):
		"""Return a normalized copy of detector-image defaults."""
		self._ensure_detector_image_defaults()
		return dict(self.detector_image_defaults)

	def set_detector_image_defaults(self, defaults):
		"""Store normalized detector-image defaults."""
		self.detector_image_defaults = self.sanitize_detector_image_defaults(defaults)

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

	def apply_detector_image_preset(self, preset_name):
		"""Apply one predefined detector-image preset to the current object state."""
		preset = self.DETECTOR_IMAGE_PRESETS.get(str(preset_name).lower())
		if preset is None:
			raise KeyError(f"Unknown detector image preset: {preset_name}")
		self.set_detector_image_defaults(preset)

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
			presentation_model=None,
			reference_transform=np.eye(4, dtype=np.float32),
			quality_profile=self.quality_profile(),
		)

	def project_scene_annotations(self):
		"""Project descendant annotations into detector-space overlay primitives."""
		self._ensure_annotation_projection_defaults()
		config = self.build_projection_config()
		geometry = config.effective_geometry()
		reference_transform = self.reference_transform()
		annotations = self.collect_projectable_annotations()
		frame_indices = self.motion_frame_indices()
		if not frame_indices or self.effective_motion_frame_mode() != "all":
			context = XRayAnnotationProjectionContext(
				geometry=geometry,
				reference_transform=reference_transform,
				object_transform_resolver=lambda scene_object: self.child_transform_relative_to_self_for_frame(
					scene_object,
					motion_frame_index=None,
				),
			)
			self.last_projected_annotations = build_overlay_projection_set(
				annotations,
				context=context,
			)
			return self.last_projected_annotations

		all_items = []
		detector_shape_hw = (int(geometry.detector_shape_hw[0]), int(geometry.detector_shape_hw[1]))
		for frame_index in frame_indices:
			context = XRayAnnotationProjectionContext(
				geometry=geometry,
				reference_transform=reference_transform,
				object_transform_resolver=lambda scene_object, _frame_index=frame_index: self.child_transform_relative_to_self_for_frame(
					scene_object,
					motion_frame_index=_frame_index,
				),
				overlay_label_suffix=f" [frame {int(frame_index):03d}]",
			)
			projection_set = build_overlay_projection_set(
				annotations,
				context=context,
			)
			all_items.extend(list(projection_set.items))
		self.last_projected_annotations = XRayOverlayProjectionSet(
			detector_shape_hw=detector_shape_hw,
			items=all_items,
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

	def project_capture(self, return_stats=False, progress_callback=None):
		"""Project the current scene and return total plus per-source detector-space outputs."""
		self._ensure_projection_cache_defaults()
		return self.build_scene().project_capture(
			self.build_projection_config(),
			return_stats=return_stats,
			progress_callback=progress_callback,
		)

	def render_projection(self, return_stats=False):
		"""Project the current scene and immediately apply the configured presentation model."""
		return self.build_scene().render(self.build_projection_config(), return_stats=return_stats)

	def project_and_cache(self, return_stats=False, progress_callback=None):
		"""Project the scene, store the raw result in `last_raw_projection`, and return it."""
		self._ensure_annotation_projection_defaults()
		self._ensure_projection_cache_defaults()
		capture = self.project_capture(return_stats=return_stats, progress_callback=progress_callback)
		self.last_line_integral_projection = np.asarray(capture.line_integral_image, dtype=np.float32)
		self.last_raw_projection = np.asarray(capture.detector_image, dtype=np.float32)
		self.last_source_projections = list(capture.source_projections)
		self.project_scene_annotations()
		if return_stats:
			return self.last_raw_projection, capture.stats
		return self.last_raw_projection

	def _projection_mode_name(self):
		"""Return the normalized projection mode name used by the current geometry."""
		return "cone" if str(self.projection_mode).lower() == "cone" else "parallel"

	def _source_to_detector_distance_map(self, geometry):
		"""Return per-pixel source-to-detector distances for cone-beam detector conversion."""
		if not geometry.is_cone_beam():
			return None
		height, width = int(geometry.detector_shape_hw[0]), int(geometry.detector_shape_hw[1])
		detector_origin = np.asarray(geometry.detector_origin_ref, dtype=np.float32)
		detector_u = np.asarray(geometry.detector_u_ref, dtype=np.float32)
		detector_v = np.asarray(geometry.detector_v_ref, dtype=np.float32)
		source_position = np.asarray(geometry.source_position_ref, dtype=np.float32)
		col_grid, row_grid = np.meshgrid(
			np.arange(width, dtype=np.float32),
			np.arange(height, dtype=np.float32),
		)
		pixel_centers = (
			detector_origin
			+ detector_u * col_grid[:, :, np.newaxis]
			+ detector_v * row_grid[:, :, np.newaxis]
		).reshape(height * width, 3)
		return np.linalg.norm(pixel_centers - source_position[np.newaxis, :], axis=1).astype(np.float32)

	def _line_integral_to_detector_image(self, line_integral_image):
		"""Convert one cached line-integral detector map into the current raw detector output."""
		line_integral_image = np.asarray(line_integral_image, dtype=np.float32)
		geometry = self.build_projection_config().effective_geometry()
		distances = self._source_to_detector_distance_map(geometry)
		return self.build_physics_model().integral_to_image(
			line_integral_image.reshape(-1),
			source_to_detector_distance_mm=distances,
			projection_mode="cone" if geometry.is_cone_beam() else "parallel",
		).reshape(line_integral_image.shape).astype(np.float32, copy=False)

	def _resolve_projection_cache(self, stage):
		"""Return the requested cached detector-space array."""
		self._ensure_projection_cache_defaults()
		stage_name = str(stage).strip().lower()
		if stage_name in {"line", "line_integral", "integral"}:
			if self.last_line_integral_projection is None:
				raise ValueError("No cached line-integral projection is available.")
			return np.asarray(self.last_line_integral_projection, dtype=np.float32)
		if stage_name in {"raw", "detector", "detector_image"}:
			if self.last_raw_projection is None:
				raise ValueError("No cached raw detector projection is available.")
			return np.asarray(self.last_raw_projection, dtype=np.float32)
		raise ValueError("stage must be one of: 'raw', 'detector', 'line_integral'.")

	@staticmethod
	def _save_projection_array(array, path):
		"""Persist one 2D detector-space array to `.npy`, `.npz`, or a text-based file."""
		path = Path(path)
		array = np.asarray(array, dtype=np.float32)
		suffix = path.suffix.lower()
		if suffix == ".npy":
			np.save(path, array)
			return path
		if suffix == ".npz":
			np.savez_compressed(path, image=array)
			return path
		delimiter = "," if suffix == ".csv" else None
		if suffix in {".txt", ".csv", ".tsv"}:
			if suffix == ".tsv":
				delimiter = "\t"
			save_kwargs = {"fmt": "%.9g"}
			if delimiter is not None:
				save_kwargs["delimiter"] = delimiter
			np.savetxt(path, array, **save_kwargs)
			return path
		raise ValueError("Supported projection export formats are: .npy, .npz, .txt, .csv, .tsv.")

	@staticmethod
	def _load_projection_array(path):
		"""Load one 2D detector-space array from `.npy`, `.npz`, or a text-based file."""
		path = Path(path)
		suffix = path.suffix.lower()
		metadata = None
		if suffix == ".npy":
			array = np.load(path)
		elif suffix == ".npz":
			with np.load(path, allow_pickle=False) as archive:
				if "image" in archive:
					array = archive["image"]
				elif len(archive.files) == 1:
					array = archive[archive.files[0]]
				else:
					raise ValueError("Projection NPZ must contain an 'image' array.")
				if "metadata_json" in archive:
					metadata_raw = archive["metadata_json"]
					metadata = json.loads(str(metadata_raw.tolist() if hasattr(metadata_raw, "tolist") else metadata_raw))
		elif suffix in {".txt", ".csv", ".tsv"}:
			delimiter = "," if suffix == ".csv" else None
			if suffix == ".tsv":
				delimiter = "\t"
			array = np.loadtxt(path, delimiter=delimiter)
		else:
			raise ValueError("Supported projection import formats are: .npy, .npz, .txt, .csv, .tsv.")
		array = np.asarray(array, dtype=np.float32)
		if array.ndim != 2:
			raise ValueError("Imported projection arrays must be 2D.")
		return array, metadata

	def export_cached_projection(self, path, stage="raw"):
		"""Save one cached detector-space projection array to disk."""
		path = Path(path)
		array = self._resolve_projection_cache(stage)
		self._ensure_detector_image_defaults()
		if path.suffix.lower() != ".npz":
			return self._save_projection_array(array, path)
		from .detectorImage import DetectorImage

		detector_image = DetectorImage()
		detector_image.sync_from_virtual_xray(self, auto_window=False)
		if str(stage).strip().lower() in {"line", "line_integral", "integral"} and "composited_line_integral" in detector_image.package_images:
			detector_image.set_active_layer("composited_line_integral", auto_window=False)
		detector_image.export_array(path)
		return path

	def import_cached_projection(self, path, stage="raw"):
		"""Load one cached detector-space projection array from disk into this object."""
		self._ensure_projection_cache_defaults()
		path = Path(path)
		if path.suffix.lower() == ".npz":
			with np.load(path, allow_pickle=False) as archive:
				if "metadata_json" in archive:
					metadata_raw = archive["metadata_json"]
					metadata = json.loads(str(metadata_raw.tolist() if hasattr(metadata_raw, "tolist") else metadata_raw))
					if isinstance(metadata, dict) and str(metadata.get("schema", "")).strip() == "virtRTG-detector-package":
						from .detectorImage import DetectorImage

						detector_image = DetectorImage()
						detector_image.import_array(path, auto_window=False)
						active_layer = detector_image.active_layer_info()
						active_stage = str(stage).strip().lower() if active_layer is None else str(active_layer.get("stage", stage)).strip().lower()
						self.last_raw_projection = None
						self.last_line_integral_projection = None
						self.last_source_projections = []
						self.last_projected_annotations = detector_image.overlay_projection_set
						for layer in detector_image.package_layers:
							layer_array = detector_image.package_images.get(layer["key"], None)
							if layer_array is None:
								continue
							if layer["role"] == "composited":
								if layer["stage"] == "raw":
									self.last_raw_projection = np.asarray(layer_array, dtype=np.float32)
								elif layer["stage"] == "line_integral":
									self.last_line_integral_projection = np.asarray(layer_array, dtype=np.float32)
							elif layer["role"] == "per_source":
								source_index = int(layer.get("source_index", 0) or 0)
								existing = next((item for item in self.last_source_projections if int(item.source_index) == source_index), None)
								if existing is None:
									existing = XRaySourceProjection(
										source_index=source_index,
										label=str(layer.get("source_label", layer["label"])),
										source_type=str(layer.get("source_type", "unknown")),
										line_integral_image=np.zeros_like(layer_array, dtype=np.float32),
										detector_image=np.zeros_like(layer_array, dtype=np.float32),
									)
									self.last_source_projections.append(existing)
								if layer["stage"] == "raw":
									existing.detector_image = np.asarray(layer_array, dtype=np.float32)
								elif layer["stage"] == "line_integral":
									existing.line_integral_image = np.asarray(layer_array, dtype=np.float32)
						if self.last_raw_projection is None and self.last_line_integral_projection is not None:
							self.last_raw_projection = self._line_integral_to_detector_image(self.last_line_integral_projection)
						if active_stage in {"line", "line_integral", "integral"}:
							return np.asarray(self.last_line_integral_projection, dtype=np.float32)
						return np.asarray(self.last_raw_projection, dtype=np.float32)
		array, metadata = self._load_projection_array(path)
		import_stage = str(stage).strip().lower()
		if isinstance(metadata, dict):
			import_stage = str(metadata.get("stage", import_stage)).strip().lower()
		if import_stage in {"line", "line_integral", "integral"}:
			self.last_line_integral_projection = array
			self.last_raw_projection = self._line_integral_to_detector_image(array)
		elif import_stage in {"raw", "detector", "detector_image"}:
			self.last_raw_projection = array
			self.last_line_integral_projection = None
		else:
			raise ValueError("stage must be one of: 'raw', 'detector', 'line_integral'.")
		self.last_source_projections = []
		self.last_projected_annotations = None
		if isinstance(metadata, dict):
			self.last_projected_annotations = overlay_projection_set_from_payload(
				metadata.get("projected_annotations", None)
			)
		return array

	def export_cached_source_projections(self, directory, stage="raw", file_format=".npy"):
		"""Export one file per cached source contribution into the selected directory."""
		self._ensure_projection_cache_defaults()
		directory = Path(directory)
		directory.mkdir(parents=True, exist_ok=True)
		file_format = str(file_format).strip().lower()
		if not file_format.startswith("."):
			file_format = f".{file_format}"
		if file_format not in {".npz", ".npy", ".txt", ".csv", ".tsv"}:
			raise ValueError("file_format must be one of: .npz, .npy, .txt, .csv, .tsv.")
		stage_name = str(stage).strip().lower()
		exports = []
		for source_projection in self.last_source_projections:
			source_label = "".join(
				char if char.isalnum() or char in {"-", "_"} else "_"
				for char in str(source_projection.label)
			)
			if stage_name in {"line", "line_integral", "integral"}:
				array = source_projection.line_integral_image
			elif stage_name in {"raw", "detector", "detector_image"}:
				array = source_projection.detector_image
			else:
				raise ValueError("stage must be one of: 'raw', 'detector', 'line_integral'.")
			export_path = directory / f"{source_projection.source_index:03d}_{source_label}{file_format}"
			self._save_projection_array(array, export_path)
			exports.append(export_path)
		return exports

	def projection_cache_payload(self):
		"""Return one JSON-ready payload with the cached detector buffers, or `None`."""
		self._ensure_projection_cache_defaults()
		if self.last_raw_projection is None and self.last_line_integral_projection is None and not self.last_source_projections:
			return None
		payload = {
			"schema": "virtRTG-projection-cache",
			"version": 1,
			"raw_projection": (
				np.asarray(self.last_raw_projection, dtype=np.float32).copy()
				if self.last_raw_projection is not None else None
			),
			"line_integral_projection": (
				np.asarray(self.last_line_integral_projection, dtype=np.float32).copy()
				if self.last_line_integral_projection is not None else None
			),
			"source_projections": [],
		}
		for source_projection in self.last_source_projections:
			payload["source_projections"].append({
				"source_index": int(source_projection.source_index),
				"label": str(source_projection.label),
				"source_type": str(source_projection.source_type),
				"raw_projection": np.asarray(source_projection.detector_image, dtype=np.float32).copy(),
				"line_integral_projection": np.asarray(source_projection.line_integral_image, dtype=np.float32).copy(),
			})
		return payload

	def apply_projection_cache_payload(self, payload):
		"""Restore cached detector buffers from one serialized payload mapping."""
		self._ensure_projection_cache_defaults()
		payload = {} if payload is None else dict(payload)
		raw_projection = payload.get("raw_projection", None)
		line_integral_projection = payload.get("line_integral_projection", None)
		self.last_raw_projection = (
			None if raw_projection is None
			else np.asarray(raw_projection, dtype=np.float32)
		)
		self.last_line_integral_projection = (
			None if line_integral_projection is None
			else np.asarray(line_integral_projection, dtype=np.float32)
		)
		if self.last_raw_projection is None and self.last_line_integral_projection is not None:
			self.last_raw_projection = self._line_integral_to_detector_image(self.last_line_integral_projection)
		self.last_source_projections = []
		for source_payload in payload.get("source_projections", []):
			source_raw = source_payload.get("raw_projection", None)
			source_line = source_payload.get("line_integral_projection", None)
			source_raw_array = (
				None if source_raw is None
				else np.asarray(source_raw, dtype=np.float32)
			)
			source_line_array = (
				None if source_line is None
				else np.asarray(source_line, dtype=np.float32)
			)
			if source_raw_array is None and source_line_array is not None:
				source_raw_array = self._line_integral_to_detector_image(source_line_array)
			if source_raw_array is None or source_line_array is None:
				continue
			self.last_source_projections.append(XRaySourceProjection(
				source_index=int(source_payload.get("source_index", len(self.last_source_projections))),
				label=str(source_payload.get("label", f"source_{len(self.last_source_projections)}")),
				source_type=str(source_payload.get("source_type", "unknown")),
				line_integral_image=source_line_array,
				detector_image=source_raw_array,
			))
		self.last_projection_image = None

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
