# -*- coding: utf-8 -*-
"""World-space X-ray projection backend prepared for multiple scene source types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import logging
import os
from time import perf_counter
from typing import Iterable, Sequence

import cv2
import numpy as np
import pydicom
from scipy.ndimage import map_coordinates

from dpVision import Mesh, Volumetric
from .xraySource import XRaySampleSource, VolumetricXRaySource, MeshXRaySource

from .xrayPresentation import XRayPresentationModel
from .xrayHelpers import _normalize_vector, _transform_point, _transform_direction

_log = logging.getLogger(__name__)


def _ray_box_intersections_vectorized(ray_origins, ray_directions, box_min, box_max):
	"""Vectorized slab-based AABB intersection for many rays.

	This is a local implementation of the standard ray-box "slab" test,
	not copied from one external project. See
	`docs/THIRD_PARTY_ATTRIBUTION.md` in this plugin for the
	algorithm-level attribution note.
	"""
	ray_origins = np.asarray(ray_origins, dtype=np.float32)
	ray_directions = np.asarray(ray_directions, dtype=np.float32)
	box_min = np.asarray(box_min, dtype=np.float32)
	box_max = np.asarray(box_max, dtype=np.float32)

	n = ray_origins.shape[0]
	t_start = np.full(n, -np.inf, dtype=np.float32)
	t_end = np.full(n, np.inf, dtype=np.float32)
	hit_mask = np.ones(n, dtype=bool)

	for axis in range(3):
		d = ray_directions[:, axis]
		o = ray_origins[:, axis]
		parallel = np.abs(d) <= 1e-8
		nonparallel = ~parallel
		hit_mask &= ~(parallel & ((o < box_min[axis]) | (o > box_max[axis])))
		safe_d = np.where(nonparallel, d, 1.0)
		inv_d = np.where(nonparallel, 1.0 / safe_d, 0.0)
		t0 = (box_min[axis] - o) * inv_d
		t1 = (box_max[axis] - o) * inv_d
		t_near = np.where(nonparallel, np.minimum(t0, t1), -np.inf)
		t_far = np.where(nonparallel, np.maximum(t0, t1), np.inf)
		t_start = np.maximum(t_start, t_near)
		t_end = np.minimum(t_end, t_far)
		hit_mask &= t_end >= t_start

	return t_start, t_end, hit_mask



def _ray_planar_slab_intersections_vectorized(ray_origins, ray_directions, slab_origin, slab_axis, slab_min, slab_max):
	"""Intersect N rays with one infinite planar slab defined in axis coordinates."""
	ray_origins = np.asarray(ray_origins, dtype=np.float32)
	ray_directions = np.asarray(ray_directions, dtype=np.float32)
	slab_origin = np.asarray(slab_origin, dtype=np.float32)
	slab_axis = _normalize_vector(slab_axis)
	slab_min = float(slab_min)
	slab_max = float(slab_max)

	n = ray_origins.shape[0]
	t_start = np.full(n, -np.inf, dtype=np.float32)
	t_end = np.full(n, np.inf, dtype=np.float32)
	hit_mask = np.ones(n, dtype=bool)

	offsets = ray_origins - slab_origin[np.newaxis, :]
	s0 = np.sum(offsets * slab_axis[np.newaxis, :], axis=1)
	ds = np.sum(ray_directions * slab_axis[np.newaxis, :], axis=1)
	parallel = np.abs(ds) <= 1e-8
	inside_parallel = parallel & (s0 >= slab_min) & (s0 <= slab_max)
	hit_mask &= (~parallel) | inside_parallel

	nonparallel = ~parallel
	if np.any(nonparallel):
		ds_safe = ds[nonparallel]
		t0 = (slab_min - s0[nonparallel]) / ds_safe
		t1 = (slab_max - s0[nonparallel]) / ds_safe
		t_near = np.minimum(t0, t1)
		t_far = np.maximum(t0, t1)
		t_start[nonparallel] = t_near.astype(np.float32)
		t_end[nonparallel] = t_far.astype(np.float32)

	hit_mask &= t_end >= t_start
	return t_start, t_end, hit_mask

def _identity_matrix():
	"""Return a 4x4 identity matrix used as a default homogeneous transform."""
	return np.eye(4, dtype=np.float32)


@dataclass
class XRayScalarPreprocessor:
	"""Optionally normalize source scalar values before the physics model consumes them."""

	mode: str = "none"
	input_low_percentile: float = 0.5
	input_high_percentile: float = 99.5
	output_low_value: float = -1000.0
	output_high_value: float = 2500.0

	def is_active(self):
		"""Return `True` when this preprocessor should modify source scalar values."""
		return str(self.mode).lower() != "none"

	def estimate_volume_stats(self, volume):
		"""Estimate robust scalar range statistics from one source volume."""
		volume = np.asarray(volume, dtype=np.float32)
		finite_values = volume[np.isfinite(volume)]
		if finite_values.size == 0:
			return None

		low_q = float(np.percentile(finite_values, np.clip(float(self.input_low_percentile), 0.0, 100.0)))
		high_q = float(np.percentile(finite_values, np.clip(float(self.input_high_percentile), 0.0, 100.0)))
		if high_q <= low_q:
			high_q = low_q + 1.0
		return {
			"input_low": low_q,
			"input_high": high_q,
		}

	def apply(self, scalar_values, stats):
		"""Return optionally normalized scalar values using previously estimated source statistics."""
		scalar_values = np.asarray(scalar_values, dtype=np.float32)
		if not self.is_active() or stats is None:
			return scalar_values

		mode = str(self.mode).lower()
		if mode == "percentile_rescale":
			input_low = float(stats["input_low"])
			input_high = float(stats["input_high"])
			output_low = float(self.output_low_value)
			output_high = float(self.output_high_value)
			normalized = np.clip((scalar_values - input_low) / max(input_high - input_low, 1e-6), 0.0, 1.0)
			return (output_low + normalized * (output_high - output_low)).astype(np.float32, copy=False)

		return scalar_values


@dataclass
class XRayProjectionGeometry:
	"""Describe the source-detector setup in a configurable reference frame."""

	detector_origin_ref: Sequence[float]
	detector_u_ref: Sequence[float]
	detector_v_ref: Sequence[float]
	detector_shape_hw: Sequence[int]
	step_mm: float = 1.0
	source_position_ref: Sequence[float] | None = None
	ray_direction_ref: Sequence[float] | None = None
	depth_window_mode: str | None = None
	depth_window_mm: Sequence[float] | None = None
	depth_window_origin_ref: Sequence[float] | None = None
	depth_window_axis_ref: Sequence[float] | None = None

	@classmethod
	def from_detector_pose(
		cls,
		detector_center_ref,
		detector_normal_ref,
		detector_up_ref,
		detector_shape_hw,
		detector_pixel_size_mm=None,
		detector_size_mm_hw=None,
		step_mm=1.0,
		source_position_ref=None,
		ray_direction_ref=None,
		depth_window_mode=None,
		depth_window_mm=None,
		depth_window_origin_ref=None,
		depth_window_axis_ref=None,
	):
		"""Build geometry from a detector center, orientation and either size or pixel spacing."""
		height = int(detector_shape_hw[0])
		width = int(detector_shape_hw[1])
		if height <= 0 or width <= 0:
			raise ValueError("detector_shape_hw must contain positive height and width.")

		normal = _normalize_vector(detector_normal_ref)
		up = np.asarray(detector_up_ref, dtype=np.float32)
		up = up - normal * float(np.dot(up, normal))
		up = _normalize_vector(up)
		u_axis = _normalize_vector(np.cross(up, normal))
		v_axis = up

		if detector_pixel_size_mm is not None and detector_size_mm_hw is not None:
			raise ValueError("Provide either detector_pixel_size_mm or detector_size_mm_hw, not both.")
		if detector_pixel_size_mm is None and detector_size_mm_hw is None:
			raise ValueError("Either detector_pixel_size_mm or detector_size_mm_hw must be provided.")

		if detector_size_mm_hw is not None:
			size_h = float(detector_size_mm_hw[0])
			size_w = float(detector_size_mm_hw[1])
			if size_h <= 0.0 or size_w <= 0.0:
				raise ValueError("detector_size_mm_hw must contain positive physical dimensions.")
			pixel_size_v = size_h / float(height)
			pixel_size_u = size_w / float(width)
		else:
			if np.isscalar(detector_pixel_size_mm):
				pixel_size_u = float(detector_pixel_size_mm)
				pixel_size_v = float(detector_pixel_size_mm)
			else:
				pixel_size_u = float(detector_pixel_size_mm[0])
				pixel_size_v = float(detector_pixel_size_mm[1])
			if pixel_size_u <= 0.0 or pixel_size_v <= 0.0:
				raise ValueError("detector_pixel_size_mm must be positive.")

		detector_u_ref = u_axis * pixel_size_u
		detector_v_ref = v_axis * pixel_size_v
		detector_center_ref = np.asarray(detector_center_ref, dtype=np.float32)
		detector_origin_ref = (
			detector_center_ref
			- detector_u_ref * (float(width - 1) / 2.0)
			- detector_v_ref * (float(height - 1) / 2.0)
		)
		return cls(
			detector_origin_ref=detector_origin_ref.astype(np.float32),
			detector_u_ref=detector_u_ref.astype(np.float32),
			detector_v_ref=detector_v_ref.astype(np.float32),
			detector_shape_hw=[height, width],
			step_mm=float(step_mm),
			source_position_ref=source_position_ref,
			ray_direction_ref=ray_direction_ref,
			depth_window_mode=depth_window_mode,
			depth_window_mm=depth_window_mm,
			depth_window_origin_ref=depth_window_origin_ref,
			depth_window_axis_ref=depth_window_axis_ref,
		)

	def detector_pixel_size_mm_uv(self):
		"""Return detector pixel pitch along the `u` and `v` detector axes."""
		return (
			float(np.linalg.norm(np.asarray(self.detector_u_ref, dtype=np.float32))),
			float(np.linalg.norm(np.asarray(self.detector_v_ref, dtype=np.float32))),
		)

	def detector_size_mm_hw(self):
		"""Return the physical detector size in millimeters as `(height_mm, width_mm)`."""
		height, width = int(self.detector_shape_hw[0]), int(self.detector_shape_hw[1])
		pixel_size_u, pixel_size_v = self.detector_pixel_size_mm_uv()
		return float(height) * pixel_size_v, float(width) * pixel_size_u

	def detector_center_ref_point(self):
		"""Return the detector center point expressed in the geometry reference frame."""
		height, width = int(self.detector_shape_hw[0]), int(self.detector_shape_hw[1])
		return (
			np.asarray(self.detector_origin_ref, dtype=np.float32)
			+ np.asarray(self.detector_u_ref, dtype=np.float32) * (float(width - 1) / 2.0)
			+ np.asarray(self.detector_v_ref, dtype=np.float32) * (float(height - 1) / 2.0)
		).astype(np.float32)

	def detector_normal_ref_vector(self):
		"""Return the detector plane normal vector derived from the `u` and `v` axes."""
		return _normalize_vector(np.cross(
			np.asarray(self.detector_u_ref, dtype=np.float32),
			np.asarray(self.detector_v_ref, dtype=np.float32),
		))

	def is_cone_beam(self):
		"""Return `True` when the geometry is driven by a point source."""
		return self.source_position_ref is not None

	def is_parallel_beam(self):
		"""Return `True` when the geometry uses one shared ray direction."""
		return self.ray_direction_ref is not None

	def has_depth_window(self):
		"""Return `True` when the geometry requests depth-limited integration."""
		return self.depth_window_mode is not None and self.depth_window_mm is not None

	def normalized_depth_window_mode(self):
		"""Return the normalized depth-window mode string or `None`."""
		if self.depth_window_mode is None:
			return None
		mode = str(self.depth_window_mode).strip().lower()
		return mode or None

	def depth_window_limits_mm(self):
		"""Return sorted depth-window limits in millimetres or `None` when disabled."""
		if self.depth_window_mm is None:
			return None
		if len(self.depth_window_mm) != 2:
			raise ValueError("depth_window_mm must contain exactly two values.")
		depth_from = float(self.depth_window_mm[0])
		depth_to = float(self.depth_window_mm[1])
		return (min(depth_from, depth_to), max(depth_from, depth_to))

	def effective_depth_window_origin_ref(self):
		"""Return the planar depth-window origin or a detector-centred default."""
		if self.depth_window_origin_ref is not None:
			return np.asarray(self.depth_window_origin_ref, dtype=np.float32)
		return self.detector_center_ref_point()

	def effective_depth_window_axis_ref(self):
		"""Return the planar depth-window axis or the detector normal by default."""
		if self.depth_window_axis_ref is not None:
			return _normalize_vector(self.depth_window_axis_ref)
		return self.detector_normal_ref_vector()

	def with_quality_profile(self, quality_profile):
		"""Return a geometry copy modified by a quality profile."""
		if quality_profile is None:
			return replace(self)
		return quality_profile.apply_to_geometry(self)

	def validate(self):
		"""Validate geometry fields before projection."""
		height, width = int(self.detector_shape_hw[0]), int(self.detector_shape_hw[1])
		if height <= 0 or width <= 0:
			raise ValueError("detector_shape_hw must contain positive height and width.")
		if float(self.step_mm) <= 0.0:
			raise ValueError("step_mm must be positive.")
		if self.source_position_ref is None and self.ray_direction_ref is None:
			raise ValueError("Either source_position_ref or ray_direction_ref must be provided.")
		if self.source_position_ref is not None and self.ray_direction_ref is not None:
			raise ValueError("Geometry must define either source_position_ref or ray_direction_ref, not both.")
		if np.linalg.norm(np.asarray(self.detector_u_ref, dtype=np.float32)) <= 1e-8:
			raise ValueError("detector_u_ref must have a non-zero length.")
		if np.linalg.norm(np.asarray(self.detector_v_ref, dtype=np.float32)) <= 1e-8:
			raise ValueError("detector_v_ref must have a non-zero length.")
		depth_mode = self.normalized_depth_window_mode()
		depth_limits = self.depth_window_limits_mm()
		if depth_mode is None:
			if depth_limits is not None:
				raise ValueError("depth_window_mm requires depth_window_mode to be enabled.")
			return
		if depth_limits is None:
			raise ValueError("depth_window_mode requires depth_window_mm to be provided.")
		if depth_mode not in {"ray", "planar"}:
			raise ValueError("depth_window_mode must be one of: 'ray', 'planar'.")
		if depth_mode == "planar":
			if np.linalg.norm(np.asarray(self.effective_depth_window_axis_ref(), dtype=np.float32)) <= 1e-8:
				raise ValueError("depth_window_axis_ref must have a non-zero length for planar depth windows.")


@dataclass
class XRayPhysicsModel:
	"""Map scalar values to attenuation and convert integrals into detector intensities.

	The intensity conversion follows the Beer-Lambert attenuation law in its
	simplified monochromatic form. The scalar-to-attenuation mapping around it
	remains project-specific and intentionally heuristic.
	"""

	mu_air: float = 0.0
	mu_water: float = 0.02
	hounsfield_air: float = -1000.0
	attenuation_scale: float = 1.0
	source_energy_kev: float = 70.0
	reference_energy_kev: float = 70.0
	attenuation_energy_exponent: float = 2.0
	output_mode: str = "integral"
	intensity_floor: float = 0.0
	source_distance_falloff_mode: str = "none"
	source_distance_reference_mm: float | None = None
	source_distance_power: float = 2.0
	material_response_mode: str = "linear"
	bone_threshold_hu: float | None = None
	bone_threshold_softness: float = 250.0
	material_window_center: float | None = None
	material_window_width: float | None = None
	material_window_mode: str = "hard"
	material_window_softness: float = 150.0

	def attenuation_energy_scale(self):
		"""Return one heuristic scale factor that decreases attenuation for higher source energy."""
		source_energy = max(1e-6, float(self.source_energy_kev))
		reference_energy = max(1e-6, float(self.reference_energy_kev))
		exponent = max(0.0, float(self.attenuation_energy_exponent))
		return float((reference_energy / source_energy) ** exponent)

	def scalar_to_mu(self, scalar_values):
		"""Convert scalar CT-like values into a linear attenuation coefficient."""
		scalar_values = np.asarray(scalar_values, dtype=np.float32)
		energy_scale = self.attenuation_energy_scale()
		mode = str(self.material_response_mode).lower()
		if mode == "piecewise_bone":
			mu = self._scalar_to_mu_piecewise_bone(scalar_values, energy_scale=energy_scale)
		elif mode == "piecewise_soft_tissue":
			mu = self._scalar_to_mu_piecewise_soft_tissue(scalar_values, energy_scale=energy_scale)
		elif mode == "bone_threshold":
			mu = self._scalar_to_mu_bone_threshold(scalar_values, energy_scale=energy_scale)
		else:
			relative_density = np.maximum(0.0, 1.0 + scalar_values / abs(float(self.hounsfield_air)))
			mu = (float(self.mu_air) + float(self.mu_water) * relative_density) * float(self.attenuation_scale) * energy_scale
		if self.material_window_center is not None and self.material_window_width is not None and float(self.material_window_width) > 0.0:
			vmin = float(self.material_window_center) - float(self.material_window_width) / 2.0
			vmax = float(self.material_window_center) + float(self.material_window_width) / 2.0
			mode = str(self.material_window_mode).lower()
			softness = max(1e-6, float(self.material_window_softness))
			if mode == "linear":
				lower = np.clip((scalar_values - (vmin - softness)) / softness, 0.0, 1.0)
				upper = np.clip(((vmax + softness) - scalar_values) / softness, 0.0, 1.0)
				weight = lower * upper
			elif mode == "sigmoid":
				lower = 1.0 / (1.0 + np.exp(-(scalar_values - vmin) / softness))
				upper = 1.0 / (1.0 + np.exp((scalar_values - vmax) / softness))
				weight = lower * upper
			else:
				weight = ((scalar_values >= vmin) & (scalar_values <= vmax)).astype(np.float32)
			mu = mu * weight.astype(np.float32, copy=False)
		return mu

	def _piecewise_linear_map(self, scalar_values, control_points):
		"""Map scalar values with a piecewise-linear attenuation curve."""
		xp = np.asarray([point[0] for point in control_points], dtype=np.float32)
		fp = np.asarray([point[1] for point in control_points], dtype=np.float32)
		return np.interp(scalar_values, xp, fp, left=fp[0], right=fp[-1]).astype(np.float32, copy=False)

	def _scalar_to_mu_piecewise_bone(self, scalar_values, energy_scale=1.0):
		"""Return one bone-emphasis attenuation curve tuned for craniofacial structures."""
		base = float(self.mu_water) * float(self.attenuation_scale) * float(energy_scale)
		control_points = [
			(-1000.0, float(self.mu_air)),
			(-300.0, 0.03 * base),
			(0.0, 0.10 * base),
			(150.0, 0.18 * base),
			(400.0, 0.35 * base),
			(800.0, 0.75 * base),
			(1200.0, 1.20 * base),
			(2000.0, 1.85 * base),
			(3000.0, 2.30 * base),
			(4000.0, 2.60 * base),
		]
		return self._piecewise_linear_map(scalar_values, control_points)

	def _scalar_to_mu_piecewise_soft_tissue(self, scalar_values, energy_scale=1.0):
		"""Return one soft-tissue-oriented attenuation curve with reduced bone dominance."""
		base = float(self.mu_water) * float(self.attenuation_scale) * float(energy_scale)
		control_points = [
			(-1000.0, float(self.mu_air)),
			(-300.0, 0.05 * base),
			(0.0, 0.45 * base),
			(80.0, 0.70 * base),
			(200.0, 0.85 * base),
			(500.0, 1.05 * base),
			(1000.0, 1.20 * base),
			(2000.0, 1.35 * base),
			(4000.0, 1.55 * base),
		]
		return self._piecewise_linear_map(scalar_values, control_points)

	def _scalar_to_mu_bone_threshold(self, scalar_values, energy_scale=1.0):
		"""Blend one neutral attenuation model with a bone-emphasis model above an HU threshold."""
		relative_density = np.maximum(0.0, 1.0 + scalar_values / abs(float(self.hounsfield_air)))
		linear_mu = (float(self.mu_air) + float(self.mu_water) * relative_density) * float(self.attenuation_scale) * float(energy_scale)
		bone_mu = self._scalar_to_mu_piecewise_bone(scalar_values, energy_scale=energy_scale)
		threshold = 350.0 if self.bone_threshold_hu is None else float(self.bone_threshold_hu)
		softness = max(1e-6, float(self.bone_threshold_softness))
		weight = 1.0 / (1.0 + np.exp(-(scalar_values - threshold) / softness))
		soft_tissue_mix = 0.85
		return (
			linear_mu * (1.0 - soft_tissue_mix * weight)
			+ bone_mu * weight
		).astype(np.float32, copy=False)

	def source_distance_gain(self, source_to_detector_distance_mm, projection_mode=None):
		"""Return one multiplicative detector gain caused by distance from the source."""
		if source_to_detector_distance_mm is None:
			return None
		if projection_mode is not None and str(projection_mode).lower() != "cone":
			return None
		mode = str(self.source_distance_falloff_mode).strip().lower()
		if mode in {"", "none", "off"}:
			return None
		if mode != "inverse_square":
			raise ValueError("source_distance_falloff_mode must be one of: 'none', 'inverse_square'.")
		distances = np.asarray(source_to_detector_distance_mm, dtype=np.float32)
		reference_distance = self.source_distance_reference_mm
		if reference_distance is None or float(reference_distance) <= 0.0:
			finite_distances = distances[np.isfinite(distances) & (distances > 0.0)]
			if finite_distances.size == 0:
				return None
			reference_distance = float(np.median(finite_distances))
		reference_distance = max(1e-6, float(reference_distance))
		power = max(0.0, float(self.source_distance_power))
		safe_distances = np.maximum(distances, 1e-6)
		return np.power(reference_distance / safe_distances, power).astype(np.float32, copy=False)

	def integral_to_image(self, line_integral, source_to_detector_distance_mm=None, projection_mode=None):
		"""Convert integrated attenuation into a detector-space image value.

		`output_mode == "intensity"` applies a Beer-Lambert style
		`I = exp(-integral)` conversion, optionally followed by a cone-beam
		distance gain term.
		"""
		line_integral = np.asarray(line_integral, dtype=np.float32)
		distance_gain = self.source_distance_gain(
			source_to_detector_distance_mm=source_to_detector_distance_mm,
			projection_mode=projection_mode,
		)
		mode = str(self.output_mode).lower()
		if mode == "integral":
			if distance_gain is None:
				return line_integral
			safe_gain = np.maximum(distance_gain, 1e-12)
			return line_integral - np.log(safe_gain)
		intensity = np.exp(-line_integral)
		if distance_gain is not None:
			intensity = intensity * distance_gain
		return np.maximum(float(self.intensity_floor), intensity)


@dataclass
class XRayProjectionQualityProfile:
	"""Describe a reusable quality preset that modifies geometry sampling density."""

	name: str
	step_mm: float | None = None
	detector_downsample: int = 1

	@classmethod
	def draft(cls):
		"""Return a low-cost profile useful for quick geometry debugging."""
		return cls(name="draft", step_mm=2.0, detector_downsample=2)

	@classmethod
	def normal(cls):
		"""Return a balanced profile for standard interactive work."""
		return cls(name="normal", step_mm=1.0, detector_downsample=1)

	@classmethod
	def high(cls):
		"""Return a high-quality profile prioritizing detail over runtime."""
		return cls(name="high", step_mm=0.5, detector_downsample=1)

	def apply_to_geometry(self, geometry):
		"""Return a geometry copy with detector sampling and ray step adapted to this profile."""
		geometry = replace(geometry)
		downsample = max(1, int(self.detector_downsample))
		if downsample > 1:
			height = max(1, int(np.ceil(int(geometry.detector_shape_hw[0]) / float(downsample))))
			width = max(1, int(np.ceil(int(geometry.detector_shape_hw[1]) / float(downsample))))
			geometry = replace(
				geometry,
				detector_shape_hw=[height, width],
				detector_u_ref=np.asarray(geometry.detector_u_ref, dtype=np.float32) * float(downsample),
				detector_v_ref=np.asarray(geometry.detector_v_ref, dtype=np.float32) * float(downsample),
			)
		if self.step_mm is not None:
			geometry = replace(geometry, step_mm=float(self.step_mm))
		return geometry


@dataclass
class XRayProjectionConfig:
	"""Bundle projection geometry, physics and optional presentation into one scenario."""

	geometry: XRayProjectionGeometry
	physics_model: XRayPhysicsModel
	presentation_model: XRayPresentationModel | None = None
	reference_transform: np.ndarray = field(default_factory=_identity_matrix)
	quality_profile: XRayProjectionQualityProfile | None = None

	def effective_geometry(self):
		"""Return the geometry after applying any quality profile override."""
		return self.geometry.with_quality_profile(self.quality_profile)

	def apply_presentation(self, image):
		"""Return either the raw image or a presentation-mapped view of it."""
		if self.presentation_model is None:
			return np.asarray(image, dtype=np.float32)
		return self.presentation_model.apply(image)


@dataclass
class XRayProjectionStats:
	"""Collect basic runtime and workload statistics for one projection call."""

	elapsed_seconds: float
	total_pixels: int
	traced_pixels: int
	total_sample_count: int
	source_count: int
	step_mm: float
	projection_mode: str
	detector_shape_hw: tuple[int, int]
	depth_window_mode: str | None = None
	# Per-phase wall-clock timing breakdown (phase_name -> seconds).
	# Keys: "ray_setup", "aabb_intersection", "depth_clipping",
	#       "direct_sources_total", "marching_total", "physics_conversion".
	phase_timings: dict = field(default_factory=dict)
	# Per-source statistics — one dict per XRaySampleSource in scene order.
	# Volumetric keys: "label", "source_type", "elapsed_s", "work_count",
	#   "volume_shape", "interpolation".
	# Mesh keys additionally: "backend", "mode", "triangle_count",
	#   "bvh_build_s" (analytic_bvh only), "stack_build_s", "stack_uv_projection_s",
	#   "stack_rasterize_s", "stack_csr_s", "integration_s".
	per_source_stats: list = field(default_factory=list)

	@property
	def average_samples_per_traced_pixel(self):
		"""Return the average sample count over rays that crossed the scene bounds."""
		if self.traced_pixels <= 0:
			return 0.0
		return float(self.total_sample_count) / float(self.traced_pixels)

	@property
	def samples_per_second(self):
		"""Return the effective throughput in attenuation samples per second."""
		if self.elapsed_seconds <= 1e-9:
			return 0.0
		return float(self.total_sample_count) / float(self.elapsed_seconds)

	@property
	def rays_per_second(self):
		"""Return the number of traced rays (pixels that hit the scene AABB) per second."""
		if self.elapsed_seconds <= 1e-9:
			return 0.0
		return float(self.traced_pixels) / float(self.elapsed_seconds)

	def format_report(self):
		"""Return a formatted multi-line performance report suitable for building tables."""
		lines = []
		h, w = self.detector_shape_hw
		lines.append("=== XRay Projection Performance Report ===")
		lines.append(f"  Detector:          {w} x {h} px  ({self.total_pixels:,} total)")
		lines.append(f"  Traced pixels:     {self.traced_pixels:,}  ({100.0 * self.traced_pixels / max(self.total_pixels, 1):.1f} %)")
		lines.append(f"  Step:              {self.step_mm:.2f} mm")
		lines.append(f"  Projection mode:   {self.projection_mode}")
		lines.append(f"  Sources:           {self.source_count}")
		lines.append(f"  Total samples:     {self.total_sample_count:,}")
		lines.append(f"  Samples/s:         {self.samples_per_second:,.0f}")
		lines.append(f"  Avg samples/ray:   {self.average_samples_per_traced_pixel:.1f}")
		lines.append(f"  Total elapsed:     {self.elapsed_seconds * 1000:.1f} ms")
		if self.phase_timings:
			lines.append("  --- Phase breakdown ---")
			phase_total = sum(self.phase_timings.values())
			for phase_name, t_s in self.phase_timings.items():
				pct = 100.0 * t_s / max(phase_total, 1e-9)
				lines.append(f"    {phase_name:<30s} {t_s * 1000:8.1f} ms  ({pct:.1f} %)")
		if self.per_source_stats:
			lines.append("  --- Per-source breakdown ---")
			for i, src in enumerate(self.per_source_stats):
				label = src.get("label", f"source_{i}")
				stype = src.get("source_type", "?")
				t_s   = src.get("elapsed_s", 0.0)
				wc    = src.get("work_count", 0)
				lines.append(f"    [{i}] {stype:<24s} '{label}'  {t_s * 1000:.1f} ms  work={wc:,}")
				if src.get("backend") is not None:
					lines.append(f"         backend={src['backend']}  mode={src.get('mode', '?')}  triangles={src.get('triangle_count', '?'):,}")
				if src.get("volume_shape") is not None:
					lines.append(f"         volume_shape={src['volume_shape']}  interpolation={src.get('interpolation', '?')}")
				for sub_key in ("bvh_build_s", "stack_build_s", "stack_uv_projection_s",
				                "stack_rasterize_s", "stack_csr_s", "integration_s"):
					if sub_key in src:
						lines.append(f"         {sub_key:<28s} {src[sub_key] * 1000:.1f} ms")
		lines.append("==========================================")
		return "\n".join(lines)

	def print_report(self):
		"""Print the formatted performance report to stdout."""
		print(self.format_report())



@dataclass
class XRayScene:
	"""Own the set of sources taking part in one X-ray acquisition scenario."""

	sample_sources: list[XRaySampleSource]

	@classmethod
	def from_sample_sources(cls, sample_sources):
		"""Build a scene from already prepared X-ray sample sources."""
		return cls(sample_sources=list(sample_sources))

	@classmethod
	def from_volumetrics(cls, volumetric_entries):
		"""Build a scene directly from volumetric objects and optional transforms."""
		sample_sources = []
		for entry in volumetric_entries:
			if isinstance(entry, VolumetricXRaySource):
				sample_sources.append(entry)
				continue
			if isinstance(entry, Volumetric):
				sample_sources.append(VolumetricXRaySource(entry))
				continue
			if not isinstance(entry, Sequence) or len(entry) == 0:
				raise TypeError("Each volumetric entry must be a Volumetric, VolumetricXRaySource or a configuration tuple.")
			volumetric = entry[0]
			global_transform = entry[1] if len(entry) > 1 else None
			interpolation = entry[2] if len(entry) > 2 else "linear"
			fill_value = entry[3] if len(entry) > 3 else None
			sample_sources.append(VolumetricXRaySource(
				volumetric=volumetric,
				global_transform=global_transform,
				interpolation=interpolation,
				fill_value=fill_value,
			))
		return cls(sample_sources=sample_sources)

	def build_projector(self):
		"""Return a projector bound to the sources stored in this scene."""
		return XRayProjector(self.sample_sources)

	def project(self, config, return_stats=False, progress_callback=None):
		"""Project the scene using a single combined configuration object."""
		return self.build_projector().project_config(config=config, return_stats=return_stats, progress_callback=progress_callback)

	def render(self, config, return_stats=False, progress_callback=None):
		"""Project the scene and optionally apply the configured presentation model."""
		if return_stats:
			raw_image, stats = self.project(config=config, return_stats=True, progress_callback=progress_callback)
			return config.apply_presentation(raw_image), stats
		return config.apply_presentation(self.project(config=config, return_stats=False, progress_callback=progress_callback))



class XRayProjector:
	"""Project one or more world-space attenuation sources onto a detector plane."""

	def __init__(self, sample_sources: Iterable[XRaySampleSource]):
		"""Store the list of sample sources used during projection."""
		self.sample_sources = list(sample_sources)
		if not self.sample_sources:
			raise ValueError("sample_sources must contain at least one X-ray source.")

	def scene_bounds_world(self):
		"""Return the world-space AABB covering every source registered in the projector."""
		mins = []
		maxs = []
		for source in self.sample_sources:
			source_min, source_max = source.bounds_world()
			mins.append(np.asarray(source_min, dtype=np.float32))
			maxs.append(np.asarray(source_max, dtype=np.float32))
		return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)

	def _detector_pixel_world(self, geometry, reference_transform, row_idx, col_idx):
		"""Return one detector pixel center in world coordinates."""
		detector_origin_world = _transform_point(reference_transform, geometry.detector_origin_ref)
		detector_u_world = _transform_direction(reference_transform, geometry.detector_u_ref)
		detector_v_world = _transform_direction(reference_transform, geometry.detector_v_ref)
		return (
			detector_origin_world
			+ detector_u_world * float(col_idx)
			+ detector_v_world * float(row_idx)
		)

	def _ray_definition_world(self, geometry, reference_transform, detector_point_world):
		"""Return one ray origin and direction in world coordinates."""
		if geometry.source_position_ref is not None:
			ray_origin_world = _transform_point(reference_transform, geometry.source_position_ref)
			ray_direction_world = _normalize_vector(detector_point_world - ray_origin_world)
			return ray_origin_world, ray_direction_world

		ray_origin_world = detector_point_world
		ray_direction_world = _normalize_vector(_transform_direction(reference_transform, geometry.ray_direction_ref))
		return ray_origin_world, ray_direction_world

	def project(self, geometry, physics_model, reference_transform=None, return_stats=False, progress_callback=None):
		"""Project all sample sources through the provided X-ray geometry (vectorized slab marching)."""
		geometry.validate()
		reference_transform = np.eye(4, dtype=np.float32) if reference_transform is None else np.asarray(reference_transform, dtype=np.float32)
		if reference_transform.shape != (4, 4):
			raise ValueError("reference_transform must be a 4x4 homogeneous matrix.")

		scene_min_world, scene_max_world = self.scene_bounds_world()
		height, width = int(geometry.detector_shape_hw[0]), int(geometry.detector_shape_hw[1])
		n_pixels = height * width
		step_mm = float(geometry.step_mm)
		start_time = perf_counter()
		_phase_timings: dict = {}
		_per_source_stats: list = []

		# Build all detector pixel centers (H*W, 3)
		detector_origin_world = _transform_point(reference_transform, geometry.detector_origin_ref).astype(np.float32)
		detector_u_world = _transform_direction(reference_transform, geometry.detector_u_ref).astype(np.float32)
		detector_v_world = _transform_direction(reference_transform, geometry.detector_v_ref).astype(np.float32)
		col_grid, row_grid = np.meshgrid(
			np.arange(width, dtype=np.float32),
			np.arange(height, dtype=np.float32),
		)
		pixel_centers = (
			detector_origin_world
			+ detector_u_world * col_grid[:, :, np.newaxis]
			+ detector_v_world * row_grid[:, :, np.newaxis]
		).reshape(n_pixels, 3)

		# Build per-ray origins and normalized directions (H*W, 3)
		if geometry.source_position_ref is not None:
			source_world = _transform_point(reference_transform, geometry.source_position_ref).astype(np.float32)
			ray_origins = np.empty((n_pixels, 3), dtype=np.float32)
			ray_origins[:] = source_world
			raw_dirs = pixel_centers - source_world
			source_to_detector_distance_mm = np.linalg.norm(raw_dirs, axis=1).astype(np.float32)
			norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
			ray_directions = raw_dirs / np.maximum(norms, 1e-8)
		else:
			ray_dir_world = _normalize_vector(
				_transform_direction(reference_transform, geometry.ray_direction_ref)
			).astype(np.float32)
			ray_origins = pixel_centers
			ray_directions = np.empty((n_pixels, 3), dtype=np.float32)
			ray_directions[:] = ray_dir_world
			source_to_detector_distance_mm = None

		_phase_timings["ray_setup"] = float(perf_counter() - start_time)

		# Vectorized AABB intersection for all rays
		_t_aabb_start = perf_counter()
		t_starts, t_ends, hit_mask = _ray_box_intersections_vectorized(
			ray_origins, ray_directions, scene_min_world, scene_max_world,
		)
		t_starts = np.maximum(t_starts, 0.0)
		hit_mask &= t_ends > t_starts
		_phase_timings["aabb_intersection"] = float(perf_counter() - _t_aabb_start)

		_t_depth_start = perf_counter()
		depth_mode = geometry.normalized_depth_window_mode()
		depth_limits = geometry.depth_window_limits_mm()
		if depth_mode == "ray" and depth_limits is not None:
			depth_start, depth_end = depth_limits
			t_starts = np.maximum(t_starts, depth_start)
			t_ends = np.minimum(t_ends, depth_end)
			hit_mask &= t_ends > t_starts
		elif depth_mode == "planar" and depth_limits is not None:
			depth_origin_world = _transform_point(
				reference_transform,
				geometry.effective_depth_window_origin_ref(),
			).astype(np.float32)
			depth_axis_world = _normalize_vector(_transform_direction(
				reference_transform,
				geometry.effective_depth_window_axis_ref(),
			)).astype(np.float32)
			depth_start, depth_end = depth_limits
			slab_t_starts, slab_t_ends, slab_hit_mask = _ray_planar_slab_intersections_vectorized(
				ray_origins,
				ray_directions,
				depth_origin_world,
				depth_axis_world,
				depth_start,
				depth_end,
			)
			t_starts = np.maximum(t_starts, slab_t_starts)
			t_ends = np.minimum(t_ends, slab_t_ends)
			hit_mask &= slab_hit_mask & (t_ends > t_starts)

		# Slab marching: one Python iteration per depth step, all active rays batched
		_phase_timings["depth_clipping"] = float(perf_counter() - _t_depth_start)
		projection_flat = np.zeros(n_pixels, dtype=np.float32)
		total_sample_count = 0
		traced_pixels = int(np.sum(hit_mask))

		if traced_pixels > 0:
			hit_indices = np.where(hit_mask)[0]
			marched_sources = []
			direct_sources = []
			for source in self.sample_sources:
				if source.uses_direct_integral():
					direct_sources.append(source)
					continue
				marched_sources.append(source)

			has_direct = len(direct_sources) > 0
			has_marched = len(marched_sources) > 0
			if has_direct and has_marched:
				direct_progress_fraction = (0.0, 0.6)
				marched_progress_fraction = (0.6, 1.0)
			elif has_direct:
				direct_progress_fraction = (0.0, 1.0)
				marched_progress_fraction = (1.0, 1.0)
			else:
				direct_progress_fraction = (0.0, 0.0)
				marched_progress_fraction = (0.0, 1.0)

			_t_direct_start = perf_counter()
			for source_idx, source in enumerate(direct_sources):
				source_progress_start = direct_progress_fraction[0]
				source_progress_end = direct_progress_fraction[1]
				if has_direct:
					source_progress_start = direct_progress_fraction[0] + (
						(direct_progress_fraction[1] - direct_progress_fraction[0]) * (float(source_idx) / float(len(direct_sources)))
					)
					source_progress_end = direct_progress_fraction[0] + (
						(direct_progress_fraction[1] - direct_progress_fraction[0]) * (float(source_idx + 1) / float(len(direct_sources)))
					)
				_t_src_start = perf_counter()
				direct_integral = source.ray_integral_world(
					ray_origins=ray_origins[hit_indices],
					ray_directions=ray_directions[hit_indices],
					t_starts=t_starts[hit_indices],
					t_ends=t_ends[hit_indices],
					physics_model=physics_model,
					step_mm=step_mm,
					progress_callback=progress_callback,
					progress_fraction=(source_progress_start, source_progress_end),
					geometry=geometry,
					reference_transform=reference_transform,
					hit_ray_indices=hit_indices,
					detector_shape_hw=(height, width),
				)
				source_integrals, source_work_count = direct_integral
				projection_flat[hit_indices] += np.asarray(source_integrals, dtype=np.float32)
				total_sample_count += int(source_work_count)
				_t_src_elapsed = float(perf_counter() - _t_src_start)
				_src_stat: dict = {
					"label": getattr(
						getattr(source, "mesh", None) or getattr(source, "volumetric", None),
						"label", type(source).__name__,
					),
					"source_type": type(source).__name__,
					"elapsed_s": _t_src_elapsed,
					"work_count": int(source_work_count),
				}
				if isinstance(source, MeshXRaySource):
					_src_stat["backend"] = source.backend
					_src_stat["mode"] = source.mode
					_src_stat["triangle_count"] = int(source._triangles_world.shape[0])
					if hasattr(source, "_last_bvh_build_s"):
						_src_stat["bvh_build_s"] = float(source._last_bvh_build_s)
					if hasattr(source, "_last_stack_timing"):
						_src_stat["stack_build_s"]         = float(source._last_stack_timing.get("total_s", 0.0))
						_src_stat["stack_uv_projection_s"] = float(source._last_stack_timing.get("uv_projection_s", 0.0))
						_src_stat["stack_rasterize_s"]     = float(source._last_stack_timing.get("rasterize_s", 0.0))
						_src_stat["stack_csr_s"]           = float(source._last_stack_timing.get("csr_s", 0.0))
					if hasattr(source, "_last_integral_timing"):
						_src_stat["integration_s"] = float(source._last_integral_timing.get("integration_s", 0.0))
				_per_source_stats.append(_src_stat)
			_phase_timings["direct_sources_total"] = float(perf_counter() - _t_direct_start)
			_t_marching_start = perf_counter()
			if marched_sources:
				t_start_global = float(np.min(t_starts[hit_mask]))
				t_global_max = float(np.max(t_ends[hit_mask]))
				t_values = np.arange(t_start_global, t_global_max + step_mm * 0.5, step_mm, dtype=np.float64)
				n_steps = len(t_values)
				_last_progress_clock = perf_counter()
				for i, t_k in enumerate(t_values):
					if progress_callback is not None and perf_counter() - _last_progress_clock >= 0.15:
						_last_progress_clock = perf_counter()
						progress_callback(
							marched_progress_fraction[0]
							+ (marched_progress_fraction[1] - marched_progress_fraction[0]) * (i / max(n_steps, 1))
						)
					t_k_f = float(t_k)
					active = hit_mask & (t_k_f >= t_starts) & (t_k_f <= t_ends)
					if not np.any(active):
						continue
					active_idx = np.where(active)[0]
					points_world = ray_origins[active_idx] + t_k_f * ray_directions[active_idx]
					total_mu = np.zeros(len(active_idx), dtype=np.float32)
					for source in marched_sources:
						total_mu += source.sample_attenuation_world(points_world, physics_model)
					projection_flat[active_idx] += total_mu * step_mm
					total_sample_count += len(active_idx)

			_phase_timings["marching_total"] = float(perf_counter() - _t_marching_start)
			for _vsrc in marched_sources:
				_vol_label = getattr(getattr(_vsrc, "volumetric", None), "label", None) or type(_vsrc).__name__
				_vsrc_stat: dict = {
					"label": _vol_label,
					"source_type": type(_vsrc).__name__,
					"elapsed_s": _phase_timings["marching_total"] / max(len(marched_sources), 1),
					"work_count": int(total_sample_count),
				}
				if isinstance(_vsrc, VolumetricXRaySource):
					_vsrc_stat["volume_shape"] = tuple(int(x) for x in _vsrc._volume.shape)
					_vsrc_stat["interpolation"] = _vsrc.interpolation
				_per_source_stats.append(_vsrc_stat)

			if progress_callback is not None:
				progress_callback(1.0)

		_t_physics_start = perf_counter()
		projection = physics_model.integral_to_image(
			projection_flat,
			source_to_detector_distance_mm=source_to_detector_distance_mm,
			projection_mode="cone" if geometry.is_cone_beam() else "parallel",
		).reshape(height, width)
		_phase_timings["physics_conversion"] = float(perf_counter() - _t_physics_start)

		if not return_stats:
			return projection

		elapsed_seconds = perf_counter() - start_time
		stats = XRayProjectionStats(
			elapsed_seconds=float(elapsed_seconds),
			total_pixels=n_pixels,
			traced_pixels=traced_pixels,
			total_sample_count=total_sample_count,
			source_count=int(len(self.sample_sources)),
			step_mm=step_mm,
			projection_mode="cone" if geometry.is_cone_beam() else "parallel",
			detector_shape_hw=(height, width),
			depth_window_mode=depth_mode,
			phase_timings=_phase_timings,
			per_source_stats=_per_source_stats,
		)
		return projection, stats

	def project_config(self, config, return_stats=False, progress_callback=None):
		"""Project the scene using a higher-level configuration object."""
		if not isinstance(config, XRayProjectionConfig):
			raise TypeError("config must be an instance of XRayProjectionConfig.")
		return self.project(
			geometry=config.effective_geometry(),
			physics_model=config.physics_model,
			reference_transform=config.reference_transform,
			return_stats=return_stats,
			progress_callback=progress_callback,
		)
