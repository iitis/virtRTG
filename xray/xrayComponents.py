"""Lightweight tube-emission and collimator configuration value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .xrayProjection import RectangularBeamGeometry


@dataclass(frozen=True)
class FocalSpotSamples:
	"""Weighted focal-spot positions in an explicitly named coordinate frame."""

	positions_mm: tuple[tuple[float, float, float], ...]
	weights: tuple[float, ...]
	coordinate_frame: str

	def __post_init__(self):
		positions = np.asarray(self.positions_mm, dtype=np.float64)
		weights = np.asarray(self.weights, dtype=np.float64)
		if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] == 0:
			raise ValueError("positions_mm must have shape (N, 3) for N >= 1")
		if weights.shape != (positions.shape[0],):
			raise ValueError("weights must have shape (N,)")
		if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
			raise ValueError("focal-spot positions and weights must be finite")
		if np.any(weights < 0.0) or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-12):
			raise ValueError("focal-spot weights must be nonnegative and sum to one")
		if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
			raise ValueError("coordinate_frame must be explicit")
		object.__setattr__(self, "positions_mm", tuple(tuple(float(v) for v in row) for row in positions))
		object.__setattr__(self, "weights", tuple(float(v) for v in weights))
		object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame))

	@classmethod
	def point(cls, position_mm, coordinate_frame):
		"""Create the current single-point source configuration."""
		return cls((tuple(float(v) for v in position_mm),), (1.0,), coordinate_frame)


@dataclass(frozen=True)
class XRayTube:
	"""Immutable emission snapshot with optional independently configured pose.

	VirtualXRay currently has no tube orientation setting independent of its
	effective beam frame, so ``tube_to_reference_rotation`` is normally ``None``.
	"""

	focal_spot_samples: FocalSpotSamples
	tube_to_reference_rotation: tuple[tuple[float, float, float], ...] | None = None
	energy_kev: float | None = None

	def __post_init__(self):
		if not isinstance(self.focal_spot_samples, FocalSpotSamples):
			raise TypeError("focal_spot_samples must be FocalSpotSamples")
		if self.tube_to_reference_rotation is not None:
			rotation = np.asarray(self.tube_to_reference_rotation, dtype=np.float64)
			if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
				raise ValueError("tube_to_reference_rotation must be a finite 3x3 rotation")
			if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-7):
				raise ValueError("tube_to_reference_rotation must be orthonormal")
			if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-7):
				raise ValueError("tube_to_reference_rotation must be right-handed")
			object.__setattr__(self, "tube_to_reference_rotation", tuple(tuple(float(v) for v in row) for row in rotation))
		if self.energy_kev is not None and (not np.isfinite(float(self.energy_kev)) or float(self.energy_kev) <= 0.0):
			raise ValueError("energy_kev must be finite and positive when provided")
		if self.energy_kev is not None:
			object.__setattr__(self, "energy_kev", float(self.energy_kev))


@dataclass(frozen=True)
class XRayCollimator:
	"""Immutable ideal angular-field snapshot referenced to one point source.

	``RectangularBeamGeometry`` contains an axis, angular field dimensions and
	transverse axes, but no finite aperture pose or position. Accordingly, this
model evaluates directions from a shared focal point; it cannot calculate
physically positioned collimation for independent source origins.
	"""

	beam_geometry: RectangularBeamGeometry | None
	enabled: bool = False

	def __post_init__(self):
		if self.beam_geometry is not None and not isinstance(self.beam_geometry, RectangularBeamGeometry):
			raise TypeError("beam_geometry must be RectangularBeamGeometry or None")
		if self.enabled and self.beam_geometry is None:
			raise ValueError("enabled collimation requires beam geometry")

	def transmission_for_directions(self, ray_directions_ref):
		"""Return batch transmission for directions from the shared focal point.

		The current ideal model returns zero or one. Ray origins are deliberately
		absent: varying them requires a positioned aperture model that this field
		geometry does not define. The projector retains its optimized detector mask.
		"""
		directions = np.asarray(ray_directions_ref, dtype=np.float64)
		if directions.shape[-1:] != (3,):
			raise ValueError("ray directions must end in a three-vector axis")
		if not np.all(np.isfinite(directions)):
			raise ValueError("ray directions must be finite")
		if not self.enabled:
			return np.ones(directions.shape[:-1], dtype=np.float32)
		beam = self.beam_geometry
		axis = np.asarray(beam.beam_axis_ref)
		horizontal = np.asarray(beam.beam_horizontal_ref)
		vertical = np.asarray(beam.beam_vertical_ref)
		forward = np.einsum("...i,i->...", directions, axis)
		limit_u = forward * beam.field_width_cm * 5.0 / beam.field_reference_distance_mm
		limit_v = forward * beam.field_height_cm * 5.0 / beam.field_reference_distance_mm
		inside = ((forward > 0.0)
			& (np.abs(np.einsum("...i,i->...", directions, horizontal)) <= limit_u)
			& (np.abs(np.einsum("...i,i->...", directions, vertical)) <= limit_v))
		return inside.astype(np.float32)

	def accept_directions(self, ray_directions_ref):
		"""Return binary acceptance for directions from the shared focal point."""
		return self.transmission_for_directions(ray_directions_ref).astype(bool)
