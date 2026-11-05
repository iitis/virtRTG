"""Vectorized primary attenuation integration for prepared X-ray ray batches."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .xraySource import MeshXRaySource, VolumetricXRaySource, XRaySampleSource


@dataclass
class XRayTransportResult:
	"""Hold flattened primary integrals and integration diagnostics.

	Integral arrays use detector C-order flattening and contain one value per
	pixel, including zeroes for pixels excluded by clipping or exposure. Keeping
	this representation avoids scattering hit-ray results back to detector maps
	inside the projector and preserves its existing accumulation order.
	"""

	total_primary_integral: np.ndarray
	per_source_primary_integrals: list[np.ndarray]
	total_work_count: int
	traced_pixels: int
	phase_timings: dict
	per_source_stats: list


class XRayTransportModel:
	"""Integrate attenuation over already generated and clipped world-space rays."""

	def __init__(self, sample_sources: list[XRaySampleSource]):
		"""Retain the selected source adapters in their scene order."""
		self.sample_sources = sample_sources
		self._source_index_map = {
			id(source): source_index
			for source_index, source in enumerate(self.sample_sources)
		}

	def integrate(
		self,
		*,
		geometry,
		physics_model,
		reference_transform,
		ray_origins,
		ray_directions,
		t_starts,
		t_ends,
		hit_mask,
		hit_indices,
		detector_shape_hw,
		step_mm,
		progress_callback=None,
	) -> XRayTransportResult:
		"""Integrate direct and sampled sources over prepared rays.

		Ray origins/directions and interval bounds are full detector-flattened
		arrays. ``hit_mask`` and ascending C-order ``hit_indices`` describe the
		already clipped rays. Direct adapters receive only the indexed hit rows;
		sampled adapters retain the existing batched depth-step loop.
		"""
		pixel_count = int(ray_origins.shape[0])
		projection_flat = np.zeros(pixel_count, dtype=np.float32)
		per_source_projection_flat = [
			np.zeros(pixel_count, dtype=np.float32)
			for _ in self.sample_sources
		]
		total_sample_count = 0
		traced_pixels = int(np.sum(hit_mask))
		phase_timings: dict = {}
		per_source_stats: list = []

		if traced_pixels > 0:
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
				scene_source_index = int(self._source_index_map[id(source)])
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
				source_physics_model = source.resolve_physics_model(physics_model)
				direct_integral = source.ray_integral_world(
					ray_origins=ray_origins[hit_indices],
					ray_directions=ray_directions[hit_indices],
					t_starts=t_starts[hit_indices],
					t_ends=t_ends[hit_indices],
					physics_model=source_physics_model,
					step_mm=step_mm,
					progress_callback=progress_callback,
					progress_fraction=(source_progress_start, source_progress_end),
					geometry=geometry,
					reference_transform=reference_transform,
					hit_ray_indices=hit_indices,
					detector_shape_hw=detector_shape_hw,
				)
				source_integrals, source_work_count = direct_integral
				source_integrals = np.asarray(source_integrals, dtype=np.float32)
				projection_flat[hit_indices] += source_integrals
				per_source_projection_flat[scene_source_index][hit_indices] += source_integrals
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
						_src_stat["stack_build_s"] = float(source._last_stack_timing.get("total_s", 0.0))
						_src_stat["stack_uv_projection_s"] = float(source._last_stack_timing.get("uv_projection_s", 0.0))
						_src_stat["stack_rasterize_s"] = float(source._last_stack_timing.get("rasterize_s", 0.0))
						_src_stat["stack_csr_s"] = float(source._last_stack_timing.get("csr_s", 0.0))
					if hasattr(source, "_last_integral_timing"):
						_src_stat["integration_s"] = float(source._last_integral_timing.get("integration_s", 0.0))
				per_source_stats.append(_src_stat)
			phase_timings["direct_sources_total"] = float(perf_counter() - _t_direct_start)
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
						scene_source_index = int(self._source_index_map[id(source)])
						source_physics_model = source.resolve_physics_model(physics_model)
						source_mu = source.sample_attenuation_world(points_world, source_physics_model)
						source_increment = np.asarray(source_mu, dtype=np.float32) * step_mm
						total_mu += np.asarray(source_mu, dtype=np.float32)
						per_source_projection_flat[scene_source_index][active_idx] += source_increment
					projection_flat[active_idx] += total_mu * step_mm
					total_sample_count += len(active_idx)

			phase_timings["marching_total"] = float(perf_counter() - _t_marching_start)
			for source in marched_sources:
				label = getattr(getattr(source, "volumetric", None), "label", None) or type(source).__name__
				stat: dict = {
					"label": label,
					"source_type": type(source).__name__,
					"elapsed_s": phase_timings["marching_total"] / max(len(marched_sources), 1),
					"work_count": int(total_sample_count),
				}
				if isinstance(source, VolumetricXRaySource):
					stat["volume_shape"] = tuple(int(x) for x in source._volume.shape)
					stat["interpolation"] = source.interpolation
				per_source_stats.append(stat)

			if progress_callback is not None:
				progress_callback(1.0)

		return XRayTransportResult(
			total_primary_integral=projection_flat,
			per_source_primary_integrals=per_source_projection_flat,
			total_work_count=total_sample_count,
			traced_pixels=traced_pixels,
			phase_timings=phase_timings,
			per_source_stats=per_source_stats,
		)
