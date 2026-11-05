"""Create and compare the frozen virtRTG 0.6.2 transport references.

Use an explicit mode from the repository root with ``.venv/Scripts/python.exe``.
The original baseline is immutable after creation; after/supplement/compare
modes verify its recorded SHA-256 values before reading it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np


PLUGIN_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = PLUGIN_DIR.parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

ACCEPTED_COMMIT = "8b7ff41883b329cfdb11afb1243c1ce34e59a81e"
BASELINE_HASHES = {
    "baseline.json": "E4044FC47A8B18D8D8275B198D4BE560B168260BA1912419CA61E2B6B8F1D062",
    "baseline_arrays.npz": "C86967088175DB26D4AADE1AA65FC7B070E5D91D07A64997C44D0512E3B30C4A",
}
WARMUPS = 2
REPETITIONS = 5
TOLERANCE = {"absolute": 1e-6, "relative": 1e-6, "relative_near_zero_abs_floor": 1e-7}
TIMING_STAT_KEYS = {
    "elapsed_seconds", "elapsed_s", "phase_timings", "bvh_build_s",
    "stack_build_s", "stack_uv_projection_s", "stack_rasterize_s",
    "stack_csr_s", "integration_s",
}

# Avoid plugins/__init__.py application bootstrap, matching tests/conftest.py.
for package_name, package_path in (
    ("plugins", REPO_ROOT / "plugins"),
    ("plugins.virtRTG", PLUGIN_DIR),
    ("plugins.virtRTG.xray", PLUGIN_DIR / "xray"),
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from dpVision import Mesh, Volumetric, SliceMetadata
from plugins.virtRTG.xray.xrayProjection import (
    RectangularBeamGeometry,
    XRayPhysicsModel,
    XRayProjectionConfig,
    XRayProjectionGeometry,
    XRayScene,
)
from plugins.virtRTG.xray.xraySource import MeshXRaySource, VolumetricXRaySource


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify_frozen_baseline():
    """Reject missing or modified original baseline files before comparison."""
    hash_manifest = OUTPUT_DIR / "baseline_hashes.json"
    if hash_manifest.exists():
        manifest = json.loads(hash_manifest.read_text(encoding="utf-8"))
        if manifest.get("files") != BASELINE_HASHES:
            raise RuntimeError("baseline_hashes.json does not match the hard-coded frozen baseline hashes")
    for filename, expected_hash in BASELINE_HASHES.items():
        path = OUTPUT_DIR / filename
        if not path.is_file():
            raise RuntimeError(f"Frozen baseline file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen baseline hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
    baseline_metadata = json.loads((OUTPUT_DIR / "baseline.json").read_text(encoding="utf-8"))
    if baseline_metadata.get("baseline_git_commit") != ACCEPTED_COMMIT:
        raise RuntimeError("Frozen baseline metadata does not identify the accepted 0.6.1 commit")


def _production_sources_match_accepted():
    """Return whether production projection and source files match accepted 0.6.1."""
    for relative_path in ("xray/xrayProjection.py", "xray/xraySource.py"):
        committed = subprocess.run(
            ["git", "-C", str(PLUGIN_DIR), "show", f"{ACCEPTED_COMMIT}:{relative_path}"],
            check=True, capture_output=True,
        ).stdout
        current = (PLUGIN_DIR / relative_path).read_bytes()
        if current != committed:
            return False
    return True


def _assert_accepted_production_source():
    """Allow baseline/supplement creation only when production code is accepted 0.6.1."""
    if not _production_sources_match_accepted():
        raise RuntimeError("Refusing reference creation: projection/source code differs from accepted 0.6.1")


def _volume_source(backend="sampling", label="volume", shape_zyx=(5, 5, 5), origin_xyz=(-2.0, -2.0, -2.0)):
    volume = Volumetric()
    volume.label = label
    volume.m_volume = np.full(shape_zyx, 0.0, dtype=np.float32)
    volume.shape = volume.m_volume.shape
    metadata = SliceMetadata()
    metadata.image_position_patient = list(origin_xyz)
    metadata.pixel_spacing = [1.0, 1.0]
    metadata.slice_distance = 1.0
    metadata.slice_thickness = 1.0
    metadata.voxel_spacing = [1.0, 1.0, 1.0]
    volume.metadata = [metadata]
    return VolumetricXRaySource(volume, volume_backend=backend, interpolation="linear")


def _mesh_source(backend="analytic_bvh", scale=1.0):
    mesh = Mesh()
    mesh.label = "cube_mesh"
    mesh.m_vertices = float(scale) * np.asarray([
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ], dtype=np.float32)
    mesh.m_faces = np.asarray([
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ], dtype=np.uint32)
    return MeshXRaySource(mesh, scalar_value=0.0, mode="solid", backend=backend)


def _geometry(mode="cone", detector_shape_hw=(5, 5), detector_z=8.0,
              source_position_ref=None, ray_direction_ref=(0.0, 0.0, -1.0), **kwargs):
    height, width = map(int, detector_shape_hw)
    common = dict(
        detector_origin_ref=[-(width - 1) / 2.0, -(height - 1) / 2.0, float(detector_z)],
        detector_u_ref=[1.0, 0.0, 0.0],
        detector_v_ref=[0.0, 1.0, 0.0],
        detector_shape_hw=[height, width],
        step_mm=0.5,
    )
    common.update(kwargs)
    if mode == "cone":
        common["source_position_ref"] = [0.0, 0.0, -8.0] if source_position_ref is None else source_position_ref
    else:
        common["ray_direction_ref"] = ray_direction_ref
        common.pop("source_position_ref", None)
    return XRayProjectionGeometry(**common)


def _collimation():
    return RectangularBeamGeometry(
        beam_axis_ref=(0, 0, 1), beam_horizontal_ref=(1, 0, 0),
        beam_vertical_ref=(0, 1, 0), field_width_cm=2.0,
        field_height_cm=2.0, field_reference_distance_mm=100.0,
    )


def _cases():
    """Return the original 12 frozen case definitions without changing their parameters."""
    return {
        "cone_volume_sampling_integral": ([_volume_source("sampling")], _geometry(), "integral"),
        "cone_volume_siddon_integral": ([_volume_source("siddon")], _geometry(), "integral"),
        "cone_mesh_analytic_bvh": ([_mesh_source("analytic_bvh")], _geometry(), "integral"),
        "cone_mesh_projected_list": ([_mesh_source("projected_intersection_list")], _geometry(), "integral"),
        "cone_mixed_volume_mesh": ([_volume_source("sampling"), _mesh_source("analytic_bvh")], _geometry(), "integral"),
        "cone_multi_source_maps": ([_volume_source("sampling", "volume_a"), _volume_source("sampling", "volume_b")], _geometry(), "integral"),
        "cone_depth_ray": ([_volume_source("sampling")], _geometry(depth_window_mode="ray", depth_window_mm=[9.0, 13.0]), "integral"),
        "cone_depth_planar": ([_volume_source("sampling")], _geometry(depth_window_mode="planar", depth_window_mm=[9.0, 13.0], depth_window_origin_ref=[0.0, 0.0, 0.0], depth_window_axis_ref=[0.0, 0.0, 1.0]), "integral"),
        "cone_collimation_disabled": ([_volume_source("sampling")], _geometry(beam_geometry=_collimation()), "integral"),
        "cone_collimation_enabled": ([_volume_source("sampling")], _geometry(beam_geometry=_collimation(), collimation_enabled=True), "integral"),
        "cone_intensity": ([_volume_source("sampling")], _geometry(), "intensity"),
        # Preserve the ineffective +Z direction in the immutable original case.
        "parallel_volume_sampling": ([_volume_source("sampling")], _geometry("parallel", ray_direction_ref=[0.0, 0.0, 1.0]), "integral"),
    }


def _supplementary_cases():
    """Return corrected effective depth/parallel cases with independently frozen references."""
    return {
        "cone_depth_planar_effective": (
            [_volume_source("sampling")],
            _geometry(depth_window_mode="planar", depth_window_mm=[7.0, 9.0],
                      depth_window_origin_ref=[0.0, 0.0, -8.0], depth_window_axis_ref=[0.0, 0.0, 1.0]),
            "integral",
        ),
        "parallel_volume_sampling_effective": (
            [_volume_source("sampling")], _geometry("parallel", ray_direction_ref=[0.0, 0.0, -1.0]), "integral",
        ),
        "cone_depth_ray_effective": (
            [_volume_source("sampling")],
            _geometry(depth_window_mode="ray", depth_window_mm=[9.0, 13.0]), "integral",
        ),
    }


def _jsonable_stats(stats):
    return {
        "elapsed_seconds": float(stats.elapsed_seconds),
        "total_pixels": int(stats.total_pixels),
        "traced_pixels": int(stats.traced_pixels),
        "total_sample_count": int(stats.total_sample_count),
        "source_count": int(stats.source_count),
        "step_mm": float(stats.step_mm),
        "projection_mode": stats.projection_mode,
        "detector_shape_hw": list(stats.detector_shape_hw),
        "depth_window_mode": stats.depth_window_mode,
        "phase_timings": {key: float(value) for key, value in stats.phase_timings.items()},
        "per_source_stats": [
            {key: (list(value) if isinstance(value, tuple) else value) for key, value in item.items()}
            for item in stats.per_source_stats
        ],
    }


def _capture_case(name, sources, geometry, output_mode):
    model = XRayPhysicsModel(output_mode=output_mode, intensity_floor=0.0)
    capture = XRayScene.from_sample_sources(sources).project_capture(
        XRayProjectionConfig(geometry=geometry, physics_model=model), return_stats=True,
    )
    if capture.stats is None or capture.exposure_mask is None:
        raise RuntimeError(f"{name}: missing projection stats or exposure mask")
    arrays = {
        f"{name}__line_integral": capture.line_integral_image,
        f"{name}__raw_output": capture.detector_image,
        f"{name}__exposure_mask": capture.exposure_mask,
    }
    source_shapes = []
    source_backends = []
    for index, source_result in enumerate(capture.source_projections):
        arrays[f"{name}__source_{index}_integral"] = source_result.line_integral_image
        arrays[f"{name}__source_{index}_output"] = source_result.detector_image
        source_shapes.append({
            "index": index, "label": source_result.label,
            "source_type": source_result.source_type,
            "shape": list(source_result.line_integral_image.shape),
            "dtype": str(source_result.line_integral_image.dtype),
        })
        source = sources[index]
        source_backends.append(
            f"volume_{source.volume_backend}" if isinstance(source, VolumetricXRaySource)
            else f"mesh_{source.backend}" if isinstance(source, MeshXRaySource)
            else type(source).__name__
        )
    for key, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name}: non-finite result in {key}")
    return arrays, {
        "output_mode": output_mode,
        "projection_mode": capture.stats.projection_mode,
        "shape": list(capture.line_integral_image.shape),
        "line_integral_dtype": str(capture.line_integral_image.dtype),
        "raw_output_dtype": str(capture.detector_image.dtype),
        "exposure_mask_dtype": str(capture.exposure_mask.dtype),
        "exposure_mask": capture.exposure_mask.tolist(),
        "source_projections": source_shapes,
        "source_backends": source_backends,
        "stats": _jsonable_stats(capture.stats),
        "array_bytes": int(sum(array.nbytes for array in arrays.values())),
    }


def _run_cases(case_definitions):
    arrays = {}
    metadata = {}
    for name, (sources, geometry, output_mode) in case_definitions.items():
        case_arrays, case_metadata = _capture_case(name, sources, geometry, output_mode)
        arrays.update(case_arrays)
        metadata[name] = case_metadata
    return arrays, metadata


def _require_effective_case(name, arrays, metadata, *, require_partial=False):
    """Assert a regression case crosses the source and produces nonzero attenuation."""
    stats = metadata[name]["stats"]
    integral = arrays[f"{name}__line_integral"]
    if stats["traced_pixels"] <= 0:
        raise RuntimeError(f"{name}: expected nonzero traced pixels")
    if stats["total_sample_count"] <= 0:
        raise RuntimeError(f"{name}: expected nonzero integration work")
    if not np.any(integral > 0.0):
        raise RuntimeError(f"{name}: expected nonzero attenuation")
    if require_partial and not np.any((integral > 0.0) & (integral < 0.1)):
        raise RuntimeError(f"{name}: expected a partially depth-restricted nonzero integral")


def _validate_supplementary(arrays, metadata):
    """Enforce live coverage for corrected planar, parallel and ray-depth scenes."""
    _require_effective_case("cone_depth_planar_effective", arrays, metadata, require_partial=True)
    _require_effective_case("parallel_volume_sampling_effective", arrays, metadata)
    _require_effective_case("cone_depth_ray_effective", arrays, metadata, require_partial=True)
    for name in ("cone_depth_planar_effective", "cone_depth_ray_effective"):
        mode = metadata[name]["stats"]["depth_window_mode"]
        if mode != ("planar" if "planar" in name else "ray"):
            raise RuntimeError(f"{name}: expected requested depth-window mode, got {mode!r}")
    if metadata["parallel_volume_sampling_effective"]["stats"]["projection_mode"] != "parallel":
        raise RuntimeError("parallel supplementary case did not execute parallel projection")
    full_arrays, _ = _capture_case(
        "unrestricted_cone_check", [_volume_source("sampling")], _geometry(), "integral",
    )
    for name in ("cone_depth_planar_effective", "cone_depth_ray_effective"):
        restricted = arrays[f"{name}__line_integral"]
        unrestricted = full_arrays["unrestricted_cone_check__line_integral"]
        if float(restricted.sum()) >= float(unrestricted.sum()):
            raise RuntimeError(f"{name}: restriction did not reduce total attenuation")
        if np.any(restricted > unrestricted + TOLERANCE["absolute"]):
            raise RuntimeError(f"{name}: restricted integral exceeds the unrestricted reference")


def _run_benchmark(factory, *, warmups=WARMUPS, repetitions=REPETITIONS):
    """Measure repeated full projections, constructing cold source adapters each iteration."""
    timings = []
    allocation_peaks = []
    phase_samples = {}
    retained_bytes = 0
    for iteration in range(warmups + repetitions):
        sources, geometry = factory()
        config = XRayProjectionConfig(geometry, XRayPhysicsModel(output_mode="integral"))
        scene = XRayScene.from_sample_sources(sources)
        tracemalloc.start()
        started = time.perf_counter()
        capture = scene.project_capture(config, return_stats=True)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if iteration >= warmups:
            timings.append(float(elapsed))
            allocation_peaks.append(int(peak))
            retained_bytes = int(capture.line_integral_image.nbytes + capture.detector_image.nbytes + capture.exposure_mask.nbytes)
            for key, value in capture.stats.phase_timings.items():
                phase_samples.setdefault(key, []).append(float(value))
            detector_shape = list(capture.stats.detector_shape_hw)
            traced_pixels = int(capture.stats.traced_pixels)
            integration_work = int(capture.stats.total_sample_count)
            source_bytes = int(sum(
                getattr(source, "_volume", np.empty(0, dtype=np.uint8)).nbytes
                + getattr(source, "_vertices_world", np.empty(0, dtype=np.uint8)).nbytes
                + getattr(source, "_faces", np.empty(0, dtype=np.uint8)).nbytes
                for source in sources
            ))
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "individual_runtime_seconds": timings,
        "median_runtime_seconds": float(statistics.median(timings)),
        "spread_seconds_min_max": [float(min(timings)), float(max(timings))],
        "phase_median_seconds": {key: float(statistics.median(values)) for key, values in phase_samples.items()},
        "tracemalloc_peak_bytes_median": int(statistics.median(allocation_peaks)),
        "tracemalloc_note": "Python allocator proxy; does not include all NumPy native allocations or peak RSS.",
        "retained_capture_array_bytes": retained_bytes,
        "detector_shape_hw": detector_shape,
        "traced_pixels": traced_pixels,
        "integration_work_count": integration_work,
        "source_geometry_array_bytes": source_bytes,
        "cache_state": "New source adapters each iteration; BVH/stack caches cold at projection start. Source setup is outside timed interval.",
    }


def _environment():
    """Collect interpreter, numerical-library and host details for repeatability."""
    versions = {}
    for module_name in ("scipy", "cv2", "pydicom"):
        try:
            module = __import__(module_name)
            versions[module_name] = module.__version__
        except Exception:
            versions[module_name] = None
    try:
        import psutil
        total_memory = int(psutil.virtual_memory().total)
    except ImportError:
        total_memory = None
    return {
        "python": platform.python_version(), "numpy": np.__version__, **versions,
        "os": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "physical_memory_total_bytes": total_memory,
    }


def _large_benchmarks():
    """Run two bounded workload references with detector sizes sized per backend."""
    cases = {
        "volume_sampling_64x64": lambda: (
            [_volume_source("sampling", shape_zyx=(16, 32, 32), origin_xyz=(-16.0, -16.0, -8.0))],
            _geometry(detector_shape_hw=(64, 64), detector_z=32.0, source_position_ref=[0.0, 0.0, -32.0]),
        ),
        "volume_siddon_32x32": lambda: (
            [_volume_source("siddon", shape_zyx=(16, 32, 32), origin_xyz=(-16.0, -16.0, -8.0))],
            _geometry(detector_shape_hw=(32, 32), detector_z=32.0, source_position_ref=[0.0, 0.0, -32.0]),
        ),
        "mixed_sampling_mesh_32x32": lambda: (
            [_volume_source("sampling", shape_zyx=(16, 32, 32), origin_xyz=(-16.0, -16.0, -8.0)), _mesh_source("analytic_bvh", scale=8.0)],
            _geometry(detector_shape_hw=(32, 32), detector_z=32.0, source_position_ref=[0.0, 0.0, -32.0]),
        ),
    }
    return {
        "environment": _environment(),
        "scene": {"volume_shape_zyx": [16, 32, 32], "volume_extent_mm_xyz": [32, 32, 16], "mesh_faces": 12, "mixed_mesh_edge_mm": 16.0, "scalar_hu": 0.0, "step_mm": 0.5},
        "benchmarks": {name: _run_benchmark(factory) for name, factory in cases.items()},
    }


def _small_benchmarks():
    return {
        name: _run_benchmark(factory)
        for name, factory in {
            "volume_sampling": lambda: ([_volume_source("sampling")], _geometry()),
            "volume_siddon": lambda: ([_volume_source("siddon")], _geometry()),
            "mixed_volume_mesh": lambda: ([_volume_source("sampling"), _mesh_source("analytic_bvh")], _geometry()),
        }.items()
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _save_npz(path, arrays):
    with path.open("wb") as output:
        np.savez_compressed(output, **arrays)


def _write_hash_manifest():
    file_hashes = {name: _sha256(OUTPUT_DIR / name) for name in BASELINE_HASHES}
    _write_json(OUTPUT_DIR / "baseline_hashes.json", {
        "baseline_git_commit": ACCEPTED_COMMIT,
        "files": file_hashes,
        "verified_sha256": file_hashes,
    })


def _baseline_mode(confirm):
    """Create original baseline paths once, only from unchanged accepted code."""
    paths = [OUTPUT_DIR / name for name in ("baseline.json", "baseline_arrays.npz")]
    existing = [path.name for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite frozen baseline artifact(s): {', '.join(existing)}")
    if not confirm:
        raise RuntimeError("Baseline creation requires --confirm-baseline-write; it cannot be an ordinary run")
    _assert_accepted_production_source()
    arrays, cases = _run_cases(_cases())
    _validate_required_original_cases(arrays, cases)
    benchmarks = _small_benchmarks()
    _save_npz(OUTPUT_DIR / "baseline_arrays.npz", arrays)
    payload = _metadata_payload("pre_refactoring_baseline", cases, benchmarks, arrays)
    _write_json(OUTPUT_DIR / "baseline.json", payload)
    _write_hash_manifest()
    print("Created baseline artifacts from unchanged accepted production sources.")


def _validate_required_original_cases(arrays, cases):
    required_backends = {
        "cone_volume_sampling_integral": "volume_sampling",
        "cone_volume_siddon_integral": "volume_siddon",
        "cone_mesh_analytic_bvh": "mesh_analytic_bvh",
        "cone_mesh_projected_list": "mesh_projected_intersection_list",
        "cone_mixed_volume_mesh": "mesh_analytic_bvh",
    }
    if not cases:
        raise RuntimeError("No baseline reference cases were executed")
    for case_name, backend_name in required_backends.items():
        if backend_name not in cases[case_name]["source_backends"]:
            raise RuntimeError(f"{case_name}: expected adapter backend {backend_name} did not execute")
    for name in ("cone_volume_siddon_integral", "cone_mesh_analytic_bvh", "cone_mesh_projected_list"):
        stats = cases[name]["stats"]
        if stats["traced_pixels"] <= 0 or stats["total_sample_count"] <= 0:
            raise RuntimeError(f"{name}: backend case did not trace rays and perform integration")
    for key, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"baseline array {key} contains non-finite values")


def _metadata_payload(status, cases, benchmarks, arrays):
    return {
        "status": status,
        "baseline_git_commit": ACCEPTED_COMMIT,
        "plugin_worktree_status_at_run": subprocess.run(
            ["git", "-C", str(PLUGIN_DIR), "status", "--porcelain=v1"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
        "environment": _environment(),
        "scene_parameters": {
            "detector_shape_hw_by_case": {name: item["shape"] for name, item in cases.items()},
            "step_mm": 0.5, "volume_shape_zyx": [5, 5, 5], "volume_scalar_hu": 0.0,
            "mesh": "closed triangulated cube, 8 vertices, 12 faces",
            "cone_source_ref_mm": [0.0, 0.0, -8.0],
            "detector_basis_mm": [[1, 0, 0], [0, 1, 0]],
        },
        "tested_integration_backends": ["volume sampling", "volume Siddon", "analytic mesh BVH", "experimental projected_intersection_list", "mixed volume and analytic mesh"],
        "cases": cases,
        "benchmarks": benchmarks,
        "numerical_tolerances_fixed_before_refactoring": TOLERANCE,
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
    }


def _supplement_mode(refresh=False):
    """Freeze corrected cases and larger accepted-code performance measurements separately."""
    _verify_frozen_baseline()
    output_paths = [OUTPUT_DIR / name for name in (
        "supplemental.json", "supplemental_arrays.npz", "large_performance.json", "baseline_hashes.json",
    )]
    existing = [path.name for path in output_paths if path.exists()]
    if existing and not refresh:
        raise RuntimeError(f"Refusing to overwrite existing supplementary artifact(s): {', '.join(existing)}")
    _assert_accepted_production_source()
    if existing:
        for path in output_paths:
            if path.exists():
                path.unlink()
    arrays, cases = _run_cases(_supplementary_cases())
    _validate_supplementary(arrays, cases)
    large_perf = _large_benchmarks()
    payload = _metadata_payload("accepted_code_supplementary_reference", cases, {}, arrays)
    payload["baseline_hashes"] = BASELINE_HASHES
    payload["coverage_assertions"] = {
        name: {"traced_pixels": item["stats"]["traced_pixels"],
               "integration_work": item["stats"]["total_sample_count"],
               "integral_sum": float(np.sum(arrays[f"{name}__line_integral"]))}
        for name, item in cases.items()
    }
    _save_npz(OUTPUT_DIR / "supplemental_arrays.npz", arrays)
    _write_json(OUTPUT_DIR / "supplemental.json", payload)
    _write_json(OUTPUT_DIR / "large_performance.json", large_perf)
    _write_hash_manifest()
    print("Created supplementary effective-coverage and larger performance references.")


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _inventory_check(expected, actual, label):
    missing = sorted(set(expected) - set(actual))
    additional = sorted(set(actual) - set(expected))
    if missing or additional:
        raise RuntimeError(f"{label} array inventory mismatch; missing={missing}, additional={additional}")


def _strip_timing(value):
    if isinstance(value, dict):
        return {key: _strip_timing(item) for key, item in value.items() if key not in TIMING_STAT_KEYS}
    if isinstance(value, list):
        return [_strip_timing(item) for item in value]
    return value


def _compare_group(reference_arrays, actual_arrays, reference_meta, actual_meta, label):
    _inventory_check(reference_arrays, actual_arrays, label)
    _inventory_check(reference_meta["cases"], actual_meta["cases"], f"{label} case")
    array_results = {}
    for key in sorted(reference_arrays):
        ref = reference_arrays[key]
        actual = actual_arrays[key]
        if ref.shape != actual.shape:
            raise RuntimeError(f"{label} {key}: shape mismatch {ref.shape} != {actual.shape}")
        if ref.dtype != actual.dtype:
            raise RuntimeError(f"{label} {key}: dtype mismatch {ref.dtype} != {actual.dtype}")
        if key.endswith("__exposure_mask"):
            equal = bool(np.array_equal(ref, actual))
            if not equal:
                raise RuntimeError(f"{label} {key}: exact exposure mask mismatch")
            array_results[key] = {"exact": True, "max_absolute_error": 0.0, "mean_absolute_error": 0.0, "max_relative_error": 0.0, "passed": True}
        else:
            delta = np.abs(ref.astype(np.float64) - actual.astype(np.float64))
            max_abs = float(delta.max(initial=0.0))
            mae = float(delta.mean()) if delta.size else 0.0
            relative = delta / np.maximum(np.abs(ref.astype(np.float64)), TOLERANCE["relative_near_zero_abs_floor"])
            max_relative = float(relative.max(initial=0.0))
            passed = bool(np.allclose(ref, actual, atol=TOLERANCE["absolute"], rtol=TOLERANCE["relative"]))
            array_results[key] = {
                "exact": bool(np.array_equal(ref, actual)),
                "max_absolute_error": max_abs,
                "mean_absolute_error": mae,
                "max_relative_error": max_relative,
                "passed": passed,
            }
            if not passed:
                raise RuntimeError(f"{label} {key}: tolerance failure (max_abs={max_abs}, max_rel={max_relative})")

    case_results = {}
    for name, ref in reference_meta["cases"].items():
        actual = actual_meta["cases"][name]
        if _strip_timing(ref) != _strip_timing(actual):
            raise RuntimeError(f"{label} {name}: non-timing statistics, backend identity, or source ordering changed")
        case_results[name] = {"non_timing_stats_equal": True, "source_order_equal": True}
    return {"arrays": array_results, "cases": case_results}


def _runtime_comparison(reference_benchmarks, actual_benchmarks):
    _inventory_check(reference_benchmarks, actual_benchmarks, "benchmark runtime")
    results = {}
    for name in sorted(reference_benchmarks):
        before = reference_benchmarks[name]
        after = actual_benchmarks[name]
        before_times = before.get("individual_runtime_seconds", [])
        after_times = after.get("individual_runtime_seconds", [])
        if not before_times or not after_times:
            raise RuntimeError(f"{name}: missing runtime samples")
        before_median = float(statistics.median(before_times))
        after_median = float(statistics.median(after_times))
        results[name] = {
            "before_samples_seconds": before_times,
            "after_samples_seconds": after_times,
            "before_median_seconds": before_median,
            "after_median_seconds": after_median,
            "before_spread_min_max_seconds": before.get("spread_seconds_min_max"),
            "after_spread_min_max_seconds": after.get("spread_seconds_min_max"),
            "runtime_ratio_after_over_before": after_median / before_median if before_median > 0 else None,
        }
    return results


def _compare_files():
    _verify_frozen_baseline()
    baseline_meta = json.loads((OUTPUT_DIR / "baseline.json").read_text(encoding="utf-8"))
    after_meta = json.loads((OUTPUT_DIR / "after.json").read_text(encoding="utf-8"))
    baseline_arrays = _load_arrays(OUTPUT_DIR / "baseline_arrays.npz")
    after_arrays = _load_arrays(OUTPUT_DIR / "after_arrays.npz")
    after_base_arrays = {key: value for key, value in after_arrays.items() if not key.startswith("supplemental__")}
    base_compare = _compare_group(baseline_arrays, after_base_arrays, baseline_meta, after_meta, "baseline")
    supplement_compare = None
    if (OUTPUT_DIR / "supplemental.json").exists():
        supplement_meta = json.loads((OUTPUT_DIR / "supplemental.json").read_text(encoding="utf-8"))
        supplement_arrays = _load_arrays(OUTPUT_DIR / "supplemental_arrays.npz")
        # Supplementary outputs are saved with a `supplemental__` prefix in after arrays.
        actual_supplement = {
            key.removeprefix("supplemental__"): value
            for key, value in after_arrays.items() if key.startswith("supplemental__")
        }
        supplemental_after_meta = after_meta["supplemental"]
        supplement_compare = _compare_group(
            supplement_arrays, actual_supplement, supplement_meta,
            {"cases": supplemental_after_meta}, "supplementary",
        )
    expected_after_inventory = set(baseline_arrays)
    if supplement_compare is not None:
        expected_after_inventory |= {f"supplemental__{key}" for key in supplement_arrays}
    _inventory_check(expected_after_inventory, after_arrays, "complete after")
    runtime_before = dict(baseline_meta.get("benchmarks", {}))
    if (OUTPUT_DIR / "large_performance.json").exists():
        runtime_before.update(json.loads((OUTPUT_DIR / "large_performance.json").read_text(encoding="utf-8"))["benchmarks"])
    runtime_after = dict(after_meta.get("benchmarks", {}))
    runtime_after.update(after_meta.get("large_benchmarks", {}))
    comparison = {
        "baseline_hashes_verified": BASELINE_HASHES,
        "tolerances": TOLERANCE,
        "original_baseline": base_compare,
        "supplementary": supplement_compare,
        "runtime_distributions": _runtime_comparison(runtime_before, runtime_after),
    }
    _write_json(OUTPUT_DIR / "comparison.json", comparison)
    return comparison


def _after_mode():
    _verify_frozen_baseline()
    supplemental_json = OUTPUT_DIR / "supplemental.json"
    supplemental_npz = OUTPUT_DIR / "supplemental_arrays.npz"
    if not supplemental_json.is_file() or not supplemental_npz.is_file():
        raise RuntimeError("Run --mode supplement first to freeze corrected coverage and larger references")
    base_arrays, base_cases = _run_cases(_cases())
    _validate_required_original_cases(base_arrays, base_cases)
    supplemental_arrays, supplemental_cases = _run_cases(_supplementary_cases())
    _validate_supplementary(supplemental_arrays, supplemental_cases)
    _inventory_check(_load_arrays(OUTPUT_DIR / "baseline_arrays.npz"), base_arrays, "after baseline")
    frozen_supplement = _load_arrays(supplemental_npz)
    _inventory_check(frozen_supplement, supplemental_arrays, "after supplementary")
    all_arrays = dict(base_arrays)
    all_arrays.update({f"supplemental__{key}": value for key, value in supplemental_arrays.items()})
    small_benchmarks = _small_benchmarks()
    large_benchmarks = _large_benchmarks()
    _save_npz(OUTPUT_DIR / "after_arrays.npz", all_arrays)
    after_status = (
        "runner_self_validation_on_accepted_0.6.1"
        if _production_sources_match_accepted() else "post_refactoring_comparison"
    )
    after_meta = _metadata_payload(after_status, base_cases, small_benchmarks, base_arrays)
    after_meta["supplemental"] = supplemental_cases
    after_meta["large_benchmarks"] = large_benchmarks["benchmarks"]
    after_meta["supplemental_array_prefix"] = "supplemental__"
    after_meta["supplemental_array_inventory"] = sorted(supplemental_arrays)
    _write_json(OUTPUT_DIR / "after.json", after_meta)
    comparison = _compare_files()
    failed = [key for group in (comparison["original_baseline"], comparison.get("supplementary") or {})
              for key, result in group.get("arrays", {}).items() if not result.get("passed", True)]
    print(json.dumps({"comparison": "comparison.json", "array_count": len(all_arrays), "failed": failed}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("baseline", "supplement", "after", "refactored", "compare"))
    parser.add_argument(
        "--confirm-baseline-write", action="store_true",
        help="Deliberately create baseline paths if absent; accepted production source hashes are checked.",
    )
    parser.add_argument(
        "--refresh-supplement", action="store_true",
        help="Replace supplementary artifacts only; never changes original baseline files.",
    )
    args = parser.parse_args()
    if args.confirm_baseline_write and args.mode != "baseline":
        parser.error("--confirm-baseline-write is valid only with --mode baseline")
    if args.refresh_supplement and args.mode != "supplement":
        parser.error("--refresh-supplement is valid only with --mode supplement")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "baseline":
            _baseline_mode(args.confirm_baseline_write)
        elif args.mode == "supplement":
            _supplement_mode(args.refresh_supplement)
        elif args.mode in {"after", "refactored"}:
            _after_mode()
        else:
            comparison = _compare_files()
            print(json.dumps({"comparison": "comparison.json", "baseline_cases": len(comparison["original_baseline"]["cases"])}, indent=2))
    except RuntimeError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
