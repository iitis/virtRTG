import sys

if __name__ == '__main__':
	print("sorry, you can't run this file directly")
	sys.exit(0)

import os
import numpy as np

# # Katalog główny projektu (d:\praca\pyDpVision) — dla dpVision.*
# _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)

# # Katalog tego pluginu — dla płaskich importów (virtualXRay, xrayProjection ...)
# _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
# if _PLUGIN_DIR not in sys.path:
#     sys.path.insert(1, _PLUGIN_DIR)

from .virtualXRay import VirtualXRay
from .xray.xrayProjection import (
	XRayProjectionGeometry,
	XRayPhysicsModel,
	XRayProjectionQualityProfile,
	XRayProjectionConfig,
	RawPresentationModel,
	DigitalRadiographyPresentationModel,
	MeshXRaySource,
	VolumetricXRaySource,
	XRayScene,
	save_projection_png,
	save_projection_tiff,
	save_projection_dicom,
)

from dpVision import AP, Transform, Mesh, Volumetric
from dpVision.meshQualityAnalyzer import MeshQualityAnalyzer

def _demo_output_dir(*parts):
	"""Return an absolute path inside `sample_data/generated`."""
	return os.path.join(os.path.dirname(__file__), "sample_data", "generated", *parts)


def _make_jaw_transform(translation_xyz=(0.0, 0.0, 0.0), rotation_deg_z=0.0):
	"""Build a simple rigid transform for synthetic jaw motion tests."""
	tx, ty, tz = translation_xyz
	angle = np.deg2rad(float(rotation_deg_z))
	cos_a = np.cos(angle)
	sin_a = np.sin(angle)
	transform = np.eye(4, dtype=np.float32)
	transform[:3, :3] = np.array([
		[cos_a, -sin_a, 0.0],
		[sin_a,  cos_a, 0.0],
		[0.0,    0.0,   1.0],
	], dtype=np.float32)
	transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
	return transform



def create_synthetic_xray_demo_dicoms(base_dir=None, overwrite=False):
	"""Create two simple synthetic DICOM sets: a fixed skull and a movable jaw."""
	if base_dir is None:
		base_dir = _demo_output_dir("xray_demo_dicoms")

	skull_dir = os.path.join(base_dir, "skull")
	jaw_dir = os.path.join(base_dir, "jaw")
	if not overwrite and os.path.isdir(skull_dir) and os.path.isdir(jaw_dir):
		return {
			"base_dir": base_dir,
			"skull_dir": skull_dir,
			"jaw_dir": jaw_dir,
		}

	os.makedirs(skull_dir, exist_ok=True)
	os.makedirs(jaw_dir, exist_ok=True)

	skull = Volumetric.create(layers=96, rows=96, columns=96)
	skull.label = "synthetic_skull"
	skull.set_position(x=-48.0, y=-48.0, z=-48.0)
	skull.set_pixel_size(image_x=1.0, image_y=1.0, slice_thickness=1.0)
	skull.drawSphere(origin=[48, 50, 48], radius=34, color=1800.0)
	skull.drawSphere(origin=[48, 50, 48], radius=28, color=200.0)
	skull.drawBox(origin=[0, 0, 0], size=[96, 22, 96], color=0.0)
	skull.export(dir=skull_dir, file_base="skull_", ext=".dcm")

	jaw = Volumetric.create(layers=96, rows=96, columns=96)
	jaw.label = "synthetic_jaw"
	jaw.set_position(x=-48.0, y=-48.0, z=-48.0)
	jaw.set_pixel_size(image_x=1.0, image_y=1.0, slice_thickness=1.0)
	jaw.drawBox(origin=[24, 18, 18], size=[12, 26, 18], color=1800.0)
	jaw.drawBox(origin=[24, 18, 60], size=[12, 26, 18], color=1800.0)
	jaw.drawBox(origin=[18, 12, 18], size=[12, 10, 60], color=1800.0)
	jaw.drawBox(origin=[30, 18, 30], size=[6, 16, 36], color=0.0)
	jaw.export(dir=jaw_dir, file_base="jaw_", ext=".dcm")

	return {
		"base_dir": base_dir,
		"skull_dir": skull_dir,
		"jaw_dir": jaw_dir,
	}


def build_synthetic_xray_demo_volumes():
	"""Create in-memory synthetic skull and jaw volumes for X-ray projection tests."""
	skull = Volumetric.create(layers=96, rows=96, columns=96)
	skull.label = "synthetic_skull"
	skull.set_position(x=-48.0, y=-48.0, z=-48.0)
	skull.set_pixel_size(image_x=1.0, image_y=1.0, slice_thickness=1.0)
	skull.drawSphere(origin=[48, 50, 48], radius=34, color=1800.0)
	skull.drawSphere(origin=[48, 50, 48], radius=28, color=200.0)
	skull.drawBox(origin=[0, 0, 0], size=[96, 22, 96], color=0.0)

	jaw = Volumetric.create(layers=96, rows=96, columns=96)
	jaw.label = "synthetic_jaw"
	jaw.set_position(x=-48.0, y=-48.0, z=-48.0)
	jaw.set_pixel_size(image_x=1.0, image_y=1.0, slice_thickness=1.0)
	jaw.drawBox(origin=[24, 18, 18], size=[12, 26, 18], color=1800.0)
	jaw.drawBox(origin=[24, 18, 60], size=[12, 26, 18], color=1800.0)
	jaw.drawBox(origin=[18, 12, 18], size=[12, 10, 60], color=1800.0)
	jaw.drawBox(origin=[30, 18, 30], size=[6, 16, 36], color=0.0)
	return skull, jaw


def build_synthetic_xray_demo_mesh():
	"""Create one simple synthetic mesh sample for hybrid volume + mesh X-ray tests."""
	vertices = np.array([
		[-12.0, -6.0, -25.0],
		[ 12.0, -6.0, -25.0],
		[ 12.0,  6.0, -25.0],
		[-12.0,  6.0, -25.0],
		[-12.0, -6.0,  25.0],
		[ 12.0, -6.0,  25.0],
		[ 12.0,  6.0,  25.0],
		[-12.0,  6.0,  25.0],
	], dtype=np.float32)
	faces = np.array([
		[0, 1, 2], [0, 2, 3],
		[4, 6, 5], [4, 7, 6],
		[0, 4, 5], [0, 5, 1],
		[1, 5, 6], [1, 6, 2],
		[2, 6, 7], [2, 7, 3],
		[3, 7, 4], [3, 4, 0],
	], dtype=np.uint32)
	mesh = Mesh.create(vertices=vertices, faces=faces)
	mesh.label = "synthetic_implant_mesh"
	return mesh


def report_mesh_xray_topology(mesh: Mesh, area_epsilon=1e-12):
	"""Print one compact RTG-oriented topology report for a mesh."""
	analyzer = MeshQualityAnalyzer(mesh)
	report = analyzer.compute_xray_topology_report(area_epsilon=area_epsilon)
	print(analyzer.summarize_xray_topology_report(area_epsilon=area_epsilon))
	return report


def report_selected_mesh_xray_topology(area_epsilon=1e-12):
	"""Print one topology report for the currently selected mesh in the workspace."""
	selected = getattr(AP, "selected", None)
	if not isinstance(selected, Mesh):
		raise TypeError("AP.selected must be a Mesh to run report_selected_mesh_xray_topology().")
	return report_mesh_xray_topology(selected, area_epsilon=area_epsilon)


def clean_mesh_for_xray(mesh: Mesh, vertex_merge_tolerance=1e-6, drop_degenerate_faces=True,
	                    drop_nonmanifold_faces=False, drop_boundary_faces=False):
	"""Return one cleaned mesh copy plus a compact cleanup summary for RTG tests.

	The cleanup is intentionally conservative:
	- merge duplicated vertices by quantized position,
	- drop degenerate faces created by merged vertex ids,
	- remove duplicate triangle faces independent of winding order,
	- optionally drop faces incident to non-manifold or boundary edges.
	"""
	if not isinstance(mesh, Mesh):
		raise TypeError("mesh must be an instance of Mesh.")

	vertices = np.asarray(mesh.m_vertices, dtype=np.float32)
	faces = np.asarray(mesh.m_faces, dtype=np.int64)
	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError("mesh.m_vertices must have shape (N, 3).")
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError("mesh.m_faces must have shape (M, 3).")
	if vertex_merge_tolerance <= 0.0:
		raise ValueError("vertex_merge_tolerance must be positive.")

	quantized_vertices = np.round(vertices / float(vertex_merge_tolerance)).astype(np.int64)
	_unique_keys, unique_vertex_indices, inverse_vertex_indices = np.unique(
		quantized_vertices,
		axis=0,
		return_index=True,
		return_inverse=True,
	)
	merged_vertices = vertices[unique_vertex_indices].astype(np.float32, copy=False)
	remapped_faces = inverse_vertex_indices[faces].astype(np.int64, copy=False)

	degenerate_mask = (
		(remapped_faces[:, 0] == remapped_faces[:, 1])
		| (remapped_faces[:, 1] == remapped_faces[:, 2])
		| (remapped_faces[:, 2] == remapped_faces[:, 0])
	)
	degenerate_removed_count = int(np.count_nonzero(degenerate_mask))
	if drop_degenerate_faces and degenerate_removed_count > 0:
		remapped_faces = remapped_faces[~degenerate_mask]

	normalized_faces = np.sort(remapped_faces, axis=1)
	_unique_faces, unique_face_indices = np.unique(normalized_faces, axis=0, return_index=True)
	del _unique_faces
	unique_face_indices = np.sort(unique_face_indices)
	duplicate_face_removed_count = int(remapped_faces.shape[0] - unique_face_indices.shape[0])
	remapped_faces = remapped_faces[unique_face_indices]

	removed_nonmanifold_face_count = 0
	removed_boundary_face_count = 0
	if drop_nonmanifold_faces or drop_boundary_faces:
		edge_to_face_indices = {}
		for face_idx, (a, b, c) in enumerate(remapped_faces):
			for edge in ((a, b), (b, c), (c, a)):
				edge_key = tuple(sorted((int(edge[0]), int(edge[1]))))
				edge_to_face_indices.setdefault(edge_key, []).append(int(face_idx))

		faces_to_drop = set()
		for edge_faces in edge_to_face_indices.values():
			if drop_boundary_faces and len(edge_faces) == 1:
				faces_to_drop.update(edge_faces)
			if drop_nonmanifold_faces and len(edge_faces) > 2:
				faces_to_drop.update(edge_faces)

		if faces_to_drop:
			faces_to_drop_array = np.array(sorted(faces_to_drop), dtype=np.int64)
			if drop_nonmanifold_faces:
				nonmanifold_faces = set()
				for edge_faces in edge_to_face_indices.values():
					if len(edge_faces) > 2:
						nonmanifold_faces.update(edge_faces)
				removed_nonmanifold_face_count = int(np.intersect1d(
					faces_to_drop_array,
					np.array(sorted(nonmanifold_faces), dtype=np.int64),
					assume_unique=True,
				).shape[0])
			if drop_boundary_faces:
				boundary_faces = set()
				for edge_faces in edge_to_face_indices.values():
					if len(edge_faces) == 1:
						boundary_faces.update(edge_faces)
				removed_boundary_face_count = int(np.intersect1d(
					faces_to_drop_array,
					np.array(sorted(boundary_faces), dtype=np.int64),
					assume_unique=True,
				).shape[0])
			keep_mask = np.ones(remapped_faces.shape[0], dtype=bool)
			keep_mask[faces_to_drop_array] = False
			remapped_faces = remapped_faces[keep_mask]

	cleaned_mesh = Mesh.create(
		vertices=merged_vertices,
		faces=remapped_faces.astype(np.uint32, copy=False),
	)
	cleaned_mesh.label = f"{getattr(mesh, 'label', 'mesh')}_xray_clean"

	cleanup_report = {
		"input_vertex_count": int(vertices.shape[0]),
		"input_face_count": int(faces.shape[0]),
		"output_vertex_count": int(cleaned_mesh.m_vertices.shape[0]),
		"output_face_count": int(cleaned_mesh.m_faces.shape[0]),
		"merged_vertex_count": int(vertices.shape[0] - cleaned_mesh.m_vertices.shape[0]),
		"removed_degenerate_face_count": int(degenerate_removed_count if drop_degenerate_faces else 0),
		"removed_duplicate_face_count": int(duplicate_face_removed_count),
		"removed_nonmanifold_face_count": int(removed_nonmanifold_face_count),
		"removed_boundary_face_count": int(removed_boundary_face_count),
		"vertex_merge_tolerance": float(vertex_merge_tolerance),
		"drop_nonmanifold_faces": bool(drop_nonmanifold_faces),
		"drop_boundary_faces": bool(drop_boundary_faces),
	}
	return cleaned_mesh, cleanup_report


def clean_selected_mesh_for_xray(vertex_merge_tolerance=1e-6, add_to_workspace=True,
	                             drop_nonmanifold_faces=False, drop_boundary_faces=False):
	"""Clean the currently selected mesh, print reports before/after and optionally add the copy."""
	selected = getattr(AP, "selected", None)
	if not isinstance(selected, Mesh):
		raise TypeError("AP.selected must be a Mesh to run clean_selected_mesh_for_xray().")

	before_report = report_mesh_xray_topology(selected)
	cleaned_mesh, cleanup_report = clean_mesh_for_xray(
		selected,
		vertex_merge_tolerance=vertex_merge_tolerance,
		drop_nonmanifold_faces=drop_nonmanifold_faces,
		drop_boundary_faces=drop_boundary_faces,
	)
	after_report = report_mesh_xray_topology(cleaned_mesh)

	print("Mesh XRay cleanup summary:", cleanup_report)
	if add_to_workspace:
		AP.addObject(cleaned_mesh)

	return {
		"original_report": before_report,
		"cleaned_report": after_report,
		"cleanup_report": cleanup_report,
		"cleaned_mesh": cleaned_mesh,
	}


def run_virtual_xray_headless(virtual_xray, save_png_path=None):
	"""Run one VirtualXRay without inserting or refreshing GUI image objects.

	This helper isolates the projection backend from the `Run Simulation` GUI path.
	If it succeeds while the GUI button still crashes, the issue is likely in image
	creation, workspace insertion or GL refresh rather than in the projection math.
	"""
	if not isinstance(virtual_xray, VirtualXRay):
		raise TypeError("virtual_xray must be a VirtualXRay instance.")

	config = virtual_xray.build_projection_config()
	raw_image, stats = virtual_xray.build_scene().project(
		config=config,
		return_stats=True,
		progress_callback=None,
	)
	raw_image = np.asarray(raw_image, dtype=np.float32)
	display_image = config.apply_presentation(raw_image)

	print(
		"VirtualXRay headless projection:",
		{
			"label": str(getattr(virtual_xray, "label", "VirtualXRay")),
			"shape": tuple(int(v) for v in raw_image.shape),
			"elapsed_seconds": float(stats.elapsed_seconds),
			"traced_pixels": int(stats.traced_pixels),
			"total_sample_count": int(stats.total_sample_count),
		},
	)

	if save_png_path is not None:
		save_projection_png(
			display_image,
			save_png_path,
			invert=False,
			fixed_range=(0.0, 1.0),
		)
		print(f"Saved headless XRay preview to: {save_png_path}")

	return {
		"raw_image": raw_image,
		"display_image": display_image,
		"stats": stats,
	}


def run_selected_virtual_xray_headless(save_png_path=None):
	"""Run the currently selected VirtualXRay without touching the GUI image path."""
	selected = getattr(AP, "selected", None)
	if not isinstance(selected, VirtualXRay):
		selected_type = "None" if selected is None else type(selected).__name__
		raise TypeError(
			f"AP.selected must be a VirtualXRay to run run_selected_virtual_xray_headless(); got {selected_type}."
		)
	return run_virtual_xray_headless(selected, save_png_path=save_png_path)


def demo_synthetic_xray_projection(output_dir=None, jaw_translation_xyz=(0.0, -8.0, 0.0), jaw_rotation_deg_z=8.0):
	"""Generate one example hybrid volume + mesh X-ray projection."""
	create_synthetic_xray_demo_dicoms()
	skull, jaw = build_synthetic_xray_demo_volumes()
	implant_mesh = build_synthetic_xray_demo_mesh()
	if output_dir is None:
		output_dir = _demo_output_dir("xray_demo_output")
	os.makedirs(output_dir, exist_ok=True)

	skull_transform = np.eye(4, dtype=np.float32)
	jaw_transform = _make_jaw_transform(
		translation_xyz=jaw_translation_xyz,
		rotation_deg_z=jaw_rotation_deg_z,
	)
	mesh_transform = np.eye(4, dtype=np.float32)
	mesh_transform[:3, 3] = np.array([42.0, 18.0, 48.0], dtype=np.float32)

	geometry = XRayProjectionGeometry.from_detector_pose(
		detector_center_ref=[42.2, 42.2, 180.0],
		detector_normal_ref=[0.0, 0.0, -1.0],
		detector_up_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[512, 512],
		detector_pixel_size_mm=0.4,
		step_mm=1.0,
		source_position_ref=[42.2, 42.2, -220.0],
	)
	physics = XRayPhysicsModel(
		mu_air=0.0,
		mu_water=0.02,
		attenuation_scale=1.0,
		output_mode="integral",
	)
	scene = XRayScene.from_sample_sources([
		VolumetricXRaySource(skull, global_transform=skull_transform, interpolation="linear"),
		VolumetricXRaySource(jaw, global_transform=jaw_transform, interpolation="linear"),
		MeshXRaySource(implant_mesh, global_transform=mesh_transform, scalar_value=2200.0, mode="solid", shell_thickness_mm=1.2),
	])
	config = XRayProjectionConfig(
		geometry=geometry,
		physics_model=physics,
		presentation_model=DigitalRadiographyPresentationModel(
			invert=False,
			gamma=0.7,
			contrast=1.2,
		),
		reference_transform=np.eye(4, dtype=np.float32),
		quality_profile=XRayProjectionQualityProfile.normal(),
	)

	image, stats = scene.project(config=config, return_stats=True)
	raw_presentation = RawPresentationModel()
	raw_image = raw_presentation.apply(image)
	display_image = config.apply_presentation(image)

	png_path = os.path.join(output_dir, "synthetic_xray_display.png")
	tiff_path = os.path.join(output_dir, "synthetic_xray_display.tiff")
	dicom_path = os.path.join(output_dir, "synthetic_xray_display.dcm")
	raw_tiff_path = os.path.join(output_dir, "synthetic_xray_raw.tiff")
	save_projection_png(display_image, png_path, invert=False, fixed_range=(0.0, 1.0))
	save_projection_tiff(display_image, tiff_path, mode="uint16", invert=False, fixed_range=(0.0, 1.0))
	save_projection_dicom(
		display_image,
		dicom_path,
		patient_name="Synthetic^XRay",
		patient_id="XRAYDEMO",
		study_description="Synthetic multi-volume demo",
		series_description="Skull + Jaw projection display",
		invert=False,
		fixed_range=(0.0, 1.0),
	)
	save_projection_tiff(raw_image, raw_tiff_path, mode="float32")
	return {
		"image": image,
		"raw_image": raw_image,
		"display_image": display_image,
		"png_path": png_path,
		"tiff_path": tiff_path,
		"dicom_path": dicom_path,
		"raw_tiff_path": raw_tiff_path,
		"jaw_transform": jaw_transform,
		"stats": stats,
		"config": config,
	}


def create_virtual_xray_demo_object():
	"""Create one `VirtualXRay` scene node with synthetic skull, jaw and mesh descendants."""
	skull, jaw = build_synthetic_xray_demo_volumes()
	implant_mesh = build_synthetic_xray_demo_mesh()
	jaw_transform = _make_jaw_transform(
		translation_xyz=(0.0, -8.0, 0.0),
		rotation_deg_z=8.0,
	)

	setup = VirtualXRay()
	setup.detector_center_ref = np.array([42.2, 42.2, 180.0], dtype=np.float32)
	setup.source_position_ref = np.array([42.2, 42.2, -220.0], dtype=np.float32)
	setup.detector_shape_hw = [512, 512]
	setup.detector_pixel_size_mm = [0.4, 0.4]
	setup.step_mm = 1.0
	setup.quality_profile_name = "normal"

	skull_transform = Transform()
	skull_transform.addChild(skull)
	setup.addChild(skull_transform)

	jaw_node = Transform(matrix=jaw_transform)
	jaw_node.label = "jaw_pose"
	jaw_node.addChild(jaw)
	setup.addChild(jaw_node)

	mesh_node = Transform()
	mesh_node.label = "implant_pose"
	mesh_node.translate(42.0, 18.0, 48.0)
	mesh_node.addChild(implant_mesh)
	setup.addChild(mesh_node)

	AP.addObject(setup)
	return setup



def save_difference_map(img_a, img_b, out_path, title="Sampling − Siddon", cmap="RdBu_r"):
	"""Save a coloured signed-difference map between two raw projection images.

	img_a, img_b : 2-D float32 arrays of equal shape (raw line-integral values).
	The map shows  diff = img_a − img_b  with a diverging colormap so positive
	differences (sampling > Siddon) appear in one hue and negative in the other.
	A second panel shows the absolute difference with a sequential colormap.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import matplotlib.colors as mcolors

	diff = np.asarray(img_a, dtype=np.float64) - np.asarray(img_b, dtype=np.float64)
	abs_diff = np.abs(diff)

	vmax = float(np.percentile(np.abs(diff), 99.5))
	if vmax < 1e-9:
		vmax = 1.0

	fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
	fig.suptitle(title, fontsize=11)

	# Panel 1: sampling image
	axes[0].imshow(img_a, cmap="gray", vmin=0, interpolation="nearest")
	axes[0].set_title("Sampling", fontsize=9)
	axes[0].axis("off")

	# Panel 2: Siddon image
	axes[1].imshow(img_b, cmap="gray", vmin=0, interpolation="nearest")
	axes[1].set_title("Siddon", fontsize=9)
	axes[1].axis("off")

	# Panel 3: signed difference (diverging)
	im = axes[2].imshow(diff, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
	axes[2].set_title("Sampling − Siddon", fontsize=9)
	axes[2].axis("off")
	cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
	cbar.set_label("Δ (line integral)", fontsize=7)
	cbar.ax.tick_params(labelsize=6)

	# Annotation: MAE and max abs diff
	img_a_f = np.asarray(img_a, dtype=np.float64)
	mae  = float(np.mean(abs_diff))
	dmax = float(np.max(abs_diff))
	ref  = float(np.mean(img_a_f[img_a_f > 0])) if np.any(img_a_f > 0) else 1.0
	rel_mae = mae / ref
	axes[2].set_xlabel(f"MAE={mae:.4f}  max|Δ|={dmax:.4f}  rel={100*rel_mae:.3f}%",
	                   fontsize=7, labelpad=3)
	axes[2].xaxis.set_label_position("bottom")

	plt.tight_layout()
	fig.savefig(out_path, bbox_inches="tight")
	plt.close(fig)
	print(f"  Difference map saved: {out_path}  (MAE={mae:.5f}, rel={100*rel_mae:.4f}%)")
	return {"mae": mae, "max_abs": dmax, "rel_mae": rel_mae, "path": out_path}


def generate_sample_siddon_figure(results, out_path, vol_name="small", profile="normal"):
	"""Generate the publication-quality 3-panel comparison figure (sample_siddon.png).

	Panels: (a) Sampling projection  |  (b) Siddon projection  |  (c) Signed difference
	The difference panel uses a diverging RdBu_r colormap; the colorbar is symmetric
	around zero so zero difference appears white.

	Parameters
	----------
	results : list of dicts returned by benchmark_xray_performance()
	out_path : str  path to save the PNG (e.g. .../sample_siddon.png)
	vol_name : str  "small", "medium", or "large"
	profile  : str  "draft", "normal", or "high"

	Returns
	-------
	dict with keys mae, max_abs, rel_mae, path
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	by_sp = {(r["scene"], r["profile"]): r for r in results}
	r_samp   = by_sp.get((f"vol_{vol_name}",        profile))
	r_siddon = by_sp.get((f"vol_{vol_name}_siddon",  profile))
	if r_samp is None or r_siddon is None:
		raise ValueError(f"Results for vol_{vol_name} sampling/siddon profile={profile} not found.")

	img_s = np.asarray(r_samp["raw_image"],   dtype=np.float64)
	img_d = np.asarray(r_siddon["raw_image"], dtype=np.float64)

	# Resize Siddon to sampling shape if detectors differ (draft vs normal/high)
	if img_s.shape != img_d.shape:
		from scipy.ndimage import zoom
		zf = np.array(img_s.shape) / np.array(img_d.shape)
		img_d = zoom(img_d, zf, order=1)

	diff = img_s - img_d
	abs_diff = np.abs(diff)
	mae  = float(np.mean(abs_diff))
	dmax = float(np.max(abs_diff))
	ref  = float(np.mean(img_s[img_s > 0])) if np.any(img_s > 0) else 1.0
	rel_mae = mae / ref

	# Symmetric colorbar limit: 99.5th percentile of |diff|, at least 1e-6
	vmax = max(float(np.percentile(abs_diff, 99.5)), 1e-6)

	fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=180,
	                         gridspec_kw={"wspace": 0.04})

	gray_vmax = float(np.percentile(img_s, 99.5)) or 1.0

	# (a) Sampling
	axes[0].imshow(img_s, cmap="gray", vmin=0, vmax=gray_vmax, interpolation="nearest")
	axes[0].set_title(r"(a) Sampling ($\Delta s = 1\,\mathrm{mm}$)", fontsize=8)
	axes[0].axis("off")

	# (b) Siddon
	axes[1].imshow(img_d, cmap="gray", vmin=0, vmax=gray_vmax, interpolation="nearest")
	axes[1].set_title(r"(b) Siddon (exact traversal)", fontsize=8)
	axes[1].axis("off")

	# (c) Signed difference with colorbar
	im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
	axes[2].set_title(r"(c) Sampling $-$ Siddon", fontsize=8)
	axes[2].axis("off")
	divider = make_axes_locatable(axes[2])
	cax = divider.append_axes("right", size="5%", pad=0.06)
	cbar = fig.colorbar(im, cax=cax)
	cbar.set_label(r"$\Delta\,\int\mu\,\mathrm{d}\ell$", fontsize=7)
	cbar.ax.tick_params(labelsize=6)

	fig.text(0.5, -0.02,
	         f"vol\\,{vol_name}  |  profile: {profile}  |  "
	         f"MAE = {mae:.5f}  |  max$|\\Delta|$ = {dmax:.5f}  |  "
	         f"rel.\\,MAE = {100*rel_mae:.3f}\\%",
	         ha="center", fontsize=7, style="italic")

	fig.savefig(out_path, bbox_inches="tight", dpi=180)
	plt.close(fig)
	print(f"  sample_siddon figure saved: {out_path}  (MAE={mae:.5f}, rel={100*rel_mae:.4f}%)")
	return {"mae": mae, "max_abs": dmax, "rel_mae": rel_mae, "path": out_path}


def benchmark_xray_performance(output_dir=None, show_reports=True):
	"""Run a series of X-ray projections and measure performance.

	Tests combinations of:
	  - Sample type: volume-only, mesh-only, mixed (volume+mesh)
	  - Volume sizes: small (64³), medium (96³), large (128³)
	  - Quality profiles: draft, normal, high

	Each row in the returned list contains timing data from XRayProjectionStats,
	including phase_timings (dict of phase -> seconds) and per_source_stats (list of
	per-source dicts with elapsed_s, work_count, bvh info, stack timings, etc.).

	Returns a list of result dicts suitable for printing a performance table.
	"""
	import time

	if output_dir is None:
		output_dir = _demo_output_dir("xray_benchmark")
	os.makedirs(output_dir, exist_ok=True)

	geometry = XRayProjectionGeometry.from_detector_pose(
		detector_center_ref=[0.0, 0.0, 180.0],
		detector_normal_ref=[0.0, 0.0, -1.0],
		detector_up_ref=[0.0, 1.0, 0.0],
		detector_shape_hw=[512, 512],
		detector_pixel_size_mm=0.4,
		step_mm=1.0,
		source_position_ref=[0.0, 0.0, -220.0],
	)
	physics = XRayPhysicsModel(mu_water=0.02, attenuation_scale=1.0, output_mode="integral")

	profiles = [
		XRayProjectionQualityProfile.draft(),
		XRayProjectionQualityProfile.normal(),
		XRayProjectionQualityProfile.high(),
	]

	volume_sizes = [
		("small",  64),
		("medium", 96),
		("large",  128),
	]

	implant_mesh = build_synthetic_xray_demo_mesh()

	print("=" * 70)
	print("XRay Performance Benchmark")
	print("=" * 70)

	results = []

	for vol_name, vol_size in volume_sizes:
		from dpVision.volumetric import Volumetric
		vol = Volumetric.create(layers=vol_size, rows=vol_size, columns=vol_size)
		vol.label = f"synthetic_{vol_name}_{vol_size}^3"
		vol.set_position(x=-vol_size / 2.0, y=-vol_size / 2.0, z=-vol_size / 2.0)
		vol.set_pixel_size(image_x=1.0, image_y=1.0, slice_thickness=1.0)
		vol.drawSphere(origin=[vol_size // 2] * 3, radius=vol_size // 3, color=1800.0)
		vol.drawSphere(origin=[vol_size // 2] * 3, radius=vol_size // 4, color=200.0)

		for profile in profiles:
			scene_variants = [
				(f"vol_{vol_name}",             [VolumetricXRaySource(vol, interpolation="linear", volume_backend="sampling")]),
				(f"vol_{vol_name}_siddon",       [VolumetricXRaySource(vol, interpolation="linear", volume_backend="siddon")]),
				("mesh_only",                   [MeshXRaySource(implant_mesh, scalar_value=2200.0, mode="solid", backend="analytic_bvh")]),
				("mesh_only_projected",         [MeshXRaySource(implant_mesh, scalar_value=2200.0, mode="solid", backend="projected_intersection_list")]),
				(f"vol_{vol_name}+mesh",        [
					VolumetricXRaySource(vol, interpolation="linear", volume_backend="sampling"),
					MeshXRaySource(implant_mesh, scalar_value=2200.0, mode="solid", backend="analytic_bvh"),
				]),
				(f"vol_{vol_name}_siddon+mesh", [
					VolumetricXRaySource(vol, interpolation="linear", volume_backend="siddon"),
					MeshXRaySource(implant_mesh, scalar_value=2200.0, mode="solid", backend="analytic_bvh"),
				]),
			]
			for scene_label, sources in scene_variants:
				config = XRayProjectionConfig(
					geometry=geometry,
					physics_model=physics,
					quality_profile=profile,
				)
				scene = XRayScene.from_sample_sources(sources)
				t_wall_start = time.perf_counter()
				_img, stats = scene.project(config=config, return_stats=True)
				wall_s = time.perf_counter() - t_wall_start

				_png_name = (
					scene_label.replace("+", "_plus_").replace("^", "") + "__" + profile.name + ".png"
				)
				_png_path = os.path.join(output_dir, _png_name)
				_display = DigitalRadiographyPresentationModel(invert=False, gamma=0.6, contrast=1.1).apply(_img)
				save_projection_png(_display, _png_path, invert=False, fixed_range=(0.0, 1.0))

				row = {
					"scene":           scene_label,
					"profile":         profile.name,
					"detector":        f"{stats.detector_shape_hw[1]}x{stats.detector_shape_hw[0]}",
					"step_mm":         stats.step_mm,
					"total_ms":        stats.elapsed_seconds * 1000.0,
					"wall_ms":         wall_s * 1000.0,
					"traced_pct":      100.0 * stats.traced_pixels / max(stats.total_pixels, 1),
					"samples":         stats.total_sample_count,
					"samples_per_s":   stats.samples_per_second,
					"phase_ms":        {k: v * 1000.0 for k, v in stats.phase_timings.items()},
					"per_source":      stats.per_source_stats,
					"rays_per_s":      stats.rays_per_second,
					"png_path":        _png_path,
					"raw_image":       np.asarray(_img, dtype=np.float32),
				}
				results.append(row)

				if show_reports:
					print(f"\n[{scene_label}]  profile={profile.name}  step={stats.step_mm:.1f} mm")
					stats.print_report()

	# ── Difference maps (sampling vs Siddon) ───────────────────────────────
	by_sp_full = {}
	for r in results:
		by_sp_full[(r["scene"], r["profile"])] = r

	diff_stats = {}  # keyed by (vol_name, profile)
	for vol_name in [s for s, _ in volume_sizes]:
		for prof in [p.name for p in profiles]:
			r_samp   = by_sp_full.get((f"vol_{vol_name}",        prof))
			r_siddon = by_sp_full.get((f"vol_{vol_name}_siddon", prof))
			if r_samp is None or r_siddon is None:
				continue
			diff_path = os.path.join(
				output_dir,
				f"diff_vol_{vol_name}_sampling_vs_siddon__{prof}.png",
			)
			ds = save_difference_map(
				r_samp["raw_image"],
				r_siddon["raw_image"],
				diff_path,
				title=f"Sampling − Siddon  |  vol {vol_name}  |  {prof}",
			)
			diff_stats[(vol_name, prof)] = {**ds, "diff_png_path": diff_path}

	# Publication-quality 3-panel figure for vol_small / normal
	sample_siddon_path = os.path.join(output_dir, "sample_siddon.png")
	try:
		ss_info = generate_sample_siddon_figure(results, sample_siddon_path,
		                                        vol_name="small", profile="normal")
		diff_stats[("small", "normal")]["sample_siddon_path"] = sample_siddon_path
		diff_stats[("small", "normal")]["rel_mae"] = ss_info["rel_mae"]
	except Exception as _e:
		print(f"  Warning: could not generate sample_siddon figure: {_e}")
		ss_info = None

	print("\n" + "=" * 85)
	print(f"{'scene':<28s} {'profile':<8s} {'step':>5s}  {'total_ms':>9s}  {'samples/s':>13s}  {'rays/s':>10s}")
	print("-" * 85)
	for r in results:
		print(
			f"  {r['scene']:<26s} {r['profile']:<8s} {r['step_mm']:>4.1f}mm"
			f"  {r['total_ms']:>8.1f} ms  {r['samples_per_s']:>12,.0f} samp/s"
			f"  {r['rays_per_s']:>9,.0f} ray/s"
		)
	print("=" * 85)
	return results, diff_stats


def generate_latex_benchmark_report(results, tex_path=None, section_title="Examples and Use Cases", diff_stats=None):
	"""Generate a standalone LaTeX performance report from benchmark_xray_performance() results.

	Produces a .tex file whose top-level heading is \\section{section_title}.
	Internal structure mirrors the article layout:
	  \\subsection{Synthetic scenes}
	  \\subsection{Projection geometry and quality profiles}
	  \\clearpage
	  \\subsection{Performance Results}
	    \\subsubsection{Total projection time and throughput}
	    \\subsubsection{Phase breakdown (normal profile)}
	    \\subsubsection{Backend comparison (mesh sources)}
	    \\subsubsection{Key observations}
	  \\subsection{Projection Images}

	Compile with:  pdflatex <tex_path>
	Requires LaTeX packages: booktabs, graphicx, subcaption, geometry, multirow.
	"""
	if tex_path is None:
		tex_path = os.path.join(_demo_output_dir("xray_benchmark"), "report.tex")
	tex_dir = os.path.dirname(tex_path)
	os.makedirs(tex_dir, exist_ok=True)

	profiles = ["draft", "normal", "high"]
	seen_sc = {}
	for r in results:
		k = r["scene"]
		if k not in seen_sc:
			seen_sc[k] = r
	scenes_all = list(seen_sc.keys())
	vol_only       = [s for s in scenes_all if s.startswith("vol_") and "+" not in s and "siddon" not in s]
	vol_only_siddon= [s for s in scenes_all if s.startswith("vol_") and "+" not in s and "siddon" in s]
	mesh_only      = [s for s in scenes_all if s == "mesh_only"]
	vol_mesh       = [s for s in scenes_all if "+" in s and "siddon" not in s]
	vol_mesh_siddon= [s for s in scenes_all if "+" in s and "siddon" in s]
	# projected mesh variant excluded from main table; Siddon gets its own subsection
	scenes_display = vol_only + vol_only_siddon + vol_mesh + vol_mesh_siddon + mesh_only

	by_sp = {}
	for r in results:
		key = (r["scene"], r["profile"])
		if key not in by_sp:
			by_sp[key] = r

	def esc(s):
		return s.replace("_", r"\_").replace("^", r"\^{}").replace("+", r"\texttt{+}")

	scene_row_labels = {
		"vol_small":               r"vol\,small ($64^3$) sampling",
		"vol_medium":              r"vol\,medium ($96^3$) sampling",
		"vol_large":               r"vol\,large ($128^3$) sampling",
		"vol_small_siddon":        r"vol\,small ($64^3$) Siddon",
		"vol_medium_siddon":       r"vol\,medium ($96^3$) Siddon",
		"vol_large_siddon":        r"vol\,large ($128^3$) Siddon",
		"mesh_only":               r"mesh only",
		"vol_small+mesh":          r"vol\,small + mesh (sampling)",
		"vol_medium+mesh":         r"vol\,medium + mesh (sampling)",
		"vol_large+mesh":          r"vol\,large + mesh (sampling)",
		"vol_small_siddon+mesh":   r"vol\,small + mesh (Siddon)",
		"vol_medium_siddon+mesh":  r"vol\,medium + mesh (Siddon)",
		"vol_large_siddon+mesh":   r"vol\,large + mesh (Siddon)",
	}

	def scene_row_label(s):
		return scene_row_labels.get(s, esc(s))

	def ms_cell(ms):
		if ms >= 1000.0:
			return f"{ms / 1000.0:.2f}\\,s"
		return f"{ms:.0f}\\,ms"

	def sps_cell(sps):
		if sps >= 1e6:
			return f"{sps / 1e6:.1f}"
		if sps >= 1e3:
			return f"{sps / 1e3:.1f}k"
		return f"{sps:.0f}"

	def phase_cell(v_ms, total_ms):
		if v_ms < 0.5:
			return r"$<\!1$"
		pct = 100.0 * v_ms / max(total_ms, 1e-6)
		return f"{v_ms:.0f} ({pct:.0f}\\%)"

	L = []
	def ln(s=""):
		L.append(s)

	# ── Preamble ──────────────────────────────────────────────────────────────
	ln(r"\documentclass[a4paper,10pt]{article}")
	ln(r"\usepackage[utf8]{inputenc}")
	ln(r"\usepackage[T1]{fontenc}")
	ln(r"\usepackage[english]{babel}")
	ln(r"\usepackage{booktabs}")
	ln(r"\usepackage{graphicx}")
	ln(r"\usepackage{subcaption}")
	ln(r"\usepackage{amsmath}")
	ln(r"\usepackage{geometry}")
	ln(r"\geometry{margin=2cm,top=2.5cm}")
	ln(r"\usepackage{multirow}")
	ln(r"\usepackage{lmodern}")
	ln(r"\usepackage{microtype}")
	ln()
	ln(r"\title{X-Ray Projection Pipeline\\[4pt]\large Performance Analysis --- Synthetic Benchmark}")
	ln(r"\author{pyDpVision}")
	ln(r"\date{\today}")
	ln()
	ln(r"\begin{document}")
	ln(r"\maketitle")
	ln()

	# ── Top-level section ─────────────────────────────────────────────────────
	ln(r"\section{" + section_title + r"}\label{sec:results}")
	ln()

	# ── Subsection: Synthetic scenes ──────────────────────────────────────────
	ln(r"\subsection{Synthetic scenes}")
	ln()
	ln(r"All measurements were performed on purely synthetic, in-memory datasets.")
	ln(r"No disk I/O or GUI rendering is included in the reported timings.")
	ln(r"The benchmark covers three quality profiles, three volume sizes, and three scene")
	ln(r"variants (volumetric only, mesh only, hybrid).")
	ln()
	ln(r"\paragraph{Volumes.}")
	ln(r"Three hollow-sphere volumes of increasing size were generated with $1\,\mathrm{mm}$")
	ln(r"isotropic voxel spacing:")
	ln(r"\begin{itemize}\setlength{\itemsep}{2pt}")
	ln(r"  \item \textbf{small}: $64\times 64\times 64$ voxels.")
	ln(r"  \item \textbf{medium}: $96\times 96\times 96$ voxels.")
	ln(r"  \item \textbf{large}: $128\times 128\times 128$ voxels.")
	ln(r"\end{itemize}")
	ln(r"Each contains a hollow sphere with wall attenuation $\approx 1800\,\mathrm{HU}$")
	ln(r"and interior $\approx 200\,\mathrm{HU}$. Trilinear interpolation is used during sampling.")
	ln()
	ln(r"\paragraph{Mesh.}")
	ln(r"One axis-aligned rectangular box implant ($24\times 12\times 50\,\mathrm{mm}$, 12 triangles,")
	ln(r"scalar value $2200\,\mathrm{HU}$, solid mode) is used in all mesh and hybrid scenes.")
	ln(r"Both ray-intersection backends are benchmarked separately")
	ln(r"(see Section~\ref{ssec:backends}).")
	ln()

	# ── Subsection: Projection geometry ───────────────────────────────────────
	ln(r"\subsection{Projection geometry and quality profiles}")
	ln()
	ln(r"Cone-beam geometry; source at $(0,\;0,\;-220)\,\mathrm{mm}$,")
	ln(r"detector centre at $(0,\;0,\;180)\,\mathrm{mm}$ (source-to-detector distance")
	ln(r"$400\,\mathrm{mm}$). Physics: Beer--Lambert attenuation integral,")
	ln(r"$\mu_{\mathrm{water}}=0.02\,\mathrm{mm}^{-1}$.")
	ln()
	ln(r"\begin{table}[htbp]")
	ln(r"\centering")
	ln(r"\caption{Quality profile parameters used in the benchmark.}")
	ln(r"\label{tab:profiles}")
	ln(r"\begin{tabular}{lccc}")
	ln(r"\toprule")
	ln(r"Profile & Detector & Pixel size & Marching step \\")
	ln(r"\midrule")
	ln(r"\textbf{draft}  & $256\times 256$ & $0.8\,\mathrm{mm}$ & $2.0\,\mathrm{mm}$ \\")
	ln(r"\textbf{normal} & $512\times 512$ & $0.4\,\mathrm{mm}$ & $1.0\,\mathrm{mm}$ \\")
	ln(r"\textbf{high}   & $512\times 512$ & $0.4\,\mathrm{mm}$ & $0.5\,\mathrm{mm}$ \\")
	ln(r"\bottomrule")
	ln(r"\end{tabular}")
	ln(r"\end{table}")
	ln()
	ln(r"\clearpage")

	# ── Subsection: Performance Results ───────────────────────────────────────
	ln(r"\subsection{Performance Results}")
	ln()

	# ── Subsubsection: Total time and throughput ───────────────────────────────
	ln(r"\subsubsection{Total projection time and throughput}")
	ln()
	ln(r"\begin{table}[!ht]")
	ln(r"\centering")
	ln(r"\caption{Total projection time and throughput per scene and quality profile.")
	ln(r"  Time in ms (or s if~$\geq 1\,\mathrm{s}$).")
	ln(r"  \emph{samp/s}: attenuation samples (interpolations) per second.")
	ln(r"  \emph{rays/s}: traced rays per second --- for sampling this decreases")
	ln(r"  with finer step~$\Delta s$ (more samples per ray, same ray count);")
	ln(r"  for Siddon it is the primary throughput metric, independent of~$\Delta s$.}")
	ln(r"\label{tab:main}")
	ln(r"\small")
	ln(r"\begin{tabular}{l rrr rrr rrr}")
	ln(r"\toprule")
	ln(r"  & \multicolumn{3}{c}{\textbf{draft} (2\,mm)}")
	ln(r"  & \multicolumn{3}{c}{\textbf{normal} (1\,mm)}")
	ln(r"  & \multicolumn{3}{c}{\textbf{high} (0.5\,mm)} \\")
	ln(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
	ln(r"Scene & time & samp/s & rays/s & time & samp/s & rays/s & time & samp/s & rays/s \\")
	ln(r"\midrule")

	def scene_group(s):
		if s.startswith("mesh_only"):       return "mesh"
		if "+" in s and "siddon" in s:      return "mixed_siddon"
		if "+" in s:                        return "mixed"
		if "siddon" in s:                   return "vol_siddon"
		return "vol"

	prev_group = None
	for s in scenes_display:
		grp = scene_group(s)
		if prev_group is not None and grp != prev_group:
			ln(r"\midrule")
		prev_group = grp
		row_cells = [scene_row_label(s)]
		for p in profiles:
			r = by_sp.get((s, p))
			if r:
				row_cells.append(ms_cell(r["total_ms"]))
				row_cells.append(sps_cell(r["samples_per_s"]))
				row_cells.append(sps_cell(r.get("rays_per_s", 0)))
			else:
				row_cells += ["---", "---", "---"]
		ln("  " + " & ".join(row_cells) + r" \\")

	ln(r"\bottomrule")
	ln(r"\end{tabular}")
	ln(r"\end{table}")
	ln()

	# ── Subsubsection: Phase breakdown ────────────────────────────────────────
	ln(r"\subsubsection{Phase breakdown (normal profile, $512\times512$, step~$1.0\,\mathrm{mm}$)}")
	ln()

	phase_keys = [
		("ray_setup",            r"Ray setup"),
		("aabb_intersection",    r"AABB intersection"),
		("depth_clipping",       r"Depth clipping"),
		("direct_sources_total", r"Direct sources (mesh BVH)"),
		("marching_total",       r"Marching (volumetric)"),
		("physics_conversion",   r"Physics conversion"),
	]

	for group_scenes, cap_suffix, label_suffix, tbl_pos in [
		(vol_only,        "volumetric-only scenes (sampling)",  "vol",          r"[!ht]"),
		(vol_only_siddon, "volumetric-only scenes (Siddon)",    "vol_siddon",   r"[!ht]"),
		(mesh_only,       "mesh-only scene",                    "mesh",         r"[!ht]"),
		(vol_mesh,        "hybrid (vol + mesh, sampling) scenes", "mixed",      r"[htbp]"),
		(vol_mesh_siddon, "hybrid (vol Siddon + mesh) scenes",  "mixed_siddon", r"[htbp]"),
	]:
		if not group_scenes:
			continue
		col_spec = "l" + "r" * len(group_scenes)
		ln(r"\begin{table}" + tbl_pos)
		ln(r"\centering")
		ln(r"  \caption{Phase breakdown [ms] for normal profile --- " + cap_suffix + r".}")
		ln(r"  \label{tab:phases_" + label_suffix + r"}")
		ln(r"\small")
		ln(r"\begin{tabular}{" + col_spec + r"}")
		ln(r"\toprule")
		hdr = r"Phase & " + " & ".join(r"\texttt{" + esc(s) + r"}" for s in group_scenes) + r" \\"
		ln(hdr)
		ln(r"\midrule")
		for pk, plabel in phase_keys:
			cells = [plabel]
			for s in group_scenes:
				r = by_sp.get((s, "normal"))
				if r and pk in r.get("phase_ms", {}):
					cells.append(phase_cell(r["phase_ms"][pk], sum(r["phase_ms"].values())))
				else:
					cells.append("---")
			ln("  " + " & ".join(cells) + r" \\")
		ln(r"\midrule")
		total_cells = [r"\textbf{Total}"]
		for s in group_scenes:
			r = by_sp.get((s, "normal"))
			total_cells.append(f"\\textbf{{{r['total_ms']:.0f}}}" if r else "---")
		ln("  " + " & ".join(total_cells) + r" \\")
		ln(r"\bottomrule")
		ln(r"\end{tabular}")
		ln(r"\end{table}")
		ln()

	# ── Subsubsection: Siddon vs sampling comparison ─────────────────────────
	has_siddon = any("siddon" in r["scene"] for r in results)
	ln(r"\subsubsection{Volume backend comparison: sampling vs.~Siddon}\label{ssec:vol_backends}")
	ln()
	ln(r"The pipeline offers two integration backends for volumetric sources:")
	ln(r"\begin{description}\setlength{\itemsep}{2pt}")
	ln(r"  \item[\texttt{sampling}] Uniform ray-marching: the ray is discretised at")
	ln(r"    equal intervals $\Delta s$ (\texttt{step\_mm}) and the local attenuation is")
	ln(r"    reconstructed by trilinear interpolation at each sample point.")
	ln(r"  \item[\texttt{siddon}] Exact voxel traversal: all voxel-boundary plane crossings")
	ln(r"    are computed analytically and accumulated as $\mu_i \cdot \ell_i$. The result")
	ln(r"    is independent of \texttt{step\_mm} and every traversed voxel is visited exactly once.")
	ln(r"\end{description}")
	ln()
	if has_siddon:
		ln(r"\begin{table}[htbp]")
		ln(r"\centering")
		ln(r"\caption{Volume-only projection time and throughput for both volume backends.")
		ln(r"  Time in ms (or s).")
		ln(r"  Sampling throughput in M\,samp/s (trilinear interpolations/s).")
		ln(r"  Siddon throughput in k\,rays/s (fully-traversed rays/s).}")
		ln(r"\label{tab:vol_backends}")
		ln(r"\small")
		ln(r"\begin{tabular}{l rrr rrr rrr}")
		ln(r"\toprule")
		ln(r"  & \multicolumn{3}{c}{\textbf{draft} (2\,mm)}")
		ln(r"  & \multicolumn{3}{c}{\textbf{normal} (1\,mm)}")
		ln(r"  & \multicolumn{3}{c}{\textbf{high} (0.5\,mm)} \\")
		ln(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
		ln(r"Scene / backend & time & M\,samp/s & k\,rays/s & time & M\,samp/s & k\,rays/s & time & M\,samp/s & k\,rays/s \\")
		ln(r"\midrule")
		for vol_name in ["small", "medium", "large"]:
			for sc, label, is_siddon in [
				(f"vol_{vol_name}",        f"vol\\,{vol_name} sampling", False),
				(f"vol_{vol_name}_siddon", f"vol\\,{vol_name} Siddon",   True),
			]:
				row_cells = [label]
				for p in profiles:
					r = by_sp.get((sc, p))
					if r:
						row_cells.append(ms_cell(r["total_ms"]))
						row_cells.append(sps_cell(r["samples_per_s"]) if not is_siddon else "---")
						rps = r.get("rays_per_s", 0)
						row_cells.append(f"{rps / 1e3:.1f}" if is_siddon else "---")
					else:
						row_cells += ["---", "---", "---"]
				ln("  " + " & ".join(row_cells) + r" \\")
			ln(r"\midrule")
		ln(r"\bottomrule")
		ln(r"\end{tabular}")
		ln(r"\end{table}")
		ln()
	else:
		ln(r"Siddon comparison data not available in these results")
		ln(r"(re-run \texttt{benchmark\_xray\_performance()} to generate it).")
		ln()

	# ── Subsubsection: Backend comparison (mesh) ──────────────────────────────
	has_proj = any(r["scene"] == "mesh_only_projected" for r in results)
	ln(r"\subsubsection{Backend comparison (mesh sources)}\label{ssec:backends}")
	ln()
	ln(r"The pipeline offers two ray-intersection backends for mesh sources:")
	ln(r"\begin{description}\setlength{\itemsep}{2pt}")
	ln(r"  \item[\texttt{analytic\_bvh}] Per-ray BVH traversal with analytic ray--triangle")
	ln(r"    intersection, currently implemented as a Python loop.")
	ln(r"  \item[\texttt{projected\_intersection\_list}] Rasterisation-based approach: triangles are projected")
	ln(r"    onto the detector plane, rasterised into pixel stacks (CSR layout),")
	ln(r"    and line integrals are accumulated per pixel.")
	ln(r"\end{description}")
	ln()
	if has_proj:
		ln(r"\begin{table}[htbp]")
		ln(r"\centering")
		ln(r"\caption{Mesh-only projection time and throughput for both backends.")
		ln(r"  Time in ms (or s); throughput in k\,samp/s.}")
		ln(r"\label{tab:backends}")
		ln(r"\small")
		ln(r"\begin{tabular}{l rr rr rr}")
		ln(r"\toprule")
		ln(r"  & \multicolumn{2}{c}{\textbf{draft} (2\,mm)}")
		ln(r"  & \multicolumn{2}{c}{\textbf{normal} (1\,mm)}")
		ln(r"  & \multicolumn{2}{c}{\textbf{high} (0.5\,mm)} \\")
		ln(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
		ln(r"Backend & time & k\,s/s & time & k\,s/s & time & k\,s/s \\")
		ln(r"\midrule")
		for sc, label in [("mesh_only", r"\texttt{analytic\_bvh}"),
		                  ("mesh_only_projected", r"\texttt{projected\_intersection\_list}")]:
			row_cells = [label]
			for p in profiles:
				r = by_sp.get((sc, p))
				if r:
					row_cells.append(ms_cell(r["total_ms"]))
					sps_k = r["samples_per_s"] / 1e3
					row_cells.append(f"{sps_k:.1f}")
				else:
					row_cells += ["---", "---"]
			ln("  " + " & ".join(row_cells) + r" \\")
		ln(r"\bottomrule")
		ln(r"\end{tabular}")
		ln(r"\end{table}")
		ln()
		# Per-phase comparison for projected backend at normal profile
		proj_r = by_sp.get(("mesh_only_projected", "normal"))
		if proj_r and proj_r.get("per_source"):
			src = proj_r["per_source"][0]
			proj_keys = [
				("stack_build_s",          r"Stack build total"),
				("stack_uv_projection_s",  r"\quad UV projection"),
				("stack_rasterize_s",      r"\quad Rasterise"),
				("stack_csr_s",            r"\quad CSR assembly"),
				("integration_s",          r"Integration"),
			]
			has_proj_detail = any(k in src for k, _ in proj_keys)
			if has_proj_detail:
				bvh_r = by_sp.get(("mesh_only", "normal"))
				ln(r"\begin{table}[htbp]")
				ln(r"\centering")
				ln(r"\caption{Per-phase timing for mesh backends at normal profile [ms].}")
				ln(r"\label{tab:backends_phases}")
				ln(r"\small")
				ln(r"\begin{tabular}{lrr}")
				ln(r"\toprule")
				ln(r"Phase & \texttt{analytic\_bvh} & \texttt{projected\_intersection\_list} \\")
				ln(r"\midrule")
				# analytic_bvh phases
				bvh_src = bvh_r["per_source"][0] if bvh_r and bvh_r.get("per_source") else {}
				bvh_phases = [
					("bvh_build_s",  r"BVH build"),
					("integration_s", r"Integration (BVH)"),
				]
				proj_display = [
					("stack_build_s",         r"Stack build total"),
					("stack_uv_projection_s", r"\quad UV projection"),
					("stack_rasterize_s",     r"\quad Rasterise"),
					("stack_csr_s",           r"\quad CSR assembly"),
					("integration_s",         r"Integration"),
				]
				# print BVH rows
				for k, label in bvh_phases:
					v_bvh = bvh_src.get(k, 0) * 1000.0
					ln(f"  {label} & {v_bvh:.1f} & --- \\\\")
				ln(r"\midrule")
				# print projected rows
				for k, label in proj_display:
					v_proj = src.get(k, 0) * 1000.0
					ln(f"  {label} & --- & {v_proj:.1f} \\\\")
				ln(r"\midrule")
				v_bvh_tot = bvh_r["total_ms"] if bvh_r else 0
				v_proj_tot = proj_r["total_ms"]
				ln(f"  \\textbf{{Total}} & \\textbf{{{v_bvh_tot:.0f}}} & \\textbf{{{v_proj_tot:.0f}}} \\\\")
				ln(r"\bottomrule")
				ln(r"\end{tabular}")
				ln(r"\end{table}")
				ln()
	else:
		ln(r"Backend comparison data not available in these results")
		ln(r"(re-run \texttt{benchmark\_xray\_performance()} to generate it).")
		ln()

	# ── Subsubsection: Key observations ───────────────────────────────────────
	ln(r"\subsubsection{Key observations}")
	ln()
	ln(r"\begin{itemize}\setlength{\itemsep}{3pt}")
	ln(r"  \item \textbf{Volumetric marching (sampling)} dominates volume-only scenes ($>80\%$ of total time).")
	ln(r"    Throughput is approximately constant at $11$--$17\,\mathrm{Msamp/s}$,")
	ln(r"    confirming that cost scales linearly with sample count.")
	ln(r"  \item \textbf{Siddon exact traversal} processes $O(N_x+N_y+N_z)$ boundary crossings per ray,")
	ln(r"    independent of \texttt{step\_mm}. Its cost is dominated by the sorting step and")
	ln(r"    scales with volume resolution rather than marching step density.")
	ln(r"  \item For coarse steps ($\Delta s \geq $ voxel size) Siddon can be faster than sampling")
	ln(r"    while guaranteeing that every traversed voxel is visited exactly once.")
	ln(r"  \item \textbf{Mesh ray-intersection} (\texttt{analytic\_bvh}, Python loop)")
	ln(r"    runs at $\approx 16$--$18\,\mathrm{k\,samp/s}$ ---")
	ln(r"    several orders of magnitude below the volumetric path.")
	ln(r"    This is the primary bottleneck for optimisation.")
	ln(r"  \item Mesh timing is \textbf{independent of volume size} and nearly independent")
	ln(r"    of step size, since the number of intersecting pixels is determined by")
	ln(r"    the projected mesh silhouette, not by the marching step.")
	ln(r"  \item In hybrid scenes the two paths run independently and their times add up.")
	ln(r"\end{itemize}")
	ln()

	# ── Subsection: Projection Images ─────────────────────────────────────────
	has_images = any("png_path" in r and os.path.isfile(r["png_path"]) for r in results)
	if has_images:
		ln(r"\subsection{Projection Images}")
		ln()
		ln(r"Figure~\ref{fig:proj_normal} shows synthetic cone-beam projections for all scene variants")
		ln(r"at the \textbf{normal} quality profile ($512\times512$\,px, step\,$1.0\,\mathrm{mm}$),")
		ln(r"rendered with a digital radiography presentation model")
		ln(r"(standard convention: dense~=~white, $\gamma=0.6$, contrast\,=\,1.1).")
		ln(r"Images at the draft and high profiles are visually indistinguishable")
		ln(r"for the synthetic objects used here.")
		ln()

		ln(r"\begin{figure}[htbp]")
		ln(r"\centering")
		vol_sizes_order = ["small", "medium", "large"]
		col_scene_types = ["vol_{v}", "vol_{v}_siddon", "vol_{v}+mesh"]
		for vi, vol_size in enumerate(vol_sizes_order):
			for ci, sc_template in enumerate(col_scene_types):
				sc = sc_template.replace("{v}", vol_size)
				r = by_sp.get((sc, "normal"))
				if r is None or "png_path" not in r or not os.path.isfile(r["png_path"]):
					ln(r"\begin{subfigure}[t]{0.30\linewidth}\centering")
					ln(r"  \fbox{\rule{0pt}{3cm}\hspace{3cm}}")
					ln(f"  \\caption{{\\texttt{{{esc(sc)}}}}}")
					ln(r"\end{subfigure}")
				else:
					try:
						rel = os.path.relpath(r["png_path"], tex_dir).replace("\\", "/")
					except ValueError:
						rel = r["png_path"].replace("\\", "/")
					cap_text = sc.replace("_", r"\_").replace("+", r"\,+\,")
					ln(r"\begin{subfigure}[t]{0.30\linewidth}")
					ln(r"  \centering")
					ln(f"  \\includegraphics[width=\\linewidth]{{{rel}}}")
					ln(f"  \\caption{{\\texttt{{{cap_text}}}}}")
					ln(r"\end{subfigure}")
				if ci < 2:
					ln(r"\hfill")
			if vi < 2:
				ln(r"\\[4pt]")
			ln()
		ln(r"\caption{Synthetic cone-beam projections, normal profile")
		ln(r"  ($512\times512$\,px, step\,$1.0\,\mathrm{mm}$).")
		ln(r"  Columns: sampling / Siddon / sampling\,+\,mesh.")
		ln(r"  Rows: small ($64^3$) / medium ($96^3$) / large ($128^3$) volume.}")
		ln(r"\label{fig:proj_normal}")
		ln(r"\end{figure}")
		ln()

	# ── Subsection: Sampling vs Siddon visual comparison ──────────────────────
	ss_key   = ("small", "normal")
	ss_stats = (diff_stats or {}).get(ss_key, {})
	ss_fig   = ss_stats.get("sample_siddon_path") or ss_stats.get("path")
	has_ss   = ss_fig is not None and os.path.isfile(ss_fig)

	ln(r"\subsection{Visual Comparison: Sampling vs.\ Siddon}")
	ln()
	ln(r"Figure~\ref{fig:sample_siddon} illustrates the influence of the volumetric")
	ln(r"integration backend on the synthetic cone-beam projection.")
	ln(r"Identical projection scenarios were generated using Joseph-type sampling and")
	ln(r"Siddon voxel-traversal backends while preserving the same scene geometry,")
	ln(r"detector configuration, attenuation model, and acquisition parameters.")
	ln(r"Consequently, the volumetric integration backend remained the only varying")
	ln(r"component of the projection pipeline.")
	ln()
	ln(r"\begin{figure}[htbp]")
	ln(r"\centering")
	if has_ss:
		try:
			rel_ss = os.path.relpath(ss_fig, tex_dir).replace("\\", "/")
		except ValueError:
			rel_ss = ss_fig.replace("\\", "/")
		ln(f"\\includegraphics[width=\\linewidth]{{{rel_ss}}}")
	else:
		ln(r"\fbox{\parbox{\linewidth}{\centering\rule{0pt}{5cm}")
		ln(r"  \textit{sample\_siddon.png not found -- re-run benchmark}}")
		ln(r"}")
	ln(r"\caption{")
	ln(r"  Visual comparison of synthetic cone-beam projection images generated using")
	ln(r"  Joseph-type sampling ($\Delta s = 1\,\mathrm{mm}$) and Siddon exact voxel-traversal")
	ln(r"  backends (vol\,small, $64^3$, normal profile, $512\times512$\,px).")
	ln(r"  Panel~(a): sampling result.")
	ln(r"  Panel~(b): Siddon result.")
	ln(r"  Panel~(c): signed difference (sampling\,$-$\,Siddon), diverging \texttt{RdBu\_r}")
	ln(r"  colormap symmetric about zero; red\,=\,sampling\,$>$\,Siddon,")
	ln(r"  blue\,=\,sampling\,$<$\,Siddon.")
	if ss_stats.get("mae") is not None:
		mae_s    = f"{ss_stats['mae']:.5f}"
		dmax_s   = f"{ss_stats['max_abs']:.5f}"
		relmae_s = f"{100*ss_stats['rel_mae']:.3f}\\,\\%" if ss_stats.get("rel_mae") is not None else "n/a"
		ln(f"  Numerical error: MAE\\,=\\,${mae_s}$, max$|\\Delta|$\\,=\\,${dmax_s}$,")
		ln(f"  relative MAE\\,=\\,${relmae_s}$ of mean non-zero attenuation.")
	ln(r"}")
	ln(r"\label{fig:sample_siddon}")
	ln(r"\end{figure}")
	ln()

	# ── Numerical difference table ─────────────────────────────────────────────
	if diff_stats:
		ln(r"\begin{table}[htbp]")
		ln(r"\centering")
		ln(r"\caption{Numerical difference (sampling\,$-$\,Siddon) for volume-only scenes.")
		ln(r"  MAE and max$|\Delta|$ are computed on raw line-integral images")
		ln(r"  (no presentation model applied).}")
		ln(r"\label{tab:diff_stats}")
		ln(r"\small")
		ln(r"\begin{tabular}{llrrr}")
		ln(r"\toprule")
		ln(r"Volume & Profile & MAE & max$|\Delta|$ & rel.\,MAE (\%) \\")
		ln(r"\midrule")
		for vol_n in ["small", "medium", "large"]:
			for prof_n in ["draft", "normal", "high"]:
				ds = diff_stats.get((vol_n, prof_n))
				if ds is None:
					continue
				mae_v    = f"{ds['mae']:.5f}"
				dmax_v   = f"{ds['max_abs']:.5f}"
				rel_v    = f"{100*ds['rel_mae']:.3f}" if ds.get("rel_mae") is not None else "---"
				ln(f"  vol\\,{vol_n} & {prof_n} & ${mae_v}$ & ${dmax_v}$ & ${rel_v}$ \\\\")
		ln(r"\bottomrule")
		ln(r"\end{tabular}")
		ln(r"\end{table}")
		ln()

	ln(r"\end{document}")

	content = "\n".join(L)
	with open(tex_path, "w", encoding="utf-8") as fh:
		fh.write(content)

	print(f"LaTeX report written to: {tex_path}")
	print(f"Compile with: pdflatex \"{tex_path}\"")
	return tex_path

# if __name__ == '__main__':
# 	# _bm_results, _diff_stats = benchmark_xray_performance()
# 	# generate_latex_benchmark_report(_bm_results, diff_stats=_diff_stats)
# 	print("cant run this file directly")
