"""Focused geometry tests for the standalone VirtualXRay collimation controls."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
from plugins.virtRTG.gui.propVirtualXRay import (
	_build_beam_aligned_to_detector,
)
from plugins.virtRTG.sceneFormat import load_virtual_xray_scene, save_virtual_xray_scene
from plugins.virtRTG.virtualXRay import (
	VirtualXRay,
	_beam_with_dimensions,
	_beam_with_offsets,
)
from plugins.virtRTG.xray.xrayProjection import (
	RectangularBeamGeometry,
	XRayPhysicsModel,
	XRayProjectionConfig,
	XRayProjectionGeometry,
	XRayScene,
)
from PyQt5.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_qapp():
	"""Return the retained offscreen Qt application for real property panel tests."""
	return QApplication.instance() or QApplication([])


class _MaskSampleSource:
	"""Minimal source used to exercise exposure masks without scene contributions."""

	def bounds_world(self):
		return np.array([1000.0, 1000.0, 1000.0]), np.array([1001.0, 1001.0, 1001.0])

	def sample_attenuation_world(self, points_world, physics_model):
		return np.zeros(points_world.shape[0], dtype=np.float32)

	def uses_direct_integral(self):
		return False

	def resolve_physics_model(self, physics_model):
		return physics_model


def test_standalone_beam_alignment_uses_detector_center_and_defaults():
	"""Initialize a useful beam without requiring device-link geometry."""
	virtual_xray = VirtualXRay()
	beam = _build_beam_aligned_to_detector(virtual_xray)

	assert np.allclose(beam.beam_axis_ref, [0.0, 0.0, 1.0])
	assert np.allclose(beam.beam_horizontal_ref, [1.0, 0.0, 0.0])
	assert np.allclose(beam.beam_vertical_ref, [0.0, 1.0, 0.0])
	assert beam.field_width_cm == 15.0
	assert beam.field_height_cm == 10.0
	assert beam.field_reference_distance_mm == 1000.0
	assert virtual_xray.directional_beam_geometry is None
	assert virtual_xray.physical_collimation_enabled is False


def test_alignment_tracks_rotated_off_axis_detector_and_preserves_up_orientation():
	"""Aim at the actual detector center and project its up axis into the beam frame."""
	virtual_xray = VirtualXRay()
	virtual_xray.source_position_ref = np.array([10.0, -5.0, -200.0], dtype=np.float32)
	virtual_xray.detector_center_ref = np.array([40.0, 20.0, 100.0], dtype=np.float32)
	virtual_xray.detector_normal_ref = np.array([0.3, 0.2, -0.93], dtype=np.float32)
	virtual_xray.detector_up_ref = np.array([0.1, 0.98, 0.15], dtype=np.float32)
	geometry = virtual_xray.build_geometry()
	beam = _build_beam_aligned_to_detector(virtual_xray)
	expected_axis = geometry.detector_center_ref_point() - virtual_xray.source_position_ref
	expected_axis /= np.linalg.norm(expected_axis)
	expected_vertical = geometry.detector_v_ref - expected_axis * np.dot(geometry.detector_v_ref, expected_axis)
	expected_vertical /= np.linalg.norm(expected_vertical)

	assert np.allclose(beam.beam_axis_ref, expected_axis, atol=1e-6)
	assert np.allclose(beam.beam_vertical_ref, expected_vertical, atol=1e-6)
	assert np.allclose(np.cross(beam.beam_horizontal_ref, beam.beam_vertical_ref), beam.beam_axis_ref, atol=1e-6)


def test_alignment_rejects_degenerate_source_detector_geometry():
	"""Report coincident source and detector center instead of creating an invalid beam."""
	virtual_xray = VirtualXRay()
	virtual_xray.detector_center_ref = virtual_xray.source_position_ref.copy()

	with pytest.raises(ValueError, match="Source-to-detector direction"):
		_build_beam_aligned_to_detector(virtual_xray)


def test_dimension_changes_preserve_orientation_and_offsets_are_stable():
	"""Keep axes fixed during field edits and compute angles from the original frame."""
	base = _build_beam_aligned_to_detector(VirtualXRay())
	changed = _beam_with_dimensions(base, 22.0, 17.0, 850.0)
	assert changed.beam_axis_ref == base.beam_axis_ref
	assert changed.beam_horizontal_ref == base.beam_horizontal_ref
	assert changed.beam_vertical_ref == base.beam_vertical_ref
	assert (changed.field_width_cm, changed.field_height_cm, changed.field_reference_distance_mm) == (22.0, 17.0, 850.0)

	first = _beam_with_offsets(base, 8.0, -5.0)
	second = _beam_with_offsets(base, 8.0, -5.0)
	assert first == second
	assert np.dot(first.beam_axis_ref, base.beam_axis_ref) < 1.0
	assert np.dot(first.beam_axis_ref, base.beam_horizontal_ref) > 0.0
	assert np.dot(first.beam_axis_ref, base.beam_vertical_ref) < 0.0

	virtual_xray = VirtualXRay()
	virtual_xray.reset_collimation_reference(base)
	virtual_xray.set_collimation_angles(8.0, -5.0, 12.0)
	effective_before_dimensions = virtual_xray.directional_beam_geometry
	virtual_xray.set_collimation_dimensions(22.0, 17.0, 850.0)
	assert virtual_xray.collimation_angles_degrees == (8.0, -5.0, 12.0)
	assert virtual_xray.collimation_reference_geometry.beam_axis_ref == base.beam_axis_ref
	assert virtual_xray.directional_beam_geometry.field_width_cm == 22.0
	assert virtual_xray.directional_beam_geometry.field_height_cm == 17.0
	assert virtual_xray.directional_beam_geometry.field_reference_distance_mm == 850.0
	assert virtual_xray.directional_beam_geometry != effective_before_dimensions


def test_pure_roll_rotates_only_the_field_axes_and_preserves_scene_geometry():
	"""Roll changes the rectangle orientation around its fixed central beam axis."""
	virtual_xray = VirtualXRay()
	source_before = virtual_xray.source_position_ref.copy()
	detector_before = virtual_xray.detector_center_ref.copy()
	base = _build_beam_aligned_to_detector(virtual_xray, 20.0, 60.0, 1000.0)
	rolled = _beam_with_offsets(base, 0.0, 0.0, 90.0)

	assert np.allclose(rolled.beam_axis_ref, base.beam_axis_ref, atol=1e-12)
	assert np.allclose(rolled.beam_horizontal_ref, base.beam_vertical_ref, atol=1e-12)
	assert np.allclose(rolled.beam_vertical_ref, -np.asarray(base.beam_horizontal_ref), atol=1e-12)
	assert (rolled.field_width_cm, rolled.field_height_cm, rolled.field_reference_distance_mm) == (20.0, 60.0, 1000.0)
	assert np.array_equal(virtual_xray.source_position_ref, source_before)
	assert np.array_equal(virtual_xray.detector_center_ref, detector_before)


def test_three_angle_edits_return_exactly_to_the_reference_without_drift():
	"""Compute each edit from the same reference and recover it at zero angles."""
	base = _build_beam_aligned_to_detector(VirtualXRay(), 20.0, 60.0, 1000.0)
	edits = [
		_beam_with_offsets(base, 10.0, 0.0, 0.0),
		_beam_with_offsets(base, 20.0, 0.0, 0.0),
		_beam_with_offsets(base, 20.0, -10.0, 0.0),
		_beam_with_offsets(base, 20.0, -10.0, 35.0),
	]
	assert all(np.isfinite(beam.beam_axis_ref).all() for beam in edits)
	reset = _beam_with_offsets(base, 0.0, 0.0, 0.0)
	for actual, expected in zip(
		(reset.beam_axis_ref, reset.beam_horizontal_ref, reset.beam_vertical_ref),
		(base.beam_axis_ref, base.beam_horizontal_ref, base.beam_vertical_ref),
	):
		assert np.allclose(actual, expected, atol=1e-12)


def test_gui_angle_sequence_and_enable_toggle_keep_one_reference_frame():
	"""Edit through actual controls and preserve model-owned angles through refreshes."""
	from plugins.virtRTG.gui.propVirtualXRay import PropVirtualXRay

	app = _get_qapp()
	virtual_xray = VirtualXRay()
	base = _build_beam_aligned_to_detector(virtual_xray)
	virtual_xray.reset_collimation_reference(base)
	panel = PropVirtualXRay(virtual_xray)
	panel._after_change = lambda obj: panel.updateProperties()
	panel.updateProperties()

	for angles in ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, -10.0, 0.0), (20.0, -10.0, 35.0)):
		for spin, value in zip(
			(panel.beamHorizontalOffsetSpin, panel.beamVerticalOffsetSpin, panel.beamRollSpin), angles,
		):
			spin.setValue(value)
		app.processEvents()

	geometry_before_toggle = virtual_xray.directional_beam_geometry
	panel.collimationCheck.setChecked(True)
	panel.collimationCheck.setChecked(False)
	assert virtual_xray.directional_beam_geometry is geometry_before_toggle
	assert virtual_xray.collimation_angles_degrees == (20.0, -10.0, 35.0)
	assert panel.beamRollSpin.value() == 35.0

	panel.beamFieldWidthSpin.setValue(22.0)
	panel.beamFieldHeightSpin.setValue(12.0)
	panel.beamReferenceDistanceSpin.setValue(900.0)
	assert virtual_xray.collimation_angles_degrees == (20.0, -10.0, 35.0)
	assert virtual_xray.directional_beam_geometry.field_width_cm == 22.0
	assert virtual_xray.directional_beam_geometry.field_height_cm == 12.0
	assert virtual_xray.directional_beam_geometry.field_reference_distance_mm == 900.0

	panel.beamHorizontalOffsetSpin.setValue(0.0)
	panel.beamVerticalOffsetSpin.setValue(0.0)
	panel.beamRollSpin.setValue(0.0)
	for actual, expected in zip(
		(
			virtual_xray.directional_beam_geometry.beam_axis_ref,
			virtual_xray.directional_beam_geometry.beam_horizontal_ref,
			virtual_xray.directional_beam_geometry.beam_vertical_ref,
		),
		(base.beam_axis_ref, base.beam_horizontal_ref, base.beam_vertical_ref),
	):
		assert np.allclose(actual, expected, atol=1e-12)
	assert virtual_xray.collimation_angles_degrees == (0.0, 0.0, 0.0)
	panel.on_align_beam_to_detector()
	assert virtual_xray.collimation_angles_degrees == (0.0, 0.0, 0.0)
	assert virtual_xray.directional_beam_geometry.field_width_cm == 22.0
	assert virtual_xray.directional_beam_geometry.field_height_cm == 12.0
	assert virtual_xray.directional_beam_geometry.field_reference_distance_mm == 900.0

	panel.deleteLater()
	app.processEvents()


def test_projection_panel_recreation_and_object_selection_retain_angle_state(monkeypatch):
	"""Projection and real property panel lifecycle preserve the model edit state."""
	from plugins.virtRTG.gui.propVirtualXRay import PropVirtualXRay

	app = _get_qapp()
	virtual_xray = VirtualXRay()
	virtual_xray.detector_shape_hw = [3, 3]
	beam = _build_beam_aligned_to_detector(virtual_xray, 15.0, 10.0, 1000.0)
	virtual_xray.reset_collimation_reference(beam)
	virtual_xray.set_collimation_angles(18.0, -7.0, 32.0)
	virtual_xray.physical_collimation_enabled = True
	panel_before = PropVirtualXRay(virtual_xray)
	panel_before.updateProperties()
	monkeypatch.setattr(virtual_xray, "build_scene", lambda: XRayScene.from_sample_sources([_MaskSampleSource()]))
	virtual_xray.project_and_cache()
	assert virtual_xray.collimation_angles_degrees == (18.0, -7.0, 32.0)
	panel_before.updateProperties()
	assert (
		panel_before.beamHorizontalOffsetSpin.value(),
		panel_before.beamVerticalOffsetSpin.value(),
		panel_before.beamRollSpin.value(),
	) == (18.0, -7.0, 32.0)
	panel_before.deleteLater()

	other_xray = VirtualXRay()
	other_panel = PropVirtualXRay(other_xray)
	other_panel.updateProperties()
	other_panel.deleteLater()
	panel_after = PropVirtualXRay(virtual_xray)
	panel_after.updateProperties()
	assert (
		panel_after.beamHorizontalOffsetSpin.value(),
		panel_after.beamVerticalOffsetSpin.value(),
		panel_after.beamRollSpin.value(),
	) == (18.0, -7.0, 32.0)
	assert virtual_xray.collimation_angles_degrees == (18.0, -7.0, 32.0)
	panel_after.deleteLater()
	app.processEvents()


def test_roll_rotates_non_square_exposure_footprint_without_steering_axis():
	"""A 90 degree roll swaps the narrow and wide sides of the rectangular field."""
	base = RectangularBeamGeometry((0, 0, 1), (1, 0, 0), (0, 1, 0), 2.0, 6.0, 100.0)
	rolled = _beam_with_offsets(base, 0.0, 0.0, 90.0)
	geometry = XRayProjectionGeometry.from_detector_pose(
		detector_center_ref=[0.0, 0.0, 100.0], detector_normal_ref=[0.0, 0.0, -1.0],
		detector_up_ref=[0.0, 1.0, 0.0], detector_shape_hw=[5, 5],
		detector_pixel_size_mm=10.0, source_position_ref=[0.0, 0.0, 0.0],
		beam_geometry=base, collimation_enabled=True,
	)
	scene = XRayScene.from_sample_sources([_MaskSampleSource()])
	model = XRayPhysicsModel(output_mode="integral")
	initial = scene.project_capture(XRayProjectionConfig(geometry, model))
	rolled_capture = scene.project_capture(XRayProjectionConfig(replace(geometry, beam_geometry=rolled), model))

	expected_initial = np.zeros((5, 5), dtype=bool)
	expected_initial[:, 1:4] = True
	expected_rolled = np.zeros((5, 5), dtype=bool)
	expected_rolled[1:4, :] = True
	assert np.array_equal(initial.exposure_mask, expected_initial)
	assert np.array_equal(rolled_capture.exposure_mask, expected_rolled)
	assert np.allclose(rolled.beam_axis_ref, base.beam_axis_ref)
	assert np.array_equal(geometry.source_position_ref, np.array([0.0, 0.0, 0.0]))
	assert np.array_equal(geometry.detector_center_ref_point(), np.array([0.0, 0.0, 100.0], dtype=np.float32))


def test_offsets_and_scene_round_trip_leave_source_detector_geometry_unchanged(tmp_path):
	"""Persist the effective adjusted frame without persisting misleading absolute angles."""
	virtual_xray = VirtualXRay()
	beam = _build_beam_aligned_to_detector(virtual_xray)
	virtual_xray.reset_collimation_reference(beam)
	virtual_xray.set_collimation_angles(4.0, 3.0, 35.0)
	virtual_xray.physical_collimation_enabled = True
	source_before = virtual_xray.source_position_ref.copy()
	detector_before = virtual_xray.detector_center_ref.copy()
	path = tmp_path / "standalone-collimation.vxrscene.xml"
	save_virtual_xray_scene(virtual_xray, path)
	loaded = load_virtual_xray_scene(path)

	assert np.array_equal(virtual_xray.source_position_ref, source_before)
	assert np.array_equal(virtual_xray.detector_center_ref, detector_before)
	assert np.array_equal(loaded.directional_beam_geometry.beam_axis_ref, virtual_xray.directional_beam_geometry.beam_axis_ref)
	assert np.array_equal(loaded.directional_beam_geometry.beam_horizontal_ref, virtual_xray.directional_beam_geometry.beam_horizontal_ref)
	assert np.array_equal(loaded.directional_beam_geometry.beam_vertical_ref, virtual_xray.directional_beam_geometry.beam_vertical_ref)
	assert loaded.directional_beam_geometry.field_width_cm == 15.0
	assert loaded.directional_beam_geometry.field_height_cm == 10.0
	assert loaded.directional_beam_geometry.field_reference_distance_mm == 1000.0
	assert loaded.physical_collimation_enabled is True
	assert np.allclose(loaded.directional_beam_geometry.beam_axis_ref, virtual_xray.directional_beam_geometry.beam_axis_ref)
	assert np.allclose(loaded.directional_beam_geometry.beam_horizontal_ref, virtual_xray.directional_beam_geometry.beam_horizontal_ref)
	assert np.allclose(loaded.directional_beam_geometry.beam_vertical_ref, virtual_xray.directional_beam_geometry.beam_vertical_ref)
	assert loaded.collimation_reference_geometry == loaded.directional_beam_geometry
	assert loaded.collimation_angles_degrees == (0.0, 0.0, 0.0)
	from plugins.virtRTG.gui.propVirtualXRay import PropVirtualXRay
	app = _get_qapp()
	loaded_panel = PropVirtualXRay(loaded)
	loaded_panel.updateProperties()
	assert loaded_panel.beamHorizontalOffsetSpin.value() == 0.0
	assert loaded_panel.beamVerticalOffsetSpin.value() == 0.0
	assert loaded_panel.beamRollSpin.value() == 0.0
	loaded_panel.deleteLater()
	app.processEvents()


def test_dimension_changes_retain_reference_orientation_for_all_angles():
	"""Updating field dimensions changes shape only, including under active angles."""
	reference = _build_beam_aligned_to_detector(VirtualXRay(), 20.0, 60.0, 1000.0)
	angled = _beam_with_offsets(reference, 20.0, -10.0, 35.0)
	dimensioned_reference = _beam_with_dimensions(reference, 25.0, 45.0, 800.0)
	updated = _beam_with_offsets(dimensioned_reference, 20.0, -10.0, 35.0)
	assert np.allclose(dimensioned_reference.beam_axis_ref, reference.beam_axis_ref)
	assert np.allclose(dimensioned_reference.beam_horizontal_ref, reference.beam_horizontal_ref)
	assert np.allclose(dimensioned_reference.beam_vertical_ref, reference.beam_vertical_ref)
	assert np.allclose(updated.beam_axis_ref, angled.beam_axis_ref)
	assert np.allclose(updated.beam_horizontal_ref, angled.beam_horizontal_ref)
	assert np.allclose(updated.beam_vertical_ref, angled.beam_vertical_ref)
	assert (updated.field_width_cm, updated.field_height_cm, updated.field_reference_distance_mm) == (25.0, 45.0, 800.0)
