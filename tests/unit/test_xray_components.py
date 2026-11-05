"""Compatibility tests for tube and collimator configuration snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from plugins.virtRTG.virtualXRay import VirtualXRay
from plugins.virtRTG.xray.xrayComponents import FocalSpotSamples, XRayCollimator, XRayTube
from plugins.virtRTG.xray.xrayProjection import RectangularBeamGeometry


def _beam():
	return RectangularBeamGeometry(
		(0, 0, 1), (1, 0, 0), (0, 1, 0), 20.0, 10.0, 1000.0,
	)


def test_focal_spot_samples_validate_contract_and_point_source():
	samples = FocalSpotSamples.point((1.0, 2.0, 3.0), "virtual_xray_ref_mm")
	assert samples.positions_mm == ((1.0, 2.0, 3.0),)
	assert samples.weights == (1.0,)
	assert samples.coordinate_frame == "virtual_xray_ref_mm"
	with pytest.raises(ValueError):
		FocalSpotSamples(((0, 0, 0), (1, 0, 0)), (0.2, 0.2), "ref")
	with pytest.raises(ValueError):
		FocalSpotSamples(((0, 0, np.inf),), (1.0,), "ref")
	with pytest.raises(ValueError):
		FocalSpotSamples(((0, 0, 0),), (1.0,), None)


def test_collimator_is_an_angular_model_from_a_shared_focal_point():
	collimator = XRayCollimator(_beam(), enabled=True)
	directions = np.array([[0, 0, 1], [0.2, 0.05, 1]], dtype=np.float64)
	accepted = collimator.accept_directions(directions)
	assert accepted.shape == (2,)
	assert accepted.tolist() == [True, False]
	assert collimator.transmission_for_directions(directions[:1]).tolist() == [1.0]
	assert XRayCollimator(_beam(), enabled=False).accept_directions(directions).all()
	assert not hasattr(collimator, "transmission_for_rays")


def test_virtual_xray_builds_fresh_component_snapshots_and_keeps_geometry_equivalent():
	virtual_xray = VirtualXRay()
	virtual_xray.directional_beam_geometry = _beam()
	virtual_xray.physical_collimation_enabled = True
	config = virtual_xray.build_projection_config()
	assert config.tube_model.focal_spot_samples.positions_mm == (
		tuple(float(value) for value in virtual_xray.source_position_ref),
	)
	assert config.tube_model.focal_spot_samples.coordinate_frame == "virtual_xray_ref_mm"
	assert config.tube_model.tube_to_reference_rotation is None
	assert config.collimator_model.beam_geometry == config.geometry.beam_geometry
	assert config.collimator_model.enabled == config.geometry.collimation_enabled
	virtual_xray.source_position_ref = np.array([2.0, 3.0, -200.0], dtype=np.float32)
	assert virtual_xray.build_tube_model().focal_spot_samples.positions_mm == ((2.0, 3.0, -200.0),)
	virtual_xray.projection_mode = "parallel"
	assert virtual_xray.build_projection_config().tube_model is None


def test_geometry_adapter_uses_the_supplied_collimator_snapshot(monkeypatch):
	virtual_xray = VirtualXRay()
	supplied = XRayCollimator(_beam(), enabled=True)
	virtual_xray.directional_beam_geometry = RectangularBeamGeometry(
		(0, 0, 1), (1, 0, 0), (0, 1, 0), 2.0, 3.0, 400.0,
	)
	virtual_xray.physical_collimation_enabled = False
	monkeypatch.setattr(
		virtual_xray,
		"build_collimator_model",
		lambda: pytest.fail("_build_geometry rebuilt the supplied snapshot"),
	)
	geometry = virtual_xray._build_geometry(supplied)
	assert geometry.beam_geometry is supplied.beam_geometry
	assert geometry.collimation_enabled is True


def test_collimator_roll_does_not_become_a_tube_pose():
	virtual_xray = VirtualXRay()
	virtual_xray.reset_collimation_reference(_beam())
	tube_before = virtual_xray.build_tube_model()
	beam_before = virtual_xray.directional_beam_geometry
	virtual_xray.set_collimation_angles(0.0, 0.0, 90.0)
	tube_after = virtual_xray.build_tube_model()
	assert tube_before.tube_to_reference_rotation is None
	assert tube_after.tube_to_reference_rotation is None
	assert virtual_xray.directional_beam_geometry != beam_before
	assert np.allclose(virtual_xray.directional_beam_geometry.beam_axis_ref, beam_before.beam_axis_ref)
	assert not np.allclose(virtual_xray.directional_beam_geometry.beam_horizontal_ref, beam_before.beam_horizontal_ref)


def test_tube_rotation_snapshot_requires_a_proper_rotation():
	samples = FocalSpotSamples.point((0.0, 0.0, 0.0), "virtual_xray_ref_mm")
	with pytest.raises(ValueError):
		XRayTube(samples, tube_to_reference_rotation=((1, 0, 0), (0, 1, 0), (0, 0, 2)))
