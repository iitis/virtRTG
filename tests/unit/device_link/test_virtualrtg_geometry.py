"""Numerical checks for the independent VirtualXRay geometry mapping."""
import math
import unittest

from plugins.virtRTG.device_link.device_adapter import DeviceState, Pose
from plugins.virtRTG.device_link.virtualrtg_geometry import IDENTITY_4, map_device_geometry, plugin_geometry_arguments


def set_pose(state, name, position_mm, rotation_deg=(0, 0, 0)):
    state.poses[name] = Pose(tuple(v / 1000 for v in position_mm),
                             tuple(math.radians(v) for v in rotation_deg))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


class VirtualXRayGeometryTests(unittest.TestCase):
    def state_for(self, preset):
        state = DeviceState()
        if preset == "lateral":
            set_pose(state, "tube", (1100, 1140, 0), (0, 0, -90))
            set_pose(state, "detector", (-820, 1140, 0), (0, 0, -90))
        elif preset == "upright":
            set_pose(state, "phantom", (0, 1050, 0), (90, 0, 0))
            set_pose(state, "tube", (0, 1120, 1150), (90, 0, 0))
            set_pose(state, "detector", (0, 1120, -680), (90, 0, 0))
        return state

    def test_ap_lateral_upright_mapping_and_ray_direction(self):
        for name, axis in (("ap", 1), ("lateral", 0), ("upright", 2)):
            with self.subTest(preset=name):
                geometry = map_device_geometry(self.state_for(name))
                ray = tuple(geometry["detector_center_ref"][i] - geometry["source_position_ref"][i] for i in range(3))
                self.assertLess(ray[axis], 0)
                self.assertAlmostEqual(geometry["source_to_detector_distance_mm"], math.sqrt(dot(ray, ray)), places=8)
                self.assertEqual(geometry["units"], "mm")
                self.assertEqual(geometry["coordinate_frame"], "R (VirtualXRay local reference)")
                self.assertTrue(geometry["geometry_consistent"])

    def test_independent_translation_rotation_and_xyz_composition(self):
        state = DeviceState()
        set_pose(state, "tube", (250, 1800, -400), (20, 35, -15))
        set_pose(state, "detector", (-300, 900, 600), (0, 0, 0))
        g = map_device_geometry(state)
        a, b, c = map(math.radians, (20, 35, -15))
        ox, oy, oz = -math.sin(c) * -430, math.cos(c) * -430, 0.0
        ox, oz = math.cos(b)*ox + math.sin(b)*oz, -math.sin(b)*ox + math.cos(b)*oz
        oy, oz = math.cos(a)*oy - math.sin(a)*oz, math.sin(a)*oy + math.cos(a)*oz
        expected_source = (250 + ox, 1800 + oy, -400 + oz)
        for actual, expected in zip(g["source_position_ref"], expected_source):
            self.assertAlmostEqual(actual, expected, places=8)
        set_pose(state, "tube", (251, 1800, -400), (20, 35, -15))
        moved = map_device_geometry(state)
        self.assertAlmostEqual(moved["source_position_ref"][0] - g["source_position_ref"][0], 1.0, places=8)
        expected_center = (-300.0, 940.0, 600.0)
        for actual, expected in zip(g["detector_center_ref"], expected_center):
            self.assertAlmostEqual(actual, expected, places=8)
        set_pose(state, "detector", (-300, 900, 600), (20, 35, -15))
        rotated = map_device_geometry(state)
        self.assertNotEqual(rotated["detector_center_ref"], expected_center)
        self.assertNotEqual(rotated["detector_up_ref"], g["detector_up_ref"])
        # A detector whose plugin normal points away from the source ray is reported.
        set_pose(state, "detector", (-300, 900, 600), (180, 0, 0))
        self.assertFalse(map_device_geometry(state)["geometry_consistent"])

    def test_detector_axes_are_orthonormal_and_pixel_handedness_is_explicit(self):
        for name in ("ap", "lateral", "upright"):
            g = map_device_geometry(self.state_for(name))
            u, v, n = (g["detector_horizontal_ref"], g["detector_up_ref"], g["detector_normal_ref"])
            for axis in (u, v, n):
                self.assertAlmostEqual(dot(axis, axis), 1.0, places=7)
            self.assertAlmostEqual(dot(u, v), 0.0, places=7)
            self.assertAlmostEqual(dot(u, n), 0.0, places=7)
            self.assertAlmostEqual(dot(v, n), 0.0, places=7)
            self.assertTrue(all(abs(a-b) < 1e-7 for a, b in zip(cross(v, n), u)))
            self.assertTrue(g["orientation_consistent"])
            self.assertTrue(all(abs(a-b) < 1e-7 for a, b in zip(u, g["browser_detector_horizontal_ref"])))
            self.assertAlmostEqual(dot(cross(g["browser_detector_horizontal_ref"], v),
                                       g["browser_detector_normal_ref"]), -1.0, places=7)
            ray = tuple(g["detector_center_ref"][i]-g["source_position_ref"][i] for i in range(3))
            self.assertGreater(dot(n, ray), 0.0)  # plugin normal follows source-to-detector ray

    def test_rigid_station_to_reference_transform_and_validation(self):
        state = self.state_for("ap")
        # R is translated and rotated 90 degrees about Y relative to S.
        t_rs = ((0.0, 0.0, 1.0, 120.0), (0.0, 1.0, 0.0, -35.0),
                (-1.0, 0.0, 0.0, 80.0), (0.0, 0.0, 0.0, 1.0))
        identity = map_device_geometry(state)
        mapped = map_device_geometry(state, t_rs)
        self.assertAlmostEqual(mapped["source_position_ref"][0], identity["source_position_ref"][2] + 120)
        self.assertAlmostEqual(mapped["source_position_ref"][1], identity["source_position_ref"][1] - 35)
        self.assertAlmostEqual(mapped["source_position_ref"][2], -identity["source_position_ref"][0] + 80)
        self.assertAlmostEqual(mapped["source_to_detector_distance_mm"], identity["source_to_detector_distance_mm"], places=7)
        self.assertEqual(map_device_geometry(state, IDENTITY_4), identity)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            map_device_geometry(state, ((2,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)))
        malformed = (
            None,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0, 0), (0, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            (("bad", 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            ((True, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            ((math.inf, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 2)),
        )
        for matrix in malformed:
            with self.subTest(matrix=matrix), self.assertRaisesRegex(ValueError, "T_RS"):
                map_device_geometry(state, matrix)

    def test_zero_sid_returns_inconsistent_geometry_without_exception(self):
        state = DeviceState()
        # Tube focal point and detector active center both become (0, 0, 0).
        set_pose(state, "tube", (0, 430, 0))
        set_pose(state, "detector", (0, -40, 0))
        geometry = map_device_geometry(state)
        self.assertEqual(geometry["source_position_ref"], geometry["detector_center_ref"])
        self.assertEqual(geometry["source_to_detector_distance_mm"], 0.0)
        self.assertIsNone(geometry["detector_normal_ray_alignment"])
        self.assertFalse(geometry["geometry_consistent"])
        self.assertTrue(geometry["orientation_consistent"])

    def test_global_virtualxray_transform_is_not_folded_into_local_geometry(self):
        state = self.state_for("lateral")
        local = map_device_geometry(state)
        # Move local points through two distinct T_WR placements and invert
        # those placements; coordinates expressed in R must be recovered.
        for rotation, translation in (
            (((1,0,0),(0,1,0),(0,0,1)), (0,0,0)),
            (((0,-1,0),(1,0,0),(0,0,1)), (500,200,-80)),
        ):
            for key in ("source_position_ref", "detector_center_ref"):
                point_r = local[key]
                point_w = tuple(sum(rotation[row][col] * point_r[col] for col in range(3)) + translation[row]
                                for row in range(3))
                recovered_r = tuple(sum(rotation[row][col] * (point_w[row] - translation[row])
                                        for row in range(3)) for col in range(3))
                for actual, expected in zip(recovered_r, point_r):
                    self.assertAlmostEqual(actual, expected, places=8)
        self.assertEqual(map_device_geometry(state), local)

    def test_phantom_pose_separate_and_mapping_is_read_only(self):
        state = self.state_for("upright")
        before = state.to_dict()
        first = map_device_geometry(state)
        second = map_device_geometry(state)
        self.assertEqual(first, second)
        self.assertEqual(state.to_dict(), before)
        self.assertIn("phantom_pose_ref", first)
        self.assertNotIn("phantom", plugin_geometry_arguments(first))


if __name__ == "__main__":
    unittest.main()
