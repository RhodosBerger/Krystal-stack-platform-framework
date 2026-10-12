"""
Unit Test Suite for Krystal-Stack Scene Composer Trajectories & Kinematics
=========================================================================
Verifies camera paths (Orbital, FlythroughSpline, CinematicDollyZoom, SpiralAscent),
ASCII raymarching projections, Godot 4.x .tscn Path3D/PathFollow3D tracks,
and invariant preservation (vital_max_hp == 6).
"""

import math
import unittest

from mimicry_engine.scene_composer import (
    CameraTrajectory,
    OrbitalTrajectory,
    FlythroughSplineTrajectory,
    CinematicDollyZoomTrajectory,
    SpiralAscentTrajectory,
    GameScene,
    SceneActor,
    build_old_town_prague_square,
    build_cyberpunk_district
)

class TestSceneComposerTrajectories(unittest.TestCase):

    def test_01_camera_trajectory_base(self):
        traj = CameraTrajectory("BaseTrack", duration=5.0)
        self.assertEqual(traj.name, "BaseTrack")
        self.assertEqual(traj.duration, 5.0)
        with self.assertRaises(NotImplementedError):
            traj.get_camera_state(0.0)

        info = traj.to_dict()
        self.assertEqual(info["type"], "CameraTrajectory")
        self.assertEqual(info["name"], "BaseTrack")
        self.assertEqual(info["duration"], 5.0)

    def test_02_orbital_trajectory(self):
        orbit = OrbitalTrajectory(
            name="Orbital_Alpha",
            center=(0.0, 1.0, 0.0),
            radius=5.0,
            height=2.0,
            angular_speed=0.5,
            elevation_oscillation=0.5,
            fov=55.0,
            duration=10.0
        )
        # t = 0
        pos0, ta0, fov0 = orbit.get_camera_state(0.0)
        self.assertAlmostEqual(pos0[0], 0.0, places=5)
        self.assertAlmostEqual(pos0[1], 3.0, places=5)  # 1.0 + 2.0
        self.assertAlmostEqual(pos0[2], -5.0, places=5)
        self.assertEqual(ta0, (0.0, 1.0, 0.0))
        self.assertEqual(fov0, 55.0)

        # t > 0
        pos1, _, _ = orbit.get_camera_state(math.pi)
        # ang = pi * 0.5 = pi/2 -> sin(pi/2)=1, cos(pi/2)=0
        self.assertAlmostEqual(pos1[0], 5.0, places=4)
        self.assertAlmostEqual(pos1[2], 0.0, places=4)

        pts = orbit.sample_path(num_points=16)
        self.assertEqual(len(pts), 16)
        d = orbit.to_dict()
        self.assertEqual(d["type"], "OrbitalTrajectory")
        self.assertEqual(d["radius"], 5.0)

    def test_03_flythrough_spline_trajectory(self):
        wps = [
            (-4.0, 2.0, -4.0),
            (-1.0, 1.5, 0.0),
            (2.0, 2.5, 2.0),
            (5.0, 3.0, -2.0)
        ]
        spline = FlythroughSplineTrajectory(
            name="Spline_Tour",
            waypoints=wps,
            duration=8.0,
            fov=68.0,
            closed=False
        )
        pos_start, target_start, fov = spline.get_camera_state(0.0)
        self.assertAlmostEqual(pos_start[0], -4.0, places=3)
        self.assertAlmostEqual(pos_start[1], 2.0, places=3)
        self.assertAlmostEqual(pos_start[2], -4.0, places=3)
        self.assertEqual(fov, 68.0)

        # Verify target is forward along direction
        dx = target_start[0] - pos_start[0]
        dy = target_start[1] - pos_start[1]
        dz = target_start[2] - pos_start[2]
        dist_to_target = math.sqrt(dx*dx + dy*dy + dz*dz)
        self.assertGreater(dist_to_target, 0.5)

        # Sample path
        pts = spline.sample_path(num_points=20)
        self.assertEqual(len(pts), 20)
        self.assertEqual(spline.to_dict()["closed"], False)

        # Test closed spline
        closed_spline = FlythroughSplineTrajectory(waypoints=wps, closed=True)
        pos_c0, _, _ = closed_spline.get_camera_state(0.0)
        pos_cend, _, _ = closed_spline.get_camera_state(closed_spline.duration)
        self.assertAlmostEqual(pos_c0[0], pos_cend[0], places=3)
        self.assertAlmostEqual(pos_c0[1], pos_cend[1], places=3)
        self.assertAlmostEqual(pos_c0[2], pos_cend[2], places=3)

    def test_04_cinematic_dolly_zoom_trajectory(self):
        target = (0.0, 0.0, 0.0)
        start_pos = (0.0, 0.0, -10.0)
        end_pos = (0.0, 0.0, -2.0)
        dolly = CinematicDollyZoomTrajectory(
            name="VertigoEffect",
            target=target,
            start_pos=start_pos,
            end_pos=end_pos,
            initial_fov=30.0,
            duration=4.0
        )
        pos0, _, fov0 = dolly.get_camera_state(0.0)
        self.assertAlmostEqual(pos0[2], -10.0, places=3)
        self.assertAlmostEqual(fov0, 30.0, places=1)

        # Camera moves closer (t = duration): FOV must broaden substantially
        pos_end, _, fov_end = dolly.get_camera_state(4.0)
        self.assertAlmostEqual(pos_end[2], -2.0, places=3)
        self.assertGreater(fov_end, fov0 + 50.0)  # Classic Vertigo compensation

        pts = dolly.sample_path(num_points=10)
        self.assertEqual(len(pts), 10)

    def test_05_spiral_ascent_trajectory(self):
        spiral = SpiralAscentTrajectory(
            name="ClocktowerAscent",
            center=(0.0, 0.0, 0.0),
            start_radius=6.0,
            end_radius=2.5,
            start_height=0.5,
            end_height=8.0,
            revolutions=2.0,
            duration=10.0,
            fov=60.0
        )
        pos0, target0, fov0 = spiral.get_camera_state(0.0)
        self.assertAlmostEqual(pos0[1], 0.5, places=3)
        radius0 = math.sqrt(pos0[0]**2 + pos0[2]**2)
        self.assertAlmostEqual(radius0, 6.0, places=3)

        pos_top, target_top, _ = spiral.get_camera_state(10.0)
        self.assertAlmostEqual(pos_top[1], 8.0, places=3)
        radius_top = math.sqrt(pos_top[0]**2 + pos_top[2]**2)
        self.assertAlmostEqual(radius_top, 2.5, places=3)
        self.assertGreater(target_top[1], target0[1])

    def test_06_scene_ascii_raymarch_with_trajectory(self):
        scene = build_cyberpunk_district()
        orbit = OrbitalTrajectory(radius=6.0, height=2.0, angular_speed=0.2)
        
        # Test ASCII render with explicit trajectory
        ascii_out = scene.render_ascii_view(width=40, height=12, t=1.0, trajectory=orbit)
        lines = ascii_out.splitlines()
        self.assertEqual(len(lines), 12)
        self.assertEqual(len(lines[0]), 40)
        # Check non-empty rendering (some actors hit)
        has_glyphs = any(c in "#*+@" for line in lines for c in line)
        self.assertTrue(has_glyphs)

        # Test ASCII render with default_trajectory
        scene.set_trajectory(orbit)
        ascii_out_default = scene.render_ascii_view(width=40, height=12, t=1.0)
        self.assertEqual(ascii_out, ascii_out_default)

    def test_07_godot_tscn_export_with_path3d(self):
        scene = build_old_town_prague_square()
        spiral = SpiralAscentTrajectory(
            name="PragueSpiral",
            center=(0.0, 0.0, 1.8),
            start_radius=7.0,
            end_radius=3.0,
            start_height=0.5,
            end_height=6.0,
            duration=12.0
        )
        scene.set_trajectory(spiral)
        tscn = scene.export_godot_tscn()

        # Invariant checks
        self.assertIn("metadata/vital_max_hp = 6", tscn)
        for actor in scene.actors:
            self.assertLessEqual(actor.vital_hp, 6)

        # Track and Curve3D checks
        self.assertIn("Curve3D_CamTrack", tscn)
        self.assertIn("PackedVector3Array(", tscn)
        self.assertIn('node name="CameraTrack" type="Path3D"', tscn)
        self.assertIn('node name="PathFollow3D" type="PathFollow3D"', tscn)
        self.assertIn('metadata/trajectory_type = "SpiralAscentTrajectory"', tscn)
        self.assertIn("metadata/duration = 12.0", tscn)

        # Serialization to dict
        d = scene.to_dict()
        self.assertIsNotNone(d["trajectory"])
        self.assertEqual(d["trajectory"]["type"], "SpiralAscentTrajectory")

if __name__ == "__main__":
    unittest.main()
