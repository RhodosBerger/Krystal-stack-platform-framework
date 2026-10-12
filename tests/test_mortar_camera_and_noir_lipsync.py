"""
Unit Test Suite for Mortar Ballistic Camera Trajectory & L.A. Noire Lip-sync
============================================================================
Verifies:
1. MortarBallisticCameraTrajectory: parabolic arc, apex height, follow-cam, bullet-time.
2. Film-Noir Interrogation Scene: 8 urban spatial rules, ASCII projection, Godot export.
3. L.A. Noire Lip-sync Engine: phonetic viseme decomposition, micro-expressions,
   ASCII portrait rendering, and Godot 4.x blendshape AnimationPlayer tracks.
"""

import math
import unittest

from mimicry_engine.scene_composer import (
    MortarBallisticCameraTrajectory,
    build_noir_mortar_interrogation_scene,
    get_scene,
    SCENES_CATALOG
)
from mimicry_engine.cinematic_character_lipsync import (
    VisemeType,
    EmotionalMicroExpression,
    DialogueScriptLine,
    LANoireCharacterLipsyncEngine,
    CinematicMovieSceneDirector
)

class TestMortarCameraAndNoirLipsync(unittest.TestCase):

    def test_01_mortar_ballistic_trajectory_math(self):
        launch = (-10.0, 1.0, -10.0)
        target = (0.0, 0.0, 0.0)
        apex = 20.0
        mortar = MortarBallisticCameraTrajectory(
            name="TestMortar",
            launch_pos=launch,
            target_pos=target,
            apex_height=apex,
            camera_mode="FOLLOW_SHELL",
            duration=5.0
        )
        # Progress s = 0.0 (launch)
        pos0 = mortar.get_shell_position(0.0)
        self.assertAlmostEqual(pos0[0], launch[0], places=3)
        self.assertAlmostEqual(pos0[1], launch[1], places=3)
        self.assertAlmostEqual(pos0[2], launch[2], places=3)

        # Progress s = 0.5 (apex)
        pos_apex = mortar.get_shell_position(0.5)
        expected_y = 0.5 * (launch[1] + target[1]) + apex
        self.assertAlmostEqual(pos_apex[1], expected_y, places=3)

        # Progress s = 1.0 (impact)
        pos_impact = mortar.get_shell_position(1.0)
        self.assertAlmostEqual(pos_impact[0], target[0], places=3)
        self.assertAlmostEqual(pos_impact[1], target[1], places=3)
        self.assertAlmostEqual(pos_impact[2], target[2], places=3)

        # Tangent velocity calculation
        vel0 = mortar.get_shell_velocity(0.0)
        self.assertGreater(vel0[1], 0.0)  # Ascending at launch
        vel_end = mortar.get_shell_velocity(1.0)
        self.assertLess(vel_end[1], 0.0)    # Plunging downward at impact

    def test_02_mortar_camera_modes(self):
        mortar_follow = MortarBallisticCameraTrajectory(camera_mode="FOLLOW_SHELL", duration=6.0)
        cam_pos, look_target, fov = mortar_follow.get_camera_state(3.0)
        self.assertGreater(look_target[1], -10.0)
        self.assertGreater(fov, 45.0)

        # Cinematic Plunge camera
        mortar_plunge = MortarBallisticCameraTrajectory(camera_mode="CINEMATIC_PLUNGE", duration=6.0)
        cam_p, look_p, _ = mortar_plunge.get_camera_state(3.0)
        # In plunge mode, camera stays near ground looking up at shell
        shell_at_3 = mortar_plunge.get_shell_position(0.5)
        self.assertAlmostEqual(look_p[0], shell_at_3[0], places=3)
        self.assertAlmostEqual(look_p[1], shell_at_3[1], places=3)

        # Bullet-Time mode
        mortar_bullet = MortarBallisticCameraTrajectory(camera_mode="BULLET_TIME", duration=6.0)
        _, _, fov_b = mortar_bullet.get_camera_state(3.0)
        self.assertGreater(fov_b, 40.0)

        # Sample path
        pts = mortar_follow.sample_path(num_points=25)
        self.assertEqual(len(pts), 25)

    def test_03_noir_mortar_interrogation_scene(self):
        scene = build_noir_mortar_interrogation_scene()
        self.assertEqual(scene.scene_id, "SCENE_NOIR_MORTAR_INTERROGATION")
        self.assertEqual(scene.genre, "film_noir_detective")
        self.assertIsNotNone(scene.default_trajectory)
        self.assertIsInstance(scene.default_trajectory, MortarBallisticCameraTrajectory)

        # Rule validation
        self.assertIsNotNone(scene.validation_result)
        self.assertTrue(scene.validation_result.passed)
        self.assertGreater(scene.validation_result.compliance_score, 0.8)

        # ASCII Raymarch Viewport
        ascii_out = scene.render_ascii_view(width=50, height=14, t=2.0)
        lines = ascii_out.splitlines()
        self.assertEqual(len(lines), 14)
        self.assertEqual(len(lines[0]), 50)

        # Godot TSCN export
        tscn = scene.export_godot_tscn()
        self.assertIn("Curve3D_CamTrack", tscn)
        self.assertIn('node name="CameraTrack" type="Path3D"', tscn)
        self.assertIn('metadata/vital_max_hp = 6', tscn)
        self.assertIn("metadata/trajectory_type = \"MortarBallisticCameraTrajectory\"", tscn)

        # Catalog lookup
        sc_from_cat = get_scene("SCENE_NOIR_MORTAR_INTERROGATION")
        self.assertIsNotNone(sc_from_cat)

    def test_04_phonetic_viseme_decomposition(self):
        # Test word to viseme sequence
        text = "mortar shell blast"
        visemes = LANoireCharacterLipsyncEngine.text_to_visemes(text)
        self.assertGreater(len(visemes), 3)

        v_types = [v for v, _ in visemes]
        self.assertIn(VisemeType.MBP, v_types)
        self.assertIn(VisemeType.SS, v_types)

        # Total duration weights sum to 1.0
        total_weight = sum(w for _, w in visemes)
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_05_la_noire_micro_expressions_and_facial_frames(self):
        # 1. Truth: Steady gaze, minimal eye dart
        line_truth = DialogueScriptLine(
            speaker="Detective",
            line="Where were you, Cole?",
            start_time=0.0,
            duration=3.0,
            intended_emotion=EmotionalMicroExpression.TRUTH_STEADY
        )
        frame_truth = LANoireCharacterLipsyncEngine.evaluate_facial_frame(1.5, line_truth)
        self.assertEqual(frame_truth.emotion, EmotionalMicroExpression.TRUTH_STEADY)
        self.assertLess(abs(frame_truth.eye_dart_x), 0.25)
        self.assertLess(frame_truth.brow_furrow, 0.3)

        # 2. Lie: Asymmetrical lip sneer, shifted gaze
        line_lie = DialogueScriptLine(
            speaker="Suspect",
            line="I saw nothing detective.",
            start_time=0.0,
            duration=3.0,
            intended_emotion=EmotionalMicroExpression.LIE_DECEPTIVE
        )
        frame_lie = LANoireCharacterLipsyncEngine.evaluate_facial_frame(1.5, line_lie)
        self.assertEqual(frame_lie.emotion, EmotionalMicroExpression.LIE_DECEPTIVE)
        self.assertGreater(frame_lie.lip_sneer, 0.4)
        self.assertLess(frame_lie.eye_dart_x, -0.4)  # Eye averted to left

        # 3. Doubt: Furrowed brows, restless eye dart
        line_doubt = DialogueScriptLine(
            speaker="Detective",
            line="The trajectory came from the warehouse.",
            start_time=0.0,
            duration=3.0,
            intended_emotion=EmotionalMicroExpression.DOUBT_EVASIVE
        )
        frame_doubt = LANoireCharacterLipsyncEngine.evaluate_facial_frame(1.5, line_doubt)
        self.assertEqual(frame_doubt.emotion, EmotionalMicroExpression.DOUBT_EVASIVE)
        self.assertGreater(frame_doubt.brow_furrow, 0.5)

    def test_06_ascii_portrait_and_godot_facial_export(self):
        script = CinematicMovieSceneDirector.get_noir_movie_script()
        self.assertEqual(len(script), 4)

        # ASCII Portrait rendering
        frame = LANoireCharacterLipsyncEngine.evaluate_facial_frame(1.0, script[0])
        ascii_face = CinematicMovieSceneDirector.render_ascii_face(frame)
        self.assertIn("L.A. NOIRE MOTIONSCAN FACIAL VIEWPORT", ascii_face)
        self.assertIn("Fedora", ascii_face)
        self.assertIn(frame.emotion.value, ascii_face)

        # Godot Blendshape TSCN export
        godot_tscn = CinematicMovieSceneDirector.export_godot_lipsync_tscn(script)
        self.assertIn('node name="NoirMovieCharacter" type="Node3D"', godot_tscn)
        self.assertIn('metadata/vital_max_hp = 6', godot_tscn)
        self.assertIn('metadata/motionscan_rig = "LA_NOIRE_PHONETIC_V4"', godot_tscn)
        self.assertIn('tracks/0/path = NodePath("CharacterHead/FaceMesh:blend_shapes/jaw_open")', godot_tscn)
        self.assertIn('tracks/1/path = NodePath("CharacterHead/FaceMesh:blend_shapes/lip_sneer")', godot_tscn)
        self.assertIn('PackedFloat32Array(', godot_tscn)

if __name__ == "__main__":
    unittest.main()
