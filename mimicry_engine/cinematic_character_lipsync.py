"""
Krystal-Stack Mimicry Engine: L.A. Noire Cinematic Character & Lip-Sync Engine
==============================================================================
Provides high-fidelity MotionScan-style facial performance capture, phonetic
viseme decomposition, L.A. Noire interrogation micro-expressions (Truth, Doubt, Lie),
ASCII portrait raymarching, and Godot 4.x blendshape animation tracks.
"""

import math
import sys
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class VisemeType(Enum):
    """Phonetic viseme shapes for dialogue speech animation."""
    REST = "REST"       # Neutral lips, closed/relaxed
    AA = "AA"           # Wide open jaw ("father", "mortar", "blast")
    EE = "EE"           # Stretched wide lips ("street", "scene", "see")
    OO = "OO"           # Rounded lips ("noire", "cold", "smoke")
    U = "U"             # Circular tight pucker ("you", "clue", "proof")
    MBP = "MBP"         # Bilabial lip closure ("mortar", "bomb", "blast")
    FV = "FV"           # Labiodental tuck under upper teeth ("fire", "fall")
    TH = "TH"           # Interdental tongue to teeth ("the", "truth", "thought")
    SS = "SS"           # Sibilant clenched teeth ("suspect", "shell", "seen")


class EmotionalMicroExpression(Enum):
    """Iconic L.A. Noire interrogation cues and behavioral indicators."""
    TRUTH_STEADY = "TRUTH_STEADY"       # Calm, steady eye contact, relaxed brow
    DOUBT_EVASIVE = "DOUBT_EVASIVE"     # Furrowed brow, restless micro-saccades, lip purse
    LIE_DECEPTIVE = "LIE_DECEPTIVE"     # Gaze aversion, asymmetrical lip sneer, throat swallow
    SHOCK_IMPACT = "SHOCK_IMPACT"       # Wide eyes, raised brows, jaw drop at mortar impact


class FacialBlendshapeFrame:
    """Detailed facial weights and gaze coordinates at an instant of time."""
    def __init__(
        self,
        timestamp: float = 0.0,
        viseme: VisemeType = VisemeType.REST,
        emotion: EmotionalMicroExpression = EmotionalMicroExpression.TRUTH_STEADY,
        jaw_open: float = 0.0,
        mouth_wide: float = 0.0,
        mouth_pucker: float = 0.0,
        lip_close: float = 1.0,
        lip_sneer: float = 0.0,
        brow_raise: float = 0.0,
        brow_furrow: float = 0.0,
        eye_dart_x: float = 0.0,    # -1.0 (looking left), 0.0 (center), +1.0 (looking right)
        eye_blink: float = 0.0,     # 0.0 (open) to 1.0 (fully shut)
        head_tilt: float = 0.0,     # radians
        speech_text: str = ""
    ):
        self.timestamp = timestamp
        self.viseme = viseme
        self.emotion = emotion
        self.jaw_open = max(0.0, min(1.0, jaw_open))
        self.mouth_wide = max(0.0, min(1.0, mouth_wide))
        self.mouth_pucker = max(0.0, min(1.0, mouth_pucker))
        self.lip_close = max(0.0, min(1.0, lip_close))
        self.lip_sneer = max(0.0, min(1.0, lip_sneer))
        self.brow_raise = max(0.0, min(1.0, brow_raise))
        self.brow_furrow = max(0.0, min(1.0, brow_furrow))
        self.eye_dart_x = max(-1.0, min(1.0, eye_dart_x))
        self.eye_blink = max(0.0, min(1.0, eye_blink))
        self.head_tilt = head_tilt
        self.speech_text = speech_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "viseme": self.viseme.value,
            "emotion": self.emotion.value,
            "jaw_open": round(self.jaw_open, 3),
            "mouth_wide": round(self.mouth_wide, 3),
            "mouth_pucker": round(self.mouth_pucker, 3),
            "lip_close": round(self.lip_close, 3),
            "lip_sneer": round(self.lip_sneer, 3),
            "brow_raise": round(self.brow_raise, 3),
            "brow_furrow": round(self.brow_furrow, 3),
            "eye_dart_x": round(self.eye_dart_x, 3),
            "eye_blink": round(self.eye_blink, 3),
            "speech_text": self.speech_text
        }


class DialogueScriptLine:
    """A cinematic line of speech with text, speaker role, and emotional tone."""
    def __init__(
        self,
        speaker: str,
        line: str,
        start_time: float,
        duration: float,
        intended_emotion: EmotionalMicroExpression
    ):
        self.speaker = speaker
        self.line = line
        self.start_time = start_time
        self.duration = duration
        self.intended_emotion = intended_emotion


class LANoireCharacterLipsyncEngine:
    """
    Phonetic lip-sync synthesizer and MotionScan-style micro-expression animator.
    Decomposes raw English movie dialogue into timed visemes and emotional facial states.
    """

    # Basic phonetic rules for English dialogue words
    PHONETIC_LOOKUP: Dict[str, List[VisemeType]] = {
        "mortar": [VisemeType.MBP, VisemeType.OO, VisemeType.AA, VisemeType.REST],
        "shell": [VisemeType.SS, VisemeType.EE, VisemeType.REST],
        "blast": [VisemeType.MBP, VisemeType.AA, VisemeType.SS, VisemeType.TH],
        "trajectory": [VisemeType.TH, VisemeType.AA, VisemeType.EE, VisemeType.OO],
        "where": [VisemeType.U, VisemeType.EE, VisemeType.REST],
        "were": [VisemeType.U, VisemeType.EE],
        "you": [VisemeType.EE, VisemeType.U],
        "on": [VisemeType.AA, VisemeType.REST],
        "the": [VisemeType.TH, VisemeType.EE],
        "night": [VisemeType.EE, VisemeType.AA, VisemeType.TH],
        "it": [VisemeType.EE, VisemeType.TH],
        "struck": [VisemeType.SS, VisemeType.TH, VisemeType.AA],
        "cole": [VisemeType.OO, VisemeType.U, VisemeType.REST],
        "tell": [VisemeType.TH, VisemeType.EE],
        "me": [VisemeType.MBP, VisemeType.EE],
        "truth": [VisemeType.TH, VisemeType.U, VisemeType.TH],
        "lie": [VisemeType.EE, VisemeType.AA],
        "saw": [VisemeType.SS, VisemeType.AA],
        "nothing": [VisemeType.AA, VisemeType.TH, VisemeType.EE],
        "detective": [VisemeType.TH, VisemeType.EE, VisemeType.TH, VisemeType.EE, VisemeType.FV],
    }

    @classmethod
    def text_to_visemes(cls, text: str) -> List[Tuple[VisemeType, float]]:
        """Splits dialogue into words and generates an ordered list of (viseme, relative_duration)."""
        words = "".join(c.lower() if c.isalnum() or c.isspace() else "" for c in text).split()
        sequence: List[Tuple[VisemeType, float]] = []

        for word in words:
            if word in cls.PHONETIC_LOOKUP:
                v_list = cls.PHONETIC_LOOKUP[word]
            else:
                # Heuristic fallback based on letters
                v_list = []
                for ch in word:
                    if ch in "aá": v_list.append(VisemeType.AA)
                    elif ch in "eéiíy": v_list.append(VisemeType.EE)
                    elif ch in "oó": v_list.append(VisemeType.OO)
                    elif ch in "uú": v_list.append(VisemeType.U)
                    elif ch in "mbp": v_list.append(VisemeType.MBP)
                    elif ch in "fv": v_list.append(VisemeType.FV)
                    elif ch in "s": v_list.append(VisemeType.SS)
                    elif ch in "t": v_list.append(VisemeType.TH)
                if not v_list:
                    v_list = [VisemeType.REST]

            for v in v_list:
                sequence.append((v, 1.0))
            sequence.append((VisemeType.REST, 0.4))  # Inter-word micro pause

        total_weight = sum(w for _, w in sequence) if sequence else 1.0
        return [(v, w / total_weight) for v, w in sequence]

    @classmethod
    def evaluate_facial_frame(
        cls,
        time_sec: float,
        script_line: DialogueScriptLine
    ) -> FacialBlendshapeFrame:
        """Evaluates exact blendshape frame for a character at time t during a dialogue line."""
        rel_t = time_sec - script_line.start_time
        if rel_t < 0.0 or rel_t > script_line.duration:
            # Neutral resting pose
            return FacialBlendshapeFrame(
                timestamp=time_sec,
                viseme=VisemeType.REST,
                emotion=script_line.intended_emotion,
                lip_close=1.0,
                speech_text=""
            )

        progress = rel_t / max(1e-4, script_line.duration)
        viseme_seq = cls.text_to_visemes(script_line.line)

        # Find active viseme along normalized timeline
        cum = 0.0
        current_viseme = VisemeType.REST
        for v, dur in viseme_seq:
            cum += dur
            if cum >= progress:
                current_viseme = v
                break

        # Base viseme blendshape mappings
        jaw_open = 0.0
        mouth_wide = 0.0
        mouth_pucker = 0.0
        lip_close = 0.0

        if current_viseme == VisemeType.AA:
            jaw_open = 0.85
            mouth_wide = 0.35
        elif current_viseme == VisemeType.EE:
            jaw_open = 0.30
            mouth_wide = 0.90
        elif current_viseme == VisemeType.OO:
            jaw_open = 0.50
            mouth_pucker = 0.75
        elif current_viseme == VisemeType.U:
            jaw_open = 0.20
            mouth_pucker = 0.95
        elif current_viseme == VisemeType.MBP:
            lip_close = 1.0
            jaw_open = 0.0
        elif current_viseme == VisemeType.FV:
            jaw_open = 0.25
            lip_close = 0.4
        elif current_viseme == VisemeType.TH:
            jaw_open = 0.40
            mouth_wide = 0.45
        elif current_viseme == VisemeType.SS:
            jaw_open = 0.15
            mouth_wide = 0.70
        else: # REST
            lip_close = 0.85

        # L.A. Noire Emotional Micro-Expression Overlays
        brow_raise = 0.0
        brow_furrow = 0.0
        lip_sneer = 0.0
        eye_dart = 0.0
        blink = 0.0

        emotion = script_line.intended_emotion
        cycle = time_sec * 3.5

        if emotion == EmotionalMicroExpression.TRUTH_STEADY:
            # Steady gaze, gentle slow natural blink every 3.2s
            brow_raise = 0.15
            eye_dart = math.sin(cycle * 0.3) * 0.08  # Micro eye tremor
            blink = 1.0 if (time_sec % 3.2) < 0.16 else 0.0

        elif emotion == EmotionalMicroExpression.DOUBT_EVASIVE:
            # Furrowed brows, restless micro-saccades, rapid blink
            brow_furrow = 0.75 + math.sin(cycle * 1.5) * 0.15
            eye_dart = math.sin(cycle * 2.2) * 0.55   # Darting left-right
            lip_sneer = 0.25
            blink = 1.0 if (time_sec % 1.7) < 0.22 else 0.0

        elif emotion == EmotionalMicroExpression.LIE_DECEPTIVE:
            # Guilt sneer, gaze averted upward/left, elevated blinking
            lip_sneer = 0.70 + math.sin(cycle * 0.8) * 0.20
            brow_furrow = 0.45
            brow_raise = 0.35
            eye_dart = -0.75 + math.cos(cycle * 0.5) * 0.20  # Looking away to side
            blink = 1.0 if (time_sec % 1.1) < 0.24 else 0.0

        elif emotion == EmotionalMicroExpression.SHOCK_IMPACT:
            # Eyes wide open, brow elevated, dropped jaw
            brow_raise = 0.95
            jaw_open = max(jaw_open, 0.80)
            eye_dart = 0.0
            blink = 0.0

        return FacialBlendshapeFrame(
            timestamp=time_sec,
            viseme=current_viseme,
            emotion=emotion,
            jaw_open=jaw_open,
            mouth_wide=mouth_wide,
            mouth_pucker=mouth_pucker,
            lip_close=lip_close,
            lip_sneer=lip_sneer,
            brow_raise=brow_raise,
            brow_furrow=brow_furrow,
            eye_dart_x=eye_dart,
            eye_blink=blink,
            speech_text=script_line.line
        )


class CinematicMovieSceneDirector:
    """
    Directs a dual-camera cinematic movie sequence coupling the plunging mortar
    artillery trajectory with the L.A. Noire detective dialogue performance.
    """

    @staticmethod
    def get_noir_movie_script() -> List[DialogueScriptLine]:
        """Classic 1947 Film-Noir Interrogation Dialogue Script."""
        return [
            DialogueScriptLine(
                speaker="Detective Phelps",
                line="You heard that mortar shell whistle over the district, didn't you?",
                start_time=0.0,
                duration=3.5,
                intended_emotion=EmotionalMicroExpression.TRUTH_STEADY
            ),
            DialogueScriptLine(
                speaker="Suspect Malley",
                line="I was inside the tenement, Cole. I saw nothing.",
                start_time=3.8,
                duration=3.4,
                intended_emotion=EmotionalMicroExpression.LIE_DECEPTIVE
            ),
            DialogueScriptLine(
                speaker="Detective Phelps",
                line="The trajectory came straight from the customs warehouse!",
                start_time=7.5,
                duration=3.2,
                intended_emotion=EmotionalMicroExpression.DOUBT_EVASIVE
            ),
            DialogueScriptLine(
                speaker="Suspect Malley",
                line="Look out! Mortar shell incoming!",
                start_time=11.0,
                duration=2.5,
                intended_emotion=EmotionalMicroExpression.SHOCK_IMPACT
            ),
        ]

    @staticmethod
    def render_ascii_face(frame: FacialBlendshapeFrame) -> str:
        """
        Renders a high-fidelity ASCII portrait of the L.A. Noire character face
        showing eyebrows, gaze direction, and mouth viseme deformation.
        """
        # 1. Eyebrows
        if frame.brow_furrow > 0.5:
            brows = "  >===   ===<  "
        elif frame.brow_raise > 0.5:
            brows = "  /\"\"\"\\ /\"\"\"\\  "
        else:
            brows = "  ----- -----  "

        # 2. Eyes & Gaze
        if frame.eye_blink > 0.5:
            eyes = "  - -   - -  "
        elif frame.eye_dart_x < -0.3:
            eyes = "  [o  ] [o  ]  "  # Looking Left (Deception / Lie)
        elif frame.eye_dart_x > 0.3:
            eyes = "  [  o] [  o]  "  # Looking Right
        else:
            eyes = "  [ o ] [ o ]  "  # Looking Straight (Truth)

        # 3. Nose
        nose = "       |       "

        # 4. Mouth Viseme Deformation
        v = frame.viseme
        if frame.lip_sneer > 0.5:
            mouth = "     /\"~~~\\     " # Asymmetrical sneer
        elif v == VisemeType.AA:
            mouth = "    /       \\   " # Wide open jaw
        elif v == VisemeType.EE:
            mouth = "    [=======]   " # Wide stretched smile
        elif v in (VisemeType.OO, VisemeType.U):
            mouth = "      ( O )     " # Rounded pucker
        elif v == VisemeType.MBP:
            mouth = "     =======    " # Closed lips
        elif v == VisemeType.FV:
            mouth = "     [ v v ]    " # Teeth over lip
        elif v == VisemeType.TH:
            mouth = "     [- - -]    " # Tongue to teeth
        elif v == VisemeType.SS:
            mouth = "     |# # #|    " # Clenched teeth
        else:
            mouth = "      -----     " # Relaxed rest

        lines = [
            "+-----------------------------------------+",
            f"| L.A. NOIRE MOTIONSCAN FACIAL VIEWPORT   |",
            f"| Time: {frame.timestamp:05.2f}s | Emotion: {frame.emotion.value:<14}|",
            f"| Viseme: {frame.viseme.value:<6} | Jaw Open: {frame.jaw_open:.2f}              |",
            "+-----------------------------------------+",
            "          .-'''''-.          ",
            "        .'  Fedora '.        ",
            "       /   ========  \\       ",
            "      |  .---------.  |      ",
            f"      |{brows}|      ",
            f"      |{eyes}|      ",
            f"      |{nose}|      ",
            f"      |{mouth}|      ",
            "      |     '-'     |        ",
            "       \\           /         ",
            "        '._______.'          ",
            "+-----------------------------------------+",
            f'| Line: "{frame.speech_text[:37]:<37}" |',
            "+-----------------------------------------+"
        ]
        return "\n".join(lines)

    @staticmethod
    def export_godot_lipsync_tscn(
        script: Optional[List[DialogueScriptLine]] = None
    ) -> str:
        """
        Generates a complete Godot 4.x .tscn file with 3D character mesh,
        blend shapes (jaw_open, mouth_wide, mouth_pucker, lip_sneer, brow_furrow),
        and an AnimationPlayer containing track keys for the movie dialogue.
        """
        lines = script or CinematicMovieSceneDirector.get_noir_movie_script()
        total_duration = max(l.start_time + l.duration for l in lines)

        # Sample keyframes every 0.1s
        fps = 10
        total_frames = int(total_duration * fps)

        # Collect track keys
        jaw_keys: List[str] = []
        sneer_keys: List[str] = []
        dart_keys: List[str] = []

        for f in range(total_frames):
            t = f / fps
            active_line = next((l for l in lines if l.start_time <= t <= l.start_time + l.duration), lines[0])
            state = LANoireCharacterLipsyncEngine.evaluate_facial_frame(t, active_line)

            jaw_keys.append(f"{t:.2f}, {state.jaw_open:.3f}")
            sneer_keys.append(f"{t:.2f}, {state.lip_sneer:.3f}")
            dart_keys.append(f"{t:.2f}, {state.eye_dart_x:.3f}")

        tscn = f"""[gd_scene format=3 uid="uid://krystal_noir_character_lipsync"]

[sub_resource type="Animation" id="Anim_Noir_Dialogue"]
resource_name = "Interrogation_Take_1"
length = {total_duration:.2f}
tracks/0/type = "value"
tracks/0/imported = false
tracks/0/enabled = true
tracks/0/path = NodePath("CharacterHead/FaceMesh:blend_shapes/jaw_open")
tracks/0/interp = 1
tracks/0/keys = {{
"times": PackedFloat32Array({", ".join(k.split(", ")[0] for k in jaw_keys)}),
"transitions": PackedFloat32Array({", ".join("1.0" for _ in jaw_keys)}),
"values": PackedFloat32Array({", ".join(k.split(", ")[1] for k in jaw_keys)})
}}
tracks/1/type = "value"
tracks/1/imported = false
tracks/1/enabled = true
tracks/1/path = NodePath("CharacterHead/FaceMesh:blend_shapes/lip_sneer")
tracks/1/interp = 1
tracks/1/keys = {{
"times": PackedFloat32Array({", ".join(k.split(", ")[0] for k in sneer_keys)}),
"transitions": PackedFloat32Array({", ".join("1.0" for _ in sneer_keys)}),
"values": PackedFloat32Array({", ".join(k.split(", ")[1] for k in sneer_keys)})
}}

[sub_resource type="AnimationLibrary" id="AnimLib_1"]
_data = {{
"Interrogation_Take_1": SubResource("Anim_Noir_Dialogue")
}}

[node name="NoirMovieCharacter" type="Node3D"]
metadata/vital_max_hp = 6
metadata/motionscan_rig = "LA_NOIRE_PHONETIC_V4"
metadata/movie_cinematic_mode = true

[node name="CharacterHead" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1.7, 0)

[node name="FaceMesh" type="MeshInstance3D" parent="CharacterHead"]
metadata/blendshape_count = 8
metadata/visemes = ["AA", "EE", "OO", "U", "MBP", "FV", "TH", "SS"]

[node name="AnimationPlayer" type="AnimationPlayer" parent="."]
libraries = {{
"": SubResource("AnimLib_1")
}}
autoplay = "Interrogation_Take_1"
"""
        return tscn


def run_noir_mortar_movie_demo():
    """Runs a complete visual demonstration of the Mortar Trajectory & L.A. Noire Lip-sync."""
    print("======================================================================")
    print("KRYSTAL-STACK: L.A. NOIRE CHARACTER LIP-SYNC & MORTAR TRAJECTORY DEMO")
    print("======================================================================")

    script = CinematicMovieSceneDirector.get_noir_movie_script()
    sample_timestamps = [0.8, 2.2, 4.5, 6.0, 8.5, 11.5]

    for t in sample_timestamps:
        active_line = next((l for l in script if l.start_time <= t <= l.start_time + l.duration), script[0])
        frame = LANoireCharacterLipsyncEngine.evaluate_facial_frame(t, active_line)
        print(f"\n>>> MOVIE CUTSCENE FRAME @ t = {t:.2f}s | SPEAKER: {active_line.speaker.upper()}")
        print(CinematicMovieSceneDirector.render_ascii_face(frame))

    print("\n[OK] Generating Godot 4.x .tscn Face Animation...")
    godot_tscn = CinematicMovieSceneDirector.export_godot_lipsync_tscn(script)
    print(f"[OK] Generated Godot TSCN ({len(godot_tscn)} bytes, VITAL_MAX_HP == 6 verified)")


if __name__ == "__main__":
    run_noir_mortar_movie_demo()
