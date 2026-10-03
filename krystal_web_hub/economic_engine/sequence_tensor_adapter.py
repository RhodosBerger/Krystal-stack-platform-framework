# ==============================================================================
# KRYSTAL-STACK: SEQUENCE TENSOR REPLICATION ADAPTER & CACHED PALETTE ENGINE
# ==============================================================================
# Implements:
#   1. Cached Palette Tensor Cache (indexed matrix lookup for visual & tile representations).
#   2. Sequence Tensor Recorder (temporal multi-frame 3D/4D matrix timeline).
#   3. Cross-Room Context Replicator (state synchronization and reproduction across match chambers).
# ==============================================================================

import time
import copy
from typing import Dict, List, Any, Optional, Tuple

# Pre-defined Palette of visual tokens (Colors, ASCII Glyphs, and Godot 4 Tile Markers)
CANONICAL_PALETTES: Dict[str, Dict[int, Dict[str, Any]]] = {
    "hex_arena_pbr": {
        0: {"symbol": ".", "name": "Void / Unclaimed", "rgb": [20, 24, 33], "material": "void_shader"},
        1: {"symbol": "C", "name": "Crystal Hexagon", "rgb": [70, 180, 255], "material": "crystal_pbr"},
        2: {"symbol": "T", "name": "Toxic Slime Slag", "rgb": [80, 220, 100], "material": "toxic_slime_pbr"},
        3: {"symbol": "D", "name": "Druidic Ancient Roots", "rgb": [210, 150, 60], "material": "druid_amber_pbr"},
        4: {"symbol": "W", "name": "Ward Energy Bubble", "rgb": [255, 230, 80], "material": "ward_shield_fx"},
        5: {"symbol": "H", "name": "Hero / Champion", "rgb": [255, 255, 255], "material": "hero_emissive"},
        6: {"symbol": "+", "name": "Healer Unit", "rgb": [120, 255, 220], "material": "healer_aura_fx"},
        7: {"symbol": "X", "name": "Hazardous Decaying Zone", "rgb": [180, 40, 60], "material": "hazard_fire_pbr"}
    },
    "ascii_spectator": {
        0: {"symbol": " ", "name": "Empty"},
        1: {"symbol": "#", "name": "Crystal Wall"},
        2: {"symbol": "~", "name": "Toxic Acid"},
        3: {"symbol": "%", "name": "Forest Canopy"},
        4: {"symbol": "O", "name": "Ward Shield"},
        5: {"symbol": "@", "name": "Supernatural Hero"},
        6: {"symbol": "&", "name": "Mage Healer"},
        7: {"symbol": "!", "name": "Mortar Impact"}
    }
}

class PaletteTensorCache:
    """
    Manages cached visual palettes, allowing fast index-based 2D matrix encoding
    and tensor frame reconstruction.
    """

    def __init__(self):
        self._palettes: Dict[str, Dict[int, Dict[str, Any]]] = copy.deepcopy(CANONICAL_PALETTES)

    def register_palette(self, name: str, palette_map: Dict[int, Dict[str, Any]]) -> None:
        self._palettes[name] = copy.deepcopy(palette_map)

    def get_palette(self, name: str) -> Dict[int, Dict[str, Any]]:
        if name not in self._palettes:
            raise KeyError(f"Palette '{name}' is not registered in PaletteTensorCache.")
        return self._palettes[name]

    def encode_matrix(self, grid_2d: List[List[int]], palette_name: str = "hex_arena_pbr") -> List[List[int]]:
        """Validates and bounds indices against the palette."""
        palette = self.get_palette(palette_name)
        max_idx = max(palette.keys())
        encoded = []
        for row in grid_2d:
            encoded_row = [max(0, min(max_idx, int(val))) for val in row]
            encoded.append(encoded_row)
        return encoded

    def reproduce_tensor_frame(
        self,
        matrix_2d: List[List[int]],
        palette_name: str = "hex_arena_pbr"
    ) -> List[List[List[int]]]:
        """
        Reproduces a 3D RGB tensor [H x W x 3] from the cached palette.
        """
        palette = self.get_palette(palette_name)
        h = len(matrix_2d)
        w = len(matrix_2d[0]) if h > 0 else 0
        tensor_frame = []
        for r in range(h):
            row_rgb = []
            for c in range(w):
                idx = matrix_2d[r][c]
                entry = palette.get(idx, palette[0])
                rgb = entry.get("rgb", [0, 0, 0])
                row_rgb.append(list(rgb))
            tensor_frame.append(row_rgb)
        return tensor_frame

    def reproduce_ascii_frame(
        self,
        matrix_2d: List[List[int]],
        palette_name: str = "hex_arena_pbr"
    ) -> str:
        """
        Reproduces an ASCII text frame representation from matrix and cached palette.
        """
        palette = self.get_palette(palette_name)
        lines = []
        for row in matrix_2d:
            line_chars = [palette.get(val, palette[0])["symbol"] for val in row]
            lines.append("".join(line_chars))
        return "\n".join(lines)


class SequenceTensorRecorder:
    """
    Records temporal series of matrix states and actions for high-efficiency
    playback and cross-match replication.
    """

    def __init__(self, room_id: str, default_palette: str = "hex_arena_pbr"):
        self.room_id = room_id
        self.default_palette = default_palette
        self.frames: List[Dict[str, Any]] = []

    def record_frame(
        self,
        matrix_2d: List[List[int]],
        action_label: str = "state_tick",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Appends a new state matrix frame to the recorded timeline."""
        frame_idx = len(self.frames)
        frame_entry = {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "matrix": [list(row) for row in matrix_2d],
            "action_label": action_label,
            "metadata": metadata or {}
        }
        self.frames.append(frame_entry)
        return frame_idx

    def get_timeline_length(self) -> int:
        return len(self.frames)

    def extract_3d_matrix_tensor(self) -> List[List[List[int]]]:
        """Returns 3D Tensor of dimensions [T x H x W]."""
        return [f["matrix"] for f in self.frames]


class CrossRoomContextReplicator:
    """
    Manages rooms/chambers and reproduces recorded sequence contexts across
    active multiplayer match rooms and spectator viewports.
    """

    def __init__(self):
        self.palette_cache = PaletteTensorCache()
        self.rooms: Dict[str, SequenceTensorRecorder] = {}
        self.room_metadata: Dict[str, Dict[str, Any]] = {}

    def create_room(
        self,
        room_id: str,
        palette_name: str = "hex_arena_pbr",
        initial_meta: Optional[Dict[str, Any]] = None
    ) -> SequenceTensorRecorder:
        recorder = SequenceTensorRecorder(room_id=room_id, default_palette=palette_name)
        self.rooms[room_id] = recorder
        self.room_metadata[room_id] = initial_meta or {
            "created_at": time.time(),
            "palette": palette_name,
            "replicated_from": None
        }
        return recorder

    def record_room_event(
        self,
        room_id: str,
        state_matrix: List[List[int]],
        action_label: str = "action",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        if room_id not in self.rooms:
            self.create_room(room_id)
        encoded_matrix = self.palette_cache.encode_matrix(
            state_matrix,
            palette_name=self.room_metadata[room_id].get("palette", "hex_arena_pbr")
        )
        return self.rooms[room_id].record_frame(encoded_matrix, action_label, metadata)

    def replicate_context_to_room(
        self,
        source_room_id: str,
        target_room_id: str
    ) -> Dict[str, Any]:
        """
        Replicates the full sequence tensor context from source room to target room.
        """
        if source_room_id not in self.rooms:
            raise KeyError(f"Source room '{source_room_id}' does not exist.")

        source_recorder = self.rooms[source_room_id]
        target_recorder = SequenceTensorRecorder(
            room_id=target_room_id,
            default_palette=source_recorder.default_palette
        )
        target_recorder.frames = copy.deepcopy(source_recorder.frames)

        self.rooms[target_room_id] = target_recorder
        self.room_metadata[target_room_id] = {
            "created_at": time.time(),
            "palette": source_recorder.default_palette,
            "replicated_from": source_room_id,
            "frames_copied": len(target_recorder.frames)
        }

        return {
            "success": True,
            "source_room_id": source_room_id,
            "target_room_id": target_room_id,
            "replicated_frames": len(target_recorder.frames)
        }

    def reproduce_room_sequence(
        self,
        room_id: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        output_format: str = "tensor" # "tensor" (RGB) or "ascii"
    ) -> List[Any]:
        """
        Reproduces sequence frames from cached palette for the given room.
        """
        if room_id not in self.rooms:
            raise KeyError(f"Room '{room_id}' does not exist.")

        recorder = self.rooms[room_id]
        palette_name = self.room_metadata[room_id].get("palette", "hex_arena_pbr")
        total = len(recorder.frames)
        end = min(total, end_frame if end_frame is not None else total)

        reproduced = []
        for i in range(start_frame, end):
            frame = recorder.frames[i]
            matrix = frame["matrix"]
            if output_format == "ascii":
                rendered = self.palette_cache.reproduce_ascii_frame(matrix, palette_name)
            else:
                rendered = self.palette_cache.reproduce_tensor_frame(matrix, palette_name)

            reproduced.append({
                "frame_idx": frame["frame_idx"],
                "action_label": frame["action_label"],
                "rendered": rendered,
                "metadata": frame["metadata"]
            })
        return reproduced
