from dataclasses import dataclass
from typing import Dict, Tuple, Any


Coord3D = Tuple[int, int, int]


@dataclass
class GridStats:
    used_cells: int
    capacity: int


class GridMemory3D:
    def __init__(self, x: int, y: int, z: int) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.capacity = x * y * z
        self._cells: Dict[Coord3D, Any] = {}

    def put(self, coord: Coord3D, value: Any) -> None:
        if not self._is_valid(coord):
            raise ValueError(f"Invalid 3D coordinate: {coord}")
        self._cells[coord] = value

    def get(self, coord: Coord3D) -> Any:
        return self._cells.get(coord)

    def auto_place(self, idx: int, value: Any) -> Coord3D:
        coord = (idx % self.x, (idx // self.x) % self.y, (idx // (self.x * self.y)) % self.z)
        self.put(coord, value)
        return coord

    def stats(self) -> GridStats:
        return GridStats(used_cells=len(self._cells), capacity=self.capacity)

    def _is_valid(self, coord: Coord3D) -> bool:
        cx, cy, cz = coord
        return 0 <= cx < self.x and 0 <= cy < self.y and 0 <= cz < self.z

