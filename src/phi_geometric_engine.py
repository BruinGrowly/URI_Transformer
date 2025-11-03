"""
Phi Geometric Engine
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PhiCoordinate:
    """A 4D coordinate in the semantic space."""
    love: float
    justice: float
    power: float
    wisdom: float

    def to_numpy(self):
        return np.array([self.love, self.justice, self.power, self.wisdom])

    def __sub__(self, other: 'PhiCoordinate') -> 'PhiCoordinate':
        """Subtracts another PhiCoordinate from this one."""
        return PhiCoordinate(
            self.love - other.love,
            self.justice - other.justice,
            self.power - other.power,
            self.wisdom - other.wisdom,
        )


def calculate_harmony_index(anchor_distance: float) -> float:
    """Calculates the harmony index from the anchor distance."""
    return 1 / (1 + anchor_distance)


class GoldenSpiral:
    """Calculates the natural distance between concepts in 4D space."""
    def distance(self, coord1: PhiCoordinate, coord2: PhiCoordinate) -> float:
        """Calculates the Euclidean distance between two 4D points."""
        v1 = coord1.to_numpy()
        v2 = coord2.to_numpy()
        return np.linalg.norm(v1 - v2)
