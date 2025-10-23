"""
Phi Geometric Engine
====================

Provides the core mathematical components for the TruthSense Transformer,
based on golden ratio (phi) mathematics.
"""

import numpy as np
from dataclasses import dataclass

PHI = 1.61803398875
PHI_INVERSE = 1 / PHI
GOLDEN_ANGLE_RAD = np.pi * (3 - np.sqrt(5))


@dataclass
class PhiCoordinate:
    """A 4D coordinate in the semantic space."""
    love: float
    justice: float
    power: float
    wisdom: float

    def to_numpy(self):
        return np.array([self.love, self.justice, self.power, self.wisdom])


class FibonacciSequence:
    """Generates Fibonacci numbers for relationship expansion."""
    def __init__(self, cache_size=100):
        self._cache = {0: 0, 1: 1}
        if cache_size > 1:
            for i in range(2, cache_size + 1):
                self._cache[i] = self._cache[i - 1] + self._cache[i - 2]

    def get(self, n):
        if n in self._cache:
            return self._cache[n]
        sqrt5 = np.sqrt(5)
        return round((PHI**n - (1 - PHI)**n) / sqrt5)


class GoldenSpiral:
    """Calculates the natural distance between concepts in 4D space."""
    def __init__(self, a=1.0, b=0.30635):
        self.a = a
        self.b = b

    def distance(self, coord1: PhiCoordinate, coord2: PhiCoordinate) -> float:
        """Calculates the golden spiral arc length between two 4D points."""
        v1 = coord1.to_numpy()
        v2 = coord2.to_numpy()
        euclidean_dist = np.linalg.norm(v1 - v2)

        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_product == 0:
            return 0.0

        cos_theta = dot_product / norm_product
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return euclidean_dist * (PHI**(theta / (np.pi / 2)))


class GoldenAngleRotator:
    """Uses the golden angle for optimal distribution of semantic concepts."""
    def rotate(self, coord: PhiCoordinate, n_rotations: int) -> PhiCoordinate:
        """Rotates a 4D coordinate by the golden angle n times."""
        angle = n_rotations * GOLDEN_ANGLE_RAD
        love, justice, power, wisdom = (
            coord.love, coord.justice, coord.power, coord.wisdom
        )  # noqa
        love_new = love * np.cos(angle) - justice * np.sin(angle)
        justice_new = love * np.sin(angle) + justice * np.cos(angle)
        return PhiCoordinate(love_new, justice_new, power, wisdom)


class PhiExponentialBinner:
    """Provides efficient semantic indexing using exponential phi-based bins."""
    def get_bin(self, magnitude: float) -> int:
        """Determines the phi-based bin for a given magnitude."""
        if magnitude <= 0:
            return 0
        return int(np.floor(np.log(magnitude) / np.log(PHI)))


class DodecahedralAnchors:
    """A network of 12 dodecahedral anchors for geometric navigation."""
    def __init__(self):
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        """Generates 11 other anchors in dodecahedral symmetry."""
        anchors = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    love = ((-1)**i) * PHI_INVERSE
                    power = ((-1)**j) * PHI_INVERSE
                    wisdom = ((-1)**k) * PHI_INVERSE
                    anchors.append(
                        PhiCoordinate(love, PHI, power, wisdom)
                    )
        return anchors

    def get_nearest_anchor(self, coord: PhiCoordinate) -> PhiCoordinate:
        """Finds the nearest dodecahedral anchor to a given coordinate."""
        min_dist = float('inf')
        nearest_anchor = None
        for anchor in self.anchors:
            dist = GoldenSpiral().distance(coord, anchor)
            if dist < min_dist:
                min_dist = dist
                nearest_anchor = anchor
        return nearest_anchor
