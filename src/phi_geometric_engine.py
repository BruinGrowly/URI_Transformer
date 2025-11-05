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


def universal_semantic_mix(primary_weights: dict) -> PhiCoordinate:
    """
    Mix semantic primaries like colors: weighted average
    Primary colors are the fundamental building blocks of all meaning
    """
    total = sum(primary_weights.values())
    if total == 0:
        # Return a neutral PhiCoordinate if no weights are provided
        return PhiCoordinate(love=0.25, justice=0.25, power=0.25, wisdom=0.25)
    
    love = primary_weights.get('love', 0) / total
    justice = primary_weights.get('justice', 0) / total
    power = primary_weights.get('power', 0) / total
    wisdom = primary_weights.get('wisdom', 0) / total

    return PhiCoordinate(love=love, justice=justice, power=power, wisdom=wisdom)


def quadratic_semantic_mix(primary_weights: dict) -> PhiCoordinate:
    """
    Emphasizes dominant primaries, creating more vivid meanings.
    """
    weights_squared = {k: v**2 for k, v in primary_weights.items()}
    total = sum(weights_squared.values())
    if total == 0:
        return PhiCoordinate(love=0.25, justice=0.25, power=0.25, wisdom=0.25)
    
    love = weights_squared.get('love', 0) / total
    justice = weights_squared.get('justice', 0) / total
    power = weights_squared.get('power', 0) / total
    wisdom = weights_squared.get('wisdom', 0) / total
    
    return PhiCoordinate(love=love, justice=justice, power=power, wisdom=wisdom)


def golden_semantic_mix(primary_weights: dict) -> PhiCoordinate:
    """
    Creates meaning with divine mathematical proportions using the Golden Ratio.
    """
    phi = 1.618  # Golden ratio
    weights_phi = {k: v**phi for k, v in primary_weights.items()}
    total = sum(weights_phi.values())
    if total == 0:
        return PhiCoordinate(love=0.25, justice=0.25, power=0.25, wisdom=0.25)
    
    love = weights_phi.get('love', 0) / total
    justice = weights_phi.get('justice', 0) / total
    power = weights_phi.get('power', 0) / total
    wisdom = weights_phi.get('wisdom', 0) / total
    
    return PhiCoordinate(love=love, justice=justice, power=power, wisdom=wisdom)
