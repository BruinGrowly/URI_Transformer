"""
Phi Geometric Engine
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


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

    # LJPW Mathematical Baselines Integration

    def get_effective_dimensions(self) -> Dict[str, float]:
        """
        Calculate coupling-adjusted effective dimensions.
        Love acts as a force multiplier for other dimensions.

        Returns:
            Dictionary with effective_L, effective_J, effective_P, effective_W
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.effective_dimensions(
            self.love, self.justice, self.power, self.wisdom
        )

    def harmonic_mean(self) -> float:
        """
        Calculate harmonic mean (robustness - weakest link metric).

        Returns:
            Harmonic mean score
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.harmonic_mean(
            self.love, self.justice, self.power, self.wisdom
        )

    def geometric_mean(self) -> float:
        """
        Calculate geometric mean (effectiveness - multiplicative interaction).

        Returns:
            Geometric mean score
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.geometric_mean(
            self.love, self.justice, self.power, self.wisdom
        )

    def coupling_aware_sum(self) -> float:
        """
        Calculate coupling-aware sum (growth potential with Love amplification).
        Note: Can exceed 1.0 due to coupling effects.

        Returns:
            Coupling-aware sum score
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.coupling_aware_sum(
            self.love, self.justice, self.power, self.wisdom
        )

    def harmony_index(self) -> float:
        """
        Calculate harmony index (balance - inverse distance from Anchor Point).

        Returns:
            Harmony index score (0.0 to 1.0)
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.harmony_index(
            self.love, self.justice, self.power, self.wisdom
        )

    def composite_score(self) -> float:
        """
        Calculate composite score (overall performance).
        Weighted combination of growth, effectiveness, robustness, and harmony.

        Returns:
            Composite score
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.composite_score(
            self.love, self.justice, self.power, self.wisdom
        )

    def distance_from_anchor(self) -> float:
        """
        Calculate Euclidean distance from Anchor Point (1,1,1,1).

        Returns:
            Distance from anchor
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.distance_from_anchor(
            self.love, self.justice, self.power, self.wisdom
        )

    def distance_from_natural_equilibrium(self) -> float:
        """
        Calculate Euclidean distance from Natural Equilibrium (0.618, 0.414, 0.718, 0.693).

        Interpretation:
        - d < 0.2: Near-optimal balance
        - 0.2 ≤ d < 0.5: Good but improvable
        - 0.5 ≤ d < 0.8: Moderate imbalance
        - d ≥ 0.8: Significant dysfunction

        Returns:
            Distance from natural equilibrium
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.distance_from_natural_equilibrium(
            self.love, self.justice, self.power, self.wisdom
        )

    def full_diagnostic(self) -> Dict:
        """
        Get complete diagnostic analysis including all baseline metrics.

        Returns:
            Dictionary with comprehensive diagnostic information
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.full_diagnostic(
            self.love, self.justice, self.power, self.wisdom
        )

    def suggest_improvements(self) -> Dict:
        """
        Get improvement suggestions based on distance from Natural Equilibrium.

        Returns:
            Dictionary with prioritized improvement suggestions
        """
        from src.ljpw_baselines import LJPWBaselines
        return LJPWBaselines.suggest_improvements(
            self.love, self.justice, self.power, self.wisdom
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
