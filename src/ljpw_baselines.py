"""
LJPW Mathematical Baselines
Version 1.0

Provides objective, non-arbitrary baselines for LJPW framework implementations.
Based on information-theoretic derivations and empirically validated coupling coefficients.

Reference: docs/LJPW-MATHEMATICAL-BASELINES.md
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class NumericalEquivalents:
    """Fundamental constants for LJPW dimensions derived from information theory"""
    L: float = (math.sqrt(5) - 1) / 2  # φ⁻¹ ≈ 0.618034 (Golden ratio inverse)
    J: float = math.sqrt(2) - 1         # √2 - 1 ≈ 0.414214 (Pythagorean ratio)
    P: float = math.e - 2               # e - 2 ≈ 0.718282 (Exponential base)
    W: float = math.log(2)              # ln(2) ≈ 0.693147 (Information unit)


@dataclass
class ReferencePoints:
    """Key reference points in LJPW space"""
    ANCHOR_POINT: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    NATURAL_EQUILIBRIUM: Tuple[float, float, float, float] = (
        0.618034,  # L - Golden ratio inverse
        0.414214,  # J - Pythagorean ratio
        0.718282,  # P - Exponential base
        0.693147   # W - Information unit
    )


class LJPWBaselines:
    """LJPW mathematical baselines and calculations"""

    # Coupling matrix - empirically validated coefficients
    # Love acts as a force multiplier for other dimensions
    COUPLING_MATRIX = {
        'LL': 1.0, 'LJ': 1.4, 'LP': 1.3, 'LW': 1.5,
        'JL': 0.9, 'JJ': 1.0, 'JP': 0.7, 'JW': 1.2,
        'PL': 0.6, 'PJ': 0.8, 'PP': 1.0, 'PW': 0.5,
        'WL': 1.3, 'WJ': 1.1, 'WP': 1.0, 'WW': 1.0,
    }

    @staticmethod
    def effective_dimensions(L: float, J: float, P: float, W: float) -> Dict[str, float]:
        """
        Calculate coupling-adjusted effective dimensions.

        Love acts as a force multiplier:
        - κ_LJ = 1.4: Love amplifies Justice by 40%
        - κ_LP = 1.3: Love amplifies Power by 30%
        - κ_LW = 1.5: Love amplifies Wisdom by 50% (strongest coupling)

        Args:
            L: Love dimension (0.0 to 1.0)
            J: Justice dimension (0.0 to 1.0)
            P: Power dimension (0.0 to 1.0)
            W: Wisdom dimension (0.0 to 1.0)

        Returns:
            Dictionary with effective_L, effective_J, effective_P, effective_W
        """
        return {
            'effective_L': L,  # Love is the source, not amplified
            'effective_J': J * (1 + 1.4 * L),  # Justice amplified by Love
            'effective_P': P * (1 + 1.3 * L),  # Power amplified by Love
            'effective_W': W * (1 + 1.5 * L),  # Wisdom amplified by Love (strongest)
        }

    @staticmethod
    def harmonic_mean(L: float, J: float, P: float, W: float) -> float:
        """
        Harmonic mean - robustness (weakest link metric).

        The system is limited by its weakest dimension.
        Use for: Robustness, fault tolerance, minimum guarantees.

        Interpretation:
        - Score near 0: At least one dimension critically weak
        - Score ≈ 0.5: All dimensions above 0.5 (competent)
        - Score ≈ 0.7: All dimensions strong

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Harmonic mean score
        """
        if L <= 0 or J <= 0 or P <= 0 or W <= 0:
            return 0.0
        return 4.0 / (1/L + 1/J + 1/P + 1/W)

    @staticmethod
    def geometric_mean(L: float, J: float, P: float, W: float) -> float:
        """
        Geometric mean - effectiveness (multiplicative interaction).

        All dimensions needed proportionally.
        Use for: Overall effectiveness, balanced performance.

        Interpretation:
        - Score < 0.5: System struggling in multiple areas
        - Score ≈ 0.6: Functional but not optimal
        - Score ≈ 0.8: High-performing system

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Geometric mean score
        """
        return (L * J * P * W) ** 0.25

    @staticmethod
    def coupling_aware_sum(L: float, J: float, P: float, W: float) -> float:
        """
        Coupling-aware weighted sum - growth potential with Love amplification.

        Use for: Growth potential, scalability, future performance.

        Interpretation:
        - Score < 1.0: Limited growth potential
        - Score ≈ 1.4: Good growth trajectory (coupling active)
        - Score > 1.8: Exceptional growth potential

        Note: This score can exceed 1.0 due to coupling amplification.

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Coupling-aware sum score (can exceed 1.0)
        """
        J_eff = J * (1 + 1.4 * L)
        P_eff = P * (1 + 1.3 * L)
        W_eff = W * (1 + 1.5 * L)
        return 0.35 * L + 0.25 * J_eff + 0.20 * P_eff + 0.20 * W_eff

    @staticmethod
    def harmony_index(L: float, J: float, P: float, W: float) -> float:
        """
        Harmony index - balance (inverse distance from Anchor Point).

        Use for: Balance, alignment, proximity to ideal perfection.

        Interpretation:
        - Score ≈ 0.33: Far from ideal (d ≈ 2.0)
        - Score ≈ 0.50: Moderate alignment (d ≈ 1.0)
        - Score ≈ 0.71: Strong alignment (d ≈ 0.4)

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Harmony index score (0.0 to 1.0)
        """
        d_anchor = math.sqrt((1-L)**2 + (1-J)**2 + (1-P)**2 + (1-W)**2)
        return 1.0 / (1.0 + d_anchor)

    @staticmethod
    def composite_score(L: float, J: float, P: float, W: float) -> float:
        """
        Composite score - overall performance.

        Weighted combination:
        - 35% Growth Potential (coupling-aware)
        - 25% Effectiveness (geometric mean)
        - 25% Robustness (harmonic mean)
        - 15% Harmony (balance)

        Interpretation:
        - Score < 0.8: System needs improvement
        - Score ≈ 1.0: Solid, functional system
        - Score > 1.2: High-performing, growth-oriented system

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Composite score
        """
        baselines = LJPWBaselines
        growth = baselines.coupling_aware_sum(L, J, P, W)
        effectiveness = baselines.geometric_mean(L, J, P, W)
        robustness = baselines.harmonic_mean(L, J, P, W)
        harmony = baselines.harmony_index(L, J, P, W)

        return 0.35 * growth + 0.25 * effectiveness + 0.25 * robustness + 0.15 * harmony

    @staticmethod
    def distance_from_anchor(L: float, J: float, P: float, W: float) -> float:
        """
        Euclidean distance from Anchor Point (1.0, 1.0, 1.0, 1.0).

        The Anchor Point represents perfect, transcendent ideal.

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Euclidean distance from anchor
        """
        return math.sqrt((1-L)**2 + (1-J)**2 + (1-P)**2 + (1-W)**2)

    @staticmethod
    def distance_from_natural_equilibrium(L: float, J: float, P: float, W: float) -> float:
        """
        Euclidean distance from Natural Equilibrium (0.618, 0.414, 0.718, 0.693).

        Natural Equilibrium is the physically achievable optimal balance point.

        Interpretation:
        - d < 0.2: Near-optimal balance
        - 0.2 ≤ d < 0.5: Good but improvable
        - 0.5 ≤ d < 0.8: Moderate imbalance
        - d ≥ 0.8: Significant dysfunction

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Euclidean distance from natural equilibrium
        """
        NE = ReferencePoints.NATURAL_EQUILIBRIUM
        return math.sqrt(
            (NE[0]-L)**2 + (NE[1]-J)**2 + (NE[2]-P)**2 + (NE[3]-W)**2
        )

    @staticmethod
    def full_diagnostic(L: float, J: float, P: float, W: float) -> Dict:
        """
        Complete diagnostic analysis of LJPW coordinates.

        Provides comprehensive analysis including:
        - Raw coordinates
        - Coupling-adjusted effective dimensions
        - Distances from reference points
        - All mixing algorithm metrics

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Dictionary with complete diagnostic information
        """
        baselines = LJPWBaselines
        eff = baselines.effective_dimensions(L, J, P, W)

        return {
            'coordinates': {'L': L, 'J': J, 'P': P, 'W': W},
            'effective_dimensions': eff,
            'distances': {
                'from_anchor': baselines.distance_from_anchor(L, J, P, W),
                'from_natural_equilibrium': baselines.distance_from_natural_equilibrium(L, J, P, W),
            },
            'metrics': {
                'harmonic_mean': baselines.harmonic_mean(L, J, P, W),
                'geometric_mean': baselines.geometric_mean(L, J, P, W),
                'coupling_aware_sum': baselines.coupling_aware_sum(L, J, P, W),
                'harmony_index': baselines.harmony_index(L, J, P, W),
                'composite_score': baselines.composite_score(L, J, P, W),
            }
        }

    @staticmethod
    def suggest_improvements(L: float, J: float, P: float, W: float) -> Dict:
        """
        Suggest which dimension to improve based on distance from Natural Equilibrium.

        Returns prioritized list of dimensions to focus on.

        Args:
            L, J, P, W: LJPW dimensions (0.0 to 1.0)

        Returns:
            Dictionary with improvement suggestions
        """
        NE = ReferencePoints.NATURAL_EQUILIBRIUM

        distances = {
            'L': abs(L - NE[0]),
            'J': abs(J - NE[1]),
            'P': abs(P - NE[2]),
            'W': abs(W - NE[3])
        }

        # Sort by distance (largest first)
        priorities = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        dimension_names = {
            'L': 'Love',
            'J': 'Justice',
            'P': 'Power',
            'W': 'Wisdom'
        }

        current_values = {'L': L, 'J': J, 'P': P, 'W': W}

        return {
            'primary_focus': dimension_names[priorities[0][0]],
            'primary_focus_key': priorities[0][0],
            'distance_from_optimal': priorities[0][1],
            'current_value': current_values[priorities[0][0]],
            'target_value': NE[['L', 'J', 'P', 'W'].index(priorities[0][0])],
            'all_priorities': [
                {
                    'dimension': dimension_names[k],
                    'key': k,
                    'distance': v,
                    'current': current_values[k],
                    'target': NE[['L', 'J', 'P', 'W'].index(k)]
                }
                for k, v in priorities
            ]
        }


def get_numerical_equivalents() -> NumericalEquivalents:
    """Get the fundamental numerical equivalents for LJPW dimensions."""
    return NumericalEquivalents()


def get_reference_points() -> ReferencePoints:
    """Get the key reference points (Anchor and Natural Equilibrium)."""
    return ReferencePoints()


# Example usage and validation
if __name__ == '__main__':
    # Example: Software team analysis
    L, J, P, W = 0.792, 0.843, 0.940, 0.724

    print("LJPW Mathematical Baselines - Diagnostic Report")
    print("=" * 70)
    print(f"Coordinates: L={L:.3f}, J={J:.3f}, P={P:.3f}, W={W:.3f}")
    print()

    baselines = LJPWBaselines()
    diagnostic = baselines.full_diagnostic(L, J, P, W)

    print("Effective Dimensions (coupling-adjusted):")
    for dim, val in diagnostic['effective_dimensions'].items():
        print(f"  {dim}: {val:.3f}")
    print()

    print("Distances:")
    print(f"  From Anchor Point (1,1,1,1): {diagnostic['distances']['from_anchor']:.3f}")
    print(f"  From Natural Equilibrium: {diagnostic['distances']['from_natural_equilibrium']:.3f}")
    print()

    print("Performance Metrics:")
    for metric, val in diagnostic['metrics'].items():
        print(f"  {metric}: {val:.3f}")
    print()

    improvements = baselines.suggest_improvements(L, J, P, W)
    print("Improvement Suggestions:")
    print(f"  Primary Focus: {improvements['primary_focus']}")
    print(f"  Current Value: {improvements['current_value']:.3f}")
    print(f"  Target Value: {improvements['target_value']:.3f}")
    print(f"  Distance from Optimal: {improvements['distance_from_optimal']:.3f}")
    print()

    print("Reference Points:")
    ne = get_numerical_equivalents()
    print(f"  Love (φ⁻¹): {ne.L:.6f}")
    print(f"  Justice (√2-1): {ne.J:.6f}")
    print(f"  Power (e-2): {ne.P:.6f}")
    print(f"  Wisdom (ln2): {ne.W:.6f}")
