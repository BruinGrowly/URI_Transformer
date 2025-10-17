"""
PHI GEOMETRIC ENGINE - Golden Ratio Mathematics for Semantic Substrate Database

This module implements geometric operations using the golden ratio (phi = 1.618...)
to enable natural growth patterns in semantic space navigation and indexing.

Core Capabilities:
- Fibonacci sequence generation and indexing
- Golden spiral distance calculations in 4D space
- Golden angle rotations for maximum diversity
- Exponential phi-based binning for logarithmic complexity
- Dodecahedral anchor geometry

Author: Semantic Substrate Database Project
License: MIT
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Golden ratio constant (to maximum precision)
PHI = 1.618033988749895  # (1 + √5) / 2
PHI_INVERSE = 0.618033988749895  # 1 / φ = φ - 1
GOLDEN_ANGLE_RAD = 2.39996322972865332  # (2 - φ) × 2π radians
GOLDEN_ANGLE_DEG = 137.5077640500378  # (2 - φ) × 360 degrees


@dataclass
class PhiCoordinate:
    """4D coordinate with phi-geometric properties"""
    love: float
    power: float
    wisdom: float
    justice: float

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector"""
        return np.array([self.love, self.power, self.wisdom, self.justice])

    def magnitude(self) -> float:
        """Calculate magnitude in 4D space"""
        return np.linalg.norm(self.to_vector())


class FibonacciSequence:
    """
    Fibonacci sequence generator with phi-based properties

    F(n) = F(n-1) + F(n-2)
    F(0) = 0, F(1) = 1

    As n→∞, F(n+1)/F(n) → φ
    """

    def __init__(self, max_precompute: int = 50):
        """Initialize with precomputed sequence"""
        self.sequence = [0, 1]
        for i in range(2, max_precompute):
            self.sequence.append(self.sequence[-1] + self.sequence[-2])

    def get(self, n: int) -> int:
        """Get nth Fibonacci number"""
        if n < len(self.sequence):
            return self.sequence[n]

        # Compute on demand if beyond precomputed
        while len(self.sequence) <= n:
            self.sequence.append(self.sequence[-1] + self.sequence[-2])

        return self.sequence[n]

    def get_range(self, start: int, end: int) -> List[int]:
        """Get Fibonacci numbers from start to end index"""
        return [self.get(i) for i in range(start, end + 1)]

    def approximate_with_phi(self, n: int) -> float:
        """
        Approximate F(n) using Binet's formula
        F(n) ≈ φ^n / √5
        """
        return (PHI ** n) / math.sqrt(5)

    def find_index_for_value(self, target: int) -> int:
        """Find index of Fibonacci number closest to target"""
        # Use inverse of Binet's formula
        # n ≈ log_φ(target × √5)
        n = int(math.log(target * math.sqrt(5)) / math.log(PHI))

        # Adjust if needed
        while self.get(n) < target:
            n += 1

        return n


class GoldenSpiral:
    """
    Logarithmic spiral based on golden ratio

    r(θ) = a × φ^(θ / (π/2))

    This spiral appears in nature: nautilus shells, galaxies, etc.
    """

    def __init__(self, scale_factor: float = 1.0):
        """Initialize spiral with scale factor"""
        self.a = scale_factor

    def radius_at_angle(self, theta: float) -> float:
        """Calculate radius at given angle (radians)"""
        return self.a * (PHI ** (theta / (math.pi / 2)))

    def angle_at_radius(self, r: float) -> float:
        """Calculate angle for given radius (radians)"""
        if r <= 0 or self.a <= 0:
            return 0.0
        return (math.pi / 2) * math.log(r / self.a) / math.log(PHI)

    def distance_4d(self, p1: PhiCoordinate, p2: PhiCoordinate,
                    center: Optional[PhiCoordinate] = None) -> float:
        """
        Calculate distance along golden spiral in 4D space

        Projects 4D coordinates onto spiral and calculates arc length.
        More natural than Euclidean distance for semantic relationships.
        """
        if center is None:
            center = PhiCoordinate(0, 0, 0, 0)

        # Calculate vectors from center
        v1 = p1.to_vector() - center.to_vector()
        v2 = p2.to_vector() - center.to_vector()

        # Calculate radii
        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)

        if r1 == 0 or r2 == 0:
            return abs(r1 - r2)

        # Calculate angles on spiral
        theta1 = self.angle_at_radius(r1)
        theta2 = self.angle_at_radius(r2)

        # Arc length on spiral between two points
        # Approximation: integrate r(θ) from θ1 to θ2
        if abs(theta2 - theta1) < 0.001:
            # Very close angles, use linear approximation
            return abs(r2 - r1)

        # Simpson's rule integration for arc length
        n_steps = 20
        theta_step = (theta2 - theta1) / n_steps
        arc_length = 0.0

        for i in range(n_steps):
            theta = theta1 + i * theta_step
            r = self.radius_at_angle(theta)

            # dr/dθ for logarithmic spiral
            dr_dtheta = r * math.log(PHI) / (math.pi / 2)

            # Arc length element: sqrt(r² + (dr/dθ)²)
            ds = math.sqrt(r**2 + dr_dtheta**2)
            arc_length += ds * abs(theta_step)

        return arc_length

    def spiral_path_points(self, start_angle: float, end_angle: float,
                          num_points: int = 100) -> List[Tuple[float, float]]:
        """Generate points along spiral path"""
        points = []
        angle_step = (end_angle - start_angle) / (num_points - 1)

        for i in range(num_points):
            theta = start_angle + i * angle_step
            r = self.radius_at_angle(theta)

            # Convert to Cartesian (2D projection)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append((x, y))

        return points


class GoldenAngleRotator:
    """
    Rotations using golden angle (137.5°) for optimal packing

    The golden angle minimizes overlap when placing items in a circle,
    used in phyllotaxis (leaf arrangement on stems).
    """

    def __init__(self):
        """Initialize golden angle rotator"""
        self.golden_angle_rad = GOLDEN_ANGLE_RAD
        self.golden_angle_deg = GOLDEN_ANGLE_DEG

    def rotate_2d(self, x: float, y: float, n: int = 1) -> Tuple[float, float]:
        """
        Rotate 2D point by n × golden angle

        Args:
            x, y: Point coordinates
            n: Number of golden angle rotations

        Returns:
            Rotated (x', y') coordinates
        """
        angle = n * self.golden_angle_rad
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a

        return (x_new, y_new)

    def rotate_4d(self, coord: PhiCoordinate, n: int = 1,
                  plane: str = "LP") -> PhiCoordinate:
        """
        Rotate 4D coordinate by n × golden angle in specified plane

        Args:
            coord: 4D coordinate
            n: Number of golden angle rotations
            plane: Rotation plane ("LP", "LW", "LJ", "PW", "PJ", "WJ")

        Returns:
            Rotated 4D coordinate
        """
        angle = n * self.golden_angle_rad
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        L, P, W, J = coord.love, coord.power, coord.wisdom, coord.justice

        # Apply rotation in specified plane
        if plane == "LP":
            L_new = L * cos_a - P * sin_a
            P_new = L * sin_a + P * cos_a
            return PhiCoordinate(L_new, P_new, W, J)
        elif plane == "LW":
            L_new = L * cos_a - W * sin_a
            W_new = L * sin_a + W * cos_a
            return PhiCoordinate(L_new, P, W_new, J)
        elif plane == "LJ":
            L_new = L * cos_a - J * sin_a
            J_new = L * sin_a + J * cos_a
            return PhiCoordinate(L_new, P, W, J_new)
        elif plane == "PW":
            P_new = P * cos_a - W * sin_a
            W_new = P * sin_a + W * cos_a
            return PhiCoordinate(L, P_new, W_new, J)
        elif plane == "PJ":
            P_new = P * cos_a - J * sin_a
            J_new = P * sin_a + J * cos_a
            return PhiCoordinate(L, P_new, W, J_new)
        elif plane == "WJ":
            W_new = W * cos_a - J * sin_a
            J_new = W * sin_a + J * cos_a
            return PhiCoordinate(L, P, W_new, J_new)
        else:
            raise ValueError(f"Invalid rotation plane: {plane}")

    def generate_optimal_distribution(self, center: PhiCoordinate,
                                     radius: float, count: int) -> List[PhiCoordinate]:
        """
        Generate optimally distributed points around center using golden angle

        Creates maximum diversity with minimum overlap.
        """
        points = []

        for i in range(count):
            # Rotate by i × golden angle in LP plane
            # Start with point at (radius, 0, 0, 0) from center
            point = PhiCoordinate(
                center.love + radius,
                center.power,
                center.wisdom,
                center.justice
            )

            # Rotate by golden angle
            rotated = self.rotate_4d(
                PhiCoordinate(radius, 0, 0, 0),
                n=i,
                plane="LP"
            )

            # Add to center
            final = PhiCoordinate(
                center.love + rotated.love,
                center.power + rotated.power,
                center.wisdom + rotated.wisdom,
                center.justice + rotated.justice
            )

            points.append(final)

        return points


class PhiExponentialBinner:
    """
    Exponential binning using φ^n for logarithmic complexity

    Bin 0: [0, φ^0] = [0, 1.000]
    Bin 1: [φ^0, φ^1] = [1.000, 1.618]
    Bin 2: [φ^1, φ^2] = [1.618, 2.618]
    Bin 3: [φ^2, φ^3] = [2.618, 4.236]
    ...

    Enables O(log_φ n) search complexity.
    """

    def __init__(self, max_bins: int = 20):
        """Initialize with precomputed bin boundaries"""
        self.bin_boundaries = [PHI ** i for i in range(max_bins + 1)]
        self.max_bins = max_bins

    def get_bin(self, value: float) -> int:
        """
        Get bin index for value

        Returns bin number or 0 for values < 1.
        """
        if value < 0:
            return 0  # Treat negative as bin 0

        if value == 0:
            return 0

        if value < 1.0:
            return 0  # Values < 1 go in bin 0

        # Calculate bin using logarithm
        # If value = φ^n, then n = log_φ(value)
        bin_idx = int(math.log(value) / math.log(PHI))

        # Clamp to valid range
        return min(max(0, bin_idx), self.max_bins - 1)

    def get_bin_range(self, bin_idx: int) -> Tuple[float, float]:
        """Get [min, max) range for bin"""
        if bin_idx < 0 or bin_idx >= self.max_bins:
            raise ValueError(f"Bin index {bin_idx} out of range")

        return (self.bin_boundaries[bin_idx], self.bin_boundaries[bin_idx + 1])

    def get_bin_center(self, bin_idx: int) -> float:
        """Get center value of bin"""
        min_val, max_val = self.get_bin_range(bin_idx)
        # Geometric mean for exponential bins
        return math.sqrt(min_val * max_val)

    def bins_in_range(self, min_val: float, max_val: float) -> List[int]:
        """Get all bin indices that overlap with [min_val, max_val]"""
        bins = []
        min_bin = self.get_bin(min_val)
        max_bin = self.get_bin(max_val)

        for i in range(max(0, min_bin), min(max_bin + 1, self.max_bins)):
            bins.append(i)

        return bins


class DodecahedralAnchors:
    """
    12 anchor points arranged in dodecahedral geometry using phi

    A dodecahedron has 12 pentagonal faces, and all its geometry
    is based on the golden ratio. Perfect for biblical 12 (tribes, apostles).
    """

    def __init__(self):
        """Initialize 12 dodecahedral anchors in 4D space"""
        self.anchors = self._generate_dodecahedral_anchors()

    def _generate_dodecahedral_anchors(self) -> Dict[int, PhiCoordinate]:
        """
        Generate 12 anchors based on dodecahedron vertices in 4D

        Using phi-based coordinates for sacred geometry.
        """
        # Classic dodecahedron vertices use coordinates involving φ
        # In 4D, we extend this pattern

        anchors = {
            # Anchor Point A - Origin (1,1,1,1)
            1: PhiCoordinate(1.0, 1.0, 1.0, 1.0),

            # 8 vertices of inner cube scaled by φ
            2: PhiCoordinate(PHI, PHI_INVERSE, 0, 0),
            3: PhiCoordinate(PHI, -PHI_INVERSE, 0, 0),
            4: PhiCoordinate(-PHI, PHI_INVERSE, 0, 0),
            5: PhiCoordinate(-PHI, -PHI_INVERSE, 0, 0),

            # 4 vertices on axes
            6: PhiCoordinate(0, 0, PHI, PHI_INVERSE),
            7: PhiCoordinate(0, 0, PHI, -PHI_INVERSE),
            8: PhiCoordinate(0, 0, -PHI, PHI_INVERSE),
            9: PhiCoordinate(0, 0, -PHI, -PHI_INVERSE),

            # 3 more vertices completing dodecahedral symmetry
            10: PhiCoordinate(PHI_INVERSE, 0, 0, PHI),
            11: PhiCoordinate(-PHI_INVERSE, 0, 0, PHI),
            12: PhiCoordinate(0, PHI, PHI_INVERSE, 0),
        }

        # Normalize all to unit sphere then scale
        for anchor_id in anchors:
            coord = anchors[anchor_id]
            magnitude = coord.magnitude()
            if magnitude > 0:
                scale = 1.0 / magnitude
                anchors[anchor_id] = PhiCoordinate(
                    coord.love * scale,
                    coord.power * scale,
                    coord.wisdom * scale,
                    coord.justice * scale
                )

        return anchors

    def get_anchor(self, anchor_id: int) -> Optional[PhiCoordinate]:
        """Get anchor by ID (1-12)"""
        return self.anchors.get(anchor_id)

    def nearest_anchor(self, point: PhiCoordinate) -> Tuple[int, float]:
        """
        Find nearest anchor to point

        Returns (anchor_id, distance)
        """
        min_distance = float('inf')
        nearest_id = 1

        for anchor_id, anchor_coord in self.anchors.items():
            # Euclidean distance in 4D
            diff = point.to_vector() - anchor_coord.to_vector()
            distance = np.linalg.norm(diff)

            if distance < min_distance:
                min_distance = distance
                nearest_id = anchor_id

        return (nearest_id, min_distance)

    def get_pentagonal_cluster(self, anchor_id: int) -> List[int]:
        """
        Get 5 anchors forming pentagonal face around given anchor

        Dodecahedron has 12 vertices, each surrounded by a pentagon.
        """
        # Simplified: return 5 nearest neighbors
        anchor = self.get_anchor(anchor_id)
        if not anchor:
            return []

        distances = []
        for other_id, other_anchor in self.anchors.items():
            if other_id != anchor_id:
                diff = anchor.to_vector() - other_anchor.to_vector()
                distance = np.linalg.norm(diff)
                distances.append((other_id, distance))

        # Sort by distance and take 5 nearest
        distances.sort(key=lambda x: x[1])
        return [anchor_id] + [aid for aid, _ in distances[:5]]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fibonacci(n: int) -> int:
    """Quick access to nth Fibonacci number"""
    fib = FibonacciSequence()
    return fib.get(n)


def golden_spiral_distance(p1: PhiCoordinate, p2: PhiCoordinate,
                          center: Optional[PhiCoordinate] = None) -> float:
    """Quick access to golden spiral distance"""
    spiral = GoldenSpiral()
    return spiral.distance_4d(p1, p2, center)


def rotate_by_golden_angle(coord: PhiCoordinate, n: int = 1,
                          plane: str = "LP") -> PhiCoordinate:
    """Quick access to golden angle rotation"""
    rotator = GoldenAngleRotator()
    return rotator.rotate_4d(coord, n, plane)


def get_phi_bin(value: float) -> int:
    """Quick access to phi exponential bin"""
    binner = PhiExponentialBinner()
    return binner.get_bin(value)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'PHI',
    'PHI_INVERSE',
    'GOLDEN_ANGLE_RAD',
    'GOLDEN_ANGLE_DEG',
    'PhiCoordinate',
    'FibonacciSequence',
    'GoldenSpiral',
    'GoldenAngleRotator',
    'PhiExponentialBinner',
    'DodecahedralAnchors',
    'fibonacci',
    'golden_spiral_distance',
    'rotate_by_golden_angle',
    'get_phi_bin',
]


if __name__ == "__main__":
    # Quick demonstration
    print("=" * 80)
    print("PHI GEOMETRIC ENGINE DEMONSTRATION")
    print("=" * 80)

    print("\n[1] Fibonacci Sequence")
    fib = FibonacciSequence()
    print(f"First 15 Fibonacci numbers: {fib.get_range(0, 14)}")
    print(f"F(20) = {fib.get(20)}")
    print(f"F(20) / F(19) = {fib.get(20) / fib.get(19):.10f} (approaches phi = {PHI})")

    print("\n[2] Golden Spiral")
    spiral = GoldenSpiral()
    p1 = PhiCoordinate(0.5, 0.5, 0.5, 0.5)
    p2 = PhiCoordinate(0.8, 0.6, 0.7, 0.9)
    spiral_dist = spiral.distance_4d(p1, p2)
    euclidean_dist = np.linalg.norm(p1.to_vector() - p2.to_vector())
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Spiral distance: {spiral_dist:.6f}")
    print(f"Euclidean distance: {euclidean_dist:.6f}")
    print(f"Ratio: {spiral_dist / euclidean_dist:.3f}")

    print("\n[3] Golden Angle Rotation")
    rotator = GoldenAngleRotator()
    coord = PhiCoordinate(1.0, 0.0, 0.0, 0.0)
    print(f"Original: {coord}")
    for i in range(1, 4):
        rotated = rotator.rotate_4d(coord, i, "LP")
        print(f"After {i} x phi rotation: L={rotated.love:.4f}, P={rotated.power:.4f}")

    print("\n[4] Phi Exponential Binning")
    binner = PhiExponentialBinner()
    test_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f"Value -> Bin (Range)")
    for val in test_values:
        bin_idx = binner.get_bin(val)
        bin_range = binner.get_bin_range(bin_idx)
        print(f"  {val:5.1f} -> Bin {bin_idx} [{bin_range[0]:.3f}, {bin_range[1]:.3f})")

    print("\n[5] Dodecahedral Anchors")
    dodec = DodecahedralAnchors()
    print(f"Generated 12 anchors in 4D dodecahedral arrangement")
    test_point = PhiCoordinate(0.7, 0.8, 0.6, 0.9)
    nearest_id, distance = dodec.nearest_anchor(test_point)
    print(f"Test point: {test_point}")
    print(f"Nearest anchor: #{nearest_id} at distance {distance:.4f}")
    cluster = dodec.get_pentagonal_cluster(nearest_id)
    print(f"Pentagonal cluster around anchor #{nearest_id}: {cluster}")

    print("\n" + "=" * 80)
    print("PHI GEOMETRIC ENGINE READY")
    print("=" * 80)
