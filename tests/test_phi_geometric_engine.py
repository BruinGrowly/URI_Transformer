"""
Unit Tests for the Phi Geometric Engine
"""

import unittest
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phi_geometric_engine import (
    PhiCoordinate,
    calculate_harmony_index,
    GoldenSpiral
)

class TestPhiGeometricEngine(unittest.TestCase):
    """A unit test suite for the core mathematical components."""

    def test_phi_coordinate_subtraction(self):
        """Tests the subtraction of two PhiCoordinate objects."""
        coord1 = PhiCoordinate(0.8, 0.7, 0.6, 0.5)
        coord2 = PhiCoordinate(0.1, 0.2, 0.3, 0.4)
        result = coord1 - coord2
        self.assertAlmostEqual(result.love, 0.7)
        self.assertAlmostEqual(result.justice, 0.5)
        self.assertAlmostEqual(result.power, 0.3)
        self.assertAlmostEqual(result.wisdom, 0.1)

    def test_calculate_harmony_index(self):
        """Tests the harmony index calculation."""
        self.assertAlmostEqual(calculate_harmony_index(0.0), 1.0)
        self.assertAlmostEqual(calculate_harmony_index(1.0), 0.5)
        self.assertLess(calculate_harmony_index(100.0), 0.01)

    def test_golden_spiral_distance(self):
        """Tests the Euclidean distance calculation in 4D."""
        coord1 = PhiCoordinate(1, 0, 0, 0)
        coord2 = PhiCoordinate(0, 1, 0, 0)
        distance = GoldenSpiral().distance(coord1, coord2)
        self.assertAlmostEqual(distance, np.sqrt(2))

        coord3 = PhiCoordinate(1, 1, 1, 1)
        coord4 = PhiCoordinate(2, 2, 2, 2)
        distance2 = GoldenSpiral().distance(coord3, coord4)
        self.assertAlmostEqual(distance2, 2.0)

if __name__ == '__main__':
    unittest.main()
