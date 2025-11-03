"""
Unit Tests for the Semantic Calculus
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.semantic_calculus import calculate_trajectory
from src.phi_geometric_engine import PhiCoordinate

class TestSemanticCalculus(unittest.TestCase):
    """A unit test suite for the Semantic Calculus."""

    def test_calculate_trajectory(self):
        """Tests the trajectory calculation."""
        coord1 = PhiCoordinate(1, 1, 1, 1)
        coord2 = PhiCoordinate(2, 2, 2, 2)

        velocity, acceleration = calculate_trajectory(coord1, coord2)

        # Velocity should be the element-wise difference
        self.assertAlmostEqual(velocity.love, 1.0)
        self.assertAlmostEqual(velocity.justice, 1.0)
        self.assertAlmostEqual(velocity.power, 1.0)
        self.assertAlmostEqual(velocity.wisdom, 1.0)

        # Acceleration should be the magnitude of the velocity vector
        self.assertAlmostEqual(acceleration, 2.0)

if __name__ == '__main__':
    unittest.main()
