"""
Unit Tests for the QLAE and GOD Frameworks
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.frameworks import QLAEFramework, GODFramework
from src.data_structures import QLAEDomain, ExecutionStrategy
from src.phi_geometric_engine import PhiCoordinate

class TestFrameworks(unittest.TestCase):
    """A unit test suite for the QLAE and GOD frameworks."""

    def setUp(self):
        self.qlae = QLAEFramework()
        self.god = GODFramework()

    def test_qlae_framework_context(self):
        """Tests that the QLAE framework maps coordinates to the correct domain."""
        coord1 = PhiCoordinate(0.5, 0.5, 0.5, 1.5)
        context1 = self.qlae.get_context(coord1)
        self.assertEqual(context1.primary_domain, QLAEDomain.IPE)

        coord2 = PhiCoordinate(1.5, 0.5, 0.5, 1.5)
        context2 = self.qlae.get_context(coord2)
        self.assertEqual(context2.primary_domain, QLAEDomain.ICE)

    def test_god_framework_plan_generation(self):
        """Tests that the GOD framework generates the correct execution plan."""
        plan1 = self.god.generate_plan(0.8, PhiCoordinate(1.5, 0.5, 0.5, 0.5))
        self.assertEqual(plan1.strategy, ExecutionStrategy.COMPASSIONATE_ACTION)

        plan2 = self.god.generate_plan(0.8, PhiCoordinate(0.5, 0.5, 1.5, 0.5))
        self.assertEqual(plan2.strategy, ExecutionStrategy.AUTHORITATIVE_COMMAND)

if __name__ == '__main__':
    unittest.main()
