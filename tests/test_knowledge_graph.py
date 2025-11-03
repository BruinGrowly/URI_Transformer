"""
Unit Tests for the Knowledge Graph
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_graph import KnowledgeGraph
from src.phi_geometric_engine import PhiCoordinate

class TestKnowledgeGraph(unittest.TestCase):
    """A unit test suite for the KnowledgeGraph."""

    def setUp(self):
        self.kg = KnowledgeGraph()

    def test_find_closest_principle(self):
        """Tests that the knowledge graph can find the closest principle."""
        # A coordinate that is very close to "Benevolent Love"
        coord1 = PhiCoordinate(1.7, 1.3, 1.1, 1.5)
        principle1 = self.kg.find_closest_principle(coord1)
        self.assertEqual(principle1.name, "Benevolent Love")

        # A coordinate that is very close to "Discerning Wisdom"
        coord2 = PhiCoordinate(1.5, 1.5, 1.3, 1.9)
        principle2 = self.kg.find_closest_principle(coord2)
        self.assertEqual(principle2.name, "Discerning Wisdom")

if __name__ == '__main__':
    unittest.main()
