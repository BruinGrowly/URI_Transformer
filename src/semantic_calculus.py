"""
Semantic Calculus
"""

from src.phi_geometric_engine import PhiCoordinate
import numpy as np

def calculate_trajectory(coord1: PhiCoordinate, coord2: PhiCoordinate) -> tuple:
    """
    Calculates the semantic trajectory between two coordinates.

    Args:
        coord1: The starting coordinate.
        coord2: The ending coordinate.

    Returns:
        A tuple containing the velocity and acceleration.
    """
    velocity = coord2 - coord1
    acceleration = np.linalg.norm(velocity.to_numpy())
    return velocity, acceleration
