"""
Unit Tests for Validation and Error Handling
=============================================

Tests for PhiCoordinate validation, model loading error handling,
and other critical validation logic.
"""

import pytest
import os
import tempfile
import torch
from src.phi_geometric_engine import PhiCoordinate, CoordinateValidationError
from src.semantic_frontend import (
    SemanticFrontEnd,
    ModelNotFoundError,
    ModelLoadError,
    ProjectionHead
)


class TestPhiCoordinateValidation:
    """Tests for PhiCoordinate input validation."""

    def test_valid_coordinates(self):
        """Test that valid coordinates are accepted."""
        # Test with values in [0, 1]
        coord = PhiCoordinate(love=0.5, justice=0.5, power=0.5, wisdom=0.5)
        assert coord.love == 0.5
        assert coord.justice == 0.5
        assert coord.power == 0.5
        assert coord.wisdom == 0.5

        # Test boundary values
        coord_min = PhiCoordinate(love=0.0, justice=0.0, power=0.0, wisdom=0.0)
        assert coord_min.love == 0.0

        coord_max = PhiCoordinate(love=1.0, justice=1.0, power=1.0, wisdom=1.0)
        assert coord_max.love == 1.0

    def test_invalid_coordinates_below_range(self):
        """Test that values below 0 raise validation error."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            PhiCoordinate(love=-0.1, justice=0.5, power=0.5, wisdom=0.5)

        assert "love=-0.1" in str(exc_info.value)
        assert "must be in range [0, 1]" in str(exc_info.value)

    def test_invalid_coordinates_above_range(self):
        """Test that values above 1 raise validation error."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            PhiCoordinate(love=0.5, justice=1.5, power=0.5, wisdom=0.5)

        assert "justice=1.5" in str(exc_info.value)
        assert "must be in range [0, 1]" in str(exc_info.value)

    def test_multiple_invalid_coordinates(self):
        """Test error message with multiple invalid coordinates."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            PhiCoordinate(love=-0.1, justice=0.5, power=1.5, wisdom=0.5)

        error_msg = str(exc_info.value)
        assert "love=" in error_msg
        assert "power=" in error_msg

    def test_non_numeric_coordinates(self):
        """Test that non-numeric values raise validation error."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            PhiCoordinate(love="invalid", justice=0.5, power=0.5, wisdom=0.5)

        assert "must be numeric" in str(exc_info.value)

    def test_to_numpy_conversion(self):
        """Test conversion to numpy array."""
        coord = PhiCoordinate(love=0.1, justice=0.2, power=0.3, wisdom=0.4)
        np_array = coord.to_numpy()

        assert np_array.shape == (4,)
        assert np_array[0] == 0.1
        assert np_array[1] == 0.2
        assert np_array[2] == 0.3
        assert np_array[3] == 0.4


class TestSemanticFrontEndErrorHandling:
    """Tests for SemanticFrontEnd error handling."""

    def test_missing_projection_head_file(self):
        """Test that missing model file raises appropriate error."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            SemanticFrontEnd(projection_head_path="nonexistent_model.pth")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "nonexistent_model.pth" in error_msg
        assert "train_semantic_frontend.py" in error_msg

    def test_initialization_without_projection_head(self):
        """Test that initialization works without projection head path."""
        frontend = SemanticFrontEnd(projection_head_path=None)
        assert frontend is not None
        assert frontend.projection_head is not None
        assert frontend.tokenizer is not None
        assert frontend.language_model is not None

    def test_corrupt_model_file(self):
        """Test that corrupt model file raises appropriate error."""
        # Create a temporary corrupt model file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pth', delete=False) as f:
            f.write("This is not a valid pytorch model file")
            temp_path = f.name

        try:
            with pytest.raises(ModelLoadError) as exc_info:
                SemanticFrontEnd(projection_head_path=temp_path)

            error_msg = str(exc_info.value)
            assert "Failed to load projection head" in error_msg
        finally:
            os.unlink(temp_path)

    def test_projection_head_architecture(self):
        """Test ProjectionHead initialization."""
        proj_head = ProjectionHead(input_dim=768, output_dim=4, dropout_rate=0.2)

        assert proj_head is not None
        assert hasattr(proj_head, 'fc1')
        assert hasattr(proj_head, 'fc2')
        assert hasattr(proj_head, 'bn1')
        assert hasattr(proj_head, 'dropout')
        assert hasattr(proj_head, 'sigmoid')

    def test_projection_head_output_shape(self):
        """Test that ProjectionHead outputs correct shape."""
        proj_head = ProjectionHead(input_dim=768, output_dim=4)
        proj_head.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 768)

        with torch.no_grad():
            output = proj_head(dummy_input)

        assert output.shape == (1, 4)

        # Check output is in valid range [0, 1] due to sigmoid
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestOutputGenerator:
    """Tests for OutputGenerator."""

    def test_output_generator_import(self):
        """Test that OutputGenerator can be imported."""
        from src.output_generator import OutputGenerator

        generator = OutputGenerator()
        assert generator is not None

    def test_output_generation(self):
        """Test output generation with valid inputs."""
        from src.output_generator import OutputGenerator
        from src.data_structures import Intent
        from src.frameworks import QLAEContext, ExecutionPlan, ExecutionStrategy, QLAEDomain

        generator = OutputGenerator()

        intent = Intent(
            purpose="To act with benevolent purpose",
            guiding_principles=["Guided by wisdom"]
        )

        context = QLAEContext(
            domains={QLAEDomain.ICE: 0.8},
            primary_domain=QLAEDomain.ICE,
            is_valid=True
        )

        execution = ExecutionPlan(
            strategy=ExecutionStrategy.COMPASSIONATE_ACTION,
            magnitude=0.7,
            description="Execute with compassion"
        )

        output = generator.generate(intent, context, execution)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "with" in output.lower()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_coordinate_with_zeros(self):
        """Test coordinate with all zeros."""
        coord = PhiCoordinate(love=0.0, justice=0.0, power=0.0, wisdom=0.0)
        assert coord.to_numpy().sum() == 0.0

    def test_coordinate_with_ones(self):
        """Test coordinate with all ones."""
        coord = PhiCoordinate(love=1.0, justice=1.0, power=1.0, wisdom=1.0)
        assert coord.to_numpy().sum() == 4.0

    def test_coordinate_with_float_precision(self):
        """Test coordinate with high precision floats."""
        coord = PhiCoordinate(
            love=0.123456789,
            justice=0.987654321,
            power=0.5555555,
            wisdom=0.3333333
        )
        assert coord.love == 0.123456789


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
