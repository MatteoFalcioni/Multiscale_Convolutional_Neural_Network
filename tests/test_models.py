import unittest
import torch
from models.scnn import SingleScaleCNN
from models.mcnn import MultiScaleCNN


class TestSingleScaleCNN(unittest.TestCase):
    def setUp(self):
        """Set up the SingleScaleCNN model for testing."""
        self.model = SingleScaleCNN()
        self.input_shape = (1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, SingleScaleCNN)

    def test_forward_pass(self):
        """Test if the model forward pass works correctly and output shape is as expected."""
        self.model.eval()  # Set model to evaluation mode
        input_tensor = torch.randn(self.input_shape)  # Create a random tensor with the input shape
        output = self.model(input_tensor)

        # Assert output shape is (1, 128, 8, 8)
        self.assertEqual(output.shape, (1, 128, 8, 8))

    def test_forward_pass_batch(self):
        """Test the forward pass with a batch of images."""
        input_tensor = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        output = self.model(input_tensor)

        # Assert output shape is (5, 128, 8, 8)
        self.assertEqual(output.shape, (5, 128, 8, 8))


class TestMultiScaleCNN(unittest.TestCase):
    def setUp(self):
        """Set up the MultiScaleCNN model for testing."""
        self.model = MultiScaleCNN()
        self.input_shape = (1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN)

    def test_forward_pass(self):
        """Test if the model forward pass works correctly and output shape is as expected."""
        self.model.eval()  # Set model to evaluation mode
        input_tensor1 = torch.randn(self.input_shape)  # Create random tensors for the 3 scales
        input_tensor2 = torch.randn(self.input_shape)
        input_tensor3 = torch.randn(self.input_shape)
        output = self.model(input_tensor1, input_tensor2, input_tensor3)

        # Assert output shape is (1, 9)
        self.assertEqual(output.shape, (1, 9))

    def test_forward_pass_batch(self):
        """Test the forward pass with a batch of images for MCNN."""
        input_tensor1 = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        input_tensor2 = torch.randn(5, *self.input_shape[1:])
        input_tensor3 = torch.randn(5, *self.input_shape[1:])
        output = self.model(input_tensor1, input_tensor2, input_tensor3)

        # Assert output shape is (5, 9)
        self.assertEqual(output.shape, (5, 9))


if __name__ == "__main__":
    unittest.main()
