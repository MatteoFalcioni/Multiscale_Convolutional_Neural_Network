import unittest
import torch
import torch.nn as nn
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
        """Test if the model forward pass works correctly in both evaluation and training modes"""

        # Evaluation Mode Test
        self.model.eval()  # Set model to evaluation mode
        input_tensor = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        output = self.model(input_tensor)
        # Assert output shape is (5, 128, 8, 8)
        self.assertEqual(output.shape, (5, 128, 8, 8))

        # Training Mode Test
        self.model.train()  # Set model to training mode
        input_tensor = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        output = self.model(input_tensor)
        # Assert output shape is (5, 128, 8, 8)
        self.assertEqual(output.shape, (5, 128, 8, 8))

    def test_gradient_computation(self):
        """Test that gradients are computed correctly for SCNN."""
        input_tensor = torch.randn(5, 3, 128, 128)  # Batch size of 5
        self.model.train()  # Set model to training mode

        # Forward pass
        output = self.model(input_tensor)

        # Compute a dummy loss (mean of output)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check if gradients are not None or zero
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.all(param.grad == 0))  # Ensure gradients are not all zero
            else:
                self.fail("Gradient is None for one or more parameters")

    def test_invalid_input_shape(self):
        """Test if the model raises an error with incorrect input shapes."""
        invalid_input_tensor = torch.randn(1, 5, 128, 128)  # Incorrect input channels
        with self.assertRaises(RuntimeError):  # Expecting a RuntimeError due to shape mismatch
            self.model(invalid_input_tensor)

    def test_feature_extraction_output_shape(self):
        """Test if the SCNN model outputs the correct feature map shape."""
        # Create a small dummy dataset
        input_tensor = torch.randn(2, 3, 128, 128)  # Batch size of 2

        # Ensure the SCNN outputs the expected feature map shape
        self.model.eval()  # Set model to evaluation mode
        output = self.model(input_tensor)  # Output from SCNN

        # Expected output shape: (batch_size, 128, 8, 8)
        expected_shape = (2, 128, 8, 8)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_different_batch_sizes(self):
        """Test if the model works correctly with different batch sizes."""
        for batch_size in [2, 5, 10]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            output = self.model(input_tensor)
            self.assertEqual(output.shape, (batch_size, 128, 8, 8))  # Adjust shape for SCNN


class TestMultiScaleCNN(unittest.TestCase):
    def setUp(self):
        """Set up the MultiScaleCNN model for testing."""
        self.model = MultiScaleCNN()
        self.input_shape = (1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN)

    def test_forward_pass(self):
        """Test if the model forward pass works correctly in both evaluation and training modes"""

        # Evaluation Mode Test
        self.model.eval()  # Set model to evaluation mode
        input_tensor1 = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        input_tensor2 = torch.randn(5, *self.input_shape[1:])
        input_tensor3 = torch.randn(5, *self.input_shape[1:])
        output = self.model(input_tensor1, input_tensor2, input_tensor3)
        # Assert output shape is (5, 9)
        self.assertEqual(output.shape, (5, 9))

        # Training Mode Test
        self.model.train()  # Set model to training mode
        input_tensor1 = torch.randn(5, *self.input_shape[1:])  # Batch size of 5
        input_tensor2 = torch.randn(5, *self.input_shape[1:])
        input_tensor3 = torch.randn(5, *self.input_shape[1:])
        output = self.model(input_tensor1, input_tensor2, input_tensor3)
        # Assert output shape is (5, 9)
        self.assertEqual(output.shape, (5, 9))

    def test_invalid_input_shape(self):
        """Test if the MCNN model raises an error with incorrect input shapes."""
        invalid_input_tensor1 = torch.randn(1, 5, 128, 128)  # Incorrect input channels
        input_tensor2 = torch.randn(1, 3, 128, 128)
        input_tensor3 = torch.randn(1, 3, 128, 128)
        with self.assertRaises(RuntimeError):  # Expecting a RuntimeError due to shape mismatch
            self.model(invalid_input_tensor1, input_tensor2, input_tensor3)

    def test_overfit_small_dataset(self):
        """Test if the MCNN model can overfit on a small dataset of 2 samples."""
        # Create a small dummy dataset
        input_tensor1 = torch.randn(2, 3, 128, 128)
        input_tensor2 = torch.randn(2, 3, 128, 128)
        input_tensor3 = torch.randn(2, 3, 128, 128)
        labels = torch.tensor([0, 1])  # Dummy labels for 2 classes

        # Set up a simple optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Overfitting loop
        self.model.train()
        for _ in range(100):  # Train for 100 iterations
            optimizer.zero_grad()
            output = self.model(input_tensor1, input_tensor2, input_tensor3)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Test if loss is very small (indicating overfitting)
        self.assertLess(loss.item(), 0.01)

    def test_different_batch_sizes(self):
        """Test if the MCNN model works correctly with different batch sizes."""
        for batch_size in [2, 5, 10]:
            input_tensor1 = torch.randn(batch_size, 3, 128, 128)
            input_tensor2 = torch.randn(batch_size, 3, 128, 128)
            input_tensor3 = torch.randn(batch_size, 3, 128, 128)
            output = self.model(input_tensor1, input_tensor2, input_tensor3)
            self.assertEqual(output.shape, (batch_size, 9))


if __name__ == "__main__":
    unittest.main()
