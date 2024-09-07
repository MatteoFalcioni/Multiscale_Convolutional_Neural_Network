import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.data_utils import prepare_dataloader
from scripts.train import train, validate, train_epochs, save_model
from utils.plot_utils import plot_loss


class TestTrainingProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run once for the entire test class
        cls.device = torch.device('cpu')  # Use CPU for testing
        cls.model = MultiScaleCNN().to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=5, gamma=0.5)

        # Mocked DataLoader with random data
        cls.train_loader = prepare_dataloader(batch_size=4)
        cls.val_loader = prepare_dataloader(batch_size=4)

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN, "Model should be an instance of MultiScaleCNN.")
        self.assertTrue(next(self.model.parameters()).is_cuda == False, "Model should be on CPU for testing.")

    def test_data_loader(self):
        """Test if data loaders are working properly."""
        batch = next(iter(self.train_loader))
        inputs, labels = batch
        self.assertEqual(inputs.size(0), 4, "Batch size should be 4.")
        self.assertEqual(inputs.ndim, 4, "Inputs should be 4-dimensional.")
        self.assertEqual(labels.ndim, 1, "Labels should be 1-dimensional.")

    def test_training_step(self):
        """Test the training step for a single batch."""
        initial_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        self.assertTrue(initial_loss >= 0, "Initial loss should be non-negative.")

    def test_validation_step(self):
        """Test the validation step for a single batch."""
        val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
        self.assertTrue(val_loss >= 0, "Validation loss should be non-negative.")

    @patch('utils.plot_utils.plot_loss')
    def test_train_epochs_with_early_stopping(self, mock_plot_loss):
        """Test the full training loop with early stopping and ensure plotting is called."""
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                     self.optimizer, self.scheduler, epochs=5, device=self.device,
                     save_dir='models/saved/', patience=2)

        mock_plot_loss.assert_called_once()  # Ensure plot_loss is called once at the end

    def test_plot_loss(self):
        """Test the plotting function."""
        train_losses = [0.8, 0.6, 0.5, 0.4]
        val_losses = [0.9, 0.7, 0.6, 0.5]
        try:
            plot_loss(train_losses, val_losses, save_path='tests/test_loss_plot.png')
        except Exception as e:
            self.fail(f"plot_loss() raised {e} unexpectedly!")
