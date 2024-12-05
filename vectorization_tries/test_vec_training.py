import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.point_cloud_data_utils import extract_num_classes
from scripts.vectorized_training import vec_train, vec_validate, vec_train_epochs
from utils.vectorized_train_utils import prepare_utilities, new_prepare_dataloader


class TestVectorizedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")
        cls.full_data_filepath = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        cls.subset_file = 'data/datasets/train_dataset.csv'
        cls.selected_features = ['intensity', 'red']
        cls.num_channels = len(cls.selected_features)
        cls.num_classes = extract_num_classes(cls.full_data_filepath)
        cls.model = MultiScaleCNN(channels=cls.num_channels, classes=cls.num_classes).to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=1, gamma=0.5)
        cls.grid_resolution = 128
        cls.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        cls.batch_size = 8
        cls.num_workers = 16

        # Prepare shared utilities
        cls.shared_objects = prepare_utilities(
            full_data_filepath=cls.full_data_filepath,
            features_to_use=cls.selected_features,
            grid_resolution=cls.grid_resolution,
            window_sizes=cls.window_sizes,
            device=cls.device
        )

        # Prepare dataloaders
        cls.train_loader, cls.val_loader = new_prepare_dataloader(
            batch_size=cls.batch_size,
            data_filepath=cls.full_data_filepath,
            window_sizes=cls.window_sizes,
            features_to_use=cls.selected_features,
            train_split=0.8,
            num_workers=cls.num_workers,
            shuffle_train=True,
            device='cpu',  # Avoid GPU here for multiprocessing compatibility
            subset_file=cls.subset_file
        )


    def test_training_step(self):
        """Test the training step with shared utilities."""
        train_loss = vec_train(
            self.model,
            self.train_loader,
            self.criterion,
            self.optimizer,
            self.device,
            self.shared_objects
        )

        self.assertTrue(train_loss >= 0, "Training loss should be non-negative.")


    def test_validation_step(self):
        """Test the validation step with shared utilities."""
        val_loss = vec_validate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            self.shared_objects
        )

        self.assertTrue(val_loss >= 0, "Validation loss should be non-negative.")


    def test_full_training_pipeline(self):
        """Test the full training pipeline over multiple epochs."""
        epochs_to_run = 2
        model_save_folder = vec_train_epochs(
            self.model,
            self.train_loader,
            self.val_loader,
            self.criterion,
            self.optimizer,
            self.scheduler,
            epochs=epochs_to_run,
            patience=5,
            device=self.device,
            shared_objects=self.shared_objects,
            save=False
        )

        self.assertIsInstance(model_save_folder, str, "The model save folder should be a valid string.")

    def test_sanity_check(self):
        """Sanity check for data loader output."""
        first_batch = next(iter(self.train_loader))
        raw_points, labels, original_indices = first_batch

        self.assertEqual(len(raw_points), self.batch_size, "Batch size mismatch.")
        self.assertEqual(len(labels), self.batch_size, "Labels size mismatch.")
        self.assertEqual(len(original_indices), self.batch_size, "Original indices size mismatch.")

