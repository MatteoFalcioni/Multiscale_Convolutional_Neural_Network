import unittest
from unittest import mock
from torch.utils.data import DataLoader
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from scripts.point_cloud_to_image import compute_point_cloud_bounds
from scipy.spatial import cKDTree
from utils.train_data_utils import PointCloudDataset, prepare_dataloader, custom_collate_fn
import torch
import numpy as np
from tqdm import tqdm


'''class TestPointCloudDataset(unittest.TestCase):
    def setUp(self):
        # Mock point cloud data for the test
        self.data_array, self.known_features = read_file_to_numpy(data_dir='data/chosen_tiles/32_687000_4930000_FP21.las')
        self.data_array, _ = remap_labels(self.data_array)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.num_channels = len(self.features_to_use)

        # Build the KDTree once for the test
        self.kdtree = cKDTree(self.data_array[:, :3])
        self.point_cloud_bounds = compute_point_cloud_bounds(self.data_array)

        # Create the dataset instance
        self.dataset = PointCloudDataset(
            data_array=self.data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features
        )

    def test_len(self):
        # Test that the length of the dataset matches the number of points
        self.assertEqual(len(self.dataset), len(self.data_array))

    def test_getitem_validity(self):
        
        num_tests = 10000
        indices = np.random.choice(len(self.dataset), num_tests, replace=False)
        
        # Test retrieving grids and labels for multiple indices
        for idx in tqdm(indices, desc="Testing getitem method", unit="processed points"):  
            result = self.dataset[idx]

            if result is not None:
                small_grid, medium_grid, large_grid, label, _ = result

                # Check grid shapes
                self.assertEqual(small_grid.shape, (self.num_channels, self.grid_resolution, self.grid_resolution),
                                 f"Invalid shape for small grid at index {idx}")
                self.assertEqual(medium_grid.shape, (self.num_channels, self.grid_resolution, self.grid_resolution),
                                 f"Invalid shape for medium grid at index {idx}")
                self.assertEqual(large_grid.shape, (self.num_channels, self.grid_resolution, self.grid_resolution),
                                 f"Invalid shape for large grid at index {idx}")

                # Check for NaN or Inf values in grids
                for grid, scale in zip([small_grid, medium_grid, large_grid], ['small', 'medium', 'large']):
                    self.assertFalse(torch.isnan(grid).any(), f"NaN values found in {scale} grid at index {idx}")
                    self.assertFalse(torch.isinf(grid).any(), f"Inf values found in {scale} grid at index {idx}")

                # Check label validity
                self.assertIsInstance(label, torch.Tensor, f"Label is not a tensor at index {idx}")
                self.assertEqual(label.item(), self.data_array[idx, -1], f"Label mismatch at index {idx}")
            
        
    @mock.patch('utils.train_data_utils.PointCloudDataset.__getitem__')
    def test_dataloader_with_none(self, mock_getitem):
        # Mock __getitem__ to return None for certain indices
        def side_effect(idx):
            if idx % 2 == 0:  # Simulate returning None for every other point
                return None
            else:
                # Return a valid tuple with grid data and label
                return (torch.randn(self.num_channels, self.grid_resolution, self.grid_resolution), \
                       torch.randn(self.num_channels, self.grid_resolution, self.grid_resolution), \
                       torch.randn(self.num_channels, self.grid_resolution, self.grid_resolution), \
                       torch.tensor(1),  
                       torch.tensor(1))

        mock_getitem.side_effect = side_effect

        # Prepare DataLoader using the mocked dataset and your custom collate function
        dataloader = DataLoader(self.dataset, batch_size=4, collate_fn=custom_collate_fn)

        # Fetch batches and ensure None entries are skipped
        for batch in dataloader:
            if batch is not None:
                small_grid, medium_grid, large_grid, labels, indices = batch
                # None values should be skipped, so we expect 2 valid entries per batch
                self.assertEqual(small_grid.size(0), 2, "Batch size should be 2 after skipping None entries.")'''


class TestPrepareDataloader(unittest.TestCase):

    def setUp(self):
        # Configurable parameters for the test
        self.batch_size = 32
        self.grid_resolution = 128
        self.train_split = 0.8
        self.data_dir = 'data/chosen_tiles/32_687000_4930000_FP21.las'  # Path to raw data for this test
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.num_channels = len(self.features_to_use)

    def test_dataloader_full_dataset(self):
        # Prepare DataLoader without train/test split (using full dataset)
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.data_dir,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=None,  # No train/test split, using full dataset
            num_workers=4,
            features_file_path=None,
            shuffle_train=True
        )

        # Ensure that the train_loader is returned properly and eval_loader is None
        self.assertIsInstance(train_loader, DataLoader, "train_loader is not a DataLoader instance.")
        self.assertIsNone(eval_loader, "eval_loader should be None when train_split=0.0.")

        # Fetch the first batch and verify its structure
        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None (all points skipped).")
        self.assertEqual(len(first_batch), 5, "Expected 5 elements in the batch (small_grid, medium_grid, large_grid, label, indexes).")

        small_grid, medium_grid, large_grid, labels, indices = first_batch
        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")

    def test_train_eval_dataloader(self):
        # Load both train and eval DataLoader
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.data_dir,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=self.train_split
        )

        # Test the train_loader
        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None (all points skipped).")
        small_grid, medium_grid, large_grid, labels, indices = first_batch
        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")
        self.assertLessEqual(len(labels), self.batch_size, "len labels should be smaller or equal to specified batch size due to skipped points.")
        self.assertLessEqual(len(indices), self.batch_size, "len indices should be smaller or equal to specified batch size due to skipped points.")
        

        # Test the eval_loader (ensure it was created and works)
        self.assertIsNotNone(eval_loader, "Evaluation DataLoader is None when it shouldn't be.")
        first_batch_eval = next(iter(eval_loader))
        self.assertIsNotNone(first_batch_eval, "First batch of eval_loader should not be None (all points skipped).")
        small_grid, medium_grid, large_grid, labels_eval, indices = first_batch_eval
        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch in eval.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch in eval.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch in eval.")
        self.assertLessEqual(len(labels_eval), self.batch_size, "len labels should be smaller or equal to specified batch size in eval due to skipped points.")
        self.assertLessEqual(len(indices), self.batch_size, "len indices should be smaller or equal to specified batch size due to skipped points.")
        
        
class TestCustomCollateFn(unittest.TestCase):
    def setUp(self):
        # Mock dataset entries
        self.mock_data = [
            (torch.randn(3, 128, 128), torch.randn(3, 128, 128), torch.randn(3, 128, 128), torch.tensor(1), 0),
            None,  # Simulate a skipped point
            (torch.randn(3, 128, 128), torch.randn(3, 128, 128), torch.randn(3, 128, 128), torch.tensor(2), 1),
        ]

    def test_collate_fn(self):
        # Apply the custom collate function
        batch = custom_collate_fn(self.mock_data)

        # Check batch structure
        self.assertIsNotNone(batch, "Batch should not be None.")
        small_grids, medium_grids, large_grids, labels, indices = batch

        # Check batch sizes
        self.assertEqual(small_grids.size(0), 2, "Batch size should match the number of valid points.")
        self.assertEqual(medium_grids.size(0), 2)
        self.assertEqual(large_grids.size(0), 2)
        self.assertEqual(labels.size(0), 2)
        self.assertEqual(indices.size(0), 2)
        

class TestDataloaderDatasetIntegration(unittest.TestCase):
    def setUp(self):
        # Mock parameters
        self.batch_size = 32
        self.grid_resolution = 128
        self.data_dir = 'data/chosen_tiles/32_687000_4930000_FP21.las'
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.num_channels = len(self.features_to_use)

    def test_dataloader_with_dataset(self):
        # Prepare DataLoader with the dataset
        inference_loader, _ = prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.data_dir,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=None,   # lets get only inference dataset, no eval
            num_workers=16,
            features_file_path=None,
            shuffle_train=True  # usualy no shuffle in inference, but in this way we can test random batches
        )

        num_batches_to_test = 5000
        for i, batch in enumerate(tqdm(inference_loader, desc="Testing batches for NaN/Inf", total=num_batches_to_test)):
            if i >= num_batches_to_test:
                break  # Test only a subset of batches
            
            if batch is None:
                print(f"Batch {i} skipped: No valid points.")
                continue

            small_grid, medium_grid, large_grid, labels, indices = batch
            
            # Validate grid shapes
            self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Small grid resolution mismatch")
            self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Medium grid resolution mismatch")
            self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Large grid resolution mismatch")

            # Check for NaN/Inf values
            for grid, scale in zip([small_grid, medium_grid, large_grid], ['small', 'medium', 'large']):
                self.assertFalse(torch.isnan(grid).any(), f"NaN values found in {scale} grid for batch {i}")
                self.assertFalse(torch.isinf(grid).any(), f"Inf values found in {scale} grid for batch {i}")

            #print(f"Batch {i} passed all NaN/Inf checks.")