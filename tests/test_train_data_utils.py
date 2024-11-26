import unittest
from unittest import mock
from torch.utils.data import DataLoader
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from scripts.point_cloud_to_image import compute_point_cloud_bounds
from models.mcnn import MultiScaleCNN
from scipy.spatial import cKDTree
from utils.train_data_utils import PointCloudDataset, prepare_dataloader, custom_collate_fn, save_model, load_model, load_parameters, save_used_parameters
import torch
import numpy as np
from tqdm import tqdm
import os
import shutil


class TestSaveLoadModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize a test model
        self.num_channels = 4
        self.num_classes = 6
        self.dummy_model = MultiScaleCNN(channels=self.num_channels, classes=self.num_classes).to(self.device)
        self.save_dir = "tests/test_saved_models"
        
        self.real_model_path = 'models/saved/mcnn_model_20241116_143003/model.pth'
        
        self.temp_dir = "tests/temp_test_model"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Define sample features and hyperparameters
        self.sample_features = ["intensity", "red", "green", "blue", "nir", "delta_z", "l1", "l2", "l3"]
        self.sample_hyperparameters = {
            'training file': 'data/training_data/21/train_21.csv',
            'num_classes' : 6,
            'number of total points' : 2680354,
            'grid_resolution': 128,
            'patience' : 2,
            'momentum' : 0.9,
            'step_size' : 5,
            'learning_rate_decay_factor' : 0.5,
            'num_workers' : 32,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "window_sizes": [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        }
        

    def tearDown(self):
        # Clean up the temporary directory after tests
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_and_load_parameters(self):
        # Save parameters to the temporary directory
        save_used_parameters(
            used_features=self.sample_features,
            hyperparameters=self.sample_hyperparameters,
            save_dir=self.temp_dir
        )

        # Load parameters back
        loaded_features, num_channels, loaded_window_sizes = load_parameters(self.temp_dir)

        # Validate features
        self.assertEqual(loaded_features, self.sample_features, "Loaded features do not match saved features.")
        self.assertEqual(num_channels, len(self.sample_features), "Number of channels does not match the number of features.")

        # Validate window sizes
        self.assertEqual(loaded_window_sizes, self.sample_hyperparameters["window_sizes"], "Loaded window sizes do not match saved window sizes.")

        # Check if the saved files exist
        features_file = os.path.join(self.temp_dir, "features_used.csv")
        hyperparameters_file = os.path.join(self.temp_dir, "hyperparameters.csv")
        self.assertTrue(os.path.exists(features_file), "Features file was not saved.")
        self.assertTrue(os.path.exists(hyperparameters_file), "Hyperparameters file was not saved.")

        # Print success message
        print(f"Parameters saved and loaded successfully. Saved in {self.temp_dir}.")


    def test_save_and_load_dummy_model(self):
        hyperparameters = {'num_classes' : 6}
        
        # Save the model
        save_path = save_model(self.dummy_model, save_dir=self.save_dir, hyperparameters=hyperparameters)

        # Load the model
        loaded_model = load_model(os.path.join(save_path, "model.pth"), self.device, num_channels=4)

        # Validate the weights
        for name, param in loaded_model.state_dict().items():
            self.assertFalse(torch.isnan(param).any(), f"NaN found in {name} after loading.")
            self.assertFalse(torch.isinf(param).any(), f"Inf found in {name} after loading.")

        print("Model saving and loading validated successfully.")
        
        
    def test_save_and_load_real_model(self):
        # Load the model
        features_list, num_loaded_channels, window_sizes = load_parameters(self.real_model_path)
        
        print(f"Loaded features for real model: {features_list}")
        
        loaded_model = load_model(self.real_model_path, self.device, num_channels=num_loaded_channels)

        # Validate the weights
        for name, param in loaded_model.state_dict().items():
            self.assertFalse(torch.isnan(param).any(), f"NaN found in {name} after loading.")
            self.assertFalse(torch.isinf(param).any(), f"Inf found in {name} after loading.")

        print("Real model saving and loading validated successfully.") 
        
    def test_compare_dummy_and_real_model(self):
        
        fresh_model = MultiScaleCNN(channels=9, classes=6).to(self.device)
        
        features_list, num_loaded_channels, window_sizes = load_parameters(self.real_model_path)
        model = load_model(self.real_model_path, self.device, num_channels=num_loaded_channels)
        
        for (name1, param1), (name2, param2) in zip(fresh_model.state_dict().items(), model.state_dict().items()):
            assert name1 == name2, "Mismatch in parameter names"
            print(f"{name1}: max_diff={torch.max(torch.abs(param1 - param2))}")
        
        
    def test_compare_load_features_and_used_features(self):
        training_file = 'data/training_data/21/train_21.csv'
        inference_file = 'data/chosen_tiles/32_687000_4930000_FP21.las'
        
        _ , original_features = read_file_to_numpy(training_file, features_to_use=None)
        training_features = ["intensity", "red", "green", "blue", "nir", "delta_z", "l1", "l2", "l3"]
        training_indices =  [original_features.index(feature) for feature in training_features]
        
        _, inference_features = read_file_to_numpy(inference_file, features_to_use=None)
        loaded_features, _, _ = load_parameters(self.real_model_path)
        loaded_indices = [inference_features.index(feature) for feature in loaded_features]
        
        print(f"\n\nknown features in orginal file: {original_features}\
                \ntraining features: {training_features} -> training indices: {training_indices}\
                \nknown features in inference file:{inference_features}\
                \nloaded features: {loaded_features} -> loaded_indinces: {loaded_indices}\n\n")


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
                self.assertEqual(small_grid.size(0), 2, "Batch size should be 2 after skipping None entries.")


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
                self.assertFalse(torch.isinf(grid).any(), f"Inf values found in {scale} grid for batch {i}")'''
