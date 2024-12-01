import unittest
from unittest import mock
from torch.utils.data import DataLoader
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values, compute_point_cloud_bounds
from models.mcnn import MultiScaleCNN
from scipy.spatial import cKDTree
from utils.train_data_utils import PointCloudDataset, prepare_dataloader, save_model, load_model, load_parameters, save_used_parameters
import torch
import numpy as np
from tqdm import tqdm
import os
import shutil
import pandas as pd
import time


'''class TestSaveLoadModel(unittest.TestCase):
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
                \nloaded features: {loaded_features} -> loaded_indinces: {loaded_indices}\n\n")'''


class TestPointCloudDataset(unittest.TestCase):
    def setUp(self):
        real_data_filepath = ''  # read a real las file to test thoroughly 
        self.real_array, self.real_known_features = read_file_to_numpy(data_dir=real_data_filepath)
        self.real_subset_file = ''   # A real subset file for testing

        print(f"\nReal data array shape {self.real_array.shape}")

        # use a sample file for pipeline tests
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir='tests/test_subtiler/32_687000_4930000_FP21_sampled_10k.las')
        self.full_data_array, _ = remap_labels(self.full_data_array)
        self.full_data_array = clean_nan_values(data_array=self.full_data_array)
        print(f"\nlen of full data array: {len(self.full_data_array)}")
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.num_channels = len(self.features_to_use)

        # Create a subset file (mock subset points for testing)
        self.n_subset = 5000
        self.seed = 42
        np.random.seed(self.seed)
        random_indices = np.random.choice(self.full_data_array.shape[0], self.n_subset, replace=False)  # Randomly select indices
        subset_points = self.full_data_array[random_indices, :3]  # Select corresponding points
        print(f"\nlen of subset points: {len(subset_points)}, dtype: {subset_points.dtype}")
        self.subset_file = "tests/test_subset.csv"
        pd.DataFrame(subset_points, columns=['x', 'y', 'z']).to_csv(self.subset_file, index=False)

        self.atol = 1e-8    # absolute tolerance for subset selection. This is really important, because LiDAR file have difference in the order of 10^-10 to 10^-16
                            # a tolerance of 1e^-8 is sufficient to avoid selection errors

        # Create the dataset instance with subset filtering
        self.dataset = PointCloudDataset(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features,
            subset_file=self.subset_file
        )


    def tearDown(self):
        # Clean up temporary subset file
        if os.path.exists(self.subset_file):
            os.remove(self.subset_file)


    def test_len(self):
        # Test that the length of the dataset matches the number of points in the selected array
        self.assertEqual(len(self.dataset), len(self.dataset.selected_array))


    def test_subset_filtering(self):
        # Verify that the subset filtering works correctly
        subset_points = pd.read_csv(self.subset_file, dtype={'x': 'float64', 'y': 'float64', 'z': 'float64'}).values
        print(f"Subset points shape: {subset_points.shape}")

        # Re-generate the same random indices to compare with the selected points
        np.random.seed(self.seed)  # Ensure reproducibility
        random_indices = np.random.choice(len(self.full_data_array), self.n_subset, replace=False)

        # Extract expected subset points from the full data array
        expected_subset_points = self.full_data_array[random_indices, :3]

        # Verify that the saved subset matches the expected points
        assert np.allclose(
            subset_points,
            expected_subset_points,
            atol=self.atol
        ), "Subset points do not match after reloading from the CSV!"

        # Check bounds
        # Always compute bounds on the full data (grids should not fall out of bounds of the full pc, not of the selected subset) 
        bounds = compute_point_cloud_bounds(self.dataset.full_data_array)
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        max_half_window = max(ws[1] / 2 for ws in self.dataset.window_sizes)
        print(f"max half window: {max_half_window}")
        print(f"Computed bounds: {bounds}")
        print(f"Max half window size: {max_half_window}")

        out_of_bounds_mask = (
            (subset_points[:, 0] - max_half_window <= x_min) |
            (subset_points[:, 0] + max_half_window >= x_max) |
            (subset_points[:, 1] - max_half_window <= y_min) |
            (subset_points[:, 1] + max_half_window >= y_max)
        )
        print(f"Out-of-bounds mask applied to subset: {np.sum(~out_of_bounds_mask)} points out of bounds.")
        print(f"Out-of-bounds mask type: {out_of_bounds_mask.dtype}, shape: {out_of_bounds_mask.shape}")

        # Subset points expected to remain after filtering
        in_bounds_points = subset_points[~out_of_bounds_mask]
        print(f"Subset points in bounds: {len(in_bounds_points)}")

        print(f"Dataset.selected_array length: {len(self.dataset.selected_array)}")
        print(f"Dataset.selected_array dtype: {self.dataset.selected_array.dtype}, shape: {self.dataset.selected_array.shape}")

        # Assert that the selected array matches the in-bounds points
        np.testing.assert_allclose(
            np.sort(self.dataset.selected_array[:, :3], axis=0),
            np.sort(in_bounds_points, axis=0),
            err_msg=f"Selected array does not match the in-bounds points from the subset file.", 
            atol=self.atol
        )


    def test_original_indices_mapping(self):
        # Test that original_indices map correctly from selected_array to full_data_array
        for idx in tqdm(range(len(self.dataset)), desc="Test index mapping", unit="processed indices"):
            original_idx = self.dataset.original_indices[idx]
            np.testing.assert_array_equal(
                self.dataset.selected_array[idx],
                self.dataset.full_data_array[original_idx],
                err_msg=f"Mapping mismatch between selected_array and full_data_array at index {idx}"
            )


    def test_getitem_validity(self):
        # Test retrieving grids and labels for multiple indices
        num_tests = 100
        indices = np.random.choice(len(self.dataset), num_tests, replace=False)

        for idx in tqdm(indices, desc="Testing getitem method", unit="processed points"):
            result = self.dataset[idx]

            if result is not None:
                small_grid, medium_grid, large_grid, label, original_idx = result

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
                self.assertEqual(
                    label.item(), self.dataset.full_data_array[original_idx, -1], 
                    f"Label mismatch between selected_array and full_data_array at index {idx}"
                )
            

    def test_no_subset_file_fall_back(self):

        fall_back_dataset = PointCloudDataset(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features,
            subset_file=None
        )
        fall_back_array = fall_back_dataset.selected_array
        full_data_array = self.full_data_array

        print(f"full data array shape: {full_data_array.shape}, dtype: {full_data_array.dtype}")
        print(f"Selected array shape without subset file selection: {fall_back_array.shape}, dtype: {fall_back_array.dtype}\n")
        print(f"Checking if this matches with manually computed in bound points...\n")

        bounds = compute_point_cloud_bounds(full_data_array)
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        max_half_window = max(ws[1] / 2 for ws in self.dataset.window_sizes)
        print(f"Computed bounds: {bounds}")

        out_of_bounds_mask = (
            (full_data_array[:, 0] - max_half_window <= x_min) |
            (full_data_array[:, 0] + max_half_window >= x_max) |
            (full_data_array[:, 1] - max_half_window <= y_min) |
            (full_data_array[:, 1] + max_half_window >= y_max)
        )

        # Subset points expected to remain after filtering
        in_bound_array = full_data_array[~out_of_bounds_mask]
        print(f"Fulla data array in bound points: {len(in_bound_array)}")

        np.testing.assert_array_equal(in_bound_array, fall_back_array, err_msg=f"The selected array without subset selection doesn't match full data array \
                                      without out of bounds points.")
        
    def test_with_real_data(self):

        # input a real huge file (about 10 million points) and a subset of usual dimensions (about 2.5 million points)
        dataset_start = time.time()
        huge_dataset = PointCloudDataset(
            full_data_array=self.real_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.real_known_features,
            subset_file=self.real_subset_file
        )
        dataset_end = time.time()
        print(f"Huge dataset created in {dataset_end-dataset_start}")

        print(f"\nFull real data array selection produced: {huge_dataset.full_data_array.shape[0]} --> {huge_dataset.selected_array.shape[0]}")

        # retrieve n_test grids for testing
        n_test = int(1e5)
        np.random.seed(self.seed)  # Ensure reproducibility
        random_indices = np.random.choice(len(self.real_data_array), n_test, replace=False)

        hugefile_start = time.time()
        for idx in tqdm(random_indices, desc="retrieving grids from real data", unit="processed points"):
            result = huge_dataset[idx]
            small_grid, medium_grid, large_grid, label, original_idx = result
        hugefile_end = time.time()
        print(f"\nHuge file input: Retrieved {n_test} dataset elements in {(hugefile_end-hugefile_start)/60} minutes")

        # Now compare with a smaller file input. You can use the subset file for this.
        # what we were doing earlier than this selection implementation was to input about 2.5 million pc points and remove out of bounds 
        small_data_array, small_known_features = read_file_to_numpy(self.real_subset_file)

        small_dataset_start = time.time()
        small_dataset = PointCloudDataset(
            full_data_array=small_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=small_known_features,
            subset_file=None
        )
        small_dataset_end = time.time()
        print(f"\nSmall dataset created in {small_dataset_end-small_dataset_start}")

        # retrieve n_test grids for testing
        n_test = int(1e5)
        np.random.seed(self.seed)  # Ensure reproducibility
        random_indices = np.random.choice(len(small_data_array), n_test, replace=False)

        smallfile_start = time.time()
        for idx in tqdm(random_indices, desc="retrieving grids from real data", unit="processed points"):
            result = small_dataset[idx]
            small_grid, medium_grid, large_grid, label, original_idx = result
        smallfile_end = time.time()
        print(f"\nSmall file input: Retrieved {n_test} dataset elements in {(smallfile_end-smallfile_start)/60} minutes")

        
        
    ''' you stopped skipping batches now, no need for this
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

'''
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
'''