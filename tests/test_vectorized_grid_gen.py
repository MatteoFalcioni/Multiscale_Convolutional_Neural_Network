import unittest
import torch
import numpy as np
import random
from scripts.old_gpu_grid_gen import create_feature_grid_gpu
from scripts.vectorized_grid_gen import vectorized_create_feature_grids
from utils.point_cloud_data_utils import read_file_to_numpy


class TestFeatureGridCreation(unittest.TestCase):

    def setUp(self):
        # Common parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grid_resolution = 128
        self.selected_features = ['intensity', 'red', 'green']
        self.channels = len(self.selected_features)

        dataset_filepath = 'data/datasets/train_dataset.csv'
        self.data_array, known_features = read_file_to_numpy(data_dir=dataset_filepath)
        print(f"Loaded data with shape {self.data_array.shape}")

        self.seed = 42
        random.seed(self.seed)
        self.num_samples = 100  # Number of random indices to sample
        self.random_indices = np.random.choice(self.data_array.shape[0], size=self.num_samples, replace=False)

        self.single_idx = self.random_indices[0]



    def test_single_point(self):
        center_point_tensor = torch.tensor(self.data_array[self.single_idx, :], device=self.device)
        window_size = 5.0

        # Old function
        old_grid, old_cell_size, old_x_coords, old_y_coords, old_z = create_feature_grid_gpu(
            center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
        )

        # New function
        center_points = center_point_tensor.unsqueeze(0)  # Add batch dimension
        window_sizes = torch.tensor([window_size], device=self.device)

        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        # Compare outputs
        self.assertAlmostEqual(new_cell_sizes[0].item(), old_cell_size, places=6)
        self.assertTrue(torch.allclose(new_grid_coords[0, 0, :, 0], old_x_coords.flatten(), atol=1e-6))
        self.assertTrue(torch.allclose(new_grid_coords[0, 0, :, 1], old_y_coords.flatten(), atol=1e-6))
        self.assertTrue(torch.allclose(new_grid_coords[0, 0, :, 2], torch.full_like(old_x_coords.flatten(), old_z), atol=1e-6))


    def test_batch_support(self):
        center_points = torch.tensor(self.data_array[self.random_indices, :], device=self.device)
        window_size = 5.0

        # Old function results for each point
        old_results = [
            create_feature_grid_gpu(
                center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
            ) for center_point_tensor in center_points
        ]

        # New function
        window_sizes = torch.tensor([window_size], device=self.device)
        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        for i, (old_grid, old_cell_size, old_x_coords, old_y_coords, old_z) in enumerate(old_results):
            self.assertAlmostEqual(new_cell_sizes[0].item(), old_cell_size, places=6)
            self.assertTrue(torch.allclose(new_grid_coords[i, 0, :, 0], old_x_coords.flatten(), atol=1e-6))
            self.assertTrue(torch.allclose(new_grid_coords[i, 0, :, 1], old_y_coords.flatten(), atol=1e-6))
            self.assertTrue(torch.allclose(new_grid_coords[i, 0, :, 2], torch.full_like(old_x_coords.flatten(), old_z), atol=1e-6))

    def test_multiscale_support(self):
        center_point_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        window_sizes = torch.tensor([1.0, 2.0], device=self.device)  # Two scales

        # Old function results for each scale
        old_results = [
            create_feature_grid_gpu(
                center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
            ) for window_size in window_sizes
        ]

        # New function
        center_points = center_point_tensor.unsqueeze(0)  # Add batch dimension
        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        for scale_idx, (old_grid, old_cell_size, old_x_coords, old_y_coords, old_z) in enumerate(old_results):
            self.assertAlmostEqual(new_cell_sizes[scale_idx].item(), old_cell_size, places=6)
            self.assertTrue(torch.allclose(new_grid_coords[0, scale_idx, :, 0], old_x_coords.flatten(), atol=1e-6))
            self.assertTrue(torch.allclose(new_grid_coords[0, scale_idx, :, 1], old_y_coords.flatten(), atol=1e-6))
            self.assertTrue(torch.allclose(new_grid_coords[0, scale_idx, :, 2], torch.full_like(old_x_coords.flatten(), old_z), atol=1e-6))
