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
        center_point_tensor = torch.tensor(self.data_array[self.single_idx, :], device=self.device, dtype=torch.float64)
        window_size = 5.0

        # Old function
        old_grid, old_cell_size, old_x_coords, old_y_coords, old_z = create_feature_grid_gpu(
            center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
        )
        # Mesh the old grid to create the full 2D grid
        old_grid_x, old_grid_y = torch.meshgrid(old_x_coords, old_y_coords, indexing='ij')
        old_grid_coords = torch.stack(
            (
                old_grid_x.flatten(),
                old_grid_y.flatten(),
                torch.full((old_grid_x.numel(),), old_z, device=self.device, dtype=torch.float64),
            ),
            dim=-1
        )

        # New function
        center_points = center_point_tensor.unsqueeze(0)  # Add batch dimension
        window_sizes = torch.tensor([window_size], device=self.device, dtype=torch.float64)

        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        # Compare cell sizes
        self.assertAlmostEqual(new_cell_sizes[0].item(), old_cell_size, places=10)

        # Compare grid coordinates
        # new_grid_coords[0,0] selects first batch, first scale 
        self.assertTrue(torch.allclose(new_grid_coords[0, 0], old_grid_coords, atol=1e-10))


    def test_batch_support(self):
        # Select random center points from the dataset
        center_points = torch.tensor(self.data_array[self.random_indices, :], device=self.device, dtype=torch.float64)
        window_size = 5.0

        # Old function results for each point
        old_results = [
            create_feature_grid_gpu(
                center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
            ) for center_point_tensor in center_points
        ]

        # Mesh the old grid for each result
        old_meshed_coords = []
        for old_grid, old_cell_size, old_x_coords, old_y_coords, old_z in old_results:
            grid_x, grid_y = torch.meshgrid(old_x_coords, old_y_coords, indexing='ij')
            old_meshed_coords.append(
                torch.stack(
                    (
                        grid_x.flatten(),
                        grid_y.flatten(),
                        torch.full((grid_x.numel(),), old_z, device=self.device, dtype=torch.float64),
                    ),
                    dim=-1
                )
            )

        # New function
        window_sizes = torch.tensor([window_size], device=self.device, dtype=torch.float64)
        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        # Compare results
        for i, old_grid_coords in enumerate(old_meshed_coords):
            self.assertAlmostEqual(new_cell_sizes[0].item(), old_results[i][1], places=10)  # Compare cell sizes
            self.assertTrue(torch.allclose(new_grid_coords[i, 0], old_grid_coords, atol=1e-10))  # Compare coordinates
            

    def test_multiscale_support(self):
        center_point_tensor = torch.tensor(self.data_array[self.random_indices[0], :], device=self.device, dtype=torch.float64)
        window_sizes = torch.tensor([1.0, 2.0], device=self.device)  # Two scales

        # Old function results for each scale
        old_results = [
            create_feature_grid_gpu(
                center_point_tensor, self.device, window_size, self.grid_resolution, self.channels
            ) for window_size in window_sizes
        ]

        # Mesh the old grid for each scale
        old_meshed_coords = []
        for old_grid, old_cell_size, old_x_coords, old_y_coords, old_z in old_results:
            grid_x, grid_y = torch.meshgrid(old_x_coords, old_y_coords, indexing='ij')
            old_meshed_coords.append(
                torch.stack(
                    (
                        grid_x.flatten(),
                        grid_y.flatten(),
                        torch.full((grid_x.numel(),), old_z, device=self.device, dtype=torch.float64),
                    ),
                    dim=-1
                )
            )

        # New function
        center_points = center_point_tensor.unsqueeze(0)  # Add batch dimension
        new_cell_sizes, new_grids, new_grid_coords = vectorized_create_feature_grids(
            center_points, window_sizes, self.grid_resolution, self.channels, self.device
        )

        # Compare results
        for scale_idx, old_grid_coords in enumerate(old_meshed_coords):
            self.assertAlmostEqual(new_cell_sizes[scale_idx].item(), old_results[scale_idx][1], places=10)  # Compare cell sizes
            self.assertTrue(torch.allclose(new_grid_coords[0, scale_idx], old_grid_coords, atol=1e-10))  # Compare coordinates

