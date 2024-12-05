import unittest
import torch
import numpy as np
import random
from scripts.vectorized_gpu_grid_gen import vectorized_create_feature_grids as torch_vectorized_create_feature_grids
from scripts.vectorized_gpu_grid_gen import vectorized_assign_features_to_grids as torch_vectorized_assign_features_to_grids
from scripts.cpu_vectorized_gen import numpy_create_feature_grids as np_vectorized_create_feature_grids
from scripts.cpu_vectorized_gen import numpy_assign_features_to_grids as np_vectorized_assign_features_to_grids
from utils.point_cloud_data_utils import read_file_to_numpy
from torch_kdtree import build_kd_tree
from scipy.spatial import cKDTree
from utils.plot_utils import visualize_grid


class TestNumpyVsTorchVectorization(unittest.TestCase):

    def setUp(self):
        # Common parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.channels = len(self.features_to_use)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.tensor_window_sizes = torch.tensor([size for _, size in self.window_sizes], dtype=torch.float64, device=self.device)
        self.np_window_sizes = np.array([size for _, size in self.window_sizes], dtype=np.float64)
        

        # full data
        dataset_filepath = 'data/datasets/train_dataset.csv'
        self.full_data_array, self.feature_names = read_file_to_numpy(data_dir=dataset_filepath)
        print(f"\n\nLoaded data with shape {self.full_data_array.shape}")
        self.feature_indices = [self.feature_names.index(feature) for feature in self.features_to_use]
        self.tensor_feature_indices = torch.tensor(self.feature_indices, device=self.device)

        # select num_samples random indices to test on
        self.seed = 42
        random.seed(self.seed)
        self.num_samples = 100  # Number of random indices to sample
        self.random_indices = np.random.choice(self.full_data_array.shape[0], size=self.num_samples, replace=False)

        self.single_idx = self.random_indices[0]
        
        self.tensor_full_data = torch.tensor(self.full_data_array, dtype=torch.float64, device=self.device)
        
        self.torch_kdtree = build_kd_tree(self.tensor_full_data[:, :3])
        self.np_kdtree = cKDTree(self.full_data_array[:, :3])


    def test_single_point(self):
        # Select a single point
        center_point_np = self.full_data_array[self.single_idx, :3]
        center_point_torch = torch.tensor(center_point_np, device=self.device, dtype=torch.float64).unsqueeze(0)

        window_size = 10.0

        # Torch vectorized function
        torch_cell_sizes, torch_grids, torch_grid_coords = torch_vectorized_create_feature_grids(
            center_point_torch, self.tensor_window_sizes[:1], self.grid_resolution, self.channels, self.device
        )

        # NumPy vectorized function
        np_cell_sizes, np_grids, np_grid_coords = np_vectorized_create_feature_grids(
            center_point_np[np.newaxis, :], self.np_window_sizes[:1], self.grid_resolution, self.channels
        )

        # Compare cell sizes
        self.assertAlmostEqual(
            torch_cell_sizes[0].item(), np_cell_sizes[0], places=10,
            msg=f"Cell size mismatch between Torch and NumPy implementations"
        )

        # Compare grid coordinates
        self.assertTrue(
            np.allclose(torch_grid_coords[0, 0].cpu().numpy(), np_grid_coords[0, 0], atol=1e-10),
            "Grid coordinate mismatch between Torch and NumPy implementations"
        )

    def test_batch_support(self):
        # Select random center points
        center_points_np = self.full_data_array[self.random_indices, :3]
        center_points_torch = torch.tensor(center_points_np, device=self.device, dtype=torch.float64)

        window_size = 10.0

        # Torch vectorized function
        torch_cell_sizes, torch_grids, torch_grid_coords = torch_vectorized_create_feature_grids(
            center_points_torch, self.tensor_window_sizes[:1], self.grid_resolution, self.channels, self.device
        )

        # NumPy vectorized function
        np_cell_sizes, np_grids, np_grid_coords = np_vectorized_create_feature_grids(
            center_points_np, self.np_window_sizes[:1], self.grid_resolution, self.channels
        )

        # Compare results for each point
        for i in range(self.num_samples):
            # Compare cell sizes
            self.assertAlmostEqual(
                torch_cell_sizes[0].item(), np_cell_sizes[0], places=10,
                msg=f"Cell size mismatch for sample {i}"
            )

            # Compare grid coordinates
            self.assertTrue(
                np.allclose(torch_grid_coords[i, 0].cpu().numpy(), np_grid_coords[i, 0], atol=1e-10),
                f"Grid coordinate mismatch for sample {i}"
            )

    def test_multiscale_support(self):
        # Select a single point
        center_point_np = self.full_data_array[self.random_indices[0], :3]
        center_point_torch = torch.tensor(center_point_np, device=self.device, dtype=torch.float64).unsqueeze(0)

        # Torch vectorized function
        torch_cell_sizes, torch_grids, torch_grid_coords = torch_vectorized_create_feature_grids(
            center_point_torch, self.tensor_window_sizes, self.grid_resolution, self.channels, self.device
        )

        # NumPy vectorized function
        np_cell_sizes, np_grids, np_grid_coords = np_vectorized_create_feature_grids(
            center_point_np[np.newaxis, :], self.np_window_sizes, self.grid_resolution, self.channels
        )

        # Compare results for each scale
        for scale_idx in range(len(self.window_sizes)):
            # Compare cell sizes
            self.assertAlmostEqual(
                torch_cell_sizes[scale_idx].item(), np_cell_sizes[scale_idx], places=10,
                msg=f"Cell size mismatch for scale {scale_idx}"
            )

            # Compare grid coordinates
            self.assertTrue(
                np.allclose(torch_grid_coords[0, scale_idx].cpu().numpy(), np_grid_coords[0, scale_idx], atol=1e-10),
                f"Grid coordinate mismatch for scale {scale_idx}"
            )
            
    def test_feature_assignment(self):
        # Select random center points
        center_points_np = self.full_data_array[self.random_indices, :3]
        center_points_torch = torch.tensor(center_points_np, device=self.device, dtype=torch.float64)

        # Torch vectorized grid generation
        torch_cell_sizes, torch_grids, torch_grid_coords = torch_vectorized_create_feature_grids(
            center_points_torch, self.tensor_window_sizes, self.grid_resolution, self.channels, self.device
        )

        # NumPy vectorized grid generation
        np_cell_sizes, np_grids, np_grid_coords = np_vectorized_create_feature_grids(
            center_points_np, self.np_window_sizes, self.grid_resolution, self.channels
        )

        # Assign features using Torch
        torch_grids = torch_vectorized_assign_features_to_grids(
            self.torch_kdtree, self.tensor_full_data,
            torch_grid_coords, torch_grids, self.tensor_feature_indices,
            self.device
        )

        # Assign features using NumPy
        np_grids = np_vectorized_assign_features_to_grids(
            self.np_kdtree, self.full_data_array, np_grid_coords, np_grids, self.feature_indices
        )

        # Compare results for each point
        for i in range(self.num_samples):
            for scale_idx in range(len(self.window_sizes)):
                torch_grid = torch_grids[i, scale_idx].cpu().numpy()
                np_grid = np_grids[i, scale_idx]
                self.assertTrue(
                    np.allclose(torch_grid, np_grid, atol=1e-10),
                    f"Feature assignment mismatch for sample {i}, scale {scale_idx}"
                )
