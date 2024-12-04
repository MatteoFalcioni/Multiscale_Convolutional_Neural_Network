import unittest
import torch
import numpy as np
import random
from scripts.old_gpu_grid_gen import create_feature_grid_gpu, apply_masks_gpu, assign_features_to_grid_gpu
from scripts.vectorized_grid_gen import vectorized_create_feature_grids, vectorized_assign_features_to_grids, vectorized_generate_multiscale_grids
from utils.point_cloud_data_utils import read_file_to_numpy
from torch_kdtree import build_kd_tree
from utils.plot_utils import visualize_grid


class TestFeatureAssignment(unittest.TestCase):

    def setUp(self):
        # Common parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.channels = len(self.features_to_use)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.tensor_window_sizes = torch.tensor([size for _, size in self.window_sizes], dtype=torch.float64, device=self.device)
        
        # full data
        dataset_filepath = 'data/datasets/train_dataset.csv'
        self.full_data_array, known_features = read_file_to_numpy(data_dir=dataset_filepath)
        print(f"\n\nLoaded data with shape {self.full_data_array.shape}")
        feature_indices = [known_features.index(feature) for feature in self.features_to_use]
        self.feature_indices_tensor = torch.tensor(feature_indices, dtype=torch.int64)
        # transfer full data to gpu
        self.tensor_full_data = torch.tensor(self.full_data_array, dtype=torch.float64, device=self.device)
        print("\nFull data array transferred to GPU")
        
        # subset file
        self.subset_filepath = 'data/datasets/train_dataset.csv'
        self.subset_array, subset_features = read_file_to_numpy(self.subset_filepath) 
        print(f"\nLoaded subset array of shape: {self.subset_array.shape}, dtype: {self.subset_array.dtype}")
        
        # select tensor subset of full data      
        self.selected_tensor, mask, bounds = apply_masks_gpu(tensor_data_array=self.tensor_full_data, window_sizes=self.window_sizes, subset_file=self.subset_filepath)
        self.selected_tensor = self.selected_tensor.to(self.device)
        print(f"Selected tensor of shape {self.selected_tensor.shape}, dtype: {self.selected_tensor.dtype}")
        
        # Build torch KDTree 
        print(f"\nBuilding torch KDTree...")
        self.gpu_tree = build_kd_tree(self.tensor_full_data[:, :3])
        print(f"Torch KDTree created successfully") 

        # select num_samples random indices to test on
        self.seed = 42
        random.seed(self.seed)
        self.num_samples = 100  
        self.random_indices = np.random.choice(self.full_data_array.shape[0], size=self.num_samples, replace=False)
        self.single_idx = self.random_indices[0]
        
        
    def test_vectorized_assign_features(self):
        # Parameters for the test
        batch_size = 1  # Use a subset of samples for testing
        center_points = self.selected_tensor[self.random_indices[:batch_size]]

        # Old implementation results
        grids_dict_old = {}
        for batch_idx, center_point in enumerate(center_points):
            for scale_idx, (_, window_size) in enumerate(self.window_sizes):
                # Generate the grid
                grid, _, x_coords, y_coords, z_coord = create_feature_grid_gpu(
                    center_point, self.device, window_size, self.grid_resolution, self.channels
                )

                # Assign features using the old function
                grid_with_features = assign_features_to_grid_gpu(
                    self.gpu_tree,
                    self.tensor_full_data,
                    grid,
                    x_coords,
                    y_coords,
                    z_coord,
                    self.feature_indices_tensor,
                    self.device,
                )

                # Store results for comparison
                grids_dict_old[(batch_idx, scale_idx)] = grid_with_features

        # New implementation results
        grids_new = vectorized_generate_multiscale_grids(
            center_points,
            self.tensor_full_data,
            self.tensor_window_sizes,
            self.grid_resolution,
            self.feature_indices_tensor,
            self.gpu_tree,
            device=self.device,
        )

        # Compare results
        for batch_idx in range(batch_size):
            for scale_idx in range(len(self.tensor_window_sizes)):
                # Extract old and new grids
                old_grid = grids_dict_old[(batch_idx, scale_idx)].permute(2, 0, 1)  # Permute old grid
                new_grid = grids_new[batch_idx, scale_idx]
                
                #print(f"old grid shape: [{old_grid.shape}, dtype:{old_grid.dtype}] VS new grid shape: [{new_grid.shape}, dtype:{new_grid.dtype}]")
                #print(f"Printing first 10 values of new grid at [batch_idx, scale_idx] = [{batch_idx}, {scale_idx}]\n{new_grid[:10]}")
                #print(f"\nPrinting first 10 values of old grid at [batch_idx, scale_idx] = [{batch_idx}, {scale_idx}]\n{old_grid[:10]}")

                # Assert shapes are the same
                self.assertEqual(
                    old_grid.shape, new_grid.shape, f"Shape mismatch for batch {batch_idx}, scale {scale_idx}"
                )
                
                visualize_grid(old_grid.cpu().numpy(), channel=0)
                visualize_grid(new_grid.cpu().numpy(), channel=0)

                # Assert values are close
                self.assertTrue(
                    torch.allclose(old_grid, new_grid, atol=1e-10),
                    f"Value mismatch for batch {batch_idx}, scale {scale_idx}",
                )
        


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


