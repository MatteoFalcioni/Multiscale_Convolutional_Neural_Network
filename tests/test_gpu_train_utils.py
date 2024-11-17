import unittest
from scripts.gpu_grid_gen import build_cuml_knn
from utils.gpu_training_utils import gpu_prepare_dataloader
from utils.train_data_utils import prepare_dataloader
from scripts.point_cloud_to_image import compute_point_cloud_bounds
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from torch.utils.data import Dataset, random_split
import cupy as cp
import numpy as np
import torch


class TestGPUGridGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sampled_las_path = 'tests/test_subtiler/32_687000_4930000_FP21_sampled_1k.las'
        cls.data_array, cls.known_features = read_file_to_numpy(cls.sampled_las_path)
        cls.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        cls.grid_resolution = 128
        cls.features_to_use = ['intensity', 'red', 'green', 'blue']
        cls.num_channels = len(cls.features_to_use)
        cls.point_cloud_bounds = compute_point_cloud_bounds(cls.data_array)

        cls.gpu_tree = build_cuml_knn(cls.data_array[:, :3])
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def test_gpu_dataloader(self):
        """
        Test that the GPU-based DataLoader works and returns grids correctly.
        """
        batch_size = 4
        train_loader, _ = gpu_prepare_dataloader(batch_size=batch_size, data_dir=self.sampled_las_path,
                                             window_sizes=self.window_sizes, grid_resolution=self.grid_resolution,
                                             features_to_use=self.features_to_use, shuffle_train=True, device='cuda')

        for small_grids, medium_grids, large_grids, labels, indices in train_loader:
            # Ensure that the grids are on the GPU
            self.assertEqual(small_grids.device.type, self.device, "Small grid is not on GPU")
            self.assertEqual(medium_grids.device.type, self.device, "Medium grid is not on GPU")
            self.assertEqual(large_grids.device.type, self.device, "Large grid is not on GPU")

            # Check that the grid shapes are correct
            self.assertEqual(small_grids.shape, (batch_size, self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid shape mismatch")
            self.assertEqual(medium_grids.shape, (batch_size, self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid shape mismatch")
            self.assertEqual(large_grids.shape, (batch_size, self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid shape mismatch")

            # Check that no values are zero in the grid (assuming that all cells should be filled)
            self.assertFalse(cp.all(small_grids == 0).item(), "Small grid contains zero cells")
            self.assertFalse(cp.all(medium_grids == 0).item(), "Medium grid contains zero cells")
            self.assertFalse(cp.all(large_grids == 0).item(), "Large grid contains zero cells")

            break  # Only test one batch for now

    def test_gpu_vs_cpu_dataloader(self):
        """
        Test if the GPU DataLoader behaves similarly to the CPU DataLoader.
        """
        # Prepare the CPU and GPU DataLoaders
        batch_size = 4
        cpu_train_loader, _ = prepare_dataloader(batch_size=batch_size, data_dir=self.sampled_las_path,
                                                    window_sizes=self.window_sizes, grid_resolution=self.grid_resolution,
                                                    features_to_use=self.features_to_use, shuffle_train=False, device=self.device)
        gpu_train_loader, _ = gpu_prepare_dataloader(batch_size=batch_size, data_dir=self.sampled_las_path,
                                                    window_sizes=self.window_sizes, grid_resolution=self.grid_resolution,
                                                    features_to_use=self.features_to_use, shuffle_train=False, device=self.device)

        # Compare one batch of CPU and GPU DataLoader
        for (cpu_small, cpu_medium, cpu_large, cpu_labels, cpu_indices), \
            (gpu_small, gpu_medium, gpu_large, gpu_labels, gpu_indices) in zip(cpu_train_loader, gpu_train_loader):

            # Check if grid shapes are the same
            self.assertEqual(cpu_small.shape, gpu_small.shape, "Grid shape mismatch between CPU and GPU")
            self.assertEqual(cpu_medium.shape, gpu_medium.shape, "Grid shape mismatch between CPU and GPU")
            self.assertEqual(cpu_large.shape, gpu_large.shape, "Grid shape mismatch between CPU and GPU")

            # Compare grid values to ensure they are identical (to a reasonable tolerance)
            np.testing.assert_almost_equal(cpu_small.cpu().numpy(), gpu_small.cpu().numpy(), decimal=5, 
                                            err_msg="Small grids do not match between CPU and GPU")
            np.testing.assert_almost_equal(cpu_medium.cpu().numpy(), gpu_medium.cpu().numpy(), decimal=5, 
                                            err_msg="Medium grids do not match between CPU and GPU")
            np.testing.assert_almost_equal(cpu_large.cpu().numpy(), gpu_large.cpu().numpy(), decimal=5, 
                                            err_msg="Large grids do not match between CPU and GPU")

            break  # Only test one batch