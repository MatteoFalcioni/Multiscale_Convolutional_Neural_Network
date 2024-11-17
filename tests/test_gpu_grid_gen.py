import unittest
import numpy as np
from utils.point_cloud_data_utils import read_file_to_numpy
from utils.plot_utils import visualize_grid  # Assuming this is your visualization function
from scripts.point_cloud_to_image import generate_multiscale_grids
from scripts.gpu_grid_gen import generate_multiscale_grids_gpu, build_cuml_knn
from scripts.point_cloud_to_image import compute_point_cloud_bounds
import cupy as cp
from scipy.spatial import cKDTree 


class TestGridGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This will run once before any tests.
        """
        cls.sampled_las_path = 'tests/test_subtiler/32_687000_4930000_FP21_sampled_1k.las'
        cls.data_array, cls.known_features = read_file_to_numpy(cls.sampled_las_path)
        cls.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        cls.grid_resolution = 128
        cls.features_to_use = ['intensity', 'red', 'green', 'blue']  # Adjust based on available features
        cls.point_cloud_bounds = compute_point_cloud_bounds(cls.point_cloud_data)  # Compute bounds

        # Initialize CPU and GPU KNN models (we'll use a small subset for testing)
        cls.cpu_kdtree = cKDTree(cls.point_cloud_data[:, :3])  # Build KDTREE for CPU
        cls.gpu_tree = build_cuml_knn(cls.point_cloud_data[:, :3])  # Build cuML KNN model for GPU

    def test_gpu_grid_generation(self):
        """
        Test that the GPU grid generation works correctly for a subset of points.
        """
        # Take a small subset of points for testing
        points_to_test = self.data_array[:50]  # Testing with first 5 points

        for center_point in points_to_test:
            # Generate grids using the GPU pipeline
            gpu_grids, _ = generate_multiscale_grids_gpu(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.features_to_use,
                cuml_knn=self.gpu_tree, point_cloud_bounds=self.point_cloud_bounds
            )

            # Check if the generated grids contain any NaN or Inf values
            for scale in ['small', 'medium', 'large']:
                grid = gpu_grids[scale]

                # Check if there are any NaN or Inf values in the grid
                self.assertFalse(cp.isnan(grid).any(), f"NaN found in {scale} grid for point {center_point}")
                self.assertFalse(cp.isinf(grid).any(), f"Inf found in {scale} grid for point {center_point}")

                # Check if the grid has the expected shape (C, H, W)
                self.assertEqual(grid.shape, (4, self.grid_resolution, self.grid_resolution),  # Assuming 4 channels (features)
                                 f"Grid shape mismatch for {scale} grid at point {center_point}")

                # Check if there are no zero cells in the grid (i.e., no cells without assigned features)
                self.assertFalse(cp.all(grid == 0), f"Grid contains zero cells for {scale} grid at point {center_point}")
    

    def test_generate_grids_cpu_vs_gpu(self):
        """
        Test if grids generated on CPU match with those generated on GPU.
        """
        # Take a small subset of points for comparison
        points_to_test = self.point_cloud_data[:50]  # Testing with first 50 points

        for center_point in points_to_test:
            # Generate grids using the CPU pipeline
            cpu_grids, skipped_cpu = generate_multiscale_grids(
                center_point, data_array=self.point_cloud_data, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.features_to_use,
                kdtree=self.cpu_kdtree, point_cloud_bounds=self.point_cloud_bounds
            )

            # Generate grids using the GPU pipeline
            gpu_grids, skipped_gpu = generate_multiscale_grids_gpu(
                center_point, data_array=self.point_cloud_data, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.features_to_use,
                cuml_knn=self.gpu_tree, point_cloud_bounds=self.point_cloud_bounds
            )

            # Compare the grids from CPU and GPU (for the same point)
            for scale in ['small', 'medium', 'large']:
                cpu_grid = cpu_grids[scale]
                gpu_grid = gpu_grids[scale]

                # Check if the grids are approximately equal
                np.testing.assert_almost_equal(cpu_grid, gpu_grid, decimal=8, 
                                               err_msg=f"Grids do not match for {scale} scale at point {center_point}")

if __name__ == '__main__':
    unittest.main()
