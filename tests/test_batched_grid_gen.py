import unittest
import numpy as np
from scipy.spatial import cKDTree
from scripts.point_cloud_to_image import compute_point_cloud_bounds, generate_multiscale_grids
from scripts.batched_pc_to_img import generate_batched_multiscale_grids

class TestBatchedMultiscaleGridGeneration(unittest.TestCase):
    def setUp(self):
        # Generate mock data
        np.random.seed(42)
        self.data_array = np.random.rand(1000, 6)  # 1000 points, x, y, z + 3 features
        self.kdtree = cKDTree(self.data_array[:, :3])
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.grid_resolution = 128
        self.feature_indices = [3, 4, 5]  # Use last 3 columns as features
        self.point_cloud_bounds = compute_point_cloud_bounds(self.data_array)

        # Select center points for testing
        self.center_points = self.data_array[:5, :3]  # First 5 points as test center points

    def test_batched_vs_non_batched_grids(self):
        # Run non-batched version on each center point individually
        non_batched_results = []
        for center_point in self.center_points:
            grids, skipped = generate_multiscale_grids(center_point, self.data_array, self.window_sizes,
                                                       self.grid_resolution, self.feature_indices,
                                                       self.kdtree, self.point_cloud_bounds)
            non_batched_results.append((grids, skipped))

        # Run batched version
        batched_grids, batched_skipped = generate_batched_multiscale_grids(self.center_points, self.data_array,
                                                                           self.window_sizes, self.grid_resolution,
                                                                           self.feature_indices, self.kdtree,
                                                                           self.point_cloud_bounds)

        # Compare batched and non-batched results
        for i, (non_batched_grid, non_batched_skipped) in enumerate(non_batched_results):
            for size_label in self.window_sizes:
                if not non_batched_skipped:
                    print('checking non skipped point...')
                    self.assertTrue(np.array_equal(non_batched_grid[size_label], batched_grids[size_label][i]),
                                    f"Mismatch in grid at center point {i} for scale {size_label}")
            self.assertEqual(non_batched_skipped, batched_skipped[i], f"Mismatch in skip status at center point {i}")
