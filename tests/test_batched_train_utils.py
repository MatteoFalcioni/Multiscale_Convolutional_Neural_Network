import unittest
import numpy as np
from scipy.spatial import cKDTree
from scripts.point_cloud_to_image import generate_multiscale_grids, compute_point_cloud_bounds
from scripts.batched_pc_to_img import generate_batched_multiscale_grids
from utils.plot_utils import visualize_grid  

class TestBatchedGridGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample data setup
        cls.data_array = np.array([
            [1.0, 2.0, 3.0, 0.5, 0.6],  # point 1
            [4.0, 5.0, 6.0, 0.7, 0.8],  # point 2
            [7.0, 8.0, 9.0, 0.9, 1.0]   # point 3
        ])
        cls.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        cls.grid_resolution = 64
        cls.feature_indices = [3, 4]  # Assuming we use the last two features
        cls.kdtree = cKDTree(cls.data_array[:, :3])
        cls.point_cloud_bounds = compute_point_cloud_bounds(cls.data_array)

    def test_batched_vs_non_batched(self):
        visualize = True  # Set this to True to enable visualization
        
        for idx in range(len(self.data_array)):
            center_point = self.data_array[idx, :3]

            # Non-batched grid generation
            grids_dict_non_batched, skipped_non_batched = generate_multiscale_grids(
                center_point=center_point,
                data_array=self.data_array,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                feature_indices=self.feature_indices,
                kdtree=self.kdtree,
                point_cloud_bounds=self.point_cloud_bounds
            )

            # Batched grid generation (single-point batch)
            grids_dict_batched, skipped_batched = generate_batched_multiscale_grids(
                center_points=[center_point],
                data_array=self.data_array,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                feature_indices=self.feature_indices,
                kdtree=self.kdtree,
                point_cloud_bounds=self.point_cloud_bounds
            )

            # Check both skip flags
            self.assertEqual(skipped_non_batched, skipped_batched[0])

            if not skipped_non_batched:
                for scale, grid_non_batched in grids_dict_non_batched.items():
                    grid_batched = grids_dict_batched[scale][0]

                    # Assert that the grids match
                    np.testing.assert_allclose(
                        grid_non_batched,
                        grid_batched,
                        rtol=1e-5,
                        atol=1e-8,
                        err_msg=f"Mismatch in {scale} scale for point index {idx}"
                    )

                    '''# Optional visualization
                    if visualize:
                        save_path = f"tests/visualizations/{scale}_grid_point_{idx}.png"
                        visualize_grid(grid_non_batched, channel=0, title=f"{scale.capitalize()} Grid for Point {idx}",
                                       save=True, file_path=save_path)
                        print(f"Visualization saved at: {save_path}")'''

if __name__ == "__main__":
    unittest.main()
