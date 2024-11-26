import unittest
import numpy as np
from utils.point_cloud_data_utils import numpy_to_dataframe, read_file_to_numpy
from scripts.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_multiscale_grids, compute_point_cloud_bounds
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
from scipy.spatial import cKDTree as KDTree
import os
from tqdm import tqdm


class TestPointCloudToImage(unittest.TestCase):

    def setUp(self):
        self.file_path = 'tests/test_subtiler/32_687000_4930000_FP21.las'# 'data/training_data/test_21.csv'
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']  # selected features
        self.channels = len(self.features_to_use)  # Number of channels based on selected features

        # Load LAS file and get data with user-selected features
        self.full_data, self.feature_names = read_file_to_numpy(data_dir=self.file_path, features_to_use=self.features_to_use)
        
        # Randomly sample num_points from the point cloud
        np.random.seed(42)  # For reproducibility
        num_points = int(5*1e4)
        random_indices = np.random.choice(self.full_data.shape[0], num_points, replace=False)
        self.sliced_data = self.full_data[random_indices, :]
        
        print(f'\nfeature names in test file: {self.feature_names}')
        self.feature_indices = [self.feature_names.index(feature) for feature in self.features_to_use]
        self.tree = KDTree(self.full_data[:, :3])
        self.point_cloud_bounds = compute_point_cloud_bounds(self.full_data)

        self.df = numpy_to_dataframe(feature_names=self.feature_names, data_array=self.full_data)

        '''# to do only if using las file without labels:
        num_points = self.full_data.shape[0]
        labels = np.random.randint(0, 5, size=num_points)
        # Append labels as a new column
        self.full_data = np.hstack((self.full_data, labels.reshape(-1, 1)))'''
        
        self.sample_size = 1000  # Subset for testing
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]
        self.idx = 100000

        # Define the window sizes for multiscale grids
        self.window_sizes = [('small', 1.0), ('medium', 2.0), ('large', 3.0)]

        self.save_imgs_bool = False  # if True, save the generated images
        self.save_imgs_dir = 'tests/test_feature_imgs'   # directory to save test images
        os.makedirs(self.save_imgs_dir, exist_ok=True)
        

    def test_create_and_assign_grids(self):

        # Check that data is not empty and has the expected structure
        self.assertIsInstance(self.full_data, np.ndarray)
        self.assertGreaterEqual(self.full_data.shape[1], 4)  # At least x, y, z, and one feature
        self.assertIsInstance(self.sliced_data, np.ndarray)
        self.assertGreaterEqual(self.sliced_data.shape[1], 4)  # At least x, y, z, and one feature

        test_window_size = 10.0
        out_of_bounds = 0

        for center_point in tqdm(self.sliced_data, desc="testing grid creation and feat. assignment", unit="processed points"):

            # Check if the center point is within the point cloud bounds
            half_window = test_window_size / 2
            if (center_point[0] - half_window < self.point_cloud_bounds['x_min'] or 
                center_point[0] + half_window > self.point_cloud_bounds['x_max'] or
                center_point[1] - half_window < self.point_cloud_bounds['y_min'] or
                center_point[1] + half_window > self.point_cloud_bounds['y_max']):
                # print(f"Out of bounds skip")
                out_of_bounds += 1
                continue

            # Create a grid around the center point
            grid, _, x_coords, y_coords, z_coord = create_feature_grid(
                center_point, window_size=test_window_size, grid_resolution=self.grid_resolution, channels=self.channels
            )
            
            # Check for NaN or Inf in the grid right after creation
            self.assertFalse(np.isnan(grid).any(), "NaN values found in grid creation.")
            self.assertFalse(np.isinf(grid).any(), "Inf values found in grid creation.")

            # Ensure grid has the correct shape
            self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))

            # Assign features using the pre-built KDTree
            grid_with_features = assign_features_to_grid(self.tree, self.full_data, grid, x_coords, y_coords, z_coord, self.feature_indices)
            grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

            # Check how many grid cells are still zero after assigning features
            non_zero_cells = np.count_nonzero(grid_with_features)
            total_cells = grid_with_features.size
            zero_cells_percentage = (total_cells - non_zero_cells) / total_cells * 100
            if zero_cells_percentage > 0:
                print(f"WARNING: percentage of zero cells in the grid: {zero_cells_percentage}%. This could be due to the use of a subsample of the point cloud\
                      during testing. Be sure to test with a full point cloud, and eventually slice the array to test on less points.\
                      subsampling leads to sparse poit clouds and deprecated results!")
                grid_with_features = np.transpose(grid_with_features, (2, 0, 1))
                for chan in range(self.channels):
                    feature_name = self.features_to_use[chan] 
                    visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {feature_name}", save=self.save_imgs_bool)
                    visualize_grid_with_comparison(grid_with_features, self.df, center_point, window_size=test_window_size, feature_names=self.feature_names,
                                       channel=chan, visual_size=30, save=self.save_imgs_bool)

            # Ensure features are assigned (grid should not be all zeros)
            self.assertFalse(np.all(grid_with_features == 0), "Grid is unexpectedly empty or all zeros.")

            # Check that no NaN or Inf values exist in the assigned grid
            self.assertFalse(np.isnan(grid_with_features).any(), "NaN values found in grid features.")
            self.assertFalse(np.isinf(grid_with_features).any(), "Inf values found in grid features.")

            self.assertFalse(np.any(grid_with_features == 0), "Assigned grid cells are unexpectedly empty.")
            
        print(f"Total number of out of bounds points: {out_of_bounds}")

        # Transpose the grid to match PyTorch's 'channels x height x width' format for visualization
        # grid_with_features = np.transpose(grid_with_features, (2, 0, 1))
        # Visualize and eventually save feature images (if save = True)
        '''for chan in range(0, self.channels):
            # Create a filename for saving the image
            feature_name = self.feature_names[3 + chan] if len(self.feature_names) > 3 + chan else f"Channel_{chan}"
            file_path = os.path.join(self.save_imgs_dir, f"Grid_Visual_window{int(self.window_size)}_{feature_name}.png")

            # Visualize and save the grid image
            print(f"Grid shape to be visualized: {grid_with_features.shape}")
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {feature_name}", save=self.save_imgs_bool, file_path=file_path)'''

        '''# visualize and eventually save feature image compared with point cloud
        chosen_chan = 3  # channel to visualize on feature image (8=nir)
        visualize_grid_with_comparison(grid_with_features, self.df, center_point, window_size=self.window_size, feature_names=self.feature_names,
                                       channel=chosen_chan, visual_size=50, save=self.save_imgs_bool, file_path=file_path)'''
        

    def test_kd_tree(self):
        """
        Test that the KDTree is constructed properly and that KDTree queries return valid indices.
        """

        num_points = len(self.full_data)
        # Verify the KDTree contains all points
        self.assertEqual(self.tree.n, num_points, "KDTree does not contain all points from the dataset.")

        test_window_size = 10.0
        
        # number of out of bounds points
        out_of_bounds = 0

        for center_point in tqdm(self.sliced_data, desc="Testing KD tree", unit="processed points"):

            # Check if the center point is within the point cloud bounds
            half_window =test_window_size / 2
            if (center_point[0] - half_window < self.point_cloud_bounds['x_min'] or 
                center_point[0] + half_window > self.point_cloud_bounds['x_max'] or
                center_point[1] - half_window < self.point_cloud_bounds['y_min'] or
                center_point[1] + half_window > self.point_cloud_bounds['y_max']):
                # print(f"Out of bounds skip")
                out_of_bounds += 1
                continue

            # Create grid and query KDTree for nearest neighbors
            grid, _, x_coords, y_coords, z_coord = create_feature_grid(
                center_point, window_size=test_window_size, grid_resolution=self.grid_resolution, channels=self.channels
            )

            # Flatten grid coordinates to query the KDTree
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
            grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, z_coord)), axis=-1)

            # Query KDTree for nearest points
            distances, indices = self.tree.query(grid_coords)

            # Ensure no NaN or Inf values are returned by the KDTree query
            self.assertFalse(np.isnan(indices).any(), "KDTree query returned NaN indices.")
            self.assertFalse(np.isinf(indices).any(), "KDTree query returned Inf indices.")

            # Check that the queried indices are within valid range
            self.assertTrue(np.all(indices >= 0), "KDTree query returned invalid (negative) indices.")
            self.assertTrue(np.all(indices < len(self.full_data)), "KDTree query returned out-of-bounds indices.")

            # Check for edge cases (e.g., near the edges of the grid)
            edge_x = grid_x[0, 0]
            edge_y = grid_y[0, 0]
            edge_coord = np.array([edge_x, edge_y, z_coord])
            _, edge_index = self.tree.query(edge_coord)

            self.assertGreaterEqual(edge_index, 0, "KDTree returned invalid index for edge case.")
            self.assertLess(edge_index, len(self.full_data), "KDTree returned out-of-bounds index for edge case.")

            # Ensure valid distances (no NaN/Inf in distances)
            self.assertFalse(np.isnan(distances).any(), "KDTree returned NaN distances.")
            self.assertFalse(np.isinf(distances).any(), "KDTree returned Inf distances.")
            
        print(f"total number of out of bound points: {out_of_bounds}")
    
    
    def test_generate_multiscale_grids(self):
        print('Testing multiscale grid generation')
        
        num_points = len(self.sliced_data)
        not_skipped_points = 0
        
        for point in tqdm(self.sliced_data, desc="Testing multiscale grids", unit="processed points"):

            grids_dict, skipped = generate_multiscale_grids(
                center_point=point,
                data_array=self.full_data,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                feature_indices=self.feature_indices,
                kdtree=self.tree,  # Pass the prebuilt KDTree
                point_cloud_bounds=self.point_cloud_bounds
            )

            if not skipped:
                not_skipped_points += 1
                # Verify that grids for each scale are generated and not empty
                for scale_label, _ in self.window_sizes:
                    self.assertIn(scale_label, grids_dict, f"Scale {scale_label} is missing from the grids_dict.")
                    self.assertIsNotNone(grids_dict[scale_label], f"Grid for scale {scale_label} is None.")
                    self.assertGreater(grids_dict[scale_label].size, 0, f"No grids generated for scale {scale_label}.")
                    
                    # Check that the generated grids have valid values
                    grid = grids_dict[scale_label]
                    self.assertFalse(np.isnan(grid).any(), f"Grid for scale {scale_label} contains NaN values.")
                    self.assertFalse(np.isinf(grid).any(), f"Grid for scale {scale_label} contains Inf values.")
                    self.assertGreater(np.count_nonzero(grid), 0, f"Grid for scale {scale_label} is all zeros.")
                    
        print(f"Total number of skipped points: {num_points - not_skipped_points} ; percentage of skipped points: {num_points - not_skipped_points} / {num_points}")

