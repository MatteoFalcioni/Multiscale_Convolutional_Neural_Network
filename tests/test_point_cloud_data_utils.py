from utils.point_cloud_data_utils import load_las_data, load_asc_data, read_las_file_to_numpy, numpy_to_dataframe
from utils.point_cloud_data_utils import combine_and_save_csv_files, subtiler, stitch_subtiles
import os
import pandas as pd
import unittest
import numpy as np
import laspy
import re


'''class TestPointCloudDataUtils(unittest.TestCase):

    def setUp(self):
        self.las_directory = 'data/raw'
        self.asc_file_path = 'data/raw/dtm.asc'
        self.features_to_extract = ['x', 'y', 'z',  'intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir',
                                    'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                                    'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
                                    'N_h', 'delta_z_fl']
        self.csv_files = ['data/classes/buildings.csv', 'data/classes/cars.csv', 'data/classes/grass.csv',
                          'data/classes/rail.csv', 'data/classes/roads.csv', 'data/classes/trees.csv']
        self.csv_save_dir = 'data/combined_data'  # Directory to save combined data

    def test_load_las_data(self):
        # Test loading LAS data
        las_files = [f for f in os.listdir(self.las_directory) if f.endswith('.las')]
        if las_files:
            points = load_las_data(os.path.join(self.las_directory, las_files[0]))
            self.assertIsInstance(points, np.ndarray, "Loaded LAS data is not a numpy array.")
            self.assertEqual(points.shape[1], 3, "LAS data does not have three coordinate columns.")
        else:
            self.skipTest("No LAS files found for testing `load_las_data`.")

    def test_load_asc_data(self):
        # Test loading ASC data
        try:
            dtm_data = load_asc_data(self.asc_file_path)
            self.assertIsInstance(dtm_data, np.ndarray, "Loaded ASC data is not a numpy array.")
            self.assertGreater(dtm_data.shape[0], 0, "Loaded ASC data is unexpectedly empty.")
        except FileNotFoundError:
            self.skipTest("ASC file not found for testing `load_asc_data`.")

    def test_read_las_file_to_numpy(self):
        # Test the read_las_file_to_numpy function
        sample_file = os.path.join(self.las_directory, 'features_F.las')
        numpy_array, feature_names = read_las_file_to_numpy(sample_file, features_to_extract=self.features_to_extract)

        # Ensure the result is a NumPy array
        self.assertIsInstance(numpy_array, np.ndarray, "Result is not a NumPy array.")

        # Ensure the shape of the array is as expected (at least 4 columns: x, y, z, features)
        self.assertGreaterEqual(numpy_array.shape[1], 4, "NumPy array does not have the expected number of columns.")
        self.assertGreater(len(feature_names), 0, "No feature names returned.")

        # Check that the first three feature names are 'x', 'y', 'z'
        self.assertEqual(feature_names[:3], ['x', 'y', 'z'], "The first three features should be 'x', 'y', 'z'.")

        # Ensure that all specified features to extract are either in the list or not available
        for feature in self.features_to_extract:
            if feature in ['x', 'y', 'z']:
                continue
            self.assertIn(feature, feature_names, f"Feature '{feature}' is not found in the returned feature names.")

        # Check some values to ensure data was loaded correctly (sample values for the first few points)
        print("Sample data from NumPy array (first 5 rows):")
        print(numpy_array[:5])
        print("Feature names extracted:")
        print(feature_names)

        # check that segment_id and label are correctly extracted from a labeled sample file
        labeled_file = os.path.join(self.las_directory, 'labeled_FSL.las')
        lbl_array, feature_names_lbl = read_las_file_to_numpy(labeled_file, features_to_extract=['x', 'y', 'z', 'segment_id', 'label'])
        self.assertIn('segment_id', feature_names_lbl, "'segment_id' is not found in the returned feature names.")
        self.assertIn('label', feature_names_lbl, "'label' is not found in the returned feature names.")

    def test_numpy_to_dataframe(self):
        sample_file = os.path.join(self.las_directory, 'features_F.las')
        numpy_array, feature_names = read_las_file_to_numpy(sample_file, features_to_extract=self.features_to_extract)

        # Convert numpy array to DataFrame
        df = numpy_to_dataframe(numpy_array, feature_names=self.features_to_extract)

        # Check the result is a DataFrame
        self.assertIsInstance(df, pd.DataFrame, "The result is not a pandas DataFrame.")

        # Check the DataFrame has the correct number of rows and columns
        self.assertEqual(df.shape[0], numpy_array.shape[0], "The number of rows in the DataFrame does not match the input NumPy array.")
        self.assertEqual(df.shape[1], numpy_array.shape[1], "The number of columns in the DataFrame does not match the input NumPy array.")

        # Check the DataFrame columns have the expected names
        self.assertListEqual(list(df.columns), feature_names, "The DataFrame columns do not match the expected names.")

        # Check if the values are correctly copied from NumPy array
        np.testing.assert_array_almost_equal(df.values, numpy_array, decimal=6, err_msg="The DataFrame values do not match the original NumPy array values.")

        # Print sample data for manual verification
        print("Sample data from DataFrame (first 5 rows):")
        print(df.head())

    def test_combine_and_save_csv_files(self):
        """ Test combining and saving CSV files into a single NumPy array. """

        # Test combining without saving
        combined_data = combine_and_save_csv_files(self.csv_files, save=False)

        # Check that the result is a NumPy array
        self.assertIsInstance(combined_data, np.ndarray, "The result is not a NumPy array.")

        # Ensure that the array has some rows
        self.assertGreater(combined_data.shape[0], 0, "Combined array is unexpectedly empty.")

        # Test combining with saving
        combined_data_saved = combine_and_save_csv_files(self.csv_files, save=True, save_dir=self.csv_save_dir)

        # Ensure the combined data was saved
        saved_file_path = os.path.join(self.csv_save_dir, 'combined_data.npy')
        self.assertTrue(os.path.exists(saved_file_path), "Combined data was not saved correctly.")

        # Load the saved file and compare it with the original combined data
        loaded_data = np.load(saved_file_path)
        np.testing.assert_array_almost_equal(combined_data_saved, loaded_data, decimal=6,
                                             err_msg="Loaded data does not match the originally combined data.")
'''

class TestSubtiler(unittest.TestCase):

    def setUp(self):
        # Set up test directory and parameters
        self.input_file = 'tests/test_subtiler/32_687000_4930000_FP21.las'  
        self.tile_size = 125
        self.overlap_size = 30
        self.output_dir = '.'
        self.final_output_dir = 'tests/test_subtiler/test_cut_and_stitch'

    """def test_subtiling(self):
        
        # Call the subtiler function
        subtile_folder = subtiler(self.input_file, self.tile_size, self.overlap_size)
        self.output_dir = subtile_folder
        
        # Verify the output folder exists
        self.assertTrue(os.path.exists(subtile_folder), "Subtile folder should exist.")
        
        # Get all subtiles from the folder
        subtile_files = [os.path.join(subtile_folder, f) for f in os.listdir(subtile_folder) if f.endswith('.las')]
        
        # Verify that there are subtiles
        self.assertGreater(len(subtile_files), 0, "There should be at least one subtile.")
        
        # Load original LAS file to compare coordinates
        original_las = laspy.read(self.input_file)
        original_points = original_las.points
        original_x, original_y = original_points['X'], original_points['Y']
        
        # Check that the coordinates of points in subtiles match expected regions
        for subtile_file in subtile_files:
            # Load the subtile
            subtile_las = laspy.read(subtile_file)
            subtile_points = subtile_las.points
            subtile_x, subtile_y = subtile_points['X'], subtile_points['Y']
            
            # Extract the lower-left coordinates of the subtile
            parts = os.path.basename(subtile_file).split('_')
            subtile_lower_left_x = int(parts[-2])
            subtile_lower_left_y = int(parts[-1].split('.')[0])
            
            subtile_upper_bound_x = subtile_lower_left_x + self.tile_size
            subtile_upper_bound_y = subtile_lower_left_y + self.tile_size
            # Take overlap into account by adjusting the bounds
            subtile_lower_left_x_with_overlap = subtile_lower_left_x - self.overlap_size
            subtile_lower_left_y_with_overlap = subtile_lower_left_y - self.overlap_size
            subtile_upper_bound_x_with_overlap = subtile_upper_bound_x + self.overlap_size
            subtile_upper_bound_y_with_overlap = subtile_upper_bound_y + self.overlap_size
            
            tol = 1e3
            # Check that the points in the subtile are within the expected region
            for x, y in zip(subtile_x, subtile_y):
                self.assertTrue(subtile_lower_left_x_with_overlap * tol <= x < subtile_upper_bound_x_with_overlap * tol, "X coordinate mismatch")
                self.assertTrue(subtile_lower_left_y_with_overlap * tol <= y < subtile_upper_bound_y_with_overlap * tol, "Y coordinate mismatch")
        
        # Optionally, check the total points in subtiles
        total_points_in_subtiles = sum([len(laspy.read(f).points) for f in subtile_files])
        self.assertEqual(total_points_in_subtiles, len(original_points), "Total points in subtiles do not match original points.")
        """
    def test_stitching(self):
        las_file = laspy.read(self.input_file)

        subtile_folder = subtiler(self.input_file, self.tile_size, self.overlap_size)
        
        stitch_subtiles(subtile_folder=subtile_folder, original_las=las_file, original_filename=self.input_file, model_directory=self.final_output_dir, overlap_size=self.overlap_size)

        '''output_pattern = re.compile(r".+_pred_\d{8}_\d{6}\.las$")
        output_dir = os.path.join('tests/test_subtiler/', 'inference', 'predictions')

        # Check if any file in the directory matches the pattern
        output_files = [f for f in os.listdir(output_dir) if output_pattern.match(f)]
        assert len(output_files) > 0, "No stitched LAS file found with the expected pattern."

        # Read the stitched file and check for overlapping points
        stitched_las = laspy.read(output_files[0])
        unique_points, unique_indices = np.unique(np.column_stack((stitched_las.x, stitched_las.y, stitched_las.z)), axis=0, return_index=True)

        # Ensure that duplicate points are handled correctly (i.e., no duplicate points left)
        assert len(unique_points) == len(stitched_las.x) - 1, "Duplicate points found in stitched file."

        print("Stitching test passed on real data!")'''

        
