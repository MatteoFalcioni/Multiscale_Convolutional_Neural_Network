import unittest
import os
import numpy as np
import pandas as pd
from data.transforms.data_loader import load_las_data, load_asc_data, visualize_dtm, load_las_features


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment. This method runs once before all tests.
        """
        # Define paths to sample test data files
        cls.base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
        cls.las_file_path = os.path.join(cls.base_data_dir, 'original.las')
        cls.asc_file_path = os.path.join(cls.base_data_dir, 'dtm.ASC')
        cls.features_las_file_path = os.path.join(cls.base_data_dir, 'features_F.las')

    def test_load_las_data(self):
        """
        Test if LAS file data is loaded correctly.
        """
        points = load_las_data(self.las_file_path)
        self.assertIsInstance(points, np.ndarray, "Output should be a NumPy array.")
        self.assertEqual(points.shape[1], 3, "Output should have 3 columns for XYZ coordinates.")
        self.assertGreater(points.shape[0], 0, "No points loaded from LAS file.")

    def test_load_asc_data(self):
        """
        Test if ASC file (DTM) data is loaded correctly.
        """
        dtm_data = load_asc_data(self.asc_file_path)
        self.assertIsInstance(dtm_data, np.ndarray, "Output should be a NumPy array.")
        self.assertGreater(dtm_data.shape[0], 0, "DTM data should have rows.")
        self.assertGreater(dtm_data.shape[1], 0, "DTM data should have columns.")

    def test_load_las_features(self):
        """
        Test if LAS file with features is loaded correctly.
        """
        features_df = load_las_features(self.features_las_file_path)
        self.assertIsInstance(features_df, pd.DataFrame, "Output should be a pandas DataFrame.")
        self.assertGreater(features_df.shape[0], 0, "No data loaded from LAS file with features.")

        # Check that all expected features are present in the DataFrame
        expected_columns = [
            'x', 'y', 'z', 'intensity', 'ndvi', 'ndwi', 'ssi',
            'l1_a', 'l2_a', 'l3_a', 'planarity_a', 'sphericity_a', 'linearity_a',
            'entropy_a', 'theta_a', 'theta_variance_a', 'mad_a', 'delta_z_a',
            'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b',
            'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
            'N_h', 'delta_z_fl'
        ]

        for col in expected_columns:
            self.assertIn(col, features_df.columns, f"Missing feature '{col}' in loaded DataFrame.")

    def test_visualize_dtm(self):
        """
        Test if the DTM visualization function runs without errors.
        """
        dtm_data = load_asc_data(self.asc_file_path)

        try:
            visualize_dtm(dtm_data)
            success = True
        except Exception as e:
            success = False
            print(f"Error visualizing DTM: {e}")

        self.assertTrue(success, "DTM visualization should run without errors.")

    def test_missing_file(self):
        """
        Test error handling for a missing file.
        """
        missing_file_path = os.path.join(self.base_data_dir, 'non_existent_file.las')

        with self.assertRaises(FileNotFoundError):
            load_las_data(missing_file_path)

