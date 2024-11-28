import unittest
import os
import shutil
from collections import defaultdict
from utils.create_dataset import pair_ground_and_offgrounds, stitch_pairs, train_test_split
from utils.point_cloud_data_utils import las_to_csv, read_file_to_numpy, combine_csv_files
import laspy
import numpy as np
import pandas as pd


class TestCreateDataset(unittest.TestCase):
    def setUp(self):
        
        self.real_file_dir = 'data/ground_and_offground'
        self.real_file_directories = ['data/ground_and_offground/32_681500', 'data/ground_and_offground/32_684000', 'data/ground_and_offground/32_686000']    #
        self.las_dir_out = 'tests/fused_las'
        self.fused_files = [filepath for filepath in self.las_dir_out if filepath.endswith('.las')]
        os.makedirs(self.las_dir_out, exist_ok=True)
        
        
    def test_pair_ground_and_offgrounds(self):
        # Call the function to test
        directories = self.real_file_directories
        file_pairs = pair_ground_and_offgrounds(input_folders=directories)
        
        print(f"number of files in pairs: {len(file_pairs)}")

        
    def test_stitch_pairs(self):
        # Step 1: Generate file pairs
        file_pairs = pair_ground_and_offgrounds(self.real_file_directories)
        self.assertGreater(len(file_pairs), 0, "No file pairs generated.")

        # Step 2: Stitch pairs and create fused LAS files
        fused_files = stitch_pairs(file_pairs, self.las_dir_out)
        self.assertEqual(len(fused_files), len(file_pairs), "Number of fused files does not match the number of pairs.")

        # Step 3: Verify the output LAS files
        for fused_file, (ground_file, off_ground_files) in zip(fused_files, file_pairs):
            # Read the fused file
            fused_las = laspy.read(fused_file)

            # Read the ground file
            ground_las = laspy.read(ground_file)

            # Read and concatenate all off-ground files
            all_off_ground_points = []
            for off_ground_file in off_ground_files:
                off_ground_las = laspy.read(off_ground_file)
                off_ground_points = np.vstack((off_ground_las.x, off_ground_las.y, off_ground_las.z)).T
                all_off_ground_points.append(off_ground_points)
            all_off_ground_points = np.concatenate(all_off_ground_points, axis=0)

            # Concatenate ground and off-ground points
            ground_points = np.vstack((ground_las.x, ground_las.y, ground_las.z)).T
            expected_points = np.vstack((ground_points, all_off_ground_points))

            # Verify point data in the fused file
            fused_points = np.vstack((fused_las.x, fused_las.y, fused_las.z)).T
            np.testing.assert_allclose(
                fused_points, expected_points, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch in fused points in file {fused_file}."
            )

            print(f"Verified fused file: {fused_file}")

        print("All fused files verified successfully!")
        
    def test_las_to_csv(self):
        # Use one of the fused LAS files for testing
        fused_file = self.fused_files[0]
        
        # Convert the LAS file to CSV
        csv_file = las_to_csv(las_file=fused_file, output_folder=self.csv_subdir)

        # Check if the CSV file was created
        self.assertTrue(os.path.exists(csv_file), f"CSV file not created: {csv_file}")
        
        # Load the LAS file and CSV file
        las_data, known_features = read_file_to_numpy(fused_file)
        csv_data = pd.read_csv(csv_file)

        # Verify that the number of points and features match
        self.assertEqual(len(csv_data), len(las_data), "Number of points in CSV does not match LAS file.")
        self.assertEqual(list(csv_data.columns), known_features, "CSV column names do not match LAS features.")

        print(f"CSV conversion verified for: {fused_file}")
        
    def test_combine_csv_files(self):
        # Convert LAS files to CSV
        csv_filepaths = []
        for fused_file in self.fused_files:
            csv_file = las_to_csv(las_file=fused_file, output_folder=self.csv_subdir)
            csv_filepaths.append(csv_file)

        # Combine the CSV files
        combined_csv_path = os.path.join(self.output_dataset_folder, "full_dataset.csv")
        combined_csv = combine_csv_files(csv_filepaths, output_csv=combined_csv_path)

        # Check if the combined CSV file was created
        self.assertTrue(os.path.exists(combined_csv), f"Combined CSV file not created: {combined_csv}")

        # Load the individual CSVs and the combined CSV
        individual_dataframes = [pd.read_csv(csv) for csv in csv_filepaths]
        combined_dataframe = pd.read_csv(combined_csv)

        # Verify that the combined CSV contains all points from the individual CSVs
        total_points = sum(len(df) for df in individual_dataframes)
        self.assertEqual(len(combined_dataframe), total_points, "Combined CSV does not contain all points.")

        print(f"CSV combination verified. Combined file saved at: {combined_csv_path}")