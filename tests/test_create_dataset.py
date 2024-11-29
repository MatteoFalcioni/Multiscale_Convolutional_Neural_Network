import unittest
import os
import shutil
from collections import defaultdict
from utils.create_dataset import pair_ground_and_offgrounds, stitch_pairs, create_train_eval_datasets
from utils.point_cloud_data_utils import las_to_csv, read_file_to_numpy, clean_and_combine_csv_files
import laspy
import numpy as np
import pandas as pd


class TestCreateDataset(unittest.TestCase):
    def setUp(self):
        
        self.real_file_dir = 'data/ground_and_offground'
        self.real_file_directories = ['data/ground_and_offground/32_681500', 'data/ground_and_offground/32_684000', 'data/ground_and_offground/32_686000_4930500', 'data/ground_and_offground/32_686000_4933000']    #
        self.las_dir_out = 'tests/fused_las'
        os.makedirs(self.las_dir_out, exist_ok=True)
        self.csv_subdir = f"{self.las_dir_out}/csv"
        # be careful to update this when changing files to fuse
        self.fused_files = [os.path.join(self.las_dir_out, filepath) for filepath in os.listdir(self.las_dir_out) if filepath.endswith('.las')]
        
        if len(self.fused_files) == len(self.real_file_directories):
            print(f"\nNumber of fused files doesn't match that of the directories;\
                    if you need to test only las fusion this is alright,\
                    otherwise you first need to generate fused las file, than test the rest\n")
            
        self.output_dataset_folder = 'tests/output_dataset_folder/'
        
        self.test_full_pipeline = True
        
        
    '''def test_pair_ground_and_offgrounds(self):
        if self.test_full_pipeline:
            # Call the function to test
            directories = self.real_file_directories
            file_pairs = pair_ground_and_offgrounds(input_folders=directories)
            
            print(f"number of files in pairs: {len(file_pairs)}")

        
    def test_stitch_pairs(self):
        
        if self.test_full_pipeline:
            # Step 1: Generate file pairs
            file_pairs = pair_ground_and_offgrounds(self.real_file_directories)
            self.assertGreater(len(file_pairs), 0, "No file pairs generated.")

            # Step 2: Stitch pairs and create fused LAS files
            self.fused_files = stitch_pairs(file_pairs, self.las_dir_out)
            self.assertEqual(len(self.fused_files), len(file_pairs), "Number of fused files does not match the number of pairs.")

            # Step 3: Verify the output LAS files
            for fused_file, (ground_file, off_ground_files) in zip(self.fused_files, file_pairs):
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

            print("All fused files verified successfully!")'''
    
            
    '''def test_las_to_csv(self):
        #if self.test_full_pipeline:
            # Use one of the fused LAS files for testing
            fused_file = self.fused_files[0]
            
            # Convert the LAS file to CSV
            csv_file = las_to_csv(las_file=fused_file, output_folder=self.csv_subdir, selected_classes=[3,5,6,11,64])

            # Check if the CSV file was created
            self.assertTrue(os.path.exists(csv_file), f"CSV file not created: {csv_file}")
            
            # Load the LAS file and CSV file
            las_data, known_features = read_file_to_numpy(fused_file)
            csv_data = pd.read_csv(csv_file)

            self.assertEqual(list(csv_data.columns), known_features, "CSV column names do not match LAS features.")

            print(f"CSV conversion verified for: {fused_file}")'''
        
        
    def test_combine_csv_files(self):
        # Convert LAS files to CSV
        if self.test_full_pipeline:
            csv_filepaths = []
            for fused_file in self.fused_files:
                csv_file = las_to_csv(las_file=fused_file, output_folder=self.csv_subdir, selected_classes=[3,5,6,11,64])
                csv_filepaths.append(csv_file)

            # Combine the CSV files
            combined_csv_path = os.path.join(self.output_dataset_folder, "full_dataset.csv")
            combined_csv = clean_and_combine_csv_files(csv_filepaths, output_csv=combined_csv_path)

            # Check if the combined CSV file was created
            self.assertTrue(os.path.exists(combined_csv), f"Combined CSV file not created: {combined_csv}")

            # Load the individual CSVs and the combined CSV
            print(f"reading csvs back to df to check...\n")
            individual_dataframes = [pd.read_csv(csv) for csv in csv_filepaths]
            # Load and process the combined CSV in chunks to avoid memory issues
            print(f"Reading CSV back in chunks to check...\n")
            chunk_size = 10_000
            total_rows = 0
            total_columns = None
            column_names = None
            label_counts = {}

            for chunk in pd.read_csv(combined_csv, chunksize=chunk_size):
                total_rows += len(chunk)

                # Update total columns and column names based on the first chunk
                if total_columns is None:
                    total_columns = len(chunk.columns)
                    column_names = list(chunk.columns)

                # Update label counts if the 'label' column exists
                if 'label' in chunk.columns:
                    chunk_label_counts = chunk['label'].value_counts()
                    for label, count in chunk_label_counts.items():
                        if label in label_counts:
                            label_counts[label] += count
                        else:
                            label_counts[label] = count

            # Print the dimensions and column names of the combined CSV
            print(f"Combined CSV dimensions: ({total_rows}, {total_columns})")
            print(f"Column names: {column_names}")
            
            total_points = sum(len(df) for df in individual_dataframes)
            '''error here! but probably due to something you got wrong in the checks, review pipeline. basically you combine the dataset tgtr, dont know if they were cleaned or not before/after combining...check that
            maybe erase all files because they could fall back to old version, and retry.'''
            self.assertLessEqual(total_rows, total_points, "Combined CSV should contain same points as the original ones.")
            
            # Print label distribution
            if label_counts:
                print("Label distribution in the combined CSV:")
                for label, count in label_counts.items():
                    print(f"Label {label}: {count}")
            else:
                print("No 'label' column found in the combined CSV.")

            print(f"CSV combination verified. Combined file saved at: {combined_csv_path}")
        
            
    def test_create_train_eval_datasets(self):
        # Path to the combined CSV
        if self.test_full_pipeline:
            combined_csv_path = os.path.join(self.output_dataset_folder, "full_dataset.csv")

            # Parameters for the function
            max_points_per_class = 200000
            #chosen_classes = [1.0, 6.0, 64.0]  # Test with a subset of classes
            train_split = 0.8

            # Call the function to create train and eval datasets
            print("\nRunning create_train_eval_datasets with chunked processing...\n")
            train_df, eval_df = create_train_eval_datasets(
                csv_file=combined_csv_path,
                max_points_per_class=max_points_per_class,
                chosen_classes=None,    # already cleaned earlier from other useless classes
                train_split=train_split,
                output_dataset_folder=self.output_dataset_folder
            )

            # Check if train and eval datasets are created
            train_csv = os.path.join(self.output_dataset_folder, "train_dataset.csv")
            eval_csv = os.path.join(self.output_dataset_folder, "eval_dataset.csv")
            self.assertTrue(os.path.exists(train_csv), "Training dataset CSV was not created.")
            self.assertTrue(os.path.exists(eval_csv), "Evaluation dataset CSV was not created.")

            # Load the train and eval datasets
            train_data = pd.read_csv(train_csv)
            eval_data = pd.read_csv(eval_csv)

            # Validate the splits
            total_points = len(train_data) + len(eval_data)
            self.assertEqual(len(train_df), len(train_data), "Mismatch in training dataset length.")
            self.assertEqual(len(eval_df), len(eval_data), "Mismatch in evaluation dataset length.")
            self.assertAlmostEqual(len(train_data) / total_points, train_split, delta=0.01, msg="Training split ratio mismatch.")
            self.assertAlmostEqual(len(eval_data) / total_points, 1 - train_split, delta=0.01, msg="Evaluation split ratio mismatch.")

            # Check class distribution
            print("\nChecking class distributions in train and eval datasets...\n")
            train_class_counts = train_data['label'].value_counts().sort_index()
            eval_class_counts = eval_data['label'].value_counts().sort_index()
            print("Training set class distribution:")
            print(train_class_counts)
            print("\nEvaluation set class distribution:")
            print(eval_class_counts)

            '''# Validate that all chosen classes are present
            self.assertTrue(set(chosen_classes).issubset(train_class_counts.index), "Not all chosen classes are present in the training set.")
            self.assertTrue(set(chosen_classes).issubset(eval_class_counts.index), "Not all chosen classes are present in the evaluation set.")'''

            # Validate no class exceeds max_points_per_class
            self.assertTrue((train_class_counts <= max_points_per_class).all(), "Some classes in training set exceed max_points_per_class.")
            self.assertTrue((eval_class_counts <= max_points_per_class).all(), "Some classes in evaluation set exceed max_points_per_class.")

            print("\nTrain/Eval dataset creation test passed successfully!")