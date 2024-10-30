import unittest
import torch
import numpy as np
import laspy
from scripts.inference import inference, inference_without_ground_truth
from utils.train_data_utils import prepare_dataloader
from models.mcnn import MultiScaleCNN
import glob
import os

'''
class TestInferenceWithDummyModel(unittest.TestCase):

    def setUp(self):
        # Create a dummy model with a simple forward method
        class DummyModel(torch.nn.Module):
            def forward(self, small_grids, medium_grids, large_grids):
                batch_size = small_grids.size(0)
                return torch.randn(batch_size, 5)  # Simulate output for 5 classes

        self.model = DummyModel()
        self.device = torch.device('cpu')  # Use CPU for the dummy model

        # Load actual point cloud data
        data_array, feature_names = read_file_to_numpy('data/sampled/sampled_data_500000.csv')

        remapped_array, _ = remap_labels(data_array)
        # Use a small sample of the data for testing
        self.point_cloud_array = remapped_array[np.random.choice(data_array.shape[0], 2000, replace=False)]
        
        # Define parameters for grid generation
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.grid_resolution = 128
        self.feature_indices = list(range(len(feature_names)))  # Use all features for testing

        # File path for saving inference results
        self.save_file = 'tests/inference/test_inference_results.csv'

    def test_inference_with_dummy_model(self):
        # Run inference with grid generation using the dummy model and real data
        predicted_labels = inference(
            model=self.model,
            data_array=self.point_cloud_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            feature_indices=self.feature_indices,  # Use feature indices
            device=self.device,
            save_file=self.save_file
        )
        
        # Check that predictions have the correct shape and type
        self.assertIsInstance(predicted_labels, torch.Tensor)

        # Ensure predictions count matches the number of valid (non-skipped) points
        num_valid_predictions = predicted_labels.shape[0]
        self.assertLessEqual(num_valid_predictions, 200)  # Should be <= 200 as some points could be skipped

        # Use glob to find the file with the timestamp
        saved_files = glob(f"{self.save_file.split('.')[0]}_*.csv")
        
        # Ensure the file was created
        self.assertTrue(len(saved_files) > 0, "The save file was not created.")
        
        # Use the first (and only) saved file
        saved_file_with_timestamp = saved_files[0]

        # Check if the labels were saved to the CSV file correctly
        with open(saved_file_with_timestamp, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Check that the file has the correct header and number of rows
            self.assertEqual(rows[0], ['True Label', 'Predicted Label'], "CSV header is incorrect.")
            self.assertEqual(len(rows) - 1, num_valid_predictions, f"Expected {num_valid_predictions} rows in the CSV, but got {len(rows) - 1}.")
            
'''
        
class TestInferenceLabelOrderWithRealData(unittest.TestCase):

    def setUp(self):
        # Setup parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 6
        self.num_channels = 4
        self.batch_size = 32
        self.grid_resolution = 128
        self.window_sizes = [("small", 1.0), ("medium", 2.0), ("large", 4.0)]
        self.num_workers = 0
        
        # Initialize the model
        self.model = MultiScaleCNN(channels=self.num_channels, classes=self.num_classes).to(self.device)
        
        # Set up the paths to the LAS files
        self.large_las_file = "data/chosen_tiles/32_680000_4928500_FP21.las"  
        self.small_las_file = "test_data_small.las"

        # Create a subsample of the large LAS file for testing
        las = laspy.read(self.large_las_file)
        mask = np.full(len(las.x), False)
        mask[:1000] = True
        subsampled_points = las.points[mask]
        
        # Write the subsampled points to the small LAS file
        small_las = laspy.LasData(las.header)
        small_las.points = subsampled_points
        small_las.write(self.small_las_file)

        # Prepare DataLoader with the small LAS file
        self.dataloader, _ = prepare_dataloader(
            batch_size=self.batch_size,
            data_dir=self.small_las_file,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=['intensity', 'return_number', 'number_of_returns', 'planarity'],
            train_split=None,
            num_workers=self.num_workers,
            shuffle_train=False
        )

    def test_inference_without_ground_truth(self):
        # Run inference which saves predictions directly to LAS file
        saved_file_path = inference_without_ground_truth(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            data_file=self.small_las_file,
            model_save_folder="tests/test_model_output"
        )

        # Verify that the predictions were correctly saved in the output LAS file
        saved_file = laspy.read(saved_file_path)
        saved_labels = saved_file.classification

        # Check that the number of labels matches the number of points
        num_points = len(self.dataloader.dataset.data_array)
        self.assertEqual(len(saved_labels), num_points, "Mismatch in the number of points and saved labels")

        # Ensure that the saved labels contain valid predictions (non -1 values) for all processed points
        non_negative_labels = np.sum(saved_labels != -1)
        self.assertEqual(non_negative_labels, num_points, "Not all labels were assigned predictions in the saved file")