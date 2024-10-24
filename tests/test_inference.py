import unittest
import torch
import os
import csv
import numpy as np
import laspy
from datetime import datetime
import glob 
from scripts.inference import inference, inference_without_ground_truth
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from utils.train_data_utils import prepare_dataloader
from models.mcnn import MultiScaleCNN
import random

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
        
def test_las_saving():
    # Simulate dataloader dataset output (mocking the dataset for testing)
    class MockDataset:
        def __init__(self):
            # Simulate data with 10 points, 3 coordinates, 5 features, and labels
            self.data_array = np.random.rand(10, 9)  # 3 coords, 5 features, 1 label
            self.known_features = ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue', 'nir', 'label']

    class MockDataloader:
        def __init__(self):
            self.dataset = MockDataset()

    dataloader = MockDataloader()

    # Simulate predicted labels
    predicted_labels_list = np.random.randint(0, 5, 10)  # 5 classes

    # Get the coordinates and features from the original dataset
    original_data = dataloader.dataset.data_array
    coordinates = original_data[:, :3]  # Assuming the first 3 columns are x, y, z
    features = original_data[:, 3:-1]  # Assuming features are after x, y, z

    # Generate timestamp for unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file_name = f"test_pred_{timestamp}.las"
    save_dir = "test_output"
    os.makedirs(save_dir, exist_ok=True)
    las_file_path = os.path.join(save_dir, pred_file_name)

    # Save predicted labels and features to LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    with laspy.open(las_file_path, mode="w", header=header) as las:
        las.x = coordinates[:, 0]
        las.y = coordinates[:, 1]
        las.z = coordinates[:, 2]
        las.classification = predicted_labels_list

        # Add extra features (e.g., intensity, red, green, etc.)
        for i, feature_name in enumerate(dataloader.dataset.known_features[3:-1]):
            feature_data = features[:, i]
            extra_dimension_info = laspy.util.ExtraBytesParams(
                name=f"{feature_name}", 
                type=np.float32
            )
            las.add_extra_dim(extra_dimension_info)
            las[feature_name] = feature_data

    print(f"Test LAS file saved at {las_file_path}")

