import unittest
import torch
import numpy as np
import laspy
from scripts.inference import predict, predict_subtiles
from utils.train_data_utils import prepare_dataloader, load_model, load_parameters
from models.mcnn import MultiScaleCNN
import glob
import os
import sys

        
class TestPredictFunction(unittest.TestCase):

    def setUp(self):
        """
        Setup the test environment by creating a sampled LAS file.
        """
        self.sampled_las_path = 'tests/test_subtiler/32_687000_4930000_FP21_sampled_10k.las'  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Set some parameters for the test
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)] 
        self.overlap_size = 30.0  # size of the largest window 
        self.batch_size = 16  
        self.grid_resolution = 128  
        self.features_to_use = ['x', 'y', 'z', 'intensity']  
        self.num_channels = len(self.features_to_use)
        self.num_workers = 0 
        self.tile_size = 125  # Tile size for subtiles
        self.min_points = 500  # Example threshold for when to subtile
        self.model = MultiScaleCNN(channels=self.num_channels, classes=6).to(self.device) 
        loaded_model_path = 'models/saved/mcnn_model_20241030_051517/model.pth'
        loaded_features, num_loaded_channels, self.window_sizes = load_parameters(loaded_model_path)
        self.features_to_use = loaded_features
        print(f'window sizes: {self.window_sizes}\nLoaded features: {loaded_features}')
        self.model = load_model(model_path=loaded_model_path, device=self.device, num_channels=num_loaded_channels) 


    def test_predict_subtiles(self):
        """
        Test the predict_subtiles function for correctness and consistency in label assignment.
        """
        # Create a temporary directory for subtile tests
        """here: put a directory with some real subtiles, with million of points"""
        subtile_test_dir = 'tests/test_subtiler/32_687000_4930000_FP21_sampled_10k_250_subtiles'
        os.makedirs(subtile_test_dir, exist_ok=True)

        # Run predict_subtiles on the test subtile directory
        prediction_folder = predict_subtiles(
            subtile_folder=subtile_test_dir,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            num_workers=self.num_workers
        )

        predicted_subtiles = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('_pred.las')]

        for subtile_path in predicted_subtiles:

            labeled_las = laspy.read(subtile_path)
            # Verify the label field was added and populated
            self.assertIn('label', labeled_las.point_format.dimension_names, "Label field not added to LAS file header.")

            # Define upper bounds based on tile size (to cut off strips of unlabeled points)
            cut_off = self.overlap_size/2

            min_x = labeled_las.x.min()
            max_x = labeled_las.x.max()
            min_y = labeled_las.y.min()
            max_y = labeled_las.y.max()
            
            upper_bound_x = max_x - cut_off    
            upper_bound_y = max_y - cut_off
            lower_bound_x = min_x + cut_off
            lower_bound_y = min_y + cut_off

            mask = (
            (labeled_las.x < upper_bound_x) &  # Exclude right overlap if not rightmost
            (labeled_las.y < upper_bound_y ) &  # Exclude top overlap if not northernmost
            (labeled_las.x > lower_bound_x) &
            (labeled_las.y > lower_bound_y)
            )

            subtile_masked = labeled_las.points[mask]   # get the 'inside' of the subtiles, without borders of -1 labels

            label_array = labeled_las.label
            # Check unprocessed labels (-1) considering borders
            unprocessed_labels = np.sum(label_array == -1)
            total_points = len(label_array)
            print(f"Unprocessed labels with borders: {unprocessed_labels} / {total_points}")

            label_array_no_borders = subtile_masked.label
            print(f"Number of points before masking: {len(label_array)}")
            print(f"Number of points after masking (inside borders): {len(subtile_masked)}")
            # Check unprocessed labels without borders  
            unprocessed_labels_no_borders = np.sum(label_array_no_borders == -1)
            total_points_no_borders = len(label_array_no_borders)
            print(f"Unprocessed labels without borders: {unprocessed_labels_no_borders} / {total_points_no_borders}")


    '''def test_predict(self):
        """
        Test the entire pipeline of subtile processing, inference, and stitching.
        """
        # Run the predict function on the sampled file
        print(f"Running predict on sampled LAS file: {self.sampled_las_path}")
        
        predict(
            file_path=self.sampled_las_path,
            model=self.model,
            model_path='tests/test_subtiler/test_predict/',
            device=self.device,
            batch_size=self.batch_size,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            num_workers=self.num_workers,
            min_points=self.min_points,
            tile_size=self.tile_size
        )'''



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