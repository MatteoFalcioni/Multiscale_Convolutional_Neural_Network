import unittest
import torch
import numpy as np
import laspy
from scripts.inference import predict, predict_subtiles
from utils.train_data_utils import prepare_dataloader, load_model, load_parameters
from utils.point_cloud_data_utils import stitch_subtiles, read_file_to_numpy, numpy_to_dataframe, clean_nan_values
from models.mcnn import MultiScaleCNN
import glob
import os
import sys

        
class TestPredictFunction(unittest.TestCase):

    def setUp(self):
        """
        Setup the test environment by creating a sampled LAS file.
        """
        self.original_las_path = 'data/chosen_tiles/32_687000_4930000_FP21.las'
        self.sampled_las_path = 'tests/test_subtiler/32_687000_4930000_FP21_sampled_10k.las'  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Set some parameters for the test
        self.batch_size = 32 
        self.grid_resolution = 128  
        self.num_workers = 16 
        self.load = True
        if self.load:
            loaded_model_path = 'models/saved/mcnn_model_20241116_143003/model.pth'
            loaded_features, num_loaded_channels, self.window_sizes = load_parameters(loaded_model_path)
            self.features_to_use = loaded_features
            self.num_channels = num_loaded_channels
            self.model = load_model(model_path=loaded_model_path, device=self.device, num_channels=num_loaded_channels)
            print(f"\nLoaded features: {self.features_to_use}")
            print(f"\nLoaded num channels: {self.num_channels}")
            print(f"\nLoaded window sizes: {self.window_sizes}")
        else:
            self.features_to_use = ['intensity', 'red', 'green', 'blue', 'nir', 'delta_z', 'l1', 'l2', 'l3']
            self.num_channels = len(self.features_to_use)
            self.model = MultiScaleCNN(channels=self.num_channels, classes=6).to(self.device)
            self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        
        print(f'window sizes: {self.window_sizes}\n')
        self.overlap_size = int([value for label, value in self.window_sizes if label == 'large'][0])   # size of the largest window 
        self.cut_off = self.overlap_size/2
        
        self.subtile_test_dir = 'tests/test_inference/two_subtiles/'  # contains two subtile from one of the 'chosen tiles'
        os.makedirs(self.subtile_test_dir, exist_ok=True)
        
    def test_inspect_data_for_bad_values(self):
        
        data_array, known_features = read_file_to_numpy(self.original_las_path, features_to_use=None)
        
        # Check for NaNs and Infs in np array
        for i, feature in enumerate(known_features):
            nan_count = np.isnan(data_array[:, i]).sum()
            inf_count = np.isinf(data_array[:, i]).sum()
            print(f"\nFeature '{feature}': NaNs: {nan_count}, Infs: {inf_count}")
            
        data_array = clean_nan_values(data_array=data_array)
        
        # Check for NaNs and Infs in np array
        for i, feature in enumerate(known_features):
            nan_count = np.isnan(data_array[:, i]).sum()
            inf_count = np.isinf(data_array[:, i]).sum()
            print("Array has beem cleaned: ")
            print(f"\nFeature '{feature}': NaNs: {nan_count}, Infs: {inf_count}")
        

    def test_predict_subtiles(self):
        """
        Test the predict_subtiles function for correctness and consistency in label assignment.
        """

        # Run predict_subtiles on the test subtile directory
        prediction_folder = predict_subtiles(
            subtile_folder=self.subtile_test_dir,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            num_workers=self.num_workers
        )

        # predicted_subtiles = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('_pred.las')]
        predicted_subtiles = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('.las')]

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
            (labeled_las.x < upper_bound_x) &  
            (labeled_las.y < upper_bound_y ) &  
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
            #print(f"Number of points before masking: {len(label_array)}")
            #print(f"Number of points after masking (inside borders): {len(subtile_masked)}")
            # Check unprocessed labels without borders  
            unprocessed_labels_no_borders = np.sum(label_array_no_borders == -1)
            total_points_no_borders = len(label_array_no_borders)
            print(f"Unprocessed labels without borders: {unprocessed_labels_no_borders} / {total_points_no_borders}")

        
    def test_predict_subtiles_and_stitching(self):

        # Run predict_subtiles on the test subtile directory
        prediction_folder = predict_subtiles(
            subtile_folder=self.subtile_test_dir,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            num_workers=self.num_workers
        )

        original_las = laspy.read(self.original_las_path)
        out_dir = 'tests/test_inference/'
        # stitch predictions back together
        predicted_filepath = stitch_subtiles(subtile_folder=prediction_folder,
                        original_las=original_las,
                        original_filename=self.original_las_path, 
                        model_directory=out_dir, 
                        overlap_size=self.overlap_size)
        
        # inspect stitched file output and compare it to the original las file
        predicted_las = laspy.read(predicted_filepath)

        # Check that the header was correctly updated in predictions to the new number of points
        print("=== Header Checks ===")
        print(f"Original LAS point count: {original_las.header.point_count}")
        print(f"Predicted LAS point count: {predicted_las.header.point_count}")
        assert predicted_las.header.point_count == len(predicted_las.x), "Mismatch between header point count and actual points!"
        self.assertNotEqual(original_las.header.point_count, predicted_las.header.point_count,
                        "Point count in the stitched file should differ due to overlap removal.")

        # Check offsets and scales
        print(f"Original Offsets: {original_las.header.offsets}")
        print(f"Predicted Offsets: {predicted_las.header.offsets}")
        self.assertTrue(np.allclose(original_las.header.offsets, predicted_las.header.offsets, atol=1e-6),
                    "Offsets in stitched LAS file should be consistent with the original.")

        print(f"Original Scales: {original_las.header.scales}")
        print(f"Predicted Scales: {predicted_las.header.scales}")
        self.assertTrue(np.allclose(original_las.header.scales, predicted_las.header.scales, atol=1e-9),
                    "Scales in stitched LAS file should be consistent with the original.")
        
        # Ensure the label field exists
        self.assertIn('label', predicted_las.point_format.dimension_names, 
                    "Label field is missing in the stitched LAS file.")

        # check how many -1 labels (unclassified points) are present in stitching
        label_array = predicted_las.label
        total_points = len(label_array)
        unprocessed_labels = np.sum(label_array == -1)
        print(f"Unprocessed labels without borders: {unprocessed_labels} / {total_points}")


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