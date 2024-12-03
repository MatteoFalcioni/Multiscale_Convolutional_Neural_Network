# Test created appositely to check point selection and matching between subsets of point clouds (in csv format) 
# and full point cloud (also in csv format)

import unittest
from utils.point_cloud_data_utils import read_file_to_numpy, apply_masks_KDTree
import numpy as np


class TestPointMatching(unittest.TestCase):
    
    def setUp(self):
        
        self.full_data_filepath = 'data/datasets/full_dataset.csv' #'data/datasets/train_&_eval_dataset.csv' 
        self.subset_filepath = 'data/datasets/train_dataset.csv'
        
        self.full_data_array, self.full_features = read_file_to_numpy(data_dir=self.full_data_filepath)
        self.subset_data_array, self.subset_features = read_file_to_numpy(data_dir=self.subset_filepath)
        print(f"\nFull data shape: {self.full_data_array.shape}, dtype: {self.full_data_array.dtype}")
        print(f"\nSubset data shape: {self.subset_data_array.shape}, dtype: {self.subset_data_array.dtype}")
        
        if self.full_features != self.subset_features:
            print("Warning: subset features don't match the full data features")
            
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.tol = 1e-10
            
        np.set_printoptions(precision=16)

                  
        
    def test_kdtree_selection(self):
        
        # Apply masks with KDTree implementation
        selected_array, mask, bounds = apply_masks_KDTree(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            subset_file=self.subset_filepath,
            tol=self.tol
        )
        
        print(f"Number of matches: {np.sum(mask)}")
        print(f"\nSelected array shape: {selected_array.shape}, dtype: {selected_array.dtype}")
        
        # Validate the number of selected points
        print(f"Selected points: {len(selected_array)}")
        self.assertGreater(len(selected_array), 0, "No points were selected; check KDTree implementation.")
        self.assertLessEqual(len(selected_array), len(self.full_data_array),
                            "More points were selected than available in the full dataset.")
        
    
    def test_different_tolerances(self):
        
        tolerances = [1e-20, 1e-10, 1e-6, 1e-4, 1e-2, 1]
        
        for tol in tolerances:
            print(f"\n=========================== Tolerance: {tol} ===========================")
            # Apply masks with KDTree implementation
            selected_array, mask, bounds = apply_masks_KDTree(
                full_data_array=self.full_data_array,
                window_sizes=self.window_sizes,
                subset_file=self.subset_filepath,
                tol=tol
            )
            
            # Validate the number of selected points
            print(f"Selected points: {len(selected_array)}")
        