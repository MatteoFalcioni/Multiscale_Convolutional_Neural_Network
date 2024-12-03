# Test created appositely to check point selection and matching between subsets of point clouds (in csv format) 
# and full point cloud (also in csv format)

import unittest
from utils.point_cloud_data_utils import read_file_to_numpy, apply_masks, isin_tolerance, apply_masks_KDTree
import numpy as np


class TestPointMatching(unittest.TestCase):
    
    def setUp(self):
        
        self.full_data_filepath = 'data/datasets/train_&_eval_dataset.csv'    # 'data/datasets/full_dataset.csv'
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
        
    '''def test_debug_matching(self):
        
        unique_full_points = np.unique(self.full_data_array[:, :3], axis=0)
        print(f"Number of unique points in full data: {len(unique_full_points)}")
        
        unique_subset_points = np.unique(self.subset_data_array[:, :3], axis=0)
        print(f"Number of unique points in subset data: {len(unique_subset_points)}")
        
        intersection_mask = np.isin(self.full_data_array[:, :3], self.subset_data_array[:, :3], assume_unique=False).all(axis=1)
        print(f"Number of exact matches (intersection): {np.sum(intersection_mask)}")
        
        for i in range(10):  # Compare the first 10 subset points
            subset_point = self.subset_data_array[i, :3]
            matches = np.isclose(self.full_data_array[:, :3], subset_point, atol=1e-10).all(axis=1)
            print(f"Subset Point {i}: {subset_point}, Matches: {np.sum(matches)}")'''
        
    '''
    def test_different_tolerances(self):
        
        tolerances = [1e-30 , 1e-20, 1e-10, 1e-6, 1e-4, 1e-2, 1]
        
        for tol in tolerances:
            print(f"\n=========================== Tolerance: {tol} ===========================")
            # Apply masks with KDTree implementation
            selected_array, mask, bounds = apply_masks(
                full_data_array=self.full_data_array,
                window_sizes=self.window_sizes,
                subset_file=self.subset_filepath,
                tol=tol
            )
            
            # Validate the number of selected points
            print(f"Selected points: {len(selected_array)}")
        '''