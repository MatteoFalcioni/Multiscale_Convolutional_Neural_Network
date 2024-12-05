import unittest
from utils.np_vectorized_train_utils import NEW_PointCloudDataset, new_prepare_dataloader, prepare_utilities
import numpy as np
from scripts.cpu_vectorized_gen import numpy_create_feature_grids, numpy_assign_features_to_grids
import random
from tqdm import tqdm
from utils.point_cloud_data_utils import read_file_to_numpy, apply_masks_KDTree

class TestPointCloudDatasetNumpy(unittest.TestCase):
    def setUp(self):
        # Common parameters
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.channels = len(self.features_to_use)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.window_sizes_np = np.array([size for _, size in self.window_sizes], dtype=np.float64)
        
        # Full data
        self.dataset_filepath = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir=self.dataset_filepath)
        print(f"\nLoaded data with shape {self.full_data_array.shape}")

        # Subset file
        self.subset_filepath = 'data/datasets/train_dataset.csv'
        self.subset_array, _ = read_file_to_numpy(self.subset_filepath)
        print(f"\nLoaded subset array of shape: {self.subset_array.shape}, dtype: {self.subset_array.dtype}")

        # Prepare utilities
        self.utilities = prepare_utilities(
            full_data_filepath=self.dataset_filepath,
            features_to_use=self.features_to_use,
            grid_resolution=self.grid_resolution,
            window_sizes=self.window_sizes
        )

        self.full_data_array = self.utilities['full_data_array']
        self.kd_tree = self.utilities['kd_tree']
        self.feature_indices = self.utilities['feature_indices']
        self.num_workers = 16

    def test_dataset_initialization(self):
        """Test initialization of the NEW_PointCloudDataset (NumPy-based)."""
        #apply masks
        selected_array, mask, _ = apply_masks_KDTree(      # here it generates another kd tree, you can make it better
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            subset_file=self.subset_filepath
        )
        original_indices = np.where(mask.cpu().numpy())[0]
        

        print(f"Initializing dataset...")
        # Initialize dataset
        dataset = NEW_PointCloudDataset(
            selected_tensor=selected_array,
            original_indices=original_indices,
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes_np,
            grid_resolution=self.grid_resolution,
            feature_indices=self.feature_indices,
            kd_tree=self.kd_tree
        )
        print(f"Dataset has been initialized")

        # Test dataset length
        self.assertEqual(len(dataset), len(selected_array))

        # Randomly select samples for testing
        random.seed(42)
        num_samples = 100
        random_indices = np.random.choice(selected_array.shape[0], size=num_samples, replace=False)

        # Test individual item retrieval
        for idx in tqdm(random_indices, desc="Testing dataset retrieval", unit="samples"):
            grids, label, original_idx = dataset[idx]
            self.assertEqual(label, selected_array[idx, -1])
            self.assertEqual(original_idx, original_indices[idx])

    def test_dataloader_creation(self):
        """Test creation and usage of the new dataloader."""
        train_loader, val_loader = new_prepare_dataloader(
            batch_size=4,
            data_filepath=self.dataset_filepath,
            window_sizes=self.window_sizes,
            features_to_use=self.features_to_use,
            train_split=0.8,
            num_workers=self.num_workers,
            shuffle_train=True,
            subset_file=self.subset_filepath
        )

        # Test train loader
        for batch_idx, (grids, labels, original_indices) in enumerate(train_loader):
            print(f"Train Batch {batch_idx}:")
            print(f"  Grids shape: {grids.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Original indices: {original_indices}")

            # Check shapes
            self.assertEqual(len(labels), 4)  # Batch size
            self.assertEqual(len(original_indices), 4)  # Batch size

            if batch_idx == 2:  # Test only the first 3 batches
                break

        # Test validation loader
        for batch_idx, (grids, labels, original_indices) in enumerate(val_loader):
            print(f"Validation Batch {batch_idx}:")
            print(f"  Grids shape: {grids.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Original indices: {original_indices}")

            # Check shapes
            self.assertGreaterEqual(len(labels), 1)  # Remaining samples
            self.assertGreaterEqual(len(original_indices), 1)  # Remaining samples

            if batch_idx == 2:  # Test only the first 3 batches
                break

    def test_utilities(self):
        """Test preparation of utilities for grid generation."""
        utilities = self.utilities
        self.assertIn('kd_tree', utilities)
        self.assertIn('full_data_array', utilities)
        self.assertIn('feature_indices', utilities)

        # Test KDTree functionality
        query_points = self.full_data_array[:5, :3]
        distances, indices = self.kd_tree.query(query_points)
        self.assertEqual(indices.shape[0], query_points.shape[0])

        print("Utilities preparation tested successfully!")
