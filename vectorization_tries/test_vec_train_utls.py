import unittest
from utils.vectorized_train_utils import NEW_PointCloudDataset, new_prepare_dataloader 
import torch
from utils.point_cloud_data_utils import read_file_to_numpy
from scripts.old_gpu_grid_gen import apply_masks_gpu
import random
import numpy as np
from tqdm import tqdm


class TestPointCloudDataset(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.channels = len(self.features_to_use)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.tensor_window_sizes = torch.tensor([size for _, size in self.window_sizes], dtype=torch.float64, device=self.device)
        
        # full data
        self.dataset_filepath = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir=self.dataset_filepath)
        print(f"\n\nLoaded data with shape {self.full_data_array.shape}")
        self.full_data_tensor = torch.tensor(self.full_data_array, dtype=torch.float64, device=self.device)
        
        # subset file
        self.subset_filepath = 'data/datasets/train_dataset.csv'
        self.subset_array, subset_features = read_file_to_numpy(self.subset_filepath) 
        print(f"\nLoaded subset array of shape: {self.subset_array.shape}, dtype: {self.subset_array.dtype}")
        
        self.num_workers = 2
                
        
    def test_dataset_initialization(self):
        """Test initialization of the NEW_PointCloudDataset."""

        # Apply masks
        selected_tensor, mask, point_cloud_bounds = apply_masks_gpu(
            tensor_data_array=torch.tensor(self.full_data_array, dtype=torch.float64, device=self.device),
            window_sizes=self.window_sizes,
            subset_file=self.subset_filepath,
        )
        original_indices = torch.where(mask.cpu())[0].numpy()

        print(f"Initializing dataset...")
        # Initialize dataset
        dataset = NEW_PointCloudDataset(selected_tensor=selected_tensor, original_indices=original_indices)
        print(f"Dataset has been initialiazed")

        # Test dataset length
        self.assertEqual(len(dataset), len(selected_tensor))
        
        # select num_samples random indices to test on
        seed = 42
        random.seed(seed)
        num_samples = 10000  
        random_indices = np.random.choice(selected_tensor.shape[0], size=num_samples, replace=False)

        # Test individual item retrieval
        for idx in tqdm(random_indices, desc="testing dataset retrieval", unit="Retrieved arguments"):
            center_point, label, original_idx = dataset[idx]
            self.assertTrue(torch.equal(center_point, selected_tensor[idx, :3]))
            self.assertEqual(label.item(), selected_tensor[idx, -1].item())
            self.assertEqual(original_idx, original_indices[idx])
            self.assertTrue(torch.equal(center_point, self.full_data_tensor[original_indices[idx], :3]))
            

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
            device=self.device,
            subset_file=self.subset_filepath,
        )

        # Test train loader
        for batch_idx, (raw_points, labels, original_indices) in enumerate(train_loader):
            print(f"Train Batch {batch_idx}:")
            print(f"  Raw points shape: {raw_points.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Original indices: {original_indices}")

            # Check shapes
            self.assertEqual(raw_points.shape[0], 4)  # Batch size
            self.assertEqual(labels.shape[0], 4)  # Batch size
            self.assertEqual(len(original_indices), 4)  # Batch size

            if batch_idx == 2:  # Test only the first 3 batches
                break

        # Test validation loader
        for batch_idx, (raw_points, labels, original_indices) in enumerate(val_loader):
            print(f"Validation Batch {batch_idx}:")
            print(f"  Raw points shape: {raw_points.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Original indices: {original_indices}")

            # Check shapes
            self.assertGreaterEqual(raw_points.shape[0], 1)  # Remaining samples
            self.assertGreaterEqual(labels.shape[0], 1)  # Remaining samples
            self.assertGreaterEqual(len(original_indices), 1)  # Remaining samples

            if batch_idx == 2:  # Test only the first 3 batches
                break
            
            
    def test_utilities(self):
        """Test the preparation of shared utilities like KDTree and feature indices tensor."""
        # Prepare utilities
        print("Testing utilities preparation...")
        from utils.vectorized_train_utils import prepare_utilities

        utilities = prepare_utilities(
            full_data_filepath=self.dataset_filepath,
            features_to_use=self.features_to_use,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # Test utilities
        self.assertIn('gpu_tree', utilities)
        self.assertIn('tensor_full_data', utilities)
        self.assertIn('feature_indices_tensor', utilities)

        # Check tensor shapes
        tensor_full_data = utilities['tensor_full_data']
        feature_indices_tensor = utilities['feature_indices_tensor']
        self.assertEqual(tensor_full_data.shape, self.full_data_tensor.shape)
        self.assertEqual(feature_indices_tensor.shape, (self.channels,))

        # Check KDTree functionality
        gpu_tree = utilities['gpu_tree']
        query_points = tensor_full_data[:5, :3]  # Test with a few points
        _, indices = gpu_tree.query(query_points)
        self.assertEqual(indices.shape[0], query_points.shape[0])  # Ensure all queries return indices
        print("Utilities preparation tested successfully!")
