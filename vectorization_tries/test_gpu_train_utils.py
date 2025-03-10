import unittest
from utils.old_gpu_training_utils import gpu_prepare_dataloader, GPU_PointCloudDataset
from scripts.old_gpu_grid_gen import apply_masks_gpu
from utils.train_data_utils import prepare_dataloader, PointCloudDataset
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torch
import os
from utils.point_cloud_data_utils import apply_masks_KDTree
import time
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.multiprocessing as mp



'''class TestMaskingCPUvsGPU(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Simulated full data array
        self.full_data_array = np.random.rand(100000, 5)  # Shape [100000, 5], includes x, y, z, feature, label
        self.subset_points = self.full_data_array[np.random.choice(100000, 20000, replace=False), :3]  # Subset (x, y, z)

        # Convert full data array to GPU tensor
        self.tensor_full_data = torch.tensor(self.full_data_array, dtype=torch.float64, device=self.device)

        # Window sizes and bounds for masking
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.point_cloud_bounds = {
            'x_min': 0.0, 'x_max': 100.0, 
            'y_min': 0.0, 'y_max': 100.0
        }

        # Temporary subset file for CPU testing
        self.subset_file = "tests/test_subset.csv"
        np.savetxt(self.subset_file, self.subset_points, delimiter=',', header='x,y,z', comments='')

        self.tol = 1e-10
        
        self.real_data = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        self.real_subset_file = 'data/datasets/train_dataset.csv'
        self.real_data_array, self.real_known_features = read_file_to_numpy(data_dir=self.real_data)
        self.tensor_real_data = torch.tensor(self.real_data_array, dtype=torch.float64, device=self.device)
        self.real_subset_array, _ = read_file_to_numpy(self.real_subset_file)

        print(f"\nFull real data array shape: {self.real_data_array.shape}, dtype: {self.real_data_array.dtype}")
        print(f"Real data tensor shape: {self.tensor_real_data.shape}")
        print(f"Subset array shape: {self.real_subset_array.shape}, dtype: {self.real_subset_array.dtype}\n")

    def test_apply_masks_cpu_vs_gpu(self):
        """Compare the CPU and GPU implementations of apply_masks for synthetic and real data."""
        for data_array, subset_file in [
            (self.full_data_array, self.subset_file),
            (self.real_data_array, self.real_subset_file)
        ]:
            # CPU masking with KDTree
            cpu_selected_array, cpu_mask, cpu_bounds = apply_masks_KDTree(
                full_data_array=data_array,
                window_sizes=self.window_sizes,
                subset_file=subset_file,
                tol=self.tol
            )

            # GPU masking
            gpu_selected_array, gpu_mask, gpu_bounds = apply_masks_gpu(
                tensor_data_array=torch.tensor(data_array, dtype=torch.float64, device=self.device),
                window_sizes=self.window_sizes,
                subset_file=subset_file,
                tol=self.tol
            )

            # Convert GPU outputs back to CPU for comparison
            gpu_selected_array = gpu_selected_array.cpu().numpy()
            gpu_mask = gpu_mask.cpu().numpy()

            # Compare selected arrays
            np.testing.assert_allclose(
                np.sort(cpu_selected_array, axis=0),
                np.sort(gpu_selected_array, axis=0),
                atol=self.tol,
                err_msg="Selected arrays from CPU and GPU masking do not match."
            )

            # Compare masks
            np.testing.assert_array_equal(
                cpu_mask,
                gpu_mask,
                err_msg="Masks from CPU and GPU masking do not match."
            )

            # Compare bounds
            self.assertEqual(cpu_bounds, gpu_bounds, "Bounds computed on CPU and GPU do not match.")


    def tearDown(self):
        # Remove temporary subset file
        import os
        if os.path.exists(self.subset_file):
            os.remove(self.subset_file)


class TestGPUPointCloudDataset(unittest.TestCase):
    def setUp(self):
        # Use a smaller dataset for testing consistency
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir='data/datasets/sampled_full_dataset/sampled_data_5251681.csv')
        self.subset_file = 'data/datasets/train_dataset.csv'   # A real subset file for testing
        subset_array, _ = read_file_to_numpy(self.subset_file)

        print(f"\nFull data array shape: {self.full_data_array.shape}, dtype: {self.full_data_array.dtype}")
        print(f"Subset array shape: {subset_array.shape}\n")
        
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.seed = 42  # for reproducibility

        # CPU Dataset
        self.cpu_dataset = PointCloudDataset(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features,
            subset_file=self.subset_file
        )

        # GPU Dataset
        self.gpu_dataset = GPU_PointCloudDataset(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features,
            device=self.device,
            subset_file=self.subset_file
        )


    def test_lengths(self):
        """Test that CPU and GPU datasets have the same length."""
        self.assertEqual(len(self.cpu_dataset), len(self.gpu_dataset), "CPU and GPU dataset lengths mismatch.")


    def test_subset_filtering(self):
        """Test subset filtering consistency between CPU and GPU datasets."""
        np.testing.assert_allclose(
            self.cpu_dataset.selected_array[:, :3],
            self.gpu_dataset.selected_tensor[:, :3].cpu().numpy(),
            atol=1e-10,
            err_msg="Mismatch between CPU and GPU subset filtering."
        )
    
    
    def test_original_indices_mapping(self):
        """Test that the original indices mapping matches between CPU and GPU."""
        np.testing.assert_array_equal(
            self.cpu_dataset.original_indices,
            self.gpu_dataset.original_indices,
            "Original indices mapping mismatch between CPU and GPU datasets."
        )'''


class TestDataloaderDatasetIntegrationGPU(unittest.TestCase):
    def setUp(self):
        mp.set_start_method(method="forkserver")
        # Mock parameters
        self.batch_size = 32
        self.grid_resolution = 128
        self.full_data_path = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        self.real_subset_file = 'data/datasets/train_dataset.csv'
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.features_to_use = ['intensity', 'red']
        self.num_channels = len(self.features_to_use)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 2


    def test_dataloader_with_dataset(self):
        # Prepare DataLoader with the GPU dataset
        loader, _ = gpu_prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.full_data_path,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=None,  # no eval
            num_workers=self.num_workers,
            shuffle_train=True,
            device=self.device,
            subset_file=self.real_subset_file  
        )

        num_batches_to_test = 1000
        
        start_time = time.time()
        for i, batch in enumerate(tqdm(loader, desc="Testing GPU dataloader + dataset integration", total=num_batches_to_test)):
            if i >= num_batches_to_test:
                break  # Test only a subset of batches

            small_grid, medium_grid, large_grid, labels, indices = batch
            
            # Validate grid shapes
            self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Small grid resolution mismatch")
            self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Medium grid resolution mismatch")
            self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution),
                                f"Large grid resolution mismatch")

            # Check for NaN/Inf values (directly on the GPU)
            for grid, scale in zip([small_grid, medium_grid, large_grid], ['small', 'medium', 'large']):
                self.assertFalse(torch.isnan(grid).any().item(), f"NaN values found in {scale} grid for batch {i}")
                self.assertFalse(torch.isinf(grid).any().item(), f"Inf values found in {scale} grid for batch {i}")
                
        end_time = time.time()
        print(f"{num_batches_to_test} batches processed in {(end_time-start_time)/60:.2f} minutes.")

'''
class TestPrepareDataloaderGPU(unittest.TestCase):
    def setUp(self):
        # Configurable parameters for the test
        self.batch_size = 32
        self.grid_resolution = 128
        self.train_split = 0.8
        self.full_data_path = 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'
        self.real_subset_file = 'data/datasets/train_dataset.csv'
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.features_to_use = ['intensity', 'red']
        self.num_channels = len(self.features_to_use)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0

    def test_dataloader_full_dataset(self):
        # Prepare DataLoader without train/test split (using full dataset)
        train_loader, eval_loader = gpu_prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.full_data_path,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=None,  # No train/test split, using full dataset
            num_workers=self.num_workers,
            shuffle_train=True,
            device=self.device,
            subset_file=self.real_subset_file
        )

        self.assertIsInstance(train_loader, DataLoader, "train_loader is not a DataLoader instance.")
        self.assertIsNone(eval_loader, "eval_loader should be None when train_split=None.")

        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None.")
        small_grid, medium_grid, large_grid, labels, indices = first_batch

        # Validate shapes
        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")


    def test_train_eval_dataloader(self):
        # Load both train and eval DataLoader
        train_loader, eval_loader = gpu_prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.full_data_path,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=self.train_split,
            device=self.device,
            subset_file=self.real_subset_file,
            num_workers=self.num_workers
        )

        # Test train_loader
        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None.")
        small_grid, medium_grid, large_grid, labels, indices = first_batch

        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")

        # Test eval_loader
        self.assertIsNotNone(eval_loader, "Evaluation DataLoader is None.")
        first_eval_batch = next(iter(eval_loader))
        self.assertIsNotNone(first_eval_batch, "First eval batch should not be None.")
        small_grid, medium_grid, large_grid, labels_eval, indices_eval = first_eval_batch

        self.assertEqual(small_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch in eval.")
        self.assertEqual(medium_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch in eval.")
        self.assertEqual(large_grid.shape[-3:], (self.num_channels, self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch in eval.")'''