import unittest
from utils.gpu_training_utils import gpu_prepare_dataloader, GPU_PointCloudDataset
from scripts.gpu_grid_gen import apply_masks_gpu, isin_tolerance_gpu
from utils.train_data_utils import prepare_dataloader, PointCloudDataset
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
import torch
import os
from utils.point_cloud_data_utils import apply_masks, isin_tolerance


class TestMaskingCPUvsGPU(unittest.TestCase):
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

        self.tol = 1e-8
        
        self.real_data = 'data/datasets/sampled_full_dataset/sampled_data_5251680.csv'
        self.real_subset_file = 'data/datasets/train_dataset.csv'
        self.real_data_array, self.real_knonw_features = read_file_to_numpy(data_dir=self.real_data)
        self.tensor_real_data = torch.tensor(self.real_data_array, dtype=torch.float64, device=self.device)
        self.real_subset_file = 'data/datasets/train_dataset.csv'   # A real subset file for testing
        self.real_subset_array, _ = read_file_to_numpy(self.real_subset_file)
        print(f"\nFull real data array shape: {self.real_data_array.shape}, dtype: {self.real_data_array.dtype}")
        print(f"\nReal data tensor shape: {self.tensor_real_data.shape}")
        print(f"Subset array shape: {self.real_subset_array.shape}, dtype: {self.real_subset_array.dtype}\n")


    def test_isin_tolerance_cpu_vs_gpu(self):
        """Compare the CPU and GPU versions of isin_tolerance with both simulated and real data."""
        for dataset, subset in [
            (self.full_data_array, self.subset_points),
            (self.real_data_array, self.real_subset_array)
        ]:
            # CPU mask
            cpu_mask = np.ones(len(dataset), dtype=bool)
            # GPU mask
            gpu_mask = torch.ones(len(dataset), dtype=torch.bool, device=self.device)

            for i in range(3):  # Iterate over x, y, z
                cpu_mask &= isin_tolerance(dataset[:, i], subset[:, i], self.tol)
                gpu_mask &= isin_tolerance_gpu(
                    torch.tensor(dataset[:, i], dtype=torch.float64, device=self.device),
                    torch.tensor(subset[:, i], dtype=torch.float64, device=self.device),
                    self.tol
                )

            # Compare results
            np.testing.assert_array_equal(
                cpu_mask, gpu_mask.cpu().numpy(),
                err_msg=f"CPU and GPU isin_tolerance masks do not match for dataset: {dataset}."
            )


    def test_apply_masks_cpu_vs_gpu(self):
        """Compare the CPU and GPU implementations of apply_masks for synthetic and real data."""
        for data_array, subset_file in [
            (self.full_data_array, self.subset_file),
            (self.real_data_array, self.real_subset_file)
        ]:
            # CPU masking
            cpu_selected_array, cpu_mask, cpu_bounds = apply_masks(
                full_data_array=data_array,
                window_sizes=self.window_sizes,
                subset_file=subset_file,
                tol=self.tol
            )

            # GPU masking
            tensor_data = torch.tensor(data_array, dtype=torch.float64, device=self.device)
            gpu_selected_array, gpu_mask, gpu_bounds = apply_masks_gpu(
                tensor_data_array=tensor_data,
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



'''class TestGPUPointCloudDataset(unittest.TestCase):
    def setUp(self):
        # Use a smaller dataset for testing consistency
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir='data/datasets/sampled_full_dataset/sampled_data_5251680.csv')
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

    def tearDown(self):
        # Clean up subset file
        if os.path.exists(self.subset_file):
            os.remove(self.subset_file)


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
        )'''

'''def test_grid_generation(self):
        """Test multiscale grid generation consistency for random indices."""
        np.random.seed(self.seed)
        indices = np.random.choice(len(self.cpu_dataset), 1000, replace=False)  # Test on 10 random points
        for idx in indices:
            # CPU grids
            cpu_small, cpu_medium, cpu_large, cpu_label, _ = self.cpu_dataset[idx]

            # GPU grids
            gpu_small, gpu_medium, gpu_large, gpu_label, _ = self.gpu_dataset[idx]

            # Compare grids
            np.testing.assert_close(cpu_small, gpu_small.cpu().numpy()), atol=1e-10, msg=f"Small grid mismatch at index {idx}")
            np.testing.assert_close(cpu_medium, gpu_medium.cpu().numpy(), atol=1e-10, msg=f"Medium grid mismatch at index {idx}")
            np.testing.assert_close(cpu_large, gpu_large.cpu().numpy(), atol=1e-10, msg=f"Large grid mismatch at index {idx}")

            # Compare labels
            self.assertEqual(cpu_label, gpu_label.item(), f"Label mismatch at index {idx}")'''

'''def test_original_indices_mapping(self):
        """Test that the original indices mapping matches between CPU and GPU."""
        np.testing.assert_array_equal(
            self.cpu_dataset.original_indices,
            self.gpu_dataset.original_indices,
            "Original indices mapping mismatch between CPU and GPU datasets."
        )'''