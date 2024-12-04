import unittest
from torch.utils.data import DataLoader
from utils.vectorized_train_utils import GPU_PointCloudDataset 
import torch
from utils.point_cloud_data_utils import read_file_to_numpy


class TestPointCloudDataset(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']
        self.channels = len(self.features_to_use)
        self.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        self.tensor_window_sizes = torch.tensor([size for _, size in self.window_sizes], dtype=torch.float64, device=self.device)
        
        # full data
        dataset_filepath = 'data/datasets/train_dataset.csv'
        self.full_data_array, self.known_features = read_file_to_numpy(data_dir=dataset_filepath)
        print(f"\n\nLoaded data with shape {self.full_data_array.shape}")
        
        # subset file
        self.subset_filepath = 'data/datasets/train_dataset.csv'
        self.subset_array, subset_features = read_file_to_numpy(self.subset_filepath) 
        print(f"\nLoaded subset array of shape: {self.subset_array.shape}, dtype: {self.subset_array.dtype}")

        # Initialize the dataset
        self.dataset = GPU_PointCloudDataset(
            full_data_array=self.full_data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features,
            device=self.device,
            subset_file=None
        )

    def test_dataset_initialization(self):
        # Check if dataset initializes correctly
        self.assertEqual(len(self.dataset), len(self.dataset.selected_tensor))
        self.assertEqual(len(self.dataset.original_indices), len(self.dataset.selected_tensor))

    def test_single_item_retrieval(self):
        # Retrieve a single item from the dataset
        idx = 0
        grids, label, original_idx = self.dataset[idx]

        # Check the shapes of the grids and label
        self.assertEqual(grids.shape, (3, self.grid_resolution, self.grid_resolution, len(self.features_to_use)))
        self.assertTrue(isinstance(label, torch.Tensor))
        self.assertTrue(isinstance(original_idx, int))

    def test_dataloader_integration(self):
        # Initialize a dataloader
        dataloader = DataLoader(self.dataset, batch_size=4, collate_fn=None, num_workers=0)

        # Iterate through one batch
        for batch in dataloader:
            grids, labels, original_indices = batch

            # Check batch dimensions
            self.assertEqual(grids.shape[0], 4)  # Batch size
            self.assertEqual(grids.shape[1:], (3, self.grid_resolution, self.grid_resolution, len(self.features_to_use)))  # Grid dimensions
            self.assertEqual(labels.shape, (4,))
            self.assertEqual(original_indices.shape, (4,))
            break


if __name__ == '__main__':
    unittest.main()
