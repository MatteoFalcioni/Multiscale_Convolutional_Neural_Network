import unittest
from scripts.vectorized_grid_gen import vectorized_generate_multiscale_grids
from utils.vectorized_train_utils import NEW_PointCloudDataset, new_prepare_dataloader 
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
        
        self.num_workers = 4

        # Prepare DataLoader
        self.train_loader, self.eval_loader = new_prepare_dataloader(
            batch_size=self.batch_size,
            data_filepath=self.dataset_filepath,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=None,
            num_workers=self.num_workers,
            shuffle_train=False,
            device=self.device,
            subset_file=self.subset_filepath
        )

    def test_dataloader_batch_generation(self):
        """
        Test that the DataLoader prepared by `gpu_prepare_dataloader` returns batches of raw points and labels
        and verify that grid generation works for these batches.
        """
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:
                continue
            
            raw_points, labels, original_indices = batch
            print(f"Batch {batch_idx}: Raw points shape: {raw_points.shape}, Labels shape: {labels.shape}")

            # Verify shapes of raw points and labels
            self.assertEqual(raw_points.shape, (self.batch_size, 3))  # x, y, z
            self.assertEqual(labels.shape, (self.batch_size,))  # Labels

    def test_grid_generation_for_batches(self):
        """
        Test grid generation using `vectorized_generate_multiscale_grids` for batches returned by the DataLoader.
        """
        for batch_idx, batch in enumerate(self.train_loader):
            if batch is None:
                continue

            raw_points, labels, _ = batch
            raw_points = raw_points.to(self.device)

            # Generate grids
            grids = vectorized_generate_multiscale_grids(
                raw_points,
                torch.tensor([size for _, size in self.window_sizes], dtype=torch.float64, device=self.device),
                self.grid_resolution,
                len(self.features_to_use),
                self.train_loader.dataset.gpu_tree,
                self.train_loader.dataset.tensor_full_data,
                self.train_loader.dataset.feature_indices_tensor,
                self.device
            )

            # Check the shape of the generated grids
            self.assertEqual(grids.shape, (self.batch_size, len(self.window_sizes), len(self.features_to_use), self.grid_resolution, self.grid_resolution))



if __name__ == '__main__':
    unittest.main()
