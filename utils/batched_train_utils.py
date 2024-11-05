import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from scipy.spatial import cKDTree
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from scripts.batched_pc_to_img import compute_point_cloud_bounds, generate_batched_multiscale_grids

class PointCloudDataset(Dataset):
    def __init__(self, data_array, window_sizes, grid_resolution, features_to_use, known_features, batch_mode=True):
        """
        Dataset class for batch-mode multiscale grid generation from point cloud data.
        Args:
        - data_array (numpy.ndarray): Entire point cloud data array.
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Grid resolution (e.g., 128x128).
        - features_to_use (list): List of feature names for generating grids.
        - known_features (list): All known feature names in the data array.
        - batch_mode (bool): Use batched grid generation if True.
        """
        self.data_array = data_array
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        self.batch_mode = batch_mode
        
        # Build KDTree for the entire dataset
        self.kdtree = cKDTree(data_array[:, :3])
        self.feature_indices = [known_features.index(feature) for feature in features_to_use]
        self.point_cloud_bounds = compute_point_cloud_bounds(data_array)

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        """
        Generates multiscale grids for the point at index `idx` and returns them as PyTorch tensors.
        """
        center_point = self.data_array[idx, :3]  # Get x, y, z coordinates
        label = self.data_array[idx, -1]         # Get the label

        if self.batch_mode:
            grids_dict, skipped = generate_batched_multiscale_grids(
                [center_point], data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                kdtree=self.kdtree, point_cloud_bounds=self.point_cloud_bounds
            )
        else:
            grids_dict, skipped = generate_batched_multiscale_grids(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                kdtree=self.kdtree, point_cloud_bounds=self.point_cloud_bounds
            )

        if skipped[0]:  # Skipped in batch mode
            return None

        # Convert grids to PyTorch tensors
        small_grid = torch.tensor(grids_dict['small'][0], dtype=torch.float32)
        medium_grid = torch.tensor(grids_dict['medium'][0], dtype=torch.float32)
        large_grid = torch.tensor(grids_dict['large'][0], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return small_grid, medium_grid, large_grid, label, idx

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    small_grids, medium_grids, large_grids, labels, indices = zip(*batch)
    small_grids = torch.stack(small_grids)
    medium_grids = torch.stack(medium_grids)
    large_grids = torch.stack(large_grids)
    labels = torch.stack(labels)
    indices = torch.tensor(indices)
    return small_grids, medium_grids, large_grids, labels, indices


def prepare_dataloader(batch_size, data_dir=None, window_sizes=None, grid_resolution=128,
                       features_to_use=None, train_split=None, features_file_path=None, 
                       num_workers=4, shuffle_train=True, batch_mode=True):
    """
    Prepares DataLoader with batch-mode multiscale grid generation.
    """
    if data_dir is None:
        raise ValueError('ERROR: Data directory not specified.')

    data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=None, features_file_path=features_file_path)
    data_array, _ = remap_labels(data_array)

    full_dataset = PointCloudDataset(
        data_array=data_array, window_sizes=window_sizes, grid_resolution=grid_resolution,
        features_to_use=features_to_use, known_features=known_features, batch_mode=batch_mode
    )

    if train_split is not None:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  collate_fn=custom_collate_fn, num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=custom_collate_fn, num_workers=num_workers)
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  collate_fn=custom_collate_fn, num_workers=num_workers)
        eval_loader = None

    return train_loader, eval_loader
