from scripts.cpu_vectorized_gen import vectorized_create_feature_grids, vectorized_assign_features_to_grids
from scipy.spatial import cKDTree
from scripts.cpu_vectorized_gen import numpy_generate_multiscale_grids
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values, apply_masks_KDTree
import numpy as np


class NEW_PointCloudDataset(Dataset):
    """
    Dataset class for point cloud data with grid generation using NumPy vectorization.
    """
    def __init__(self, selected_tensor, original_indices, full_data_array, window_sizes, grid_resolution, feature_indices, kd_tree):
        """
        Args:
        - selected_tensor (torch.Tensor): Preprocessed tensor with selected points and labels.
        - original_indices (np.ndarray): Original indices of the points in the full dataset.
        - full_data_array (np.ndarray): Full point cloud data array.
        - window_sizes (list): List of window sizes for grid generation.
        - grid_resolution (int): Grid resolution for grids.
        - feature_indices (np.ndarray): Indices of features to include.
        - kd_tree (scipy.spatial.cKDTree): KDTree built on the full data for feature assignment.
        """
        self.selected_tensor = selected_tensor
        self.original_indices = original_indices
        self.full_data_array = full_data_array
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.feature_indices = feature_indices
        self.kd_tree = kd_tree

    def __len__(self):
        return len(self.selected_tensor)

    def __getitem__(self, idx):
        """
        Retrieves a single point, its label, and generates multiscale grids.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - grids (np.ndarray): Generated grids of shape (scales, channels, grid_res, grid_res).
        - label (int): Label of the point.
        - original_idx (int): Original index of the point in the full data array.
        """
        # Extract the center point and label
        center_point = self.selected_tensor[idx, :3]  # x, y, z coordinates
        label = int(self.selected_tensor[idx, -1])    # Label as an integer
        original_idx = self.original_indices[idx]     # Map to original dataset index

        # Generate grids using NumPy vectorization

        grids = numpy_generate_multiscale_grids()

        return grids, label, original_idx


def new_prepare_dataloader(batch_size, data_filepath, window_sizes, features_to_use, train_split=None, num_workers=4, shuffle_train=True, subset_file=None):
    """
    Prepares the DataLoader with NumPy-based grid generation.

    Args:
    - batch_size (int): The batch size for training.
    - data_filepath (str): Path to the raw data file.
    - window_sizes (list): List of window sizes.
    - features_to_use (list): List of feature names for grid generation.
    - train_split (float): Ratio for training and validation split.
    - num_workers (int): Number of workers for data loading.
    - shuffle_train (bool): Whether to shuffle the training data.
    - subset_file (str, optional): Path to a file for subset selection.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - val_loader (DataLoader): DataLoader for validation (if train_split is provided).
    - shared_objects (dict): Contains prepared utilities.
    """
    # Prepare utilities
    shared_objects = prepare_utilities(data_filepath, features_to_use, grid_resolution=128, window_sizes=window_sizes)

    # Apply masking to select points if subset_file is provided
    data_array = shared_objects['full_data_array']
    selected_array, mask, _ = apply_masks_KDTree(      # here it generates another kd tree, you can make it better
        full_data_array=data_array,
        window_sizes=window_sizes,
        subset_file=subset_file
    )
    original_indices = np.where(mask.cpu().numpy())[0]

    # Initialize dataset
    dataset = NEW_PointCloudDataset(
        selected_tensor=selected_array,
        original_indices=original_indices,
        full_data_array=data_array,
        window_sizes=shared_objects['window_sizes'],
        grid_resolution=shared_objects['grid_resolution'],
        feature_indices=shared_objects['feature_indices'],
        kd_tree=shared_objects['kd_tree']
    )

    # Split dataset
    if train_split:
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = None

    return train_loader, val_loader


def prepare_utilities(full_data_filepath, features_to_use, grid_resolution, window_sizes):
    """
    Prepares shared resources like KDTree, feature indices, the full data array, and window sizes.

    Args:
    - full_data_filepath (str): Path to the raw data file.
    - features_to_use (list): List of features to include in the grids.
    - grid_resolution (int): Resolution of the grid (e.g., 128).
    - window_sizes (list): List of window sizes (e.g., [10.0, 20.0]).

    Returns:
    - dict: Contains KDTree, feature indices, full data array, window sizes, and grid resolution.
    """
    # Read raw data and preprocess
    data_array, known_features = read_file_to_numpy(full_data_filepath)
    data_array, _ = remap_labels(data_array)
    data_array = clean_nan_values(data_array)

    # Build NumPy KDTree
    print("Building KDTree...")
    kd_tree = cKDTree(data_array[:, :3])
    print("KDTree successfully built.")

    # Prepare feature indices
    feature_indices = np.array([known_features.index(feature) for feature in features_to_use], dtype=np.int64)

    return {
        'kd_tree': kd_tree,
        'full_data_array': data_array,
        'feature_indices': feature_indices,
        'window_sizes': np.array(window_sizes, dtype=np.float64),
        'grid_resolution': grid_resolution
    }
