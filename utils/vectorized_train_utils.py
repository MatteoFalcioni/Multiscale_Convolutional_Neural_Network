from scripts.old_gpu_grid_gen import apply_masks_gpu
from scripts.vectorized_grid_gen import vectorized_generate_multiscale_grids
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values
from torch_kdtree import build_kd_tree


class NEW_PointCloudDataset(Dataset):
    """
    Dataset class for point cloud data with original index mapping.
    """
    def __init__(self, selected_tensor, original_indices):
        """
        Args:
        - selected_tensor (torch.Tensor): Preprocessed tensor with selected points and labels.
        - original_indices (np.ndarray): Original indices of the points in the full dataset.
        """
        self.selected_tensor = selected_tensor
        self.original_indices = original_indices

    def __len__(self):
        return len(self.selected_tensor)

    def __getitem__(self, idx):
        """
        Retrieves a single point, its label, and the original index.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - torch.Tensor: Coordinates of the point (x, y, z).
        - torch.Tensor: Label of the point.
        - int: Original index of the point in the full data array.
        """
        center_point = self.selected_tensor[idx, :3]  # x, y, z coordinates
        label = self.selected_tensor[idx, -1].long()  # Label as long
        original_idx = self.original_indices[idx]  # Map to the original dataset index
        return center_point, label, original_idx



def new_prepare_dataloader(batch_size, data_filepath=None, window_sizes=None, features_to_use=None, 
                           train_split=None, num_workers=4, shuffle_train=True, device='cuda:0', subset_file=None):
    """
    Prepares the DataLoader using an external KDTree and feature index tensor for multiprocessing.

    Args:
    - batch_size (int): The batch size to be used for training.
    - data_filepath (str): Path to the raw data (e.g., .las or .csv file). Default is None.
    - window_sizes (list): List of window sizes to use for grid generation. Default is None.
    - features_to_use (list): List of feature names to use for grid generation. Default is None.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data). Default is None.
    - num_workers (int): Number of workers for parallelized data loading. Default is 4.
    - shuffle_train (bool): Whether to shuffle the data for training. Default is True.
    - device (str): The device ('cuda' or 'cpu') for tensor operations.
    - subset_file (str, optional): Path to a CSV file containing coordinates of points to be selected. If None, all are selected.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - val_loader (DataLoader): DataLoader for validation (if train_split is not None, else val_loader=None).
    - shared_objects (dict): Contains shared KDTree, full data tensor, and feature indices.
    """
    # Read raw data, remap labels and clean nan values
    data_array, known_features = read_file_to_numpy(data_filepath)
    data_array, _ = remap_labels(data_array)
    data_array = clean_nan_values(data_array)

    # Move data to the appropriate device
    tensor_full_data = torch.tensor(data_array, dtype=torch.float64, device=device)

    # Build the KDTree
    print("Building KDTree...")
    gpu_tree = build_kd_tree(tensor_full_data[:, :3])
    print("KDTree successfully built.")

    # Prepare feature indices tensor
    feature_indices = [known_features.index(feature) for feature in features_to_use]
    feature_indices_tensor = torch.tensor(feature_indices, dtype=torch.int64, device=device)

    # Apply masking to select points if subset_file is provided
    selected_tensor, mask, point_cloud_bounds = apply_masks_gpu(
        tensor_data_array=tensor_full_data,
        window_sizes=window_sizes,
        subset_file=subset_file
    )
    original_indices = torch.where(mask.cpu())[0].numpy()  # Map from selected array to original data indices

    # Initialize dataset
    dataset = NEW_PointCloudDataset(selected_tensor=selected_tensor, original_indices=original_indices)

    # Split dataset into training and validation sets
    if train_split:
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = None

    # Share the reusable components
    shared_objects = {
        'gpu_tree': gpu_tree,
        'tensor_full_data': tensor_full_data,
        'feature_indices_tensor': feature_indices_tensor
    }

    return train_loader, val_loader, shared_objects