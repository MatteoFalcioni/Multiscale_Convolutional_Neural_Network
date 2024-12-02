from scripts.gpu_grid_gen import build_gpu_tree, generate_multiscale_grids_gpu_masked, apply_masks_gpu
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values


class GPU_PointCloudDataset(Dataset):
    def __init__(self, full_data_array, window_sizes, grid_resolution, features_to_use, known_features, device, subset_file=None):
        """
        Dataset class for streaming multiscale grid generation from point cloud data on the GPU.

        Args:
        - full_data_array (numpy.ndarray): The entire point cloud data array (already remapped).
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Grid resolution (e.g., 128x128).
        - features_to_use (list): List of feature names for generating grids.
        - known_features (list): All known feature names in the data array.
        - device (torch.device): device to work on with torch tensors.
        - subset_file (str, optional): Path to a csv file containing coordinates of points to be selected from the full data. If None, all are selected.
        """
        self.device = device
        self.tensor_full_data = torch.tensor(full_data_array, dtype=torch.float64).to(device=self.device)  # too big for gpu?
        
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        feature_indices = [known_features.index(feature) for feature in features_to_use]
        self.feature_indices_tensor = torch.Tensor(feature_indices, device=self.device)
        
        # Build torch kdtree model on the GPU 
        self.gpu_tree = build_gpu_tree(self.tensor_data_array[:, :3])
        
        # Apply masking and compute bounds
        self.selected_array, mask, point_cloud_bounds = apply_masks_gpu(
            tensor_data_array=self.tensor_full_data,
            window_sizes=self.window_sizes,
            subset_file=subset_file
        )
        
        # Store the original indices corresponding to the selected points. We need it to assign correctly the predicted labels during inference.
        self.original_indices = torch.where(mask.cpu())[0].numpy()  # Map from selected array to the original array
        
    def __len__(self):
        return len(self.selected_tensor)

    def __getitem__(self, idx):
        """
        Generates multiscale grids for the point at index `idx` and returns them as PyTorch tensors, along with the index.
        """
        # Extract the single point's data using `idx`
        center_point_tensor = self.selected_tensor[idx, :3]  # Get the x, y, z coordinates
        label = self.selected_tensor[idx, -1].long()  # Get the label for this point as long

        # Generate multiscale grids for this point using the GPU
        grids_dict = generate_multiscale_grids_gpu_masked(
            center_point_tensor=center_point_tensor, tensor_data_array=self.tensor_data_array, window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution, feature_indices_tensor=self.feature_indices_tensor,
            gpu_tree=self.gpu_tree, device=self.device
        )

        # Unpack grids
        small_grid = grids_dict['small']
        medium_grid = grids_dict['medium']
        large_grid = grids_dict['large']

        # Return the grids and label
        return small_grid, medium_grid, large_grid, label, idx


def gpu_prepare_dataloader(batch_size, data_filepath=None, 
                       window_sizes=None, grid_resolution=128, features_to_use=None, 
                       train_split=None, num_workers=4, shuffle_train=True, device='cuda:0', subset_file=None):
    """
    Prepares the DataLoader by loading the raw data and streaming multiscale grid generation on the GPU.
    
    Args:
    - batch_size (int): The batch size to be used for training.
    - data_filepath (str): Path to the raw data (e.g., .las or .csv file). Default is None.
    - window_sizes (list): List of window sizes to use for grid generation. Default is None.
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for grid generation. Default is None.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data). Default is None.
    - features_file_path: File path to feature metadata, needed if using raw data in .npy format. Default is None.
    - num_workers (int): number of workers for parallelized process. Default is 4.
    - shuffle_train (bool): Whether to shuffle the data for training. Default is True.
    - device (str): The device ('cuda' or 'cpu') for tensor operations.
    - subset_file (str, optional): Path to a csv file containing coordinates of points to be selected from the full data. If None, all are selected.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split is not None, else eval_loader=None).
    """
    
    # Check if data directory was passed as input
    if data_filepath is None:
        raise ValueError('ERROR: Data file path was not passed as input to the dataloader.')

    # Read the raw point cloud data 
    data_array, known_features = read_file_to_numpy(data_dir=data_filepath, features_to_use=None)

    # Remap labels to ensure they vary continuously (needed for CrossEntropyLoss)
    data_array, _ = remap_labels(data_array)
    # clean data from nan/inf values
    data_array = clean_nan_values(data_array)

    # Create the dataset 
    full_dataset = GPU_PointCloudDataset(
        full_data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        known_features=known_features,
        device=device,
        subset_file=subset_file
    )

    # Split the dataset into training and evaluation sets (if train_split is provided)
    if train_split is not None:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,  num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        # If no train/test split, create one DataLoader for the full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        eval_loader = None

    return train_loader, eval_loader


