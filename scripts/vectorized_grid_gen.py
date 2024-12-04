import torch


def vectorized_create_feature_grids(center_points, window_sizes, grid_resolution, channels, device):
    """
    Initialize grids for a batch of center points and multiple scales, fully vectorized.
    
    Args:
    - center_points (torch.Tensor): Tensor of shape (batch_size, 3) with (x, y, z) coordinates.
    - window_sizes (torch.Tensor): Tensor of shape (scales,) with window sizes (floats).
    - grid_resolution (int): Number of cells per dimension.
    - channels (int): Number of feature channels.
    - device (str): CUDA device.

    Returns:
    - cell_sizes (torch.Tensor): Tensor of shape (scales,) with cell sizes for each scale.
    - grids (torch.Tensor): Initialized grids of shape (batch_size, scales, channels, grid_resolution, grid_resolution).
    - grid_coords (torch.Tensor): Flattened grid coordinates of shape (batch_size, scales, grid_resolution^2, 3).
    """
    
    '''Note: put asserts for float64 of center point, need that float64 for matching'''
    batch_size = center_points.shape[0]
    scales = len(window_sizes)

    # Compute cell sizes for each scale
    cell_sizes = window_sizes / grid_resolution  # Shape: (scales,)
    #print(f"Cell sizes for each scale: {cell_sizes}")  # Debugging print

    # Precompute indices for grid resolution
    indices = torch.arange(grid_resolution, device=device)
    i_indices, j_indices = torch.meshgrid(indices, indices, indexing="ij")
    half_resolution_minus_half = torch.tensor((grid_resolution / 2) - 0.5, device=device, dtype=torch.float64)
    
    # Precompute offsets
    x_offsets = (j_indices.flatten() - half_resolution_minus_half).to(dtype=torch.float64)
    y_offsets = (i_indices.flatten() - half_resolution_minus_half).to(dtype=torch.float64)
    # Expand offsets to match scales
    x_offsets = x_offsets.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, grid_resolution^2)
    y_offsets = y_offsets.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, grid_resolution^2)
    # Temporary broadcasting view of cell_sizes
    cell_sizes_broadcasted = cell_sizes.view(1, scales, 1)  # Shape: (1, scales, 1)

    # Compute grid coordinates for all scales and center points
    x_coords = center_points[:, 0:1].unsqueeze(1) + x_offsets * cell_sizes_broadcasted  # Broadcasting produces shape (B, S, res^2)
    y_coords = center_points[:, 1:2].unsqueeze(1) + y_offsets * cell_sizes_broadcasted
    z_coords = center_points[:, 2:3].unsqueeze(1).expand(-1, scales, grid_resolution**2)

    # Stack into grid_coords tensor
    grid_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # Shape: (batch_size, scales, grid_resolution^2, 3)
    print(f"grid coords shape: {grid_coords.shape}, dtype: {grid_coords.dtype}")

    grids = torch.zeros(batch_size, scales, channels, grid_resolution, grid_resolution, dtype=torch.float64, device=device)
    print(f"Initialized grids with shape: {grids.shape}")  # Debugging print

    return cell_sizes, grids, grid_coords

def vectorized_assign_features_to_grids_gpu(gpu_tree, tensor_data_array, grid_coords, grids, feature_indices_tensor, device):
    """
    Assign features to all grids for all scales and batches in a fully vectorized manner.

    Args:
    - gpu_tree (torch_kdtree): KDTree for nearest neighbor search.
    - tensor_data_array (torch.Tensor): Point cloud data (on GPU).
    - grid_coords (torch.Tensor): Flattened grid coordinates (batch_size, scales, grid_resolution^2, 3).
    - grids (torch.Tensor): Initialized grids (batch_size, scales, channels, grid_resolution, grid_resolution).
    - feature_indices_tensor (torch.Tensor): Indices of features to use.
    - device (str): CUDA device.

    Returns:
    - grids (torch.Tensor): Grids filled with features.
    """
    batch_size, scales, num_cells, _ = grid_coords.shape
    channels, grid_resolution = grids.shape[2], grids.shape[-1]

    # Flatten grid_coords for KDTree query
    flattened_coords = grid_coords.view(-1, 3)  # Shape: (batch_size * scales * grid_resolution^2, 3)

    # Query KDTree for nearest neighbors
    _, indices = gpu_tree.query(flattened_coords)  # Shape: (batch_size * scales * grid_resolution^2,)
    print(f"returned indices shape {indices.shape}")
    indices = indices.view(batch_size, scales, num_cells)  # Reshape back
    print(f"returned indices after reshaping {indices.shape}")
    
    # Fetch features for nearest neighbors
    # tensor_data_array[indices, :] -> (batch_size, scales, num_cells, num_features), 
    # [:, :, :, feature_indices_tensor] -> (batch_size, scales, num_cells, channels)
    features = tensor_data_array[indices, :][:, :, :, feature_indices_tensor]  # Shape: (batch_size, scales, num_cells, channels)
    print(f"features shape: {features.shape}")  
    
    # Reshape and assign features to grids
    grids[:] = features.view(batch_size, scales, grid_resolution, grid_resolution, channels).permute(0, 1, 4, 2, 3)

    return grids
