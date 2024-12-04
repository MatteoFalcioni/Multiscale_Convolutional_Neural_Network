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
