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
    batch_size = center_points.shape[0]
    scales = len(window_sizes)

    # Step 1: Compute cell sizes for each scale
    cell_sizes = window_sizes / grid_resolution
    print(f"Cell sizes for each scale: {cell_sizes}")  # Debugging print

    # Step 2: Precompute indices for grid resolution
    indices = torch.arange(grid_resolution, device=device)
    i_indices, j_indices = torch.meshgrid(indices, indices, indexing="ij")
    print(f"Grid indices shape: i: {i_indices.shape}, j: {j_indices.shape}")  # Debugging print

    half_resolution = grid_resolution / 2

    # Step 3: Compute grid coordinates dynamically for all batches and scales
    grid_coords = torch.empty(batch_size, scales, grid_resolution**2, 3, device=device)

    for scale_idx, cell_size in enumerate(cell_sizes):
        print(f"\nProcessing scale {scale_idx + 1}/{scales}, cell size: {cell_size}")  # Debugging print

        # Dynamic calculation for all center points
        x_coords = center_points[:, 0:1] + (j_indices.flatten() - half_resolution) * cell_size
        y_coords = center_points[:, 1:2] + (i_indices.flatten() - half_resolution) * cell_size
        z_coords = center_points[:, 2:3]

        print(f"x_coords shape: {x_coords.shape}")  # Debugging print
        print(f"y_coords shape: {y_coords.shape}")  # Debugging print
        print(f"z_coords shape: {z_coords.shape}")  # Debugging print

        # Store grid coordinates
        grid_coords[:, scale_idx, :, 0] = x_coords
        grid_coords[:, scale_idx, :, 1] = y_coords
        grid_coords[:, scale_idx, :, 2] = z_coords

    # Step 4: Initialize grids (empty for now)
    grids = torch.zeros(batch_size, scales, channels, grid_resolution, grid_resolution, device=device)
    print(f"Initialized grids with shape: {grids.shape}")  # Debugging print

    return cell_sizes, grids, grid_coords
