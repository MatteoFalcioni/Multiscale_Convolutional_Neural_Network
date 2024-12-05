import numpy as np
from scipy.spatial import cKDTree

def numpy_create_feature_grids(center_points, window_sizes, grid_resolution, channels):
    """
    Initialize grids for a batch of center points and multiple scales, fully vectorized using NumPy.

    Args:
    - center_points (np.ndarray): Array of shape (batch_size, 3) with (x, y, z) coordinates.
    - window_sizes (np.ndarray): Array of shape (scales,) with window sizes (floats).
    - grid_resolution (int): Number of cells per dimension.
    - channels (int): Number of feature channels.

    Returns:
    - cell_sizes (np.ndarray): Array of shape (scales,) with cell sizes for each scale.
    - grids (np.ndarray): Initialized grids of shape (batch_size, scales, channels, grid_resolution, grid_resolution).
    - grid_coords (np.ndarray): Flattened grid coordinates of shape (batch_size, scales, grid_resolution^2, 3).
    """
    batch_size = center_points.shape[0]
    scales = len(window_sizes)

    # Compute cell sizes for each scale
    cell_sizes = window_sizes / grid_resolution  # Shape: (scales,)

    # Precompute indices for grid resolution
    indices = np.arange(grid_resolution)
    i_indices, j_indices = np.meshgrid(indices, indices, indexing="ij")
    half_resolution_minus_half = (grid_resolution / 2) - 0.5

    # Precompute offsets
    x_offsets = (j_indices.flatten() - half_resolution_minus_half).astype(np.float64)  # (grid_resolution^2,)
    y_offsets = (i_indices.flatten() - half_resolution_minus_half).astype(np.float64)

    # Compute grid coordinates for all scales and center points
    x_coords = center_points[:, 0:1, np.newaxis] + x_offsets[np.newaxis, np.newaxis, :] * cell_sizes[np.newaxis, :, np.newaxis]
    y_coords = center_points[:, 1:2, np.newaxis] + y_offsets[np.newaxis, np.newaxis, :] * cell_sizes[np.newaxis, :, np.newaxis]
    z_coords = np.tile(center_points[:, 2:3, np.newaxis], (1, scales, grid_resolution**2))

    # Stack into grid_coords array
    grid_coords = np.stack([x_coords, y_coords, z_coords], axis=-1)  # Shape: (batch_size, scales, grid_resolution^2, 3)

    # Initialize empty grids
    grids = np.zeros((batch_size, scales, channels, grid_resolution, grid_resolution), dtype=np.float64)

    return cell_sizes, grids, grid_coords


def numpy_assign_features_to_grids(cpu_tree, full_data_array, grid_coords, grids, feature_indices):
    """
    Assign features to all grids for all scales and batches in a fully vectorized manner using NumPy.

    Args:
    - cpu_tree (scipy.spatial.KDTree): KDTree for nearest neighbor search.
    - full_data_array (np.ndarray): Point cloud data.
    - grid_coords (np.ndarray): Flattened grid coordinates (batch_size, scales, grid_resolution^2, 3).
    - grids (np.ndarray): Initialized grids (batch_size, scales, channels, grid_resolution, grid_resolution).
    - feature_indices (np.ndarray): Indices of features to use.

    Returns:
    - grids (np.ndarray): Grids filled with features.
    """
    batch_size, scales, num_cells, _ = grid_coords.shape
    grid_resolution = grids.shape[-1]

    # Flatten grid_coords for KDTree query
    flattened_coords = grid_coords.reshape(-1, 3)  # Shape: (batch_size * scales * grid_resolution^2, 3)

    # Query KDTree for nearest neighbors
    _, indices = cpu_tree.query(flattened_coords)  # Shape: (batch_size * scales * grid_resolution^2,)

    # Fetch features for nearest neighbors
    features = full_data_array[indices][:, feature_indices]  # Shape: (batch_size * scales * grid_resolution^2, channels)

    # Reshape and assign features to grids
    reshaped_features = features.reshape(batch_size, scales, grid_resolution, grid_resolution, len(feature_indices))
    grids[:] = np.transpose(reshaped_features, axes=(0, 1, 4, 2, 3))

    return grids


def numpy_generate_multiscale_grids(center_points, full_data_array, window_sizes, grid_resolution, feature_indices, cpu_tree):
    """
    Generate multiscale grids for a batch of points using NumPy.

    Args:
    - center_points (np.ndarray): (batch_size, 3) coordinates of the points.
    - full_data_array (np.ndarray): Point cloud data.
    - window_sizes (np.ndarray): Array of window sizes (floats).
    - grid_resolution (int): Resolution of the grid.
    - feature_indices (np.ndarray): Indices of features to use.
    - cpu_tree (scipy.spatial.KDTree): KDTree for nearest neighbor search.

    Returns:
    - grids (np.ndarray): Generated grids filled with features.
    """
    # Generate grids and grid coordinates
    cell_sizes, grids, grid_coords = numpy_create_feature_grids(
        center_points, window_sizes, grid_resolution, len(feature_indices)
    )

    # Assign features to grids
    grids = numpy_assign_features_to_grids(
        cpu_tree, full_data_array, grid_coords, grids, feature_indices
    )

    return grids
