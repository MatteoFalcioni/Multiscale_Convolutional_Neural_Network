import torch
import pandas as pd
from torch_kdtree import build_kd_tree


def create_feature_grid_gpu(center_point_tensor, device, window_size, grid_resolution=128, channels=3):
    """
    Creates a grid around the center point and initializes cells to store feature values, on the GPU.
    Args:
    - center_point_tensor (torch.Tensor): (x, y, z) coordinates of the center of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels in the resulting image.

    Returns:
    - cell_size (float): The size of each cell in meters.
    - x_coords (torch.Tensor): Array of x coordinates for the centers of the grid cells.
    - y_coords (torch.Tensor): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros on GPU; each cell will eventually hold feature values
    grid = torch.zeros((grid_resolution, grid_resolution, channels), dtype=torch.float64, device=device) 

    # Generate cell coordinates for the grid based on the center point, on the GPU
    i_indices = torch.arange(grid_resolution, device=device)
    j_indices = torch.arange(grid_resolution, device=device)

    half_resolution_minus_half = torch.tensor((grid_resolution / 2) - 0.5, device=device, dtype=torch.float64)

    x_coords = center_point_tensor[0] - (half_resolution_minus_half - j_indices) * cell_size
    y_coords = center_point_tensor[1] - (half_resolution_minus_half - i_indices) * cell_size

    constant_z = center_point_tensor[2]  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, constant_z


def assign_features_to_grid_gpu(gpu_tree, tensor_data_array, grid, x_coords, y_coords, constant_z, feature_indices_tensor, device):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid using torch_kdtree's GPU-accelerated KDTree,
    and directly assigns them to the grid on GPU.
    """
    # Generate the grid coordinates on the GPU
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = torch.stack(
            (
                grid_x.flatten(),
                grid_y.flatten(),
                torch.full((grid_x.numel(),), constant_z, device=device, dtype=torch.float64)
            ),
            dim=-1
        )
    # Query the GPU KNN model for nearest neighbors using the GPU-based KDTree
    _, indices = gpu_tree.query(grid_coords)

    indices_flattened = indices.flatten()

    grid[:, :, :] = tensor_data_array[indices_flattened, :][:, feature_indices_tensor].reshape(grid.shape)

    return grid


def generate_multiscale_grids_gpu_masked(center_point_tensor, tensor_data_array, window_sizes, grid_resolution, feature_indices_tensor, gpu_tree, device="cuda:0"):
    """
    Generate multiscale grids for a single point on GPU without redundant checks.

    Args:
    - center_point_tensor (torch.Tensor): (x, y, z) coordinates of the point.
    - tensor_data_array (torch.Tensor): Point cloud data (on GPU).
    - window_sizes (list): List of tuples with grid window sizes (e.g., [('small', 1.0), ...]).
    - grid_resolution (int): Resolution of the grid.
    - feature_indices_tensor (torch.Tensor): Indices of features to use.
    - gpu_tree (torch_kdtree): KDTree for nearest neighbor search.
    - device (str): CUDA device.

    Returns:
    - grids_dict (dict): Dictionary containing generated grids for each scale.
    """
    grids_dict = {}
    channels = len(feature_indices_tensor)

    for size_label, window_size in window_sizes:
        # Create grid on GPU
        grid, _, x_coords, y_coords, z_coord = create_feature_grid_gpu(
            center_point_tensor, device, window_size, grid_resolution, channels
        )

        # Assign features using KDTree
        grid_with_features = assign_features_to_grid_gpu(
            gpu_tree, tensor_data_array, grid, x_coords, y_coords, z_coord, feature_indices_tensor, device
        )

        # Convert grid to PyTorch format (channels first: C, H, W)
        grid_with_features = grid_with_features.permute(2, 0, 1)
        grids_dict[size_label] = grid_with_features

    return grids_dict


def apply_masks_gpu(tensor_data_array, window_sizes, subset_file=None, tol=1e-8):
    """
    Applies masking operations on a point cloud dataset:
    1. Selects points based on a subset file (if provided) using KDTree-based matching on GPU.
    2. Masks out-of-bounds points based on grid window sizes and dataset bounds.

    Args:
    - tensor_data_array (torch.Tensor): Full point cloud dataset on GPU (shape: [N, features]).
    - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ...]).
    - subset_file (str, optional): Path to a CSV file with subset points (columns: x, y, z).
    - tol (float): Tolerance for approximate matching.

    Returns:
    - selected_tensor (torch.Tensor): The filtered data tensor after applying all masks.
    - final_mask (torch.Tensor): Boolean tensor applied to the original data tensor.
    - bounds (dict): Bounds computed from the full dataset.
    """
    # Initialize mask with all True
    final_mask = torch.ones(tensor_data_array.shape[0], dtype=torch.bool, device=tensor_data_array.device)

    # Apply subset file mask (if provided)
    if subset_file is not None:
        # Read subset points and convert to GPU tensor
        subset_points = torch.tensor(
            pd.read_csv(subset_file)[['x', 'y', 'z']].values,
            dtype=torch.float64,
            device=tensor_data_array.device
        )
        # Build KDTree equivalent on GPU (torch_kdtree assumed for GPU KDTree functionality)
        kdtree = build_kd_tree(subset_points)  # Placeholder for actual GPU KDTree implementation

        # Query KDTree for distances within tolerance
        distances, _ = kdtree.query(tensor_data_array[:, :3], nr_nns_searches=1)
        distances = distances.squeeze()  # Ensure it's a 1D GPU tensor
        subset_mask = distances <= tol  # Points within tolerance
        final_mask &= subset_mask

        print(f"GPU subset mask: {torch.sum(final_mask).item()} points match subset within tolerance {tol}.")

    # Filter the tensor data array based on the combined mask
    selected_tensor = tensor_data_array[final_mask]
    print(f"GPU selected array length after masking with subset: {len(selected_tensor)}")

    # Compute bounds on the full data tensor
    bounds = {
        'x_min': tensor_data_array[:, 0].min().item(),
        'x_max': tensor_data_array[:, 0].max().item(),
        'y_min': tensor_data_array[:, 1].min().item(),
        'y_max': tensor_data_array[:, 1].max().item(),
    }

    # Apply out-of-bounds mask
    max_half_window = max(window_size / 2 for _, window_size in window_sizes)
    out_of_bounds_mask = (
        (selected_tensor[:, 0] - max_half_window >= bounds['x_min']) &
        (selected_tensor[:, 0] + max_half_window <= bounds['x_max']) &
        (selected_tensor[:, 1] - max_half_window >= bounds['y_min']) &
        (selected_tensor[:, 1] + max_half_window <= bounds['y_max'])
    )

    # Update the final mask to include out-of-bounds filtering
    final_mask[torch.where(final_mask)[0]] &= out_of_bounds_mask
    selected_tensor = tensor_data_array[final_mask]
    print(f"GPU selected array length after masking out of bounds: {len(selected_tensor)}")

    return selected_tensor, final_mask, bounds













'''def isin_tolerance_gpu(A, B, tol):
    """
    Checks if elements of tensor A are approximately in tensor B within a specified tolerance using GPU operations.

    Args:
    - A (torch.Tensor): Tensor to check (shape: [N]).
    - B (torch.Tensor): Tensor to match against (shape: [M]).
    - tol (float): Tolerance for approximate matching.

    Returns:
    - mask (torch.Tensor): Boolean tensor indicating matches within tolerance (shape: [N]).
    """
    # Ensure tensors are contiguous
    A = A.contiguous()
    B_sorted, _ = torch.sort(B.contiguous())

    # Perform searchsorted and clamp indices
    idx = torch.searchsorted(B_sorted, A)
    idx_clamped = torch.clamp(idx, max=len(B_sorted) - 1)

    # Compute distances to nearest neighbors
    lval = torch.abs(A - B_sorted.gather(0, idx_clamped))
    idx1 = torch.clamp(idx - 1, min=0)
    rval = torch.abs(A - B_sorted.gather(0, idx1))

    # Return mask for elements within tolerance
    return (torch.min(lval, rval) <= tol)'''



'''
def apply_masks_gpu(tensor_data_array, window_sizes, subset_file=None, tol=1e-8):
    """
    Applies masking operations on a point cloud dataset:
    1. Selects points based on a subset file (if provided) using tolerance-based matching.
    2. Masks out-of-bounds points based on grid window sizes and dataset bounds.

    Args:
    - tensor_data_array (torch.Tensor): Full point cloud dataset on GPU (shape: [N, features]).
    - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ...]).
    - subset_file (str, optional): Path to a CSV file with subset points (columns: x, y, z).
    - tol (float): Tolerance for approximate matching.

    Returns:
    - selected_tensor (torch.Tensor): The filtered data tensor after applying all masks.
    - final_mask (torch.Tensor): Boolean tensor applied to the original data tensor.
    - bounds (dict): Bounds computed from the full dataset.
    """
    # Initialize mask with all True
    final_mask = torch.ones(tensor_data_array.shape[0], dtype=torch.bool, device=tensor_data_array.device)

    # Apply subset file mask (if provided)
    if subset_file is not None:
        subset_points = torch.tensor(
            pd.read_csv(subset_file)[['x', 'y', 'z']].values,
            dtype=torch.float64,  # Match numpy default
            device=tensor_data_array.device
        )
        for i in range(3):  # Apply isin_tolerance_gpu for each coordinate
            subset_mask = isin_tolerance_gpu(tensor_data_array[:, i], subset_points[:, i], tol)
            final_mask &= subset_mask

        print(f"GPU subset mask: {torch.sum(final_mask).item()} points match subset within tolerance {tol}.")

    # Filter the tensor data array based on the combined mask
    selected_tensor = tensor_data_array[final_mask]
    print(f"GPU selected array length after masking with subset: {len(selected_tensor)}")

    # Compute bounds on the full data tensor
    bounds = {
        'x_min': tensor_data_array[:, 0].min().item(),
        'x_max': tensor_data_array[:, 0].max().item(),
        'y_min': tensor_data_array[:, 1].min().item(),
        'y_max': tensor_data_array[:, 1].max().item(),
    }

    # Apply out-of-bounds mask
    max_half_window = max(window_size / 2 for _, window_size in window_sizes)
    out_of_bounds_mask = (
        (selected_tensor[:, 0] - max_half_window >= bounds['x_min']) &
        (selected_tensor[:, 0] + max_half_window <= bounds['x_max']) &
        (selected_tensor[:, 1] - max_half_window >= bounds['y_min']) &
        (selected_tensor[:, 1] + max_half_window <= bounds['y_max'])
    )

    final_mask[torch.where(final_mask)[0]] &= out_of_bounds_mask
    selected_tensor = tensor_data_array[final_mask]
    print(f"GPU selected array length after masking out of bounds: {len(selected_tensor)}")

    return selected_tensor, final_mask, bounds



def old_generate_multiscale_grids_gpu(center_point_tensor, tensor_data_array, window_sizes, grid_resolution, feature_indices_tensor, gpu_tree, point_cloud_bounds, device):
    """
    Generates multiscale grids for a single point in the data array using GPU-based operations.
    
    Args:
    - center_point_tensor (torch.Tensor): (x, y, z) coordinates of the point in the point cloud.
    - tensor_data_array (torch.Tensor): 2D array containing the point cloud data (on GPU).
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size).
                           Example: [('small', 2.5), ('medium', 5.0), ('large', 10.0)].
    - grid_resolution (int): Resolution of the grid (e.g., 128 for 128x128 grids).
    - feature_indices_tensor (torch.Tensor): List of feature indices to be selected from the full list of features, in tensor form.
    - gpu_tree (torch_kdtree): Prebuilt torch_kdtree model for nearest neighbor search.
    - point_cloud_bounds (dict): Dictionary containing point cloud boundaries in every dimension (x, y, z).
    
    Returns:
    - grids_dict (dict): Dictionary of generated grids for each scale.
    - status (str or None): "out_of_bounds", "nan/inf", or None (valid).
    """
    
    grids_dict = {}  # To store grids for each scale
    channels = len(feature_indices_tensor)
    status = None   # Default: point is not skipped

    # Generate grid coordinates directly on the GPU for each scale
    for size_label, window_size in window_sizes:
        half_window = window_size / 2

        # Check if the point is within bounds
        if (center_point_tensor[0] - half_window < point_cloud_bounds['x_min'] or
            center_point_tensor[0] + half_window > point_cloud_bounds['x_max'] or
            center_point_tensor[1] - half_window < point_cloud_bounds['y_min'] or
            center_point_tensor[1] + half_window > point_cloud_bounds['y_max']):
            status = "out_of_bounds"
            break

        # Create grid coordinates on GPU
        grid, _, x_coords, y_coords, z_coord = create_feature_grid_gpu(
            center_point_tensor, device, window_size, grid_resolution, channels
        )

        # Assign features from the nearest point in the data array using the GPU KNN model
        grid_with_features = assign_features_to_grid_gpu(
            gpu_tree, tensor_data_array, grid, x_coords, y_coords, z_coord, feature_indices_tensor, device
        )

        # Check for NaN or Inf in the grid
        if torch.isnan(grid_with_features).any() or torch.isinf(grid_with_features).any():
            status = "nan/inf"
            break

        # Convert grid to PyTorch format (channels first: C, H, W)
        grid_with_features = grid_with_features.permute(2, 0, 1)  # (channels, height, width)
        grids_dict[size_label] = grid_with_features

    return grids_dict, status


def mask_out_of_bounds_points_gpu(tensor_data_array, window_sizes, point_cloud_bounds):
    """
    Masks points that are too close to the boundaries of the dataset using GPU operations and precomputed bounds.

    Args:
    - tensor_data_array (torch.Tensor): Point cloud data on GPU (shape: [N, 3]).
    - point_cloud_bounds (dict): Precomputed bounds of the point cloud {'x_min', 'x_max', 'y_min', 'y_max'}.
    - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 1.0), ...]).

    Returns:
    - masked_tensor (torch.Tensor): Points that are not out of bounds.
    - mask (torch.Tensor): Boolean tensor indicating valid points.
    """
    max_half_window = max(window_size / 2 for _, window_size in window_sizes)

    # Extract bounds
    x_min, x_max = point_cloud_bounds['x_min'], point_cloud_bounds['x_max']
    y_min, y_max = point_cloud_bounds['y_min'], point_cloud_bounds['y_max']

    # Apply mask logic on GPU
    mask = (
        (tensor_data_array[:, 0] - max_half_window >= x_min) &
        (tensor_data_array[:, 0] + max_half_window <= x_max) &
        (tensor_data_array[:, 1] - max_half_window >= y_min) &
        (tensor_data_array[:, 1] + max_half_window <= y_max) 
    )

    masked_tensor = tensor_data_array[mask]  # Apply mask to get valid points
    return masked_tensor, mask'''

'''def build_cuml_knn(data_array, n_neighbors=1):
        """
        Builds and returns a cuML KNN model on the GPU for nearest neighbor search.
        """
        data_gpu = cp.array(data_array, dtype=cp.float64) 
        cuml_knn = cuKNN(n_neighbors=n_neighbors)
        cuml_knn.fit(data_gpu)
        return cuml_knn'''