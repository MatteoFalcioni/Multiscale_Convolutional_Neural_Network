import cuml
from cuml.neighbors import NearestNeighbors as cuKNN
import cupy as cp
from torch_kdtree import build_kd_tree
import torch


def build_gpu_tree(data_array, n_neighbors=1):
    data_array_gpu = torch.tensor(data_array[:, :3], dtype=torch.float32).to(device="cuda")
    gpu_kdtree = build_kd_tree(data_array_gpu)
    return gpu_kdtree


def build_cuml_knn(data_array, n_neighbors=1):
        """
        Builds and returns a cuML KNN model on the GPU for nearest neighbor search.
        """
        data_gpu = cp.array(data_array, dtype=cp.float64) 
        cuml_knn = cuKNN(n_neighbors=n_neighbors)
        cuml_knn.fit(data_gpu)
        return cuml_knn


def create_feature_grid_gpu(center_point_tensor, window_size, grid_resolution=128, channels=3):
    """
    Creates a grid around the center point and initializes cells to store feature values, on the GPU.
    Args:
    - center_point_tensor (torch.Tensor): (x, y, z) coordinates of the center of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels in the resulting image.

    Returns:
    - grid (torch.Tensor): A 2D grid initialized to zeros, which will store feature values.
    - cell_size (float): The size of each cell in meters.
    - x_coords (torch.Tensor): Array of x coordinates for the centers of the grid cells.
    - y_coords (torch.Tensor): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros on GPU; each cell will eventually hold feature values
    grid = torch.zeros((grid_resolution, grid_resolution, channels), dtype=torch.float32, device="cuda") 

    # Generate cell coordinates for the grid based on the center point, on the GPU
    i_indices = torch.arange(grid_resolution, device="cuda")
    j_indices = torch.arange(grid_resolution, device="cuda")

    half_resolution_minus_half = (grid_resolution / 2) - 0.5

    x_coords = center_point_tensor[0] - (half_resolution_minus_half - j_indices) * cell_size
    y_coords = center_point_tensor[1] - (half_resolution_minus_half - i_indices) * cell_size

    constant_z = center_point_tensor[2]  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, constant_z


def assign_features_to_grid_gpu(gpu_tree, gpu_data_array, grid, x_coords, y_coords, constant_z, feature_indices, device):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid using torch_kdtree's GPU-accelerated KNN,
    and directly assigns them to the grid on GPU.
    """
    # Generate the grid coordinates on the GPU
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), constant_z, device=device)), dim=-1)

    # Query the GPU KNN model for nearest neighbors using the GPU-based KDTree
    _, indices = gpu_tree.query(grid_coords)

    indices = indices.flatten()
    selected_features = gpu_data_array[indices, :][:, feature_indices]

    grid[:, :, :] = selected_features.reshape(grid.shape)

    return grid


def generate_multiscale_grids_gpu(center_point_tensor, data_array, window_sizes, grid_resolution, feature_indices, gpu_tree, point_cloud_bounds):
    """
    Generates multiscale grids for a single point in the data array using GPU-based operations.
    
    Args:
    - center_point_tensor (torch.Tensor): (x, y, z) coordinates of the point in the point cloud.
    - data_array (torch.Tensor): 2D array containing the point cloud data (on GPU).
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size).
                           Example: [('small', 2.5), ('medium', 5.0), ('large', 10.0)].
    - grid_resolution (int): Resolution of the grid (e.g., 128 for 128x128 grids).
    - feature_indices (list): List of feature indices to be selected from the full list of features.
    - gpu_tree (torch_kdtree): Prebuilt torch_kdtree model for nearest neighbor search.
    - point_cloud_bounds (dict): Dictionary containing point cloud boundaries in every dimension (x, y, z).
    
    Returns:
    - grids_dict (dict): Dictionary of generated grids for each scale.
    - skipped (bool): Flag indicating if the point was skipped.
    """
    # Convert the data array to a tensor if it's not already on GPU
    gpu_data_array = torch.tensor(data_array, dtype=torch.float32).to(device="cuda")
    
    grids_dict = {}  # To store grids for each scale
    channels = len(feature_indices)
    skipped = False

    # Generate grid coordinates directly on the GPU for each scale
    for size_label, window_size in window_sizes:
        half_window = window_size / 2

        # Check if the point is within bounds
        if (center_point_tensor[0] - half_window < point_cloud_bounds['x_min'] or
            center_point_tensor[0] + half_window > point_cloud_bounds['x_max'] or
            center_point_tensor[1] - half_window < point_cloud_bounds['y_min'] or
            center_point_tensor[1] + half_window > point_cloud_bounds['y_max']):
            skipped = True
            break

        # Create grid coordinates on GPU
        grid, _, x_coords, y_coords, z_coord = create_feature_grid_gpu(
            center_point_tensor, window_size, grid_resolution, channels
        )

        # Assign features from the nearest point in the data array using the GPU KNN model
        grid_with_features = assign_features_to_grid_gpu(
            gpu_tree, gpu_data_array, grid, x_coords, y_coords, z_coord, feature_indices
        )

        # Check for NaN or Inf in the grid
        if torch.isnan(grid_with_features).any() or torch.isinf(grid_with_features).any():
            skipped = True
            break

        # Convert grid to PyTorch format (channels first: C, H, W)
        grid_with_features = grid_with_features.permute(2, 0, 1)  # (channels, height, width)
        grids_dict[size_label] = grid_with_features

    return grids_dict, skipped