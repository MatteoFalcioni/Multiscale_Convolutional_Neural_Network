import numpy as np
from scipy.spatial import cKDTree

def compute_point_cloud_bounds(data_array, padding=0.0):
    """
    Computes the spatial boundaries (min and max) of the point cloud data.
    """
    x_min = data_array[:, 0].min() - padding
    x_max = data_array[:, 0].max() + padding
    y_min = data_array[:, 1].min() - padding
    y_max = data_array[:, 1].max() + padding

    bounds_dict = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
    return bounds_dict

def batched_create_feature_grids(center_points, window_size, grid_resolution=128, channels=3):
    """
    Creates multiple grids for a batch of center points.
    """
    cell_size = window_size / grid_resolution
    grids = np.zeros((len(center_points), grid_resolution, grid_resolution, channels))

    x_coords_batch, y_coords_batch = [], []
    half_res_minus_half = (grid_resolution / 2) - 0.5

    for center_point in center_points:
        i_indices = np.arange(grid_resolution)
        j_indices = np.arange(grid_resolution)
        
        x_coords = center_point[0] - (half_res_minus_half - j_indices) * cell_size
        y_coords = center_point[1] - (half_res_minus_half - i_indices) * cell_size
        x_coords_batch.append(x_coords)
        y_coords_batch.append(y_coords)

    z_coords = np.full(len(center_points), center_points[0][2])
    return grids, cell_size, x_coords_batch, y_coords_batch, z_coords

def batched_assign_features_to_grids(tree, data_array, grids, x_coords_batch, y_coords_batch, z_coords, feature_indices):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid for a batch of grids.
    """
    for idx, (x_coords, y_coords) in enumerate(zip(x_coords_batch, y_coords_batch)):
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, z_coords[idx])), axis=-1)
        
        _, indices = tree.query(grid_coords)
        grids[idx, :, :, :] = data_array[indices, :][:, feature_indices].reshape(grids[idx].shape)

    return grids

def generate_batched_multiscale_grids(center_points, data_array, window_sizes, grid_resolution, feature_indices, kdtree, point_cloud_bounds):
    """
    Generates multiscale grids for multiple center points in a single batch.
    """
    batched_grids = {scale_label: [] for scale_label, _ in window_sizes}
    skipped = []

    for center_point in center_points:
        grids_dict = {}
        point_skipped = False

        for size_label, window_size in window_sizes:
            half_window = window_size / 2

            # Check bounds for the center point
            if (center_point[0] - half_window < point_cloud_bounds['x_min'] or
                center_point[0] + half_window > point_cloud_bounds['x_max'] or
                center_point[1] - half_window < point_cloud_bounds['y_min'] or
                center_point[1] + half_window > point_cloud_bounds['y_max']):
                point_skipped = True
                break

            # Create grid coordinates for all cells around the center point
            grid, cell_size, x_coords, y_coords, z_coord = batched_create_feature_grids(
                center_points, window_size, grid_resolution, channels=len(feature_indices)
            )

            # Assign features to the grid cells
            grid_with_features = batched_assign_features_to_grids(kdtree, data_array, grid, x_coords, y_coords, z_coord, feature_indices)
            grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

            if np.isnan(grid_with_features).any() or np.isinf(grid_with_features).any():
                point_skipped = True
                break

            grids_dict[size_label] = grid_with_features

        if point_skipped:
            skipped.append(True)
        else:
            for size_label, grid in grids_dict.items():
                batched_grids[size_label].append(grid)
            skipped.append(False)

    return batched_grids, skipped
