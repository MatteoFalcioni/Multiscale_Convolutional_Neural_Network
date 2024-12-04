from scripts.old_gpu_grid_gen import apply_masks_gpu
from scripts.vectorized_grid_gen import vectorized_generate_multiscale_grids
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values
from torch_kdtree import build_kd_tree


class GPU_PointCloudDataset(Dataset):
    def __init__(self, full_data_array, window_sizes, grid_resolution, features_to_use, known_features, device, subset_file=None):
        """
        Dataset class for streaming multiscale grid generation from point cloud data on the GPU.
        """
        self.device = device
        self.tensor_full_data = torch.tensor(full_data_array, dtype=torch.float64, device=self.device) 
        
        '''fix this thing where you need window sizes in the two formats '''
        self.window_sizes_tensor = torch.tensor([size for _, size in window_sizes], dtype=torch.float64, device=self.device) 
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        self.feature_indices_tensor = torch.tensor([known_features.index(feature) for feature in features_to_use], dtype=torch.int64)

        # Build GPU KDTree 
        self.gpu_tree = build_kd_tree(self.tensor_full_data[:, :3])
        
        # Apply masking and compute bounds
        self.selected_tensor, mask, point_cloud_bounds = apply_masks_gpu(
            tensor_data_array=self.tensor_full_data,
            window_sizes=window_sizes,
            subset_file=subset_file
        )
        
        # Store the original indices for selected points
        self.original_indices = torch.where(mask.cpu())[0].numpy()

    def __len__(self):
        return len(self.selected_tensor)

    def __getitem__(self, idx):
        """
        Fetches the center point, label, and original index for a single point.
        """
        center_point_tensor = self.selected_tensor[idx, :3]  # Center point coordinates
        label = self.selected_tensor[idx, -1].long()  # Label
        original_idx = self.original_indices[idx]  # Map back to original index

        return center_point_tensor, label, original_idx

    def generate_batch_grids(self, batch_center_points):
        """
        Generates multiscale grids for a batch of center points using vectorized implementation.
        """
        grids = vectorized_generate_multiscale_grids(
            center_points=batch_center_points,
            tensor_data_array=self.tensor_full_data,
            window_sizes=self.window_sizes_tensor,
            grid_resolution=self.grid_resolution,
            feature_indices_tensor=self.feature_indices_tensor,
            gpu_tree=self.gpu_tree,
            device=self.device
        )

        return grids
    

def collate_fn(batch):
    """
    Custom collate function to handle batch generation of grids and labels.
    """
    center_points, labels, original_indices = zip(*batch)
    center_points = torch.stack(center_points)
    labels = torch.stack(labels)
    original_indices = torch.tensor(original_indices)

    return center_points, labels, original_indices


def gpu_prepare_dataloader(batch_size, data_filepath=None, window_sizes=None, grid_resolution=128, 
                           features_to_use=None, train_split=None, num_workers=4, shuffle_train=True, device='cuda:0', subset_file=None):
    """
    Prepares the DataLoader with optimized batch processing of grids.
    """
    # Read and preprocess data
    data_array, known_features = read_file_to_numpy(data_dir=data_filepath)
    data_array, _ = remap_labels(data_array)
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

    # Split dataset
    if train_split is not None:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, collate_fn=collate_fn)
        eval_loader = None

    return train_loader, eval_loader



