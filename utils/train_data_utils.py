import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import os
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from scripts.point_cloud_to_image import generate_multiscale_grids, compute_point_cloud_bounds
from datetime import datetime
import pandas as pd
from scipy.spatial import cKDTree
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from models.mcnn import MultiScaleCNN


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PointCloudDataset(Dataset):
    def __init__(self, data_array, window_sizes, grid_resolution, features_to_use, known_features):
        """
        Dataset class for streaming multiscale grid generation from point cloud data.

        Args:
        - data_array (numpy.ndarray): The entire point cloud data array (already remapped).
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Grid resolution (e.g., 128x128).
        - features_to_use (list): List of feature names for generating grids.
        - known_features (list): All known feature names in the data array.
        """
        self.data_array = data_array
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        
        # Build KDTree once for the entire dataset
        self.kdtree = cKDTree(data_array[:, :3])  # Use coordinates for KDTree
        self.feature_indices = [known_features.index(feature) for feature in features_to_use]
        
        self.point_cloud_bounds = compute_point_cloud_bounds(data_array)

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        """
        Generates multiscale grids for the point at index `idx` and returns them as PyTorch tensors.
        """
        # Extract the single point's data using `idx`
        center_point = self.data_array[idx, :3]  # Get the x, y, z coordinates
        label = self.data_array[idx, -1]  # Get the label for this point

        # Generate multiscale grids for this point
        grids_dict, skipped = generate_multiscale_grids(center_point, data_array=self.data_array, window_sizes=self.window_sizes, grid_resolution=self.grid_resolution, feature_indices=self.feature_indices, kdtree=self.kdtree, point_cloud_bounds=self.point_cloud_bounds)

        if not skipped: 
            # Convert grids to PyTorch tensors
            small_grid = torch.tensor(grids_dict['small'], dtype=torch.float32)
            medium_grid = torch.tensor(grids_dict['medium'], dtype=torch.float32)
            large_grid = torch.tensor(grids_dict['large'], dtype=torch.float32)

            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)
        else:
            return None

        # Return the grids and label
        return small_grid, medium_grid, large_grid, label
    

def custom_collate_fn(batch):
    """
    Custom collate function to filter out None values (skipped points).
    """
    # Filter out any None values (i.e., skipped points)
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty (all points were skipped), return None
    if len(batch) == 0:
        return None
    
    # Unpack the batch into grids and labels
    small_grids, medium_grids, large_grids, labels = zip(*batch)
    
    # Stack the grids and labels to create tensors for the batch
    small_grids = torch.stack(small_grids)
    medium_grids = torch.stack(medium_grids)
    large_grids = torch.stack(large_grids)
    labels = torch.stack(labels)
    
    return small_grids, medium_grids, large_grids, labels
    


def prepare_dataloader(batch_size, data_dir=None, 
                       window_sizes=None, grid_resolution=128, features_to_use=None, 
                       train_split=0.8, features_file_path=None, num_workers = 4):
    """
    Prepares the DataLoader by loading the raw data and streaming multiscale grid generation.
    
    Args:
    - batch_size (int): The batch size to be used for training.
    - data_dir (str): Path to the raw data (e.g., .las or .csv file). Default is None.
    - window_sizes (list): List of window sizes to use for grid generation. Default is None.
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for grid generation. Default is None.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data). Default is 0.8 (80%).
    - features_file_path: File path to feature metadata, needed if using raw data in .npy format. Default is None.
    - num_workers (int): number of worlkers for parallelized process. Default is 0. 

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split > 0).
    """
    
    # check if raw data directory was passed as input
    if data_dir is None:
        raise ValueError('ERROR: Raw data directory was not passe as input to the dataloader.')

    # Step 1: Read the raw point cloud data into memory
    data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=None, features_file_path=features_file_path)   # with None as features_to_use we get the known features (all the feats in the data)
    
    # Step 2: remap labels to esnure they vary continously (needed for CrossEntropyLoss)
    data_array, _ = remap_labels(data_array)

    # Step 2: Create the dataset using the new streaming-based approach
    full_dataset = PointCloudDataset(
        data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        known_features=known_features,
    )

    # Step 3: Split the dataset into training and evaluation sets (if train_split is provided)
    if train_split > 0.0:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Step 6: Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)
    else:
        # If no train/test split, create one DataLoader for the full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)
        eval_loader = None

    return train_loader, eval_loader


def save_model(model, save_dir='models/saved', used_features=None, hyperparameters=None):
    """
    Saves the PyTorch model along with the features and hyperparameters used in training.

    Args:
    - model (nn.Module): The trained model to be saved.
    - save_dir (str): The directory where the model and parameters will be saved. Default is 'models/saved'.
    - used_features (list): List of features used for training the model.
    - hyperparameters (dict): Dictionary of hyperparameters used during training.

    Returns:
    - model_save_foder (str): Name of the folder where the model is saved. Needed later for inference.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the model folder with the timestamp
    model_save_folder = os.path.join(save_dir, f"mcnn_model_{current_time}")
    os.makedirs(model_save_folder, exist_ok=True)

    # Save the model in the folder
    model_filename = f"model.pth"
    model_save_path = os.path.join(model_save_folder, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save the used features and hyperparameters
    if used_features or hyperparameters:
        save_used_parameters(used_features, hyperparameters, model_save_folder)

    return model_save_folder


def load_model(model_path, device, num_channels, num_classes):
    """
    Loads a saved PyTorch model, initializes it, and sets it to evaluation mode.

    Args:
    - model_path (str): Path to the saved model state dictionary.
    - device (torch.device): Device where the model will be loaded (CPU or GPU).
    - num_channels (int): Number of input channels for the model.
    - num_classes (int): Number of output classes for the model.

    Returns:
    - model (torch.nn.Module): The loaded and initialized model set to evaluation mode.
    """
    # Initialize the model
    model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)
    
    # Load the saved model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

    
    
def save_used_parameters(used_features=None, hyperparameters=None, save_dir='models/saved'):
    """
    Saves the features and hyperparameters used during training to a CSV file.

    Args:
    - used_features (list): List of features used in training.
    - hyperparameters (dict): Dictionary (name, value) of hyperparameters used during training. 
    - save_dir (str): Directory where the used features and the model's hyperparameters are to be saved.

    Returns:
    - None
    """

    # Save the features
    if used_features is not None:
        feature_file = os.path.join(save_dir, 'features_used.csv')
        with open(feature_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Used Features'])  # Header
            writer.writerow(used_features)
        print(f"Features saved to {feature_file}")
    else:
        print('used features were not specified in saving, so they could not be saved together with the model.')

    # Save the hyperparameters
    if hyperparameters is not None:
        hyperparameters_file = os.path.join(save_dir, 'hyperparameters.csv')
        with open(hyperparameters_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hyperparameter', 'Value'])  # Header
            for param, value in hyperparameters.items():
                writer.writerow([param, value])
        print(f"Hyperparameters saved to {hyperparameters_file}")
    else:
        print('hyperparameters were not specified in saving, so they could not be saved together with the model.')


def save_inference_results(conf_matrix, class_report, save_dir, class_names):
    """
    Saves the confusion matrix as an image and the classification report as a CSV file.

    Args:
    - conf_matrix (np.array): The confusion matrix.
    - class_report (str): The classification report as a string.
    - save_dir (str): Directory where the files should be saved.
    - class_names (list): List of class names for labeling the confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save confusion matrix as an image
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved at {confusion_matrix_path}")
    plt.close()

    # Parse and save classification report as CSV
    report_data = []
    lines = class_report.split('\n')
    for line in lines[2:-3]:
        row_data = line.split()
        if len(row_data) > 0:
            report_data.append({
                'class': row_data[0],
                'precision': row_data[1],
                'recall': row_data[2],
                'f1-score': row_data[3],
                'support': row_data[4],
            })
    
    df = pd.DataFrame.from_records(report_data)
    class_report_path = os.path.join(save_dir, 'classification_report.csv')
    df.to_csv(class_report_path, index=False)
    print(f"Classification report saved at {class_report_path}")







