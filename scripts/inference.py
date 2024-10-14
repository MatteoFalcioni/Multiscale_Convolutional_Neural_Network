import torch
from scripts.point_cloud_to_image import generate_multiscale_grids, compute_point_cloud_bounds
from utils.point_cloud_data_utils import remap_labels
import numpy as np
import csv
from datetime import datetime
import os
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix, classification_report


def inference(model, dataloader, device, save=False, save_dir=None):
    """
    Runs inference on the provided data and returns the predicted and true labels.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the data for inference.
    - device (torch.device): The device (CPU or GPU) where computations will be performed.

    Returns:
    - Confusion matrix and classification report
    """

    model.eval()  # Set model to evaluation mode
    all_preds = []  # To store all predictions
    all_labels = []  # To store all true labels

    with torch.no_grad():  # No gradient calculation during inference
        for batch in dataloader:
            if batch is None:
                continue

            # Unpack the batch
            small_grids, medium_grids, large_grids, labels = batch
            small_grids, medium_grids, large_grids, labels = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
            )

            # Forward pass to get outputs
            outputs = model(small_grids, medium_grids, large_grids)
            preds = torch.argmax(outputs, dim=1)  # Get predicted labels

            # Append predictions and true labels to lists
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate lists to arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report (including precision, recall, F1-score)
    class_report = classification_report(all_labels, all_preds)

    return conf_matrix, class_report


def inference_with_csv(model, data_array, window_sizes, grid_resolution, feature_indices, device, save_file=None, subsample_size=None):
    """
    Perform inference with the MCNN model, generating grids from point cloud points and comparing predicted labels with known true labels.

    Args:
    - model (nn.Module): The trained MCNN model.
    - data_array (np.ndarray): Array of points from the point cloud on which we want to perform inference.
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - feature_indices (list): List of feature indices to be selected from the full list of features.
    - device (torch.device): The device (CPU or GPU) to perform inference on.
    - save_file (str): The full path to the file where labels will be saved.
    - subsample_size (int): Number of points to randomly sample for inference.

    Returns:
    - true_labels_list, predicted_labels_list (lists): True and predicted class labels.
    """
    
    # remap labels to match the remapping in the original data (it was eneded to use cross entropy loss)
    data_array, _ = remap_labels(data_array)

    # Build the KDTree once for the entire dataset
    kdtree = cKDTree(data_array[:, :3])

    # Compute point cloud bounds once
    point_cloud_bounds = compute_point_cloud_bounds(data_array)

    if subsample_size is not None:
        # Subsample points for inference
        subsample_indices = np.random.choice(data_array.shape[0], subsample_size, replace=False)
    else: 
        subsample_indices = range(data_array.shape[0])

    # Initialize lists to store predicted and true labels
    predicted_labels_list = []
    true_labels_list = []

    # Perform inference point by point
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        for idx in subsample_indices:
            center_point = data_array[idx, :3]
            true_label = data_array[idx, -1]  # Get true label from the data array

            # Generate multiscale grids for this point
            grids_dict, skipped = generate_multiscale_grids(
                center_point=center_point,
                data_array=data_array,
                window_sizes=window_sizes,
                grid_resolution=grid_resolution,
                feature_indices=feature_indices,
                kdtree=kdtree,
                point_cloud_bounds=point_cloud_bounds
            )

            if skipped:
                continue  # Skip points with invalid grids

            # Convert grids to tensors and move to device
            small_grid = torch.tensor(grids_dict['small'], dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            medium_grid = torch.tensor(grids_dict['medium'], dtype=torch.float32).unsqueeze(0).to(device)
            large_grid = torch.tensor(grids_dict['large'], dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass through the model
            outputs = model(small_grid, medium_grid, large_grid)

            # Get predicted class label (highest probability for each sample)
            _, predicted_label = torch.max(outputs, dim=1)

            # Append predicted and true labels to the lists
            predicted_labels_list.append(predicted_label.item())
            true_labels_list.append(true_label)

    # Optionally save the true and predicted labels to a file
    # If save_file is provided, ensure the directory exists
    if save_file:
        save_dir = os.path.dirname(save_file)
        os.makedirs(save_dir, exist_ok=True)

        # Add timestamp to filename to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_extension = os.path.splitext(save_file)
        save_file_with_timestamp = f"{file_name}_{timestamp}{file_extension}"

        # Open the file for writing and save the true/predicted labels
        with open(save_file_with_timestamp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True Label', 'Predicted Label'])  # Header
            for true, pred in zip(true_labels_list, predicted_labels_list):
                writer.writerow([int(true), int(pred)])

    return true_labels_list, predicted_labels_list