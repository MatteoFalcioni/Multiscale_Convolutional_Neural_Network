import torch
import numpy as np
import pandas as pd
import laspy
from utils.train_data_utils import prepare_dataloader
from utils.point_cloud_data_utils import subtiler, stitch_subtiles
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch.multiprocessing as mp
import sys
import shutil



def predict(file_path, model, model_path, device, batch_size, window_sizes, grid_resolution, features_to_use, num_workers, min_points=1000000, tile_size=50):
    """
    Checks if a LAS file is large, eventually subtiles it, performs inference on each subtile, 
    and return the predictions stitched together. The function also deletes the subtiles once they have been processed. 
    
    Args:
    - file_path (str): Path to the input LAS file.
    - model (nn.Module): The trained PyTorch model.
    - model_path (str): File path to where the trained PyTorch model is stored.
    - device (torch.device): Device (CPU or GPU) to perform inference on.
    - batch_size (int): The batch size to use for inference.
    - window_sizes (list): List of window sizes for grid preparation.
    - grid_resolution (int): Grid resolution used for data preprocessing.
    - features_to_use (list): List of features used for training.
    - num_workers (int): Number of workers for loading data.
    - min_points (int): Minimum number of points to decide if the file should be subtiled. Default is 1 million.
    - tile_size (int): Size of each subtile in meters.
    - overlap_size (int): Size of the overlap between subtiles in meters.

    Returns:
    - None: This function performs inference and saves results to disk.
    """
    # get the model direcotry from its path
    model_directory = os.path.dirname(model_path)
    
    # Load the original LAS file
    las_file = laspy.read(file_path)
    total_points = len(las_file.x)

    # get overlap size from window sizes: it's the dimension of the largest window size
    overlap_size = int([value for label, value in window_sizes if label == 'large'][0])
    # print(f'overlap size: {overlap_size}')

    # print(f"Total points in the file: {total_points}")
    
    # If the file has more than 'min_points', we proceed with subtile logic
    if total_points > min_points:
        print(f"File is too big to be processed in one go. Subtiling is needed before processing...\n")
        
        # Call subtiler function to split the file into subtiles and save them
        subtile_folder = subtiler(file_path, tile_size, overlap_size) 
        
        # Once subtiles are generated, we perform inference on each of them
        prediction_folder = predict_subtiles(subtile_folder, model, device, batch_size, window_sizes, grid_resolution, features_to_use, num_workers)

        # stitch subtiles back together to construct final file with predictions
        output_filepath = stitch_subtiles(subtile_folder=prediction_folder, original_las=las_file, original_filename=file_path, model_directory=model_directory, overlap_size=overlap_size)

        # Teardown: Remove the subtile folder and its content
        # shutil.rmtree(subtile_folder)  # Removes the entire sub-tile folder

        print(f'\nInference completed succesfully. File saved at {output_filepath}')
            
    else:
        print(f"File has less than {min_points} points. Performing inference directly on the entire file.")
        
        # adapt predict_subtiles logic to handle a file with less than min points


def predict_subtiles(subtile_folder, model, device, batch_size, window_sizes, grid_resolution, features_to_use, num_workers):
    """
    Prepares the DataLoader for the given subtile file, runs inference and updates
    the labels of the file with the predicted labels.

    Args:
    - subtile_folder (str): Directory where the subtiles are stored.
    - model (nn.Module): The trained PyTorch model.
    - device (torch.device): Device (CPU or GPU) to perform inference on.
    - batch_size (int): The batch size to use for inference.
    - window_sizes (list): List of window sizes for grid preparation.
    - grid_resolution (int): Grid resolution used for data preprocessing.
    - features_to_use (list): List of features used for training.
    - num_workers (int): Number of workers for loading data.

    Returns:
    - prediction_folder (str): subfolder where the predicted subtiles have been saved.
    """

    # Get all subtile files from the subtile folder
    subtile_files = [os.path.join(subtile_folder, f) for f in os.listdir(subtile_folder) if f.endswith('.las')]

    # Create a subfolder for predicted subtiles
    predictions_folder = os.path.join(subtile_folder, 'subtiles_predicted')
    os.makedirs(predictions_folder, exist_ok=True)
    
    file_counter = 0
    total_files = len(subtile_files)
    
    # Iterate over all subtiles and run inference
    for file_path in subtile_files: 
        file_counter += 1 
        
        print(f'Processing subtile {file_path} : {file_counter}/{total_files}')
        
        # Prepare the DataLoader for the current file
        inference_loader, _ = prepare_dataloader(
            batch_size=batch_size,
            data_filepath=file_path,
            window_sizes=window_sizes,
            grid_resolution=grid_resolution,
            features_to_use=features_to_use,
            train_split=None,  # no train/eval split
            num_workers=num_workers,
            shuffle_train=False  # don't shuffle data for inference
        )

        # Open the subtile file to copy header information
        original_subtile_file = laspy.read(file_path)
        header = original_subtile_file.header

        # Check and add 'label' as needed
        if 'label' not in header.point_format.dimension_names:
            extra_dims = [laspy.ExtraBytesParams(name="label", type=np.int8)]
            header.add_extra_dims(extra_dims)

        # Initialize the label field with -1 values (-1 = not classified)
        label_array = np.full(len(original_subtile_file.x), -1, dtype=np.int8)

        # Perform inference 
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():  # No gradient calculation during inference
            for batch in tqdm(inference_loader, desc="Performing inference"):
                if batch is None:
                    continue

                # Unpack the batch and move to the correct device
                small_grids, medium_grids, large_grids, _, indices = batch
                small_grids, medium_grids, large_grids = (
                    small_grids.to(device), medium_grids.to(device), large_grids.to(device)
                )

                # Run model inference on the current batch
                outputs = model(small_grids, medium_grids, large_grids)
                preds = torch.argmax(outputs, dim=1)  # Get predicted labels

                # Assign predictions directly to the label array for the current subtile
                cpu_preds = preds.cpu()
                label_array[indices] = cpu_preds.numpy()   

        # Save the updated subtile with predictions written into the 'label' field
        new_las = laspy.LasData(header)
        new_las.points = original_subtile_file.points  # Copy original points
        new_las.label = label_array  # Assign the labels to the new dimension
        
        # Save the file to the predictions folder with a '_pred' suffix
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        pred_file_path = os.path.join(predictions_folder, f"{base_name}_pred{ext}")
        new_las.write(pred_file_path)

    return predictions_folder


def evaluate_model(model, dataloader, device, class_names, model_save_folder, inference_file_path, save=False):
    """
    Runs inference on the provided data, returns the confusion matrix and classification report,
    and optionally saves them to the 'inference' subfolder of the model save directory.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the data for inference.
    - device (torch.device): The device (CPU or GPU) where computations will be performed.
    - class_names (list): List of class names for displaying in the confusion matrix.
    - model_save_folder (str): The folder where the model is saved and where inference results will be saved.
    - inference_file_path (str): The path to the file used for inference.
    - save (bool): If True, saves the confusion matrix and classification report.

    Returns:
    - conf_matrix (np.ndarray): The confusion matrix.
    - class_report (str): The classification report as a string.
    """
    # Check that model_save_folder is the actual directory and not a filepath to the model
    if model_save_folder.endswith('.pth'):
        model_save_folder = os.path.dirname(model_save_folder)
    
    # Create 'inference' subfolder within the model save folder, with a time stamp suffix to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_dir = os.path.join(model_save_folder, f'inference_{timestamp}')
    os.makedirs(inference_dir, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    all_preds = []  # To store all predictions
    all_labels = []  # To store all true labels

    with torch.no_grad():  # No gradient calculation during inference
        for batch in tqdm(dataloader, desc="Performing model evaluation"):
            if batch is None:
                continue

            # Unpack the batch
            small_grids, medium_grids, large_grids, labels, _ = batch
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
    
    # Classification report
    class_report = classification_report(all_labels, all_preds)

    # Save the confusion matrix and classification report together
    if save:
        save_inference_results(conf_matrix, class_report, inference_dir, class_names)
        # Save a log of the file used for inference
        log_file_path = os.path.join(inference_dir, 'inference_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Inference performed on file: {inference_file_path}\n")

    return conf_matrix, class_report


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

        # Save confusion matrix as an image using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Annotate each cell with the count value
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center")

    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    print(f"Confusion matrix saved at {confusion_matrix_path}")
    plt.close()

    # Parse and save classification report as CSV
    report_data = []
    lines = class_report.split('\n')
    
    for line in lines[2:-3]:
        row_data = line.split()
        
        # Ensure the line has enough elements (5 in this case)
        if len(row_data) == 5:
            report_data.append({
                'class': row_data[0],
                'precision': row_data[1],
                'recall': row_data[2],
                'f1-score': row_data[3],
                'support': row_data[4],
            })
        else:
            # print(f"Skipping line due to unexpected format: {line}")
            print('')
    
    df = pd.DataFrame.from_records(report_data)
    class_report_path = os.path.join(save_dir, 'classification_report.csv')
    df.to_csv(class_report_path, index=False)
    print(f"Classification report saved at {class_report_path}")


'''
def inference_without_ground_truth(model, dataloader, device, data_file, model_path, save_subfolder="predictions"):
    """
    Runs inference and writes predictions directly to a LAS file in a batch-wise manner.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the point cloud data for inference.
    - device (torch.device): Device to perform inference on (CPU or GPU).
    - data_file (str): Path to the input file (used for naming the output file).
    - model_path (str): Path to the model to be used for inference.
    - save_subfolder (str): Subdirectory of the model's folder where the LAS file with predictions will be saved.

    Returns:
    - las_file_path (str): File path to the saved LAS file. 
    """
    
    mp.set_sharing_strategy('file_system')  # trying to fix too many open files error

    model.eval()
    # Set up output path
    model_save_folder = os.path.dirname(model_path)  # Get the directory in which the model file is stored (the parent directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file_name = f"{os.path.splitext(os.path.basename(data_file))[0]}_CNN_{timestamp}.las"
    
    # Create timestamped folder inside the predictions subfolder
    save_dir = os.path.join(model_save_folder, save_subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    las_file_path = os.path.join(save_dir, pred_file_name)

    # Open the input LAS file to copy header information
    original_file = laspy.read(data_file)
    header = original_file.header
    
    # Check and add 'label' as needed
    if 'label' not in header.point_format.dimension_names:
        extra_dims = [laspy.ExtraBytesParams(name="label", type=np.int8)]
        header.add_extra_dims(extra_dims)

    # Initialize the label field with -1 values (-1 = not classified)
    total_points = len(original_file.x)
    label_array = np.full(total_points, -1, dtype=np.int8)
    
    all_predictions = []  # List to store predictions
    all_indices = []  # List to store indices

    # Perform inference 
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Performing inference", ascii=True, dynamic_ncols=True, file=sys.stdout):
            if batch is None:
                continue

            small_grids, medium_grids, large_grids, _, indices = batch
            small_grids, medium_grids, large_grids = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device)
            )

            # Run model inference
            outputs = model(small_grids, medium_grids, large_grids)
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and indices
            all_predictions.append(preds.cpu().numpy())
            all_indices.append(indices)
    
    # Concatenate all predictions and indices
    all_predictions = np.concatenate(all_predictions)
    all_indices = np.concatenate(all_indices) 
    
    # Directly assign predictions to the label array
    label_array[all_indices] = all_predictions
            
    # Create a new LasData object and assign fields for all point data and the label array
    new_las = laspy.LasData(header)
    new_las.points = original_file.points  # Copy original points
    new_las.label = label_array            # Assign the labels to the new dimension
    
    new_las.write(las_file_path)    # Write to the new file 
        
    return las_file_path'''