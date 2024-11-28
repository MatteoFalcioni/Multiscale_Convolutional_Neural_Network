import laspy
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv
from datetime import datetime


def load_las_data(file_path):
    """
    Carica il file LAS e restituisce le coordinate XYZ.

    Args:
    - file_path (str): Percorso al file LAS.

    Returns:
    - np.ndarray: Coordinate XYZ dei punti.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points


def load_asc_data(file_path):
    """
    Carica il file ASC (DTM) e restituisce un array numpy.

    Args:
    - file_path (str): Percorso al file ASC.

    Returns:
    - np.ndarray: Dati del DTM.
    """
    dtm_data = np.loadtxt(file_path, skiprows=6)  # Skips metadata lines (i.e., skips header)
    return dtm_data


def read_las_file_to_numpy(file_path, features_to_extract=None):
    """
    Reads a LAS file, extracts coordinate data (x, y, z), specific features and labels,
    and returns them as a numpy array.

    Parameters:
    - file_path (str): The path to the LAS file.
    - features_to_extract (list): List of features to extract from the LAS file.
                                  If None, all available features except 'x', 'y', 'z' will be selected.
                                  Notice that 'segment_id', and 'label' are always included in the extracted features. 

    Returns:
    - np.ndarray: A numpy array containing the extracted data from the LAS file.
    - feature_names (list of str): List of feature names corresponding to the columns in the array.
    """
    # Read the LAS file
    # print(f"Processing {file_path}...")
    las_data = laspy.read(file_path)

    # Initialize a list to store the features and their names
    data = []
    feature_names = ['x', 'y', 'z']  # Always include coordinates

    # Check if x, y, z coordinates are present
    if hasattr(las_data, 'x') and hasattr(las_data, 'y') and hasattr(las_data, 'z'):
        if len(las_data.x) > 0 and len(las_data.y) > 0 and len(las_data.z) > 0:
            # Add x, y, z as the first columns
            data.append(las_data.x)
            data.append(las_data.y)
            data.append(las_data.z)
        else:
            print(f"Warning: One of the coordinate arrays (x, y, z) is empty in {file_path}.")
            return None
    else:
        print(f"Warning: LAS data in {file_path} does not have 'x', 'y', or 'z' attributes.")
        return None

    # If features_to_extract is None, select all available features except 'x', 'y', 'z', 'label', and 'segment_id'
    if features_to_extract is None:
        features_to_extract = [dim for dim in las_data.point_format.dimension_names if dim not in ['x', 'y', 'z', 'label', 'segment_id']]

    # Extract additional features
    available_features = []
    missing_features = []
    for feature in features_to_extract:
        if feature in ['x', 'y', 'z']:
            continue  # Skip if feature is x, y, or z since they are already added
        if feature in las_data.point_format.dimension_names:
            data.append(las_data[feature])
            available_features.append(feature)
        else:
            missing_features.append(feature)

    # Warn if any requested features are missing
    if missing_features:
        print(f"Warning: The following features were not found in the LAS file: {missing_features}")

    # Add selected features to feature_names
    feature_names += available_features
    
    # Check for segment_id 
    if 'segment_id' in las_data.point_format.dimension_names:
        data.append(las_data['segment_id'])
        feature_names.append('segment_id')

    # Check for label field
    if 'label' in las_data.point_format.dimension_names:
        data.append(las_data['label'])
        feature_names.append('label')
    else:
        print('***The LAS data does not contain a label column, which is needed for training. If you are training, choose a different file, with labels.***')

    # Convert the data list to a numpy array and transpose to match the expected shape (N, num_features)
    data_array = np.vstack(data).T
    # print(f"Loaded NumPy array with shape: {data_array.shape}")

    return data_array, feature_names


def read_csv_file_to_numpy(file_path, features_to_extract=None):
    """
    Reads a CSV file and extracts the specified features along with coordinates and labels.

    Args:
    - file_path (str): Path to the CSV file.
    - features_to_extract (list of str): List of feature names to extract. 
                                         If None, all columns will be selected.

    Returns:
    - np.ndarray: Numpy array containing the extracted features and coordinates.
    - feature_names (list of str): List of feature names corresponding to the columns in the array.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure 'x', 'y', 'z' coordinates are present
    if not all(coord in df.columns for coord in ['x', 'y', 'z']):
        raise ValueError(f"CSV file {file_path} is missing required coordinates ('x', 'y', 'z').")
    
    # Always include 'x', 'y', 'z' coordinates
    coords = df[['x', 'y', 'z']].values
    
    # Initialize feature_names with 'x', 'y', 'z'
    feature_names = ['x', 'y', 'z']
    
    # If features_to_extract is None, select all columns except 'x', 'y', 'z', 'segment_id' and 'label' (since these are always included)
    if features_to_extract is None:
        features_to_extract = [col for col in df.columns if col not in ['x', 'y', 'z', 'segment_id', 'label']]

    # Extract the features
    available_features = [f for f in features_to_extract if f in df.columns]
    
    # Check for features not present in the CSV and eventually print a warning
    missing_features = [f for f in features_to_extract if f not in df.columns]
    if missing_features:
        print(f"Warning: The following features were not found in the CSV: {missing_features}")
    
    feature_data = df[available_features].values
    
    # Add selected features to feature_names
    feature_names += available_features
    
    # Optionally handle segment_id if present
    if 'segment_id' in df.columns:
        segment_id = df[['segment_id']].values
        combined_data = np.hstack((coords, feature_data, segment_id))
        feature_names += ['segment_id']
    else:
        combined_data = np.hstack((coords, feature_data))
    
    # Check if the label column exists in the CSV
    if 'label' in df.columns:
        labels = df[['label']].values
        # Combine coordinates, features, segment_id and labels into a single array
        combined_data = np.hstack((combined_data, labels))
        # Add the label to the feature names
        feature_names += ['label']
    else: 
        raise ValueError('Labels are not present in the csv files. Process aborted.')

    return combined_data, feature_names


def numpy_to_dataframe(data_array, feature_names=None):
    """
    Converts a NumPy array to a pandas DataFrame.

    Args:
    - data_array (numpy.ndarray): The NumPy array to convert.
    - feature_names (list): List of column names for the DataFrame. If None, default names will be generated.

    Returns:
    - pandas.DataFrame: The resulting DataFrame.
    """
    # Check the number of columns in the data array
    num_columns = data_array.shape[1]

    # If feature_names is None, generate default names
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(num_columns)]
    elif len(feature_names) != num_columns:
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match the number of columns in data_array ({num_columns}).")

    # Convert the numpy array to a pandas DataFrame
    return pd.DataFrame(data_array, columns=feature_names)


def load_features_for_np(features_file_path):
    """
    Loads features from a CSV file. Needed when raw data is in .npy format to get the info for features used.

    Args:
    - features_file_path (str): Path to the CSV file containing the features.

    Returns:
    - list: The loaded features as a list.

    Raises:
    - FileNotFoundError: If the features file is not found.
    - ValueError: If the file is empty or not in the expected format.
    """
    
    # Check if the file exists
    if not os.path.exists(features_file_path):
        raise FileNotFoundError(f"Features file not found at {features_file_path}")

    # Load the features from the CSV file
    try:
        with open(features_file_path, 'r') as f:
            reader = csv.reader(f)
            features_list = next(reader)  # Assuming the first row contains the feature names

        if not features_list:
            raise ValueError(f"The features file at {features_file_path} is empty or invalid.")

        return features_list

    except Exception as e:
        raise ValueError(f"Error loading features from {features_file_path}: {e}")


def read_file_to_numpy(data_dir, features_to_use=None, features_file_path=None):
    """
    Loads the raw data from a .npy, .las, or .csv file and returns the data array along with the known features.

    Args:
    - data_dir (str): Path to the raw data file.
    - features_to_use (list): List of features to extract from the file.
    - features_file_path (str): Path to the features file (only used for .npy files).

    Returns:
    - data_array (np.ndarray): The raw data array (x, y, z, features).
    - known_features (list): List of feature names corresponding to the data array.
    """
    if data_dir.endswith('.npy'):  # Directly using a NumPy array as raw data
        print("Loading raw data from numpy file...")
        data_array = np.load(data_dir)
        try:
            known_features = load_features_for_np(features_file_path)
            print(f"Features loaded from {features_file_path}: {known_features}")
        except Exception as e:
            raise ValueError(f"Unable to load features from {features_file_path}: {e}")

    elif data_dir.endswith('.las'):  # LAS file
        # print("Loading raw data from LAS file...")
        data_array, known_features = read_las_file_to_numpy(data_dir, features_to_extract=features_to_use)

    elif data_dir.endswith('.csv'):  # CSV file
        # print("Loading raw data from CSV file...")
        data_array, known_features = read_csv_file_to_numpy(data_dir, features_to_extract=features_to_use)

    else:
        raise ValueError("Unsupported data format. Please provide a .npy, .las, or .csv file.")

    return data_array, known_features


def combine_csv_files(csv_files, output_csv):
    """
    Combines multiple CSV files into a single CSV file efficiently, processing them in chunks.

    Args:
    - csv_files (list of str): List of paths to CSV files to combine.
    - output_csv (str): Path to save the combined CSV file.

    Returns:
    - output_csv (str) : path to the saved combined CSV file
    """
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w') as outfile:
        # Initialize the progress bar
        with tqdm(total=len(csv_files), desc="Combining CSV files", unit="file") as pbar:
            for i, file in enumerate(csv_files):
                # Read the CSV file in chunks
                for chunk in pd.read_csv(file, chunksize=10_000):
                    # Write the header only for the first file
                    chunk.to_csv(outfile, index=False, header=(i == 0), mode='a')
                # Update the progress bar
                pbar.update(1)

    print(f"Combined CSV saved to {output_csv}")

    return output_csv


def sample_data(input_file, sample_size, save=False, save_dir='data/sampled_data', feature_to_use=None, features_file_path=None):
    """
    Samples a subset of the data from a CSV, NumPy, or LAS file. Optionally saves the sampled data as a CSV file.

    Args:
    - input_file (str): Path to the input file (either a CSV, NumPy file, or LAS file).
    - sample_size (int): The number of samples to extract.
    - save (bool): Whether to save the sampled data to a file. Default is False.
    - save_dir (str): Directory where the sampled data will be saved. Default is 'data/sampled_data'.
    - feature_to_use (list): List of feature names to select from the data.
    - features_file_path (str): File path to known features of the .npy data (required only if input is NumPy).

    Returns:
    - np.ndarray: The sampled data array.
    """
    # Load the data
    data_array, feature_names = read_file_to_numpy(data_dir=input_file, features_to_use=feature_to_use, features_file_path=features_file_path)

    # Ensure the label column is included in the features
    if 'label' not in feature_names:
        feature_names.append('label')

    # Check if sample_size is greater than the dataset size
    if sample_size > data_array.shape[0]:
        raise ValueError(f"Sample size {sample_size} is larger than the dataset size {data_array.shape[0]}.")

    # Sample the data
    print(f"Sampling {sample_size} rows from the dataset...")
    sampled_data = data_array[np.random.choice(data_array.shape[0], sample_size, replace=False)]

    # Optionally save the sampled data as a CSV file
    if save:
        os.makedirs(save_dir, exist_ok=True)
        sample_file_path = os.path.join(save_dir, f'sampled_data_{sample_size}.csv')
        df_sample = pd.DataFrame(sampled_data, columns=feature_names)
        df_sample.to_csv(sample_file_path, index=False)
        print(f"Sampled data saved to {sample_file_path}")

    return sampled_data


def remap_labels(data_array, label_column_index=-1):
    """
    Automatically remaps the labels in the given data array to a continuous range starting from 0. Needed for
    training purpose (in order to feed labels as targets to the loss).
    Stores the mapping for future reference.

    Args:
    - data_array (np.ndarray): The input data array where the last column (by default) contains the labels.
    - label_column_index (int): The index of the column containing the labels (default is the last column).

    Returns:
    - np.ndarray: The data array with the labels remapped.
    - dict: A dictionary that stores the original to new label mapping.
    """
    # Extract the label column
    labels = data_array[:, label_column_index]

    # Get the unique labels
    unique_labels = np.unique(labels)

    # Create a mapping from the unique labels to continuous integers
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Apply the mapping to the labels
    remapped_labels = np.array([label_mapping[label] for label in labels])

    # Replace the original labels in the data array with the remapped labels
    data_array[:, label_column_index] = remapped_labels

    # Return the remapped data array and the mapping dictionary
    return data_array, label_mapping


def extract_num_classes(raw_file_path=None):
    """
    Extracts the number of unique classes from raw data (LAS, CSV, or NPY).

    Args:
    - raw_file_path (str): Path to the input LAS, CSV, or NPY file.

    Returns:
    - int: The number of unique classes.
    """

    if raw_file_path is None: 
        raise ValueError('ERROR: File path to raw data must be provided to extract the number of classes.')

    # Load data from raw files
    data_array, _ = read_file_to_numpy(raw_file_path, features_to_use=None, features_file_path=None)
    
    # Extract class labels from the last column of the data array
    class_labels = data_array[:, -1]

    # Extract the unique number of classes
    num_classes = len(np.unique(class_labels))

    # print(f"Number of unique classes: {num_classes}")
    
    return num_classes


def subtiler(file_path, tile_size=50, overlap_size=10):
    """
    Subdivides a single LAS file into smaller tiles with overlaps and saves the subtiles in a new subdirectory.
    Ensures that no strip is left out, adjusting the dimensions for northernmost and rightmost subtiles if needed.

    Parameters:
    - file_path (str): Path to the LAS file to be subdivided.
    - tile_size (int): Size of each subtile in meters.
    - overlap_size (int): Size of the overlap between subtiles in meters.
    Returns:
    - output_dir (str): Path to the directory where subtiles were saved.
    """
    
    # Load the LiDAR file
    las_file = laspy.read(file_path)

    # Create subdirectory for the subtiles
    output_dir = f"{os.path.splitext(file_path)[0]}_{tile_size:03d}_subtiles"
    os.makedirs(output_dir, exist_ok=True)

    # Extract the lower left coordinates from the filename (assuming filename format includes coordinates)
    parts = os.path.basename(file_path).split('_')
    lower_left_x = int(parts[1])
    lower_left_y = int(parts[2])

    x_coords = las_file.x
    y_coords = las_file.y

    # Calculate the number of steps required, ensuring the last subtile adjusts dynamically
    total_size = 500  # Assuming the full tile is 500x500 meters
    steps_x = (total_size - overlap_size) // (tile_size - overlap_size)
    steps_y = (total_size - overlap_size) // (tile_size - overlap_size)

    progress_bar = tqdm(total=(steps_x * steps_y), desc="Processing subtiles")

    # Loop to create subtiles
    for i in range(steps_x + 1):  # Include an extra step for the last strip
        for j in range(steps_y + 1):  # Include an extra step for the last strip
            # Calculate the starting position for the subtile
            subtile_lower_left_x = lower_left_x + i * (tile_size - overlap_size)
            subtile_lower_left_y = lower_left_y + j * (tile_size - overlap_size)

            # Determine subtile dimensions
            subtile_upper_right_x = subtile_lower_left_x + tile_size
            subtile_upper_right_y = subtile_lower_left_y + tile_size

            # Adjust for the last subtile to ensure no strip is left out
            if i == steps_x:
                subtile_upper_right_x = lower_left_x + total_size
            if j == steps_y:
                subtile_upper_right_y = lower_left_y + total_size

            # Determine points within the subtile
            mask = (
                (x_coords >= subtile_lower_left_x) & 
                (x_coords < subtile_upper_right_x) & 
                (y_coords >= subtile_lower_left_y) & 
                (y_coords < subtile_upper_right_y)
            )

            # Create a new LAS file header and set properties
            new_header = laspy.LasHeader(point_format=las_file.header.point_format,
                                         version=las_file.header.version)
            new_header.offsets = las_file.header.offsets
            new_header.scales = las_file.header.scales
            new_las = laspy.LasData(new_header)
            new_las.points = las_file.points[mask]

            # Only proceed if the subtile has points
            if len(new_las.x) > 0:
                new_las.update_header()

                # Generate the filename with lower-left coordinates
                subtile_file_name = f"{output_dir}/subtile_{subtile_lower_left_x}_{subtile_lower_left_y}"

                # Save the new LAS file
                new_las.write(f"{subtile_file_name}.las")

            # Update progress bar
            progress_bar.update(1)

    print('\nSubtiles created successfully.\n')
    return output_dir


def stitch_subtiles(subtile_folder, original_las, original_filename, model_directory, overlap_size=30):
    """
    Stitches subtiles back together into the original LAS file.
    
    Args:
    - subtile_folder (str): Folder containing the sub-tile files to be stitched.
    - original_las (laspy.LasData): The loaded LASData object of the original LAS file.
    - original_filename (str) : File name of the original LAS file.
    - model_directory (str): Directory where the trained PyTorch model is stored.
    - overlap_size (int): Size of the overlap between subtiles in meters.

    Return:
    - output_filepath (str): File path to the output stitched file.
    """

    # Create a new header with the correct point format, scales, and offsets 
    new_header = laspy.LasHeader(point_format=original_las.header.point_format, version=original_las.header.version)
    new_header.offsets = original_las.header.offsets
    new_header.scales = original_las.header.scales

    # Create a new LasData object with the new header
    stitched_las = laspy.LasData(new_header)
    
    if 'label' not in stitched_las.point_format.dimension_names:
        # Add labels as a dimension to the stitched file if it does not exist
        stitched_las.add_dimension('label')
        
    # Initialize lists to store points, labels, and other features
    all_points = []
    all_labels = []
    all_intensitites = []
    all_red = []
    all_green = []
    all_blue = []
    all_nir = []
    all_return_number = []
    all_number_of_returns = []
    all_classification = []

    # Get the lower-left coordinates of all subtiles from their filenames
    lower_left_coords = []

    # Get all subtile files from the subtile folder
    subtile_files = [os.path.join(subtile_folder, f) for f in os.listdir(subtile_folder) if f.endswith('_pred.las')]

    for subtile_file in subtile_files:
        # Extract coordinates from filename
        filename = os.path.basename(subtile_file)

        # Split filename to get coordinates (x, y) - expect format : "subtile_x_y.las"
        parts = filename.split('_')
        lower_left_x = int(parts[1])
        lower_left_y = int(parts[2].split('.')[0])  # Extract y (before the file extension)
        lower_left_coords.append((lower_left_x, lower_left_y))
    
    # define size of strip to cut off
    cut_off = overlap_size/2
    
    # Iterate over each subtile
    for subtile_file in subtile_files:
        # Read the subtile
        subtile_las = laspy.read(subtile_file)
        
        if subtile_las.point_format != original_las.point_format:
            print(f"Warning: Point format mismatch in {subtile_file}, converting to original format.")
            subtile_las.points = subtile_las.points.convert_to(original_las.point_format)
        
        filename = os.path.basename(subtile_file)
        # Split filename to get coordinates (x, y)
        parts = filename.split('_')
        lower_left_x = int(parts[1])
        lower_left_y = int(parts[2].split('.')[0])  

        # Define upper bounds based on tile size (to cut off strips of unlabeled points)
        min_x = subtile_las.x.min()
        max_x = subtile_las.x.max()
        min_y = subtile_las.y.min()
        max_y = subtile_las.y.max()
        
        upper_bound_x = max_x - cut_off    
        upper_bound_y = max_y - cut_off
        lower_bound_x = min_x + cut_off
        lower_bound_y = min_y + cut_off

        mask = (
            (subtile_las.x < upper_bound_x) &  
            (subtile_las.y < upper_bound_y ) & 
            (subtile_las.x > lower_bound_x) &
            (subtile_las.y > lower_bound_y)
        )

        subtile_masked = subtile_las.points[mask]

        # Extract only x, y, z coordinates and labels 
        masked_points = np.vstack((subtile_masked.x, subtile_masked.y, subtile_masked.z)).T
        masked_labels = subtile_masked.label  
        masked_intensitites = subtile_masked.intensity
        masked_red = subtile_masked.red
        masked_green = subtile_masked.green
        masked_blue = subtile_masked.blue
        masked_nir = subtile_masked.nir
        masked_return_number = subtile_masked.return_number
        masked_number_of_returns = subtile_masked.number_of_returns
        masked_classification = subtile_masked.classification
            
        # Append the masked points and labels to the final lists
        all_points.append(masked_points)
        all_labels.append(masked_labels)
        all_intensitites.append(masked_intensitites)
        all_red.append(masked_red)
        all_green.append(masked_green)
        all_blue.append(masked_blue)
        all_nir.append(masked_nir)
        all_return_number.append(masked_return_number)
        all_number_of_returns.append(masked_number_of_returns)
        all_classification.append(masked_classification)
        
    # Concatenate all points and labels from all sub-tiles
    all_points = np.concatenate(all_points)
    all_labels = np.concatenate(all_labels)
    all_intensitites = np.concatenate(all_intensitites)
    all_red = np.concatenate(all_red)
    all_green = np.concatenate(all_green)
    all_blue = np.concatenate(all_blue)
    all_nir = np.concatenate(all_nir)
    all_return_number = np.concatenate(all_return_number)
    all_number_of_returns = np.concatenate(all_number_of_returns)
    all_classification = np.concatenate(all_classification)

    # Verify lengths
    print(f"Total points: {len(all_points)}, Total labels: {len(all_labels)}")
    assert len(all_points) == len(all_labels), "Mismatch between points and labels!"
    
    stitched_las.x = all_points[:, 0]
    stitched_las.y = all_points[:, 1]
    stitched_las.z = all_points[:, 2]
    stitched_las.label = all_labels
    stitched_las.intensity = all_intensitites
    stitched_las.red = all_red
    stitched_las.green = all_green
    stitched_las.blue = all_blue
    stitched_las.nir = all_nir
    stitched_las.return_number = all_return_number 
    stitched_las.number_of_returns = all_number_of_returns
    stitched_las.classification = all_classification

    # Update header
    stitched_las.update_header()

    # Construct the path for saving the final stitched file inside the model's directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(model_directory, 'predictions')
    os.makedirs(model_save_dir, exist_ok=True)
    base_filename = os.path.basename(original_filename)     # Get the base filename without extension
    base_filename_without_ext = os.path.splitext(base_filename)[0]

    # Construct the final output file path
    output_filepath = os.path.join(model_save_dir, f"{base_filename_without_ext}_pred_{timestamp}.las")

    # Save the stitched file
    stitched_las.write(output_filepath)

    # print(f"Stitching completed. Stitched file saved at: {output_filepath}")

    return output_filepath
    

def clean_nan_values(data_array, default_value=0.0):
    """
    Cleans NaN and Inf values in a NumPy array by replacing them with a specified default value.

    Args:
    - data_array (numpy.ndarray): The NumPy array to clean.
    - default_value (numeric): The value to replace NaN and Inf values with. Default is 0.

    Returns:
    - numpy.ndarray: The cleaned NumPy array.
    """
    cleaned_array = np.nan_to_num(data_array, nan=default_value, posinf=default_value, neginf=default_value)
    total_nans = np.isnan(data_array).sum()
    total_infs = np.isinf(data_array).sum()
    print(f"Cleaning data array: Replaced {total_nans} NaN values and {total_infs} Inf values with {default_value}.")
    return cleaned_array


def mask_out_of_bounds_points(data_array, window_sizes, bounds):
    """
    Masks points that are too close to the boundaries of the dataset to generate grids.

    Args:
    - data_array (numpy.ndarray): Full point cloud data.
    - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 1.0), ...]).
    - bounds (dict): Precomputed point cloud bounds.

    Returns:
    - valid_points (numpy.ndarray): Points that are not out of bounds.
    - mask (numpy.ndarray): Boolean array indicating valid points.
    """
    # Determine the maximum half window size from all scales
    max_half_window = max(window_size / 2 for _, window_size in window_sizes)

    # Compute dataset boundaries
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']

    # Apply the mask to filter out out-of-bound points
    mask = (
        (data_array[:, 0] - max_half_window >= x_min) & (data_array[:, 0] + max_half_window <= x_max) &
        (data_array[:, 1] - max_half_window >= y_min) & (data_array[:, 1] + max_half_window <= y_max)
    )

    # Return valid points and the mask
    return data_array[mask], mask



def compute_point_cloud_bounds(data_array, padding=0.0):
    """
    Computes the spatial boundaries (min and max) of the point cloud data.
    
    Args:
    - data_array (numpy.ndarray): Array containing point cloud data where the first three columns are (x, y, z) coordinates.
    - padding (float): Optional padding to extend the boundaries by a fixed amount in all directions.
    
    Returns:
    - bounds_dict (dict): Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' defining the spatial limits of the point cloud.
    """
    # Calculate the min and max values for x and y coordinates
    x_min = data_array[:, 0].min() - padding
    x_max = data_array[:, 0].max() + padding
    y_min = data_array[:, 1].min() - padding
    y_max = data_array[:, 1].max() + padding

    # Construct the boundaries dictionary
    bounds_dict = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }

    return bounds_dict


def las_to_csv(las_file, output_folder):
    """
    Converts a LAS file to a CSV file by extracting its data and features.

    This function reads a LAS file, converts its contents into a NumPy array, 
    and transforms the array into a Pandas DataFrame. The DataFrame is then saved 
    as a CSV file at the specified output location.

    Args:
    - las_file (str): Path to the input LAS file.
    - output_folder (str): Folder to save the output CSV file. The file name is derived
                           from the LAS file name by replacing .las with .csv.

    Returns:
    - output_csv_filepath (str): Path to the saved CSV file.
    """

    # Derive the output CSV file path
    las_filename = os.path.basename(las_file)  # Extract file name
    csv_filename = os.path.splitext(las_filename)[0] + ".csv"  # Replace .las with .csv
    output_csv_filepath = os.path.join(output_folder, csv_filename)

    # Read the LAS file to numpy and get the features
    data_array, known_features = read_file_to_numpy(las_file)

    # Convert the array to a DataFrame
    df = numpy_to_dataframe(data_array=data_array, feature_names=known_features)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_filepath, index=False)

    print(f"Converted {las_file} to {output_csv_filepath}")
    return output_csv_filepath




''' THIS WAS INSIDE STITCH
# Find the maximum x and y values (rightmost and northernmost tiles) this was outside for loop
    # up_y = max(lower_left_coords, key=lambda x: x[1])[1]  # Northernmost tiles (largest y)
    # right_x = max(lower_left_coords, key=lambda x: x[0])[0]  # Rightmost tiles (largest x)
is_northernmost = False
        is_rightmost = False

        if lower_left_y == up_y:
            is_northernmost = True
        if lower_left_x == right_x:
            is_rightmost = True

        if is_northernmost and is_rightmost:
            # top right corner subtile: exclude bottom strip and left strip 
            mask = (
                (subtile_las.x >= lower_bound_x) & 
                (subtile_las.y >= lower_bound_y)  
            )

        elif is_northernmost:
            # northermost subtiles: exclude bottom strip and right strip of unclassified points
            mask = (
                (subtile_las.x < upper_bound_x) &  
                (subtile_las.y >= lower_bound_y) 
            )
        
        elif is_rightmost:
            # rightmost subtiles: exclude left strip and top strip
            mask = (
                (subtile_las.x >= lower_bound_x) & 
                (subtile_las.y < upper_bound_y )  
            )
            
        else:
            # general subtile: exclude right strip and top strip
            print(f"min x: {min_x}\n")
            print(f"min y: {min_y}\n")
            print(f"max x: {max_x}\n")
            print(f"max y: {max_y}\n")
            
            print("this is a general subtile, so we have to cut off :\n")
            print(f"x until the upper bound: {upper_bound_x}\n")
            print(f"y until the upper bound: {upper_bound_y}\n")
            # eventually implement an is_leftmost and is_lowermost if you want to keep the lower and left -1 pts'''



'''def old_subtiler(file_path, tile_size=50, overlap_size=10):
    """
    Subdivides a single LAS file into smaller tiles with overlaps and saves the subtiles in a new subdirectory.
    Only processes the file if it contains more than the specified minimum number of points.

    Parameters:
    - file_path (str): Path to the LAS file to be subdivided.
    - tile_size (int): Size of each subtile in meters.
    - overlap_size (int): Size of the overlap between subtiles in meters.
    Returns:
    - output_dir (str): Path to the direcotry where subtiles were saved
    """
    
    # Load the LiDAR file
    las_file = laspy.read(file_path)
    # num_points = len(las_file.x)

    # Create subdirectory for the subtiles
    output_dir = f"{os.path.splitext(file_path)[0]}_{tile_size:03d}_subtiles"
    os.makedirs(output_dir, exist_ok=True)

    # Extract the lower left coordinates from the filename (assuming filename format includes coordinates)
    parts = os.path.basename(file_path).split('_')
    lower_left_x = int(parts[1])
    lower_left_y = int(parts[2])

    x_coords = las_file.x
    y_coords = las_file.y

    total_points = 0
    steps = 500 // tile_size  # Increase steps to account for overlap

    progress_bar = tqdm(total=(steps * steps), desc="Processing subtiles")

    # Loop to create subtiles
    for i in range(steps):
        for j in range(steps):
            # Calculate the starting position for the subtile
            subtile_lower_left_x = lower_left_x + i * (tile_size - overlap_size)
            subtile_lower_left_y = lower_left_y + j * (tile_size - overlap_size)

            # Determine points within the subtile
            mask = (
                (x_coords >= subtile_lower_left_x) &
                (x_coords <= subtile_lower_left_x + tile_size) &
                (y_coords >= subtile_lower_left_y) &
                (y_coords <= subtile_lower_left_y + tile_size)
            )

            # Create a new LAS file header and set properties
            new_header = laspy.LasHeader(point_format=las_file.header.point_format,
                                         version=las_file.header.version)
            new_header.offsets = las_file.header.offsets
            new_header.scales = las_file.header.scales
            new_las = laspy.LasData(new_header)
            new_las.points = las_file.points[mask]
            
            # Only proceed if the subtile has points
            if len(new_las.x) > 0:
                new_las.update_header()
                # total_points += len(new_las.x)

                # Generate the filename with lower-left and lower-right suffixes
                subtile_file_name = f"{output_dir}/subtile_{subtile_lower_left_x}_{subtile_lower_left_y}"

                # Save the new LAS file
                new_las.write(f"{subtile_file_name}.las")
                # print(f"Saved subtile: {subtile_file_name}, with {len(new_las.x)} points")

            # Update progress bar
            progress_bar.update(1)
    
    # print(f"Total points in generated subtiles: {total_points}")
    # print(f"Original number of points: {num_points}")
    print('\nSubtiles created succesfully.\n')

    return output_dir'''


'''
def reservoir_sample_data(input_file, sample_size, save=False, save_dir='data/sampled_data', feature_to_use=None, chunk_size=100000):
    """
    Samples a random subset of the data from a large CSV file using reservoir sampling.

    Args:
    - input_file (str): Path to the input CSV file.
    - sample_size (int): The number of samples to extract.
    - save (bool): Whether to save the sampled data to a file. Default is False.
    - save_dir (str): Directory where the sampled data will be saved. Default is 'data/sampled_data'.
    - feature_to_use (list): List of feature names to select from the data.
    - chunk_size (int): Number of rows to process per chunk. Default is 100000.

    Returns:
    - pd.DataFrame: The sampled data DataFrame.
    """
    reservoir = []  # List to store the sampled rows
    total_rows = 0  # Total number of rows processed

    # Iterate over chunks of the data and add tqdm for the progress bar
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size, usecols=feature_to_use), desc="Processing chunks"):
        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):
            if len(reservoir) < sample_size:
                # If the reservoir is not full, add the row
                reservoir.append(row)
            else:
                # Randomly decide whether to replace an existing element in the reservoir
                replace_idx = random.randint(0, total_rows - 1)
                if replace_idx < sample_size:
                    reservoir[replace_idx] = row

    # Convert the reservoir to a DataFrame
    sampled_data = pd.DataFrame(reservoir, columns=chunk.columns)

    # Optionally save the sampled data
    if save:
        os.makedirs(save_dir, exist_ok=True)
        sample_file_path = os.path.join(save_dir, f'sampled_data_{sample_size}.csv')
        sampled_data.to_csv(sample_file_path, index=False)
        print(f"Sampled data saved to {sample_file_path}")

    return sampled_data
'''

'''
def get_feature_indices(features_to_use, known_features):
    """
    Given a list of chosen features and the known features in the data array, this function
    returns the indices of the chosen features.

    Args:
    - features_to_use (list of str): List of feature names to use (e.g., ['intensity', 'red', 'green', 'blue']).
    - known_features (list of str): List of all known features in the point cloud data (e.g., ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue']).

    Returns:
    - list of int: The indices of the chosen features in the known features array.
    """
    try:
        feature_indices = [known_features.index(feature) for feature in features_to_use]
    except ValueError as e:
        raise ValueError(f"Feature {str(e).split()[0]} not found in known features: {known_features}")
    
    return feature_indices
'''

'''def extract_num_channels(preprocessed_data_dir):
    """
    Extracts the number of channels from the preprocessed grid files.

    Args:
    - preprocessed_data_dir (str): Path to the preprocessed data directory.

    Returns:
    - int: The number of channels in the preprocessed grids.
    """
    # Assuming grids are stored in the 'small' directory (or any scale's directory)
    grid_files = os.listdir(os.path.join(preprocessed_data_dir, 'small'))
    
    if not grid_files:
        raise FileNotFoundError(f"No grid files found in the directory: {os.path.join(preprocessed_data_dir, 'small')}")

    # Load one grid file to check its shape
    sample_grid = np.load(os.path.join(preprocessed_data_dir, 'small', grid_files[0]))
    
    # The number of channels is the first dimension in the shape (assuming format: [channels, height, width])
    num_channels = sample_grid.shape[0]
    
    return num_channels'''