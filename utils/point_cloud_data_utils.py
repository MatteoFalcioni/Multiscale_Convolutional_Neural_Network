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
                                  If None, all available features except 'x', 'y', 'z', 'segment_id', and 'label' will be selected.

    Returns:
    - np.ndarray: A numpy array containing the extracted data from the LAS file.
    - feature_names (list of str): List of feature names corresponding to the columns in the array.
    """
    # Read the LAS file
    print(f"Processing {file_path}...")
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
        raise ValueError('The LAS data does not contain a label column, which is needed for training. Process stopped.')

    # Convert the data list to a numpy array and transpose to match the expected shape (N, num_features)
    data_array = np.vstack(data).T
    print(f"Loaded NumPy array with shape: {data_array.shape}")

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



def combine_and_save_csv_files(csv_files, save=False, save_dir='data/combined_data'):
    """
    Combines multiple CSV files into a single NumPy array and optionally saves the combined data to a file.

    Args:
    - csv_files (list of str): List of paths to CSV files.
    - save (bool): Whether to save the combined data to a file. Default is False.
    - save_dir (str): Directory where the combined NumPy array will be saved. Default is 'combined_data'.

    Returns:
    - np.ndarray: Combined data from all CSV files as a NumPy array.
    """
    combined_data = []

    # Loop through each CSV file and read its contents
    print("Reading CSV files:")
    for file in tqdm(csv_files, desc="Reading", unit="file"):
        # Read the CSV file into a NumPy array
        data = pd.read_csv(file).values
        combined_data.append(data)

    # Combine all data into a single NumPy array
    combined_array = np.vstack(combined_data)

    # Optionally save the combined data
    if save:
        os.makedirs(save_dir, exist_ok=True)
        output_file_path = os.path.join(save_dir, 'combined_data.npy')
        np.save(output_file_path, combined_array)
        print(f"Combined data saved to {output_file_path}")

    return combined_array


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

    print(f"Number of unique classes: {num_classes}")
    
    return num_classes


def subtiler(file_path, tile_size=50, overlap_size=10):
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
    print(f'----------------------------- Subtiling file: {file_path} -----------------------------')
    
    # Load the LiDAR file
    las_file = laspy.read(file_path)
    num_points = len(las_file.x)

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
    steps = (500 // tile_size) + 1  # Increase steps to account for overlap

    progress_bar = tqdm(total=(steps * steps), desc="Processing subtiles", ascii=True, dynamic_ncols=True)

    # Loop to create subtiles
    for i in range(steps):
        for j in range(steps):
            # Calculate the starting position for the subtile
            subtile_lower_left_x = lower_left_x + i * (tile_size - overlap_size)
            subtile_lower_left_y = lower_left_y + j * (tile_size - overlap_size)

            # Determine points within the subtile
            mask = (
                (x_coords >= subtile_lower_left_x) &
                (x_coords < subtile_lower_left_x + tile_size) &
                (y_coords >= subtile_lower_left_y) &
                (y_coords < subtile_lower_left_y + tile_size)
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
                total_points += len(new_las.x)

                # Generate the filename with lower-left and lower-right suffixes
                subtile_file_name = f"{output_dir}/subtile_{subtile_lower_left_x}_{subtile_lower_left_y}"

                # Save the new LAS file
                new_las.write(f"{subtile_file_name}.las")
                print(f"Saved subtile: {subtile_file_name}, with {len(new_las.x)} points")

            # Update progress bar
            progress_bar.update(1)
    
    print(f"Total points in generated subtiles: {total_points}")
    print(f"Original number of points: {num_points}")

    return output_dir


def stitch_subtiles(subtile_files, original_file, model_directory, tile_size=50, overlap_size=30):
    """
    Stitches subtiles back together into the original LAS file.
    
    Args:
    - subtile_files (list): List of paths to the sub-tile files to be stitched.
    - original_file (str): Path to the original LAS file, used for copying the header.
    - model_directory (str): Directory where the trained PyTorch model is stored.
    - tile_size (int): Size of each subtile in meters.
    - overlap_size (int): Size of the overlap between subtiles in meters.
    """
    # Open the original file to get header info
    original_las = laspy.read(original_file)
    og_header = laspy.LasHeader(point_format=original_las.header.point_format,
                                         version=original_las.header.version)
    og_header.offsets = original_las.header.offsets
    og_header.scales = original_las.header.scales
    
    # Create the stitched LAS file and copy the header
    stitched_las = laspy.LasData(og_header) 
    
    if 'label' not in stitched_las.point_format.dimension_names:
        # Add labels as a dimension to the stitched file if it does not exist
        stitched_las.add_dimension('label')
        
    # Initialize lists to store points, labels, and other features
    all_points = []
    all_labels = []
    
    # Iterate over each subtile
    for subtile_file in subtile_files:
        # Read the subtile
        subtile_las = laspy.read(subtile_file)
        
        if subtile_las.point_format != original_las.point_format:
            print(f"Warning: Point format mismatch in {subtile_file}, converting to original format.")
            subtile_las.points = subtile_las.points.convert_to(original_las.point_format)
        
        # Calculate the overlap boundaries to exclude overlap
        x_min, y_min = subtile_las.x.min(), subtile_las.y.min()
        x_max, y_max = subtile_las.x.max(), subtile_las.y.max()
        
        # Apply the mask to exclude points within the overlap region
        mask = (
            (subtile_las.x > (x_min + overlap_size)) &
            (subtile_las.x < (x_max - overlap_size)) &
            (subtile_las.y > (y_min + overlap_size)) &
            (subtile_las.y < (y_max - overlap_size))
        )
        
        masked_points = subtile_las.points[mask]
        masked_labels = subtile_las.label[mask]
        
        # Append the masked points and labels to the final lists
        all_points.append(masked_points)
        all_labels.append(masked_labels)
        
    # Concatenate all points and labels from all sub-tiles
    all_points = np.concatenate(all_points)
    all_labels = np.concatenate(all_labels)
    
    stitched_las.points = all_points
    stitched_las.labels = all_labels

    # Construct the path for saving the final stitched file inside the model's directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(model_directory, 'inference', 'predictions')
    os.makedirs(model_save_dir, exist_ok=True)
    base_filename = os.path.basename(original_file)     # Get the base filename without extension
    base_filename_without_ext = os.path.splitext(base_filename)[0]

    # Construct the final output file path
    output_file = os.path.join(model_save_dir, f"{base_filename_without_ext}_pred_{timestamp}.las")

    # Save the stitched file
    stitched_las.write(output_file)

    print(f"Stitching completed. File saved at: {output_file}")
    

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
    
    return feature_indices'''


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