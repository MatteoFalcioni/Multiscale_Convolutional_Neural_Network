# (1) Look for all files that end with FGLn and FSLn in a folder
# then of course you will need to match them based on the number in the beginning

# (2) once you get all the files, simply use stitcher to stitch the off grounds (w/ 0 overlap)
# and (3) fuse them together with the ground (prob simple write to new file of all points, not overlapping)

# (4) class rebalancing and trai/eval split: you dont want cars to be too under-represented so you downsample 
# the most represented classes. Then you split. *All this is probably better to do on csv instead of las


import os
import laspy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.point_cloud_data_utils import las_to_csv, clean_and_combine_csv_files


def create_dataset(input_folders, fused_las_folder, max_points_per_class, output_dataset_folder=None, chosen_classes=[3,5,6,10,11,64], train_split=0.8):
    """
    Creates a dataset for training and evaluation from LAS files by processing ground and off-ground data.

    This function:
    - Finds and pairs ground and off-ground LAS files.
    - Stitches them into fused LAS files.
    - Converts the fused LAS files into CSVs.
    - Combines the CSV files into a single dataset.
    - Rebalances the dataset by downsampling overrepresented classes.
    - Splits the dataset into training and evaluation sets.

    Args:
    - input_folders (List): List of paths to the folders containing the input LAS files.
    - fused_las_folder (str): Folder to save fused LAS and intermediate CSV files.
    - max_points_per_class (int): Maximum number of points per class for rebalancing.
    - output_dataset_folder (str): Folder to save the final dataset.
    - chosen_classes (list, optional): List of class labels to include in the dataset.
    - train_split (float): Proportion of data allocated to training (default: 0.8).
    

    Returns:
    - None
    """
    # get ground + off ground las file pairs
    file_pairs = pair_ground_and_offgrounds(input_folders=input_folders)

    # stitch the pairs together and save them inside fused_las_folder
    fused_files = stitch_pairs(file_pairs=file_pairs, output_folder=fused_las_folder)

    # convert the las into csv files and save them in a fused_las_folder/csv/ subdirectory
    csv_subdir = f"{fused_las_folder}/csv" 
    os.makedirs(csv_subdir, exist_ok=True)
    csv_filepaths = []
    for las_filepath in fused_files:
        csv_path = las_to_csv(las_file=las_filepath, output_folder=csv_subdir)
        csv_filepaths.append(csv_path)

    # combine the csvs together to get one big file, and save it. Also cleans nan/inf values of combined file internally
    combined_csv = clean_and_combine_csv_files(csv_filepaths, output_csv=f"{output_dataset_folder}/full_dataset.csv")

    # finally rebalance the combined csv and create train/test datasets, saving them as csv inside output_dataset_folder
    train_df, eval_df = create_train_eval_datasets(csv_file=combined_csv,
                               max_points_per_class=max_points_per_class,
                               chosen_classes=chosen_classes,
                               train_split=train_split,
                               output_dataset_folder=output_dataset_folder)
    
    # Inspection: Print dataset summary
    print("\nDataset Summary:")
    print(f"\n\nTotal points in training set: {len(train_df)}")
    print("Class distribution in training set:")
    print(train_df['label'].value_counts().sort_index())
    print("\nColumns in training set:")
    print(train_df.columns.tolist())
    print('\n\n')

    print(f"Total points in evaluation set: {len(eval_df)}")
    print("\nClass distribution in evaluation set:")
    print(eval_df['label'].value_counts().sort_index())
    print("\nColumns in evaluation set:")
    print(eval_df.columns.tolist())

    return None


def pair_ground_and_offgrounds(input_folders):
    """
    Identifies and pairs FGL and FSL LAS files in the given folder based on their shared prefix (e.g., "32_681500").

    Args:
    - input_folders (list): List of path to the folders containing LAS files (each folder should have ground + off ground).

    Returns:
    - file_pairs (list): A list of tuples (ground_file, off_ground_files) where:
        - ground_file (str): Path to the ground LAS file (FGL).
        - off_ground_files (list): List of paths to the off-ground LAS files (FSL) sharing the same prefix.
    """
    file_pairs = []

    # Process each directory separately
    for input_folder in input_folders:
        # Find all ground and off-ground files based on suffix
        ground_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith("FGLn.las")]
        off_ground_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith("FSLn.las")]

        # Pair each ground file with all off-ground files in the same directory
        for ground_file in ground_files:
            if off_ground_files:
                file_pairs.append((ground_file, off_ground_files))
            else:
                print(f"Warning: No off-ground files found for ground file: {ground_file}")
    
    return file_pairs



def stitch_pairs(file_pairs, output_folder):
    """
    Stitches and fuses multiple pairs of ground and off-ground LAS files.

    Args:
    - file_pairs (list): List of tuples, where each tuple contains:
        - ground_file (str): Path to the ground LAS file.
        - off_ground_files (list): List of paths to the off-ground LAS files.
    - output_folder (str): Folder to save the fused LAS files.

    Returns:
    - fused_files (list): List of file paths to the fused LAS files.
    """
    fused_files = []

    print(f"Processing {len(file_pairs)} file pairs (ground + off grounds)")

    for ground_file, off_ground_files in file_pairs:
        # Load the ground LAS file
        if not ground_file:
            raise ValueError(f"Error: No ground file found.")
        ground_las = laspy.read(ground_file)

        # Check that at least one off-ground file exists
        if not off_ground_files:
            raise ValueError(f"Error: No off-ground files found for ground file {ground_file}")

        # Get the dimensions in the ground file
        ground_dimensions = set(ground_las.point_format.dimension_names)

        # Prepare lists to collect all points and attributes
        all_points = [np.vstack((ground_las.x, ground_las.y, ground_las.z)).T]
        all_labels = [ground_las.label]

        # Collect other attributes if they exist
        all_intensities = [ground_las.intensity] 
        all_red = [ground_las.red]
        all_green = [ground_las.green] 
        all_blue = [ground_las.blue] 
        all_nir = [ground_las.nir] 
        all_return_number = [ground_las.return_number] 
        all_number_of_returns = [ground_las.number_of_returns] 
        all_classification = [ground_las.classification] 

        # Loop through off-ground files and collect their data
        for file in off_ground_files:
            off_ground_las = laspy.read(file)

            # check that dimensions match between ground and off ground 
            off_ground_dimensions = set(off_ground_las.point_format.dimension_names)

            if ground_dimensions != off_ground_dimensions:
                raise ValueError(
                    f"Dimension mismatch between ground file and off-ground file:\n"
                    f"Ground dimensions: {ground_dimensions}\n"
                    f"Off-ground dimensions: {off_ground_dimensions}"
                )

            all_points.append(np.vstack((off_ground_las.x, off_ground_las.y, off_ground_las.z)).T)
            all_labels.append(off_ground_las.label)


            all_intensities.append(off_ground_las.intensity)

            all_red.append(off_ground_las.red)

            all_green.append(off_ground_las.green)

            all_blue.append(off_ground_las.blue)

            all_nir.append(off_ground_las.nir)

            all_return_number.append(off_ground_las.return_number)

            all_number_of_returns.append(off_ground_las.number_of_returns)

            all_classification.append(off_ground_las.classification)

        # Concatenate all points and attributes
        all_points = np.concatenate(all_points)
        all_labels = np.concatenate(all_labels)

        all_intensities = np.concatenate(all_intensities)

        all_red = np.concatenate(all_red)

        all_green = np.concatenate(all_green)

        all_blue = np.concatenate(all_blue)

        all_nir = np.concatenate(all_nir)

        all_return_number = np.concatenate(all_return_number)

        all_number_of_returns = np.concatenate(all_number_of_returns)

        all_classification = np.concatenate(all_classification)

        # Create a new LAS file for the fused data
        # Create a new header with the correct point format, scales, and offsets 
        fused_header = laspy.LasHeader(point_format=ground_las.header.point_format, version=ground_las.header.version)
        fused_header.offsets = ground_las.header.offsets
        fused_header.scales = ground_las.header.scales
        fused_las = laspy.LasData(fused_header)

        fused_las.x = all_points[:, 0]
        fused_las.y = all_points[:, 1]
        fused_las.z = all_points[:, 2]
        fused_las.label = all_labels

        fused_las.intensity = all_intensities

        fused_las.red = all_red

        fused_las.green = all_green

        fused_las.blue = all_blue

        fused_las.nir = all_nir

        fused_las.return_number = all_return_number

        fused_las.number_of_returns = all_number_of_returns

        fused_las.classification = all_classification

        # Update header
        fused_las.update_header()

        # Save the fused LAS file
        os.makedirs(output_folder, exist_ok=True)
        fused_filepath = os.path.join(output_folder, f"fused_{os.path.basename(ground_file)}")
        fused_las.write(fused_filepath)

        print(f"Fused LAS file saved at: {fused_filepath}")
        fused_files.append(fused_filepath)

    return fused_files


def create_train_eval_datasets(csv_file, max_points_per_class, chosen_classes=None, train_split=0.8, output_dataset_folder=None):
    """
    Analyzes the class distribution in a CSV file, filters chosen classes, rebalances by downsampling 
    overrepresented classes, splits the dataset into training and evaluation sets, and saves
    the splits to new CSV files inside the specified folder.

    Args:
    - csv_file (str): Path to the input CSV file.
    - max_points_per_class (int): Maximum number of points allowed per class.
    - chosen_classes (list, optional): List of class labels to extract and rebalance. If None, use all classes.
    - train_split (float): Proportion of data to allocate to the training set (default: 0.8).
    - output_dataset_folder (str): Folder where the train and eval csv will be saved.

    Returns:
    - train_df (pd.DataFrame): A Pandas DataFrame containing the training set.
    - eval_df (pd.DataFrame): A Pandas DataFrame containing the evaluation set.
    """
    # Check directory to save the datasets
    if output_dataset_folder is None:
        raise ValueError("Output dataset folder not specified.")
    os.makedirs(output_dataset_folder, exist_ok=True)
    
    # Initialize variables for processing the dataset in chunks
    chunksize = 10_000
    filtered_dfs = []  # For storing filtered chunks
    
    # Read and process the CSV in chunks
    print(f"Reading and processing the dataset in chunks...")
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        # Filter chosen classes
        if chosen_classes is not None:
            chunk = chunk[chunk['label'].isin(chosen_classes)]

        # Append filtered chunk to list
        filtered_dfs.append(chunk)

    # Combine all filtered chunks into a single DataFrame
    if filtered_dfs:
        df = pd.concat(filtered_dfs, ignore_index=True)
    else:
        raise ValueError("No data found after filtering for chosen classes.")

    # Print class distribution before rebalancing
    print("\nOriginal class distribution:")
    class_counts = df['label'].value_counts().sort_index()
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} points")

    # Rebalance the dataset by downsampling overrepresented classes
    rebalanced_dfs = []
    for class_label, count in class_counts.items():
        class_subset = df[df['label'] == class_label]
        if count > max_points_per_class:
            # Downsample to the specified max_points_per_class
            class_subset = class_subset.sample(max_points_per_class, random_state=42)
        rebalanced_dfs.append(class_subset)

    # Combine the rebalanced subsets
    rebalanced_df = pd.concat(rebalanced_dfs, ignore_index=True)

    # Print class distribution after rebalancing
    print("\nRebalanced class distribution:")
    rebalanced_class_counts = rebalanced_df['label'].value_counts().sort_index()
    for class_label, count in rebalanced_class_counts.items():
        print(f"Class {class_label}: {count} points")

    # Split the rebalanced dataset into training and evaluation sets
    train_df, eval_df = train_test_split(
        rebalanced_df, 
        test_size=(1 - train_split), 
        random_state=42, 
        stratify=rebalanced_df['label']
    )

    # Print dataset split summary
    print(f"\nDataset split into:")
    print(f"- Training set: {len(train_df)} points ({train_split * 100:.0f}%)")
    print(f"- Evaluation set: {len(eval_df)} points ({(1 - train_split) * 100:.0f}%)")
  
    train_csv = f"{output_dataset_folder}/train_dataset.csv"
    eval_csv = f"{output_dataset_folder}/eval_dataset.csv"
    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)

    print(f"\nTraining dataset saved to: {train_csv}")
    print(f"Evaluation dataset saved to: {eval_csv}")

    return train_df, eval_df




