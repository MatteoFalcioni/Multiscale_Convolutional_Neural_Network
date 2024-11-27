# (1) Look for all files that end with FGLn and FSLn in a folder
# then of course you will need to match them based on the number in the beginning

# (2) once you get all the files, simply use stitcher to stitch the off grounds (w/ 0 overlap)
# and (3) fuse them together with the ground (prob simple write to new file of all points, not overlapping)

# (4) class rebalancing and trai/eval split: you dont want cars to be too under-represented so you downsample 
# the most represented classes. Then you split. *All this is probably better to do on csv instead of las

# (5) eventually you can choose to reate a 'big' dataset where you downsample less. Then if gpu implementation works
# we might be able to use that one. But maybe it's not even necessary.

from collections import defaultdict
import os
import laspy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from point_cloud_data_utils import las_to_csv, combine_csv_files


def create_dataset(input_folder, fused_las_folder, max_points_per_class, output_dataset_folder=None, chosen_classes=None, train_split=0.8):
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
    - input_folder (str): Path to the folder containing the input LAS files.
    - fused_las_folder (str): Folder to save fused LAS and intermediate CSV files.
    - max_points_per_class (int): Maximum number of points per class for rebalancing.
    - output_dataset_folder (str): Folder to save the final dataset.
    - chosen_classes (list, optional): List of class labels to include in the dataset.
    - train_split (float): Proportion of data allocated to training (default: 0.8).
    

    Returns:
    - None
    """
    # get ground + off ground las file pairs
    file_pairs = pair_ground_and_offgrounds(input_folder=input_folder)

    # stitch the pairs together and save them inside fused_las_folder
    fused_files = stitch_pairs(file_pairs=file_pairs, output_folder=fused_las_folder)

    # convert the las into csv files and save them in a fused_las_folder/csv/ subdirectory
    csv_subdir = f"{fused_las_folder}/csv" 
    os.makedirs(csv_subdir, exist_ok=True)
    csv_filepaths = []
    for las_filepath in fused_files:
        csv_path = las_to_csv(las_file=las_filepath, output_folder=csv_subdir)
        csv_filepaths.append(csv_path)

    # combine the csvs together to get one big file, and save it
    combined_csv = combine_csv_files(csv_filepaths, output_csv=f"{output_dataset_folder}/full_dataset.csv")

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






def pair_ground_and_offgrounds(input_folder):
    """
    Identifies and pairs FGL and FSL LAS files in the given folder based on their shared prefix (e.g., "32_681500").

    Args:
    - input_folder (str): Path to the folder containing LAS files.

    Returns:
    - file_pairs (list): A list of tuples (ground_file, off_ground_files) where:
        - ground_file (str): Path to the ground LAS file (FGL).
        - off_ground_files (list): List of paths to the off-ground LAS files (FSL) sharing the same prefix.
    """
    # Initialize a dictionary to group ground and off-ground files by their prefix
    file_groups = defaultdict(lambda: {"FGL": None, "FSL": []})
    
    # Iterate through all files in the folder
    for file in os.listdir(input_folder):
        if file.endswith(".las"):
            # Split filename into parts to extract prefix and suffix
            parts = file.split("_")
            if len(parts) < 4:
                continue  # Skip files that don't match the expected format
            
            # Extract the first two parts of the prefix (e.g., "32_681500")
            major_prefix = "_".join(parts[:2])
            suffix = parts[-1].split(".")[0]  # Extract the suffix (e.g., "FGL", "FSL")

            # Categorize files based on their suffix
            if suffix == "FGL":
                file_groups[major_prefix]["FGL"] = os.path.join(input_folder, file)
            elif suffix == "FSL":
                file_groups[major_prefix]["FSL"].append(os.path.join(input_folder, file))
    
    # Create the list of paired files
    file_pairs = []
    for prefix, files in file_groups.items():
        ground_file = files["FGL"]
        off_ground_files = files["FSL"]
        
        # Only include pairs where both ground and off-ground files exist
        if ground_file and off_ground_files:
            file_pairs.append((ground_file, off_ground_files))
    
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
        all_intensities = [ground_las.intensity] if "intensity" in ground_dimensions else None
        all_red = [ground_las.red] if "red" in ground_dimensions else None
        all_green = [ground_las.green] if "green" in ground_dimensions else None
        all_blue = [ground_las.blue] if "blue" in ground_dimensions else None
        all_nir = [ground_las.nir] if "nir" in ground_dimensions else None
        all_return_number = [ground_las.return_number] if "return_number" in ground_dimensions else None
        all_number_of_returns = [ground_las.number_of_returns] if "number_of_returns" in ground_dimensions else None
        all_classification = [ground_las.classification] if "classification" in ground_dimensions else None

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

            if all_intensities is not None:
                all_intensities.append(off_ground_las.intensity)
            if all_red is not None:
                all_red.append(off_ground_las.red)
            if all_green is not None:
                all_green.append(off_ground_las.green)
            if all_blue is not None:
                all_blue.append(off_ground_las.blue)
            if all_nir is not None:
                all_nir.append(off_ground_las.nir)
            if all_return_number is not None:
                all_return_number.append(off_ground_las.return_number)
            if all_number_of_returns is not None:
                all_number_of_returns.append(off_ground_las.number_of_returns)
            if all_classification is not None:
                all_classification.append(off_ground_las.classification)

        # Concatenate all points and attributes
        all_points = np.concatenate(all_points)
        all_labels = np.concatenate(all_labels)

        if all_intensities is not None:
            all_intensities = np.concatenate(all_intensities)
        if all_red is not None:
            all_red = np.concatenate(all_red)
        if all_green is not None:
            all_green = np.concatenate(all_green)
        if all_blue is not None:
            all_blue = np.concatenate(all_blue)
        if all_nir is not None:
            all_nir = np.concatenate(all_nir)
        if all_return_number is not None:
            all_return_number = np.concatenate(all_return_number)
        if all_number_of_returns is not None:
            all_number_of_returns = np.concatenate(all_number_of_returns)
        if all_classification is not None:
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

        if all_intensities is not None:
            fused_las.intensity = all_intensities
        if all_red is not None:
            fused_las.red = all_red
        if all_green is not None:
            fused_las.green = all_green
        if all_blue is not None:
            fused_las.blue = all_blue
        if all_nir is not None:
            fused_las.nir = all_nir
        if all_return_number is not None:
            fused_las.return_number = all_return_number
        if all_number_of_returns is not None:
            fused_las.number_of_returns = all_number_of_returns
        if all_classification is not None:
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
     # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        raise ValueError("The CSV file does not contain a 'label' column.")

    # Filter dataset to include only chosen classes
    if chosen_classes is not None:
        df = df[df['label'].isin(chosen_classes)]
        print(f"Filtering dataset to include only chosen classes: {chosen_classes}")

    # Count the number of points for each class
    class_counts = df['label'].value_counts().sort_index()

    # Print the original class distribution
    print("Original class distribution:")
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

    # Print the rebalanced class distribution
    rebalanced_class_counts = rebalanced_df['label'].value_counts().sort_index()
    print("\nRebalanced class distribution:")
    for class_label, count in rebalanced_class_counts.items():
        print(f"Class {class_label}: {count} points")

    # Split the rebalanced dataset into training and evaluation sets
    train_df, eval_df = train_test_split(rebalanced_df, test_size=(1 - train_split), random_state=42, stratify=rebalanced_df['label'])

    print(f"\nDataset split into:")
    print(f"- Training set: {len(train_df)} points ({train_split * 100:.0f}%)")
    print(f"- Evaluation set: {len(eval_df)} points ({(1 - train_split) * 100:.0f}%)")

    # save the csv files
    if output_dataset_folder is None:
        raise ValueError(f"Error: Output directory for csv files containing training and eval data was not specified.")
    os.makedirs(output_dataset_folder, exist_ok=True)
    train_csv = f"{output_dataset_folder}/train_dataset.csv"
    eval_csv = f"{output_dataset_folder}/eval_dataset.csv"

    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)

    print(f"\nTraining dataset saved to: {train_csv}")
    print(f"Evaluation dataset saved to: {eval_csv}")

    return train_df, eval_df




