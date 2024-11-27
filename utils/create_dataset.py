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


def stitch_and_fuse_multiple(file_pairs, output_folder):
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






