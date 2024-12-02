from utils.point_cloud_data_utils import sample_data, read_file_to_numpy, reservoir_sample_data, reservoir_sample_with_subset, filter_features_in_csv
import pandas as pd
from utils.point_cloud_data_utils import subtiler
import laspy
import numpy as np
import os
from datetime import datetime
from utils.create_dataset import create_dataset, create_train_eval_datasets

'''still, error in matching subset with selected array. probably due to precision issues'''
# - creare nuovo dataset, i.e. file enorme con tanti punti + train/eval con train sui 2 milioni/2.5 di punti *DONE
# - finire di testare train_data_utils con dati reali, e vedere se è molto più lento    *DONE
# - testare che il training funzioni come al solito con sta nuova selection, con test_training  *DONE


"""train_df, eval_df = create_train_eval_datasets(csv_file='data/datasets/full_dataset.csv',
                               max_points_per_class=500_000,
                               chosen_classes=[3, 5, 6, 10, 11, 64],
                               train_split=0.8,
                               output_dataset_folder='data/datasets')"""


input_csv = 'data/datasets/full_dataset.csv'
csv_name = 'full_dataset.csv'
output_csv=f'data/datasets/filtered/{csv_name}'

filter_features_in_csv(input_csv=input_csv, output_csv=output_csv)

df = pd.read_csv(output_csv)
print(f"len of eval dataset: {len(df)}")
# Inspect column names
print("Column Names in the DataFrame:")
print(df.columns.tolist())


'''
subset_array, _ = read_file_to_numpy(data_dir='data/datasets/train_dataset.csv')
print(f"Subset array size: {subset_array.shape}")
sample_size = len(subset_array)*2.5
print(f"Sample size: {sample_size}")
sampled_df = reservoir_sample_with_subset(input_file='data/datasets/full_dataset.csv', subset_file='data/datasets/train_dataset.csv', sample_size=sample_size, save_dir='data/datasets/sampled_full_dataset', save=True)

sampled_df = reservoir_sample_data(input_file='data/datasets/full_dataset.csv', sample_size=sample_size, save_dir='data/datasets/sampled_full_dataset', save=True)'''

'''input_file = 'data/datasets/sampled_full_dataset/sampled_data_5000000.csv'
array, known_features = read_file_to_numpy(input_file)
print(f"array dimension before duplicate cleaning: {array.shape}")
cleaned_array = remove_duplicates_with_tolerance(data_array=array, tolerance=1e-8) 
print(f"array cleaned of duplicates shape: {cleaned_array.shape}")'''

'''input_folders = ['data/ground_and_offground/32_681000_4933500','data/ground_and_offground/32_690500_4930000', 'data/ground_and_offground/32_681500', 'data/ground_and_offground/32_684000', 'data/ground_and_offground/32_686000_4930500', 'data/ground_and_offground/32_686000_4933000']

create_dataset(input_folders=input_folders, 
               fused_las_folder = 'data/fused_ground_off_ground', 
               max_points_per_class=500000, 
               output_dataset_folder='data/datasets')'''



'''
input_file = 'data/training_data/21/train_21.csv'

# sampled_array = sample_data(input_file=input_file, sample_size=1000000, save_dir='data/training_data/overfitting_test/train/', save=True)

sampled_file = 'data/training_data/overfitting_test/train/sampled_data_1000000.csv'

file_to_read = input_file

data_array, features = read_file_to_numpy(file_to_read)
print(f'features {features}')

df = pd.read_csv(file_to_read)
print(df.columns)

# Count the number of points per class
class_counts = df['label'].value_counts()
print("Number of points per class:")
print(class_counts)

# Check for NaN values in the entire DataFrame
nan_mask = df.isna()

# Count the total number of NaN values
total_nan = nan_mask.sum().sum()

# Count the number of NaN values per column
nan_per_column = nan_mask.sum()

# Print the results
print(f"Total NaN values in the CSV file: {total_nan}")
print("Number of NaN values per column:")
print(nan_per_column)'''

'''def compare_dimensions(original_file, subtile_files):
    # Read the original LAS file to get its dimensions
    original_las = laspy.read(original_file)
    original_dimensions = set(original_las.point_format.dimension_names)
    
    # Print dimensions of the original file
    print(f"Original file ({original_file}) dimensions:")
    print(original_dimensions)
    
    # Iterate over each subtile and check its dimensions
    for subtile_file in subtile_files:
        # Read the subtile LAS file
        subtile_las = laspy.read(subtile_file)
        subtile_dimensions = set(subtile_las.point_format.dimension_names)
        
        # Print dimensions of the subtile
        print(f"\nSubtile file ({subtile_file}) dimensions:")
        print(subtile_dimensions)
        
        # Compare dimensions between the original file and the current subtile
        common_dimensions = original_dimensions & subtile_dimensions
        missing_from_original = subtile_dimensions - original_dimensions
        missing_from_subtile = original_dimensions - subtile_dimensions
        
        # Display the comparison results
        print(f"Common dimensions: {common_dimensions}")
        print(f"Dimensions in subtile but not in original: {missing_from_original}")
        print(f"Dimensions in original but not in subtile: {missing_from_subtile}")


# Example usage:
original_file = 'tests/test_subtiler/32_687000_4930000_FP21.las'  # Path to your original LAS file
subtile_files = ['tests/test_subtiler/32_687000_4930000_FP21_125_subtiles/subtile_687000_4930000.las']  # List of your subtile files

# Compare dimensions
compare_dimensions(original_file, subtile_files)'''

'''window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
large_value = int([value for label, value in window_sizes if label == 'large'][0])
print(large_value)


def sample_points_from_las(input_las_path, output_las_path, num_samples=1000):
    """
    Sample a specified number of points from an existing LAS file.

    Args:
    - input_las_path (str): Path to the original LAS file.
    - num_samples (int): Number of points to sample from the original LAS file.

    Returns:
    - sampled_points (dict): Dictionary with NumPy arrays for sampled x, y, z, and intensity.
    """
    # Read the LAS file
    las_data = laspy.read(input_las_path)

    # Get the total number of points in the original file
    total_points = len(las_data.x)
    
    # Ensure we don't sample more points than available
    num_samples = min(num_samples, total_points)

    # Randomly sample indices from the original points
    sampled_indices = np.random.choice(total_points, num_samples, replace=False)
    
    # Extract the sampled points and intensity
    sampled_x = las_data.x[sampled_indices]
    sampled_y = las_data.y[sampled_indices]
    sampled_z = las_data.z[sampled_indices]
    sampled_intensity = las_data.intensity[sampled_indices] if 'intensity' in las_data.point_format.dimension_names else np.zeros(num_samples)

    # Create a new header with the correct number of points (sampled points)
    new_header = laspy.LasHeader(point_format=las_data.header.point_format, version=las_data.header.version)
    new_header.offsets = las_data.header.offsets
    new_header.scales = las_data.header.scales
    # Create a new LasData object with the new header
    new_las = laspy.LasData(new_header)

    # Assign the sampled points to the new LasData object
    new_las.x = sampled_x
    new_las.y = sampled_y
    new_las.z = sampled_z
    new_las.intensity = sampled_intensity

    # Write the new LAS file with the sampled points
    new_las.write(output_las_path)

    print(f"New LAS file created with sampled points at {output_las_path}")


# Example usage
input_las_path = 'path_to_your_large_file.las'
output_las_path = 'sampled_points.las'
sample_points_from_las(input_las_path='tests/test_subtiler/32_687000_4930000_FP21.las', output_las_path=f'tests/test_subtiler/32_687000_4930000_FP21_sampled_1k.las', num_samples=3000000)


"""
Applies a mask to the given subtile LAS data to exclude overlap regions, saves the masked subtile to a new file,
and returns the path to the saved file along with the LAS data.

Args:
- subtile_filepath (str): The LAS subtile file path.
- tile_size (int): The size of the subtile in meters.
- overlap_size (int): The size of the overlap to exclude.
- is_northernmost (bool): If True, this subtile is the northernmost one (no upper overlap to exclude).
- is_rightmost (bool): If True, this subtile is the rightmost one (no right overlap to exclude).
- output_dir (str): Directory where the new subtile should be saved. If None, saves in the same directory as the original.

Returns:
- str: Path to the saved LAS subtile file.
- laspy.LasData: The LAS data object with the mask applied.
"""


"""def apply_mask_to_subtile(subtile_filepath, tile_size=125, overlap_size=30, is_northernmost=False, is_rightmost=False, output_dir=None):

    # Extract lower-left coordinates from the subtile filename
    filename = os.path.basename(subtile_filepath)
    parts = filename.split('_')
    lower_left_x = int(parts[1])  # Extract lower-left x from filename
    lower_left_y = int(parts[2].split('.')[0])  # Extract lower-left y from filename

    print(f'lower left x:{lower_left_x}')
    print(f'lower left y:{lower_left_y}')

    # Load the subtile LAS file
    subtile_las = laspy.read(subtile_filepath)

    # Define the bounds of the subtile
    upper_right_x = lower_left_x + tile_size
    upper_right_y = lower_left_y + tile_size

    min_x = subtile_las.x.min()
    max_x = subtile_las.x.max()
    min_y = subtile_las.y.min()
    max_y = subtile_las.y.max()

    print(f"Subtile bounds:")
    print(f"  Min X: {min_x}, Max X: {max_x}")
    print(f"  Min Y: {min_y}, Max Y: {max_y}")

    upper_bound_x = max_x - (overlap_size/2)
    upper_bound_y = max_y - (overlap_size/2)

    lower_bound_x = min_x + (overlap_size/2)
    lower_bound_y = min_y + (overlap_size/2)

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
        mask = (
            (subtile_las.x < upper_bound_x) &  # Exclude right overlap if not rightmost
            (subtile_las.y < upper_bound_y )  # Exclude top overlap if not northernmost
        )
        

    # Create a new LAS data object with the same header
    new_las = laspy.LasData(subtile_las.header)

    # Apply the mask to filter points and copy them to the new LAS data object
    new_las.points = subtile_las.points[mask]
    new_las.label = subtile_las.label[mask]

    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(subtile_filepath)  # Default to the same directory as the original

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct the output filename
    output_filename = os.path.join(output_dir, f"masked__{timestamp}_{filename}")

    # Save the masked subtile
    new_las.write(output_filename)

    print(f"Masked subtile saved at: {output_filename}")

    return output_filename, new_las

output_filename, new_las = apply_mask_to_subtile(subtile_filepath='tests/test_subtiler/32_687000_4930000_FP21_sampled_10k_250_subtiles/subtile_687220_4930220.las', 
                                                 tile_size=250, overlap_size=30, is_northernmost=True, is_rightmost=True, output_dir='tests/test_subtiler/cut_tests')'''''
                                                 

'''cleaned_las = clean_bugged_subtile(bugged_las_path='tests/fused_las/fused_32_686000_4930500_FGLn.las')'''