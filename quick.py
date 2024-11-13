from utils.point_cloud_data_utils import sample_data, combine_and_save_csv_files, read_file_to_numpy
import pandas as pd
from utils.point_cloud_data_utils import subtiler
import laspy

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

window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
large_value = int([value for label, value in window_sizes if label == 'large'][0])
print(large_value)