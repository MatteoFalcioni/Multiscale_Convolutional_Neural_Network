from utils.point_cloud_data_utils import sample_data, combine_and_save_csv_files, read_file_to_numpy, reservoir_sample_data
import pandas as pd

input_file = 'data/training_data/overfitting_test/inference/sampled_rebalanced_data.csv'


# sampled_data = reservoir_sample_data(input_file, sample_size=1000000, save=True, save_dir='data/training_data/overfitting_test/train/', feature_to_use=None, chunk_size=1000000)
# sampled_array = sample_data(input_file=input_file, sample_size=1000000, save_dir='data/training_data/overfitting_test/train/', save=True)

sampled_file = 'data/training_data/overfitting_test/train/sampled_data_1000000.csv'



file_to_read = input_file

data_array, features = read_file_to_numpy(file_to_read)
print(f'features {features}')

df = pd.read_csv(file_to_read)

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
print(nan_per_column)