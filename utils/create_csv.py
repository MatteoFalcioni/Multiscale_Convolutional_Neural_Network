from os import makedirs
import pandas as pd
import numpy as np
import laspy
import glob
from tqdm import tqdm

# Path to the directory containing LAS files
LAS_DIRECTORY = 'data/training_data/overfitting_test/train/'

# Features and labels to keep
VARIABLES = ['x', 'y', 'z', 'intensity', 'return_number', 'number_of_returns',
             'classification', 'red', 'green', 'blue', 'nir',
             'ndvi', 'ndwi', 'ssi',
             'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b',
             'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
             'N_h', 'delta_z_fl', 'label']

# Mapping to rename columns
RENAME = {'l1_b': 'l1', 'l2_b': 'l2', 'l3_b': 'l3', 'planarity_b': 'planarity',
          'sphericity_b': 'sphericity', 'linearity_b': 'linearity', 'entropy_b': 'entropy',
          'theta_b': 'theta', 'theta_variance_b': 'theta_variance', 'mad_b': 'mad',
          'delta_z_b': 'delta_z'}

# Define the classes of interest and the desired sample size per class
CLASS_MAP = {
    'grass': 3,
    'trees': 5,
    'buildings': 6,
    'railway' : 10,
    'roads': 11,
    'cars': 64
}   

sample_size = 2500000
target_class_size = int(sample_size / len(CLASS_MAP))  # For defined classes

# Get a list of all LAS files in the directory
las_files = glob.glob(LAS_DIRECTORY + '32_*00/32_*Ln.las')

# Initialize lists to accumulate data for each class
class_data = {class_name: [] for class_name in CLASS_MAP}

# Iterate over each LAS file and extract the relevant classes
for las_file in tqdm(las_files, desc="Processing LAS files"):
    las_data = laspy.read(las_file)

    for class_name, label in CLASS_MAP.items():
        class_las = las_data.points[las_data.points['label'] == label]
        if len(class_las.x) > 0:
            class_df = pd.DataFrame({name: np.array(class_las[name]) for name in VARIABLES})
            class_df.rename(columns=RENAME, inplace=True)
            class_data[class_name].append(class_df)

# Rebalance classes: sample target_class_size points from each class
sampled_dataframes = []
for class_name, dfs in tqdm(class_data.items(), desc="Rebalancing classes"):
    combined_class_df = pd.concat(dfs, ignore_index=True)  # Concatenate all dataframes for this class at once
    if len(combined_class_df) > target_class_size:
        sampled_dataframes.append(combined_class_df.sample(target_class_size))
    else:
        sampled_dataframes.append(combined_class_df)

# Combine the rebalanced samples into a single DataFrame
combined_data = pd.concat(sampled_dataframes, ignore_index=True)

# Shuffle the final dataset to mix the classes
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save the combined, rebalanced dataset
output_dir = LAS_DIRECTORY
makedirs(output_dir, exist_ok=True)
combined_data.to_csv(f'{output_dir}/sampled_rebalanced_data.csv', index=False)
print(f'Sampled and rebalanced data saved to {output_dir}/sampled_rebalanced_data.csv')
