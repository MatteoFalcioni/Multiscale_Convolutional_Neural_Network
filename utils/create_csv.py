from os import makedirs
import pandas as pd
import numpy as np
import laspy
import glob
from tqdm import tqdm  # For progress bar

# Path to the directory containing LAS files
LAS_DIRECTORY = 'data/training_data/overfitting_test/inference/'

VARIABLES = ['x', 'y', 'z', 'intensity', 'return_number', 'number_of_returns',
             'classification', 'red', 'green', 'blue', 'nir',
             'ndvi', 'ndwi', 'ssi',
             'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b',
             'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
             'N_h', 'delta_z_fl', 'label']

RENAME = {'l1_b': 'l1', 'l2_b' : 'l2', 'l3_b' : 'l3',
          'planarity_b' : 'planarity', 'sphericity_b' : 'sphericity',
          'linearity_b' : 'linearity', 'entropy_b' : 'entropy',
          'theta_b' : 'theta', 'theta_variance_b' : 'theta_variance',
          'mad_b' : 'mad', 'delta_z_b' : 'delta_z'}

# Get a list of all LAS files in the directory
las_files = glob.glob(LAS_DIRECTORY + '32_*00/32_*Ln.las')

print(f"Processing {len(las_files)} LAS files.")

# List to store all the data
all_data = []

# Iterate over each LAS file
for las_file in tqdm(las_files, desc="Processing LAS files", unit="file"):
    las_data = laspy.read(las_file)

    # Convert the LAS data to a DataFrame
    las_df = pd.DataFrame({name: np.array(las_data[name]) for name in VARIABLES})

    # Rename columns to match the required format
    las_df.rename(columns=RENAME, inplace=True)

    # Append this file's data to the all_data list
    all_data.append(las_df)

print('combining data into a single file')
# Combine all the data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save the combined data to a CSV file
combined_data.to_csv('data/training_data/overfitting_test/inference/inference_data.csv', index=False)

print(f"Combined data saved with {len(combined_data)} points.")


