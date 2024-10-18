from os import makedirs
import pandas as pd
import numpy as np
import laspy
import glob

# Path to the directory containing LAS files
LAS_DIRECTORY = 'data/training_data/overfitting_test/train/'

VARIABLES = ['x', 'y', 'z', 'intensity', 'return_number', 'number_of_returns',
             'classification', 'red', 'green', 'blue', 'nir',
             'ndvi', 'ndwi', 'ssi',
             'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b',
             'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
             'N_h', 'delta_z_fl',
             'label']

RENAME = {'l1_b': 'l1', 'l2_b' : 'l2', 'l3_b' : 'l3',
          'planarity_b' : 'planarity', 'sphericity_b' : 'sphericity',
          'linearity_b' : 'linearity',
          'entropy_b' : 'entropy', 'theta_b' : 'theta',
          'theta_variance_b' : 'theta_variance', 'mad_b' : 'mad',
          'delta_z_b' : 'delta_z'}

# Get a list of all LAS files in the directory
las_files = glob.glob(LAS_DIRECTORY + '32_*00/32_*Ln.las')

print("\n\n", len(las_files), "\n\n")

# Create empty DataFrames to store the combined data
combined_df = pd.DataFrame(columns=VARIABLES)

# Iterate over each LAS file
for las_file in las_files:
    print(las_file)
    # Read the LAS file and extract the data
    las_data = laspy.read(las_file)
    
    # Extract all labels 
    for label, class_name in [(3, 'Grass'), (5, 'Trees'), (6, 'Buildings'), (10, 'Rail'), (11, 'Roads'), (64, 'Cars')]:
        class_las = las_data.points[las_data.points['label'] == label]
        if len(class_las.x) > 0:
            class_df = pd.DataFrame({name: np.array(class_las[name]) for name in VARIABLES})
            combined_df = pd.concat([combined_df, class_df], ignore_index=True)

# Shuffle the combined DataFrame
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Rename columns for consistency
combined_df.rename(columns=RENAME, errors='raise')

# Save the combined dataset as CSV
output_csv_path = 'data/training_data/overfitting_test/train/combined_train_data.csv'
combined_df.to_csv(output_csv_path, index=False)

print(f"Combined dataset saved to {output_csv_path}")
