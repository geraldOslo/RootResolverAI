# Some of the clips are of low quality
# After manually deleting the image this script updates csv

import os
import pandas as pd

# Path to the directory where the CSV and image files are stored
data_dir = "/fp/homes01/u01/ec-gerald/ec192/data/endo-radiographs/charlies/clips"

# Backup the original CSV file
os.rename(os.path.join(data_dir, "codefile.csv"), os.path.join(data_dir, "codefile_backup.csv"))

# Load the CSV into a DataFrame
df = pd.read_csv(os.path.join(data_dir, "codefile_backup.csv"))

# Initialize an empty list to store indices of rows to drop
rows_to_drop = []

# Loop through the DataFrame to find missing image files
for index, row in df.iterrows():
    image_path = os.path.join(data_dir, row['filename'])
    if not os.path.exists(image_path):
        rows_to_drop.append(index)
        print("Deleting line: " + image_path)

# Drop rows corresponding to missing image files
df.drop(rows_to_drop, inplace=True)

# Save the cleaned DataFrame back to CSV
df.to_csv(os.path.join(data_dir, "codefile.csv"), index=False)
