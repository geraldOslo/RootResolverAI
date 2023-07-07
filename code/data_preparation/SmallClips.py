# Import necessary libraries
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the directories for the original and new data
data_dir = "/fp/homes01/u01/ec-gerald/ec192/data/endo-radiographs/deniz/clips"
new_data =  "/fp/homes01/u01/ec-gerald/ec192/data/endo-radiographs/deniz/clips_128"

# Load the CSV file with filenames and labels
df = pd.read_csv(os.path.join(data_dir, "codefile.csv"))

# Create the new directory if it does not exist
if not os.path.exists(new_data):
    os.makedirs(new_data)

def crop_center(filename, label, crop_size=(128, 128)):
    """
    This function opens an image, crops the center and saves the new image.
    """
    with Image.open(os.path.join(data_dir, filename)) as img:
        img_width, img_height = img.size
        left = (img_width - crop_size[0]) / 2
        top = (img_height - crop_size[1]) / 2
        right = (img_width + crop_size[0]) / 2
        bottom = (img_height + crop_size[1]) / 2
        new_img = img.crop((left, top, right, bottom))
        
        # Create new filename and save the cropped image
        new_filename = f"{os.path.splitext(filename)[0]}_center.tiff"
        new_img.save(os.path.join(new_data, new_filename))
        
        return new_filename, label

# Apply the crop_center function to all images in the DataFrame
new_rows = [crop_center(row.filename, row.prediction) for _, row in df.iterrows()]

# Create a new DataFrame with the new filenames and predictions
new_df = pd.DataFrame(new_rows, columns=['filename', 'prediction'])
new_df.to_csv(os.path.join(new_data, 'codefile.csv'), index=False)

def plot_images(df):
    """
    This function plots images in a grid, with 8x8 images per page.
    """
    filenames = df['filename'].tolist()
    num_images = len(filenames)
    
    # Calculate number of pages
    num_pages = num_images // (8*8)
    if num_images % (8*8):
        num_pages += 1

    for page in range(num_pages):
        # Extract filenames for this page
        page_filenames = filenames[page*64 : (page+1)*64]
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))

        for ax, filename in zip(axes.flatten(), page_filenames):
            img = Image.open(os.path.join(new_data, filename))
            ax.imshow(np.array(img), cmap='gray')
            ax.axis('off')

        plt.show()

plot_images(new_df)
