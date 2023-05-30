# -*- coding: utf-8 -*-
"""
This file extracts data to use in AI project:
    # datafile using the Endodontic plugin to ImageJ
    # Images are cropped around the diagnosed apex to fov x fov
    # Missing area is padded with black pixels
    # Data augmentation is applied: noise, zoom, rotation
    # Data augmentation is done so that there will be a balanced set of pai
    # i.e. more copies are generated for images with less common pai
    # CSV-file with image name and PAI is written
    # gerald@odont.uio.no 2023
"""

# -*- coding: utf-8 -*-


import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# source_datafile = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized/codefile.csv"
# source_images_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized"

# destination_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/clips"
# destination_datafile = "codefile1.csv"


source_images_path = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\anonymized"
source_datafile = os.path.join(source_images_path, "codefile.csv")

destination_path = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\clips_balanced"
destination_datafile = "codefile1.csv"

# Prepare a DataFrame for saving the output data
output_df = pd.DataFrame(columns=['filename', 'pai'])


apex_coord = {}
filenames ={}
n = 0
fov = 256   # final clip size
fov2 = fov//2
clip = int(fov*1.2) # raw clip for data augmentation
clip2 = clip//2
augments = 10 # number of data augmentations per image


# load the CSV file using pandas
df = pd.read_csv(source_datafile, usecols=[1,2,3,4])

# augment each image as many times as needed to get the desired number for each class
# create a count for each class
total_counts = df['pai'].value_counts().to_dict()
desired_number = 300  # set this to the maximum number you want for each class
#aug_counts = df['pai'].value_counts().apply(lambda x: max(0, desired_number - x))



# This adds the possibility for noise in data augmentation:
class ImageDataGeneratorWithNoise(ImageDataGenerator):
    def __init__(self, noise_stddev_range=(0.02, 0.15), **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev_range = noise_stddev_range

    def random_transform(self, x):
        x = super().random_transform(x)
        noise_stddev = np.random.uniform(self.noise_stddev_range[0], self.noise_stddev_range[1])
        noise = np.random.normal(0, noise_stddev, x.shape)
        x = x + noise
        x = np.clip(x, 0., 255.)  # ensure values stay within [0, 255]
        return x

# Define image data generator
datagen = ImageDataGeneratorWithNoise(
    rotation_range=15,       # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1,        # Randomly zoom image 
    horizontal_flip=True,    # randomly flip images
    vertical_flip=False,     # do not randomly flip images vertically
    fill_mode='constant',
    cval=0.0,                # this means padding will be black
    noise_stddev_range=(0.08, 0.2)  # add Gaussian noise with random standard deviation between 0.02 and 0.15
)


# Read csv file, open image, apply augmentation and output files to target directory:


n = 1 # Counter for output file names
for _, row in df.iterrows():
    filename = "{:03d}.tiff".format(int(row['anon file number']))  # Convert int to filename
    path = os.path.join(source_images_path, filename)

    x = float(row['x'])
    y = float(row['y'])
    pai = int(row['pai'])
    augments = max(0, desired_number - total_counts[pai])
    
    if total_counts[pai] >= desired_number:
        continue

        
    try:
        im = Image.open(path)
        width, height = im.size
        res = im.info['resolution'][0]
        x = round(x*res)
        y = round(y*res)
        
        
        # Crop image, add black pixels if apex is to near to the border
        # Calculate how much to add on each side
        left = max(0, clip2 - x)
        top = max(0, clip2 - y)
        right = max(0, x + clip2 - im.width)
        bottom = max(0, y + clip2 - im.height)

        # Add the border
        im = ImageOps.expand(im, border=(left, top, right, bottom), fill='black')

        # Now, calculate the new point coordinates
        x += left
        y += top

        box = (x - clip2, y - clip2, x + clip2, y + clip2)
        img = im.crop(box)
        

        # Apply the data generator
        i = 0 # Counter for augmentations
        img_array = np.array(img).reshape((1,) + img.size + (1,))  # reshape to 4D for ImageDataGenerator
        for batch in datagen.flow(img_array, batch_size=1):
            if i >= augments:  # limit the number of augmented images per original image
                break
        
            generated_img = Image.fromarray(batch[0].astype('uint8').reshape(img.size))
        
            # Calculate coordinates to center crop the image to 256x256
            width, height = generated_img.size
            left = (width - fov)/2
            top = (height - fov)/2
            right = (width + fov)/2
            bottom = (height + fov)/2
        
            # Crop the image
            generated_img = generated_img.crop((left, top, right, bottom))
        
            new_filename = str(n).zfill(3) + ".tiff"
            new_row = pd.DataFrame({'filename': [new_filename], 'pai': [pai]})
            output_df = pd.concat([output_df, new_row], ignore_index=True)

            new_path = os.path.join(destination_path, new_filename)
            generated_img.save(new_path)
        
            i += 1
            n += 1
            total_counts[pai] += 1  # Increment the count for this class after an image is augmented and saved
        
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")

# Save the output DataFrame to a CSV file
output_df.to_csv(os.path.join(destination_path, destination_datafile), index=False)
print(n)      

