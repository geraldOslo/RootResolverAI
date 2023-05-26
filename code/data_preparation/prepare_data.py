# -*- coding: utf-8 -*-
"""
This file extracts data to use in AI project:
    # datafile using the Endodontic plugin to ImageJ
    # Images are cropped around the diagnosed apex to fov x fov
    # Missing area is padded with black pixels
    # Data augmentation is applied: noise, zoom, rotation
    # CSV-file with image name and PAI is written
    # gerald@odont.uio.no 2023
"""


import os
import csv
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# source_datafile = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized/codefile.csv"
# source_images_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized"

# destination_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/clips"
# destination_datafile = "codefile1.csv"

source_datafile = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\anonymized\codefile.csv"
source_images_path = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\anonymized"

destination_path = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\clips"
destination_datafile = "codefile1.csv"


apex_coord = {}
filenames ={}
n = 0
fov = 256   # final clip size
fov2 = fov//2
clip = int(fov*1.2) # raw clip for data augmentation
clip2 = clip//2
augments = 10 # number of data augmentations per image


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
with open(os.path.join(destination_path, destination_datafile), 'w', newline='') as outfile:
    with open(source_datafile) as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        next(read)  # This skips the first row of the CSV file.
        writer = csv.writer(outfile)
        
        n = 1 # Counter for output file names
        for row in read:
            filename = "{:03d}.tiff".format(int(row[1])) # Convert int to filename
            filenames[n] = filename
            path = os.path.join(source_images_path, filename)
            print(path)

            x = float(row[2])
            y = float(row[3])
            pai = int(row[4])

                
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
                    new_path = os.path.join(destination_path, new_filename)
                
                    # write to the new csv file
                    writer.writerow([new_filename, pai])
                    generated_img.save(new_path)
                
                    i += 1
                    n += 1
                
                
            except Exception as e:
                print("Oops!", e.__class__, "occurred.")


print(n)      
csvfile.close()
outfile.close()
