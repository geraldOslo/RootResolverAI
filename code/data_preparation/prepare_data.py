# -*- coding: utf-8 -*-
"""
This file extracts data to use in AI project:
    # datafile using the Endodontic plugin to ImageJ
    # Images are cropped around the diagnosed apex to fov x fov
    # Missing area is padded with black pixels
    # CSV-file with image name and PAI is written
    # gerald@odont.uio.no 2023
"""


import pandas as pd
import csv
import cv2
import os


# source_datafile = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized/codefile.csv"
# source_images_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/anonymized"

# destination_path = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/clips"
# destination_datafile = "codefile1.csv"

#source_images_path = "//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/dag/anon/"

source_images_path = '//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/deniz/anonymized'

csv_path = os.path.join(source_images_path, "codefile.csv")

#destination_path = "//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/dag/clips/"
destination_path = '//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/deniz/clips'

destination_datafile = "codefile.csv"


fov = 256   # final clip size


df = pd.read_csv(csv_path, delimiter=',')

# Step 2: Process each row in the DataFrame
for index, row in df.iterrows():
    filename = "{:03d}.tiff".format(int(row['anon_file_number']))  # Convert int to filename
    image_file = os.path.join(source_images_path, filename)
    image = cv2.imread(image_file)

    # Step 3: Crop the image around the Ape, ApeY points
    ape_x = int(row['x'] * 25)  # Converting mm to pixels
    ape_y = int(row['y'] * 25)  # Converting mm to pixels
    crop_size = fov
    x = max(0, ape_x - crop_size // 2)
    y = max(0, ape_y - crop_size // 2)
    crop = image[y:y + crop_size, x:x + crop_size]

    # Step 4: Save the cropped image
    destination_file = os.path.join(destination_path, filename)
    cv2.imwrite(destination_file, crop)

    # Step 5: Write the CSV file
    csv_data = {'fileName': filename, 'PAI': row['pai']}
    csv_destination_path = os.path.join(destination_path, 'output.csv')
    with open(csv_destination_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['fileName', 'PAI'])
        if os.stat(csv_destination_path).st_size == 0:
            writer.writeheader()
        writer.writerow(csv_data)