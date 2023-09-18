# -*- coding: utf-8 -*-
"""
Created on Sun May 28 07:27:27 2023
Script compares labels and predictions for n random roots

@author: ec-gerald
"""

import os
import numpy as np
import pandas as pd
#import random
#import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


from keras.preprocessing import image

target_size = 256

# load the saved model
saved_model = load_model(r'C:\temp\my_model.h5')

data_dir = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\clips"
csv_file = os.path.join(data_dir, "codefile.csv")

# load the CSV file using pandas
df = pd.read_csv(csv_file)
df.columns = ['filename', 'label']

# shuffle dataframe
df = df.sample(frac=1).reset_index(drop=True)

# prepare a dataframe to store results
result_df = pd.DataFrame(columns=['filename', 'label', 'prediction'])

# choose n images at random
n = 30  # for example
for _, row in df.head(n).iterrows():
    filename = row['filename']
    label = row['label']
    
    # # load image and prepare it for prediction
    # img_path = os.path.join(data_dir, filename)
    # img = load_img(img_path, target_size=(target_size, target_size), color_mode='grayscale')
    # img_array = img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.  # same normalization as before
    # generate random grayscale image and prepare it for prediction
    
    # Test on random generated images:
    img_array = np.random.rand(target_size, target_size, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    prediction = saved_model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # add result to dataframe
    new_row = pd.DataFrame({'filename': [filename], 'label': [label], 'prediction': [predicted_class]})
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    
# save results to a CSV file
result_df.to_csv('results.csv', index=False)
