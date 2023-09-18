# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:54:37 2023

@author: ec-gerald
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
import random

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # output layer with 5 nodes

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_data(image_path, csv_path):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img = Image.open(os.path.join(image_path, row['fileName']))
        img = img.resize((256, 256))
        img = np.array(img)
        img = img.reshape((256, 256, 1))  # Reshape to add the channel dimension
        images.append(img)

        # Subtract 1 from the label to make them 0-based (i.e., from 1-5 to 0-4)
        labels.append(row['PAI'] - 1)

    return np.array(images), np.array(labels)

image_path = "//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/dag/clips/"
csv_path = "//aspasia.ad.fp.educloud.no/ec192/data/endo-radiographs/dag/clips/codefile.csv"
X, y = load_data(image_path, csv_path)

# One-hot encode the labels
y = to_categorical(y, num_classes=5)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images up to 10 degrees
    zoom_range=0.1,  # Zoom in/out up to 10%
    horizontal_flip=True,  # Allow horizontal flipping
    fill_mode='nearest',  # Fill in newly created pixels
    preprocessing_function=add_noise  # Add noise
)

datagen.fit(X_train)

model = create_model()
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    validation_data=(X_test, y_test),
                    epochs=50)

plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'blue', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)

model.save("my_model.h5")

