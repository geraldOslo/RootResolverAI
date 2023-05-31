# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:58:45 2023
First attempt to PAI scorling by AI using a model based on the VGG16 architecture

The model was far too complicated and did not work
@author: ec-gerald
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

image_size = 256
epochs = 30

system = "win"

# specify your directory and CSV file paths
if system == "linux":
    data_dir = "/fp/homes01/u01/ec-gerald/My Projects/ec192/data/endo-radiographs/clips_balanced"
else:
    data_dir = r"\\aspasia.ad.fp.educloud.no\ec192\data\endo-radiographs\clips"
    
csv_file = os.path.join(data_dir, "codefile.csv")

# load the CSV file using pandas
df = pd.read_csv(csv_file)
df.columns = ['filename', 'prediction']

# convert labels to str because ImageDataGenerator treats all inputs as strings
df['prediction'] = df['prediction'].astype(str)

# split the data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# specify your target image size (this will be the input shape for your CNN)
target_size = (image_size, image_size)

# specify batch size
batch_size = 16  # adjust as needed

# create generators
# create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col="filename",  # this might be "image_name" or something similar depending on your CSV
    y_col='prediction',  # this might be "diagnosis" or something similar depending on your CSV
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # use 'categorical' since we have now one-hot encoded labels
    color_mode='grayscale'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=data_dir,
    x_col="filename",  # this might be "image_name" or something similar depending on your CSV
    y_col='prediction',  # this might be "diagnosis" or something similar depending on your CSV
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # use 'categorical' since we have now one-hot encoded labels
    color_mode='grayscale'
)

def create_model():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))  # grayscale image
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # output layer with 5 nodes

    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# create the model
model = create_model()

# print the model summary
model.summary()

# Define the checkpoint and early stopping
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')

# This is just an example, the actual function depends on your needs
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_schedule_callback = LearningRateScheduler(scheduler)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,  # steps_per_epoch should typically be equal to the total number of samples divided by the batch size
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps= len(valid_df) // batch_size,  # similar rule as steps_per_epoch but for the validation data
    callbacks=[checkpoint, early, lr_schedule_callback]
)

model.save("my_model.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.savefig('accuracy.png')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig('history.png')
