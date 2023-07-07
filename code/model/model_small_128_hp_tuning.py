# -*- coding: utf-8 -*-
"""
Created on Wed May 31 2023
Second attempt to PAI scorling by AI using a simple CNN model

@author: ec-gerald
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch


#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf


system = "linux"

data_dir =  "/fp/homes01/u01/ec-gerald/ec192/data/endo-radiographs/deniz/clips_128"
    
       
csv_file = os.path.join(data_dir, "codefile.csv")

image_size = 128
epochs = 30



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


max_conv_layers = 5
max_dense_layers = 5

def build_model(hp):
    model = Sequential()
    
    # Tune the number of convolutional layers
    for i in range(hp.Int('num_conv_layers', 1, max_conv_layers)):
        # Tune the number of filters
        model.add(Conv2D(hp.Choice(f'filters_layer_{i+1}', values=[32, 64, 128]), (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    # Tune the number and size of dense layers
    for i in range(hp.Int('num_dense_layers', 1, max_dense_layers)):
        model.add(Dense(units=hp.Int(f'units_dense_{i+1}', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dropout(0.5))
        
    model.add(Dense(5, activation='softmax'))  # output layer with 5 nodes

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # number of different models to try
)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min')
tuner.search(train_generator,
             validation_data=valid_generator,
             epochs=10,
             callbacks=[early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of convolutional layers is {best_hps.get('num_conv_layers')}, the optimal number of filters in the convolutional layers are {', '.join([str(best_hps.get(f'filters_layer_{i+1}')) for i in range(best_hps.get('num_conv_layers'))])}, the optimal number of dense layers is {best_hps.get('num_dense_layers')}, the optimal number of neurons in the dense layers are {', '.join([str(best_hps.get(f'units_dense_{i+1}')) for i in range(best_hps.get('num_dense_layers'))])}, and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Now build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# print the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,  # steps_per_epoch should typically be equal to the total number of samples divided by the batch size
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps= len(valid_df) // batch_size,  # similar rule as steps_per_epoch but for the validation data
    callbacks=[early]
)

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

model.save("my_model.h5")

