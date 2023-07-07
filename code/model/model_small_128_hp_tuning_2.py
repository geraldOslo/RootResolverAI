"""
Created on Wed May 31 2023
Second attempt to PAI scorling by AI using a simple CNN model
@author: ec-gerald
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


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
    x_col="filename",
    y_col='prediction',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=data_dir,
    x_col="filename",
    y_col='prediction',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# convert generators to arrays
# Note: This may consume a lot of memory if your dataset is large
X_train, y_train = next(train_generator)
X_val, y_val = next(valid_generator)


def build_model(num_conv_layers=1, filters_layer_1=32, num_dense_layers=1, units_dense_1=32, learning_rate=0.01):
    model = Sequential()
    for i in range(num_conv_layers):
        model.add(Conv2D(filters_layer_1, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    for i in range(num_dense_layers):
        model.add(Dense(units=units_dense_1, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# wrap the model with KerasClassifier
model = KerasClassifier(build_fn=build_model, epochs=epochs)

# define the grid search parameters
param_grid = {
    'num_conv_layers': [1, 2, 3],
    'filters_layer_1': [32, 64, 128],
    'num_dense_layers': [1, 2, 3],
    'units_dense_1': [32, 64, 128],
    'learning_rate': [0.1, 0.01, 0.001]
}

# prepare the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# fit the grid search
grid_result = grid.fit(X_train, np.argmax(y_train, axis=1))

# print results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")

# Now build the model with the optimal hyperparameters
model = build_model(**grid_result.best_params_)

# create early stopping callback
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
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

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model
model.save("my_model.h5")
