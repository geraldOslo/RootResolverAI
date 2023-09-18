from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_model.h5')

# Print the structure of the model
model.summary()