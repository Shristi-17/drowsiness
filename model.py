from tensorflow.keras.models import load_model

# Load the model
model = load_model("drowsiness_model.h5")

# Print the model architecture
model.summary()
