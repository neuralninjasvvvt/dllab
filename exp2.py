import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Download dataset
dataset_path = kagglehub.dataset_download(
    "sachinpatel21/az-handwritten-alphabets-in-csv-format"
)

# Load only 20,000 rows to reduce RAM and training time
data = pd.read_csv(
    os.path.join(dataset_path, "A_Z Handwritten Data.csv"),
    nrows=20000
)

# Separate images and labels
X = data.drop("0", axis=1).values
y = data["0"].values

# Reshape images to 28x28 grayscale
X = X.reshape(-1, 28, 28, 1) / 255.0

# Convert labels into one-hot encoding
y = to_categorical(y, 26)

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Build CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(26, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=64,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot training loss
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
