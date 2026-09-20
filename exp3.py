import kagglehub
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Download dataset
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print("Dataset path:", path)

# Load LFW dataset
faces = fetch_lfw_people(
    min_faces_per_person=70,
    color=True
)

# Use only 2000 images
X = faces.images[:2000] / 255.0
y = faces.target[:2000]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# CNN model
model = Sequential([
    Input(shape=X.shape[1:]),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation="relu"),

    Dense(
        len(faces.target_names),
        activation="softmax"
    )
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=2,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Prediction
prediction = model.predict(X_test[:1], verbose=0)

print(
    "Predicted Person:",
    faces.target_names[np.argmax(prediction)]
)

# Accuracy graph
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()
