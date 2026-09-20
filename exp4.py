import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Sample names
names = ["alex", "alice", "anna", "bob", "brad",
         "carla", "david", "daniel", "eric", "emma"]

# Characters
chars = sorted(set("".join(names)))
char_to_num = {c:i for i,c in enumerate(chars)}

# Create input and output
X = []
y = []

for name in names:
    for i in range(len(name)-1):
        X.append([char_to_num[c] for c in name[:i+1]])
        y.append(char_to_num[name[i+1]])

# Padding
X = tf.keras.utils.pad_sequences(X, padding="pre")
y = np.array(y)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 16),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(X, y, epochs=100)

# Plot loss
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
