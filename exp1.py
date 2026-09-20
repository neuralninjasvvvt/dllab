import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR input and output
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0,1,1,0])

# Create neural network
model = Sequential()

# Hidden layer with 4 neurons
model.add(Dense(4, input_dim=2, activation="relu"))

# Output layer with 1 neuron
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model and store loss history
history = model.fit(X, y, epochs=500, verbose=0)

# Predict XOR outputs
p = (model.predict(X, verbose=0) > 0.5).astype(int)

print("Predictions:", p.ravel())

# Evaluate accuracy and loss
loss, accuracy = model.evaluate(X, y, verbose=0)

print("Loss:", loss)
print("Accuracy:", accuracy)

# Plot loss graph
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
