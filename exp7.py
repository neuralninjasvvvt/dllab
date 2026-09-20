import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Data
english = [
    "hello","good morning","good night","how are you","i am fine",
    "thank you","what is your name","goodbye","welcome","i am happy"
]

french = [
    "bonjour","bonjour","bonne nuit","comment allez vous","je vais bien",
    "merci","quel est votre nom","au revoir","bienvenue","je suis heureux"
]

# Characters
chars = sorted(set("".join(english + french)))
char = {c:i for i,c in enumerate(chars)}
n = len(chars)

# Convert to numbers
X = [[char[c] for c in s] for s in english]
Y = [[char[c] for c in s] for s in french]

# Padding
X = tf.keras.utils.pad_sequences(X, padding="post")
Y = tf.keras.utils.pad_sequences(Y, padding="post")

# Model
model = Sequential([
    Embedding(n, 32),
    LSTM(64),
    Dense(n, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X, Y[:, -1],
    epochs=50,
    verbose=0
)

# Evaluate
loss, accuracy = model.evaluate(
    X, Y[:, -1],
    verbose=0
)

print("Training completed!")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Prediction
p = model.predict(X, verbose=0)
print("\nSample Output:")

for i in range(5):
    print(english[i], "->", french[i])

# Graph
plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
