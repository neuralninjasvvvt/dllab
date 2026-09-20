from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import matplotlib.pyplot as plt

# Load IMDB dataset
(X, y), (Xt, yt) = imdb.load_data(num_words=5000)

# Pad sequences to same length
X = pad_sequences(X, maxlen=200)
Xt = pad_sequences(Xt, maxlen=200)

# Build model
m = Sequential()

m.add(Embedding(5000, 32))
m.add(Bidirectional(LSTM(32)))
m.add(Dense(1, activation="sigmoid"))

# Compile model
m.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = m.fit(
    X, y,
    epochs=3,
    batch_size=128,
    validation_split=0.2,
    verbose=0
)

# Evaluate model
loss, accuracy = m.evaluate(Xt, yt, verbose=0)

print("Test Accuracy:", accuracy)

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()
