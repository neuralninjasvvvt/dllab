import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed

# Data
sentences = [
    "the cat sits", "the dog runs", "a boy plays",
    "a girl sings", "the student reads", "the teacher teaches",
    "the man walks", "the woman cooks",
    "birds fly", "dogs eat food"
]

tags = [
    "DT NN VBZ", "DT NN VBZ", "DT NN VBZ",
    "DT NN VBZ", "DT NN VBZ", "DT NN VBZ",
    "DT NN VBZ", "DT NN VBZ",
    "NNS VB", "NNS VB NN"
]

# Tokenize
wt = Tokenizer(oov_token="<UNK>")
wt.fit_on_texts(sentences)
X = wt.texts_to_sequences(sentences)

tt = Tokenizer()
tt.fit_on_texts(tags)
Y = tt.texts_to_sequences(tags)

# Padding
X = pad_sequences(X, maxlen=5, padding="post")
Y = pad_sequences(Y, maxlen=5, padding="post")

# Model
model = Sequential([
    Embedding(len(wt.word_index) + 1, 32),
    Bidirectional(LSTM(32, return_sequences=True)),
    TimeDistributed(
        Dense(len(tt.word_index) + 1, activation="softmax")
    )
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X, Y, epochs=20, verbose=0)

print("Model training completed.")

# Evaluate
loss, accuracy = model.evaluate(X, Y, verbose=0)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# User input
sentence = input("\nEnter a sentence: ")

seq = pad_sequences(
    wt.texts_to_sequences([sentence]),
    maxlen=5,
    padding="post"
)

# Predict
pred = np.argmax(
    model.predict(seq, verbose=0)[0],
    axis=1
)

# Display POS tags
print("\nPOS Tags:")

for word, tag in zip(sentence.lower().split(), pred):
    print(
        word,
        "->",
        tt.index_word.get(tag, "UNKNOWN")
    )
