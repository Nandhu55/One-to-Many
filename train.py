
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Configuration
# -----------------------------
DATA_FILE = "names.txt"          # Place your dataset beside this file
MODEL_FILE = "name_generator.keras"
TOKENIZER_FILE = "char_tokenizer.pkl"
MAX_EPOCHS = 50
BATCH_SIZE = 128
EMBED_DIM = 64
RNN_UNITS = 128

# -----------------------------
# Load dataset
# -----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Rename your uploaded dataset to 'names.txt'."
    )

with open(DATA_FILE, "r", encoding="utf-8") as f:
    names = [line.strip().lower() for line in f if line.strip()]

# remove duplicates
names = list(dict.fromkeys(names))

# add start/end tokens
names = [f"^{n}$" for n in names]

chars = sorted(set("".join(names)))
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding
idx2char = {i: c for c, i in char2idx.items()}

with open(TOKENIZER_FILE, "wb") as f:
    pickle.dump(
        {
            "char2idx": char2idx,
            "idx2char": idx2char,
            "maxlen": max(len(x) for x in names)
        },
        f
    )

X = []
y = []

for word in names:
    encoded = [char2idx[c] for c in word]
    for i in range(1, len(encoded)):
        X.append(encoded[:i])
        y.append(encoded[i])

maxlen = max(len(x) for x in X)
X = pad_sequences(X, maxlen=maxlen, padding="pre")
y = np.array(y)

vocab_size = len(char2idx) + 1

model = Sequential([
    Embedding(vocab_size, EMBED_DIM, input_length=maxlen),
    SimpleRNN(RNN_UNITS),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early],
    verbose=1
)

model.save(MODEL_FILE)

print("\\nTraining Complete!")
print(f"Model saved as: {MODEL_FILE}")
print(f"Tokenizer saved as: {TOKENIZER_FILE}")
print(f"Vocabulary Size: {vocab_size}")
print(f"Training Samples: {len(X)}")
