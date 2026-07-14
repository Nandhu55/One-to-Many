
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_FILE = "name_generator.keras"
TOKENIZER_FILE = "char_tokenizer.pkl"

st.set_page_config(page_title="AI Name Generator", page_icon="🧠", layout="centered")
st.title("🧠 One-to-Many SimpleRNN Name Generator")

@st.cache_resource
def load_resources():
    model = load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tok = pickle.load(f)
    return model, tok

try:
    model, tok = load_resources()
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

char2idx = tok["char2idx"]
idx2char = tok["idx2char"]
maxlen = tok["maxlen"]

def generate_name(seed="", temperature=0.8, max_chars=30):
    seed = "^" + seed.lower()
    generated = seed

    while len(generated) < max_chars:
        seq = [char2idx.get(c, 0) for c in generated]
        seq = pad_sequences([seq], maxlen=maxlen-1, padding="pre")

        preds = model.predict(seq, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_idx = np.random.choice(len(preds), p=preds)

        if next_idx == 0:
            continue

        ch = idx2char[next_idx]

        if ch == "$":
            break

        generated += ch

    return generated.replace("^", "").strip()

prefix = st.text_input("Starting letters (optional)", "")
temp = st.slider("Creativity (Temperature)", 0.2, 1.5, 0.8, 0.1)
count = st.slider("Number of Names", 1, 20, 5)

if st.button("Generate Names"):
    st.subheader("Generated Names")
    shown = set()
    while len(shown) < count:
        name = generate_name(prefix, temp).title()
        if name:
            shown.add(name)
    for n in sorted(shown):
        st.success(n)
