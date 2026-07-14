import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Name Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
background:
linear-gradient(135deg,#0f172a,#111827,#1e3a8a);
background-attachment:fixed;
color:white;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
}

.hero{

background:rgba(255,255,255,.08);

backdrop-filter:blur(18px);

border-radius:25px;

padding:35px;

box-shadow:0 10px 40px rgba(0,0,0,.45);

border:1px solid rgba(255,255,255,.15);

margin-bottom:25px;

}

.title{

font-size:56px;

font-weight:800;

text-align:center;

background:linear-gradient(90deg,#60a5fa,#22d3ee,#a855f7);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}

.subtitle{

text-align:center;

font-size:18px;

color:#cbd5e1;

margin-top:10px;

margin-bottom:15px;

}

.card{

background:rgba(255,255,255,.08);

backdrop-filter:blur(20px);

padding:20px;

border-radius:18px;

border:1px solid rgba(255,255,255,.12);

margin-bottom:18px;

box-shadow:0 6px 25px rgba(0,0,0,.3);

}

.result{

background:#0f172a;

padding:15px;

margin:10px 0;

border-radius:12px;

font-size:22px;

font-weight:bold;

border-left:5px solid cyan;

transition:.3s;

}

.result:hover{

transform:translateX(6px);

background:#111827;

}

div.stButton>button{

width:100%;

height:55px;

font-size:18px;

font-weight:bold;

background:linear-gradient(90deg,#2563eb,#7c3aed);

color:white;

border:none;

border-radius:14px;

}

div.stButton>button:hover{

transform:scale(1.02);

}

.metric{

background:#111827;

padding:20px;

border-radius:15px;

text-align:center;

}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO
# -------------------------------------------------

st.markdown("""
<div class="hero">

<div class="title">

🤖 AI NAME GENERATOR

</div>

<div class="subtitle">

Generate Human-Like Names using Deep Learning (SimpleRNN)

</div>

</div>

""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_resources():

    model = load_model("name_generator.keras")

    with open("char_tokenizer.pkl","rb") as f:

        tokenizer = pickle.load(f)

    return model, tokenizer


try:

    model, tokenizer = load_resources()

except Exception as e:

    st.error(e)

    st.stop()

char2idx = tokenizer["char2idx"]

idx2char = tokenizer["idx2char"]

maxlen = tokenizer["maxlen"]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.title("⚙ Generator Settings")

    prefix = st.text_input("Starting Letters","")

    temperature = st.slider(

        "Creativity",

        0.2,

        1.5,

        0.8,

        0.1

    )

    total = st.slider(

        "Number of Names",

        1,

        20,

        10

    )

    st.markdown("---")

    st.info("""

Lower Temperature

→ More realistic names



Higher Temperature

→ More creative names

""")

    st.markdown("---")

    st.success("Model : SimpleRNN")


# -------------------------------------------------
# RANDOM PREFIX
# -------------------------------------------------

RANDOM_PREFIXES = [
    "A","Ar","An","Aa","Ad","Ak",
    "Vi","Kr","Jo","Ka","Ra","Sa",
    "Ni","Ri","De","Ma","Sh","Pr"
]

if st.sidebar.button("🎲 Random Prefix"):
    prefix = random.choice(RANDOM_PREFIXES)

# -------------------------------------------------
# NAME GENERATOR
# -------------------------------------------------

def generate_name(seed="", temperature=0.8, max_chars=30):

    text = "^" + seed.lower()

    while len(text) < max_chars:

        sequence = [char2idx.get(c, 0) for c in text]

        sequence = pad_sequences(
            [sequence],
            maxlen=maxlen-1,
            padding="pre"
        )

        prediction = model.predict(sequence, verbose=0)[0]

        prediction = np.log(prediction + 1e-9) / temperature

        prediction = np.exp(prediction)

        prediction = prediction / np.sum(prediction)

        next_index = np.random.choice(
            len(prediction),
            p=prediction
        )

        if next_index == 0:
            continue

        character = idx2char[next_index]

        if character == "$":
            break

        text += character

    return text.replace("^","").strip().title()

# -------------------------------------------------
# GENERATOR PANEL
# -------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

left, right = st.columns([3,1])

with left:

    st.subheader("✨ Generate AI Names")

    st.write(
        "Generate unique names trained from your dataset."
    )

with right:

    generate = st.button(
        "🚀 Generate"
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# GENERATION
# -------------------------------------------------

generated_names = []

if generate:

    progress = st.progress(0)

    status = st.empty()

    for i in range(total):

        status.info(
            f"Generating {i+1} of {total}..."
        )

        progress.progress(
            (i+1)/total
        )

        generated_names.append(
            generate_name(
                prefix,
                temperature
            )
        )

        time.sleep(0.08)

    progress.empty()

    status.success("Generation Complete!")

    generated_names = list(
        dict.fromkeys(generated_names)
    )
    # -------------------------------------------------
# RESULTS
# -------------------------------------------------

if generated_names:

    st.markdown("## ✨ Generated Names")

    download_text = ""

    for index, name in enumerate(generated_names, start=1):

        st.markdown(
            f"""
            <div class="result">
                <b>{index}.</b> ✨ {name}
            </div>
            """,
            unsafe_allow_html=True
        )

        download_text += name + "\n"

    st.markdown("### 📋 Copy Generated Names")

    st.text_area(
        "",
        download_text,
        height=220
    )

    st.download_button(
        label="📥 Download Names",
        data=download_text,
        file_name="generated_names.txt",
        mime="text/plain",
        use_container_width=True
    )

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

st.markdown("---")

st.subheader("📊 Model Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Model",
        "SimpleRNN"
    )

with c2:
    st.metric(
        "Vocabulary",
        len(char2idx)
    )

with c3:
    st.metric(
        "Max Length",
        maxlen
    )

with c4:
    st.metric(
        "Generated",
        len(generated_names)
    )

# -------------------------------------------------
# ABOUT
# -------------------------------------------------

with st.expander("ℹ About this Project"):

    st.write("""
This AI Name Generator uses a **Character-Level SimpleRNN**
trained on your names dataset.

Features:

• Character-Level Learning

• Deep Learning (TensorFlow)

• Temperature Sampling

• Random Name Generation

• Streamlit Interactive Dashboard

• Download Generated Names
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center;
                color:#94a3b8;
                padding:20px;">

    <h4>🤖 AI Name Generator</h4>

    <p>
    Built using TensorFlow • Streamlit • SimpleRNN
    </p>

    <p style="font-size:14px;">
    Modern Glassmorphism Interface
    </p>

    </div>
    """,
    unsafe_allow_html=True
)    
