import streamlit as st
import time
import random
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import ssl


# ----------------------------- #
# 1) Define AttentionLayer
# ----------------------------- #
class AttentionLayer(layers.Layer):
    def init(self, **kwargs):
        super(AttentionLayer, self).init(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.matmul(inputs, self.W)
        e = tf.squeeze(e, -1)
        e = e + tf.squeeze(self.b, -1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ----------------------------- #
# 2) Cache-loading Model & Tokenizer
# ----------------------------- #
@st.cache_resource
def load_model():
    model_path = "models/bilstm_att_model.h5"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer_path = "models/tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# ----------------------------- #
# 3) Text Preprocessing
# ----------------------------- #


def advanced_text_cleaning(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ----------------------------- #
# 4) Prediction Function
# ----------------------------- #
def predict_sentiment(text, model, tokenizer, max_length=50):
    cleaned_text = advanced_text_cleaning(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
    probabilities = model.predict(padded)
    label_idx = np.argmax(probabilities, axis=1)[0]
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return mapping[label_idx]

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Call this before accessing any NLTK resources
if not download_nltk_data():
    st.error("Failed to download required NLTK data. Please try again.")
    st.stop()

# Now initialize stopwords and lemmatizer
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Base styles */
    body {
        background-color: #000000 !important;
        margin: 0;
        padding: 0;
    }

    .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: #000000;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.speech-bubble-loader {
    position: relative;
    width: 120px;
    height: 80px;
    background: transparent;
    border-radius: 20px;
}

.bubble-part {
    position: absolute;
    border-radius: 50%;
    opacity: 0;
}

.bubble-part:nth-child(1) {
    width: 40px;
    height: 40px;
    background: #FF6B6B;  /* Coral Red */
    top: 0;
    left: 0;
    animation: bubble-animation 2s ease infinite;
}

.bubble-part:nth-child(2) {
    width: 35px;
    height: 35px;
    background: #FFDF22;  /* Turquoise */
    top: 0;
    right: 0;
    animation: bubble-animation 2s ease infinite 0.5s;
}

.bubble-part:nth-child(3) {
    width: 30px;
    height: 30px;
    background: #45B7D1;  /* Sky Blue */
    bottom: 0;
    right: 20px;
    animation: bubble-animation 2s ease infinite 1s;
}

.bubble-part:nth-child(4) {
    width: 25px;
    height: 25px;
    background: #228B22;  /* Mint Green */
    bottom: -10px;
    left: 20px;
    transform: rotate(45deg);
    animation: bubble-tail-animation 2s ease infinite 1.5s;
}

@keyframes bubble-animation {
    0% {
        transform: scale(0) translateY(20px);
        opacity: 0;
    }
    30% {
        transform: scale(1) translateY(0);
        opacity: 1;
    }
    70% {
        transform: scale(1) translateY(0);
        opacity: 1;
    }
    100% {
        transform: scale(0) translateY(-20px);
        opacity: 0;
    }
}

@keyframes bubble-tail-animation {
    0% {
        transform: scale(0) rotate(45deg) translateY(20px);
        opacity: 0;
    }
    30% {
        transform: scale(1) rotate(45deg) translateY(0);
        opacity: 1;
    }
    70% {
        transform: scale(1) rotate(45deg) translateY(0);
        opacity: 1;
    }
    100% {
        transform: scale(0) rotate(45deg) translateY(-20px);
        opacity: 0;
    }
}
.emoji-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 1000;
        overflow: hidden;
    }

    .floating-emoji {
        position: absolute;
        font-size: 24px;
        opacity: 0;
        bottom: -50px;  /* Start below the viewport */
        animation: float-up 3s ease-out forwards;
        transform-origin: center center;
    }

    @keyframes float-up {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        100% {
            transform: translateY(-120vh) rotate(359deg);  /* Move beyond viewport */
            opacity: 0;
        }
    }

    /* Add this new class for staggered animations */
    .staggered {
        animation-play-state: paused;
    }
    

    @keyframes pop-and-float {
        0% {
            transform: scale(0) translateY(0);
            opacity: 0;
        }
        10% {
            transform: scale(1.2) translateY(0);
            opacity: 1;
        }
        20% {
            transform: scale(1) translateY(0);
            opacity: 1;
        }
        100% {
            transform: scale(1) translateY(-100vh);
            opacity: 0;
        }
    }


    @keyframes pulse-glow {
        from { box-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
        to { box-shadow: 0 0 20px #00ff00, 0 0 30px #00ff00, 0 0 40px #00ff00; }
    }

    /* "Let Us Talk" fullscreen text */
    .fullscreen-text {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #000000;
        font-size: min(10vw, 72px);
        font-weight: bold;
        color: #00ff00;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
        animation: fade-in-glow 10s ease-in-out forwards;
        white-space: nowrap;
    }

    @keyframes fade-in-glow {
        0% { opacity: 0; text-shadow: 0 0 0 #00ff00; }
        50% { opacity: 1; text-shadow: 0 0 30px #00ff00, 0 0 60px #00ff00; }
        100% { opacity: 0; text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00; }
    }

    /* Title and text input styling */
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #00ff00;
        text-align: center;
        margin-top: 20px;
        text-shadow: 0 0 10px #00ff00;
    }

    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
    }

    /* Sentiment output styles */
    .sentiment-output {
        position: relative;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        animation: slide-up 0.5s ease-out;
    }

    .sentiment-positive {
        color: #ffff00 !important;
        text-shadow: 0 0 10px #ffff00, 0 0 20px #ffff00;
        animation: glow-positive 2s infinite;
    }

    .sentiment-neutral {
        color: #0088ff !important;
        text-shadow: 0 0 10px #0088ff, 0 0 20px #0088ff;
        animation: glow-neutral 2s infinite;
    }

    .sentiment-negative {
        color: #ff0000 !important;
        text-shadow: 0 0 10px #ff0000, 0 0 20px #ff0000;
        animation: glow-negative 2s infinite;
    }

    /* Emoji rain animation */
    .emoji-rain {
        position: fixed;
        bottom: -50px;
        width: 100%;
        height: 50px;
        text-align: center;
        animation: rain-up 1.5s ease-out forwards;
        z-index: 1000;
    }

    .emoji {
        display: inline-block;
        margin: 0 5px;
        font-size: 24px;
        animation: float-up 2s ease-out infinite;
    }

    @keyframes rain-up {
        0% { transform: translateY(50px); opacity: 0; }
        100% { transform: translateY(-100vh); opacity: 1; }
    }

    @keyframes float-up {
        0% { transform: translateY(0) rotate(0deg); }
        100% { transform: translateY(-20px) rotate(359deg); }
    }

    /* Dark theme overrides */
    .stButton button {
        background-color: #1a1a1a !important;
        color: #00ff00 !important;
        border: 2px solid #00ff00 !important;
        text-shadow: 0 0 5px #00ff00;
        box-shadow: 0 0 10px #00ff00;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #00ff00 !important;
        color: #000000 !important;
        box-shadow: 0 0 20px #00ff00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Step 1: Loading screen (helix animation)
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    st.markdown(
        """
        <div class="loading-screen">
            <div class="speech-bubble-loader">
                <div class="bubble-part"></div>
                <div class="bubble-part"></div>
                <div class="bubble-part"></div>
                <div class="bubble-part"></div>
                
 
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(4)
    st.session_state.model_loaded = True
    st.rerun()

# Step 2: "Let Us Talk" fullscreen display
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if not st.session_state.intro_shown:
    st.markdown(
        """
        <div class="fullscreen-text">Let Us Talk...</div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(10)  # Increased to 10 seconds
    st.session_state.intro_shown = True
    st.rerun()

# Step 3: Main app layout
st.markdown('<h1 class="title">Let Us Talk...</h1>', unsafe_allow_html=True)

def generate_emoji_style():
    x = random.randint(0, 90)  # Limit to 90 to prevent right-side cutoff
    delay = random.uniform(0, 1.5)  # Increased delay variation
    return f'''
        left: {x}vw;
        animation-delay: {delay}s;
        animation-duration: {random.uniform(2.5, 3.5)}s;
    '''

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

# Main app logic
st.title("Sentiment Analyzer")
st.markdown("### Enter your text below to analyze sentiment:")

user_input = st.text_area("Input Text", placeholder="Type something...", max_chars=500)
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input, model, tokenizer)
        emoji_map = {
            "Positive": ["😊", "😄", "😎", "🎉", "🎊", "✨", "🌟", "👍", "💖", "🔥", "🥳", "🤩", "🌈", "🫶", "🥰"],
            "Neutral": ["😐", "🤔", "🤷", "💭", "😶", "😑", "📘", "⚖", "🔵", "🌓", "⭐", "💫", "🌍", "🛋"],
            "Negative": ["😞", "😢", "😡", "💔", "🌧", "🌩", "⛈", "😠", "👎", "😭", "😨", "⚡", "😔", "🙁", "😿"]
        }

        # Generate unique container ID for this analysis
        container_id = f"emoji-container-{random.randint(1000, 9999)}"
        
        # Create emojis with staggered animation
        emojis = ""
        for i in range(15):
            emoji = random.choice(emoji_map[sentiment])
            style = generate_emoji_style()
            emojis += f'<div class="floating-emoji" style="{style}">{emoji}</div>'
        
        st.markdown(
            f'''
            <div class="sentiment-output">
                <p class="sentiment-{sentiment.lower()}">
                    <strong>Sentiment Found:</strong> {sentiment} {random.choice(emoji_map[sentiment])}
                </p>
            </div>
            <div id="{container_id}" class="emoji-container">
                {emojis}
            </div>
            <script>
                // Force animation restart by removing and re-adding the container
                setTimeout(() => {{
                    const container = document.getElementById("{container_id}");
                    const parent = container.parentNode;
                    const newContainer = container.cloneNode(true);
                    parent.removeChild(container);
                    parent.appendChild(newContainer);
                }}, 100);
            </script>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text!")