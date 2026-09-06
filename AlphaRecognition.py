"""
Alphabet Recognition using an Artificial Neural Network (ANN) - Streamlit App
------------------------------------------------------------------------------
This app:
  1. Generates a synthetic training dataset of the letters A-Z by rendering
     them with different fonts, rotations, shifts, and noise (data augmentation).
  2. Builds and trains a simple ANN (Dense layers) using TensorFlow/Keras.
  3. Lets the user draw a letter on a canvas and predicts which letter it is.

Run with:
    streamlit run alphabet_ann_streamlit.py

Required packages:
    pip install streamlit tensorflow pillow numpy pandas scikit-learn streamlit-drawable-canvas
"""

import os
import random
import string

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
LETTERS = list(string.ascii_uppercase)   # A-Z
IMG_SIZE = 28                            # final image size fed to the ANN
RENDER_SIZE = 64                         # larger canvas used before downscaling

st.set_page_config(page_title="Alphabet ANN Classifier", page_icon="🔤", layout="centered")


# --------------------------------------------------------------------------------
# Helper: find available TrueType fonts on the system (falls back to default)
# --------------------------------------------------------------------------------
def get_font_paths():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    found = [p for p in candidates if os.path.exists(p)]
    return found


FONT_PATHS = get_font_paths()


def load_fonts(sizes=(36, 42, 48)):
    fonts = []
    if FONT_PATHS:
        for path in FONT_PATHS:
            for size in sizes:
                try:
                    fonts.append(ImageFont.truetype(path, size))
                except Exception:
                    pass
    if not fonts:
        # Fallback: PIL's built-in bitmap font (fixed size, still works)
        fonts.append(ImageFont.load_default())
    return fonts


FONTS = load_fonts()


# --------------------------------------------------------------------------------
# Render a single augmented letter image as a numpy array (IMG_SIZE x IMG_SIZE)
# --------------------------------------------------------------------------------
def render_letter(letter, font, angle_range=15, shift_range=4, noise_std=0.03):
    img = Image.new("L", (RENDER_SIZE, RENDER_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    try:
        bbox = draw.textbbox((0, 0), letter, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (RENDER_SIZE - w) / 2 - bbox[0], (RENDER_SIZE - h) / 2 - bbox[1]
    except AttributeError:
        w, h = draw.textsize(letter, font=font)
        x, y = (RENDER_SIZE - w) / 2, (RENDER_SIZE - h) / 2

    draw.text((x, y), letter, fill=255, font=font)

    # Random rotation
    angle = random.uniform(-angle_range, angle_range)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # Random shift
    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    # Downscale to final size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0

    # Add slight gaussian noise
    arr = arr + np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# --------------------------------------------------------------------------------
# Build the synthetic dataset (cached so it isn't regenerated every rerun)
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_dataset(samples_per_class=150, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    X, y = [], []
    for idx, letter in enumerate(LETTERS):
        for _ in range(samples_per_class):
            font = random.choice(FONTS)
            arr = render_letter(letter, font)
            X.append(arr)
            y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# --------------------------------------------------------------------------------
# Build the ANN model
# --------------------------------------------------------------------------------
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(LETTERS), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------------
# Preprocess a user-drawn canvas image into the model's expected input format
# --------------------------------------------------------------------------------
def preprocess_canvas_image(image_data):
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("🔤 Alphabet Recognition with an ANN")
st.write(
    "This app trains a simple Artificial Neural Network on synthetically "
    "generated A-Z letter images, then lets you draw a letter and predicts it."
)

if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None

# ---- Sidebar: training controls ----
st.sidebar.header("⚙️ Training Settings")
samples_per_class = st.sidebar.slider("Samples per letter", 50, 500, 150, step=50)
epochs = st.sidebar.slider("Epochs", 5, 50, 15)
train_button = st.sidebar.button("🚀 Train Model")

if train_button:
    with st.spinner("Generating dataset..."):
        X, y = build_dataset(samples_per_class=samples_per_class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    with st.spinner("Training ANN..."):
        model = build_model()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=64,
            verbose=0,
        )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    st.session_state.model = model
    st.session_state.history = history.history
    st.success(f"Training complete! Test accuracy: {test_acc:.2%}")

# ---- Show training curves if available ----
if st.session_state.history is not None:
    st.subheader("📈 Training History")
    hist_df = pd.DataFrame({
        "accuracy": st.session_state.history["accuracy"],
        "val_accuracy": st.session_state.history["val_accuracy"],
    })
    st.line_chart(hist_df)

st.divider()

# ---- Input method: draw on canvas OR upload an image (e.g. from Paint) ----
st.subheader("✏️ Provide a Letter")

input_mode = st.radio("Choose input method:", ["Draw on canvas", "Upload image (e.g. from Paint)"])

final_arr = None  # will hold the final 28x28 normalized array to predict on

if st.session_state.model is None:
    st.info("Train the model first using the sidebar button.")

else:
    if input_mode == "Draw on canvas":
        if not CANVAS_AVAILABLE:
            st.warning(
                "The `streamlit-drawable-canvas` package is not installed. "
                "Install it with:  pip install streamlit-drawable-canvas"
            )
        else:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=15,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=200,
                height=200,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None:
                final_arr = preprocess_canvas_image(canvas_result.image_data)

    else:  # Upload image
        st.markdown(
            "**Tips for drawing in Paint:**\n"
            "- Use a square canvas (e.g. resize image to 200x200 pixels).\n"
            "- Draw the letter big and centered, filling most of the canvas.\n"
            "- A thick brush works better than a thin one.\n"
            "- Black letter on white background is fine — just leave "
            "'Invert colors' checked below (that's Paint's default)."
        )

        uploaded_file = st.file_uploader(
            "Upload your letter image", type=["png", "jpg", "jpeg", "bmp"]
        )
        invert_colors = st.checkbox(
            "Invert colors (use this if your image is a dark letter on a light background)",
            value=True,
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("L")  # grayscale
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            if invert_colors:
                arr = 1.0 - arr
            final_arr = arr

            st.image(img, caption="Uploaded image (original)", width=150)

    predict_button = st.button("🔍 Predict Letter")

    if predict_button and final_arr is not None:
        pred_input = np.expand_dims(final_arr, axis=0)
        probs = st.session_state.model.predict(pred_input, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_letter = LETTERS[pred_idx]
        confidence = probs[pred_idx]

        st.markdown(f"### Predicted Letter: **{pred_letter}** ({confidence:.1%} confidence)")

        prob_df = pd.DataFrame({"Probability": probs}, index=LETTERS)
        st.bar_chart(prob_df)
    elif predict_button and final_arr is None:
        st.warning("Please draw or upload a letter first.")

st.divider()
st.caption(
    "Note: this model is trained on rendered fonts, not real handwriting, "
    "so accuracy on hand-drawn letters is a fun demo rather than production-grade."
)
