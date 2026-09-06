"""
Alphabet Recognition using an Artificial Neural Network (ANN) - Streamlit App
------------------------------------------------------------------------------
This version uses a from-scratch NumPy ANN (no TensorFlow/Keras, no scikit-learn)
so it works on any modern Python version, including brand-new releases like 3.14
where heavier ML frameworks may not have published wheels yet.

This app:
  1. Generates a synthetic training dataset of the letters A-Z by rendering
     them with different fonts, rotations, shifts, and noise (data augmentation).
  2. Builds and trains a simple feedforward ANN (manual forward/backward pass,
     Adam optimizer) implemented purely in NumPy.
  3. Lets you upload an image of a hand-drawn letter (e.g. drawn in Paint)
     and predicts which letter it is.

Run with:
    streamlit run alphabet_ann_streamlit.py

Required packages (all lightweight, Python-3.14-friendly, no fragile
third-party UI components):
    pip install streamlit numpy pandas pillow
"""

import os
import random
import string

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
LETTERS = list(string.ascii_uppercase)   # A-Z
IMG_SIZE = 28                            # final image size fed to the ANN
RENDER_SIZE = 64                         # larger canvas used before downscaling
N_INPUTS = IMG_SIZE * IMG_SIZE           # 784 flattened pixels

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
    return [p for p in candidates if os.path.exists(p)]


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
        fonts.append(ImageFont.load_default())
    return fonts


FONTS = load_fonts()


# --------------------------------------------------------------------------------
# Crop the drawn content to its bounding box, pad it back to a centered square,
# then resize to IMG_SIZE. This is the same trick MNIST-style pipelines use so
# that position/scale differences between training and real drawings stop
# mattering as much. Used for BOTH synthetic training data and real input
# (canvas / uploaded image) so the two stay consistent.
# --------------------------------------------------------------------------------
def crop_and_center(arr, size=IMG_SIZE, pad_ratio=0.25, threshold=0.15):
    # Auto-stretch contrast first: this boosts faint/light pen strokes to
    # near-full intensity and pushes background to near-zero, regardless of
    # how light, thin, or low-contrast the original drawing was. Safe no-op
    # on already-high-contrast synthetic training images.
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-6:
        arr = (arr - lo) / (hi - lo)

    ys, xs = np.where(arr > threshold)
    if len(xs) == 0 or len(ys) == 0:
        img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = arr[y0:y1 + 1, x0:x1 + 1]

    # Push near-background pixels inside the crop fully to 0, and re-stretch,
    # so the extracted letter reads as solid/bold rather than washed out.
    cropped = np.where(cropped > threshold, cropped, 0.0)
    c_hi = cropped.max()
    if c_hi > 1e-6:
        cropped = np.clip(cropped / c_hi, 0.0, 1.0)

    h, w = cropped.shape
    side = max(h, w)
    pad = max(2, int(side * pad_ratio))
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=arr.dtype)
    y_off = (canvas_side - h) // 2
    x_off = (canvas_side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = cropped

    img = Image.fromarray((canvas * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


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

    angle = random.uniform(-angle_range, angle_range)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    # Randomly thicken or thin the stroke so the model sees a range of pen/brush
    # widths, not just one uniform font weight. This mirrors real handwriting
    # variability much better.
    r = random.random()
    if r < 0.35:
        img = img.filter(ImageFilter.MaxFilter(3))   # thicker strokes
    elif r < 0.55:
        img = img.filter(ImageFilter.MinFilter(3))   # thinner strokes

    arr_full = np.array(img, dtype=np.float32) / 255.0
    arr = crop_and_center(arr_full, size=IMG_SIZE)

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
# Manual stratified train/test split (replaces sklearn.train_test_split)
# --------------------------------------------------------------------------------
def train_test_split_manual(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = rng.permutation(np.array(train_idx))
    test_idx = rng.permutation(np.array(test_idx))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# --------------------------------------------------------------------------------
# Pure NumPy Artificial Neural Network (feedforward, Adam optimizer)
# --------------------------------------------------------------------------------
class NumpyANN:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(2.0 / fan_in)  # He initialization (good for ReLU)
            self.weights.append(rng.normal(0, limit, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

        # Adam optimizer state
        self.mW = [np.zeros_like(w) for w in self.weights]
        self.vW = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []
        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            zs.append(Z)
            activations.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = self.softmax(Z)
        zs.append(Z)
        activations.append(A)
        return activations, zs

    def compute_loss_acc(self, X, y):
        activations, _ = self.forward(X)
        probs = activations[-1]
        n = X.shape[0]
        logp = -np.log(probs[np.arange(n), y] + 1e-9)
        loss = float(np.mean(logp))
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        return loss, acc

    def train_batch(self, X, y, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                    label_smoothing=0.1, weight_decay=1e-4):
        n = X.shape[0]
        num_classes = self.layer_sizes[-1]
        activations, zs = self.forward(X)
        probs = activations[-1]

        # Label smoothing: instead of a hard 0/1 target, spread a little
        # probability mass across all classes. This caps how extremely
        # confident the softmax is allowed to become, which reduces the
        # "100% confidence, still wrong" failure mode.
        Y_onehot = np.zeros_like(probs)
        Y_onehot[np.arange(n), y] = 1
        Y_smooth = Y_onehot * (1 - label_smoothing) + label_smoothing / num_classes

        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        dZ = (probs - Y_smooth) / n
        grads_W[-1] = activations[-2].T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.weights[-1].T
        for i in reversed(range(len(self.weights) - 1)):
            dZ = dA_prev * self.relu_deriv(zs[i])
            grads_W[i] = activations[i].T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ @ self.weights[i].T

        self.t += 1
        for i in range(len(self.weights)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (grads_W[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            # Decoupled weight decay (L2 regularization) alongside Adam's
            # update — keeps weights smaller, which reduces overfitting to
            # the narrow synthetic-font training distribution.
            self.weights[i] -= lr * (mW_hat / (np.sqrt(vW_hat) + eps) + weight_decay * self.weights[i])

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grads_b[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def fit_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, seed=0):
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    for _ in range(epochs):
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        for start in range(0, n, batch_size):
            end = start + batch_size
            model.train_batch(X_shuf[start:end], y_shuf[start:end], lr=lr)

        train_loss, train_acc = model.compute_loss_acc(X_train, y_train)
        val_loss, val_acc = model.compute_loss_acc(X_val, y_val)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history



# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("🔤 Alphabet Recognition with an ANN")
st.write(
    "This app trains a simple Artificial Neural Network (pure NumPy, no "
    "TensorFlow required) on synthetically generated A-Z letter images, then "
    "lets you draw or upload a letter and predicts it."
)

if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None

# ---- Sidebar: training controls ----
st.sidebar.header("⚙️ Training Settings")
samples_per_class = st.sidebar.slider("Samples per letter", 50, 800, 350, step=50)
epochs = st.sidebar.slider("Epochs", 5, 60, 30)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.003
)
train_button = st.sidebar.button("🚀 Train Model")

if train_button:
    with st.spinner("Generating dataset..."):
        X, y = build_dataset(samples_per_class=samples_per_class)
        X_flat = X.reshape(len(X), -1)  # flatten 28x28 -> 784
        X_train, X_test, y_train, y_test = train_test_split_manual(X_flat, y, test_size=0.2)

    with st.spinner("Training ANN..."):
        model = NumpyANN(layer_sizes=[N_INPUTS, 256, 128, len(LETTERS)])
        history = fit_model(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=64, lr=learning_rate,
        )

    test_loss, test_acc = model.compute_loss_acc(X_test, y_test)
    st.session_state.model = model
    st.session_state.history = history
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

# ---- Input: upload an image of a hand-drawn letter (e.g. from Paint) ----
st.subheader("✏️ Upload a Letter")

final_arr = None  # will hold the final 28x28 normalized array to predict on

if st.session_state.model is None:
    st.info("Train the model first using the sidebar button.")

else:
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
        img = Image.open(uploaded_file).convert("L")
        arr_full = np.array(img, dtype=np.float32) / 255.0
        if invert_colors:
            arr_full = 1.0 - arr_full

        # Slightly bolden thin/faint strokes so they better resemble the
        # bolder strokes used during training.
        boosted = Image.fromarray((arr_full * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
        arr_full = np.array(boosted, dtype=np.float32) / 255.0

        final_arr = crop_and_center(arr_full, size=IMG_SIZE)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded image (original)", width=150)
        with col2:
            preview_img = Image.fromarray((final_arr * 255).astype(np.uint8)).resize(
                (150, 150), Image.NEAREST
            )
            st.image(preview_img, caption="What the model actually sees (28x28, enlarged)")

        st.caption(
            "⚠️ If the right-hand preview looks blank, inverted, or doesn't "
            "resemble your letter, toggle 'Invert colors' above — that's "
            "almost always the cause of wrong predictions."
        )

    predict_button = st.button("🔍 Predict Letter")

    if predict_button and final_arr is not None:
        pred_input = final_arr.reshape(1, -1)  # flatten to (1, 784)
        probs = st.session_state.model.predict_proba(pred_input)[0]

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
    "so accuracy on hand-drawn letters is a fun demo rather than production-grade. "
    "The ANN itself is implemented in pure NumPy, so it has no TensorFlow "
    "dependency and works on any Python version, including 3.14."
)
