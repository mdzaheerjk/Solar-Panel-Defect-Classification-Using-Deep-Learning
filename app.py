import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

# ------------------ Page Configuration ------------------ #
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

# ------------------ Title ------------------ #
st.title("☀️ Solar Panel Defect Classifier")
st.markdown(
    """
    Upload a **Solar Panel Image** to automatically detect defects using a
    fine-tuned **EfficientNet** model.
    """
)

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Models/effnet_finetune.h5")
    return model

with st.spinner("Loading AI Model..."):
    model = load_model()

# ------------------ Class Names ------------------ #
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered"
]

# ------------------ File Upload ------------------ #
uploaded_file = st.file_uploader(
    "📤 Upload a Solar Panel Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------ Prediction ------------------ #
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Center Image
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(image, width=300, caption="Uploaded Image")

    st.write("")

    if st.button("🔍 Analyze Panel", type="primary", use_container_width=True):

        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))

        with st.spinner("Analyzing the panel..."):

            predictions = model.predict(img_array, verbose=0)

            predicted_idx = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_idx]
            confidence = predictions[0][predicted_idx]

        # ------------------ Result ------------------ #
        st.success(f"### 🔍 Prediction: {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2%}")

        if predicted_class == "Clean":
            st.success("✅ The solar panel appears to be clean and operating normally.")
        else:
            st.warning(
                f"⚠️ Detected: **{predicted_class}**\n\nMaintenance or inspection is recommended."
            )

        st.divider()

        # ------------------ Top 3 Predictions ------------------ #
        st.subheader("🏆 Top 3 Predictions")

        top_indices = np.argsort(predictions[0])[-3:][::-1]

        medals = ["🥇", "🥈", "🥉"]

        for medal, idx in zip(medals, top_indices):
            st.write(f"{medal} **{CLASSES[idx]}** — {predictions[0][idx]:.2%}")

        st.divider()

        # ------------------ Probability Chart ------------------ #
        with st.expander("📊 View Class Probabilities"):

            for cls, prob in zip(CLASSES, predictions[0]):
                st.write(f"**{cls}**")
                st.progress(float(prob))
                st.caption(f"{prob:.2%}")