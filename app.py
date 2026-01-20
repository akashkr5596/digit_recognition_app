import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (28x28)")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    st.success(f"🎯 Predicted Digit: {result}")
