import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("MNIST Digit Classifier")
st.write("Upload a 28×28 grayscale image of a handwritten digit and the model will predict the number.")

model = tf.keras.models.load_model("mnist_cnn.h5")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L").resize((28,28))
    img_array = np.array(img).reshape(1,28,28,1) / 255.0

    st.image(img, caption="Uploaded Image", width=150)

    pred = np.argmax(model.predict(img_array))

    st.subheader(f"Prediction: **{pred}**")
