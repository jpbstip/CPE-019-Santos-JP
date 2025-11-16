
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Cat Detector")

st.write("""
This app demonstrates a simple deep learning project using a convolutional neural network. 
The model was trained on the cat class from CIFAR-10 and converted to TensorFlow Lite for deployment 
on Streamlit Cloud.

Upload an image, and the app will predict whether it contains a cat.

- Images are automatically resized to **32×32 pixels** (RGB) before prediction.  
- Large or high-resolution images will be downscaled, which may reduce detail.  
- For best results, use clear images where the cat is visible and occupies most of the frame.

""")


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="anime_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img):
    img = img.resize((32, 32))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
    return pred

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)
    pred = predict(img)
    st.write("Prediction:", "Anime" if pred > 0.5 else "Not Anime")
