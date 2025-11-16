
import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("Anime Detector (Simple Deep Learning Demo)")

interpreter = tflite.Interpreter(model_path="anime_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img):
    img = img.resize((32,32))
    arr = np.array(img).astype('float32')/255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return pred

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)
    pred = predict(img)
    st.write("Prediction:", "Anime" if pred>0.5 else "Not Anime")
