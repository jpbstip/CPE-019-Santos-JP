
import streamlit as st
import torch
from model import SmallAnimeCNN
from PIL import Image
import torchvision.transforms as T

st.title("Anime Mini Model Demo")

model = SmallAnimeCNN()
model.load_state_dict(torch.load('anime_model.pth', map_location='cpu'))
model.eval()

uploaded = st.file_uploader("Upload anime-style image", type=['png','jpg','jpeg'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img)
    transform = T.Compose([T.Resize((32,32)), T.ToTensor()])
    x = transform(img).unsqueeze(0)
    preds = model(x)
    label = torch.argmax(preds,1).item()
    st.write(f"Prediction: {label}")
