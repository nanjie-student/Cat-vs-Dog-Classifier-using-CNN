# streamlit_app.py
import streamlit as st # type: ignore
from PIL import Image # type: ignore
import torch
from torchvision import transforms # type: ignore
from model import SimpleCNN

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
model.eval()

# 预测函数
def predict_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ["Cat", "Dog"]
        return class_names[predicted.item()]

# 网页界面
st.title("🐾 Cat vs Dog Classifier")
st.markdown("Upload an image, and the model will predict whether it's a **Cat** or a **Dog**!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    result = predict_image(img)
    st.success(f"Prediction: **{result}**")

