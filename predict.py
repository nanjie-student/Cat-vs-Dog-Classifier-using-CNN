import argparse
import torch
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from model import SimpleCNN

# 1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型结构
model = SimpleCNN().to(device)

# 3. 加载训练好的参数（先确保你有这个文件）
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
model.eval()

# 4. 加载并预处理图片
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 5. 模型预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ["Cat", "Dog"]
        prediction = class_names[predicted.item()]
        return prediction

# 主函数入口，支持默认路径 + 命令行传参
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat vs Dog Classifier")
    parser.add_argument("--image", type=str, help="Path to image (optional)")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
        print(f"Using custom image path: {image_path}")
    else:
        image_path = "test_cat.jpg"  # 默认测试图片
        print(f"Using default image: {image_path}")

    result = predict_image(image_path)
    print(f"The model predicts: {result}")