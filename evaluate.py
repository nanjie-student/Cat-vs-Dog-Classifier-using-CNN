import torch
from model import SimpleCNN
from dataset import get_cat_dog_dataloader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
model.eval()

# 2. 加载数据
dataloader = get_cat_dog_dataloader(batch_size=32) #每次预测 32 张图片，加速处理

# 3. 预测并收集标签
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 4. 打印指标
acc = accuracy_score(y_true, y_pred)
print(f" Accuracy: {acc * 100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Cat", "Dog"]))

# 5. 画混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
