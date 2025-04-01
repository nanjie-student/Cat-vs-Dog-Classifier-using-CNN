import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    # def __init__(self):
    #     super(SimpleCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # 输入3通道，输出32通道
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    #     self.fc1 = nn.Linear(64 * 53 * 53, 128)  # 注意这里的尺寸计算
    #     self.fc2 = nn.Linear(128, 2)  # 二分类（猫 or 狗）

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))   # -> [batch, 32, 109, 109]
    #     x = self.pool(F.relu(self.conv2(x)))   # -> [batch, 64, 53, 53]
    #     #手动.view(-1, N)，灵活性差，会造成模型脆弱，所以我们选择动态推导flatten size
    #     x = x.view(-1, self.flatten_size)           # Flatten
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 使用 dummy 数据计算 Flatten 后尺寸
        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_output = self._forward_conv(dummy_input)
        self.flatten_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.flatten_size)  # 这里动态展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
