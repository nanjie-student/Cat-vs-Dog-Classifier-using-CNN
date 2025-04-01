import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        dummy_input = torch.zeros(1, 3, 64, 64)
        dummy_output = self._forward_conv(dummy_input)
        self.flatten_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.fc2 = nn.Linear(32, 2)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
