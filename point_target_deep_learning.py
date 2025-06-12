# -*- coding: utf-8 -*-
"""
一个简单的点目标识别深度学习示例。
使用PyTorch构建一个小型卷积神经网络，
用于区分包含单个点目标的图像和空白图像。
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class PointTargetDataset(Dataset):
    """生成随机点目标数据集"""

    def __init__(self, size=1000, img_size=16):
        self.size = size
        self.img_size = img_size
        self.data = []
        self.labels = []
        for _ in range(size):
            img = torch.zeros(1, img_size, img_size)
            if torch.rand(1).item() > 0.5:
                # 有点目标
                x = torch.randint(0, img_size, (1,)).item()
                y = torch.randint(0, img_size, (1,)).item()
                img[0, y, x] = 1.0
                self.labels.append(1)
            else:
                # 无目标
                self.labels.append(0)
            self.data.append(img)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleCNN(nn.Module):
    """非常简单的卷积网络"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 16 * 16, 2)
        )

    def forward(self, x):
        return self.model(x)


def train():
    dataset = PointTargetDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        for imgs, labels in loader:
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), "point_target_model.pth")


if __name__ == "__main__":
    train()
