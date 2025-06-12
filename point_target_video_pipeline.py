# -*- coding: utf-8 -*-
"""
针对视频中多点目标的完整深度学习检测与跟踪示例。
该脚本生成合成训练数据，训练一个输出热点图的卷积网络，
随后给定视频文件对其中的点目标进行检测、跟踪并将结果以红框标记到视频中。

依赖: torch, numpy, opencv-python
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    import cv2  # 用于读取和写入视频
except ImportError:  # 在没有安装依赖的环境下允许导入失败
    cv2 = None


class PointFrameDataset(Dataset):
    """生成带噪声的随机点目标帧数据集"""

    def __init__(
        self,
        num_frames: int = 1000,
        img_size: int = 32,
        max_points: int = 3,
        noise_points: int = 5,
    ):
        self.img_size = img_size
        self.frames = []
        self.targets = []
        for _ in range(num_frames):
            frame = torch.zeros(1, img_size, img_size)
            target = torch.zeros(1, img_size, img_size)
            n = torch.randint(1, max_points + 1, (1,)).item()
            for _ in range(n):
                x = torch.randint(0, img_size, (1,)).item()
                y = torch.randint(0, img_size, (1,)).item()
                frame[0, y, x] = 1.0
                target[0, y, x] = 1.0
            # 添加噪声点但不计入目标
            noise_n = torch.randint(0, noise_points + 1, (1,)).item()
            for _ in range(noise_n):
                x = torch.randint(0, img_size, (1,)).item()
                y = torch.randint(0, img_size, (1,)).item()
                frame[0, y, x] = 1.0
            self.frames.append(frame)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        return self.frames[idx], self.targets[idx]


class HeatmapCNN(nn.Module):
    """输出热点图的简单卷积网络"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_detector(output_path: str = "point_detector.pth") -> nn.Module:
    """训练网络并保存模型"""
    dataset = PointFrameDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = HeatmapCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        for imgs, targets in loader:
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), output_path)
    return model


def load_model(model_path: str) -> nn.Module:
    model = HeatmapCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def detect_points(model: nn.Module, frame: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """对单帧图像进行点目标检测"""
    img = torch.tensor(frame / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        heat = torch.sigmoid(model(img))[0, 0]
    ys, xs = torch.where(heat > threshold)
    return [(int(x.item()), int(y.item())) for x, y in zip(xs, ys)]


def track_detections(
    detections: List[List[Tuple[int, int]]],
    max_dist: float = 2.0,
    max_missing: int = 2,
    min_length: int = 3,
) -> List[List[Tuple[int, int]]]:
    """根据逐帧检测结果进行简单的连续运动跟踪并滤除瞬时噪声"""

    class Track:
        def __init__(self, pos: Tuple[int, int]):
            self.points = [pos]
            self.last = pos
            self.missed = 0

    states: List[Track] = []

    for frame_points in detections:
        used = [False] * len(frame_points)

        for state in states:
            best_i = None
            best_d = max_dist
            for i, p in enumerate(frame_points):
                if used[i]:
                    continue
                d = np.linalg.norm(np.array(p) - np.array(state.last))
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                state.points.append(frame_points[best_i])
                state.last = frame_points[best_i]
                state.missed = 0
                used[best_i] = True
            else:
                state.points.append(None)
                state.missed += 1

        for i, p in enumerate(frame_points):
            if not used[i]:
                states.append(Track(p))

        states = [s for s in states if s.missed <= max_missing]

    valid_tracks = [
        t.points
        for t in states
        if sum(pt is not None for pt in t.points) >= min_length
    ]
    return valid_tracks


def process_video(
    model: nn.Module,
    video_path: str,
    output_path: str = "output.mp4",
    img_size: int = 32,
    box_size: int = 4,
) -> List[List[Tuple[int, int]]]:
    """对视频进行检测、跟踪并保存带红框标记的新视频"""
    if cv2 is None:
        raise RuntimeError("cv2 not installed")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))

    detections = []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (img_size, img_size))
        points = detect_points(model, small)
        detections.append(points)
        frames.append(frame)
    cap.release()

    tracks = track_detections(detections)

    for frame_idx, frame in enumerate(frames):
        scale_x = frame.shape[1] / img_size
        scale_y = frame.shape[0] / img_size
        for track in tracks:
            if frame_idx < len(track) and track[frame_idx] is not None:
                x, y = track[frame_idx]
                sx = int(x * scale_x)
                sy = int(y * scale_y)
                half = box_size // 2
                cv2.rectangle(
                    frame,
                    (sx - half, sy - half),
                    (sx + half, sy + half),
                    (0, 0, 255),
                    1,
                )
        out.write(frame)
    out.release()
    return tracks


if __name__ == "__main__":
    if not os.path.exists("point_detector.pth"):
        print("training detector...")
        train_detector()
    else:
        print("loading detector...")
    model = load_model("point_detector.pth")
    if cv2 is not None and os.path.exists("input.mp4"):
        res = process_video(model, "input.mp4", "output.mp4")
        print("tracks:", res)
        print("结果已保存至 output.mp4")
    else:
        print("cv2 未安装或未找到 input.mp4，无法演示视频跟踪")
