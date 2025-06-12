# -*- coding: utf-8 -*-
"""
基于傳統方法的點目標檢測與跟蹤基線。
使用簡化的 CFAR 檢測、卡爾曼濾波和匈牙利算法數據關聯。
依賴: numpy, opencv-python, scipy(可選)
"""

import os
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class KalmanTrack:
    """簡單的常速度卡爾曼濾波跟蹤器"""

    def __init__(self, x: float, y: float, track_id: int):
        self.track_id = track_id
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.A = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 1.0
        self.missed = 0
        self.history: List[Tuple[int, int]] = []

    def predict(self):
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
        pred = self.state[:2]
        self.history.append((int(pred[0]), int(pred[1])))
        return pred

    def update(self, z: np.ndarray):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        self.history[-1] = (int(self.state[0]), int(self.state[1]))
        self.missed = 0


class Tracker:
    def __init__(self, max_dist: float = 10.0, max_missed: int = 5):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks: List[KalmanTrack] = []
        self.next_id = 0

    def update(self, detections: List[Tuple[int, int]]):
        # 預測階段
        preds = [t.predict() for t in self.tracks]

        if len(preds) == 0:
            for det in detections:
                self.tracks.append(KalmanTrack(det[0], det[1], self.next_id))
                self.next_id += 1
            return

        cost = np.zeros((len(preds), len(detections)), dtype=np.float32)
        for i, p in enumerate(preds):
            for j, d in enumerate(detections):
                cost[i, j] = np.linalg.norm(p - np.array(d))

        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            # 簡單貪婪匹配
            row_idx, col_idx = [], []
            used_det = set()
            for i in range(len(preds)):
                min_d = self.max_dist
                min_j = -1
                for j in range(len(detections)):
                    if j in used_det:
                        continue
                    d = cost[i, j]
                    if d < min_d:
                        min_d = d
                        min_j = j
                if min_j >= 0:
                    row_idx.append(i)
                    col_idx.append(min_j)
                    used_det.add(min_j)

        assigned = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] > self.max_dist:
                continue
            self.tracks[r].update(np.array(detections[c], dtype=np.float32))
            assigned.add(r)
            assigned.add(("det", c))

        for i, t in enumerate(self.tracks):
            if i not in assigned:
                t.missed += 1

        for j, det in enumerate(detections):
            if ("det", j) not in assigned:
                self.tracks.append(KalmanTrack(det[0], det[1], self.next_id))
                self.next_id += 1

        # 移除長時間未匹配的軌跡
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def get_active_tracks(self):
        return [t for t in self.tracks if t.missed == 0]


def cfar_detect(gray: np.ndarray, ksize: int = 7, thresh: float = 20) -> List[Tuple[int, int]]:
    """非常簡化的 CFAR 檢測"""
    mean = cv2.blur(gray, (ksize, ksize))
    diff = gray.astype(np.float32) - mean.astype(np.float32)
    mask = diff > thresh
    ys, xs = np.where(mask)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def process_video(video_path: str, output_path: str = "baseline_out.mp4"):
    if cv2 is None:
        raise RuntimeError("cv2 not installed")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (width, height),
    )

    tracker = Tracker()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = cfar_detect(gray)
        tracker.update(dets)
        for track in tracker.get_active_tracks():
            x, y = track.history[-1]
            cv2.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), 1)
            cv2.putText(
                frame,
                str(track.track_id),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()


if __name__ == "__main__":
    if cv2 is not None and os.path.exists("input.mp4"):
        process_video("input.mp4")
        print("基線處理完成，結果保存在 baseline_out.mp4")
    else:
        print("cv2 未安裝或未找到 input.mp4")
