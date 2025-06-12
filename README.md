# 点目标深度学习示例

本仓库包含三个示例：

1. `point_target_deep_learning.py`：区分图像中是否存在单个点目标的简单示例。
2. `point_target_video_pipeline.py`：从训练到检测并跟踪视频中多个点目标的完整流程，支持在噪声中利用运动信息过滤误检。
3. `point_target_baseline.py`：基于 CFAR 检测和卡尔曼 + 匈牙利算法的传统跟踪基线。

## 运行方式

1. 安装依赖：
    ```bash
    pip install torch numpy opencv-python
    ```
2. 运行单帧训练脚本：
    ```bash
    python point_target_deep_learning.py
    ```
    训练完成后会生成 `point_target_model.pth` 模型文件。

3. 运行视频检测跟踪脚本（可选）：
    ```bash
    python point_target_video_pipeline.py
    ```
    如果当前目录存在 `input.mp4` 且安装了 OpenCV，会在 `output.mp4` 中生成
    带红框标记的跟踪结果，并在终端打印轨迹坐标。
    脚本会对检测结果进行连续帧关联，只保留持续存在的真实目标。
4. 运行传统基线脚本（可选）：
    ```bash
    python point_target_baseline.py
    ```
    需要安装 `scipy` 才能使用匈牙利算法；否则脚本会退化为简单的贪婪匹配。

`point_target_deep_learning.py` 会随机生成带或不带点目标的 16x16 单通道图像，并训练一个小型卷积神经网络进行二分类；`point_target_video_pipeline.py` 则演示了合成训练、在含噪视频中检测点目标，并利用目标的连续移动排除虚假点的全过程。
