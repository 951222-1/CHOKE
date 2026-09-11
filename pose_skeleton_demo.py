# -*- coding: utf-8 -*-
"""
MediaPipe Pose Landmarker (身體姿態骨架偵測與劇烈晃動分析) 獨立學習模組
包含：
1. MediaPipe Tasks API 身體 33 關鍵點 (Pose Landmarks) 模型載入與自動下載
2. 軀幹與四肢骨架連線結構 (POSE_CONNECTIONS) 定義
3. 身體劇烈晃動/掙扎無因次化計算 (Scale-Invariant Displacement)
4. OpenCV 畫全身體骨架 (綠色關節點 + 藍色骨架連線)
5. 完整的攝影機即時測試 Main 流程
"""

import os
import sys
import math
import time
import urllib.request
from collections import deque
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks API 引用
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

# 身體姿態模型下載網址與本地儲存檔名 (pose_landmarker_lite.task 約 9.1MB)
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

# 身體 33 個關節點連線結構 (Pose Skeleton Connections)
POSE_CONNECTIONS = [
    # 面部器官 Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 軀幹 Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 上肢 / 雙臂 Upper Limbs
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # 左臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # 右臂
    # 下肢 / 雙腿 Lower Limbs
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),            # 左腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)             # 右腿
]


def create_pose_landmarker(model_path=POSE_MODEL_PATH):
    """
    初始化 MediaPipe PoseLandmarker (VIDEO 模式)
    若模型不存在則自動從 Google Storage 下載。
    """
    if not os.path.exists(model_path):
        print("【提示】首次執行：正在下載身體姿態模型 pose_landmarker_lite.task (約 9.1MB)...")
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
            print("【完成】身體姿態模型下載成功。")
        except Exception as e:
            print(f"【錯誤】下載身體姿態模型失敗: {e}")
            return None

    try:
        model_buffer = open(model_path, "rb").read()
        base_options = mp_tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # 影片/鏡頭串流模式
            num_poses=1,                                # 最多偵測 1 人
            min_pose_detection_confidence=0.5,          # 姿態偵測門檻
            min_tracking_confidence=0.5                # 姿態追蹤門檻
        )
        return mp_vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"【錯誤】初始化 PoseLandmarker 失敗: {e}")
        return None


def calculate_body_shaking(nose_history, face_size):
    """
    計算無因次化身體/頭部劇烈晃動位移 (Scale-Invariant Displacement)
    :param nose_history: 1 秒內 (約 30 幀) 鼻尖/頭部 (x, y) 像素座標佇列
    :param face_size: 目前頭部/臉部參考長度 (像素)
    :return: 歸一化晃動係數 (例如 > 1.5 代表 1 秒內位移超過 1.5 倍臉高，視為劇烈晃動/掙扎)
    """
    if len(nose_history) < 2 or face_size <= 0:
        return 0.0

    total_disp_px = 0.0
    for i in range(1, len(nose_history)):
        dx = nose_history[i][0] - nose_history[i - 1][0]
        dy = nose_history[i][1] - nose_history[i - 1][1]
        total_disp_px += math.hypot(dx, dy)

    return total_disp_px / face_size


def draw_pose_skeleton(frame, pose_landmarks_list, min_visibility=0.5):
    """
    在 OpenCV 畫面上繪製身體姿態 33 點骨架與連線
    :param frame: BGR 影像矩陣
    :param pose_landmarks_list: MediaPipe 回傳的姿態地標列表 (包含 33 個 Landmark)
    :param min_visibility: 關節可見度繪製門檻
    """
    h, w, _ = frame.shape
    for pose in pose_landmarks_list:
        # 1. 畫連線 (根據 POSE_CONNECTIONS 拓撲結構)
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection[0], connection[1]
            pt1 = pose[idx1]
            pt2 = pose[idx2]

            # 檢查兩端點的可見度 (visibility)
            v1 = getattr(pt1, 'visibility', 1.0)
            v2 = getattr(pt2, 'visibility', 1.0)
            if v1 >= min_visibility and v2 >= min_visibility:
                p1x, p1y = int(pt1.x * w), int(pt1.y * h)
                p2x, p2y = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (p1x, p1y), (p2x, p2y), (255, 180, 0), 2)  # 水藍色連線 (BGR)

        # 2. 畫關節點 (33 個點)
        for pt in pose:
            v = getattr(pt, 'visibility', 1.0)
            if v >= min_visibility:
                px, py = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)  # 鮮綠色圓點 (BGR)


def main():
    print("🚀 啟動身體姿態骨架偵測展示模組...")
    landmarker = create_pose_landmarker()
    if landmarker is None:
        print("❌ 無法初始化 PoseLandmarker，程式結束。")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機 (Index 0)")
        return

    start_time = time.time()
    nose_history = deque(maxlen=30)  # 保存過去約 1 秒 (30 幀) 的頭部/鼻尖座標

    print("🎥 攝影機已開啟，請在視窗中查看身體姿態骨架。按 'q' 鍵退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻轉影像 (像鏡子一樣)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 轉 BGR 為 RGB 建立 MP Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)

        # 執行身體姿態偵測
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            draw_pose_skeleton(frame, result.pose_landmarks)

            # 抓取鼻尖 (Index 0) 與 兩肩 (Index 11, 12) 計算體幅基準
            nose = pose[0]
            l_shoulder, r_shoulder = pose[11], pose[12]

            nose_px = (int(nose.x * w), int(nose.y * h))
            nose_history.append(nose_px)

            # 以雙肩像素距離作為基準尺寸
            shoulder_dist_px = math.hypot((r_shoulder.x - l_shoulder.x) * w,
                                          (r_shoulder.y - l_shoulder.y) * h)
            shoulder_dist_px = max(shoulder_dist_px, 1.0)

            # 計算位移晃動值
            shake_ratio = calculate_body_shaking(nose_history, shoulder_dist_px)
            is_shaking = shake_ratio > 1.5

            # 畫面上顯示晃動分析結果
            color = (0, 0, 255) if is_shaking else (0, 255, 0)
            cv2.putText(frame, f"Body Shake Score: {shake_ratio:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if is_shaking:
                cv2.putText(frame, "[WARNING] Body Struggling/Shaking Detected!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Body Skeleton Detection (MediaPipe Pose Tasks)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 使用者關閉視窗。")
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
