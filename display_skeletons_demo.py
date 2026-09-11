# -*- coding: utf-8 -*-
"""
全骨架渲染與繪製技術 (Unified Skeleton Renderer: Face, Mouth, Hands & Body Pose) 獨立學習模組
包含：
1. 手部 21 點骨架繪製演算法 (draw_hand_skeleton)
2. 身體 33 點姿態骨架繪製演算法 (draw_pose_skeleton)
3. 臉部 478 點地標與嘴唇輪廓繪製演算法 (draw_face_and_mouth_landmarks)
4. 綜合繪製調色盤 (Color Palette) 與半透明骨架疊加效果 (Transparent Overlay)
5. 完整的攝影機即時展示 Main 流程
"""

import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks API 引用
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

# 模型路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

# =============================================================================
# 🦴 骨架連線結構 (Skeleton Connections Topology)
# =============================================================================

# 1. 手部 21 點連線定義
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # 姆指 Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # 食指 Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指 Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # 無名指 Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # 小指 Pinky
]

# 2. 身體 33 點連線定義
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # 面部
    (11, 12), (11, 23), (12, 24), (23, 24),                                   # 軀幹
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),               # 左臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),               # 右臂
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),                         # 左腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)                          # 右腿
]

# 3. 嘴唇外圍輪廓索引
LIP_OUTER_CONNECTIONS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]


# =============================================================================
# 🎨 骨架繪製核心函數 (Skeleton Drawing Functions)
# =============================================================================

def draw_hand_skeleton(frame, hand_landmarks_list, color_joints=(0, 255, 255), color_lines=(0, 200, 255)):
    """
    【手部骨架繪製】
    :param frame: BGR 影像矩陣
    :param hand_landmarks_list: MediaPipe 回傳的手部地標列表
    :param color_joints: 關節圓點顏色 (BGR)
    :param color_lines: 骨架線條顏色 (BGR)
    """
    h, w, _ = frame.shape
    for hand in hand_landmarks_list:
        # 1. 畫關節連線 (Line)
        for connection in HAND_CONNECTIONS:
            pt1 = hand[connection[0]]
            pt2 = hand[connection[1]]
            p1x, p1y = int(pt1.x * w), int(pt1.y * h)
            p2x, p2y = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (p1x, p1y), (p2x, p2y), color_lines, 2, cv2.LINE_AA)

        # 2. 畫關節點 (Circle)
        for pt in hand:
            px, py = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (px, py), 4, color_joints, -1, cv2.LINE_AA)


def draw_pose_skeleton(frame, pose_landmarks_list, min_visibility=0.5, color_joints=(0, 255, 0), color_lines=(255, 180, 0)):
    """
    【身體姿態骨架繪製】
    :param frame: BGR 影像矩陣
    :param pose_landmarks_list: MediaPipe 回傳的身體地標列表
    :param min_visibility: 關節可見度繪製門檻
    """
    h, w, _ = frame.shape
    for pose in pose_landmarks_list:
        # 1. 畫骨架連線
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection[0], connection[1]
            pt1, pt2 = pose[idx1], pose[idx2]

            v1 = getattr(pt1, 'visibility', 1.0)
            v2 = getattr(pt2, 'visibility', 1.0)
            if v1 >= min_visibility and v2 >= min_visibility:
                p1x, p1y = int(pt1.x * w), int(pt1.y * h)
                p2x, p2y = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (p1x, p1y), (p2x, p2y), color_lines, 2, cv2.LINE_AA)

        # 2. 畫關節點
        for pt in pose:
            v = getattr(pt, 'visibility', 1.0)
            if v >= min_visibility:
                px, py = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (px, py), 5, color_joints, -1, cv2.LINE_AA)


def draw_face_and_mouth_landmarks(frame, face_landmarks_list, draw_all_mesh=False):
    """
    【人臉與嘴部輪廓繪製】
    :param frame: BGR 影像矩陣
    :param face_landmarks_list: 人臉 478 關鍵點地標
    :param draw_all_mesh: 是否繪製全臉網格 (若為 True 會畫出全臉 478 個小細點)
    """
    h, w, _ = frame.shape
    for face in face_landmarks_list:
        # 1. 選擇性畫出全臉細微點位
        if draw_all_mesh:
            for pt in face:
                px, py = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

        # 2. 高亮畫嘴唇外圍輪廓
        lip_pts = []
        for idx in LIP_OUTER_CONNECTIONS:
            pt = face[idx]
            lip_pts.append((int(pt.x * w), int(pt.y * h)))

        for i in range(len(lip_pts) - 1):
            cv2.line(frame, lip_pts[i], lip_pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

        # 3. 標示嘴部關鍵地標 (13 上唇, 14 下唇, 78 左嘴角, 308 右嘴角, 4 鼻尖, 152 下巴)
        key_indices = [
            (4, (0, 255, 255)),    # 鼻尖 (黃)
            (152, (0, 255, 255)),  # 下巴 (黃)
            (13, (0, 0, 255)),     # 上唇 (紅)
            (14, (0, 0, 255)),     # 下唇 (紅)
            (78, (255, 0, 0)),     # 左嘴角 (藍)
            (308, (255, 0, 0))     # 右嘴角 (藍)
        ]
        for idx, color in key_indices:
            pt = face[idx]
            px, py = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)


# =============================================================================
# 🚀 主程式流程
# =============================================================================

def main():
    print("🚀 啟動全骨架顯示與繪製展示模組...")

    # 初始化偵測器
    face_landmarker = None
    hand_landmarker = None
    pose_landmarker = None

    if os.path.exists(FACE_MODEL_PATH):
        buf = open(FACE_MODEL_PATH, "rb").read()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=buf),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=1)
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    if os.path.exists(HAND_MODEL_PATH):
        buf = open(HAND_MODEL_PATH, "rb").read()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=buf),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=2)
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    if os.path.exists(POSE_MODEL_PATH):
        buf = open(POSE_MODEL_PATH, "rb").read()
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=buf),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=1)
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機")
        return

    start_time = time.time()
    print("🎥 視窗說明：按 'q' 退出程式。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - start_time) * 1000)

        # 1. 人臉與嘴部骨架
        if face_landmarker:
            res = face_landmarker.detect_for_video(mp_image, ts_ms)
            if res.face_landmarks:
                draw_face_and_mouth_landmarks(frame, res.face_landmarks, draw_all_mesh=False)

        # 2. 手部骨架
        if hand_landmarker:
            res = hand_landmarker.detect_for_video(mp_image, ts_ms)
            if res.hand_landmarks:
                draw_hand_skeleton(frame, res.hand_landmarks)

        # 3. 身體姿態骨架
        if pose_landmarker:
            res = pose_landmarker.detect_for_video(mp_image, ts_ms)
            if res.pose_landmarks:
                draw_pose_skeleton(frame, res.pose_landmarks)

        # OS 圖例說明
        cv2.putText(frame, "Yellow/Cyan: Mouth & Face", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Orange/Yellow: Hand Skeleton", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "Blue/Green: Body Pose Skeleton", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        cv2.imshow("All Skeletons Renderer (Face, Mouth, Hands & Pose)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
