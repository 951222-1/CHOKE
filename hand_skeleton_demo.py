# -*- coding: utf-8 -*-
"""
MediaPipe Hand Landmarker (手部骨架偵測與手勢識別) 獨立學習模組
包含：
1. MediaPipe Tasks API 手部關節模型載入與自動下載
2. 手部 21 個關鍵點 (21 Landmarks) 與 骨架連線定義 (HAND_CONNECTIONS)
3. 幾何位置判斷 (手部是否接近喉嚨 / 窒息手勢)
4. OpenCV 畫手部骨架 (黃色關節點 + 橘色連接線)
5. 完整的攝影機即時測試 Main 流程
"""

import os
import sys
import math
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks API 引用
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

# 手部模型下載網址與本地儲存檔名
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# 手部 21 個關節點的連線結構 (Skeleton Connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # 姆指 Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # 食指 Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指 Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # 無名指 Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # 小指 Pinky
]


def create_hand_landmarker(model_path=HAND_MODEL_PATH):
    """
    初始化 MediaPipe HandLandmarker (VIDEO 模式)
    若模型不存在則自動從 Google Storage 下載。
    """
    if not os.path.exists(model_path):
        alt_path = r"C:\Users\ph090\hand_landmarker.task"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"【提示】首次執行：正在下載手部模型 hand_landmarker.task (約 7.5MB)...")
            try:
                urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
                print("【完成】手部模型下載成功。")
            except Exception as e:
                print(f"【錯誤】下載手部模型失敗: {e}")
                return None

    try:
        model_buffer = open(model_path, "rb").read()
        base_options = mp_tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # 影片/鏡頭串流模式
            num_hands=2,                                # 最多偵測 2 隻手
            min_hand_detection_confidence=0.5,           # 偵測門檻
            min_tracking_confidence=0.5                 # 追蹤門檻
        )
        return mp_vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"【錯誤】初始化 HandLandmarker 失敗: {e}")
        return None


def estimate_throat(nose, chin, extend=0.6):
    """
    估算喉嚨位置：根據「鼻子 (nose) -> 下巴 (chin)」的方向向量向下延伸。
    nose, chin 皆為 normalized 座標 (x, y) [0.0 ~ 1.0]
    """
    tx = chin[0] + extend * (chin[0] - nose[0])
    ty = chin[1] + extend * (chin[1] - nose[1])
    return tx, ty


def hand_near_throat(hands, nose, chin, w, h, radius_scale=0.6):
    """
    判斷是否有任一隻手的任一關節點落入喉嚨附近的警戒範圍。
    :param hands: 手部座標列表，格式為 [[(x0, y0), (x1, y1), ...], ...] (已正規化)
    :param nose: 鼻子 (x, y)
    :param chin: 下巴 (x, y)
    :param w: 影像寬度 (像素)
    :param h: 影像高度 (像素)
    :param radius_scale: 警戒範圍半徑比例 (相對於臉長)
    :return: True / False
    """
    if not hands or nose is None or chin is None:
        return False

    tx, ty = estimate_throat(nose, chin)

    # 以臉長 (鼻子到下巴的實際像素距離) 作為動態半徑基準
    face_length_px = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    radius_px = max(face_length_px * radius_scale, 1.0)

    # 檢查手部每一個 point 到喉嚨預測點的距離
    for hand in hands:
        for (x, y) in hand:
            dist_px = math.hypot((x - tx) * w, (y - ty) * h)
            if dist_px < radius_px:
                return True
    return False


def draw_hand_skeleton(frame, hand_landmarks_list):
    """
    在 OpenCV 畫面上繪製手部骨架與關節點
    :param frame: BGR 影像矩陣
    :param hand_landmarks_list: MediaPipe 回傳的手部地標列表 (包含 21 個 NormalizedLandmark)
    """
    h, w, _ = frame.shape
    for hand in hand_landmarks_list:
        # 1. 畫關節點 (21 個點)
        for pt in hand:
            px, py = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)  # 黃色圓點 (BGR)

        # 2. 畫連線 (根據 HAND_CONNECTIONS 拓撲結構)
        for connection in HAND_CONNECTIONS:
            pt1 = hand[connection[0]]
            pt2 = hand[connection[1]]
            p1x, p1y = int(pt1.x * w), int(pt1.y * h)
            p2x, p2y = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 200, 255), 2)  # 橘色骨架線 (BGR)


def main():
    print("🚀 啟動手部骨架偵測展示模組...")
    landmarker = create_hand_landmarker()
    if landmarker is None:
        print("❌ 無法初始化 HandLandmarker，程式結束。")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機 (Index 0)")
        return

    start_time = time.time()
    print("🎥 攝影機已開啟，請在視窗中查看手部骨架。按 'q' 鍵退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻轉影像 (像鏡子一樣)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 將 OpenCV BGR 轉為 RGB 格式並建立 MP Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO 模式需要傳入嚴格遞增的時間戳 (ms)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 進行手部偵測
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 繪製手部骨架
        if result.hand_landmarks:
            draw_hand_skeleton(frame, result.hand_landmarks)
            
            # 顯示偵測到的手數量
            cv2.putText(frame, f"Hands Detected: {len(result.hand_landmarks)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Hand Skeleton Detection (MediaPipe Tasks)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 使用者關閉視窗。")
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
