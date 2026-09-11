# -*- coding: utf-8 -*-
"""
MediaPipe Face Landmarker (嘴部開合角度 MAR、咀嚼與唇色發紺檢測) 獨立學習模組
包含：
1. MediaPipe Tasks API 人臉 478 關鍵點 (Face Landmarks) 模型載入與自動下載
2. 嘴部關鍵點定義 (13 上唇, 14 下唇, 78 左嘴角, 308 右嘴角, 4 鼻尖, 152 下巴)
3. 嘴部開合比例 (Mouth Aspect Ratio - MAR) 與 EMA 指數滑動平滑計算
4. 嘴唇發紺 (Lip Cyanosis / Blueness) 色彩缺氧分析演算法
5. OpenCV 畫嘴部地標輪廓與動態數值顯示
6. 完整的攝影機即時測試 Main 流程
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

# 人臉姿態模型下載網址與本地儲存檔名 (face_landmarker.task 約 3.7MB)
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# 嘴部與周圍參考關鍵點索引 (MediaPipe Face Mesh 478 點索引)
LIP_TOP_IDX = 13       # 上唇內緣
LIP_BOTTOM_IDX = 14    # 下唇內緣
LIP_LEFT_IDX = 78      # 嘴唇左角
LIP_RIGHT_IDX = 308    # 嘴唇右角
NOSE_TIP_IDX = 4       # 鼻尖
CHIN_IDX = 152         # 下巴
LEFT_EYE_IDX = 33      # 左眼外角 (供瞳孔距離間距歸一化)
RIGHT_EYE_IDX = 263    # 右眼外角

# 嘴唇外圍全輪廓 (用於畫圖繪製)
LIP_OUTER_CONNECTIONS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]


def create_face_landmarker(model_path=FACE_MODEL_PATH):
    """
    初始化 MediaPipe FaceLandmarker (VIDEO 模式)
    若模型不存在則自動從 Google Storage 下載。
    """
    if not os.path.exists(model_path):
        print("【提示】首次執行：正在下載人臉模型 face_landmarker.task (約 3.7MB)...")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, model_path)
            print("【完成】人臉模型下載成功。")
        except Exception as e:
            print(f"【錯誤】下載人臉模型失敗: {e}")
            return None

    try:
        model_buffer = open(model_path, "rb").read()
        base_options = mp_tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # 影片/鏡頭串流模式
            num_faces=1,                                # 最多偵測 1 張臉
            min_face_detection_confidence=0.5,          # 人臉偵測門檻
            min_tracking_confidence=0.5                 # 人臉追蹤門檻
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"【錯誤】初始化 FaceLandmarker 失敗: {e}")
        return None


def calculate_mar(lip_top, lip_bottom, lip_left, lip_right, img_w, img_h):
    """
    計算嘴部開合比例 (Mouth Aspect Ratio - MAR)
    MAR = 嘴唇垂直距離 (Vertical Distance) / 嘴唇水平距離 (Horizontal Distance)
    """
    v_dist = math.hypot((lip_top.x - lip_bottom.x) * img_w, (lip_top.y - lip_bottom.y) * img_h)
    h_dist = math.hypot((lip_left.x - lip_right.x) * img_w, (lip_left.y - lip_right.y) * img_h)

    if h_dist <= 0:
        return 0.0

    return v_dist / h_dist


def analyze_lip_cyanosis(frame, lip_top, lip_bottom, img_w, img_h, blue_th=0.38):
    """
    分析嘴唇發紺 (Cyanosis / 缺氧青紫) 色彩
    在上下唇交界處採樣 RGB 色彩，計算藍光比例 (Blueness) = Blue / (Red + Green + Blue)
    """
    cx = int((lip_top.x + lip_bottom.x) / 2 * img_w)
    cy = int((lip_top.y + lip_bottom.y) / 2 * img_h)

    # 確保採樣點在影像邊界內
    cx = max(2, min(img_w - 3, cx))
    cy = max(2, min(img_h - 3, cy))

    # 取 5x5 區域的平均 BGR 顏色
    patch = frame[cy - 2:cy + 3, cx - 2:cx + 3]
    if patch.size == 0:
        return 0.0, False

    b, g, r = patch.mean(axis=(0, 1))
    s = r + g + b
    blueness = (b / s) if s > 0 else 0.0
    is_cyanotic = blueness >= blue_th

    return blueness, is_cyanotic


def draw_mouth_landmarks(frame, face_landmarks):
    """
    在 OpenCV 畫面上標示嘴唇關鍵點與連線輪廓
    """
    h, w, _ = frame.shape

    # 畫嘴唇輪廓線
    pts = []
    for idx in LIP_OUTER_CONNECTIONS:
        pt = face_landmarks[idx]
        pts.append((int(pt.x * w), int(pt.y * h)))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 1)  # 黃色邊框

    # 高亮標示 MAR 的 4 個核心地標 (上下唇內緣、左右嘴角)
    for idx, color in [(LIP_TOP_IDX, (0, 0, 255)),       # 紅色: 上唇
                       (LIP_BOTTOM_IDX, (0, 0, 255)),    # 紅色: 下唇
                       (LIP_LEFT_IDX, (255, 0, 0)),      # 藍色: 左嘴角
                       (LIP_RIGHT_IDX, (255, 0, 0))]:    # 藍色: 右嘴角
        pt = face_landmarks[idx]
        px, py = int(pt.x * w), int(pt.y * h)
        cv2.circle(frame, (px, py), 4, color, -1)


def main():
    print("🚀 啟動嘴部開合 (MAR) 與唇色分析展示模組...")
    landmarker = create_face_landmarker()
    if landmarker is None:
        print("❌ 無法初始化 FaceLandmarker，程式結束。")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機 (Index 0)")
        return

    start_time = time.time()
    mar_smooth = None  # EMA 指數滑動平均平滑值
    EMA_ALPHA = 0.4

    print("🎥 攝影機已開啟，請在視窗中查看嘴部開合 (MAR) 數據。按 'q' 鍵退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face = result.face_landmarks[0]
            lip_top = face[LIP_TOP_IDX]
            lip_bottom = face[LIP_BOTTOM_IDX]
            lip_left = face[LIP_LEFT_IDX]
            lip_right = face[LIP_RIGHT_IDX]

            # 1. 計算原始 MAR 與 EMA 平滑值
            mar_raw = calculate_mar(lip_top, lip_bottom, lip_left, lip_right, w, h)
            mar_smooth = mar_raw if mar_smooth is None else (EMA_ALPHA * mar_raw + (1 - EMA_ALPHA) * mar_smooth)

            # 2. 嘴唇發紺 (缺氧青紫) 色彩分析
            blueness, is_cyanotic = analyze_lip_cyanosis(frame, lip_top, lip_bottom, w, h)

            # 3. 畫嘴部地標
            draw_mouth_landmarks(frame, face)

            # 4. 顯示狀態與數值
            mouth_status = "Mouth OPEN" if mar_smooth > 0.15 else "Mouth CLOSED"
            status_color = (0, 255, 0) if mar_smooth <= 0.15 else (0, 165, 255)

            cv2.putText(frame, f"MAR (Mouth Aspect Ratio): {mar_smooth:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mouth State: {mouth_status}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Lip Blueness: {blueness:.3f} (Cyanosis: {is_cyanotic})", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_cyanotic else (255, 255, 0), 2)

        cv2.imshow("Mouth Landmarks & MAR Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 使用者關閉視窗。")
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
