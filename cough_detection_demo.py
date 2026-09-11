# -*- coding: utf-8 -*-
"""
咳嗽與嗆咳偵測 (Dual-Stream Cough Detection: Audio YAMNet + Vision Jerk) 獨立學習模組
包含：
1. 聽覺嗆咳偵測：Google YAMNet AI 模型 (16kHz 麥克風音訊串流 + 語音對抗抑制)
2. 影像嗆咳偵測：下巴/頭部快速抽動與位移特徵 (Jaw Velocity & Jerk Analysis)
3. 雙流咳嗽融合 (Audio-Vision Fused Cough Detection)
4. 完整的即時測試與數值視覺化 Main 流程
"""

import os
import sys
import math
import time
import csv
import sqlite3
import threading
from collections import deque
import cv2
import numpy as np

# 音訊相關套件相容性檢查
try:
    import sounddevice as sd
    import tensorflow as tf
    import tensorflow_hub as hub
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

# MediaPipe
import mediapipe as mp
mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
DB_PATH = os.environ.get("ANTICHOKE_DB", os.path.join(BASE_DIR, "eating_records.db"))
_last_cough_db_time = 0.0


def save_silent_cough_record(patient_id="PATIENT_001", patient_name="個案A", reason="嗆咳事件"):
    """
    嗆咳事件（原 L1 / L2 合併）：
    取消劇烈晃動條件，偵測到咳嗽（聲音或影像頭部抽動）時，
    僅於背景非阻塞寫入 SQLite 資料庫（標籤為 嗆咳紀錄），不會跳出 UI 彈窗、不閃紅橫幅。
    """
    global _last_cough_db_time
    now = time.time()
    if now - _last_cough_db_time < 3.0:
        return
    _last_cough_db_time = now

    def _worker():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eating_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT, patient_name TEXT, timestamp TEXT,
                    chew_count INTEGER, status TEXT
                )
            """)
            now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cursor.execute("""
                INSERT INTO eating_records (patient_id, patient_name, timestamp, chew_count, status)
                VALUES (?, ?, ?, ?, ?)
            """, (patient_id, patient_name, now_str, 0, "嗆咳紀錄"))
            conn.commit()
            conn.close()
            print(f"💾 【後台靜默紀錄】已成功將嗆咳事件寫入 SQLite ({now_str}, 標籤: 嗆咳紀錄, 原因: {reason})")
        except Exception as e:
            print(f"❌ 後台靜默寫入 SQLite 失敗: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# =============================================================================
# 🎤 1. 聽覺嗆咳偵測器 (Audio YAMNet Cough Detector)
# =============================================================================
class AudioCoughDetector:
    """
    基於 Google YAMNet 深度學習音訊模型的咳嗽/嗆咳聲辨識器
    """
    AUDIO_SR = 16000          # YAMNet 需要 16kHz 單聲道
    AUDIO_WINDOW_SEC = 0.975  # YAMNet 的輸入音訊視窗 (秒)
    AUDIO_HOP_SEC = 0.5       # 每 0.5 秒分析一次

    # 音效分類關鍵字
    CHOKE_KEYWORDS = ["choking", "gagging", "cough", "gasp", "wheeze", "throat"]
    SPEECH_KEYWORDS = ["speech", "conversation", "narration"]  # 用於扣除說話誤判

    def __init__(self):
        self.yamnet_model = None
        self.choke_indices = []
        self.speech_indices = []
        self.is_running = False
        self.lock = threading.Lock()

        # 急劇高音量防環境音誤報機制
        self.high_db_since = None
        self.sustained_high_db_alert = False

        self.latest_state = {
            "choke_score": 0.0,
            "top_label": "-",
            "energy": 0.0,
            "db_spl": 0.0,
            "high_db_duration": 0.0,
            "sustained_high_db_alert": False,
            "updated": 0.0
        }

    def load_model(self):
        """載入 YAMNet 模型並自動對應 521 種聲音類別中的咳嗽/說話索引"""
        if not HAS_AUDIO_LIBS:
            print("⚠️ 未安裝 tensorflow / tensorflow_hub / sounddevice，跳過聽覺咳嗽偵測。")
            return False

        print("⏳ 正在載入 YAMNet 音訊 AI 模型 (首次執行需下載約 17MB)...")
        try:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self.yamnet_model.class_map_path().numpy().decode("utf-8")

            with open(class_map_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row["display_name"].lower()
                    idx = int(row["index"])
                    if any(k in name for k in self.CHOKE_KEYWORDS):
                        self.choke_indices.append(idx)
                    if any(k in name for k in self.SPEECH_KEYWORDS):
                        self.speech_indices.append(idx)

            print(f"✅ YAMNet 模型載入成功！捕捉到 {len(self.choke_indices)} 個嗆咳相關類別。")
            return True
        except Exception as e:
            print(f"❌ YAMNet 模型載入失敗: {e}")
            return False

    def score_waveform(self, waveform):
        """輸入 16kHz float32 波形向量，計算咳嗽與說話抑制分數"""
        if self.yamnet_model is None:
            return 0.0, "-"

        scores, _, _ = self.yamnet_model(waveform)
        mean_scores = scores.numpy().mean(axis=0)

        choke_raw = float(mean_scores[self.choke_indices].sum())
        speech_raw = float(mean_scores[self.speech_indices].sum())

        # 語音抗干擾扣分：正常講話聲音會適度扣抵咳嗽分數，降低聊天誤判率
        choke_score = max(0.0, choke_raw - 0.5 * speech_raw)
        top_idx = int(mean_scores.argmax())

        return choke_score, str(top_idx)

    def start_listening(self):
        """啟動麥克風背景即時監聽與分析 Thread"""
        if not HAS_AUDIO_LIBS or self.yamnet_model is None:
            return

        ring_buffer = np.zeros(int(self.AUDIO_SR * 1.5), dtype=np.float32)
        ring_lock = threading.Lock()

        def audio_callback(indata, frames, t, status):
            nonlocal ring_buffer
            mono = indata[:, 0].astype(np.float32)
            with ring_lock:
                ring_buffer = np.roll(ring_buffer, -len(mono))
                ring_buffer[-len(mono):] = mono

        def analyzer_thread():
            self.is_running = True
            while self.is_running:
                time.sleep(self.AUDIO_HOP_SEC)
                with ring_lock:
                    wave = ring_buffer.copy()

                try:
                    win_samples = int(self.AUDIO_SR * self.AUDIO_WINDOW_SEC)
                    choke_score, top_label = self.score_waveform(wave[-win_samples:])
                    energy = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))

                    # 計算即時分貝值 (dB SPL 校正)
                    if energy > 1e-7:
                        db_spl = max(0.0, min(120.0, 20.0 * math.log10(energy) + 90.0))
                    else:
                        db_spl = 0.0

                    now = time.time()
                    # 急劇高音量(>=80dB)且持續3秒以上未降回平均音量(<=60dB)邏輯
                    if db_spl >= 80.0:
                        if self.high_db_since is None:
                            self.high_db_since = now
                        elif now - self.high_db_since >= 3.0:
                            self.sustained_high_db_alert = True
                    elif db_spl <= 60.0:
                        self.high_db_since = None
                        self.sustained_high_db_alert = False

                    high_duration = (now - self.high_db_since) if self.high_db_since else 0.0

                    with self.lock:
                        self.latest_state = {
                            "choke_score": choke_score,
                            "top_label": top_label,
                            "energy": energy,
                            "db_spl": db_spl,
                            "high_db_duration": high_duration,
                            "sustained_high_db_alert": self.sustained_high_db_alert,
                            "updated": now
                        }
                except Exception as e:
                    print(f"⚠️ 聲音分析異常: {e}")

        try:
            stream = sd.InputStream(samplerate=self.AUDIO_SR, channels=1,
                                    blocksize=int(self.AUDIO_SR * 0.1), callback=audio_callback)
            stream.start()
            threading.Thread(target=analyzer_thread, daemon=True).start()
            print("🎤 麥克風即時嗆咳聲音與 80dB 高音量防誤報監聽已啟動！")
        except Exception as e:
            print(f"❌ 無法開啟麥克風輸入裝置: {e}")

    def get_state(self):
        with self.lock:
            return self.latest_state.copy()


# =============================================================================
# 👁️ 2. 影像咳嗽/抽動分析器 (Vision Cough Detector)
# =============================================================================
class VisionCoughDetector:
    """
    根據人臉下巴/頭部快速抽動 (Jerk) 與震幅進行影像嗆咳分析
    """
    def __init__(self, movement_std_th=0.015, confirm_frames=2):
        self.jaw_history = deque(maxlen=30)
        self.cough_frame_counter = 0
        self.movement_std_th = movement_std_th
        self.confirm_frames = confirm_frames

    def update(self, face_landmarks, img_w, img_h):
        """
        輸入 MediaPipe Face Landmarks，計算下巴移動標準差 (Movement Std)
        """
        if not face_landmarks:
            self.cough_frame_counter = 0
            return False, 0.0

        p4 = face_landmarks[4]      # 鼻尖
        p152 = face_landmarks[152]  # 下巴

        # 估算像素級下巴與頭部垂直相對距離
        jaw_y = (p152.y - p4.y) * img_h
        self.jaw_history.append(jaw_y)

        if len(self.jaw_history) < 10:
            return False, 0.0

        # 計算近期 10 幀下巴特徵點變化的標準差
        recent_jaw = list(self.jaw_history)[-10:]
        movement_std = float(np.std(recent_jaw))

        # 若標準差高於動態門檻，視為有痙攣/咳嗽抽動
        if movement_std > self.movement_std_th:
            self.cough_frame_counter += 1
        else:
            self.cough_frame_counter = 0

        # 連續若干幀達標才確認為一次影像咳嗽事件
        if self.cough_frame_counter >= self.confirm_frames:
            self.cough_frame_counter = 0
            return True, movement_std

        return False, movement_std


# =============================================================================
# 🚀 主程式流程
# =============================================================================

def main():
    print("🚀 啟動雙流 (Audio + Vision) 咳嗽偵測展示模組...")

    # 1. 啟動聽覺咳嗽偵測
    audio_detector = AudioCoughDetector()
    has_audio = audio_detector.load_model()
    if has_audio:
        audio_detector.start_listening()

    # 2. 啟動影像咳嗽偵測
    vision_detector = VisionCoughDetector()

    if not os.path.exists(FACE_MODEL_PATH):
        print("❌ 找不到 face_landmarker.task 模型")
        return

    model_buffer = open(FACE_MODEL_PATH, "rb").read()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=model_buffer),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1)
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機")
        return

    start_time = time.time()
    print("🎥 視窗說明：對麥克風咳嗽或對鏡頭快速上下晃動頭部進行測試。按 'q' 鍵退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - start_time) * 1000)

        # 進行人臉地標偵測
        result = face_landmarker.detect_for_video(mp_image, ts_ms)
        face_landmarks = result.face_landmarks[0] if result.face_landmarks else None

        # 影像咳嗽檢測
        vision_cough, movement_std = vision_detector.update(face_landmarks, w, h)

        # 聽覺咳嗽與急劇分貝檢測
        audio_state = audio_detector.get_state()
        audio_score = audio_state["choke_score"]
        db_spl = audio_state.get("db_spl", 0.0)
        high_db_dur = audio_state.get("high_db_duration", 0.0)
        sustained_high_db_alert = audio_state.get("sustained_high_db_alert", False)
        audio_cough = audio_score > 0.15

        # 雙流咳嗽融合判讀
        dual_cough_fused = vision_cough and audio_cough

        # 靜默後台紀錄 (原 L1 / L2 合併為嗆咳紀錄，無身體晃動強制綁定，無彈窗、無紅橫幅)
        if vision_cough or audio_cough:
            reason = "影像抽動與聲音咳嗽雙流" if dual_cough_fused else ("聲音咳嗽" if audio_cough else "影像頭部抽動")
            save_silent_cough_record(patient_id="PATIENT_001", patient_name="個案A", reason=reason)

        # 畫面上標示資訊
        cv2.putText(frame, f"Audio Cough Score (YAMNet): {audio_score:.3f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if audio_cough else (255, 255, 255), 2)
        cv2.putText(frame, f"Audio Volume (dB SPL): {db_spl:.1f} dB (Th: >80dB, Rec: <=60dB)", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255) if db_spl >= 80.0 else (0, 255, 0), 2)
        cv2.putText(frame, f"High Volume Hold (>80dB): {high_db_dur:.1f}s / 3.0s", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255) if high_db_dur > 0 else (200, 200, 200), 2)
        cv2.putText(frame, f"Vision Jerk Std: {movement_std:.3f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if vision_cough else (255, 255, 255), 2)

        # 狀態警示
        if sustained_high_db_alert:
            cv2.putText(frame, "🚨 [ALERT] SUSTAINED HIGH VOLUME (>80dB for 3s without recovery to 60dB)!", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif dual_cough_fused:
            cv2.putText(frame, "🚨 [HIGH RISK] DUAL-STREAM COUGH FUSED!", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        elif vision_cough:
            cv2.putText(frame, "⚠️ Vision Cough Motion Detected", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif audio_cough:
            cv2.putText(frame, "🎤 Audio Cough Sound Detected", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Dual-Stream Cough Detection Engine Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()


if __name__ == "__main__":
    main()
