# -*- coding: utf-8 -*-
#
# 防哽咽即時監測 —— 主程式
#
# 這支程式用鏡頭看臉、用麥克風聽聲音,判斷有沒有人在吃東西時噎到。
# 一發現可疑就警報(畫面紅色橫幅、電腦嗶嗶叫、彈窗、還會發 LINE)。
#
# 判斷同時看好幾種線索:嘴巴開合、咀嚼、有沒有嗆咳聲(用 Google 的 YAMNet 模型)、
# 手有沒有抓喉嚨、嘴唇有沒有發紫、以及「安靜卻不動」的無聲窒息。這些線索分別在
# gesture.py / silent_choke.py / cyanosis.py / evidence_fusion.py 這幾個小檔裡,
# 想看演算法怎麼寫的話,從那幾個檔看最快,它們最短也最好懂。
#
# 沒裝聲音相關套件也沒關係,程式會自動退成「只看影像」模式,不會掛掉。
#
# 要看整支怎麼跑,直接跳到最底下的 main():開鏡頭 -> 一格一格處理畫面 -> 判斷 -> 警報。
#

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')
import subprocess

# ==========================================
# 📦 0. 套件自我檢查
#    ⚠️ 必須在重量級 import(cv2 等)之前執行,才能在缺套件時
#       給出友善提示而非直接 ImportError。故此段維持模組層級。
# ==========================================
REQUIRED_PACKAGES = {          # 核心套件(缺了就不能跑)
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "requests": "requests",
}
AUDIO_PACKAGES = {             # 聲音模組套件(缺了自動降級為純影像)
    "tensorflow": "tensorflow",
    "tensorflow_hub": "tensorflow-hub",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
}

def check_packages(pkgs, required=True):
    missing = []
    for module_name, pip_name in pkgs.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("=" * 55)
        tag = "❌ 缺少必要套件" if required else "⚠️ 缺少聲音模組套件(將降級為純影像模式)"
        print(f"{tag},安裝指令:")
        print(f"   pip install {' '.join(missing)}")
        print("=" * 55)
        if required:
            ans = input("要現在自動安裝嗎?(y/n): ").strip().lower()
            if ans == "y":
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                print("✅ 安裝完成,請重新執行本程式。")
            sys.exit(1)
        return False
    return True

check_packages(REQUIRED_PACKAGES, required=True)
AUDIO_AVAILABLE = check_packages(AUDIO_PACKAGES, required=False)
print("✅ 核心套件檢查通過。" + ("🎤 聲音模組可用。" if AUDIO_AVAILABLE else ""))

import cv2
import math
import time
import requests
import numpy as np
import json
import os
import csv
import sqlite3
import threading
from collections import deque

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import tkinter as tk
    from tkinter import messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ==========================================
# 🔀 1. 模式選擇:即時 or 錄製檔分析
# ==========================================
MODE = "LIVE"          # 💡 "LIVE" = 即時鏡頭+麥克風;"FILE" = 分析錄好的影片

# 📁 路徑一律可用環境變數覆寫,未設定則以本腳本所在資料夾為預設(可攜、不綁單機)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 三端共用門檻(shared/thresholds.json):讓桌面/Pi/ESP32 用同一套數值,避免各自寫死不一致。
def _load_shared_thresholds():
    defaults = {
        "AUDIO_CHOKE_TH_FUSION": 0.15, "AUDIO_CHOKE_TH_SOLO": 0.35,
        "AUDIO_SOLO_CONFIRM": 2, "ALERT_COOLDOWN_SEC": 10.0,
        "AUDIO_SAMPLE_RATE": 16000, "AUDIO_WINDOW_SEC": 0.975, "AUDIO_HOP_SEC": 0.5,
    }
    path = os.environ.get("ANTICHOKE_THRESHOLDS",
                          os.path.join(BASE_DIR, "shared", "thresholds.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults.update({k: v for k, v in data.items() if not k.startswith("_")})
        print(f"✅ 已載入共用門檻:{path}")
    except Exception:
        print("⚠️ 找不到 shared/thresholds.json,使用內建預設門檻。")
    return defaults

SHARED_TH = _load_shared_thresholds()
VIDEO_PATH = os.environ.get("ANTICHOKE_VIDEO",
                            os.path.join(BASE_DIR, "test_eating.mp4"))   # FILE 模式的影片路徑

# ==========================================
# 👥 2. 個案設定
# ==========================================
CURRENT_PATIENT_ID = "A_Grandma"
JSON_PATH = os.environ.get("ANTICHOKE_PATIENTS",
                           os.path.join(BASE_DIR, "patients.json"))

def load_patient_config(patient_id):
    default_config = {
        "name": "未登錄個案",
        "MAR_OPEN_THRESHOLD": 0.15,
        "MAR_CLOSE_THRESHOLD": 0.06,
        "MAX_CHEW_TIME": 4.0,
        "CHEW_WAVE_THRESHOLD": 2.0,
        "SWALLOW_STILL_THRESHOLD": 0.8,
    }
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if patient_id in data:
                print(f"✅ 成功載入【{data[patient_id]['name']}】的個別化參數設定!")
                return {**default_config, **data[patient_id]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ patients.json 格式錯誤 ({e}),改用預設參數。")
    print("⚠️ 找不到設定檔或個案,使用系統預設參數。")
    return default_config

# ==========================================
# 🗄️ 3. SQLite 資料庫
# ==========================================
DB_PATH = os.environ.get("ANTICHOKE_DB",
                         os.path.join(BASE_DIR, "eating_records.db"))

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eating_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, patient_name TEXT, timestamp TEXT,
            chew_count INTEGER, status TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_eating_record(patient_id, patient_name, chew_count, status):
    # 在背景執行緒寫入,避免磁碟 I/O 在嗆咳當下凍結影像/分析主迴圈
    def _write():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cursor.execute("""
                INSERT INTO eating_records (patient_id, patient_name, timestamp, chew_count, status)
                VALUES (?, ?, ?, ?, ?)
            """, (patient_id, patient_name, now_str, int(chew_count), status))
            conn.commit()
            conn.close()
            print(f"💾 【資料庫同步】已寫入 {patient_name} 的進食紀錄 ({status})!")
        except Exception as e:
            print(f"❌ 資料庫寫入失敗: {e}")
    threading.Thread(target=_write, daemon=True).start()

# ==========================================
# 🔒 4. LINE 通知(背景執行緒 + 冷卻)
# ==========================================
LINE_TOKEN = os.environ.get("LINE_TOKEN", "在這裡貼上你的 Channel Access Token")
ALERT_COOLDOWN = SHARED_TH["ALERT_COOLDOWN_SEC"]
_last_alert_time = {}

def _line_worker(message):
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    try:
        res = requests.post(url, json={"messages": [{"type": "text", "text": message}]},
                            headers=headers, timeout=5)
        if res.status_code != 200:
            print(f"❌ LINE 發送失敗: {res.status_code}, {res.text}")
    except Exception as e:
        print(f"❌ LINE 連線異常: {e}")

def send_line_notification(message, alert_type="general"):
    if not LINE_TOKEN or "貼上" in LINE_TOKEN:
        return
    now = time.time()
    if now - _last_alert_time.get(alert_type, 0) < ALERT_COOLDOWN:
        return
    _last_alert_time[alert_type] = now
    threading.Thread(target=_line_worker, args=(message,), daemon=True).start()

# ==========================================
# 🚨 5. UI 緊急警報(橫幅 + 蜂鳴 + 彈窗,非阻塞)
# ==========================================
alert_banner_until = 0.0
alert_banner_text = ""

# Tkinter 對「非主執行緒 + 多個 root 併發」很敏感,用鎖串行化避免連續警報時當掉
_popup_lock = threading.Lock()
_popup_active = False

def trigger_emergency_popup(title, message):
    global alert_banner_until, alert_banner_text, _popup_active
    alert_banner_until = time.time() + 5.0
    alert_banner_text = title

    # 已有彈窗在顯示就跳過(蜂鳴與紅色橫幅仍會提示),避免同時存在多個 Tk root
    with _popup_lock:
        if _popup_active:
            return
        _popup_active = True

    def _worker():
        global _popup_active
        try:
            if HAS_WINSOUND:
                for _ in range(3):
                    winsound.Beep(1500, 300)
            if HAS_TK:
                try:
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showwarning(title, message, parent=root)
                    root.destroy()
                except Exception:
                    pass
        finally:
            with _popup_lock:
                _popup_active = False

    threading.Thread(target=_worker, daemon=True).start()

# ==========================================
# 🎤 6. YAMNet 聲音哽咽偵測模組
# ==========================================
AUDIO_SR = SHARED_TH["AUDIO_SAMPLE_RATE"]        # YAMNet 固定要 16kHz 單聲道
AUDIO_WINDOW_SEC = SHARED_TH["AUDIO_WINDOW_SEC"] # YAMNet 一個分析窗
AUDIO_HOP_SEC = SHARED_TH["AUDIO_HOP_SEC"]       # 每 0.5 秒分析一次
CHOKE_KEYWORDS = ["choking", "gagging", "cough", "gasp", "wheeze", "throat"]
SPEECH_KEYWORDS = ["speech", "conversation", "narration"]  # 用來抑制講話誤判

AUDIO_CHOKE_TH_FUSION = SHARED_TH["AUDIO_CHOKE_TH_FUSION"]  # 融合時聲音只要有一點跡象即可
AUDIO_CHOKE_TH_SOLO = SHARED_TH["AUDIO_CHOKE_TH_SOLO"]      # 純聲音觸發警報的高門檻
AUDIO_SOLO_CONFIRM = SHARED_TH["AUDIO_SOLO_CONFIRM"]        # 純聲音需連續 N 個分析窗達標

yamnet_model = None
choke_class_idx = []
speech_class_idx = []

# 執行緒共享的即時聲音狀態
audio_state = {"choke_score": 0.0, "top_label": "-", "updated": 0.0, "energy": 0.0}
audio_state_lock = threading.Lock()

def load_yamnet():
    """載入 YAMNet 並找出哽咽相關類別的索引"""
    global yamnet_model, choke_class_idx, speech_class_idx
    import tensorflow_hub as hub
    print("⏳ 正在載入 YAMNet 聲音模型(第一次執行需下載約 17MB)...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["display_name"].lower()
            idx = int(row["index"])
            if any(k in name for k in CHOKE_KEYWORDS):
                choke_class_idx.append(idx)
            if any(k in name for k in SPEECH_KEYWORDS):
                speech_class_idx.append(idx)
    print(f"✅ YAMNet 載入完成!哽咽相關類別數: {len(choke_class_idx)}")

def score_waveform(waveform):
    """對一段 16kHz float32 波形計分,回傳 (嗆咳分數, 最高類別名)"""
    scores, _, _ = yamnet_model(waveform)
    mean_scores = scores.numpy().mean(axis=0)
    choke = float(mean_scores[choke_class_idx].sum())
    speech = float(mean_scores[speech_class_idx].sum())
    # 講話聲音會蓋過嗆咳分數 → 適度抑制,減少聊天誤判
    choke = max(0.0, choke - 0.5 * speech)
    top_idx = int(mean_scores.argmax())
    return choke, top_idx, mean_scores

def start_live_audio_thread():
    """LIVE 模式:麥克風背景執行緒,持續更新 audio_state"""
    import sounddevice as sd

    ring = np.zeros(int(AUDIO_SR * 1.0), dtype=np.float32)
    ring_lock = threading.Lock()

    def callback(indata, frames, t, status):
        nonlocal ring
        mono = indata[:, 0].astype(np.float32)
        with ring_lock:
            ring = np.roll(ring, -len(mono))
            ring[-len(mono):] = mono

    def analyzer():
        while True:
            time.sleep(AUDIO_HOP_SEC)
            with ring_lock:
                wave = ring.copy()
            try:
                # 對齊 FILE 模式的視窗長度(AUDIO_WINDOW_SEC),使門檻調校可轉移
                win = int(AUDIO_SR * AUDIO_WINDOW_SEC)
                choke, top_idx, _ = score_waveform(wave[-win:])
                energy = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
                with audio_state_lock:
                    audio_state["choke_score"] = choke
                    audio_state["top_label"] = str(top_idx)
                    audio_state["energy"] = energy      # S1: 整體音量能量
                    audio_state["updated"] = time.time()
            except Exception as e:
                print(f"⚠️ 聲音分析異常: {e}")

    stream = sd.InputStream(samplerate=AUDIO_SR, channels=1,
                            blocksize=int(AUDIO_SR * 0.1), callback=callback)
    stream.start()
    threading.Thread(target=analyzer, daemon=True).start()
    print("🎤 即時麥克風聲音監測已啟動!")
    return stream

def precompute_file_audio_scores(video_path):
    """FILE 模式:用 ffmpeg 抽出音軌,預先算好整段影片每 0.5 秒的嗆咳分數"""
    import soundfile as sf
    wav_path = os.path.join(os.path.dirname(video_path) or ".", "_tmp_audio_16k.wav")
    print("⏳ 正在用 ffmpeg 抽取影片音軌...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(AUDIO_SR),
             "-vn", wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ 找不到 ffmpeg 或抽取失敗!FILE 模式將以純影像分析。")
        print("   請至 https://ffmpeg.org 下載並加入系統 PATH。")
        return None

    wave, sr = sf.read(wav_path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    hop = int(AUDIO_SR * AUDIO_HOP_SEC)
    win = int(AUDIO_SR * AUDIO_WINDOW_SEC)
    timeline = []   # [(秒數, 嗆咳分數), ...]
    print("⏳ 正在對整段音軌進行 YAMNet 嗆咳分析...")
    for start in range(0, max(1, len(wave) - win), hop):
        seg = wave[start:start + win]
        choke, _, _ = score_waveform(seg)
        timeline.append((start / AUDIO_SR, choke))
    print(f"✅ 音軌分析完成,共 {len(timeline)} 個分析窗。")
    try:
        os.remove(wav_path)
    except OSError:
        pass
    return timeline

def lookup_file_audio_score(timeline, t_sec):
    """依影片播放時間查詢對應的聲音嗆咳分數"""
    if not timeline:
        return 0.0
    idx = min(max(int(t_sec / AUDIO_HOP_SEC), 0), len(timeline) - 1)  # 夾 0..end,防負索引繞尾
    return timeline[idx][1]

# ==========================================
# 🎛️ 7. 狀態機常數與訊號處理參數(模組層級常數,run_loop 直接讀取)
# ==========================================
ST_IDLE, ST_INGEST, ST_CHEW, ST_SWALLOW, ST_CHECK = "IDLE", "INGEST", "CHEW", "SWALLOW", "CHECK"

EMA_ALPHA_MAR = 0.4
EMA_ALPHA_JAW = 0.5
REF_IOD_PX = 100.0
MIN_CHEW_INTERVAL = 0.18
COUGH_STD_TH = 15.0
COUGH_VEL_TH = 8.0
COUGH_CONFIRM_FRAMES = 3
FACE_LOST_ALERT_SEC = 3.0

# --- S1 無聲窒息 ---
SILENT_HOLD_SEC = 4.0          # 嘴開+靜止+無聲持續多久算無聲窒息
SILENT_QUIET_TH = 0.02         # 音訊能量低於此視為「無聲」(0~1)
# --- S2 發紺 ---
CYANOSIS_BLUE_TH = 0.40        # 嘴唇藍佔比門檻(需現場校正)
# --- S3 時間窗多模態融合 ---
FUSION_WINDOW_SEC = 1.8
FUSION_THRESHOLD = 1.0
FUSION_COOLDOWN_SEC = 10.0
W_AUDIO = 2.0                  # audio_choke 乘權(0.3 分 -> 0.6)
W_GESTURE = 0.6               # 手抓喉嚨(高特異性)
W_CYANOSIS = 0.7             # 發紺(高特異性)
W_STILL_OPEN = 0.3           # 嘴開+靜止(低特異性)

# ==========================================
# 🤖 8. MediaPipe FaceLandmarker(Tasks API)
#    新版 mediapipe(>=0.10.x)移除了舊的 solutions.face_mesh,
#    改用 Tasks API 的 FaceLandmarker(同一套 478 點網格,索引不變)。
# ==========================================
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# 臉部模型檔(可用環境變數覆寫,預設放程式資料夾)
FACE_MODEL_PATH = os.environ.get("ANTICHOKE_FACE_MODEL",
                                 os.path.join(BASE_DIR, "face_landmarker.task"))
# 手部模型檔(A1 手抓喉嚨手勢偵測用)
HAND_MODEL_PATH = os.environ.get("ANTICHOKE_HAND_MODEL",
                                 os.path.join(BASE_DIR, "hand_landmarker.task"))
import gesture           # 純幾何,判斷手是否接近喉嚨
import silent_choke      # S1: 無聲窒息偵測
import evidence_fusion   # S3: 時間窗多模態融合
import cyanosis          # S2: 嘴唇發紺

# ==========================================
# 🧩 9. 啟動用工廠函數(把原本的模組層級副作用收攏成可呼叫的函數)
# ==========================================
def setup_audio(mode, video_path, audio_available):
    """初始化聲音模組。回傳 (mic_stream, file_audio_timeline, audio_available)。
    載入失敗時自動把 audio_available 降級為 False。"""
    mic_stream = None
    file_audio_timeline = None
    if audio_available:
        try:
            load_yamnet()
            if mode == "LIVE":
                mic_stream = start_live_audio_thread()
            else:
                file_audio_timeline = precompute_file_audio_scores(video_path)
        except Exception as e:
            print(f"⚠️ 聲音模組初始化失敗 ({e}),降級為純影像模式。")
            audio_available = False
    return mic_stream, file_audio_timeline, audio_available

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def ensure_face_model():
    """模型檔不存在就自動下載(對應舊版 FaceMesh 自動下載模型的行為)。"""
    if os.path.exists(FACE_MODEL_PATH):
        return
    print("【提示】首次執行:下載臉部模型 face_landmarker.task(約 3.7MB)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print("【完成】臉部模型下載完成。")
    except Exception as e:
        print(f"【錯誤】臉部模型下載失敗({e})。")
        print(f"        請手動下載放到:{FACE_MODEL_PATH}")
        print(f"        來源:{FACE_MODEL_URL}")
        sys.exit(1)

def create_face_landmarker():
    """建立 MediaPipe FaceLandmarker(Tasks API,VIDEO 模式含追蹤)。
    需要 face_landmarker.task 模型檔(478 點,等同舊版 refine_landmarks=True)。"""
    ensure_face_model()
    model_buffer = open(FACE_MODEL_PATH, "rb").read()
    base_options = mp_tasks.BaseOptions(model_asset_buffer=model_buffer)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.6)
    return mp_vision.FaceLandmarker.create_from_options(options)

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def create_hand_landmarker():
    """建立 MediaPipe HandLandmarker(Tasks API,VIDEO 模式)。A1 手勢偵測用。
    模型檔不存在就自動下載。回傳 landmarker,失敗回 None(降級:不做手勢偵測)。"""
    if not os.path.exists(HAND_MODEL_PATH):
        print("【提示】首次執行:下載手部模型 hand_landmarker.task(約 7.5MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
            print("【完成】手部模型下載完成。")
        except Exception as e:
            print(f"【警告】手部模型下載失敗({e}),停用手勢偵測。")
            return None
    try:
        model_buffer = open(HAND_MODEL_PATH, "rb").read()
        base_options = mp_tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        return mp_vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"【警告】手部模型初始化失敗({e}),停用手勢偵測。")
        return None

def open_capture(mode, video_path):
    """依模式開啟影像來源。回傳 (cap, video_fps)。失敗時 sys.exit(1)。"""
    if mode == "FILE":
        if not os.path.exists(video_path):
            print(f"❌ 找不到影片檔:{video_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # NaN 是 truthy 會漏過 `or`;負值/0/NaN/Inf 一律退回 30(否則 int(nan) 崩潰)
        if not (isinstance(video_fps, (int, float)) and math.isfinite(video_fps) and video_fps > 0):
            video_fps = 30.0
        print(f"📼 錄製檔分析模式啟動:{video_path} (FPS={video_fps:.1f})")
        return cap, video_fps

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟鏡頭!請確認沒有其他程式佔用。")
        sys.exit(1)
    print("🚀 即時鏡頭啟動成功!請將臉對準鏡頭...")
    return cap, 30.0     # LIVE 模式不使用 video_fps,給預設值

# ==========================================
# 🔁 10. 主迴圈(所有迴圈狀態都是本函數的區域變數)
#     參數刻意沿用 CONFIG / AUDIO_AVAILABLE / MODE 等原名,
#     使迴圈內每一行判斷邏輯維持不變。
# ==========================================
def run_loop(cap, face_landmarker, hand_landmarker, CONFIG, AUDIO_AVAILABLE, MODE, video_fps, file_audio_timeline):
    # ---- 迴圈區域狀態(原本散在模組層級,收進函數避免全域污染)----
    current_state = ST_IDLE
    state_start_time = time.time()

    prev_jaw_y = None
    jaw_movement_history = deque(maxlen=30)   # 自動汰舊,append O(1),免手動 pop(0)
    chew_count = 0

    mar_smooth = None
    jaw_smooth = None
    last_cough_time = 0.0
    hand_on_neck_start_time = None
    choke_double_start_time = None
    last_choke_alert_time = 0.0
    prev_nose = None
    nose_speed_history = deque(maxlen=30)
    last_hand_landmarks = []
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]
    
    last_chew_flip_time = 0.0
    cough_frame_counter = 0
    audio_solo_counter = 0
    face_lost_since = None
    frame_idx = 0
    last_ts_ms = 0            # FaceLandmarker VIDEO 模式需嚴格遞增的時間戳(ms)
    silent_detector = silent_choke.SilentChokeDetector(SILENT_HOLD_SEC)   # S1
    evfusion = evidence_fusion.EvidenceFusion(                            # S3
        FUSION_WINDOW_SEC, FUSION_THRESHOLD, FUSION_COOLDOWN_SEC)
    last_nose = None              # #2: 快取喉嚨錨點,臉短暫消失時手勢仍可運作
    last_chin = None
    last_anchor_ts = 0.0
    ANCHOR_GRACE_SEC = 2.0

    def handle_gesture(nose_xy, chin_xy):
        """偵測手抓喉嚨與胸前區域，回傳 (near_throat, near_chest)。"""
        nonlocal last_hand_landmarks
        if hand_landmarker is None:
            last_hand_landmarks = []
            return False, False
        hres = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        last_hand_landmarks = hres.hand_landmarks
        hnds = [[(pt.x, pt.y) for pt in hand] for hand in hres.hand_landmarks]
        near_throat = gesture.hand_near_throat(hnds, nose_xy, chin_xy, w, h)
        near_chest = gesture.hand_near_chest(hnds, nose_xy, chin_xy, w, h)
        return near_throat, near_chest

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if MODE == "FILE":
                print("📼 影片分析完畢。")
            else:
                print("❌ 無法讀取鏡頭畫面。")
            break

        if MODE == "LIVE":
            frame = cv2.flip(frame, 1)
            current_time = time.time()
        else:
            frame_idx += 1
            # FILE 模式優先用實際時間戳(POS_MSEC),對變動幀率(VFR)影片才不會漂移;
            # 無效時退回 frame_idx/fps
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if isinstance(pos_ms, (int, float)) and math.isfinite(pos_ms) and pos_ms > 0:
                current_time = pos_ms / 1000.0
            else:
                current_time = frame_idx / video_fps

        h, w, _ = frame.shape

        # ---- 取得目前聲音嗆咳分數 ----
        audio_stale = False
        if AUDIO_AVAILABLE:
            if MODE == "LIVE":
                with audio_state_lock:
                    fresh = (time.time() - audio_state["updated"]) < 1.5
                    audio_choke = audio_state["choke_score"] if fresh else 0.0
                    audio_energy = audio_state["energy"] if fresh else 1.0
                audio_stale = not fresh    # 分析執行緒停擺 -> 聲音偵測靜默失效,須顯示
            else:
                audio_choke = lookup_file_audio_score(file_audio_timeline, current_time)
                audio_energy = 1.0     # FILE 模式無即時能量,不做無聲窒息(S1)
        else:
            audio_choke = 0.0
            audio_energy = 1.0         # 無音訊 -> 不判無聲窒息(避免誤報)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Tasks API: 包成 mp.Image,VIDEO 模式需嚴格遞增時間戳
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(current_time * 1000)
        if timestamp_ms <= last_ts_ms:
            timestamp_ms = last_ts_ms + 1
        last_ts_ms = timestamp_ms
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_detected = bool(results.face_landmarks)
        movement_std = 0.0
        mar = 0.0
        gesture_now = False
        vision_cough = False
        body_shaking = False

        if face_detected:
            face_lost_since = None
            # Tasks API 回傳扁平 list(每點有 .x/.y/.z),索引與舊版一致
            face_landmarks = results.face_landmarks[0]
            p13 = face_landmarks[13]
            p14 = face_landmarks[14]
            p78 = face_landmarks[78]
            p308 = face_landmarks[308]
            p152 = face_landmarks[152]
            p4 = face_landmarks[4]
            p33 = face_landmarks[33]
            p263 = face_landmarks[263]

            v_dist = math.hypot(p13.x - p14.x, p13.y - p14.y) * h
            h_dist = math.hypot(p78.x - p308.x, p78.y - p308.y) * w
            mar_raw = v_dist / h_dist if h_dist > 0 else 0
            # 防 NaN/Inf 永久污染 EMA:壞值直接沿用上一幀(或 0),避免 FSM 永久凍結
            if not math.isfinite(mar_raw):
                mar_raw = mar_smooth if mar_smooth is not None else 0.0
            mar_smooth = mar_raw if mar_smooth is None else (
                EMA_ALPHA_MAR * mar_raw + (1 - EMA_ALPHA_MAR) * mar_smooth)
            mar = mar_smooth

            iod = math.hypot(p33.x - p263.x, p33.y - p263.y) * w
            scale = REF_IOD_PX / iod if iod > 1 else 1.0
            jaw_raw = (p152.y - p4.y) * h * scale
            if not math.isfinite(jaw_raw):
                jaw_raw = jaw_smooth if jaw_smooth is not None else 0.0
            jaw_smooth = jaw_raw if jaw_smooth is None else (
                EMA_ALPHA_JAW * jaw_raw + (1 - EMA_ALPHA_JAW) * jaw_smooth)
            jaw_relative_y = jaw_smooth

            jaw_movement_history.append(jaw_relative_y)   # deque(maxlen=30) 自動汰舊

            jaw_velocity = 0.0
            if prev_jaw_y is not None:
                jaw_velocity = jaw_relative_y - prev_jaw_y
            prev_jaw_y = jaw_relative_y

            if len(jaw_movement_history) >= 15:
                # deque 不支援切片,需先轉 list 再取最後 15 筆
                movement_std = float(np.std(list(jaw_movement_history)[-15:]))

            # ==========================================
            # A1. 手抓喉嚨手勢偵測(國際窒息手勢,獨立於狀態機、任何狀態皆檢查)
            # ==========================================
            last_nose = (p4.x, p4.y)          # #2: 快取錨點供臉短暫消失時續用
            last_chin = (p152.x, p152.y)
            last_anchor_ts = current_time
            gesture_now, hand_chest_now = handle_gesture(last_nose, last_chin)

            # --- 影像嗆咳跡象(連續幀確認)---
            vision_cough = False
            if movement_std > COUGH_STD_TH and abs(jaw_velocity) > COUGH_VEL_TH:
                cough_frame_counter += 1
            else:
                cough_frame_counter = 0
            if cough_frame_counter >= COUGH_CONFIRM_FRAMES:
                cough_frame_counter = 0
                vision_cough = True

            # --- 身體劇烈晃動偵測 (struggling) ---
            face_size = math.hypot(p4.x - p152.x, p4.y - p152.y)
            if prev_nose is not None and face_size > 0.001:
                displacement = math.hypot(p4.x - prev_nose[0], p4.y - prev_nose[1])
                norm_disp = displacement / face_size
                nose_speed_history.append(norm_disp)
                if sum(nose_speed_history) > 1.5:
                    body_shaking = True
            prev_nose = (p4.x, p4.y)

            # ==========================================
            # S1. 無聲窒息 & S3. 多模態融合 (已停用彈窗，依需求一律以 5秒劇烈晃動+胸口/脖子手勢 為唯一哽噎判定)
            # ==========================================
            # if AUDIO_AVAILABLE and silent_detector.update(...): pass
            # if evfusion.check(): pass

            # ==========================================
            # 🧠 11. 狀態機 + 多模態融合判斷
            # ==========================================
            if current_state == ST_IDLE:
                if mar > CONFIG["MAR_OPEN_THRESHOLD"]:
                    current_state = ST_INGEST
                    state_start_time = current_time
                    chew_count = 0
                    print("【通知】偵測到張嘴:食物入口 🍛")

            elif current_state == ST_INGEST:
                if mar < CONFIG["MAR_CLOSE_THRESHOLD"]:
                    current_state = ST_CHEW
                    state_start_time = current_time
                    print("【通知】開始閉嘴咀嚼食物 🦷")

            elif current_state == ST_CHEW:
                if movement_std > CONFIG["CHEW_WAVE_THRESHOLD"] and abs(jaw_velocity) > 1.5:
                    if len(jaw_movement_history) >= 3:
                        d1 = jaw_movement_history[-1] - jaw_movement_history[-2]
                        d2 = jaw_movement_history[-2] - jaw_movement_history[-3]
                        if d1 * d2 < 0 and (current_time - last_chew_flip_time) > MIN_CHEW_INTERVAL:
                            chew_count += 0.5
                            last_chew_flip_time = current_time

                # --- 影像嗆咳跡象(連續幀確認)---
                if vision_cough:
                    state_start_time = current_time

                # --- 吞嚥判定優先: 咀嚼後靜止 = 正常吞嚥,不可誤判為卡喉 ---
                elif movement_std < CONFIG["SWALLOW_STILL_THRESHOLD"] \
                        and len(jaw_movement_history) >= 15:
                    current_state = ST_SWALLOW
                    state_start_time = current_time
                    print("【通知】咀嚼停止,下巴上提(定格吞嚥中...)")

            elif current_state == ST_SWALLOW:
                if (current_time - state_start_time) > CONFIG["SWALLOW_DURATION"]:
                    current_state = ST_CHECK
                    state_start_time = current_time
                    print(f"🎉【數據分析】吞嚥成功!本次咀嚼約 {int(chew_count)} 次。進入安全期。")
                    save_eating_record(CURRENT_PATIENT_ID, CONFIG.get("name", "未知個案"),
                                       chew_count, "吞嚥成功")
                elif movement_std > CONFIG["CHEW_WAVE_THRESHOLD"]:
                    current_state = ST_CHEW
                    print("【狀態回退】非吞嚥,恢復咀嚼。")

            elif current_state == ST_CHECK:
                if (current_time - state_start_time) > CONFIG["SAFETY_CHECK_DURATION"]:
                    print("💖【數據分析】安全通過進食觀察期。")
                    current_state = ST_IDLE

            # ---- UI:每幀繪製 ----
            # 1. 繪製嘴部輪廓
            outer_pts = []
            for idx in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]:
                pt = face_landmarks[idx]
                outer_pts.append((int(pt.x * w), int(pt.y * h)))
            for i in range(len(outer_pts)):
                cv2.line(frame, outer_pts[i], outer_pts[(i + 1) % len(outer_pts)], (0, 0, 255), 1)

            # 2. 繪製嘴部與眼睛中心點
            for idx in [13, 14, 78, 308, 4, 152]:
                pt = face_landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 0), -1)
            cv2.line(frame, (int(p13.x * w), int(p13.y * h)),
                     (int(p14.x * w), int(p14.y * h)), (255, 0, 0), 1)
            cv2.line(frame, (int(p4.x * w), int(p4.y * h)),
                     (int(p152.x * w), int(p152.y * h)), (0, 0, 255), 2)

            # 3. 繪製喉嚨偵測圈圈 (脖子範圍)
            tx = p152.x + 0.6 * (p152.x - p4.x)
            ty = p152.y + 0.6 * (p152.y - p4.y)
            cx_px, cy_px = int(tx * w), int(ty * h)
            r_px = int(face_size * 0.6 * w)
            cv2.circle(frame, (cx_px, cy_px), r_px, (255, 0, 255), 2)
            cv2.circle(frame, (cx_px, cy_px), 4, (255, 0, 255), -1)
            cv2.putText(frame, "Throat Zone", (cx_px - 40, cy_px - r_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if current_state == ST_SWALLOW:
                cv2.putText(frame, "SWALLOWING...", (20, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (140, int(h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # #2: 臉短暫消失時(如病人低頭嗆咳),若手仍可見,用快取錨點續抓手勢
            gesture_now = False
            hand_chest_now = False
            if last_nose is not None and (current_time - last_anchor_ts) < ANCHOR_GRACE_SEC:
                gesture_now, hand_chest_now = handle_gesture(last_nose, last_chin)
            prev_nose = None
            nose_speed_history.clear()
            body_shaking = False
            if MODE == "LIVE" and current_state in (ST_INGEST, ST_CHEW, ST_SWALLOW):
                if face_lost_since is None:
                    face_lost_since = current_time
                elif current_time - face_lost_since > FACE_LOST_ALERT_SEC:
                    lost_msg = "⚠️【異常警報】進食過程中人臉消失,請確認個案狀況!"
                    print(lost_msg)
                    send_line_notification(lost_msg, alert_type="face_lost")
                    trigger_emergency_popup("⚠️ 人臉消失警報", lost_msg)
                    face_lost_since = current_time

        # ==========================================
        # 🚨 12. 哽噎與嗆咳決策判斷
        #     - 哽咽：身體搖晃 ＋ 手在胸前核心區域 雙重動作同時持續 5 秒 -> 緊急哽噎警報
        #     - 嗆咳：咳嗽/嗆咳僅紀錄至後台 SQLite 資料庫，不跳彈窗、不閃紅橫幅、不發 LINE 通知
        # ==========================================
        # ---- 追蹤哽咽雙重動作 (動作一：身體前後左右搖晃 ＋ 動作二：胸前手勢) ----
        choke_double_active = body_shaking and (hand_chest_now or gesture_now)

        if choke_double_active:
            if choke_double_start_time is None:
                choke_double_start_time = current_time
            choke_double_duration = current_time - choke_double_start_time
        else:
            choke_double_start_time = None
            choke_double_duration = 0.0

        # ---- 追蹤咳嗽/嗆咳偵測狀態 (聲音或影像) ----
        cough_detected_this_frame = False
        if vision_cough:
            cough_detected_this_frame = True

        audio_cough_detected = False
        if AUDIO_AVAILABLE:
            if audio_choke > AUDIO_CHOKE_TH_SOLO:
                audio_solo_counter += 1
            else:
                audio_solo_counter = 0
            if audio_solo_counter >= AUDIO_SOLO_CONFIRM:
                audio_solo_counter = 0
                audio_cough_detected = True
            elif face_detected and audio_choke > AUDIO_CHOKE_TH_FUSION:
                audio_cough_detected = True

        if audio_cough_detected:
            cough_detected_this_frame = True

        if cough_detected_this_frame:
            last_cough_time = current_time
            # 僅靜默寫入後台 SQLite 資料庫 (每 3.0 秒最多紀錄一次)，不跳警報、不閃紅橫幅、不發 LINE
            if current_time - _last_alert_time.get("cough_db_log", 0) >= 3.0:
                _last_alert_time["cough_db_log"] = current_time
                save_eating_record(CURRENT_PATIENT_ID, CONFIG.get("name", "未知個案"),
                                   chew_count, "嗆咳紀錄")

        # ---- 警報決策：哽咽緊急警報 (雙重動作同時持續滿 5.0 秒) ----
        if choke_double_duration >= 5.0:
            if current_time - last_choke_alert_time >= FUSION_COOLDOWN_SEC:
                last_choke_alert_time = current_time
                gmsg = "🚨【緊急・哽噎警報】偵測到身體連續搖晃與手部撫胸/拍胸雙重動作同時持續 5.0 秒，判定為急性哽噎！請立即協助！"
                print(gmsg)
                send_line_notification(gmsg, alert_type="choke_emergency")
                trigger_emergency_popup("緊急哽噎警報", gmsg)
                save_eating_record(CURRENT_PATIENT_ID, CONFIG.get("name", "未知個案"),
                                   chew_count, "緊急哽噎警報(雙重動作5秒)")
                choke_double_start_time = None
                choke_double_duration = 0.0

        # ---- 共用資訊面板 ----
        mode_tag = "LIVE" if MODE == "LIVE" else "FILE"
        cv2.putText(frame, f"Patient: {CONFIG.get('name', 'Unknown')}  [{mode_tag}]", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        cv2.putText(frame, f"STATE: {current_state}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Jaw Std: {movement_std:.2f}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Chew Count: {int(chew_count)}", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

        # ---- 聲音嗆咳分數即時視覺化(顏色隨危險程度變化 + 分數條)----
        if AUDIO_AVAILABLE and audio_stale:
            # 分析執行緒停擺 -> 聲音偵測已靜默失效,醒目提示操作員
            cv2.putText(frame, "! AUDIO STALE - sound detection DOWN !", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif AUDIO_AVAILABLE:
            bar_color = (0, 255, 0)
            if audio_choke > AUDIO_CHOKE_TH_SOLO:
                bar_color = (0, 0, 255)
            elif audio_choke > AUDIO_CHOKE_TH_FUSION:
                bar_color = (0, 165, 255)
            cv2.putText(frame, f"Audio Choke: {audio_choke:.2f}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
            bar_len = int(min(audio_choke / 0.6, 1.0) * 200)
            cv2.rectangle(frame, (200, 198), (200 + bar_len, 212), bar_color, -1)
            cv2.rectangle(frame, (200, 198), (400, 212), (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Audio: OFF (vision only)", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

        # ---- 緊急警報紅色閃爍橫幅 ----
        if time.time() < alert_banner_until:
            if int(time.time() * 4) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(frame, "!!! EMERGENCY ALERT !!!", (int(w * 0.18), 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        # ---- 繪製手部骨架 ----
        for hand in last_hand_landmarks:
            for pt in hand:
                px, py = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)  # Yellow joints
            for connection in HAND_CONNECTIONS:
                pt1 = hand[connection[0]]
                pt2 = hand[connection[1]]
                p1x, p1y = int(pt1.x * w), int(pt1.y * h)
                p2x, p2y = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 200, 255), 2)  # Orange lines

        cv2.imshow("Anti-Choking Multimodal System v4.2", frame)
        # FILE 模式依 FPS 控制播放速度;LIVE 模式維持最快更新
        wait_ms = 1 if MODE == "LIVE" else max(1, int(1000 / video_fps))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("🛑 使用者關閉系統。")
            break

# ==========================================
# 🚀 13. 程式進入點
# ==========================================
def main():
    # 個案參數(執行期參數在此補上,與原本模組層級行為一致)
    config = load_patient_config(CURRENT_PATIENT_ID)
    config["SWALLOW_DURATION"] = 0.6
    config["SAFETY_CHECK_DURATION"] = 4.0

    init_database()

    # 先做「可能 sys.exit 的初始化」(模型/鏡頭),全部成功後才啟動麥克風執行緒,
    # 否則早期失敗會跳過 finally、留下孤兒麥克風執行緒。
    face_landmarker = create_face_landmarker()   # 模型下載失敗會 sys.exit
    hand_landmarker = create_hand_landmarker()   # A1: 手勢偵測(失敗回 None 自動停用)
    cap, video_fps = open_capture(MODE, VIDEO_PATH)   # 開不了鏡頭會 sys.exit
    # 影像/模型就緒後才啟動麥克風(含背景分析執行緒)
    mic_stream, file_audio_timeline, audio_available = setup_audio(
        MODE, VIDEO_PATH, AUDIO_AVAILABLE)

    try:
        run_loop(cap, face_landmarker, hand_landmarker, config, audio_available,
                 MODE, video_fps, file_audio_timeline)
    finally:
        # ---- 收尾:釋放所有資源,避免麥克風/攝影機被佔用 ----
        cap.release()
        if mic_stream is not None:
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass
        try:
            face_landmarker.close()
        except Exception:
            pass
        if hand_landmarker is not None:
            try:
                hand_landmarker.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
