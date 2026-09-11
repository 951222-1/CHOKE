# -*- coding: utf-8 -*-
"""
多模態緊急警報與融合決策 (Multi-Modal Alert & Evidence Fusion) 獨立學習模組
包含：
1. EvidenceFusion 多模態證據融合評分器 (時域滑動視窗 + 權重累加 + 冷卻機制)
2. 視覺警報：OpenCV 螢幕紅色閃烁橫幅 (Flashing Banner Overlay)
3. 聽覺警報：winsound 非阻塞聲音蜂鳴 (Audio Beep)
4. GUI 彈窗警報：Tkinter 跨執行緒防鎖定跳窗 (Topmost Popup Box)
5. 遠端推送警報：LINE Notify / Webhook 異步冷卻發送 (Async Line Broadcast)
6. 資料庫紀錄：SQLite 警報事件紀錄存檔 (Database Logging)
7. 完整測試 Main 流程
"""

import os
import sys
import time
import sqlite3
import threading
import requests
import cv2
import numpy as np

# 音效與 GUI 套件相容性檢查
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


# =============================================================================
# 1. EvidenceFusion (多模態證據融合評分器)
# =============================================================================
class EvidenceFusion:
    """
    把多個微弱或分散的線索 (如聲音嗆咳、手抓喉嚨、嘴開靜止、嘴唇發紫) 綁在一起評分。
    當近幾秒內累積總分超過門檻，且達到冷卻時間，才觸發警報，大幅減少誤報。
    """
    def __init__(self, window_sec=3.0, threshold=0.8, cooldown_sec=10.0, clock=time.time):
        self.window = window_sec       # 時間窗長度 (秒)
        self.threshold = threshold     # 警報觸發分數門檻
        self.cooldown = cooldown_sec   # 警報觸發後的冷卻時間 (秒)
        self.clock = clock
        self.sources = {}              # 保存各來源最新時間點與分數: {source: (timestamp, weight)}
        self.last_fire = -1e9          # 上次警報觸發時間點

    def observe(self, source, weight):
        """更新/記錄單一訊號源的分數與發生時間 (同訊號只保留最新一次)"""
        if weight > 0:
            self.sources[source] = (self.clock(), weight)

    def score(self):
        """計算時間窗內未過期線索的總分"""
        now = self.clock()
        # 移除過期的線索
        self.sources = {k: (t, w) for k, (t, w) in self.sources.items()
                        if now - t <= self.window}
        return sum(w for (_, w) in self.sources.values())

    def check(self):
        """檢查是否達到觸發標準 (總分 >= 門檻 且 已過冷卻期)"""
        now = self.clock()
        if self.score() >= self.threshold and (now - self.last_fire >= self.cooldown):
            self.last_fire = now
            return True
        return False


# =============================================================================
# 2. 遠端 LINE 通報系統 (異步 + 冷卻機制)
# =============================================================================
class LineAlertSystem:
    def __init__(self, token=None, cooldown_sec=10.0):
        self.token = token or os.environ.get("LINE_TOKEN", "")
        self.cooldown = cooldown_sec
        self.last_alert_times = {}

    def _worker(self, message):
        if not self.token:
            print(f"[LINE 模擬發送]: {message}")
            return
        url = "https://api.line.me/v2/bot/message/broadcast"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        try:
            res = requests.post(url, json={"messages": [{"type": "text", "text": message}]},
                                headers=headers, timeout=5)
            if res.status_code == 200:
                print("✅ LINE 通報發送成功。")
            else:
                print(f"❌ LINE 發送失敗: {res.status_code}, {res.text}")
        except Exception as e:
            print(f"❌ LINE 連線失敗: {e}")

    def send(self, message, alert_type="general"):
        now = time.time()
        if now - self.last_alert_times.get(alert_type, 0) < self.cooldown:
            return  # 在冷卻期內，忽略發送

        self.last_alert_times[alert_type] = now
        # 使用背景執行緒發送，避免阻塞主 UI 畫面
        threading.Thread(target=self._worker, args=(message,), daemon=True).start()


# =============================================================================
# 3. 多重警報管理器 (UI 閃光橫幅 + 蜂鳴 + 彈窗)
# =============================================================================
class AlertManager:
    def __init__(self, db_path="eating_records.db"):
        self.banner_until = 0.0
        self.banner_text = ""
        self.popup_lock = threading.Lock()
        self.popup_active = False
        self.line_system = LineAlertSystem()
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化警報日誌 SQLite 資料庫"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    patient_id TEXT,
                    alert_title TEXT,
                    alert_message TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"資料庫初始化失敗: {e}")

    def log_to_db(self, patient_id, title, message):
        """紀錄警報事件至資料庫"""
        def _db_worker():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alert_logs (timestamp, patient_id, alert_title, alert_message)
                    VALUES (datetime('now', 'localtime'), ?, ?, ?)
                ''', (patient_id, title, message))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"紀錄警報日誌失敗: {e}")
        threading.Thread(target=_db_worker, daemon=True).start()

    def log_silent_cough(self, patient_id="PATIENT_001", message="偵測到嗆咳（聲音咳嗽或影像頭部抽動）"):
        """
        嗆咳事件（原 L1 / L2 合併）：
        取消劇烈晃動條件，僅於背景寫入 SQLite 資料庫（標籤為 嗆咳紀錄），
        不會跳出 UI 彈窗、不閃紅橫幅、不發出蜂鳴聲與不發送 LINE。
        """
        title = "嗆咳紀錄"
        print(f"🤫 【後台靜默紀錄】已寫入 SQLite 資料庫 -> 標籤: [{title}], 內容: {message}")
        self.log_to_db(patient_id, title, message)

    def trigger_alert(self, title, message, patient_id="PATIENT_001"):
        """觸發完整警報：閃光橫幅 + 音效 + 彈窗 + LINE + 資料庫紀錄"""
        print(f"\n🚨 【緊急警報發起】{title}: {message}")
        self.banner_until = time.time() + 4.0  # 橫幅持續 4 秒
        self.banner_text = title

        # 紀錄至資料庫與發送 LINE 通報
        self.log_to_db(patient_id, title, message)
        self.line_system.send(f"🚨【{title}】\n{message}")

        # 避免同時打開多個 Tkinter 彈窗造成 GUI 死鎖
        with self.popup_lock:
            if self.popup_active:
                return
            self.popup_active = True

        def _worker():
            try:
                # 1. 聽覺警報 (Beep 嗶嗶聲)
                if HAS_WINSOUND:
                    for _ in range(3):
                        winsound.Beep(1500, 200)

                # 2. GUI 彈窗警報
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
                with self.popup_lock:
                    self.popup_active = False

        threading.Thread(target=_worker, daemon=True).start()

    def draw_alert_banner(self, frame):
        """在 OpenCV 畫面上繪製紅色閃爍緊急橫幅"""
        if time.time() < self.banner_until:
            # 每 0.25 秒閃爍一次
            if int(time.time() * 4) % 2 == 0:
                h, w, _ = frame.shape
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                cv2.putText(frame, f"!!! EMERGENCY: {self.banner_text} !!!", (int(w * 0.1), 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)


def main():
    print("🚀 啟動警報與多模態融合展示模組...")
    alert_mgr = AlertManager()
    fusion = EvidenceFusion(window_sec=3.0, threshold=0.8, cooldown_sec=5.0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟預設攝影機")
        return

    print("🎥 視窗說明：")
    print(" - 按 '1' 鍵：模擬輸入聲音嗆咳線索 (+0.4 分)")
    print(" - 按 '2' 鍵：模擬輸入手抓喉嚨線索 (+0.5 分)")
    print(" - 按 '3' 鍵：模擬輸入嘴唇發紺缺氧線索 (+0.6 分)")
    print(" - 按 '4' 鍵：模擬咳嗽事件 (靜默寫入 SQLite 標籤「嗆咳紀錄」，無彈窗、無紅橫幅)")
    print(" - 按 'q' 鍵：退出程式")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 取得當前多模態融合總分
        current_score = fusion.score()

        # 檢查是否達到觸發警報條件
        if fusion.check():
            alert_mgr.trigger_alert("窒息多模態融合警報",
                                   f"綜合多模態分數達 {current_score:.2f} (>= 0.8)，判定為窒息危急狀態！")

        # 繪製警報橫幅 (若在警報時間內)
        alert_mgr.draw_alert_banner(frame)

        # 顯示當前融合分數資訊
        cv2.putText(frame, f"Evidence Fusion Score: {current_score:.2f} / 0.80", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press '1': +Audio, '2': +Gesture, '3': +Cyanosis, '4': Silent Cough Log", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Multi-Modal Alert & Fusion Engine Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            print("🎤 [模擬輸入] 聲音嗆咳訊號 (+0.4)")
            fusion.observe("audio", 0.4)
        elif key == ord('2'):
            print("✋ [模擬輸入] 手抓喉嚨手勢 (+0.5)")
            fusion.observe("gesture", 0.5)
        elif key == ord('3'):
            print("👄 [模擬輸入] 嘴唇發紺缺氧 (+0.6)")
            fusion.observe("cyanosis", 0.6)
        elif key == ord('4'):
            print("🤫 [模擬輸入] 偵測到咳嗽（聲音或影像抽動）-> 執行靜默後台紀錄（SQLite 標籤「嗆咳紀錄」）")
            alert_mgr.log_silent_cough("PATIENT_001", "模擬咳嗽/頭部抽動偵測，背景靜默存檔")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
