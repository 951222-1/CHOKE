# -*- coding: utf-8 -*-
"""
PC 即時 MJPEG 影像串流伺服器
這支程式會在電腦上啟動 HTTP 伺服器 (Port 80)，擷取本機 webcam 畫面並推送至 http://192.168.68.54/stream。
讓樹莓派 (Raspberry Pi) 可以直接連線並獲取即時影像進行防哽咽辨識！
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 全域最新影像 Frame 暫存與鎖
latest_jpeg = None
frame_lock = threading.Lock()
is_running = True

def camera_capture_thread():
    global latest_jpeg, is_running
    print("🎥 正在開啟本機攝影機 (Camera Index 0)...")
    
    # 嘗試用 DirectShow 啟動攝影機 (Windows 優化)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 無法開啟本機攝影機！請確認攝影機沒有被其他程式佔用。")
        is_running = False
        return

    print("✅ 攝影機擷取成功，開始持續產生 MJPEG 畫面...")

    while is_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 將畫面編碼為 JPEG
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with frame_lock:
                latest_jpeg = jpeg.tobytes()

        time.sleep(0.03)  # 控制在 ~30 FPS

    cap.release()
    print("🛑 攝影機已關閉。")

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpeg
        
        # 只要路徑是 /stream 或 / 或是 /video 都正常提供 MJPEG
        print(f"📡 收到來自 {self.client_address[0]} 的串流請求: {self.path}")
        
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        try:
            while is_running:
                with frame_lock:
                    jpg_data = latest_jpeg

                if jpg_data is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg_data)))
                    self.end_headers()
                    self.wfile.write(jpg_data)
                    self.wfile.write(b"\r\n")

                time.sleep(0.04)  # ~25 FPS 輸出
        except (ConnectionResetError, BrokenPipeError):
            print(f"🔌 客戶端 {self.client_address[0]} 已中斷連線。")
        except Exception as e:
            print(f"⚠️ 串流傳輸錯誤: {e}")

def main():
    # 啟動攝影機讀取執行緒
    cap_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    cap_thread.start()

    # 等待第一張畫面產生
    time.sleep(1.0)

    server_address = ("0.0.0.0", 80)
    httpd = ThreadingHTTPServer(server_address, MJPEGStreamHandler)
    print("=============================================================")
    print("🚀 PC MJPEG 影像串流伺服器已成功啟動！")
    print("📡 樹莓派連線網址: http://192.168.68.54/stream")
    print("=============================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 正在關閉串流伺服器...")
    finally:
        global is_running
        is_running = False
        httpd.server_close()

if __name__ == "__main__":
    main()
