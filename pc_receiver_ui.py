# -*- coding: utf-8 -*-
"""
ESP32-S3 Sense 影音雙串流接收端 (Wi-Fi 方案 B)
這支程式用來接收來自 Seeed Studio XIAO ESP32S3 Sense 的即時影像與聲音串流，並在 PC 上提供 UI 視窗。
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import socket
import time
import cv2
import numpy as np
import requests
import sounddevice as sd
from PIL import Image, ImageTk

class ESP32StreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32-S3 Sense 影音串流監控器")
        self.root.geometry("680x600")
        self.root.configure(bg="#1e1e1e")

        # 狀態控制變數
        self.is_connected = False
        self.video_thread = None
        self.audio_thread = None

        self.setup_ui()

    def setup_ui(self):
        # 樣式設定
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")
        style.configure("TButton", font=("Microsoft JhengHei", 10, "bold"))

        # 頂部控制面板
        control_frame = tk.Frame(self.root, bg="#2d2d2d", pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        lbl_ip = tk.Label(control_frame, text="ESP32-S3 IP 地址:", bg="#2d2d2d", fg="#ffffff", font=("Microsoft JhengHei", 10, "bold"))
        lbl_ip.pack(side=tk.LEFT, padx=10)

        self.entry_ip = tk.Entry(control_frame, font=("Courier New", 11), width=18)
        self.entry_ip.insert(0, "192.168.68.53")  # ESP32-S3 IP on mkc_ssd
        self.entry_ip.pack(side=tk.LEFT, padx=5)

        self.btn_connect = tk.Button(control_frame, text="連接連線", command=self.toggle_connection, bg="#4CAF50", fg="white", font=("Microsoft JhengHei", 10, "bold"), relief=tk.FLAT, padx=15)
        self.btn_connect.pack(side=tk.LEFT, padx=15)

        self.lbl_status = tk.Label(control_frame, text="狀態: 未連接", fg="#ff9800", bg="#2d2d2d", font=("Microsoft JhengHei", 10, "bold"))
        self.lbl_status.pack(side=tk.RIGHT, padx=15)

        # 影像顯示區
        self.video_label = tk.Label(self.root, text="🎥 等待影像串流連線...", bg="#121212", fg="#888888", font=("Microsoft JhengHei", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 底部狀態列
        status_bar = tk.Frame(self.root, bg="#2d2d2d", height=25)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_info = tk.Label(status_bar, text="提示: 樹莓派熱點開啟後，請確認本機外接網卡已連上 aaaa Wi-Fi", bg="#2d2d2d", fg="#aaaaaa", font=("Microsoft JhengHei", 9))
        self.lbl_info.pack(side=tk.LEFT, padx=10)

    def toggle_connection(self):
        if not self.is_connected:
            self.start_connection()
        else:
            self.stop_connection()

    def start_connection(self):
        ip = self.entry_ip.get().strip()
        if not ip:
            messagebox.showerror("錯誤", "請輸入有效的 IP 地址！")
            return

        self.is_connected = True
        self.btn_connect.config(text="中斷連線", bg="#f44336")
        self.lbl_status.config(text="狀態: 連線中...", fg="#2196F3")
        self.video_label.config(text="🔄 正在建立影音連線，請稍候...")

        # 啟動影音接收執行緒
        self.video_thread = threading.Thread(target=self.receive_video, args=(ip,), daemon=True)
        self.audio_thread = threading.Thread(target=self.receive_audio, args=(ip,), daemon=True)
        self.video_thread.start()
        self.audio_thread.start()

    def stop_connection(self):
        self.is_connected = False
        self.btn_connect.config(text="連接連線", bg="#4CAF50")
        self.lbl_status.config(text="狀態: 已斷開", fg="#ff9800")
        self.video_label.config(image="")
        self.video_label.config(text="🎥 等待影像串流連線...")
        self.lbl_info.config(text="連線已中斷。")

    def receive_video(self, ip):
        url = f"http://{ip}:80/"
        print(f"[Video] 開始連線至 {url}")
        
        try:
            # 建立 HTTP 串流請求
            r = requests.get(url, stream=True, timeout=5)
            if r.status_code != 200:
                raise Exception(f"HTTP 狀態碼: {r.status_code}")
                
            self.lbl_status.config(text="狀態: 影音連線成功", fg="#4CAF50")
            self.lbl_info.config(text=f"已連線至 ESP32 影像伺服器 ({ip}:80)")

            bytes_data = b""
            # 讀取 MJPEG 串流
            for chunk in r.iter_content(chunk_size=1024):
                if not self.is_connected:
                    break
                bytes_data += chunk
                
                # 搜尋 JPEG 起始與結束標記
                a = bytes_data.find(b"\xff\xd8")
                b = bytes_data.find(b"\xff\xd9")
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a : b + 2]
                    bytes_data = bytes_data[b + 2 :]
                    
                    # 將 JPEG 解碼為影像
                    img_np = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 轉為 RGB 格式並縮放到視窗大小
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame)
                        
                        # 保持等比例縮放
                        img.thumbnail((640, 480))
                        photo = ImageTk.PhotoImage(image=img)
                        
                        # 更新 UI 畫面
                        if self.is_connected:
                            self.video_label.config(image=photo)
                            self.video_label.image = photo
                            
        except Exception as e:
            print(f"[Video] 發生錯誤: {e}")
            if self.is_connected:
                self.root.after(0, self.handle_connect_error, f"影像連線失敗: {e}")

    def receive_audio(self, ip):
        url = f"http://{ip}:81/"
        print(f"[Audio] 開始連線至 {url}")
        
        try:
            # 建立音訊 TCP 連線
            r = requests.get(url, stream=True, timeout=5)
            if r.status_code != 200:
                raise Exception(f"HTTP 狀態碼: {r.status_code}")

            # 初始化 sounddevice 音訊輸出串流 (16kHz, 16-bit, 單聲道)
            stream = sd.RawOutputStream(
                samplerate=16000,
                blocksize=512,
                channels=1,
                dtype="int16"
            )
            
            with stream:
                # 讀取 PCM 串流位元組並寫入聲卡播放
                for chunk in r.iter_content(chunk_size=1024):
                    if not self.is_connected:
                        break
                    stream.write(chunk)
                    
        except Exception as e:
            print(f"[Audio] 發生錯誤: {e}")

    def handle_connect_error(self, err_msg):
        self.stop_connection()
        messagebox.showerror("連線錯誤", err_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = ESP32StreamApp(root)
    root.mainloop()
