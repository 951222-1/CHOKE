# -*- coding: utf-8 -*-
"""
===============================================================================
防哽咽即時監測系統 —— 模組驗收與獨立測試中心 (Interactive Test Center)
===============================================================================
提供可視化互動選單，讓使用者一鍵選擇欲驗收之感測/警報模組或啟動完整系統。
===============================================================================
"""

import os
import sys
import subprocess
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODULES = {
    "1": {
        "title": "✋ 手部骨架與抓喉手勢偵測模組",
        "file": "hand_skeleton_demo.py",
        "desc": "測試 21 個手部關節點追蹤、黃/橘色骨架繪製與喉嚨警戒區碰撞分析。",
        "tips": "【測試方法】將手靠近鏡頭前或脖子喉嚨處，觀察骨架繪製與手部偵測數量。"
    },
    "2": {
        "title": "🧍 身體姿態骨架與劇烈晃動分析模組",
        "file": "pose_skeleton_demo.py",
        "desc": "測試 33 個身體關節點追蹤、歸一化位移計算與身體晃動/掙扎得分。",
        "tips": "【測試方法】站在鏡頭前左右晃動或快速位移頭部，觀察 Body Shake Score 與 Warning 警示。"
    },
    "3": {
        "title": "👄 嘴部 MAR 開合角度與唇色發紺檢測模組",
        "file": "face_mouth_demo.py",
        "desc": "測試人臉 478 點 Mesh 嘴部 MAR (Mouth Aspect Ratio) 與缺氧發紺色度分析。",
        "tips": "【測試方法】對著鏡頭張嘴與閉嘴，觀察 MAR 數值變化 (Mouth OPEN / CLOSED) 及唇色分析。"
    },
    "4": {
        "title": "🎤 雙流咳嗽與嗆咳偵測模組 (Audio YAMNet + Vision Jerk)",
        "file": "cough_detection_demo.py",
        "desc": "測試 Google YAMNet 音訊 AI 咳嗽模型與臉部下巴/頭部快速抽動震幅 (Jerk Analysis)。",
        "tips": "【測試方法】對著麥克風咳嗽或快速上下晃動頭部，觀察音訊分數與 Vision Jerk 指標。"
    },
    "5": {
        "title": "🚨 多模態警報與證據融合測試模組",
        "file": "alert_system_demo.py",
        "desc": "測試時域融合評分器 (EvidenceFusion)、OpenCV 閃爍橫幅、蜂鳴聲、GUI 彈窗與 LINE/SQLite。",
        "tips": "【測試方法】開啟後在鍵盤按下 '1' (聲音)、'2' (手勢)、'3' (發紺) 累積分數測試警報。"
    },
    "6": {
        "title": "🦴 繪製全骨架渲染與連線展示模組",
        "file": "display_skeletons_demo.py",
        "desc": "測試綜合繪製演算法 (draw_face_and_mouth_landmarks, draw_hand_skeleton, draw_pose_skeleton)。",
        "tips": "【測試方法】觀察黃/藍/紅點標示鼻尖 (4)、下巴 (152)、上下唇 (13,14) 與嘴角 (78,308) 及手/姿骨架。"
    },
    "7": {
        "title": "🛡️ 啟動【完整防哽咽即時監測系統】(主程式)",
        "file": "防哽咽裝置4_2.py",
        "desc": "執行五階段進食狀態機、多模態融合、相機 UI Overlay 與音訊 YAMNet 咳嗽偵測主程式。",
        "tips": "【測試方法】完整運作模式，按 'q' 鍵退出。"
    }
}


def print_header():
    print("\n" + "=" * 72)
    print("🛡️   防哽咽即時監測系統 —— 獨立模組驗收與測試選單   🛡️")
    print("=" * 72)
    print("請選擇您想要驗收與測試的模組編號：\n")
    for key, mod in MODULES.items():
        print(f"  [{key}] {mod['title']}")
        print(f"      說明: {mod['desc']}")
        print(f"      提示: {mod['tips']}")
        print("-" * 72)
    print("  [0] 🚪 退出測試選單")
    print("=" * 72)


def run_module(choice):
    if choice not in MODULES:
        print("\n❌ 無效的選項，請重新輸入。")
        return

    mod = MODULES[choice]
    script_path = os.path.join(BASE_DIR, mod['file'])

    if not os.path.exists(script_path):
        print(f"\n❌ 找不到檔案：{script_path}")
        return

    print("\n" + "=" * 72)
    print(f"🚀 正在啟動：{mod['title']}")
    print(f"📄 執行檔案：{mod['file']}")
    print("💡 注意：視窗開啟後，在攝影機視窗內按 'q' 鍵可結束該測試並返回選單。")
    print("=" * 72 + "\n")
    time.sleep(1)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        subprocess.run([sys.executable, script_path], env=env)
    except Exception as e:
        print(f"\n❌ 執行時發生錯誤: {e}")

    print("\n" + "-" * 72)
    print(f"✅ 【{mod['title']}】測試已結束。")
    print("-" * 72)
    input("按 Enter 鍵返回主選單...")


def main():
    while True:
        print_header()
        try:
            choice = input("👉 請輸入模組編號 (0-7): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 已退出測試選單。")
            break

        if choice == '0':
            print("\n👋 感謝使用，已退出驗收測試中心！")
            break
        else:
            run_module(choice)


if __name__ == "__main__":
    main()
