# -*- coding: utf-8 -*-
"""
===============================================================================
防哽咽即時監測系統 —— 獨立核心偵測演算法模組 (Standalone Detection Core)
===============================================================================
本檔案將專案中所有「偵測」相關的演算法、幾何計算與時間窗特徵獨立抽取出來，
包含：
 1. 手抓喉嚨窒息手勢估算 (gesture.py)
 2. 無聲窒息與咀嚼吞嚥狀態機 (silent_choke.py)
 3. 多模態時間窗證據加權融合 (evidence_fusion.py)
 4. 嘴唇藍光比率與缺氧發紺分析 (cyanosis.py)
 5. 身體劇烈晃動無因次化位移量計算 (Scale-Invariant Shaking)
 6. 嘴部開合角度 (MAR) 計算
===============================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import math
import time


# =============================================================================
# 1. 手抓喉嚨窒息手勢偵測 (Choke Gesture Detection - A1)
# =============================================================================
def estimate_throat(nose, chin, extend=0.6):
    """
    沿「鼻子 → 下巴」向量延伸估算喉嚨中心的座標 (tx, ty)
    """
    tx = chin[0] + extend * (chin[0] - nose[0])
    ty = chin[1] + extend * (chin[1] - nose[1])
    return tx, ty


def hand_near_throat(hands, nose, chin, w, h, radius_scale=0.6):
    """
    判斷任何一隻手的 21 個關鍵點是否有任何一點落在「喉嚨自適應偵測圈」內。
    - radius_scale: 預設為 0.6 * 臉高 (自適應隨著靠近/遠離鏡頭縮放)
    """
    if not hands:
        return False
    tx, ty = estimate_throat(nose, chin)
    # 圈圈半徑以「鼻尖至下巴距離」做自適應基準
    face_v = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    radius = max(face_v * radius_scale, 1.0)

    for hand in hands:
        for (x, y) in hand:
            if math.hypot((x - tx) * w, (y - ty) * h) < radius:
                return True
    return False


def estimate_chest(nose, chin, extend=1.35):
    """估算胸前感應區 (Chest Zone)：沿「鼻子→下巴」方向向下延伸 1.35 倍臉長"""
    cx = chin[0] + extend * (chin[0] - nose[0])
    cy = chin[1] + extend * (chin[1] - nose[1])
    return cx, cy


def hand_near_chest(hands, nose, chin, w, h, radius_scale=0.7):
    """判斷是否有手部關節點落在胸前核心感應區 (Chest Zone) 內"""
    if not hands or nose is None or chin is None:
        return False
    cx, cy = estimate_chest(nose, chin)
    face_v = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    radius = max(face_v * radius_scale, 1.0)
    for hand in hands:
        for (x, y) in hand:
            if math.hypot((x - cx) * w, (y - cy) * h) < radius:
                return True
    return False



# =============================================================================
# 2. 嘴部開合比率 (Mouth Aspect Ratio - MAR) & 身體劇烈晃動計算
# =============================================================================
def calculate_mar(lip_top, lip_bottom, lip_left, lip_right):
    """
    計算嘴部開合垂直與水平距離比率 (MAR)
    """
    v_dist = math.hypot(lip_top[0] - lip_bottom[0], lip_top[1] - lip_bottom[1])
    h_dist = math.hypot(lip_left[0] - lip_right[0], lip_left[1] - lip_right[1])
    if h_dist <= 0:
        return 0.0
    return v_dist / h_dist


def calculate_body_shaking(nose_history, face_size):
    """
    無因次化身體/頭部劇烈晃動位移計算 (Scale-Invariant Displacement)
    - nose_history: 1 秒內 (約 30 幀) 鼻尖 (x,y) 座標歷史紀錄
    - face_size: 目前臉高 (像素)
    """
    if len(nose_history) < 2 or face_size <= 0:
        return 0.0

    total_disp = 0.0
    for i in range(1, len(nose_history)):
        dx = nose_history[i][0] - nose_history[i - 1][0]
        dy = nose_history[i][1] - nose_history[i - 1][1]
        total_disp += math.hypot(dx, dy)

    # 除以臉高進行無因次化 (歸一化)
    return total_disp / face_size


# =============================================================================
# 3. 無聲窒息與咀嚼吞嚥狀態機 (Silent Choke & Chewing Timeout)
# =============================================================================
class SilentChokeDetector:
    """
    無聲窒息偵測：當氣管完全堵塞時無法發聲，
    偵測連續好幾秒符合「嘴開 + 身體不動 + 聲音安靜」的危險組合。
    """

    def __init__(self, hold_sec=3.0, clock=time.time):
        self.hold = hold_sec  # 條件需連續成立的秒數
        self.clock = clock
        self.since = None  # 開始成立的時間點

    def update(self, mar, movement_std, audio_energy, mouth_open_th=0.3, still_th=0.05, quiet_th=0.1):
        cond = (mar > mouth_open_th and movement_std < still_th and audio_energy < quiet_th)
        now = self.clock()
        if cond:
            if self.since is None:
                self.since = now
            elif now - self.since >= self.hold:
                self.since = None
                return True
        else:
            self.since = None
        return False


# =============================================================================
# 4. 多模態時間窗證據加權融合演算法 (Evidence Fusion)
# =============================================================================
class EvidenceFusion:
    """
    多模態分數融合：避免單一訊號誤判，在滑動時間窗內收集手勢、聲音、晃動等線索加權算分。
    """

    def __init__(self, window_sec=5.0, threshold=1.0, cooldown_sec=10.0, clock=time.time):
        self.window = window_sec
        self.threshold = threshold
        self.cooldown = cooldown_sec
        self.clock = clock
        self.sources = {}
        self.last_fire = -1e9

    def observe(self, source, weight):
        if weight > 0:
            self.sources[source] = (self.clock(), weight)

    def score(self):
        now = self.clock()
        # 移除已超過時間窗的過期線索
        self.sources = {k: (t, w) for k, (t, w) in self.sources.items() if now - t <= self.window}
        return sum(w for (_, w) in self.sources.values())

    def check(self):
        now = self.clock()
        if self.score() >= self.threshold and now - self.last_fire >= self.cooldown:
            self.last_fire = now
            return True
        return False


# =============================================================================
# 5. 嘴唇藍光比率與發紺缺氧分析 (Cyanosis Detection)
# =============================================================================
def blueness(lip_rgb):
    """
    計算嘴唇顏色中藍色佔整體 RGB 的比例
    """
    r, g, b = lip_rgb
    s = r + g + b
    if s <= 0:
        return 0.0
    return b / s


def is_cyanotic(lip_rgb, blue_th=0.38):
    """
    當嘴唇藍光比例高於門檻藍光比率 (預設 0.38) 時判定為可能發紺缺氧
    """
    return blueness(lip_rgb) >= blue_th


# =============================================================================
# 測試展示主程式 (Demo Runs)
# =============================================================================
if __name__ == "__main__":
    print("=====================================================================")
    print("防哽咽即時監測系統 —— 獨立偵測演算法模組單元測試")
    print("=====================================================================")

    # 1. 測試手抓喉嚨估算
    nose = (0.5, 0.4)
    chin = (0.5, 0.6)
    tx, ty = estimate_throat(nose, chin)
    print(f"[1. 喉嚨圈估算] 鼻尖 {nose}, 下巴 {chin} -> 估算喉嚨中心: ({tx:.2f}, {ty:.2f})")

    # 模擬手部位置落在喉嚨圈內
    hand_points = [[(0.5, 0.7)]]  # 接近喉嚨點 (0.5, 0.72)
    is_hand_choke = hand_near_throat(hand_points, nose, chin, 640, 480, radius_scale=0.6)
    print(f"[1. 手勢測試] 手部點落在喉嚨附近 -> 手抓喉嚨成立: {is_hand_choke}")

    # 2. 測試嘴唇藍光發紺
    normal_lip = (200, 50, 50)  # 紅潤
    cyanotic_lip = (60, 60, 160)  # 發紫
    print(f"[2. 發紺測試] 正常嘴唇 {normal_lip} 藍光比: {blueness(normal_lip):.2f} -> 發紺: {is_cyanotic(normal_lip)}")
    print(f"[2. 發紺測試] 缺氧嘴唇 {cyanotic_lip} 藍光比: {blueness(cyanotic_lip):.2f} -> 發紺: {is_cyanotic(cyanotic_lip)}")

    # 3. 測試多模態證據融合
    fusion = EvidenceFusion(window_sec=5.0, threshold=1.0, cooldown_sec=5.0)
    fusion.observe("hand_throat", 0.6)
    fusion.observe("cough_sound", 0.5)
    print(f"[3. 證據融合] 目前融合分數: {fusion.score():.2f} / 門檻 1.0 -> 觸發警報: {fusion.check()}")
    print("=====================================================================")
