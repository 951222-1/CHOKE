# -*- coding: utf-8 -*-
# 手抓喉嚨偵測。人噎到時常會反射性抓自己的喉嚨,這是全世界通用的窒息手勢,
# 拿來當判斷依據很可靠。這裡只做純幾何,不碰相機,所以很好測試。
import math


def estimate_throat(nose, chin, extend=0.6):
    # 沒有喉嚨的地標,就沿「鼻子→下巴」方向再往下延伸,估一個喉嚨的位置
    tx = chin[0] + extend * (chin[0] - nose[0])
    ty = chin[1] + extend * (chin[1] - nose[1])
    return tx, ty


def hand_near_throat(hands, nose, chin, w, h, radius_scale=0.6):
    """任何一隻手的任何一點,落在喉嚨附近的圈圈裡就回 True。"""
    if not hands:
        return False
    tx, ty = estimate_throat(nose, chin)
    # 圈圈半徑用臉的大小當基準(人靠近鏡頭臉會變大,半徑也要跟著大)
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

