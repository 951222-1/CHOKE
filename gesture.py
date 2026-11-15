# -*- coding: utf-8 -*-
# 手抓喉嚨與撫胸偵測。人噎到時常會反射性抓自己的喉嚨或拍撫上胸口,這是全世界通用的窒息手勢,
# 拿來當判斷依據很可靠。這裡只做純幾何,不碰相機,並輸出與畫面繪製 100% 對齊的感應圈幾何。
import math

THROAT_EXTEND = 0.50
THROAT_RADIUS_SCALE = 0.72  # TEST ONLY: modestly larger detection zone
THROAT_MIN_RADIUS_PX = 24.0

CHEST_EXTEND = 1.15
CHEST_RADIUS_SCALE = 1.25  # TEST ONLY: modestly larger detection zone
CHEST_MIN_RADIUS_PX = 44.0


def estimate_throat(nose, chin, extend=THROAT_EXTEND):
    # 沒有喉嚨的地標,就沿「鼻子→下巴」方向再往下延伸,估一個喉嚨的位置
    tx = chin[0] + extend * (chin[0] - nose[0])
    ty = chin[1] + extend * (chin[1] - nose[1])
    return tx, ty


def estimate_chest(nose, chin, extend=CHEST_EXTEND):
    """估算上胸口/鎖骨感應區 (Chest Zone)：沿「鼻子→下巴」方向向下延伸 1.15 倍臉長，避免超出鏡頭下緣"""
    cx = chin[0] + extend * (chin[0] - nose[0])
    cy = chin[1] + extend * (chin[1] - nose[1])
    return cx, cy


def danger_zone_geometry(nose, chin, w, h):
    """回傳與實際碰撞判定 100% 一致的像素圓心與半徑，供 OpenCV 疊圖與幾何判定共用。"""
    face_v = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    tx, ty = estimate_throat(nose, chin)
    cx, cy = estimate_chest(nose, chin)
    throat_r = max(face_v * THROAT_RADIUS_SCALE, THROAT_MIN_RADIUS_PX)
    chest_r = max(face_v * CHEST_RADIUS_SCALE, CHEST_MIN_RADIUS_PX)
    return {
        "face_v": face_v,
        "throat_center": (tx * w, ty * h),
        "throat_radius": throat_r,
        "chest_center": (cx * w, cy * h),
        "chest_radius": chest_r,
    }


def analyze_hands_in_danger_zones(hands, nose, chin, w, h, min_pts=2):
    """分析單手/雙手進入喉嚨與上胸口感應圈的狀況，回傳與視覺圈完全對齊的結果。"""
    if nose is None or chin is None:
        return {
            "near_throat": False,
            "near_chest": False,
            "near_any": False,
            "hands_in_zone": 0,
            "throat_center": None,
            "throat_radius": 0.0,
            "chest_center": None,
            "chest_radius": 0.0,
        }

    geom = danger_zone_geometry(nose, chin, w, h)
    tx_px, ty_px = geom["throat_center"]
    cx_px, cy_px = geom["chest_center"]
    throat_r = geom["throat_radius"]
    chest_r = geom["chest_radius"]
    req_pts = max(1, int(min_pts))

    near_throat = False
    near_chest = False
    hands_in_zone = 0

    for hand in (hands or []):
        pts_throat = 0
        pts_chest = 0
        pts_either = 0
        for (x, y) in hand:
            px, py = x * w, y * h
            in_t = math.hypot(px - tx_px, py - ty_px) < throat_r
            in_c = math.hypot(px - cx_px, py - cy_px) < chest_r
            if in_t:
                pts_throat += 1
            if in_c:
                pts_chest += 1
            if in_t or in_c:
                pts_either += 1

        if pts_throat >= req_pts:
            near_throat = True
        if pts_chest >= req_pts:
            near_chest = True
        if pts_either >= req_pts:
            hands_in_zone += 1

    return {
        "near_throat": near_throat,
        "near_chest": near_chest,
        "near_any": bool(near_throat or near_chest or hands_in_zone > 0),
        "hands_in_zone": hands_in_zone,
        **geom,
    }


def hand_near_throat(hands, nose, chin, w, h, radius_scale=THROAT_RADIUS_SCALE, min_pts=1):
    """判斷是否有手部關節點落在喉嚨感應區 (Throat Zone) 內。"""
    if not hands or nose is None or chin is None:
        return False
    tx, ty = estimate_throat(nose, chin)
    face_v = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    radius = max(face_v * radius_scale, THROAT_MIN_RADIUS_PX)
    req_pts = max(1, int(min_pts))
    for hand in hands:
        pts = sum(1 for (x, y) in hand if math.hypot((x - tx) * w, (y - ty) * h) < radius)
        if pts >= req_pts:
            return True
    return False


def hand_near_chest(hands, nose, chin, w, h, radius_scale=CHEST_RADIUS_SCALE, min_pts=1):
    """判斷是否有手部關節點落在上胸口核心感應區 (Chest Zone) 內。"""
    if not hands or nose is None or chin is None:
        return False
    cx, cy = estimate_chest(nose, chin)
    face_v = math.hypot((chin[0] - nose[0]) * w, (chin[1] - nose[1]) * h)
    radius = max(face_v * radius_scale, CHEST_MIN_RADIUS_PX)
    req_pts = max(1, int(min_pts))
    for hand in hands:
        pts = sum(1 for (x, y) in hand if math.hypot((x - cx) * w, (y - cy) * h) < radius)
        if pts >= req_pts:
            return True
    return False
