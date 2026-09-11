# -*- coding: utf-8 -*-
# 嘴唇發紫(發紺)偵測。缺氧到後期嘴唇會轉青紫色,這是很明確的「真的缺氧了」的訊號。
# 這裡用最簡單的做法:算嘴唇顏色裡「藍」佔的比例,太高就當作發紫。
# (真正部署要針對膚色/光線校正,這邊先給核心概念。)


def blueness(lip_rgb):
    r, g, b = lip_rgb
    s = r + g + b
    if s <= 0:
        return 0.0
    return b / s


def is_cyanotic(lip_rgb, blue_th=0.38):
    return blueness(lip_rgb) >= blue_th
