# -*- coding: utf-8 -*-
# 無聲窒息偵測。大家以為噎到都會咳得很大聲,但氣管完全塞住時人是發不出聲的,
# 這種安靜的噎到反而最危險、也最容易被漏掉。所以這裡反過來抓:
# 嘴開著、身體不動、又幾乎沒聲音,而且持續好幾秒 -> 很可疑。
import time


class SilentChokeDetector:
    def __init__(self, hold_sec, clock=time.time):
        self.hold = hold_sec      # 條件要連續成立幾秒才算數
        self.clock = clock        # 測試時可換成假時鐘
        self.since = None         # 條件從何時開始成立

    def update(self, mar, movement_std, audio_energy,
               mouth_open_th, still_th, quiet_th):
        # mar=嘴張多開, movement_std=晃動程度, audio_energy=音量
        cond = (mar > mouth_open_th
                and movement_std < still_th
                and audio_energy < quiet_th)
        now = self.clock()
        if cond:
            if self.since is None:
                self.since = now
            elif now - self.since >= self.hold:
                self.since = None     # 報一次就歸零,不重複洗版
                return True
        else:
            self.since = None         # 中間只要斷一次就重來
        return False
