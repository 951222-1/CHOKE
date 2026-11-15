# -*- coding: utf-8 -*-
# 把多個線索一起看。單一訊號常會誤判(例如「嘴開又不動」可能只是在吞東西),
# 但「嘴開不動 + 手抓喉嚨 + 嘴唇發紫」在短時間內都出現,那多半是真的出事。
# 做法:把最近幾秒的線索各給一個分數加起來,超過門檻才報。
import time


class EvidenceFusion:
    def __init__(self, window_sec, threshold, cooldown_sec, clock=time.time):
        self.window = window_sec
        self.threshold = threshold
        self.cooldown = cooldown_sec
        self.clock = clock
        self.sources = {}
        self.last_fire = -1e9

    def observe(self, source, weight):
        # 同一個線索只留最新的一次,不然它一直成立分數會一直累加(講久一點就爆表了)
        if weight > 0:
            self.sources[source] = (self.clock(), weight)

    def score(self):
        now = self.clock()
        self.sources = {k: (t, w) for k, (t, w) in self.sources.items()
                        if now - t <= self.window}   # 丟掉過期的線索
        return sum(w for (_, w) in self.sources.values())

    def check(self):
        now = self.clock()
        if self.score() >= self.threshold and now - self.last_fire >= self.cooldown:
            self.last_fire = now
            return True
        return False
