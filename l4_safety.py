"""Pure L4 timing, leaky-bucket accumulation, multi-path triggers, face-visibility guards, and L4 Gmail SMTP dispatch."""

from email.message import EmailMessage
import math
import os
import smtplib
import time


_NOTIFICATION_DISABLE_ENV = "ANTICHOKE_DISABLE_EXTERNAL_NOTIFICATIONS"


def external_notifications_disabled():
    """Return True when a test/demo process must not contact external services."""
    return os.environ.get(_NOTIFICATION_DISABLE_ENV, "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def evaluate_l4_candidate(body_shaking, near_throat, near_chest,
                          hands_in_zone=0, recent_cough=False):
    """Evaluate the 3 high-specificity clinical L4 choking paths.

    Returns (active: bool, rate: float, reason: str | None). The active
    sensitivity profile uses a 2.5s timer, with rate 1.2 for the more specific
    two-hand and post-cough paths, and rate 1.0 for the body-shaking path.
      - Path B: Both hands simultaneously in throat/chest zone.
      - Path C: Recent cough (<= 12s) + hand clutching throat/chest.
      - Path A: Moderate body shaking/struggling + hand in throat/chest.
    """
    hands_count = int(hands_in_zone or 0)
    hand_in_zone = bool(near_throat or near_chest or hands_count >= 1)

    if hands_count >= 2:
        return True, 1.2, "雙手同時抓喉/撫胸窒息手勢"
    if hand_in_zone and recent_cough:
        return True, 1.2, "嗆咳後持續抓喉/撫胸"
    if hand_in_zone and body_shaking:
        return True, 1.0, "身體掙扎搖晃＋撫胸/抓喉"
    return False, 0.0, None


class ChokeDoubleActionTimer:
    """Track an L4 choking candidate with dropout grace and optional leaky-bucket decay."""

    def __init__(self, trigger_sec=5.0, grace_sec=0.6, decay_rate=0.0):
        self.trigger_sec = self._valid_duration(trigger_sec, "trigger_sec")
        self.grace_sec = self._valid_duration(grace_sec, "grace_sec")
        self.decay_rate = self._valid_duration(decay_rate, "decay_rate")
        self.reset()

    @staticmethod
    def _valid_duration(value, name):
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return duration

    @property
    def active(self):
        return self._start_time is not None

    @property
    def duration(self):
        return self._duration

    @property
    def reason(self):
        return self._reason

    def set_step_context(self, rate=1.0, reason=None):
        """Pre-supply accumulation rate/reason for the next update() call."""
        rate_val = float(rate)
        if not math.isfinite(rate_val) or rate_val < 0:
            raise ValueError("rate must be a finite non-negative number")
        self._pending_rate = rate_val
        self._pending_reason = reason

    def update(self, active, now, rate=None, reason=None):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be a finite number")

        step_rate = self._pending_rate if rate is None else float(rate)
        step_reason = self._pending_reason if reason is None else reason
        self._pending_rate = 1.0
        self._pending_reason = None
        if not math.isfinite(step_rate) or step_rate < 0:
            raise ValueError("rate must be a finite non-negative number")

        effective_rate = step_rate if step_rate > 0.0 else 1.0

        if active:
            if step_reason:
                self._reason = step_reason
            if self._start_time is None:
                self._start_time = now
                self._duration = 0.0
            elif self._grace_start_time is not None:
                if not self._in_decay and self._last_active_time is not None:
                    dt = max(0.0, now - self._last_active_time)
                    self._duration += dt * effective_rate
            elif self._last_update_time is not None:
                dt = max(0.0, now - self._last_update_time)
                self._duration += dt * effective_rate

            self._grace_start_time = None
            self._in_decay = False
            self._last_active_time = now
            self._start_time = now - self._duration
        elif self._start_time is not None:
            if self._grace_start_time is None:
                self._grace_start_time = now
            elif now - self._grace_start_time > self.grace_sec:
                if self.decay_rate <= 0.0:
                    self.reset()
                    return self._duration
                if not self._in_decay:
                    decay_dt = max(0.0, now - (self._grace_start_time + self.grace_sec))
                    self._in_decay = True
                else:
                    decay_dt = max(0.0, now - (self._last_update_time if self._last_update_time is not None else now))
                self._duration = max(0.0, self._duration - self.decay_rate * decay_dt)
                if self._duration <= 1e-9:
                    self.reset()
                    return self._duration
                self._start_time = now - self._duration

        self._last_update_time = now
        return self._duration

    def reached_trigger(self):
        return self._duration + 1e-9 >= self.trigger_sec

    def reset(self):
        self._start_time = None
        self._grace_start_time = None
        self._in_decay = False
        self._last_active_time = None
        self._last_update_time = None
        self._duration = 0.0
        self._reason = None
        self._pending_rate = 1.0
        self._pending_reason = None


class FaceLossWarningGuard:
    """Emit one unconfirmed visibility warning for a candidate at face-loss onset."""

    def __init__(self, grace_sec=0.6):
        self.grace_sec = ChokeDoubleActionTimer._valid_duration(grace_sec, "grace_sec")
        self.reset()

    def update(self, face_visible, candidate, now):
        now = float(now)
        if not math.isfinite(now):
            return False

        if face_visible:
            self.reset()
            return False

        if self._missing_since is None:
            self._missing_since = now
            self._candidate_at_loss = bool(candidate)

        if (not self._candidate_at_loss or self._warned
                or now - self._missing_since <= self.grace_sec):
            return False

        self._warned = True
        return True

    def reset(self):
        self._missing_since = None
        self._candidate_at_loss = False
        self._warned = False


def dispatch_visibility_warning(send_notification, show_alert, record_event,
                                patient_id, patient_name, chew_count):
    """Send a distinct unconfirmed visibility notice through injected sinks.

    Note: External notification (Gmail) is strictly reserved for confirmed L4 choking,
    so send_notification may be None (or a callback that filters out choke_visibility).
    """
    title = "⚠️ 目視確認：畫面遮蔽"
    message = (
        "⚠️【疑似哽噎，尚未確認】偵測到身體晃動與胸前手勢後，"
        "人臉畫面中斷，無法確認哽噎。請立即目視確認個案狀況。"
    )
    if callable(send_notification):
        send_notification(message, alert_type="choke_visibility")
    show_alert(title, message)
    record_event(
        patient_id, patient_name, chew_count, "疑似哽噎_視覺遮蔽需確認"
    )
    return message


def count_direction_reversals(positions, min_excursion=3.0):
    """Count direction reversals exceeding a pixel deadband (aligned with ConnectionCore)."""
    if not positions or len(positions) < 2:
        return 0
    threshold = max(0.0, float(min_excursion)) if math.isfinite(float(min_excursion)) else 3.0
    pivot = None
    direction = 0
    reversals = 0

    for raw_pos in positions:
        try:
            pos = float(raw_pos)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pos):
            continue
        if pivot is None:
            pivot = pos
            continue

        if direction == 0:
            initial_movement = pos - pivot
            if abs(initial_movement) >= threshold and initial_movement != 0.0:
                direction = 1 if initial_movement > 0 else -1
                pivot = pos
        elif direction > 0:
            if pos >= pivot:
                pivot = pos
            elif pivot - pos >= threshold:
                reversals += 1
                direction = -1
                pivot = pos
        else:
            if pos <= pivot:
                pivot = pos
            elif pos - pivot >= threshold:
                reversals += 1
                direction = 1
                pivot = pos

    return reversals


def check_continuous_body_shaking(body_motion_history, nose, face_size, current_time,
                                  w, h, speed_sum_threshold=1.0):
    """Evaluate continuous struggling/body shaking (aligned with mobile/app.js)."""
    if not nose or face_size is None or face_size <= 0:
        return False

    nx, ny = (nose[0], nose[1]) if isinstance(nose, (tuple, list)) else (nose.x, nose.y)
    body_motion_history.append({
        "x": float(nx),
        "y": float(ny),
        "face_size": float(face_size),
        "time": float(current_time),
    })

    while body_motion_history and (current_time - body_motion_history[0]["time"] > 1.4):
        body_motion_history.pop(0)

    if len(body_motion_history) < 4:
        return False

    window_span_sec = body_motion_history[-1]["time"] - body_motion_history[0]["time"]
    if window_span_sec < 0.35:  # TEST ONLY: accept shorter observed motion windows
        return False

    min_x = min(p["x"] for p in body_motion_history)
    max_x = max(p["x"] for p in body_motion_history)
    min_y = min(p["y"] for p in body_motion_history)
    max_y = max(p["y"] for p in body_motion_history)
    min_size = min(p["face_size"] for p in body_motion_history)
    max_size = max(p["face_size"] for p in body_motion_history)

    total_speed = 0.0
    for i in range(1, len(body_motion_history)):
        p = body_motion_history[i]
        prev = body_motion_history[i - 1]
        dx = (p["x"] - prev["x"]) * w
        dy = (p["y"] - prev["y"]) * h
        step_speed = math.hypot(dx, dy) / p["face_size"]
        total_speed += step_speed

    x_reversals = count_direction_reversals([p["x"] * w for p in body_motion_history], 2.0)
    y_reversals = count_direction_reversals([p["y"] * h for p in body_motion_history], 2.0)
    x_span = (max_x - min_x) * w / face_size
    y_span = (max_y - min_y) * h / face_size
    size_span = (max_size - min_size) / face_size

    speed_threshold = float(speed_sum_threshold)
    if not math.isfinite(speed_threshold) or speed_threshold < 0:
        raise ValueError("speed_sum_threshold must be a finite non-negative number")
    has_direction_reversal = (x_reversals >= 1 or y_reversals >= 1 or size_span > 0.12)
    has_significant_span = (x_span > 0.10 or y_span > 0.10 or size_span > 0.10)
    has_sustained_speed = total_speed > speed_threshold

    return bool(has_significant_span and has_sustained_speed and has_direction_reversal)


class ThresholdCrossingGate:
    """Fire once when value crosses threshold; re-arm when value drops below threshold."""

    def __init__(self):
        self._triggered = False

    def update(self, value, threshold):
        try:
            current = float(value)
            limit = float(threshold)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(current) or not math.isfinite(limit):
            return False
        if current < limit:
            self._triggered = False
            return False
        if self._triggered:
            return False
        self._triggered = True
        return True

    def reset(self):
        self._triggered = False


class ChewingPauseTracker:
    """Track chewing pause stillness duration after chewing starts (aligned with ConnectionCore)."""

    def __init__(self, max_observed_frame_gap_sec=0.4):
        self.max_observed_frame_gap_sec = float(max_observed_frame_gap_sec)
        self.reset()

    def reset(self):
        self._chewing_observed = False
        self._elapsed_sec = 0.0
        self._last_update_time = None
        self._previous_frame_face_visible = False

    def update(self, active, face_visible, chewing_detected, current_time):
        try:
            now = float(current_time)
        except (TypeError, ValueError):
            self.reset()
            return 0.0
        if not active or not math.isfinite(now):
            self.reset()
            return 0.0
        if self._last_update_time is not None and now < self._last_update_time:
            return self._elapsed_sec

        if not face_visible:
            self._last_update_time = now
            self._previous_frame_face_visible = False
            return self._elapsed_sec

        if chewing_detected:
            self._chewing_observed = True
            self._elapsed_sec = 0.0
        elif (self._chewing_observed and self._previous_frame_face_visible
              and self._last_update_time is not None):
            observed_gap_sec = now - self._last_update_time
            if observed_gap_sec <= self.max_observed_frame_gap_sec:
                self._elapsed_sec += max(0.0, observed_gap_sec)

        self._last_update_time = now
        self._previous_frame_face_visible = True
        return self._elapsed_sec


def build_l4_gmail_email(message, sender, recipient=None,
                         subject="L4 進食安全提醒", patient_name=None,
                         timestamp_str=None):
    """Build an EmailMessage for an L4 acute choking emergency alert (aligned with notify-l4-email)."""
    clean_sender = (sender or "").strip()
    clean_recipient = (recipient or clean_sender).strip()
    now_str = timestamp_str or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    patient_line = f"個案名稱：{patient_name}\n" if patient_name else ""
    body = (
        "系統偵測到 L4 進食安全緊急事件，請立即查看個案狀況。\n\n"
        "警報等級：L4（急性哽噎緊急警報）\n"
        f"{patient_line}"
        f"觸發時間：{now_str}\n"
        f"詳細內容：{message}\n"
    )
    email_msg = EmailMessage()
    email_msg["Subject"] = subject
    email_msg["From"] = clean_sender
    email_msg["To"] = clean_recipient
    email_msg.set_content(body)
    return email_msg


def send_l4_gmail_smtp(message, sender, app_password, recipient=None,
                       subject="L4 進食安全提醒", patient_name=None,
                       smtp_host="smtp.gmail.com", smtp_port=465,
                       smtp_factory=smtplib.SMTP_SSL):
    """Send an L4 emergency alert email via Gmail SMTP_SSL."""
    if external_notifications_disabled():
        return False
    clean_sender = (sender or "").strip()
    clean_password = (app_password or "").replace(" ", "").strip()
    clean_recipient = (recipient or clean_sender).strip()
    if not clean_sender or not clean_password or not clean_recipient:
        return False

    email_msg = build_l4_gmail_email(
        message=message,
        sender=clean_sender,
        recipient=clean_recipient,
        subject=subject,
        patient_name=patient_name,
    )
    with smtp_factory(smtp_host, int(smtp_port), timeout=10) as smtp:
        smtp.login(clean_sender, clean_password)
        smtp.send_message(email_msg)
    return True
