import json
from pathlib import Path
import unittest

import gesture
from l4_safety import (
    ChewingPauseTracker,
    ChokeDoubleActionTimer,
    FaceLossWarningGuard,
    ThresholdCrossingGate,
    check_continuous_body_shaking,
    count_direction_reversals,
    dispatch_visibility_warning,
    evaluate_l4_candidate,
)

MAIN_SOURCE = Path(__file__).with_name("防哽咽裝置4_2.py").read_text(encoding="utf-8")
THRESHOLDS = json.loads(
    (Path(__file__).parent / "shared" / "thresholds.json").read_text(encoding="utf-8")
)


class ChokeDoubleActionTimerTests(unittest.TestCase):
    def test_short_detection_gap_preserves_candidate_duration(self):
        timer = ChokeDoubleActionTimer(trigger_sec=5.0, grace_sec=0.6)

        timer.update(True, 10.0)
        timer.update(True, 14.7)
        timer.update(False, 14.71)
        duration = timer.update(True, 15.1)

        self.assertAlmostEqual(duration, 5.1)
        self.assertTrue(timer.reached_trigger())

    def test_gap_longer_than_grace_resets_candidate(self):
        timer = ChokeDoubleActionTimer(trigger_sec=5.0, grace_sec=0.6)

        timer.update(True, 10.0)
        timer.update(True, 14.0)
        timer.update(False, 14.01)
        timer.update(False, 14.62)

        self.assertFalse(timer.active)
        self.assertEqual(timer.duration, 0.0)
        self.assertFalse(timer.reached_trigger())

    def test_leaky_bucket_decay_survives_motion_blur_gap_and_reaches_3s_trigger(self):
        timer = ChokeDoubleActionTimer(trigger_sec=3.0, grace_sec=0.6, decay_rate=0.5)

        timer.update(True, 10.0)
        timer.update(True, 12.2)   # 2.2s accumulated
        timer.update(False, 12.2)  # motion-blur dropout begins
        timer.update(False, 13.2)  # 0.6s grace + 0.4s decay * 0.5 = 0.2s lost -> 2.0s kept
        self.assertAlmostEqual(timer.duration, 2.0)
        self.assertTrue(timer.active)

        timer.update(True, 13.2)   # re-acquired
        duration = timer.update(True, 14.2)  # +1.0s -> 3.0s
        self.assertAlmostEqual(duration, 3.0)
        self.assertTrue(timer.reached_trigger())

    def test_leaky_bucket_decays_brief_false_positive_back_to_zero(self):
        timer = ChokeDoubleActionTimer(trigger_sec=3.0, grace_sec=0.6, decay_rate=0.5)

        timer.update(True, 10.0)
        timer.update(True, 10.4)   # brief 0.4s hand pass
        timer.update(False, 10.4)
        timer.update(False, 12.0)  # 0.6s grace + 1.0s * 0.5 = 0.5s decay -> clamped at 0.0
        self.assertFalse(timer.active)
        self.assertEqual(timer.duration, 0.0)


class MultiPathL4CandidateTests(unittest.TestCase):
    def test_three_clinical_paths_and_specificity_guards(self):
        # Single hand still without cough -> NOT an L4 candidate
        active, rate, _ = evaluate_l4_candidate(
            body_shaking=False, near_throat=True, near_chest=False,
            hands_in_zone=1, recent_cough=False
        )
        self.assertFalse(active)
        self.assertEqual(rate, 0.0)

        # Path A: body shaking + hand on chest/throat -> 1.0x (3.0s)
        active, rate, reason = evaluate_l4_candidate(
            body_shaking=True, near_throat=False, near_chest=True,
            hands_in_zone=1, recent_cough=False
        )
        self.assertTrue(active)
        self.assertAlmostEqual(rate, 1.0)
        self.assertIn("掙扎", reason)

        # Path B: two hands in danger zone -> 1.2x (2.5s), even without shaking
        active, rate, reason = evaluate_l4_candidate(
            body_shaking=False, near_throat=True, near_chest=True,
            hands_in_zone=2, recent_cough=False
        )
        self.assertTrue(active)
        self.assertAlmostEqual(rate, 1.2)
        self.assertIn("雙手", reason)

        # Path C: post-cough clutch -> 1.2x (2.5s)
        timer = ChokeDoubleActionTimer(trigger_sec=3.0, grace_sec=0.6, decay_rate=0.5)
        active, rate, reason = evaluate_l4_candidate(
            body_shaking=False, near_throat=True, near_chest=False,
            hands_in_zone=1, recent_cough=True
        )
        self.assertTrue(active)
        timer.update(active, 10.0, rate=rate, reason=reason)
        timer.update(active, 12.5, rate=rate, reason=reason)
        self.assertTrue(timer.reached_trigger())
        self.assertIn("嗆咳後", timer.reason)


class DangerZoneGeometryTests(unittest.TestCase):
    def test_zone_geometry_matches_hand_analysis_and_min_pts_relaxation(self):
        nose = (0.5, 0.35)
        chin = (0.5, 0.50)
        w, h = 640, 480
        geom = gesture.danger_zone_geometry(nose, chin, w, h)
        cx, cy = geom["chest_center"]

        # Place 1 landmark inside chest zone and 20 landmarks far away
        hand = [(cx / w, cy / h)] + [(0.05, 0.05)] * 20
        still_res = gesture.analyze_hands_in_danger_zones([hand], nose, chin, w, h, min_pts=2)
        shaking_res = gesture.analyze_hands_in_danger_zones([hand], nose, chin, w, h, min_pts=1)

        self.assertFalse(still_res["near_any"])
        self.assertTrue(shaking_res["near_chest"])
        self.assertEqual(shaking_res["hands_in_zone"], 1)
        self.assertEqual(shaking_res["chest_radius"], geom["chest_radius"])
        self.assertEqual(shaking_res["throat_radius"], geom["throat_radius"])


class FaceLossWarningGuardTests(unittest.TestCase):
    def test_prior_candidate_warns_once_after_grace(self):
        guard = FaceLossWarningGuard(grace_sec=0.6)

        self.assertFalse(guard.update(True, False, 10.0))
        self.assertFalse(guard.update(False, True, 10.0))
        self.assertFalse(guard.update(False, True, 10.6))
        self.assertTrue(guard.update(False, True, 10.61))
        self.assertFalse(guard.update(False, True, 11.0))

    def test_candidate_starting_after_face_loss_does_not_warn(self):
        guard = FaceLossWarningGuard(grace_sec=0.6)

        self.assertFalse(guard.update(True, False, 10.0))
        self.assertFalse(guard.update(False, False, 10.1))
        self.assertFalse(guard.update(False, True, 10.8))

    def test_face_reacquisition_resets_warning_episode(self):
        guard = FaceLossWarningGuard(grace_sec=0.6)

        guard.update(True, False, 10.0)
        self.assertFalse(guard.update(False, True, 10.0))
        self.assertTrue(guard.update(False, True, 10.61))
        self.assertFalse(guard.update(True, False, 10.7))
        self.assertFalse(guard.update(False, True, 11.0))
        self.assertTrue(guard.update(False, True, 11.61))


class DesktopL4WiringTests(unittest.TestCase):
    def test_main_loop_uses_dropout_timer_and_face_loss_visibility_guard(self):
        self.assertIn("choke_timer.update(", MAIN_SOURCE)
        self.assertIn("choke_double_active, current_time)", MAIN_SOURCE)
        self.assertIn("visibility_warning_guard.update(", MAIN_SOURCE)
        self.assertIn("dispatch_visibility_warning(", MAIN_SOURCE)
        self.assertIn("CHOKE_DOUBLE_TRIGGER_SEC = 2.5", MAIN_SOURCE)
        self.assertIn("last_cough_time = -1e9", MAIN_SOURCE)
        self.assertIn("gesture.danger_zone_geometry(", MAIN_SOURCE)
        self.assertIn("evaluate_l4_candidate(", MAIN_SOURCE)

    def test_visibility_warning_notifies_and_records_separately_from_confirmed_l4(self):
        calls = {"line": [], "popup": [], "records": []}

        def notify(message, alert_type):
            calls["line"].append((message, alert_type))

        def popup(title, message):
            calls["popup"].append((title, message))

        def record(patient_id, patient_name, chew_count, status):
            calls["records"].append((patient_id, patient_name, chew_count, status))

        message = dispatch_visibility_warning(
            notify, popup, record, "P001", "測試個案", 12
        )

        self.assertIn("尚未確認", message)
        self.assertEqual(calls["line"][0][1], "choke_visibility")
        self.assertIn("目視確認", calls["popup"][0][0])
        self.assertEqual(calls["records"][0][3], "疑似哽噎_視覺遮蔽需確認")
        self.assertNotEqual(calls["records"][0][3], "緊急哽噎警報(雙重動作5秒)")

    def test_sensitivity_profile_thresholds_and_wiring(self):
        self.assertEqual(THRESHOLDS["AUDIO_CHOKE_TH_FUSION"], 0.12)
        self.assertEqual(THRESHOLDS["AUDIO_COUGH_RECORD_TH"], 0.18)
        self.assertEqual(THRESHOLDS["AUDIO_CHOKE_TH_SOLO"], 0.28)
        self.assertEqual(THRESHOLDS["AUDIO_SOLO_CONFIRM"], 1)
        self.assertEqual(THRESHOLDS["COUGH_STD_TH"], 15.0)
        self.assertEqual(THRESHOLDS["COUGH_VEL_TH"], 8.0)
        self.assertEqual(THRESHOLDS["COUGH_CONFIRM_FRAMES"], 4)
        self.assertEqual(THRESHOLDS["POUCHING_HINT_SEC"], 10.0)
        self.assertIn("check_continuous_body_shaking(", MAIN_SOURCE)
        self.assertIn("jaw_stillness_tracker.update(", MAIN_SOURCE)
        self.assertIn("l1_pause_reminder_gate.update(", MAIN_SOURCE)
        self.assertIn("AUDIO_COUGH_RECORD_TH", MAIN_SOURCE)

    def test_body_shaking_and_l1_pause_tracker_match_online_rules(self):
        self.assertEqual(count_direction_reversals([10, 15, 11, 16], 2.0), 2)
        history = []
        w, h = 640, 480
        face_size = 100.0
        # Oscillating nose motion across >0.45s with span > 0.12 and speed > 1.2
        pts = [
            (0.50, 0.50, 0.0),
            (0.56, 0.50, 0.2),
            (0.46, 0.50, 0.4),
            (0.55, 0.50, 0.6),
        ]
        shaking = False
        for x, y, t in pts:
            shaking = check_continuous_body_shaking(history, (x, y), face_size, t, w, h)
        self.assertTrue(shaking)

        tracker = ChewingPauseTracker()
        gate = ThresholdCrossingGate()
        self.assertEqual(tracker.update(True, True, True, 0.0), 0.0)
        t_val = 0.0
        for step in range(1, 51):
            t_val = step * 0.25
            still = tracker.update(True, True, False, t_val)
        self.assertGreaterEqual(still, 10.0)
        self.assertTrue(gate.update(still, 10.0))
        self.assertFalse(gate.update(still + 0.25, 10.0))


if __name__ == "__main__":
    unittest.main()
