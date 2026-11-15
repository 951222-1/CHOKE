import json
import unittest
from pathlib import Path

from l4_safety import check_continuous_body_shaking

ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = (ROOT / "防哽咽裝置4_2.py").read_text(encoding="utf-8")
THRESHOLDS = json.loads((ROOT / "shared" / "thresholds.json").read_text(encoding="utf-8"))


class L4SensitivityProfileTests(unittest.TestCase):
    def test_shorter_trigger_and_dropout_settings_are_present(self):
        self.assertIn("CHOKE_DOUBLE_TRIGGER_SEC = 2.5", MAIN_SOURCE)
        self.assertIn("CHOKE_DOUBLE_GRACE_SEC = 0.9", MAIN_SOURCE)
        self.assertIn("CHOKE_DECAY_RATE = 0.25", MAIN_SOURCE)
        self.assertIn("POST_COUGH_WINDOW_SEC = 12.0", MAIN_SOURCE)

    def test_body_motion_between_old_and_trial_speed_thresholds_is_detected(self):
        history = []
        w, h, face_size = 640, 480, 100.0
        x_positions_px = [320, 335, 320, 335, 320, 335, 320, 335]
        result = False
        for index, x_px in enumerate(x_positions_px):
            t = index * 0.08
            result = check_continuous_body_shaking(
                history, (x_px / w, 0.50), face_size, t, w, h
            )
        self.assertTrue(result)

    def test_visual_cough_trial_values_are_present(self):
        self.assertEqual(THRESHOLDS["COUGH_STD_TH"], 15.0)
        self.assertEqual(THRESHOLDS["COUGH_VEL_TH"], 8.0)
        self.assertEqual(THRESHOLDS["COUGH_CONFIRM_FRAMES"], 4)


if __name__ == "__main__":
    unittest.main()
