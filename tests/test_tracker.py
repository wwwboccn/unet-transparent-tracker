#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化单元测试：验证全局快捷键解析、屏幕鼠标坐标控制器以及合成透明旋转目标的 U-Net 追踪闭环。
运行方式: python -m unittest discover -s tests
"""

import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from unet_tracker import (
    ScreenMouseController,
    UniversalUNetTracker,
    parse_hotkey_string,
)


class TestUNetTrackerProject(unittest.TestCase):
    def test_hotkey_parser(self):
        mods, vk, name = parse_hotkey_string("F8")
        self.assertEqual(mods, [])
        self.assertEqual(vk, 0x77)
        self.assertEqual(name, "F8")

        mods2, vk2, name2 = parse_hotkey_string("Ctrl + Shift + F9")
        self.assertEqual(mods2, [0x11, 0x10])
        self.assertEqual(vk2, 0x78)
        self.assertEqual(name2, "Ctrl+Shift+F9")

        mods3, vk3, name3 = parse_hotkey_string("MOUSE_X1")
        self.assertEqual(mods3, [])
        self.assertEqual(vk3, 0x05)
        self.assertEqual(name3, "MOUSE_X1")

    def test_mouse_controller_offset(self):
        mc = ScreenMouseController()
        mc.offset_x = 10
        mc.offset_y = -5
        ix, iy = mc.move_to_screen_point(300.2, 400.8)
        self.assertEqual((ix, iy), (310, 396))

    def test_synthetic_transparent_polygon_tracking(self):
        tracker = UniversalUNetTracker(fps=30.0, perf_mode="turbo")
        rng = np.random.default_rng(42)
        bg = rng.integers(40, 140, size=(360, 480, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (7, 7), 1.5)

        cx, cy = 180.0, 180.0
        for idx in range(25):
            frame = bg.copy()
            pts = np.array([
                [cx, cy - 45],
                [cx + 42, cy],
                [cx, cy + 45],
                [cx - 42, cy],
            ], dtype=np.int32)
            if idx < 8:
                cv2.fillPoly(frame, [pts], (245, 245, 245))
            else:
                cx += 5.0
                cy += 3.0
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (210, 210, 210))
                frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            st = tracker.update(frame, dt_sec=1.0 / 30.0)
            self.assertIn("cx", st)
            self.assertIn("cy", st)

        err = float(np.hypot(st["cx"] - cx, st["cy"] - cy))
        self.assertLess(err, 35.0)


if __name__ == "__main__":
    unittest.main()
