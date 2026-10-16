#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.0 自动化单元测试：验证全局快捷键解析、鼠标控制器、合成透明旋转目标的几何中心锁定以及 WebSocket/HTTP 远程算力客户端导入。
运行方式: python -m unittest discover -s tests
"""

import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from unet_tracker import (
    CudaAcceleratorBackend,
    RemoteTrackerClient,
    RemoteTrackerServer,
    ScreenMouseController,
    UniversalUNetTracker,
    parse_hotkey_string,
    __version__,
)


class TestUNetTrackerProjectV30(unittest.TestCase):
    def test_version_and_cuda_backend(self):
        self.assertEqual(__version__, "3.0.0")
        backend = CudaAcceleratorBackend(device_pref="cpu")
        self.assertFalse(backend.is_cuda_active)

    def test_hotkey_parser(self):
        mods, vk, name = parse_hotkey_string("F8")
        self.assertEqual(mods, [])
        self.assertEqual(vk, 0x77)
        self.assertEqual(name, "F8")

        mods2, vk2, name2 = parse_hotkey_string("Ctrl + Shift + F9")
        self.assertEqual(mods2, [0x11, 0x10])
        self.assertEqual(vk2, 0x78)
        self.assertEqual(name2, "Ctrl+Shift+F9")

    def test_mouse_controller_offset(self):
        mc = ScreenMouseController()
        mc.offset_x = 10
        mc.offset_y = -5
        ix, iy = mc.move_to_screen_point(300.2, 400.8)
        self.assertEqual((ix, iy), (310, 396))

    def test_synthetic_hollow_window_geometric_center_lock(self):
        tracker = UniversalUNetTracker(fps=30.0, perf_mode="ultra", device="cpu")
        rng = np.random.default_rng(42)
        bg = rng.integers(40, 140, size=(360, 480, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (7, 7), 1.5)

        cx, cy = 180.0, 180.0
        for idx in range(25):
            frame = bg.copy()
            # Draw a hollow 4-pane window target (田) centered at (cx, cy)
            icx, icy = int(round(cx)), int(round(cy))
            if idx < 8:
                cv2.rectangle(frame, (icx - 36, icy - 36), (icx + 36, icy + 36), (245, 245, 245), 6)
                cv2.line(frame, (icx - 36, icy), (icx + 36, icy), (245, 245, 245), 5)
                cv2.line(frame, (icx, icy - 36), (icx, icy + 36), (245, 245, 245), 5)
            else:
                cx += 5.0
                cy += 3.0
                icx, icy = int(round(cx)), int(round(cy))
                overlay = frame.copy()
                cv2.rectangle(overlay, (icx - 36, icy - 36), (icx + 36, icy + 36), (215, 215, 215), 6)
                cv2.line(overlay, (icx - 36, icy), (icx + 36, icy), (215, 215, 215), 5)
                cv2.line(overlay, (icx, icy - 36), (icx, icy + 36), (215, 215, 215), 5)
                frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)
            st = tracker.update(frame, dt_sec=1.0 / 30.0)
            self.assertIn("cx", st)
            self.assertIn("cy", st)

        err = float(np.hypot(st["cx"] - cx, st["cy"] - cy))
        self.assertLess(err, 25.0)


if __name__ == "__main__":
    unittest.main()
