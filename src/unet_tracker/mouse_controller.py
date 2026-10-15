#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 鼠标实时控制器：将系统鼠标移动到屏幕对应的追踪中心点坐标 (screen_x, screen_y)。
支持 Win32 SetCursorPos 与 SendInput (MOUSEEVENTF_ABSOLUTE) 双驱动模式、速度前馈补偿与像素微调。
"""

import ctypes
import platform
from typing import Tuple


class ScreenMouseController:
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.user32 = ctypes.windll.user32 if self.is_windows else None
        self.enabled = False
        self.driver_mode = "setcursorpos"  # "setcursorpos" | "sendinput"
        self.only_transparent = False      # True=仅透明阶段移动鼠标, False=白色+透明全程跟随
        self.lead_ms = 15.0                # 速度前馈补偿毫秒数
        self.offset_x = 0                  # 屏幕 X 偏移微调 (px)
        self.offset_y = 0                  # 屏幕 Y 偏移微调 (px)
        self.last_screen_pos: Tuple[int, int] = (0, 0)

    def move_to_screen_point(self, screen_x: float, screen_y: float) -> Tuple[int, int]:
        ix = int(round(screen_x)) + int(self.offset_x)
        iy = int(round(screen_y)) + int(self.offset_y)

        if self.is_windows and self.user32 is not None:
            sw = max(1, int(self.user32.GetSystemMetrics(0)))
            sh = max(1, int(self.user32.GetSystemMetrics(1)))
            ix = max(2, min(sw - 3, ix))
            iy = max(2, min(sh - 3, iy))

            if self.driver_mode == "sendinput":
                self._send_input_absolute(ix, iy, sw, sh)
            else:
                self.user32.SetCursorPos(ix, iy)

        self.last_screen_pos = (ix, iy)
        return ix, iy

    def _send_input_absolute(self, x: int, y: int, sw: int, sh: int) -> None:
        if not self.is_windows or self.user32 is None:
            return
        try:
            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_ABSOLUTE = 0x8000
            norm_x = int(round((x * 65535.0) / max(1, sw - 1)))
            norm_y = int(round((y * 65535.0) / max(1, sh - 1)))
            self.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, norm_x, norm_y, 0, 0)
        except Exception:
            self.user32.SetCursorPos(x, y)
