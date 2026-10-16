#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 屏幕鼠标坐标实时控制器 (支持 SetCursorPos 与底层 SendInput)。
"""

import ctypes
import platform
from typing import Tuple

# ==============================================================================
# 2. Windows 鼠标实时控制器 (将鼠标精准移动到屏幕对应追踪中心点)
# ==============================================================================
class ScreenMouseController:
    """控制系统鼠标实时移动到屏幕对应的追踪中心点坐标 (screen_x, screen_y)"""

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.user32 = ctypes.windll.user32 if self.is_windows else None
        self.enabled = False
        self.driver_mode = "setcursorpos"  # "setcursorpos" | "sendinput"
        self.only_transparent = False
        self.lead_ms = 15.0
        self.offset_x = 0
        self.offset_y = 0
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


