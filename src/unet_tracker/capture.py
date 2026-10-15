#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零开销 Windows 屏幕区域捕获器：优先复用持久化 MSS 实例，自动回退至 PIL.ImageGrab 或原生 Win32 GDI BitBlt。
"""

import ctypes
import platform
from typing import Optional, Tuple
import cv2
import numpy as np


class ScreenRegionCapturer:
    def __init__(self):
        self._mss_inst = None
        try:
            import mss
            self._mss_inst = mss.mss()
        except Exception:
            self._mss_inst = None

    def grab_bgr(self, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        w = max(2, (x2 - x1) & ~1)
        h = max(2, (y2 - y1) & ~1)

        if self._mss_inst is not None:
            try:
                mon = {"left": int(x1), "top": int(y1), "width": int(w), "height": int(h)}
                shot = np.array(self._mss_inst.grab(mon), dtype=np.uint8)
                return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(bbox=(int(x1), int(y1), int(x1 + w), int(y1 + h)))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        if platform.system() == "Windows":
            return self._grab_win32_gdi(int(x1), int(y1), int(w), int(h))
        return None

    @staticmethod
    def _grab_win32_gdi(x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            hwnd = user32.GetDesktopWindow()
            hdc_screen = user32.GetWindowDC(hwnd)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            gdi32.SelectObject(hdc_mem, hbmp)
            SRCCOPY = 0x00CC0020
            gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, SRCCOPY)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0

            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bmi), 0)
            gdi32.DeleteObject(hbmp)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(hwnd, hdc_screen)

            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None
