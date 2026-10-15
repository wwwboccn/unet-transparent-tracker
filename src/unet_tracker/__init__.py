#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net 全图形自适应透明旋转目标实时追踪系统 (V2.3)
"""

import os

# 针对 6 核 CPU (如 Intel Core i5-9400F) 默认绑定 4 个计算线程，预留核心给系统 GUI 与鼠标控制
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2

cv2.setNumThreads(4)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

from .tracker import UniversalUNetTracker
from .visualizer import draw_unet_overlay
from .capture import ScreenRegionCapturer
from .hotkeys import SystemGlobalHotkeyManager, parse_hotkey_string
from .mouse_controller import ScreenMouseController

__version__ = "2.3.0"
__all__ = [
    "UniversalUNetTracker",
    "draw_unet_overlay",
    "ScreenRegionCapturer",
    "SystemGlobalHotkeyManager",
    "parse_hotkey_string",
    "ScreenMouseController",
]
