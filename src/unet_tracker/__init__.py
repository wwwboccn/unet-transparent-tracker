#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net 全图形自适应透明旋转目标实时追踪系统 (V3.7 无偏几何中心精确定位 + CUDA + 远程算力版)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2

cv2.setNumThreads(4)

from .tracker import CudaAcceleratorBackend, UniversalUNetTracker
from .remote import RemoteTrackerClient, RemoteTrackerServer
from .visualizer import draw_unet_overlay
from .capture import ScreenRegionCapturer
from .hotkeys import SystemGlobalHotkeyManager, parse_hotkey_string
from .mouse_controller import ScreenMouseController

__version__ = "3.7.0"
__all__ = [
    "CudaAcceleratorBackend",
    "UniversalUNetTracker",
    "RemoteTrackerClient",
    "RemoteTrackerServer",
    "draw_unet_overlay",
    "ScreenRegionCapturer",
    "SystemGlobalHotkeyManager",
    "parse_hotkey_string",
    "ScreenMouseController",
]
