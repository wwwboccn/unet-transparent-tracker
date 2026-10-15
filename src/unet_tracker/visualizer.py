#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化渲染模块：任意图形多边形轮廓 + U-Net 概率热力图 + 洋红鼠标锁定准星与实时 HUD
"""

from typing import List, Tuple
import cv2
import numpy as np
from .tracker import UniversalUNetTracker


def draw_unet_overlay(
    frame_bgr: np.ndarray,
    state: dict,
    trail: List[Tuple[float, float]],
    tracker: UniversalUNetTracker,
    show_heatmap: bool = False,
    show_hud: bool = True,
    clock_drift_sec: float = 0.0,
    mouse_active: bool = False,
    screen_xy: Tuple[int, int] = (0, 0),
    hotkey_hint: str = "F8:Run/Pause | F9:Mouse | F10:Stop",
) -> np.ndarray:
    canvas = frame_bgr.copy()
    H, W = canvas.shape[:2]

    if show_heatmap and tracker.last_unet_prob_s is not None and state["phase"] != "WHITE":
        prob_full = cv2.resize(tracker.last_unet_prob_s, (W, H), interpolation=cv2.INTER_LINEAR)
        prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_TURBO)
        active = prob_full > 0.24
        if np.any(active):
            canvas[active] = cv2.addWeighted(canvas[active], 0.58, heat[active], 0.42, 0.0)

    if len(trail) >= 2:
        pts_tr = np.array(trail[-65:], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_tr], False, (0, 215, 255), 2, cv2.LINE_AA)

    cx, cy = state["cx"], state["cy"]
    w, h, ang = state["w"], state["h"], state["angle"]
    color = (0, 255, 0) if state["phase"] == "WHITE" else (
        (0, 255, 255) if state["phase"] == "TRANSPARENT" else (0, 140, 255)
    )

    poly = tracker.get_current_polygon_full(cx, cy, ang)
    cv2.drawContours(canvas, [poly], 0, color, 2, cv2.LINE_AA)

    box = np.int32(cv2.boxPoints(((cx, cy), (w, h), ang)))
    cv2.drawContours(canvas, [box], 0, (180, 180, 180), 1, cv2.LINE_AA)

    icx, icy = int(round(cx)), int(round(cy))
    cv2.circle(canvas, (icx, icy), 4, (0, 0, 255), -1, cv2.LINE_AA)
    if mouse_active:
        cv2.circle(canvas, (icx, icy), 14, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (icx - 20, icy), (icx + 20, icy), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (icx, icy - 20), (icx, icy + 20), (255, 0, 255), 1, cv2.LINE_AA)

    rad = np.radians(ang)
    hx = int(round(cx + 0.45 * w * np.cos(rad)))
    hy = int(round(cy + 0.45 * w * np.sin(rad)))
    cv2.arrowedLine(canvas, (icx, icy), (hx, hy), (255, 80, 255), 2, cv2.LINE_AA, tipLength=0.25)

    if show_hud:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (575, 138), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        fps_inst = 1000.0 / max(0.5, state["latency_ms"])
        n_step = state.get("n_step", 1.0)
        mouse_tag = f"MOUSE LOCKED -> ({screen_xy[0]},{screen_xy[1]})" if mouse_active else "MOUSE OFF"
        mouse_col = (255, 100, 255) if mouse_active else (170, 170, 170)

        cv2.putText(
            canvas,
            f"Frame #{state['frame_idx']:03d} | {state['phase']} | {state['shape']}",
            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Local:({cx:5.1f},{cy:5.1f}) | Angle:{ang:5.1f}deg | {mouse_tag}",
            (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mouse_col, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"SNR:{state['snr']:4.2f} | Conf:{state['conf']*100:4.1f}% | {state['latency_ms']:4.1f}ms ({fps_inst:4.1f} FPS) | Step:{n_step:3.1f}x",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (120, 255, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Global Hotkeys: {hotkey_hint}",
            (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 225, 255), 1, cv2.LINE_AA
        )
    return canvas
