#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行推理、视频导出与 GUI 启动入口。
"""

import argparse
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .gui import WinTrackerGUIApp
from .tracker import UniversalUNetTracker
from .visualizer import draw_unet_overlay


def process_video_cli(
    input_path: str,
    output_path: Optional[str] = None,
    perf_mode: str = "turbo",
    sim_clock_fps: Optional[float] = None,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    tracker = UniversalUNetTracker(fps=fps, perf_mode=perf_mode)
    trail: List[Tuple[float, float]] = []
    states: List[dict] = []

    stride = max(1, int(round(fps / sim_clock_fps))) if sim_clock_fps else 1
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < 170 or idx % stride == 0:
            dt_sim = (stride / fps) if idx >= 170 else (1.0 / fps)
            st = tracker.update(frame, dt_sec=dt_sim)
            states.append(st)
            trail.append((st["cx"], st["cy"]))
            if writer is not None:
                vis = draw_unet_overlay(frame, st, trail, tracker, show_heatmap=True, show_hud=True)
                writer.write(vis)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    trans_states = [s for s in states if s["phase"] != "WHITE"]
    locked = [s for s in trans_states if s["conf"] >= 0.60]
    lock_rate = (len(locked) / max(1, len(trans_states))) * 100.0
    mean_snr = float(np.mean([s["snr"] for s in trans_states])) if trans_states else 0.0
    mean_ms = float(np.mean([s["latency_ms"] for s in trans_states])) if trans_states else 0.0
    summary = {
        "video": os.path.basename(input_path),
        "shape": tracker.shape_desc,
        "total_frames": len(states),
        "trans_frames": len(trans_states),
        "lock_rate": lock_rate,
        "mean_snr": mean_snr,
        "mean_ms": mean_ms,
        "fps": 1000.0 / max(0.1, mean_ms),
    }
    print(
        f"[{summary['video']}] Shape={summary['shape']} | "
        f"LockRate={len(locked)}/{len(trans_states)} ({lock_rate:.2f}%) | "
        f"MeanSNR={mean_snr:.2f} | Speed={mean_ms:.2f}ms/frame ({summary['fps']:.1f} FPS)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="全图形自适应 U-Net 透明目标追踪系统 V2.3")
    parser.add_argument("--gui", action="store_true", help="启动 Windows 可视化 GUI 界面")
    parser.add_argument("--video", type=str, default=None, help="输入测试视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出带追踪叠加的 MP4 视频路径")
    parser.add_argument("--mode", type=str, default="turbo", choices=["turbo", "balanced"], help="性能档位: turbo(默认60FPS) 或 balanced")
    parser.add_argument("--sim-clock-fps", type=float, default=None, help="模拟低算力时钟跳帧同步测试")
    args = parser.parse_args()

    if args.gui or (args.video is None):
        app = WinTrackerGUIApp(initial_video=args.video)
        app.run()
    else:
        process_video_cli(args.video, args.output, perf_mode=args.mode, sim_clock_fps=args.sim_clock_fps)


if __name__ == "__main__":
    main()
