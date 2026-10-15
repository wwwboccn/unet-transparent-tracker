#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视频自动化基准评测脚本：测试 30FPS 满帧速度/锁定率与 10FPS 真实时钟跳帧鲁棒性。
用法: python scripts/benchmark.py video1.mp4 video2.mp4 ...
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from unet_tracker.cli import process_video_cli


def run_benchmarks(video_paths):
    print("=== 1. 30 FPS 满帧逐帧推理基准测试 (Turbo 模式) ===")
    for vp in video_paths:
        process_video_cli(vp, perf_mode="turbo")

    print("\n=== 2. 10 FPS 真实时钟同步跳帧鲁棒性测试 (dt=0.10s, n_step=3.0x) ===")
    for vp in video_paths:
        process_video_cli(vp, perf_mode="turbo", sim_clock_fps=10.0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/benchmark.py <video1.mp4> [video2.mp4 ...]")
        sys.exit(0)
    run_benchmarks(sys.argv[1:])
