#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行视频追踪推理、远程算力服务端启动與 GUI 入口 (V3.0)。
"""

import argparse
import glob
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .gui import WinTrackerGUIApp
from .remote import RemoteTrackerClient, RemoteTrackerServer
from .tracker import UniversalUNetTracker
from .visualizer import draw_unet_overlay

# ==============================================================================
# 7. 命令行单视频 & 多样本批量测试接口
# ==============================================================================
def process_video_cli(
    input_path: str,
    output_path: Optional[str] = None,
    perf_mode: str = "ultra",
    sim_clock_fps: Optional[float] = None,
    device: str = "auto",
    gpu_id: int = 0,
    remote_url: Optional[str] = None,
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

    if remote_url or device == "remote":
        tracker = RemoteTrackerClient(server_url=(remote_url or "ws://127.0.0.1:8765/ws"), fps=fps, perf_mode=perf_mode)
        tracker.reset()
    else:
        tracker = UniversalUNetTracker(fps=fps, perf_mode=perf_mode, device=device, gpu_id=gpu_id)
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
    end_pos = (states[-1]["cx"], states[-1]["cy"]) if states else (0.0, 0.0)
    summary = {
        "video": os.path.basename(input_path),
        "shape": tracker.shape_desc,
        "total_frames": len(states),
        "trans_frames": len(trans_states),
        "lock_rate": lock_rate,
        "mean_snr": mean_snr,
        "mean_ms": mean_ms,
        "fps": 1000.0 / max(0.1, mean_ms),
        "end_pos": end_pos,
    }
    print(
        f"[{summary['video']}] Device={getattr(tracker, 'remote_device_name', getattr(getattr(tracker, 'cuda_backend', None), 'device_name', 'CPU'))} | Shape={summary['shape']} | "
        f"LockRate={len(locked)}/{len(trans_states)} ({lock_rate:.2f}%) | "
        f"MeanSNR={mean_snr:.2f} | EndPos=({end_pos[0]:.1f},{end_pos[1]:.1f}) | "
        f"Speed={mean_ms:.2f}ms/frame ({summary['fps']:.1f} FPS)"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="全图形自适应 U-Net 透明目标追踪系统 V3.0 (远程算力 HTTP/WebSocket + RTX 3060/T4 CUDA 加速 + 精度优先 Ultra + 全局热键 + 鼠标跟随版)"
    )
    parser.add_argument("--gui", action="store_true", help="启动 Windows 可视化 GUI 操作台 (支持本地与远程算力一键切换)")
    parser.add_argument("--server", action="store_true", help="启动远程 GPU 算力服务端 (同时监听 HTTP 与 WebSocket)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="远程算力服务端监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="远程算力服务端监听端口 (默认 8765)")
    parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="客户端调用的远程算力地址 (例如 'ws://192.168.1.100:8765/ws' 或 'http://192.168.1.100:8765')",
    )
    parser.add_argument("--video", type=str, nargs="*", default=None, help="输入一个或多个测试视频路径（或包含视频的文件夹路径）")
    parser.add_argument("--output", type=str, default=None, help="输出带追踪叠加的 MP4 视频路径")
    parser.add_argument("--mode", type=str, default="ultra", choices=["ultra", "balanced", "turbo"], help="性能档位: ultra(默认精度优先), balanced, turbo")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算加速设备: 'cuda' (RTX 3060 / T4 显卡), 'remote' (调用远程算力), 'auto' (自动), 或 'cpu'",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="指定 CUDA 显卡序号 (默认 0)")
    parser.add_argument("--sim-clock-fps", type=float, default=None, help="模拟低算力时钟跳帧同步测试")
    args = parser.parse_args()

    if args.server:
        srv_dev = args.device if args.device != "auto" else "cuda"
        srv = RemoteTrackerServer(
            host=args.host, port=args.port, device=srv_dev, gpu_id=args.gpu_id, perf_mode=args.mode
        )
        srv.start_forever()
        return

    if args.gui or not args.video:
        first_vid = args.video[0] if (args.video and len(args.video) > 0 and os.path.isfile(args.video[0])) else None
        init_dev = "remote" if args.remote_url else args.device
        app = WinTrackerGUIApp(
            initial_video=first_vid,
            device=init_dev,
            gpu_id=args.gpu_id,
            remote_url=(args.remote_url or "ws://127.0.0.1:8765/ws"),
        )
        app.run()
    else:
        video_list: List[str] = []
        for v_arg in args.video:
            if os.path.isdir(v_arg):
                video_list.extend(sorted(glob.glob(os.path.join(v_arg, "*.mp4"))))
            elif os.path.isfile(v_arg):
                video_list.append(v_arg)
            else:
                video_list.extend(sorted(glob.glob(v_arg)))
        for idx, vp in enumerate(video_list):
            out_p = args.output if (len(video_list) == 1) else None
            process_video_cli(
                vp, out_p, perf_mode=args.mode, sim_clock_fps=args.sim_clock_fps,
                device=args.device, gpu_id=args.gpu_id, remote_url=args.remote_url
            )



if __name__ == "__main__":
    main()
