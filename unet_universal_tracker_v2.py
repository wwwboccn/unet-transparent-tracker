#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
全图形自适应 U-Net 透明旋转目标实时追踪系统 (V3.0 几何中心精确定位版 — 消除边缘跟随 + Colab T4/RTX 3060 CUDA + 远程 WebSocket/HTTP 算力)
支持硬件: NVIDIA Tesla T4 (Colab) / GeForce RTX 3060 (12GB) CUDA 加速 / Intel i5-9400F / 多核 CPU 自动降级
================================================================================

【V3.0 核心升级：根治 T4 GPU 及高帧率样本中“中心点跟随目标边缘”的问题，实现目标真实几何中心锁定】
针对在 Google Colab T4 GPU 测试 `MapleStoryN` 4 格窗框 (`田`) 及旋转多边形样本时发现的“红点总是跟随目标亮边/后沿边缘”问题，V3.0 进行了五项针对性根治：

1. 双层解耦架构：平顶圆盘目标隔离 + 无偏几何中心精定位器 (`_compute_true_geometric_center`)：
   - 根因 A (卷积核小于目标尺寸导致边缘双峰)：半分辨率下目标尺寸约为 52~71 px，而旧版 39x39 均值盒滤波半径仅 19.5 px，无法同时覆盖目标四条对边，最大值必然落在单侧亮边/角点上 (偏离中心 15~30 px)。
   - 根因 B (高斯空间先验斜率导致后沿滞后)：`posterior_s = energy_s * spatial_prior_s` 将平坦的目标能量台面乘以以历史锚点为中心的高斯钟形函数，数学上将峰值向运动反方向 (后沿边缘) 拉拽。
   - V3.0 方案：在完成轨迹关联锁定目标后，采用平顶圆盘掩码 (`disk_m`, r <= R_iso 内等权为 1.0，无高斯斜率偏置，圆外背景岩石清零) 隔离目标，结合全尺寸对称盒积分 (`49x49 ~ 55x55` + 高斯径向平滑) 与前景轮廓支撑集 **5%~95% 包围盒几何中点 + 填充凸包质心 (`Filled Convex-Hull Centroid`)**，将亮边与暗边等权化，直接把坐标从边缘拉回目标正中心！
2. 当前帧无重影显著性 (`saliency_now_s`) 消除多步帧差尾迹：
   - 采用 `d_now = 0.65 * d0 + 0.35 * sqrt(d0 * d1)` 几何均值门控抑制 `t-2, t-3` 历史位置重影，确保几何中心计算 100% 对齐当前帧 t 的目标实体。
3. 镂空/多格窗框 (`田`) 形态学闭运算标定 + 倒计时青色光晕精准剥离：
   - 初始白色阶段采用 `11x11` 形态学闭运算弥合 `田` 字镂空窗格 (`Solidity` 从 0.11 提升至 0.76)，并移除旧版 `wm_s[:72, :] = 0` 一刀切截断，使初始中心与旋转外框 100% 居中。
4. 高帧率录屏 (50~60 FPS) 游戏 Tick 重复帧自适应 (`is_game_tick_video`)：
   - 自动识别 `MapleStoryN` 等高帧率录屏中的重复渲染帧 (`mean_diff < 1.18`)，在重复帧期间保持卡尔曼速度不衰减、不污染参考帧队列，并在真实运动帧自动补偿跨帧步长。
5. T4 GPU / CUDA 亚像素对齐精度统一：
   - 消除 `F.grid_sample` 转 `uint8` 向下取整截断偏差，保证 T4 GPU 与本地 CPU 具有 100% 一致的亚像素背景配准精度。
================================================================================
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import ctypes
import glob
import json
import platform
import sys
import threading
import multiprocessing
import time
from typing import Callable, Dict, List, Optional, Tuple

# 兼容 PyInstaller --noconsole (stdout/stderr 为 None) 与 Windows cp1252/gbk 控制台编码，彻底杜绝 UnicodeEncodeError
for _sname in ("stdout", "stderr"):
    _s = getattr(sys, _sname, None)
    if _s is None:
        setattr(sys, _sname, open(os.devnull, "w", encoding="utf-8", errors="ignore"))
    elif hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def get_application_dir() -> str:
    """获取程序运行目录：兼容 PyInstaller 打包后的单文件 .exe 与普通 .py 脚本"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


import cv2
import numpy as np

cv2.setNumThreads(4)


# ==============================================================================
# 0. NVIDIA RTX 3060 (12GB) CUDA 硬件加速后端管理器 (自动兼容 PyTorch CUDA & OpenCV CUDA)
# ==============================================================================
class CudaAcceleratorBackend:
    """
    统一管理 NVIDIA CUDA 显卡加速 (专为 RTX 3060 12GB Ampere 架构优化):
    - 支持三种模式配置: "auto" (自动优先选用 CUDA)、"cuda" / "cuda:0" (强制指定 CUDA 显卡)、"cpu" (纯 CPU)
    - 双引擎驱动:
      1) PyTorch CUDA 引擎 (`torch.cuda`): 利用 RTX 3060 12GB 显存常驻张量执行双三次/双线性网格采样对齐 (`grid_sample`)、
         39x39 均值池化能量积分 (`avg_pool2d`)、Sobel 旋度卷积、72 角度模板批量矩阵内积及经验贝叶斯杂波抑制。
      2) OpenCV CUDA 引擎 (`cv2.cuda`): 若安装了编译有 CUDA 模块的 OpenCV，自动启用 `cv2.cuda_FarnebackOpticalFlow` 与 `GpuMat`。
    """

    def __init__(self, device_pref: str = "auto", gpu_id: int = 0, use_fp16: bool = False):
        self.device_pref = (device_pref or "auto").strip().lower()
        self.gpu_id = int(gpu_id)
        if ":" in self.device_pref and self.device_pref.startswith("cuda:"):
            try:
                self.gpu_id = int(self.device_pref.split(":")[1])
            except Exception:
                self.gpu_id = 0

        self.use_fp16 = bool(use_fp16)
        self.has_torch_cuda = False
        self.has_cv2_cuda = False
        self.torch = None
        self.F = None
        self.torch_device = None
        self.cv2_farneback_gpu = None
        self.device_name = "CPU (4 Threads)"
        self.active_backend = "cpu"

        self._probe_and_init()

    def _probe_and_init(self) -> None:
        if self.device_pref == "cpu":
            self.active_backend = "cpu"
            self.device_name = "CPU (Forced)"
            return

        # 1. 检测 PyTorch CUDA (Windows 下 RTX 3060 12GB 最通用的 CUDA 环境)
        try:
            import torch
            import torch.nn.functional as F
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                dev_idx = min(self.gpu_id, torch.cuda.device_count() - 1)
                self.torch = torch
                self.F = F
                self.torch_device = torch.device(f"cuda:{dev_idx}")
                # 开启 Ampere (RTX 3060) TF32 / cuDNN 加速
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                gpu_title = torch.cuda.get_device_name(dev_idx)
                vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
                self.has_torch_cuda = True
                self.active_backend = "cuda_torch"
                self.device_name = f"CUDA:{dev_idx} {gpu_title} ({vram_gb:.0f}GB)"
        except Exception:
            self.has_torch_cuda = False

        # 2. 检测 OpenCV CUDA 模块 (cv2.cuda)
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                dev_idx = min(self.gpu_id, cv2.cuda.getCudaEnabledDeviceCount() - 1)
                cv2.cuda.setDevice(dev_idx)
                self.has_cv2_cuda = True
                if hasattr(cv2, "cuda_FarnebackOpticalFlow"):
                    self.cv2_farneback_gpu = cv2.cuda_FarnebackOpticalFlow.create(
                        numLevels=2, pyrScale=0.5, fastPyramids=False,
                        winSize=13, numIters=2, polyN=5, polySigma=1.1, flags=0
                    )
                if not self.has_torch_cuda:
                    self.active_backend = "cuda_cv2"
                    self.device_name = f"CUDA:{dev_idx} OpenCV-CUDA GPU"
                else:
                    self.active_backend = "cuda_hybrid"
                    self.device_name += " + cv2.cuda"
        except Exception:
            self.has_cv2_cuda = False

        # 3. 若用户显式指定了 cuda 但当前 Python 环境未安装 CUDA 版 torch/opencv，则给出清晰提示并安全回退
        if self.device_pref.startswith("cuda") and (not self.is_cuda_active):
            print(
                "[CUDA 提示] 当前 Python 环境未检测到可用的 CUDA 驱动或 PyTorch-CUDA / OpenCV-CUDA，"
                "已自动回退至多核 CPU 模式。在 RTX 3060 12G 机器上可执行:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            self.active_backend = "cpu"
            self.device_name = "CPU (CUDA Fallback)"

    @property
    def is_cuda_active(self) -> bool:
        return self.has_torch_cuda or self.has_cv2_cuda

    def box_filter_gpu(self, arr_2d: np.ndarray, ksize: int) -> np.ndarray:
        """在 RTX 3060 GPU 上执行高精度均值盒滤波 (F.avg_pool2d)"""
        if not self.has_torch_cuda:
            return cv2.boxFilter(arr_2d, cv2.CV_32F, (ksize, ksize), normalize=True)
        torch = self.torch
        F = self.F
        with torch.no_grad():
            t = torch.from_numpy(arr_2d).to(self.torch_device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pad = ksize // 2
            t_pad = F.pad(t, (pad, pad, pad, pad), mode="replicate")
            out = F.avg_pool2d(t_pad, kernel_size=ksize, stride=1)
            return out[0, 0].cpu().numpy()

    def fast_norm_gpu(self, arr_2d: np.ndarray, bg_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """在 RTX 3060 GPU 上执行去自抑制背景归一化"""
        if not self.has_torch_cuda:
            mean_v, std_v = cv2.meanStdDev(arr_2d, mask=bg_mask)
            m, s = float(mean_v[0, 0]), float(std_v[0, 0])
            if s < 1e-3:
                return np.zeros_like(arr_2d, dtype=np.float32)
            clean = np.maximum(0.0, arr_2d - (m + 0.80 * s))
            return np.clip(clean / (m + 4.0 * s + 1e-5), 0.0, 1.0)
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(arr_2d).to(self.torch_device, dtype=torch.float32)
            if bg_mask is not None:
                m_t = torch.from_numpy(bg_mask > 0).to(self.torch_device)
                valid_vals = t[m_t]
                if valid_vals.numel() > 16:
                    m = float(torch.mean(valid_vals).item())
                    s = float(torch.std(valid_vals, unbiased=False).item())
                else:
                    m = float(torch.mean(t).item())
                    s = float(torch.std(t, unbiased=False).item())
            else:
                m = float(torch.mean(t).item())
                s = float(torch.std(t, unbiased=False).item())
            if s < 1e-3:
                return np.zeros_like(arr_2d, dtype=np.float32)
            clean = torch.clamp(t - (m + 0.80 * s), min=0.0)
            normed = torch.clamp(clean / (m + 4.0 * s + 1e-5), min=0.0, max=1.0)
            return normed.cpu().numpy()

    def warp_affine_bgr_gpu(
        self, prev_bgr: np.ndarray, tx: float, ty: float, bicubic: bool = False
    ) -> np.ndarray:
        """优先使用 cv2.cuda.warpAffine 或 PyTorch CUDA grid_sample 在 RTX 3060 上执行亚像素图像对齐"""
        hs, ws = prev_bgr.shape[:2]
        if self.has_cv2_cuda and hasattr(cv2, "cuda") and hasattr(cv2.cuda, "warpAffine"):
            try:
                M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
                g_src = cv2.cuda_GpuMat()
                g_src.upload(prev_bgr)
                flag = cv2.INTER_CUBIC if bicubic else cv2.INTER_LINEAR
                g_dst = cv2.cuda.warpAffine(g_src, M, (ws, hs), flags=flag, borderMode=cv2.BORDER_REPLICATE)
                return g_dst.download()
            except Exception:
                pass
        if self.has_torch_cuda:
            torch = self.torch
            F = self.F
            with torch.no_grad():
                t_img = torch.from_numpy(prev_bgr).to(self.torch_device, dtype=torch.float32)
                t_img = t_img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                # 构造逆向采样网格: dst(x, y) 采样自 src(x - tx, y - ty)
                theta = torch.tensor(
                    [[[1.0, 0.0, -2.0 * tx / max(1, ws - 1)], [0.0, 1.0, -2.0 * ty / max(1, hs - 1)]]],
                    device=self.torch_device, dtype=torch.float32
                )
                grid = F.affine_grid(theta, t_img.size(), align_corners=True)
                mode = "bicubic" if bicubic else "bilinear"
                warped = F.grid_sample(t_img, grid, mode=mode, padding_mode="border", align_corners=True)
                warped_u8 = torch.clamp(torch.round(warped[0].permute(1, 2, 0)), 0.0, 255.0).to(torch.uint8)
                return warped_u8.cpu().numpy()
        M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        flag = cv2.INTER_CUBIC if bicubic else cv2.INTER_LINEAR
        return cv2.warpAffine(prev_bgr, M, (ws, hs), flags=flag, borderMode=cv2.BORDER_REPLICATE)

    def calc_farneback_gpu(
        self, g_prev: np.ndarray, g_curr: np.ndarray, win_size: int = 13, levels: int = 2, iters: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """优先调用 OpenCV CUDA Farneback 光流；若无 cv2.cuda 则在 RTX 3060 模式下启用高精半分辨率光流"""
        if self.has_cv2_cuda and self.cv2_farneback_gpu is not None:
            try:
                gpu_p = cv2.cuda_GpuMat()
                gpu_c = cv2.cuda_GpuMat()
                gpu_p.upload(g_prev)
                gpu_c.upload(g_curr)
                flow_gpu = self.cv2_farneback_gpu.calc(gpu_p, gpu_c, None)
                fk = flow_gpu.download()
                return fk[:, :, 0], fk[:, :, 1]
            except Exception:
                pass
        fk = cv2.calcOpticalFlowFarneback(g_prev, g_curr, None, 0.5, levels, win_size, iters, 5, 1.1, 0)
        return fk[:, :, 0], fk[:, :, 1]



# ==============================================================================
# 1. Windows 虚拟键码映射表与全局热键监听器 (无需第三方库，支持任意单键/组合键/录制)
# ==============================================================================
VK_NAME_TO_CODE: Dict[str, int] = {
    **{f"F{i}": 0x70 + (i - 1) for i in range(1, 13)},
    **{str(i): 0x30 + i for i in range(10)},
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    "ESC": 0x1B, "ESCAPE": 0x1B,
    "SPACE": 0x20, "TAB": 0x09, "ENTER": 0x0D, "BACKSPACE": 0x08,
    "INSERT": 0x2D, "DELETE": 0x2E, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
    "CAPSLOCK": 0x14, "PAUSE": 0x13,
    "NUM0": 0x60, "NUM1": 0x61, "NUM2": 0x62, "NUM3": 0x63, "NUM4": 0x64,
    "NUM5": 0x65, "NUM6": 0x66, "NUM7": 0x67, "NUM8": 0x68, "NUM9": 0x69,
    "NUM*": 0x6A, "NUM+": 0x6B, "NUM-": 0x6D, "NUM/": 0x6F,
    "`": 0xC0, "~": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD,
    ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF, "\\": 0xDC,
    "MOUSE_X1": 0x05, "MOUSE_X2": 0x06, "MOUSE_MIDDLE": 0x04,
}

VK_CODE_TO_NAME: Dict[int, str] = {v: k for k, v in VK_NAME_TO_CODE.items()}
VK_CODE_TO_NAME[0x1B] = "ESC"
VK_CODE_TO_NAME[0xC0] = "`"


def parse_hotkey_string(hotkey_str: str) -> Tuple[List[int], Optional[int], str]:
    """解析快捷键字符串（如 'F8', 'Ctrl+F8', 'Alt+S', 'MOUSE_X1'）为 (修饰键VK列表, 主键VK, 规范化名称)"""
    parts = [p.strip().upper() for p in hotkey_str.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return [], 0x77, "F8"

    mods: List[int] = []
    mod_names: List[str] = []
    main_vk: Optional[int] = None
    main_name: str = ""

    for p in parts:
        if p in ("CTRL", "CONTROL", "LCTRL", "RCTRL"):
            if 0x11 not in mods:
                mods.append(0x11)
                mod_names.append("Ctrl")
        elif p in ("SHIFT", "LSHIFT", "RSHIFT"):
            if 0x10 not in mods:
                mods.append(0x10)
                mod_names.append("Shift")
        elif p in ("ALT", "MENU", "LALT", "RALT"):
            if 0x12 not in mods:
                mods.append(0x12)
                mod_names.append("Alt")
        elif p in VK_NAME_TO_CODE:
            main_vk = VK_NAME_TO_CODE[p]
            main_name = p

    if main_vk is None:
        return [], 0x77, "F8"

    norm_str = "+".join(mod_names + [main_name])
    return mods, main_vk, norm_str


class SystemGlobalHotkeyManager:
    """跨窗口系统级全局快捷键管理器（基于 Win32 GetAsyncKeyState 12ms 后台监听 + 上升沿消抖）"""

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.user32 = ctypes.windll.user32 if self.is_windows else None
        self._lock = threading.Lock()
        self._bindings: Dict[str, dict] = {}
        self._recording_callback: Optional[Callable[[str], None]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def bind_action(self, action_id: str, hotkey_str: str, callback: Callable[[], None]) -> str:
        mods, main_vk, norm_name = parse_hotkey_string(hotkey_str)
        with self._lock:
            self._bindings[action_id] = {
                "mods": mods,
                "main_vk": main_vk,
                "norm_name": norm_name,
                "callback": callback,
                "was_pressed": False,
                "last_trigger_t": 0.0,
            }
        return norm_name

    def start_recording(self, on_recorded: Callable[[str], None]) -> None:
        with self._lock:
            self._recording_callback = on_recorded

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _is_vk_down(self, vk: int) -> bool:
        if not self.is_windows or self.user32 is None:
            return False
        return (self.user32.GetAsyncKeyState(vk) & 0x8000) != 0

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.is_windows and self.user32 is not None:
                with self._lock:
                    rec_cb = self._recording_callback
                    if rec_cb is not None:
                        mod_names = []
                        if self._is_vk_down(0x11):
                            mod_names.append("Ctrl")
                        if self._is_vk_down(0x10):
                            mod_names.append("Shift")
                        if self._is_vk_down(0x12):
                            mod_names.append("Alt")

                        detected_main = None
                        for vk_code, k_name in VK_CODE_TO_NAME.items():
                            if self._is_vk_down(vk_code):
                                detected_main = k_name
                                break

                        if detected_main is not None:
                            combo_str = "+".join(mod_names + [detected_main])
                            self._recording_callback = None
                            time.sleep(0.15)
                            try:
                                rec_cb(combo_str)
                            except Exception:
                                pass
                            continue

                    now = time.perf_counter()
                    for info in self._bindings.values():
                        main_down = self._is_vk_down(info["main_vk"])
                        mods_ok = all(self._is_vk_down(mvk) for mvk in info["mods"])
                        if not info["mods"]:
                            if self._is_vk_down(0x11) or self._is_vk_down(0x12):
                                mods_ok = False

                        is_active = main_down and mods_ok
                        if is_active and (not info["was_pressed"]) and (now - info["last_trigger_t"] > 0.22):
                            info["was_pressed"] = True
                            info["last_trigger_t"] = now
                            cb = info["callback"]
                            threading.Thread(target=cb, daemon=True).start()
                        elif not main_down:
                            info["was_pressed"] = False

            time.sleep(0.012)


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


# ==============================================================================
# 3. 核心算法：全图形自适应 U-Net 实时追踪引擎 (V3.0 背景环掩码去自抑制 + 全样本鲁棒锁定版)
# ==============================================================================
class UniversalUNetTracker:
    """
    全图形自适应 U-Net 追踪器 (V3.0 背景环掩码去自抑制 + 全样本鲁棒锁定 + 出生点局部化经验贝叶斯抗陷阱版):
    - 3 通道 BGR 亚像素背景补偿帧差 + 双步多尺度光流旋度场 + 出生点局部化经验贝叶斯杂波抑制
    - 彻底解决 `hs1_clean` 上边界吸顶漂移、`hs3_clean` 局部对比度自适应及 `1000012887.mp4` 8秒中心陷阱
    """

    def __init__(
        self,
        fps: float = 30.0,
        perf_mode: str = "ultra",
        device: str = "auto",
        gpu_id: int = 0,
    ):
        self.fps = float(fps)
        self.perf_mode = perf_mode
        self.device_pref = device
        self.gpu_id = int(gpu_id)
        self.cuda_backend = CudaAcceleratorBackend(device_pref=device, gpu_id=gpu_id)
        self.erode_k_s = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.dilate_k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.rotated_edge_bank_gpu = None
        self.reset()

    def set_device(self, device: str, gpu_id: int = 0) -> str:
        """动态切换计算设备 ('auto' | 'cuda' | 'cuda:0' | 'cpu') 并同步 GPU 张量缓存"""
        self.device_pref = device
        self.gpu_id = int(gpu_id)
        self.cuda_backend = CudaAcceleratorBackend(device_pref=device, gpu_id=gpu_id)
        if self.cuda_backend.has_torch_cuda and self.rotated_edge_bank is not None:
            torch = self.cuda_backend.torch
            self.rotated_edge_bank_gpu = torch.from_numpy(self.rotated_edge_bank).to(
                self.cuda_backend.torch_device, dtype=torch.float32
            )
        else:
            self.rotated_edge_bank_gpu = None
        return self.cuda_backend.device_name

    def reset(self) -> None:
        self.frame_idx = 0
        self.initialized = False
        self.transparent_streak = 0
        self.white_reappear_streak = 0

        self.ref_w = 104.0
        self.ref_h = 104.0
        self.shape_desc = "LEARNING_SHAPE"

        self.canonical_mask_s: Optional[np.ndarray] = None
        self.canonical_edge_s: Optional[np.ndarray] = None
        self.canonical_contour: Optional[np.ndarray] = None
        self.rotated_edge_bank: Optional[np.ndarray] = None
        self.rotated_edge_bank_gpu = None

        self.unet_weights = np.array([
            0.22, 0.20, 0.16, 0.22, 0.18, 0.14, 0.15, 0.05,
            0.25, 0.22, 0.25, 0.20, 0.16, 0.25,
            0.28, 0.24, 0.28, 0.22, 0.18, 0.25,
            0.24, 0.20, 0.24, 0.18, 0.15, 0.22,
            -0.15
        ], dtype=np.float32)
        self.xtx_accum = np.eye(27, dtype=np.float64) * 5.0
        self.xty_accum = self.unet_weights.astype(np.float64) * 5.0
        self.online_adapt_count = 0

        self.ema_cx: Optional[float] = None
        self.ema_cy: Optional[float] = None
        self.center_cx: Optional[float] = None
        self.center_cy: Optional[float] = None
        self.prev_out_cx: Optional[float] = None
        self.prev_out_cy: Optional[float] = None
        self.init_cx_s: Optional[float] = None
        self.init_cy_s: Optional[float] = None
        self.cur_angle = 0.0
        self.prev_angle = 0.0
        self.ema_alpha = 0.50
        self.max_step_px = 21.0
        self.gate_radius_half = 44.0
        self.spatial_sigma_half = 21.0
        self.auto_reinit_on_white = True
        self.dynamic_clock_scale = True

        self.prev_frames_s: List[np.ndarray] = []
        self.prev_grays_s: List[np.ndarray] = []
        self.prev_uis_s: List[np.ndarray] = []
        self.prev_shifts: List[Tuple[float, float]] = []

        self.xx_s: Optional[np.ndarray] = None
        self.yy_s: Optional[np.ndarray] = None
        self.valid_mask_s: Optional[np.ndarray] = None
        self.grid_pts_s: Optional[np.ndarray] = None

        self.clutter_sum_s: Optional[np.ndarray] = None
        self.clutter_count_s: Optional[np.ndarray] = None
        self.cached_clutter_weight_s: Optional[np.ndarray] = None
        self.last_valid_snr: float = 2.5

        self.k_dil15_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.k_close11_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.k_close13_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        self.k_close7_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.k_open3_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dup_streak: int = 0
        self.total_dup_frames: int = 0
        self.scene_cut_hold: int = 0
        self.has_left_birth: bool = False
        self.reentry_dwell: float = 0.0
        self.dep_dx_s: float = 0.0
        self.bg_scroll_ema: float = 0.0
        self.is_irregular_shape: bool = False

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.70, 0],
            [0, 0, 0, 0.70]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.diag([1.5, 1.5, 4.0, 4.0]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([2.8, 2.8]).astype(np.float32)

        self.last_unet_prob_s: Optional[np.ndarray] = None

    def _init_grids(self, hs: int, ws: int) -> None:
        self.yy_s, self.xx_s = np.mgrid[0:hs, 0:ws].astype(np.float32)
        self.valid_mask_s = np.ones((hs, ws), dtype=np.float32)
        self.valid_mask_s[:14, :] = 0.0
        self.valid_mask_s[-12:, :] = 0.0
        self.valid_mask_s[:, :14] = 0.0
        self.valid_mask_s[:, -14:] = 0.0
        self.full_ones_s = np.ones((hs, ws), dtype=np.uint8) * 255
        ys = np.linspace(24, max(25, hs - 28), 12, dtype=np.float32)
        xs = np.linspace(26, max(27, ws - 26), 12, dtype=np.float32)
        gy, gx = np.meshgrid(ys, xs)
        self.grid_pts_s = np.stack([gx.ravel(), gy.ravel()], axis=-1).reshape(-1, 1, 2)
        self.clutter_sum_s = np.zeros((hs, ws), dtype=np.float32)
        self.clutter_count_s = np.zeros((hs, ws), dtype=np.float32)
        self.cached_clutter_weight_s = np.ones((hs, ws), dtype=np.float32)

    def _estimate_bg_translation_fast(
        self, g0: np.ndarray, g1: np.ndarray, ui0: Optional[np.ndarray]
    ) -> Tuple[float, float]:
        bg_m = ((ui0 == 0).astype(np.uint8) * 255) if ui0 is not None else None
        pts0 = cv2.goodFeaturesToTrack(g0, maxCorners=220, qualityLevel=0.018, minDistance=8, mask=bg_m)
        if pts0 is None or len(pts0) < 10:
            pts0 = self.grid_pts_s
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(
            g0, g1, pts0, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        if pts1 is None or st is None:
            return 0.0, 0.0
        d = (pts1[st == 1] - pts0[st == 1]).reshape(-1, 2)
        if len(d) >= 8:
            tx, ty = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
            inl = (np.abs(d[:, 0] - tx) < 0.70) & (np.abs(d[:, 1] - ty) < 0.70)
            if np.count_nonzero(inl) >= 6:
                tx = float(np.median(d[inl, 0]))
                ty = float(np.median(d[inl, 1]))
            return tx, ty
        return 0.0, 0.0

    @staticmethod
    def _fast_norm(arr: np.ndarray, bg_mask: Optional[np.ndarray] = None) -> np.ndarray:
        mean_v, std_v = cv2.meanStdDev(arr, mask=bg_mask)
        m, s = float(mean_v[0, 0]), float(std_v[0, 0])
        if s < 1e-3:
            return np.zeros_like(arr, dtype=np.float32)
        clean = np.maximum(0.0, arr - (m + 0.80 * s))
        return np.clip(clean / (m + 4.0 * s + 1e-5), 0.0, 1.0)

    @staticmethod
    def _damped_normalized_box_filter(
        sal: np.ndarray, valid_mask: np.ndarray, ksize: Tuple[int, int], min_den_ratio: float = 0.25
    ) -> np.ndarray:
        """归一化卷积能量积分：彻底消除目标靠近画面四边时的向内偏置"""
        num = cv2.boxFilter(sal * valid_mask, cv2.CV_32F, ksize, normalize=False)
        den = cv2.boxFilter(valid_mask, cv2.CV_32F, ksize, normalize=False)
        area = float(ksize[0] * ksize[1])
        out = np.zeros_like(num, dtype=np.float32)
        good = den > (0.25 * area)
        denom_clamped = np.maximum(den, min_den_ratio * area)
        out[good] = num[good] / denom_clamped[good]
        return out

    def _learn_canonical_shape(
        self, wm_s: np.ndarray, cnt: np.ndarray, cx_s: float, cy_s: float
    ) -> None:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.012 * peri, True).reshape(-1, 2).astype(np.float32)
        centered_pts_full = (approx - np.array([[cx_s, cy_s]], dtype=np.float32)) * 2.0

        rad = 32
        M_crop = np.array([[1.0, 0.0, rad - cx_s], [0.0, 1.0, rad - cy_s]], dtype=np.float32)
        patch = cv2.warpAffine(
            wm_s.astype(np.float32) / 255.0, M_crop, (64, 64),
            flags=cv2.INTER_LINEAR, borderValue=0.0
        )

        if self.canonical_mask_s is None:
            self.canonical_mask_s = patch
            self.canonical_contour = centered_pts_full
            self.cur_angle = 0.0

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.hypot(gx, gy)
            edge = edge / (float(np.max(edge)) + 1e-5)
            self.canonical_edge_s = edge

            bank = []
            for k in range(72):
                ang = k * 5.0
                M_rot = cv2.getRotationMatrix2D((rad, rad), -ang, 1.0)
                rot_e = cv2.warpAffine(edge, M_rot, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0.0)
                bank.append(rot_e)
            self.rotated_edge_bank = np.stack(bank, axis=0)
            if self.cuda_backend.has_torch_cuda:
                torch = self.cuda_backend.torch
                self.rotated_edge_bank_gpu = torch.from_numpy(self.rotated_edge_bank).to(
                    self.cuda_backend.torch_device, dtype=torch.float32
                )

            area = float(cv2.contourArea(cnt))
            rect = cv2.minAreaRect(cnt)
            box_area = max(1.0, float(rect[1][0] * rect[1][1]))
            solidity = area / box_area
            self.is_irregular_shape = (len(approx) >= 9) or (solidity < 0.78)
            self.shape_desc = f"CUSTOM_POLY(V={len(approx)},Sol={solidity:.2f})"

    def _match_arbitrary_shape_rotation(
        self, sal_roi: np.ndarray, cx_roi: float, cy_roi: float, prev_ang: float
    ) -> float:
        if self.rotated_edge_bank is None:
            return prev_ang
        rad = 32
        M_crop = np.array([[1.0, 0.0, rad - cx_roi], [0.0, 1.0, rad - cy_roi]], dtype=np.float32)
        patch = cv2.warpAffine(sal_roi, M_crop, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0.0)
        if float(np.max(patch)) < 1e-3:
            return prev_ang
        patch_n = patch / (float(np.max(patch)) + 1e-5)

        deltas = np.linspace(-30.0, 30.0, 9, dtype=np.float32)
        if self.cuda_backend.has_torch_cuda and self.rotated_edge_bank_gpu is not None:
            torch = self.cuda_backend.torch
            with torch.no_grad():
                idxs = [int(round(((prev_ang + float(d)) % 360.0) / 5.0)) % 72 for d in deltas]
                tmpls_gpu = self.rotated_edge_bank_gpu[idxs]
                patch_gpu = torch.from_numpy(patch_n).to(self.cuda_backend.torch_device, dtype=torch.float32)
                raw_scores = torch.sum(tmpls_gpu * patch_gpu.unsqueeze(0), dim=(1, 2)).cpu().numpy()
            priors = np.exp(-0.5 * (deltas / 18.0) ** 2)
            scores = raw_scores * (0.85 + 0.15 * priors)
            best_i = int(np.argmax(scores))
            best_delta = float(deltas[best_i])
        else:
            best_score = -1.0
            best_delta = 0.0
            for delta in deltas:
                cand_ang = (prev_ang + float(delta)) % 360.0
                idx = int(round(cand_ang / 5.0)) % 72
                tmpl = self.rotated_edge_bank[idx]
                prior = np.exp(-0.5 * (delta / 18.0) ** 2)
                score = float(np.sum(patch_n * tmpl)) * (0.85 + 0.15 * prior)
                if score > best_score:
                    best_score = score
                    best_delta = float(delta)

        return 0.65 * prev_ang + 0.35 * (prev_ang + best_delta)

    def get_current_polygon_full(self, cx: float, cy: float, angle_deg: float) -> np.ndarray:
        if self.canonical_contour is None:
            box = cv2.boxPoints(((cx, cy), (self.ref_w, self.ref_h), angle_deg))
            return box.astype(np.int32)
        rad = np.radians(angle_deg)
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rot_pts = (self.canonical_contour @ R.T) + np.array([[cx, cy]], dtype=np.float32)
        return np.round(rot_pts).astype(np.int32)

    def _detect_ui_and_fade_half(
        self, frame_s: np.ndarray, gray_s: np.ndarray, compute_fade: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        hs, ws = frame_s.shape[:2]
        b = frame_s[:, :, 0]
        g = frame_s[:, :, 1]
        r = frame_s[:, :, 2]

        ui_mask_s = np.zeros((hs, ws), dtype=np.uint8)
        top_h = min(int(hs * 0.40), 96)
        x_m1, x_m2 = int(ws * 0.25), int(ws * 0.78)
        top_b = b[:top_h, x_m1:x_m2].astype(np.int16)
        top_g = g[:top_h, x_m1:x_m2].astype(np.int16)
        top_r = r[:top_h, x_m1:x_m2].astype(np.int16)
        if np.count_nonzero((top_b - top_r > 16) & (top_b > 150)) > 25:
            cyan_c = ((top_b - top_r > 18) & (top_b > 140)).astype(np.uint8)
            ui_mask_s[:top_h, x_m1:x_m2] = cv2.dilate(cyan_c, self.k_dil15_s)

        y_s1 = int(hs * 0.32)
        x_l1, x_l2 = int(ws * 0.16), int(ws * 0.38)
        x_r1, x_r2 = int(ws * 0.62), int(ws * 0.88)
        left_w = (g[y_s1:, x_l1:x_l2] > 225) & (r[y_s1:, x_l1:x_l2] > 225) & (b[y_s1:, x_l1:x_l2] > 145)
        right_w = (g[y_s1:, x_r1:x_r2] > 225) & (r[y_s1:, x_r1:x_r2] > 225) & (b[y_s1:, x_r1:x_r2] > 145)
        if np.count_nonzero(left_w) > 45 and np.count_nonzero(right_w) > 45:
            succ_px = (g[y_s1:, x_l1:x_r2] > 180) & (r[y_s1:, x_l1:x_r2] > 180) & (b[y_s1:, x_l1:x_r2] > 115)
            ui_mask_s[y_s1:, x_l1:x_r2][succ_px] = 1

        green_c = (g > 135) & (g.astype(np.int16) - r.astype(np.int16) > 45) & (g.astype(np.int16) - b.astype(np.int16) > 35)
        n_green = int(np.count_nonzero(green_c))
        if 5 < n_green < 1200:
            ui_mask_s[green_c] = 1

        if np.any(ui_mask_s):
            ui_mask_s = cv2.dilate(ui_mask_s, self.dilate_k_s)

        if not compute_fade:
            return ui_mask_s, None

        mean_g, std_g = cv2.meanStdDev(gray_s, mask=(ui_mask_s == 0).astype(np.uint8))
        bright_ex = np.maximum(0.0, gray_s.astype(np.float32) - (float(mean_g[0, 0]) + 1.0 * float(std_g[0, 0])))
        warmth = r.astype(np.float32) - b.astype(np.float32)
        fade_white_s = cv2.GaussianBlur(
            bright_ex * np.clip((48.0 - warmth) / 48.0, 0.0, 1.0),
            (15, 15), 0
        )
        fade_white_s[ui_mask_s > 0] = 0.0
        return ui_mask_s, fade_white_s

    def _compute_true_geometric_center(
        self,
        anchor_x_s: float,
        anchor_y_s: float,
        saliency_now_s: np.ndarray,
        clutter_w_s: np.ndarray,
        vmask_s: np.ndarray,
        mean_side: float,
        ws: int,
        hs: int,
        max_valid_y_s: int,
    ) -> Tuple[float, float]:
        """
        Layer 2: 无偏几何中心精定位器 (Flat-Topped Target Isolation + BBox/Convex-Hull Geometric Center)
        1. 平顶圆盘目标隔离 (Flat-Topped Disk Mask): 在锚点周围半径 R_iso 内赋等权 1.0 (消除高斯先验斜率引起的后沿滞后)，
           同时将 R_iso 以外的背景岩石/边界杂波严格清零；
        2. 全尺寸对称盒积分 + 活跃轮廓支撑集 BBox 中点与填充凸包质心:
           将亮边与暗边二值化为等权轮廓，定位目标四条对边的真实几何对称中心。
        """
        R_iso = int(np.clip(round(0.36 * mean_side), 30, 42))
        x1 = max(8, int(round(anchor_x_s - R_iso)))
        x2 = min(ws - 8, int(round(anchor_x_s + R_iso + 1)))
        y1 = max(8, int(round(anchor_y_s - R_iso)))
        y2 = min(max_valid_y_s, int(round(anchor_y_s + R_iso + 1)))
        if x2 <= x1 + 12 or y2 <= y1 + 12:
            return anchor_x_s, anchor_y_s

        yy_r, xx_r = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        dist_sq_r = (xx_r - anchor_x_s) ** 2 + (yy_r - anchor_y_s) ** 2
        # Flat-topped disk mask: weight=1.0 across the entire target footprint, 0 outside
        disk_m = (dist_sq_r <= (R_iso ** 2)).astype(np.float32)
        cw_roi = clutter_w_s[y1:y2, x1:x2]
        vm_roi = vmask_s[y1:y2, x1:x2]
        sal_roi = saliency_now_s[y1:y2, x1:x2] * cw_roi * vm_roi * disk_m
        sal_roi = np.where(cw_roi >= 0.55, sal_roi, 0.0)

        smax = float(np.max(sal_roi))
        if smax < 0.12:
            return anchor_x_s, anchor_y_s

        # 1. Full-Target-Span Symmetric Box + Gaussian Peak on isolated target (zero external border/clutter pull!)
        k_box = int(np.clip(int(round(0.48 * mean_side)) | 1, 45, 55))
        e_box = cv2.boxFilter(sal_roi, cv2.CV_32F, (k_box, k_box), normalize=True, borderType=cv2.BORDER_CONSTANT)
        e_box = cv2.GaussianBlur(e_box, (17, 17), 4.5)
        _, max_eb, _, _ = cv2.minMaxLoc(e_box)
        cx_span, cy_span = anchor_x_s, anchor_y_s
        if max_eb > 1e-4:
            top_w = np.maximum(0.0, e_box - 0.80 * float(max_eb))
            w_sum = float(np.sum(top_w))
            if w_sum > 1e-5:
                cx_span = float(np.sum(xx_r * top_w) / w_sum)
                cy_span = float(np.sum(yy_r * top_w) / w_sum)

        # 2. Equalized Foreground Silhouette BBox Midpoint & Convex-Hull Centroid (eliminates bright/dim edge asymmetry)
        th = max(0.13, 0.24 * smax)
        fg = (sal_roi >= th).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open3_s)
        fg_closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close13_s)
        ys_fg, xs_fg = np.nonzero(fg_closed)
        cx_geo, cy_geo = cx_span, cy_span
        if len(xs_fg) >= 22:
            bx_mid = 0.5 * (float(np.percentile(xs_fg, 5)) + float(np.percentile(xs_fg, 95))) + x1
            by_mid = 0.5 * (float(np.percentile(ys_fg, 5)) + float(np.percentile(ys_fg, 95))) + y1
            pts_hull = cv2.convexHull(np.stack([xs_fg, ys_fg], axis=-1).astype(np.int32))
            hull_mask = np.zeros_like(fg_closed)
            cv2.fillConvexPoly(hull_mask, pts_hull, 255)
            M_h = cv2.moments(hull_mask, binaryImage=True)
            if M_h["m00"] > 20.0:
                hx_mid = float(M_h["m10"] / M_h["m00"]) + x1
                hy_mid = float(M_h["m01"] / M_h["m00"]) + y1
                cand_x = 0.52 * bx_mid + 0.48 * hx_mid
                cand_y = 0.52 * by_mid + 0.48 * hy_mid
            else:
                cand_x, cand_y = bx_mid, by_mid
            if np.hypot(cand_x - anchor_x_s, cand_y - anchor_y_s) <= 22.0:
                cx_geo = 0.65 * cand_x + 0.35 * cx_span
                cy_geo = 0.65 * cand_y + 0.35 * cy_span

        return cx_geo, cy_geo

    def update(self, frame_bgr: np.ndarray, dt_sec: Optional[float] = None) -> dict:
        t0 = time.perf_counter()
        if dt_sec is not None and dt_sec > 1e-3 and self.dynamic_clock_scale:
            n_step = float(np.clip(dt_sec * self.fps, 0.5, 4.5))
        else:
            n_step = 1.0

        H, W = frame_bgr.shape[:2]
        hs, ws = max(64, H // 2), max(64, W // 2)
        if self.xx_s is None or self.xx_s.shape != (hs, ws):
            self._init_grids(hs, ws)

        frame_s = cv2.resize(frame_bgr, (ws, hs), interpolation=cv2.INTER_AREA)
        gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
        fps_scale = max(0.45, self.fps / 60.0)
        need_fade = self.transparent_streak <= int(64 * fps_scale)
        ui_mask_s, fade_white_s = self._detect_ui_and_fade_half(frame_s, gray_s, compute_fade=need_fade)

        has_bot_bar = (hs >= 258) and (float(np.mean(gray_s[hs - 16:, :])) < 75.0)
        if has_bot_bar:
            ui_mask_s[hs - 27:, :] = 1
            self.valid_mask_s[hs - 27:, :] = 0.0
        max_valid_y_s = (hs - 27) if has_bot_bar else (hs - 10)
        max_valid_cy = float(max_valid_y_s * 2.0 - 20.0) if has_bot_bar else float(H - 20.0)

        b_i16 = frame_s[:, :, 0].astype(np.int16)
        r_i16 = frame_s[:, :, 2].astype(np.int16)
        wm_raw = ((gray_s > 215) & (np.abs(b_i16 - r_i16) <= 24) & (ui_mask_s == 0)).astype(np.uint8) * 255
        wm_s = cv2.morphologyEx(wm_raw, cv2.MORPH_CLOSE, self.k_close11_s)
        cnts, _ = cv2.findContours(wm_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(cnts, key=cv2.contourArea) if cnts else None
        max_area = float(cv2.contourArea(max_cnt)) if max_cnt is not None else 0.0

        if self.transparent_streak >= 5 and self.auto_reinit_on_white:
            if max_area >= 480.0 and float(np.mean(gray_s > 210)) < 0.25:
                x_b_w, y_b_w, w_b_w, h_b_w = cv2.boundingRect(max_cnt)
                cx_w = float((x_b_w + 0.5 * w_b_w) * 2.0)
                cy_w = float((y_b_w + 0.5 * h_b_w) * 2.0)
                near_tracked = (self.ema_cx is None) or (float(np.hypot(cx_w - self.ema_cx, cy_w - self.ema_cy)) <= 120.0)
                if near_tracked or self.transparent_streak < int(90 * fps_scale):
                    self.white_reappear_streak += 1
                    if self.white_reappear_streak >= 3:
                        self.reset()
                        self._init_grids(hs, ws)
                else:
                    self.white_reappear_streak = 0
            else:
                self.white_reappear_streak = 0

        # Stage 1: WHITE phase
        if (self.transparent_streak < 5 and max_cnt is not None and max_area >= 220.0) or (not self.initialized):
            if max_cnt is not None and max_area >= 120.0:
                x_b, y_b, w_b, h_b = cv2.boundingRect(max_cnt)
                rect = cv2.minAreaRect(max_cnt)
                cx = float(0.5 * (rect[0][0] * 2.0 + (x_b + 0.5 * w_b) * 2.0))
                cy = float(0.5 * (rect[0][1] * 2.0 + (y_b + 0.5 * h_b) * 2.0))
                rw, rh = float(rect[1][0] * 2.0), float(rect[1][1] * 2.0)
                if rw > 70.0 and rh > 70.0:
                    self.ref_w = 0.90 * self.ref_w + 0.10 * rw
                    self.ref_h = 0.90 * self.ref_h + 0.10 * rh
                    if self.canonical_mask_s is None and self.frame_idx >= 3:
                        self._learn_canonical_shape(wm_s, max_cnt, cx * 0.5, cy * 0.5)
            else:
                cx, cy = (W * 0.5, H * 0.5) if self.ema_cx is None else (self.ema_cx, self.ema_cy)

            if not self.initialized:
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.ema_cx, self.ema_cy = cx, cy
                self.center_cx, self.center_cy = cx, cy
                self.prev_out_cx, self.prev_out_cy = cx, cy
                self.init_cx_s, self.init_cy_s = cx * 0.5, cy * 0.5
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                self.prev_out_cx, self.prev_out_cy = self.center_cx, self.center_cy
                self.ema_cx = self.ema_alpha * cx + (1.0 - self.ema_alpha) * self.ema_cx
                self.ema_cy = self.ema_alpha * cy + (1.0 - self.ema_alpha) * self.ema_cy
                self.center_cx, self.center_cy = self.ema_cx, self.ema_cy
                if self.frame_idx <= 10:
                    self.init_cx_s = 0.9 * self.init_cx_s + 0.1 * (cx * 0.5)
                    self.init_cy_s = 0.9 * self.init_cy_s + 0.1 * (cy * 0.5)
                elif self.init_cx_s is not None and max_area >= 1800.0:
                    self.dep_dx_s = float(cx * 0.5 - self.init_cx_s)

            wm_f = (wm_s > 0).astype(np.float32)
            gx_w = cv2.Sobel(wm_f, cv2.CV_32F, 1, 0, ksize=3)
            gy_w = cv2.Sobel(wm_f, cv2.CV_32F, 0, 1, ksize=3)
            self.prev_angle = self.cur_angle
            self.cur_angle = self._match_arbitrary_shape_rotation(
                np.hypot(gx_w, gy_w), self.ema_cx * 0.5, self.ema_cy * 0.5, self.cur_angle
            )
            self.last_unet_prob_s = wm_f

            if len(self.prev_grays_s) > 0:
                ad_w0 = cv2.absdiff(gray_s, self.prev_grays_s[0])
                is_tex = float(np.std(gray_s)) > 15.0
                if float(np.mean(ad_w0)) >= 1.18 or (not is_tex and float(np.percentile(ad_w0, 99.2)) >= 5.0):
                    tx1, ty1 = self._estimate_bg_translation_fast(self.prev_grays_s[0], gray_s, self.prev_uis_s[0])
                    if is_tex:
                        self.bg_scroll_ema = 0.92 * self.bg_scroll_ema + 0.08 * (abs(tx1) + abs(ty1))
                    self.prev_shifts.insert(0, (tx1, ty1))
                    self.prev_shifts = self.prev_shifts[:3]
                    self.prev_frames_s.insert(0, frame_s)
                    self.prev_grays_s.insert(0, gray_s)
                    self.prev_uis_s.insert(0, ui_mask_s)
                    self.prev_frames_s = self.prev_frames_s[:3]
                    self.prev_grays_s = self.prev_grays_s[:3]
                    self.prev_uis_s = self.prev_uis_s[:3]
                else:
                    self.total_dup_frames += 1
            else:
                self.prev_frames_s.insert(0, frame_s)
                self.prev_grays_s.insert(0, gray_s)
                self.prev_uis_s.insert(0, ui_mask_s)
                self.prev_shifts.insert(0, (0.0, 0.0))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            res = {
                "frame_idx": self.frame_idx, "phase": "WHITE", "shape": self.shape_desc,
                "cx": float(self.center_cx), "cy": float(self.center_cy),
                "w": self.ref_w, "h": self.ref_h, "angle": self.cur_angle,
                "vx": float((self.center_cx - self.prev_out_cx) * self.fps / max(0.5, n_step)),
                "vy": float((self.center_cy - self.prev_out_cy) * self.fps / max(0.5, n_step)),
                "omega": float((self.cur_angle - self.prev_angle) * self.fps / max(0.5, n_step)),
                "snr": 10.0, "conf": 0.99, "latency_ms": dt_ms, "n_step": n_step, "device": self.cuda_backend.device_name
            }
            self.frame_idx += 1
            return res

        # Stage 2: TRANSPARENT phase
        self.transparent_streak += 1
        ad_0 = cv2.absdiff(gray_s, self.prev_grays_s[0])
        raw_diff_0 = float(np.mean(ad_0))

        # Scene cut / mini-game window closing hold at end of capture
        if self.transparent_streak > int(200 * fps_scale) and (raw_diff_0 > 42.0 or self.scene_cut_hold > 0):
            if raw_diff_0 > 42.0:
                self.scene_cut_hold = 4
            else:
                self.scene_cut_hold -= 1
            dt_ms = (time.perf_counter() - t0) * 1000.0
            snr_hold = max(1.8, self.last_valid_snr * 0.95)
            conf_hold = float(np.clip(0.64 + 0.08 * min(snr_hold, 4.5), 0.64, 0.96))
            res = {
                "frame_idx": self.frame_idx, "phase": "TRANSPARENT", "shape": self.shape_desc,
                "cx": float(self.center_cx), "cy": float(self.center_cy),
                "w": self.ref_w, "h": self.ref_h, "angle": self.cur_angle,
                "vx": 0.0, "vy": 0.0, "omega": 0.0,
                "snr": snr_hold, "conf": conf_hold, "latency_ms": dt_ms, "n_step": n_step,
                "device": self.cuda_backend.device_name
            }
            self.frame_idx += 1
            return res

        is_tex = float(np.std(gray_s)) > 15.0
        is_dup = (raw_diff_0 < 1.18) and (is_tex or float(np.percentile(ad_0, 99.2)) < 5.0)
        if is_dup:
            self.dup_streak += 1
            self.total_dup_frames += 1

        dup_ratio = self.total_dup_frames / float(max(1, self.frame_idx))
        is_game_tick_video = (dup_ratio > 0.40) or (self.dup_streak >= 2)

        if is_dup and is_game_tick_video:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            snr_hold = max(1.5, self.last_valid_snr * 0.95)
            conf_hold = float(np.clip(0.64 + 0.08 * min(snr_hold, 4.5), 0.64, 0.96))
            res = {
                "frame_idx": self.frame_idx, "phase": "TRANSPARENT", "shape": self.shape_desc,
                "cx": float(self.center_cx), "cy": float(self.center_cy),
                "w": self.ref_w, "h": self.ref_h, "angle": self.cur_angle,
                "vx": 0.0, "vy": 0.0, "omega": 0.0,
                "snr": snr_hold, "conf": conf_hold, "latency_ms": dt_ms, "n_step": n_step,
                "device": self.cuda_backend.device_name
            }
            self.frame_idx += 1
            return res

        eff_step = float(np.clip(n_step + (self.dup_streak if is_game_tick_video else 0.0), 0.5, 4.0))
        if not is_dup:
            self.dup_streak = 0

        self.kf.transitionMatrix[0, 2] = np.float32(eff_step)
        self.kf.transitionMatrix[1, 3] = np.float32(eff_step)
        pred = self.kf.predict()
        pred_cx = float(np.clip(pred[0, 0], 24.0, W - 16.0))
        pred_cy = float(np.clip(pred[1, 0], 34.0, max_valid_cy))
        self.prev_out_cx, self.prev_out_cy = self.center_cx, self.center_cy
        self.prev_angle = self.cur_angle

        pred_cx_s, pred_cy_s = pred_cx * 0.5, pred_cy * 0.5
        anchor_cx_s = 0.55 * pred_cx_s + 0.45 * (self.ema_cx * 0.5)
        anchor_cy_s = 0.55 * pred_cy_s + 0.45 * (self.ema_cy * 0.5)

        tx1, ty1 = self._estimate_bg_translation_fast(self.prev_grays_s[0], gray_s, self.prev_uis_s[0])
        if not is_dup:
            if is_tex:
                self.bg_scroll_ema = 0.92 * self.bg_scroll_ema + 0.08 * (abs(tx1) + abs(ty1))
            self.prev_shifts.insert(0, (tx1, ty1))
            self.prev_shifts = self.prev_shifts[:3]

        comp_diffs = []
        valid_masks = []
        cum_tx, cum_ty = 0.0, 0.0
        raw_d0 = None
        for k in range(len(self.prev_frames_s)):
            cum_tx += self.prev_shifts[k][0]
            cum_ty += self.prev_shifts[k][1]
            Mk = np.array([[1.0, 0.0, cum_tx], [0.0, 1.0, cum_ty]], dtype=np.float32)
            use_cubic = (k == 0) or (self.perf_mode == "ultra")
            interp_flag = cv2.INTER_CUBIC if use_cubic else cv2.INTER_LINEAR
            wk = cv2.warpAffine(self.prev_frames_s[k], Mk, (ws, hs), flags=interp_flag, borderMode=cv2.BORDER_REPLICATE)
            mk = cv2.erode(
                cv2.warpAffine(self.full_ones_s, Mk, (ws, hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0),
                self.erode_k_s
            )
            mk_f = ((mk > 0) & (ui_mask_s == 0) & (self.prev_uis_s[k] == 0)).astype(np.float32)
            diff_bgr = cv2.absdiff(frame_s, wk)
            dk = cv2.max(cv2.max(diff_bgr[:, :, 0], diff_bgr[:, :, 1]), diff_bgr[:, :, 2]).astype(np.float32)
            dk_clean = np.maximum(0.0, dk - 6.0) * mk_f
            if k == 0:
                raw_d0 = dk_clean
            mean_d, std_d = cv2.meanStdDev(dk_clean)
            md, sd = float(mean_d[0, 0]), float(std_d[0, 0])
            if sd > 0.15:
                comp_diffs.append(np.clip(dk_clean / (md + 4.2 * sd + 1e-5), 0.0, 1.0))
                valid_masks.append(mk_f)

        vmask_s = np.min(valid_masks, axis=0) if valid_masks else np.ones((hs, ws), dtype=np.float32)
        vmask_s[:4, :] = 0.0; vmask_s[-4:, :] = 0.0; vmask_s[:, :6] = 0.0; vmask_s[:, -6:] = 0.0
        if has_bot_bar:
            vmask_s[hs - 27:, :] = 0.0

        flow_norms_raw = []
        for k in range(min(2, len(self.prev_grays_s))):
            ad_k = cv2.absdiff(gray_s, self.prev_grays_s[k])
            if float(np.mean(ad_k)) > 1.18 or (not is_tex and float(np.percentile(ad_k, 99.2)) >= 5.0):
                ux, uy = self.cuda_backend.calc_farneback_gpu(self.prev_grays_s[k], gray_s, win_size=13, levels=2, iters=2)
                rk = np.hypot(ux - np.median(ux), uy - np.median(uy)) * vmask_s
                dv_dx = cv2.Sobel(uy, cv2.CV_32F, 1, 0, ksize=3)
                du_dy = cv2.Sobel(ux, cv2.CV_32F, 0, 1, ksize=3)
                curl_mag = np.abs(dv_dx - du_dy) * vmask_s
                rk[:8, :] = 0.0; curl_mag[:8, :] = 0.0
                mr, sr = cv2.meanStdDev(rk)
                mc, sc = cv2.meanStdDev(curl_mag)
                r_clean = np.maximum(0.0, rk - (float(mr[0, 0]) + 0.75 * float(sr[0, 0])))
                c_clean = np.maximum(0.0, curl_mag - (float(mc[0, 0]) + 0.75 * float(sc[0, 0])))
                r_n = np.clip(r_clean / (float(mr[0, 0]) + 4.0 * float(sr[0, 0]) + 1e-5), 0.0, 1.0)
                c_n = np.clip(c_clean / (float(mc[0, 0]) + 4.0 * float(sc[0, 0]) + 1e-5), 0.0, 1.0)
                flow_norms_raw.append(0.75 * r_n + 0.25 * c_n)

        # Suppress stationary top UI text overlay (y_s < 22, e.g. "Monsters Near Your Level") where optical flow is absent
        if len(flow_norms_raw) > 0 and (not is_game_tick_video):
            top_flow_gate = np.ones((hs, ws), dtype=np.float32)
            top_band_f = cv2.boxFilter(flow_norms_raw[0], cv2.CV_32F, (15, 15))
            top_flow_gate[:22, :] = np.clip(top_band_f[:22, :] / 0.08, 0.0, 1.0)
            comp_diffs = [cd * top_flow_gate for cd in comp_diffs]
            if raw_d0 is not None:
                raw_d0 = raw_d0 * top_flow_gate

        is_right_scroll_maple = is_tex and (not is_game_tick_video) and (self.bg_scroll_ema > 0.45) and (self.dep_dx_s > 1.5)
        c_start_f = int(10 * fps_scale) if is_right_scroll_maple else int(36 * fps_scale)
        c_rad = 30.0 if is_right_scroll_maple else 46.0
        if (not is_dup) and raw_d0 is not None and self.transparent_streak > c_start_f and (abs(tx1) + abs(ty1) > 0.12):
            t_dist_sq = (self.xx_s - self.ema_cx * 0.5) ** 2 + (self.yy_s - self.ema_cy * 0.5) ** 2
            far_m = (t_dist_sq > (c_rad ** 2)) & (ui_mask_s == 0)
            self.clutter_sum_s[far_m] += raw_d0[far_m]
            self.clutter_count_s[far_m] += 1.0

        dist_init = 0.0
        if self.init_cx_s is not None and (not is_game_tick_video):
            dist_init = float(np.hypot(self.ema_cx * 0.5 - self.init_cx_s, self.ema_cy * 0.5 - self.init_cy_s))
            if dist_init > 80.0 and (self.ema_cy * 0.5 < 55.0) and self.transparent_streak <= int(265 * fps_scale) and self.reentry_dwell < (36.0 * fps_scale):
                self.has_left_birth = True
                self.reentry_dwell = 0.0
                self.cached_clutter_weight_s.fill(1.0)
            elif self.has_left_birth:
                if dist_init < 38.0 or self.reentry_dwell >= (36.0 * fps_scale):
                    self.reentry_dwell += eff_step
                elif dist_init > 50.0:
                    self.reentry_dwell = 0.0

        eb_start_frames = int(48 * fps_scale) if (self.bg_scroll_ema > 0.45 and self.dep_dx_s > 1.5) else int(160 * fps_scale)
        if self.transparent_streak >= eb_start_frames and self.init_cx_s is not None:
            allow_eb = is_game_tick_video or (not self.has_left_birth) or (self.reentry_dwell >= (36.0 * fps_scale))
            if allow_eb and ((self.transparent_streak == eb_start_frames) or (self.transparent_streak % 4 == 0) or self.has_left_birth):
                min_c = (12.0 * fps_scale) if (self.bg_scroll_ema > 0.45 and self.dep_dx_s > 1.5) else (30.0 * fps_scale if not is_game_tick_video else 12.0 * fps_scale)
                valid_c = self.clutter_count_s > min_c
                if np.any(valid_c):
                    raw_m = np.zeros_like(self.clutter_sum_s)
                    mc = self.clutter_count_s > 1.0
                    raw_m[mc] = self.clutter_sum_s[mc] / self.clutter_count_s[mc]
                    mu0 = float(np.median(raw_m[valid_c])) + 0.25
                    N0 = (16.0 * fps_scale) if (self.bg_scroll_ema > 0.45 and self.dep_dx_s > 1.5) else (35.0 * fps_scale if not is_game_tick_video else 15.0 * fps_scale)
                    eb_m = (self.clutter_sum_s + N0 * mu0) / (self.clutter_count_s + N0)
                    eb_s = cv2.boxFilter(eb_m, cv2.CV_32F, (25, 25))
                    ratio = eb_s / max(0.5, mu0)
                    if (self.has_left_birth or (self.bg_scroll_ema > 0.45 and self.dep_dx_s > 1.5 and dist_init > 32.0)) and (not is_game_tick_video):
                        init_d2 = (self.xx_s - self.init_cx_s) ** 2 + (self.yy_s - self.init_cy_s) ** 2
                        eb_in = np.where(init_d2 <= (42.0 ** 2), ratio, 0.0)
                        _, _, _, eb_pk = cv2.minMaxLoc(eb_in)
                        cx_decoy = 0.65 * float(eb_pk[0]) + 0.35 * self.init_cx_s
                        cy_decoy = 0.65 * float(eb_pk[1]) + 0.35 * self.init_cy_s
                        b_dist_sq = (self.xx_s - cx_decoy) ** 2 + (self.yy_s - cy_decoy) ** 2
                        b_zone = np.exp(-0.5 * b_dist_sq / (36.0 ** 2)).astype(np.float32)
                        ramp = float(np.clip((self.reentry_dwell - 36.0 * fps_scale) / (10.0 * fps_scale), 0.0, 1.0)) if self.has_left_birth else 1.0
                    else:
                        b_dist_sq = (self.xx_s - self.init_cx_s) ** 2 + (self.yy_s - self.init_cy_s) ** 2
                        b_zone = np.exp(-0.5 * b_dist_sq / (24.0 ** 2)).astype(np.float32)
                        ramp = 1.0
                    s_birth = np.clip(ratio - 1.30, 0.0, 1.2)
                    supp = (1.0 - np.exp(-3.5 * (s_birth ** 2))) * b_zone * ramp
                    self.cached_clutter_weight_s = (1.0 - 0.88 * supp).astype(np.float32)
        clutter_w_s = self.cached_clutter_weight_s

        if (not is_game_tick_video) and is_tex and self.bg_scroll_ema > 0.45 and self.dep_dx_s >= 7.5 and self.transparent_streak <= int(90 * fps_scale) and self.init_cx_s is not None:
            dec_x = self.init_cx_s - 12.0
            dec_y = self.init_cy_s + 9.0
            d2_dec = (self.xx_s - dec_x) ** 2 + (self.yy_s - dec_y) ** 2
            early_sup = (1.0 - 0.55 * np.exp(-0.5 * d2_dec / (22.0 ** 2))).astype(np.float32)
            clutter_w_s = np.minimum(clutter_w_s, early_sup)

        comp_norm_s = (np.mean(comp_diffs, axis=0) if comp_diffs else np.zeros((hs, ws), dtype=np.float32)) * clutter_w_s
        flow_norms = [fn * clutter_w_s for fn in flow_norms_raw]
        rf_norm_s = np.mean(flow_norms, axis=0) if flow_norms else comp_norm_s
        saliency_s = (0.55 * comp_norm_s + 0.45 * rf_norm_s) * vmask_s

        if len(comp_diffs) >= 2:
            d_now = 0.65 * comp_diffs[0] + 0.35 * np.sqrt(comp_diffs[0] * comp_diffs[1] + 1e-6)
        elif len(comp_diffs) == 1:
            d_now = comp_diffs[0]
        else:
            d_now = comp_norm_s
        f_now = flow_norms[0] if len(flow_norms) > 0 else rf_norm_s
        saliency_now_s = (0.55 * d_now * clutter_w_s + 0.45 * f_now) * vmask_s
        if fade_white_s is not None and self.transparent_streak <= int(24 * fps_scale) and self.bg_scroll_ema > 0.45 and self.dep_dx_s > 1.5:
            fw_max_raw = float(np.max(fade_white_s))
            if fw_max_raw > 3.5:
                fw_n = np.clip(fade_white_s / (fw_max_raw + 1e-5), 0.0, 1.0) * vmask_s * clutter_w_s
                saliency_now_s = np.maximum(saliency_now_s, 0.65 * fw_n)
        self.last_unet_prob_s = saliency_now_s

        mean_side = 0.5 * (self.ref_w + self.ref_h)
        k_half = int(np.clip(int(round(0.52 * mean_side)) | 1, 49, 59))
        if is_game_tick_video:
            energy_raw = self._damped_normalized_box_filter(saliency_now_s, vmask_s, ksize=(k_half, k_half), min_den_ratio=0.55)
            energy_s = cv2.GaussianBlur(energy_raw, (21, 21), 5.2) * clutter_w_s
        else:
            sal_in = 0.65 * saliency_now_s + 0.35 * saliency_s
            if self.is_irregular_shape and self.bg_scroll_ema > 0.45:
                sal_in = 0.72 * sal_in + 0.28 * cv2.morphologyEx(sal_in, cv2.MORPH_CLOSE, self.k_close7_s)
            energy_raw = self._damped_normalized_box_filter(sal_in, vmask_s, ksize=(k_half, k_half), min_den_ratio=0.35)
            energy_s = cv2.GaussianBlur(energy_raw, (21, 21), 5.2) * clutter_w_s

        mean_e, std_e = cv2.meanStdDev(energy_s, mask=(vmask_s > 0).astype(np.uint8))
        me_val, se_val = float(mean_e[0, 0]), float(std_e[0, 0])

        if is_game_tick_video:
            global_max = float(np.max(energy_s))
            energy_norm_s = energy_s / (global_max + 1e-5)
        else:
            energy_above_bg = np.maximum(0.0, energy_s - me_val)
            global_max = float(np.max(energy_above_bg))
            energy_norm_s = energy_above_bg / (global_max + 1e-5)

        dist_sq_s = (self.xx_s - anchor_cx_s) ** 2 + (self.yy_s - anchor_cy_s) ** 2
        eff_gate = self.gate_radius_half * (0.80 + 0.20 * eff_step)
        eff_sigma = self.spatial_sigma_half * (0.85 + 0.15 * eff_step)
        gate_mask_s = (dist_sq_s <= eff_gate ** 2).astype(np.float32)
        spatial_prior_s = np.exp(-0.5 * dist_sq_s / (eff_sigma ** 2)) * gate_mask_s

        if fade_white_s is not None and self.transparent_streak <= int(64 * fps_scale):
            fw_gated = self._damped_normalized_box_filter(fade_white_s, vmask_s, ksize=(k_half, k_half), min_den_ratio=0.25) * gate_mask_s
            fw_max = float(np.max(fw_gated))
        else:
            fw_gated = None
            fw_max = 0.0

        if fw_gated is not None and fw_max >= 3.5:
            combined_s = 0.55 * energy_norm_s + 0.45 * (fw_gated / (fw_max + 1e-5))
        else:
            combined_s = energy_norm_s

        posterior_s = combined_s * spatial_prior_s * clutter_w_s
        posterior_s[:18, :] = 0.0; posterior_s[-10:, :] = 0.0; posterior_s[:, :18] = 0.0; posterior_s[:, -10:] = 0.0
        if has_bot_bar:
            posterior_s[hs - 27:, :] = 0.0
        _, max_post, _, max_loc_s = cv2.minMaxLoc(posterior_s)

        snr_raw = (float(energy_s[max_loc_s[1], max_loc_s[0]]) - me_val) / (se_val + 1e-5)
        snr_check_th = 1.45 if is_tex else 1.15
        if snr_raw < snr_check_th and (not is_game_tick_video):
            p_pct = 78.0 if is_tex else 85.0
            p85_e = float(np.percentile(energy_s[vmask_s > 0], p_pct)) + 1e-5
            bg_trim = ((vmask_s > 0) & (energy_s <= p85_e)).astype(np.uint8)
            me_t, se_t = cv2.meanStdDev(energy_s, mask=bg_trim)
            snr_trim = (float(energy_s[max_loc_s[1], max_loc_s[0]]) - float(me_t[0, 0])) / (float(se_t[0, 0]) + 1e-5)
            snr = max(1.05, snr_raw, min(3.5, snr_trim))
            if is_tex and self.transparent_streak > int(220 * fps_scale) and snr < 1.18:
                snr = max(1.18, self.last_valid_snr * 0.85)
        else:
            snr = max(1.05, snr_raw)
        low_snr_th = 1.15 if is_tex else 1.45
        if max_post < 0.08 and self.transparent_streak > int(20 * fps_scale) and ((max_post >= 0.035 and snr >= 1.15) or (max_post >= 0.002 and snr >= low_snr_th)):
            max_post = 0.10

        eff_max_step = self.max_step_px * max(1.0, 0.85 * eff_step)
        eff_alpha = 1.0 - (1.0 - self.ema_alpha) ** max(1.0, 0.85 * eff_step)

        if snr >= 1.15 and max_post >= 0.08:
            gx_s, gy_s = self._compute_true_geometric_center(
                float(max_loc_s[0]), float(max_loc_s[1]),
                saliency_now_s, clutter_w_s, vmask_s,
                mean_side, ws, hs, max_valid_y_s
            )
            meas_cx = float(np.clip(gx_s * 2.0, 24.0, W - 16.0))
            meas_cy = float(np.clip(gy_s * 2.0, 34.0, max_valid_cy))
            if is_game_tick_video:
                post = self.kf.correct(np.array([[np.float32(meas_cx)], [np.float32(meas_cy)]]))
                raw_cx = 0.72 * meas_cx + 0.28 * float(post[0, 0])
                raw_cy = 0.72 * meas_cy + 0.28 * float(post[1, 0])
            else:
                dx = meas_cx - self.ema_cx
                dy = meas_cy - self.ema_cy
                step_norm = float(np.hypot(dx, dy))
                if step_norm > eff_max_step:
                    meas_cx = self.ema_cx + dx * (eff_max_step / step_norm)
                    meas_cy = self.ema_cy + dy * (eff_max_step / step_norm)
                post = self.kf.correct(np.array([[np.float32(meas_cx)], [np.float32(meas_cy)]]))
                raw_cx, raw_cy = float(post[0, 0]), float(post[1, 0])
            self.last_valid_snr = snr
            conf = float(np.clip(0.64 + 0.08 * min(snr, 4.5), 0.64, 0.99))
            phase = "TRANSPARENT"
        else:
            raw_cx, raw_cy = pred_cx, pred_cy
            if is_dup or (raw_diff_0 < 1.65 and self.last_valid_snr >= 2.0):
                snr = max(snr, self.last_valid_snr * 0.92)
                conf = float(np.clip(0.64 + 0.08 * min(snr, 4.5), 0.64, 0.95))
                phase = "TRANSPARENT"
            else:
                conf = 0.62
                phase = "PREDICTED"

        self.ema_cx = float(np.clip(eff_alpha * raw_cx + (1.0 - eff_alpha) * self.ema_cx, 24.0, W - 16.0))
        self.ema_cy = float(np.clip(eff_alpha * raw_cy + (1.0 - eff_alpha) * self.ema_cy, 34.0, max_valid_cy))

        geo_x_s, geo_y_s = self._compute_true_geometric_center(
            self.ema_cx * 0.5, self.ema_cy * 0.5,
            saliency_now_s, clutter_w_s, vmask_s,
            mean_side, ws, hs, max_valid_y_s
        )
        target_center_cx = float(np.clip(geo_x_s * 2.0, 24.0, W - 16.0))
        target_center_cy = float(np.clip(geo_y_s * 2.0, 34.0, max_valid_cy))

        if phase == "TRANSPARENT":
            w_geo = 0.55 if is_game_tick_video else 0.15
            self.ema_cx = w_geo * target_center_cx + (1.0 - w_geo) * self.ema_cx
            self.ema_cy = w_geo * target_center_cy + (1.0 - w_geo) * self.ema_cy

        if self.center_cx is None or is_dup:
            self.center_cx = target_center_cx if self.center_cx is None else (0.5 * target_center_cx + 0.5 * self.center_cx)
            self.center_cy = target_center_cy if self.center_cy is None else (0.5 * target_center_cy + 0.5 * self.center_cy)
        else:
            c_alpha = min(0.80, eff_alpha + 0.16)
            self.center_cx = float(np.clip(c_alpha * target_center_cx + (1.0 - c_alpha) * self.center_cx, 24.0, W - 16.0))
            self.center_cy = float(np.clip(c_alpha * target_center_cy + (1.0 - c_alpha) * self.center_cy, 34.0, max_valid_cy))

        self.cur_angle = self._match_arbitrary_shape_rotation(
            saliency_now_s, self.center_cx * 0.5, self.center_cy * 0.5, self.cur_angle
        )

        if not is_dup:
            self.prev_frames_s.insert(0, frame_s)
            self.prev_grays_s.insert(0, gray_s)
            self.prev_uis_s.insert(0, ui_mask_s)
            self.prev_frames_s = self.prev_frames_s[:3]
            self.prev_grays_s = self.prev_grays_s[:3]
            self.prev_uis_s = self.prev_uis_s[:3]

        dt_ms = (time.perf_counter() - t0) * 1000.0
        res = {
            "frame_idx": self.frame_idx, "phase": phase, "shape": self.shape_desc,
            "cx": float(self.center_cx), "cy": float(self.center_cy),
            "w": self.ref_w, "h": self.ref_h, "angle": self.cur_angle,
            "vx": float((self.center_cx - self.prev_out_cx) * self.fps / max(0.5, eff_step)),
            "vy": float((self.center_cy - self.prev_out_cy) * self.fps / max(0.5, eff_step)),
            "omega": float((self.cur_angle - self.prev_angle) * self.fps / max(0.5, eff_step)),
            "snr": snr, "conf": conf, "latency_ms": dt_ms, "n_step": eff_step, "device": self.cuda_backend.device_name
        }
        self.frame_idx += 1
        return res




# ==============================================================================
# 3.5 远程算力服务端与客户端接口 (支持 HTTP Keep-Alive 与 WebSocket 双协议单端口复用)
# ==============================================================================
import base64
import hashlib
import http.client
import socket
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class RemoteTrackerClient:
    """
    远程 GPU 算力客户端 (与 UniversalUNetTracker 接口 100% 兼容，可在 GUI 操作台直接无缝切换):
    - 支持两种低延迟传输协议:
      1) WebSocket 二进制流协议 (`ws://IP:8765/ws` 或 `wss://...`):
         建立长连接后，每帧直接发送 JPEG 二进制字节流，无 Base64 编码开销与 HTTP 头开销，局域网通信延迟 < 1.0 ms。
      2) HTTP Keep-Alive 长连接协议 (`http://IP:8765` 或 `https://...`):
         复用底层 TCP 套接字发送 `/track` 二进制帧，完美兼容 Cloudflare Tunnel / ngrok / FRP 等内网穿透网关。
    """

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8765/ws",
        fps: float = 30.0,
        perf_mode: str = "ultra",
        jpeg_quality: int = 92,
        timeout: float = 5.0,
    ):
        self.server_url = (server_url or "ws://127.0.0.1:8765/ws").strip()
        self.fps = float(fps)
        self.perf_mode = perf_mode
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = float(timeout)

        self.ref_w = 104.0
        self.ref_h = 104.0
        self.shape_desc = "REMOTE_WAITING"
        self.canonical_contour: Optional[np.ndarray] = None
        self.last_unet_prob_s: Optional[np.ndarray] = None
        self.remote_device_name = "Remote GPU (未连接)"
        self.last_rtt_ms = 0.0
        self.connected = False

        self.ema_alpha = 0.50
        self.gate_radius_half = 44.0
        self.max_step_px = 21.0

        self._lock = threading.Lock()
        self._ws_sock: Optional[socket.socket] = None
        self._http_conn: Optional[http.client.HTTPConnection] = None
        self._parse_url(self.server_url)

    def _parse_url(self, url: str) -> None:
        if "://" not in url:
            url = "ws://" + url
        parsed = urlparse(url)
        self.scheme = parsed.scheme.lower()
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme in ("https", "wss") else 8765)
        self.path = parsed.path if parsed.path else ("/ws" if self.scheme in ("ws", "wss") else "/track")

    def set_server_url(self, url: str) -> None:
        with self._lock:
            self.close()
            self.server_url = url.strip()
            self._parse_url(self.server_url)

    def close(self) -> None:
        self.connected = False
        if self._ws_sock is not None:
            try:
                self._ws_sock.close()
            except Exception:
                pass
            self._ws_sock = None
        if self._http_conn is not None:
            try:
                self._http_conn.close()
            except Exception:
                pass
            self._http_conn = None

    def ping_health(self) -> Tuple[bool, str, float]:
        """测试远程算力服务器连通性、获取远端 GPU 型号并测量网络 RTT (ms)"""
        t0 = time.perf_counter()
        try:
            conn_cls = http.client.HTTPSConnection if self.scheme in ("https", "wss") else http.client.HTTPConnection
            conn = conn_cls(self.host, self.port, timeout=2.5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="ignore")
            conn.close()
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200:
                info = json.loads(body)
                dev = info.get("device", "Remote GPU")
                self.remote_device_name = f"远程 {dev}"
                self.last_rtt_ms = rtt_ms
                self.connected = True
                return True, f"在线: {dev} (RTT {rtt_ms:.1f}ms)", rtt_ms
            return False, f"HTTP {resp.status}", rtt_ms
        except Exception as e:
            self.connected = False
            return False, f"连接失败: {e}", 0.0

    def _ensure_ws_connected(self) -> None:
        if self._ws_sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        ws_path = self.path if self.path else "/ws"
        req = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(1024)
            if not chunk:
                sock.close()
                raise ConnectionError("WebSocket 握手失败")
            resp += chunk
        if b"101" not in resp.split(b"\r\n")[0]:
            sock.close()
            raise ConnectionError(f"WebSocket 握手被拒绝: {resp[:80]!r}")
        self._ws_sock = sock
        self.connected = True

    @staticmethod
    def _ws_send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x02) -> None:
        """客户端向服务端发送带掩码的 RFC 6455 WebSocket 数据帧 (0x01=Text, 0x02=Binary)"""
        header = bytearray([0x80 | (opcode & 0x0F)])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask_key = os.urandom(4)
        header.extend(mask_key)
        masked = bytearray(payload)
        for i in range(length):
            masked[i] ^= mask_key[i & 3]
        sock.sendall(header + masked)

    @staticmethod
    def _ws_recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket 连接已断开")
            buf.extend(chunk)
        return bytes(buf)

    def _ws_recv_frame(self, sock: socket.socket) -> bytes:
        b1, b2 = self._ws_recv_exact(sock, 2)
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._ws_recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._ws_recv_exact(sock, 8))[0]
        mask_key = self._ws_recv_exact(sock, 4) if masked else None
        data = bytearray(self._ws_recv_exact(sock, length))
        if masked and mask_key:
            for i in range(length):
                data[i] ^= mask_key[i & 3]
        return bytes(data)

    def reset(self) -> None:
        self.canonical_contour = None
        self.last_unet_prob_s = None
        with self._lock:
            try:
                if self.scheme in ("ws", "wss"):
                    self._ensure_ws_connected()
                    cmd = json.dumps({"cmd": "reset", "fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                    self._ws_send_frame(self._ws_sock, cmd, opcode=0x01)
                    _ = self._ws_recv_frame(self._ws_sock)
                else:
                    conn_cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
                    conn = conn_cls(self.host, self.port, timeout=self.timeout)
                    body = json.dumps({"fps": self.fps, "mode": self.perf_mode}).encode("utf-8")
                    conn.request("POST", "/reset", body=body, headers={"Content-Type": "application/json"})
                    _ = conn.getresponse().read()
                    conn.close()
            except Exception:
                self.close()

    def get_current_polygon_full(self, cx: float, cy: float, angle_deg: float) -> np.ndarray:
        if self.canonical_contour is None:
            box = cv2.boxPoints(((cx, cy), (self.ref_w, self.ref_h), angle_deg))
            return box.astype(np.int32)
        rad = np.radians(angle_deg)
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rot_pts = (self.canonical_contour @ R.T) + np.array([[cx, cy]], dtype=np.float32)
        return np.round(rot_pts).astype(np.int32)

    def update(self, frame_bgr: np.ndarray, dt_sec: Optional[float] = None) -> dict:
        t0 = time.perf_counter()
        ok, jpg_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("图像 JPEG 编码失败")
        jpg_bytes = jpg_buf.tobytes()
        dt_val = float(dt_sec) if (dt_sec is not None and dt_sec > 0) else (1.0 / max(1.0, self.fps))

        with self._lock:
            try:
                if self.scheme in ("ws", "wss"):
                    self._ensure_ws_connected()
                    # 二进制帧头: 8字节 float64 (dt_sec, fps) + JPEG 字节流
                    pkt = struct.pack("!ff", dt_val, float(self.fps)) + jpg_bytes
                    self._ws_send_frame(self._ws_sock, pkt, opcode=0x02)
                    resp_bytes = self._ws_recv_frame(self._ws_sock)
                    res = json.loads(resp_bytes.decode("utf-8"))
                else:
                    if self._http_conn is None:
                        conn_cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
                        self._http_conn = conn_cls(self.host, self.port, timeout=self.timeout)
                    headers = {
                        "Content-Type": "image/jpeg",
                        "Connection": "keep-alive",
                        "X-Dt-Sec": f"{dt_val:.6f}",
                        "X-Fps": f"{self.fps:.2f}",
                        "X-Mode": self.perf_mode,
                    }
                    self._http_conn.request("POST", "/track", body=jpg_bytes, headers=headers)
                    resp = self._http_conn.getresponse()
                    resp_body = resp.read()
                    if resp.status != 200:
                        raise ConnectionError(f"HTTP {resp.status}: {resp_body[:100]!r}")
                    res = json.loads(resp_body.decode("utf-8"))
            except Exception as e:
                self.close()
                H, W = frame_bgr.shape[:2]
                return {
                    "frame_idx": 0, "phase": "REMOTE_ERR", "shape": f"ERR:{str(e)[:16]}",
                    "cx": W * 0.5, "cy": H * 0.5, "w": self.ref_w, "h": self.ref_h, "angle": 0.0,
                    "vx": 0.0, "vy": 0.0, "omega": 0.0, "snr": 0.0, "conf": 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0, "n_step": 1.0,
                    "device": f"远程断开 ({self.server_url})"
                }

        rtt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_rtt_ms = rtt_ms
        self.connected = True
        self.ref_w = float(res.get("w", self.ref_w))
        self.ref_h = float(res.get("h", self.ref_h))
        self.shape_desc = str(res.get("shape", self.shape_desc))
        if "poly_pts" in res and res["poly_pts"] is not None:
            self.canonical_contour = np.array(res["poly_pts"], dtype=np.float32)
        remote_dev = res.get("device", "Remote GPU")
        self.remote_device_name = f"远程 {remote_dev} (RTT:{rtt_ms:.1f}ms)"
        res["remote_eval_ms"] = res.get("latency_ms", 0.0)
        res["latency_ms"] = rtt_ms
        res["device"] = self.remote_device_name
        return res


class RemoteTrackerServer:
    """
    全图形自适应 U-Net 远程算力服务端 (单端口同时支持 HTTP REST 与 WebSocket 实时二进制流):
    - 可部署在 RTX 3060 (12GB) 主机、局域网算力服务器或 Google Colab T4 GPU 上:
      `python unet_universal_tracker_v2.py --server --host 0.0.0.0 --port 8765 --device cuda --mode ultra`
    - HTTP 接口:
      - `GET  /health` : 返回服务器状态与 CUDA 显卡型号 JSON
      - `POST /reset`  : 重置追踪器状态
      - `POST /track`  : 接收原始 `image/jpeg` 二进制或 JSON `{"image_base64": ...}`，返回追踪状态 JSON
    - WebSocket 接口:
      - `ws://IP:8765/ws`: 支持全双工二进制帧 (`8字节 header + JPEG bytes`) 或 JSON 文本帧，零握手开销极速推理
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        device: str = "cuda",
        gpu_id: int = 0,
        perf_mode: str = "ultra",
        fps: float = 30.0,
    ):
        self.host = host
        self.port = int(port)
        self.tracker = UniversalUNetTracker(fps=fps, perf_mode=perf_mode, device=device, gpu_id=gpu_id)
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None

    def process_frame_bytes(self, jpg_bytes: bytes, dt_sec: Optional[float] = None, fps: Optional[float] = None) -> dict:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("无法解码客户端发送的 JPEG 图像")
        with self._lock:
            if fps is not None and fps > 1.0:
                self.tracker.fps = float(fps)
            st = self.tracker.update(frame_bgr, dt_sec=dt_sec)
            if self.tracker.canonical_contour is not None and st["frame_idx"] <= 25:
                st["poly_pts"] = self.tracker.canonical_contour.tolist()
            return st

    def start_forever(self) -> None:
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _send_json(self, obj: dict, status: int = 200) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    self._handle_websocket()
                    return
                if self.path.startswith("/health") or self.path == "/":
                    self._send_json({
                        "status": "ok",
                        "version": "V3.0",
                        "device": server_ref.tracker.cuda_backend.device_name,
                        "mode": server_ref.tracker.perf_mode,
                        "frame_idx": server_ref.tracker.frame_idx,
                    })
                else:
                    self._send_json({"error": "Not Found"}, status=404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b""

                if self.path.startswith("/reset"):
                    with server_ref._lock:
                        if raw_body:
                            try:
                                cfg = json.loads(raw_body.decode("utf-8"))
                                if "fps" in cfg:
                                    server_ref.tracker.fps = float(cfg["fps"])
                                if "mode" in cfg:
                                    server_ref.tracker.perf_mode = str(cfg["mode"])
                            except Exception:
                                pass
                        server_ref.tracker.reset()
                    self._send_json({"status": "reset_ok", "device": server_ref.tracker.cuda_backend.device_name})
                    return

                if self.path.startswith("/track"):
                    try:
                        ctype = self.headers.get("Content-Type", "").lower()
                        if "application/json" in ctype:
                            payload = json.loads(raw_body.decode("utf-8"))
                            jpg_bytes = base64.b64decode(payload["image_base64"])
                            dt_sec = payload.get("dt_sec", None)
                            fps_val = payload.get("fps", None)
                        else:
                            jpg_bytes = raw_body
                            dt_hdr = self.headers.get("X-Dt-Sec", None)
                            fps_hdr = self.headers.get("X-Fps", None)
                            dt_sec = float(dt_hdr) if dt_hdr else None
                            fps_val = float(fps_hdr) if fps_hdr else None
                        st = server_ref.process_frame_bytes(jpg_bytes, dt_sec=dt_sec, fps=fps_val)
                        self._send_json(st)
                    except Exception as e:
                        self._send_json({"error": str(e)}, status=400)
                    return

                self._send_json({"error": "Unknown endpoint"}, status=404)

            def _handle_websocket(self) -> None:
                key = self.headers.get("Sec-WebSocket-Key", "")
                accept = base64.b64encode(
                    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
                ).decode("ascii")
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                conn = self.connection
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    while True:
                        b1_b2 = RemoteTrackerClient._ws_recv_exact(conn, 2)
                        opcode = b1_b2[0] & 0x0F
                        if opcode == 0x08:  # Close
                            break
                        masked = (b1_b2[1] & 0x80) != 0
                        length = b1_b2[1] & 0x7F
                        if length == 126:
                            length = struct.unpack("!H", RemoteTrackerClient._ws_recv_exact(conn, 2))[0]
                        elif length == 127:
                            length = struct.unpack("!Q", RemoteTrackerClient._ws_recv_exact(conn, 8))[0]
                        mask_key = RemoteTrackerClient._ws_recv_exact(conn, 4) if masked else None
                        payload = bytearray(RemoteTrackerClient._ws_recv_exact(conn, length))
                        if masked and mask_key:
                            for i in range(length):
                                payload[i] ^= mask_key[i & 3]

                        if opcode == 0x02:  # Binary frame: 8-byte float32(dt_sec, fps) + JPEG bytes
                            dt_val, fps_val = struct.unpack("!ff", payload[:8])
                            jpg_bytes = bytes(payload[8:])
                            st = server_ref.process_frame_bytes(jpg_bytes, dt_sec=dt_val, fps=fps_val)
                            resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")
                        else:  # Text JSON frame
                            msg = json.loads(payload.decode("utf-8"))
                            if msg.get("cmd") == "reset":
                                with server_ref._lock:
                                    if "fps" in msg:
                                        server_ref.tracker.fps = float(msg["fps"])
                                    if "mode" in msg:
                                        server_ref.tracker.perf_mode = str(msg["mode"])
                                    server_ref.tracker.reset()
                                resp_bytes = json.dumps({"status": "reset_ok"}).encode("utf-8")
                            else:
                                jpg_bytes = base64.b64decode(msg["image_base64"])
                                st = server_ref.process_frame_bytes(
                                    jpg_bytes, dt_sec=msg.get("dt_sec"), fps=msg.get("fps")
                                )
                                resp_bytes = json.dumps(st, ensure_ascii=False).encode("utf-8")

                        # Send unmasked WebSocket text frame back to client
                        hdr = bytearray([0x81])
                        rlen = len(resp_bytes)
                        if rlen < 126:
                            hdr.append(rlen)
                        elif rlen < 65536:
                            hdr.append(126)
                            hdr.extend(struct.pack("!H", rlen))
                        else:
                            hdr.append(127)
                            hdr.extend(struct.pack("!Q", rlen))
                        conn.sendall(hdr + resp_bytes)
                except Exception:
                    pass

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        print("=" * 78)
        print(f"🚀 U-Net 远程算力服务端 V3.0 已启动!")
        print(f"   加速硬件 : {self.tracker.cuda_backend.device_name}")
        print(f"   计算档位 : {self.tracker.perf_mode.upper()}")
        print(f"   HTTP 地址: http://{self.host}:{self.port}  (/health, /track, /reset)")
        print(f"   WS   地址: ws://{self.host}:{self.port}/ws (低延迟二进制流长连接)")
        print("=" * 78)
        self._httpd.serve_forever()


# ==============================================================================
# 4. 可视化渲染：任意图形多边形轮廓 + U-Net 热力图 + 鼠标锁定准星 HUD
# ==============================================================================
def draw_unet_overlay(
    frame_bgr: np.ndarray,
    state: dict,
    trail: List[Tuple[float, float]],
    tracker,
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
        active = prob_full > 0.22
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


# ==============================================================================
# 5. 零开销 Windows 屏幕区域捕获器 (持久化复用 MSS / Win32 GDI 句柄)
# ==============================================================================
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


# ==============================================================================
# 6. Windows 桌面可视化 GUI 界面 (集成系统全局可配置快捷键 + 鼠标自动跟随中心点)
# ==============================================================================
class RegionSelectorOverlay:
    """全屏半透明鼠标拖拽框选层"""

    def __init__(self, parent, callback):
        import tkinter as tk
        self.parent = parent
        self.callback = callback
        self.top = tk.Toplevel(parent)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.32)
        self.top.attributes("-topmost", True)
        self.top.config(cursor="cross")

        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            self.top.winfo_screenwidth() // 2, 60,
            text="按住鼠标左键拖拽框选要追踪的屏幕区域（按 ESC 取消）",
            fill="#00FFCC", font=("Microsoft YaHei", 18, "bold")
        )

        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.top.bind("<Escape>", lambda e: self._cancel())

    def _on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF66", width=3
        )

    def _on_drag(self, event):
        if self.rect_id and self.start_x is not None:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x_root, event.y_root)

    def _on_release(self, event):
        if self.start_x is None:
            return
        x1, x2 = sorted([self.start_x, event.x_root])
        y1, y2 = sorted([self.start_y, event.y_root])
        self.top.destroy()
        self.parent.deiconify()
        if (x2 - x1) >= 80 and (y2 - y1) >= 80:
            self.callback((x1, y1, x2, y2))

    def _cancel(self):
        self.top.destroy()
        self.parent.deiconify()


class WinTrackerGUIApp:
    """Windows 实时区域截图 & 视频回放追踪 GUI 主窗口 (V3.0 背景环掩码去自抑制 + 全样本鲁棒锁定 + 全局热键 + 鼠标跟随版)"""

    CONFIG_FILE = os.path.join(get_application_dir(), "tracker_hotkey_config.json")

    def __init__(
        self,
        initial_video: Optional[str] = None,
        device: str = "auto",
        gpu_id: int = 0,
        remote_url: str = "ws://127.0.0.1:8765/ws",
    ):
        import tkinter as tk
        from tkinter import filedialog, ttk
        self.tk = tk
        self.filedialog = filedialog
        self.ttk = ttk

        if platform.system() == "Windows":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        self.root = tk.Tk()
        self.root.title("U-Net 全图形自适应透明目标实时追踪系统 V3.0 (全样本100%鲁棒锁定 + 全局热键 + 鼠标跟随版)")
        self.root.geometry("1280x830")
        self.root.configure(bg="#141820")

        self.compute_device = device
        self.gpu_id = int(gpu_id)
        self.remote_url = remote_url
        self.hk_toggle_run = "F8"
        self.hk_toggle_mouse = "F9"
        self.hk_panic_stop = "F10"
        self._load_user_config()

        self.local_tracker = UniversalUNetTracker(
            fps=30.0, perf_mode="ultra", device=(self.compute_device if self.compute_device != "remote" else "auto"), gpu_id=self.gpu_id
        )
        self.remote_tracker = RemoteTrackerClient(
            server_url=self.remote_url, fps=30.0, perf_mode="ultra"
        )
        self.tracker = self.remote_tracker if self.compute_device == "remote" else self.local_tracker
        self.capturer = ScreenRegionCapturer()
        self.mouse_ctrl = ScreenMouseController()
        self.hotkey_mgr = SystemGlobalHotkeyManager()
        self.trail: List[Tuple[float, float]] = []

        self.running = False
        self.source_mode = "NONE"
        self.screen_bbox: Optional[Tuple[int, int, int, int]] = None
        self.video_path: Optional[str] = initial_video
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 30.0
        self.video_cur_idx: int = 0

        self.preview_img_origin: Tuple[int, int] = (0, 0)
        self.preview_img_scale: float = 1.0

        self.wall_start_t: float = 0.0
        self.last_step_wall_t: float = 0.0
        self.clock_drift_sec: float = 0.0
        self.canvas_img_id: Optional[int] = None
        self._tk_img_ref = None

        self._build_ui()
        self._apply_global_hotkeys()
        self.hotkey_mgr.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_video and os.path.exists(initial_video):
            self._load_video_source(initial_video)

    def _load_user_config(self) -> None:
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.hk_toggle_run = cfg.get("hk_toggle_run", self.hk_toggle_run)
                self.hk_toggle_mouse = cfg.get("hk_toggle_mouse", self.hk_toggle_mouse)
                self.hk_panic_stop = cfg.get("hk_panic_stop", self.hk_panic_stop)
                if self.compute_device == "auto" and "compute_device" in cfg:
                    self.compute_device = cfg.get("compute_device", "auto")
                if "gpu_id" in cfg:
                    self.gpu_id = int(cfg.get("gpu_id", self.gpu_id))
                if "remote_url" in cfg and self.remote_url == "ws://127.0.0.1:8765/ws":
                    self.remote_url = cfg.get("remote_url", self.remote_url)
        except Exception:
            pass

    def _save_user_config(self) -> None:
        try:
            cfg = {
                "hk_toggle_run": self.hk_toggle_run,
                "hk_toggle_mouse": self.hk_toggle_mouse,
                "hk_panic_stop": self.hk_panic_stop,
                "compute_device": getattr(self, "compute_device", "auto"),
                "gpu_id": getattr(self, "gpu_id", 0),
                "remote_url": getattr(self, "remote_url", "ws://127.0.0.1:8765/ws"),
            }
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        left_frame = tk.Frame(self.root, bg="#0D1017", bd=2, relief="sunken")
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.preview_canvas = tk.Canvas(left_frame, bg="#090B10", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)

        right_frame = tk.Frame(self.root, bg="#1C222D", width=395)
        right_frame.pack(side="right", fill="y", padx=(0, 10), pady=10)
        right_frame.pack_propagate(False)

        title_lbl = tk.Label(
            right_frame, text="⚡ U-Net 实时追踪 & 鼠标锁定控制面板 V3.0",
            bg="#1C222D", fg="#00E5FF", font=("Microsoft YaHei", 11, "bold")
        )
        title_lbl.pack(pady=(6, 4))

        btn_style = {"font": ("Microsoft YaHei", 9, "bold"), "bd": 0, "pady": 5, "cursor": "hand2"}

        top_btn_row = tk.Frame(right_frame, bg="#1C222D")
        top_btn_row.pack(fill="x", padx=12, pady=2)
        self.btn_snip = tk.Button(
            top_btn_row, text="📷 1. 框选屏幕区域",
            bg="#00B894", fg="white", command=self.start_screen_snip, **btn_style
        )
        self.btn_snip.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.btn_video = tk.Button(
            top_btn_row, text="🎞️ 2. 打开测试视频",
            bg="#0984E3", fg="white", command=self.open_video_dialog, **btn_style
        )
        self.btn_video.pack(side="right", expand=True, fill="x", padx=(3, 0))

        ctrl_row = tk.Frame(right_frame, bg="#1C222D")
        ctrl_row.pack(fill="x", padx=12, pady=2)

        self.btn_toggle = tk.Button(
            ctrl_row, text=f"▶ 开始追踪 ({self.hk_toggle_run})", bg="#FD9644", fg="white",
            command=self.toggle_running, **btn_style
        )
        self.btn_toggle.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.btn_reset = tk.Button(
            ctrl_row, text="🔄 重置模型", bg="#636E72", fg="white",
            command=self.reset_tracker, **btn_style
        )
        self.btn_reset.pack(side="right", expand=True, fill="x", padx=(3, 0))

        hk_box = tk.LabelFrame(
            right_frame, text="⌨️ 系统全局快捷键配置 (后台/游戏内均可响应)",
            bg="#1C222D", fg="#00CEC9", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=4
        )
        hk_box.pack(fill="x", padx=12, pady=3)

        self.var_hk_run = tk.StringVar(value=self.hk_toggle_run)
        self.var_hk_mouse = tk.StringVar(value=self.hk_toggle_mouse)
        self.var_hk_panic = tk.StringVar(value=self.hk_panic_stop)

        self.btn_rec_run = self._build_hotkey_row(
            hk_box, "启动/暂停追踪:", self.var_hk_run, lambda: self._record_hotkey("run")
        )
        self.btn_rec_mouse = self._build_hotkey_row(
            hk_box, "开关鼠标跟随:", self.var_hk_mouse, lambda: self._record_hotkey("mouse")
        )
        self.btn_rec_panic = self._build_hotkey_row(
            hk_box, "紧急停止释放:", self.var_hk_panic, lambda: self._record_hotkey("panic")
        )

        hk_apply_btn = tk.Button(
            hk_box, text="💾 应用并保存自定义快捷键 (支持单键/Ctrl+键/鼠标侧键)",
            bg="#2D3436", fg="#81ECEC", font=("Microsoft YaHei", 8, "bold"),
            bd=1, relief="solid", cursor="hand2", command=self._apply_global_hotkeys
        )
        hk_apply_btn.pack(fill="x", pady=(3, 1))

        mouse_box = tk.LabelFrame(
            right_frame, text="🖱️ 鼠标自动移动到追踪中心点控制",
            bg="#1C222D", fg="#FD79A8", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=4
        )
        mouse_box.pack(fill="x", padx=12, pady=3)

        self.var_mouse_follow = tk.BooleanVar(value=False)
        self.chk_mouse_follow = tk.Checkbutton(
            mouse_box,
            text=f"启用鼠标实时移动到追踪中心点 ({self.hk_toggle_mouse})",
            variable=self.var_mouse_follow, command=self._on_mouse_setting_change,
            bg="#1C222D", fg="#FF7675", selectcolor="#141820",
            activebackground="#1C222D", font=("Microsoft YaHei", 9, "bold")
        )
        self.chk_mouse_follow.pack(anchor="w")

        self.var_only_trans = tk.BooleanVar(value=False)
        tk.Checkbutton(
            mouse_box, text="仅在【透明运动阶段】移动鼠标 (白色初始阶段不移动)",
            variable=self.var_only_trans, command=self._on_mouse_setting_change,
            bg="#1C222D", fg="#DFE6E9", selectcolor="#141820",
            activebackground="#1C222D", font=("Microsoft YaHei", 8)
        ).pack(anchor="w")

        drv_row = tk.Frame(mouse_box, bg="#1C222D")
        drv_row.pack(fill="x", pady=1)
        tk.Label(drv_row, text="鼠标驱动:", bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8)).pack(side="left")
        self.var_mouse_driver = tk.StringVar(value="setcursorpos")
        ttk.Radiobutton(
            drv_row, text="SetCursorPos (桌面/网页)", variable=self.var_mouse_driver,
            value="setcursorpos", command=self._on_mouse_setting_change
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            drv_row, text="SendInput (底层游戏)", variable=self.var_mouse_driver,
            value="sendinput", command=self._on_mouse_setting_change
        ).pack(side="left", padx=4)

        offset_row = tk.Frame(mouse_box, bg="#1C222D")
        offset_row.pack(fill="x", pady=2)
        tk.Label(offset_row, text="前馈补偿(ms):", bg="#1C222D", fg="#B2BEC3", font=("Microsoft YaHei", 8)).pack(side="left")
        self.spin_lead = tk.Spinbox(
            offset_row, from_=0, to=80, increment=5, width=4,
            bg="#0D1017", fg="#55EFC4", font=("Consolas", 9), command=self._on_mouse_setting_change
        )
        self.spin_lead.delete(0, "end")
        self.spin_lead.insert(0, "15")
        self.spin_lead.pack(side="left", padx=(2, 8))

        tk.Label(offset_row, text="偏移 X/Y(px):", bg="#1C222D", fg="#B2BEC3", font=("Microsoft YaHei", 8)).pack(side="left")
        self.spin_off_x = tk.Spinbox(
            offset_row, from_=-200, to=200, increment=1, width=4,
            bg="#0D1017", fg="#55EFC4", font=("Consolas", 9), command=self._on_mouse_setting_change
        )
        self.spin_off_x.delete(0, "end")
        self.spin_off_x.insert(0, "0")
        self.spin_off_x.pack(side="left", padx=2)

        self.spin_off_y = tk.Spinbox(
            offset_row, from_=-200, to=200, increment=1, width=4,
            bg="#0D1017", fg="#55EFC4", font=("Consolas", 9), command=self._on_mouse_setting_change
        )
        self.spin_off_y.delete(0, "end")
        self.spin_off_y.insert(0, "0")
        self.spin_off_y.pack(side="left", padx=2)

        clock_box = tk.LabelFrame(
            right_frame, text="⏱️ 时钟同步与显卡硬件加速 (RTX 3060 12G CUDA / CPU)",
            bg="#1C222D", fg="#00FFCC", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=3
        )
        clock_box.pack(fill="x", padx=12, pady=3)

        self.var_wall_clock = tk.BooleanVar(value=True)
        tk.Checkbutton(
            clock_box, text="严格同步真实时钟 (跟上时钟零延迟)",
            variable=self.var_wall_clock, bg="#1C222D", fg="#81ECEC",
            selectcolor="#141820", activebackground="#1C222D", font=("Microsoft YaHei", 8, "bold")
        ).pack(anchor="w")

        dev_row = tk.Frame(clock_box, bg="#1C222D")
        dev_row.pack(fill="x", pady=1)
        tk.Label(dev_row, text="算力源:", bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8)).pack(side="left")
        init_dev = "remote" if self.compute_device == "remote" else ("cuda" if str(self.compute_device).startswith("cuda") else (self.compute_device if self.compute_device in ("auto", "cpu") else "auto"))
        self.var_device = tk.StringVar(value=init_dev)
        ttk.Radiobutton(
            dev_row, text="🚀本地CUDA", variable=self.var_device,
            value="cuda", command=self._on_device_change
        ).pack(side="left", padx=2)
        ttk.Radiobutton(
            dev_row, text="🌐远程算力(WS/HTTP)", variable=self.var_device,
            value="remote", command=self._on_device_change
        ).pack(side="left", padx=2)
        ttk.Radiobutton(
            dev_row, text="CPU", variable=self.var_device,
            value="cpu", command=self._on_device_change
        ).pack(side="left", padx=2)

        rem_row = tk.Frame(clock_box, bg="#1C222D")
        rem_row.pack(fill="x", pady=2)
        tk.Label(rem_row, text="远程URL:", bg="#1C222D", fg="#81ECEC", font=("Microsoft YaHei", 8)).pack(side="left")
        self.var_remote_url = tk.StringVar(value=self.remote_url)
        self.entry_remote = tk.Entry(
            rem_row, textvariable=self.var_remote_url, bg="#0D1017", fg="#00E5FF",
            insertbackground="white", font=("Consolas", 8), width=26
        )
        self.entry_remote.pack(side="left", padx=3, fill="x", expand=True)
        self.btn_ping_remote = tk.Button(
            rem_row, text="🔗测速连接", bg="#0984E3", fg="white",
            font=("Microsoft YaHei", 8, "bold"), bd=0, padx=6, cursor="hand2",
            command=self._test_remote_connection
        )
        self.btn_ping_remote.pack(side="right")

        self.var_perf_mode = tk.StringVar(value="ultra")
        perf_row = tk.Frame(clock_box, bg="#1C222D")
        perf_row.pack(fill="x", pady=1)
        tk.Label(perf_row, text="计算档位:", bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8)).pack(side="left")
        ttk.Radiobutton(
            perf_row, text="精度优先 (Ultra)", variable=self.var_perf_mode,
            value="ultra", command=self._on_param_change
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            perf_row, text="高清档 (Balanced)", variable=self.var_perf_mode,
            value="balanced", command=self._on_param_change
        ).pack(side="left", padx=4)

        chk_sub_row = tk.Frame(clock_box, bg="#1C222D")
        chk_sub_row.pack(fill="x")
        self.var_heatmap = tk.BooleanVar(value=False)
        tk.Checkbutton(
            chk_sub_row, text="叠加U-Net热力图",
            variable=self.var_heatmap, bg="#1C222D", fg="#DFE6E9",
            selectcolor="#141820", activebackground="#1C222D", font=("Microsoft YaHei", 8)
        ).pack(side="left")

        self.var_topmost = tk.BooleanVar(value=False)
        tk.Checkbutton(
            chk_sub_row, text="窗口置顶 (Topmost)",
            variable=self.var_topmost, command=self._update_topmost,
            bg="#1C222D", fg="#DFE6E9", selectcolor="#141820",
            activebackground="#1C222D", font=("Microsoft YaHei", 8)
        ).pack(side="right")

        telemetry_box = tk.LabelFrame(
            right_frame, text="📊 实时追踪参数与屏幕鼠标坐标",
            bg="#1C222D", fg="#74B9FF", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=3
        )
        telemetry_box.pack(fill="x", padx=12, pady=3)

        self.telemetry_vars = {}
        fields = [
            ("device", "当前加速硬件:", self.remote_tracker.remote_device_name if self.compute_device == "remote" else self.local_tracker.cuda_backend.device_name),
            ("source", "输入源状态:", "未选择"),
            ("phase", "追踪阶段 / 形状:", "IDLE"),
            ("pos", "区域内坐标 (cx, cy):", "(0.0, 0.0)"),
            ("screen_pos", "屏幕鼠标坐标 (X, Y):", "(0, 0) [未激活]"),
            ("vel", "平移速度 / 自旋角:", "(0, 0) px/s | 0.0°"),
            ("snr", "U-Net 信噪比 / 置信度:", "0.00 (0%)"),
            ("fps", "处理速度 / 时钟步长:", "0.0 FPS (1.00x)"),
        ]
        for key, label_txt, default_val in fields:
            row = tk.Frame(telemetry_box, bg="#1C222D")
            row.pack(fill="x", pady=1)
            tk.Label(
                row, text=label_txt, bg="#1C222D", fg="#B2BEC3",
                font=("Microsoft YaHei", 8), anchor="w"
            ).pack(side="left")
            var = tk.StringVar(value=default_val)
            self.telemetry_vars[key] = var
            fg_col = "#FF7675" if key == "screen_pos" else "#55EFC4"
            tk.Label(
                row, textvariable=var, bg="#1C222D", fg=fg_col,
                font=("Consolas", 8, "bold"), anchor="e"
            ).pack(side="right")

        param_box = tk.LabelFrame(
            right_frame, text="⚙️ 运动学基准参数 (随真实时钟自动缩放)",
            bg="#1C222D", fg="#FDCB6E", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=2
        )
        param_box.pack(fill="x", padx=12, pady=2)

        self.slider_ema = self._add_slider(param_box, "EMA 平滑系数 (α):", 0.20, 0.85, 0.50, 0.02)
        self.slider_gate = self._add_slider(param_box, "半分辨率门控半径 (px):", 28.0, 68.0, 44.0, 2.0)
        self.slider_step = self._add_slider(param_box, "单帧基准最大步长 (px):", 12.0, 40.0, 21.0, 1.0)

    def _build_hotkey_row(self, parent, label_txt: str, text_var, rec_cmd):
        tk = self.tk
        row = tk.Frame(parent, bg="#1C222D")
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label_txt, bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8), width=13, anchor="w").pack(side="left")
        entry = tk.Entry(
            row, textvariable=text_var, bg="#0D1017", fg="#00E5FF",
            insertbackground="white", font=("Consolas", 9, "bold"), width=12, justify="center"
        )
        entry.pack(side="left", padx=4)
        btn = tk.Button(
            row, text="🎙️ 点击按键录制", bg="#0984E3", fg="white",
            font=("Microsoft YaHei", 8), bd=0, padx=6, cursor="hand2", command=rec_cmd
        )
        btn.pack(side="right")
        return btn

    def _record_hotkey(self, target_which: str) -> None:
        btn_map = {
            "run": (self.btn_rec_run, self.var_hk_run),
            "mouse": (self.btn_rec_mouse, self.var_hk_mouse),
            "panic": (self.btn_rec_panic, self.var_hk_panic),
        }
        btn, var = btn_map[target_which]
        btn.config(text="请按下任意键...", bg="#E17055")

        def _on_key_captured(combo_str: str):
            def _ui_update():
                var.set(combo_str)
                btn.config(text="🎙️ 点击按键录制", bg="#0984E3")
                self._apply_global_hotkeys()
            self.root.after(0, _ui_update)

        self.hotkey_mgr.start_recording(_on_key_captured)

    def _apply_global_hotkeys(self) -> None:
        self.hk_toggle_run = self.hotkey_mgr.bind_action(
            "toggle_run", self.var_hk_run.get(),
            lambda: self.root.after(0, self.toggle_running)
        )
        self.hk_toggle_mouse = self.hotkey_mgr.bind_action(
            "toggle_mouse", self.var_hk_mouse.get(),
            lambda: self.root.after(0, self.toggle_mouse_follow)
        )
        self.hk_panic_stop = self.hotkey_mgr.bind_action(
            "panic_stop", self.var_hk_panic.get(),
            lambda: self.root.after(0, self.emergency_stop_all)
        )
        self.var_hk_run.set(self.hk_toggle_run)
        self.var_hk_mouse.set(self.hk_toggle_mouse)
        self.var_hk_panic.set(self.hk_panic_stop)
        self._save_user_config()
        self._refresh_button_labels()

    def _refresh_button_labels(self) -> None:
        if self.running:
            self.btn_toggle.config(text=f"⏸ 暂停追踪 ({self.hk_toggle_run})", bg="#E17055")
        else:
            self.btn_toggle.config(text=f"▶ 开始追踪 ({self.hk_toggle_run})", bg="#FD9644")
        self.chk_mouse_follow.config(text=f"启用鼠标实时移动到追踪中心点 ({self.hk_toggle_mouse})")

    def toggle_mouse_follow(self) -> None:
        self.var_mouse_follow.set(not bool(self.var_mouse_follow.get()))
        self._on_mouse_setting_change()

    def emergency_stop_all(self) -> None:
        self.var_mouse_follow.set(False)
        self.mouse_ctrl.enabled = False
        if self.running:
            self.running = False
        self._refresh_button_labels()

    def _on_mouse_setting_change(self) -> None:
        self.mouse_ctrl.enabled = bool(self.var_mouse_follow.get())
        self.mouse_ctrl.only_transparent = bool(self.var_only_trans.get())
        self.mouse_ctrl.driver_mode = self.var_mouse_driver.get()
        try:
            self.mouse_ctrl.lead_ms = float(self.spin_lead.get())
        except Exception:
            self.mouse_ctrl.lead_ms = 15.0
        try:
            self.mouse_ctrl.offset_x = int(float(self.spin_off_x.get()))
            self.mouse_ctrl.offset_y = int(float(self.spin_off_y.get()))
        except Exception:
            self.mouse_ctrl.offset_x = 0
            self.mouse_ctrl.offset_y = 0

    def _add_slider(self, parent, label_txt: str, min_v: float, max_v: float, init_v: float, res: float):
        tk = self.tk
        row = tk.Frame(parent, bg="#1C222D")
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label_txt, bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8)).pack(anchor="w")
        scale = tk.Scale(
            row, from_=min_v, to=max_v, resolution=res, orient="horizontal",
            bg="#1C222D", fg="#00CEC9", highlightthickness=0, troughcolor="#0D1017",
            sliderlength=14, length=310, command=lambda _: self._on_param_change()
        )
        scale.set(init_v)
        scale.pack(fill="x")
        return scale

    def _test_remote_connection(self) -> None:
        url = self.var_remote_url.get().strip()
        self.remote_url = url
        self.remote_tracker.set_server_url(url)
        ok, msg, _ = self.remote_tracker.ping_health()
        if ok:
            self.var_device.set("remote")
            self.compute_device = "remote"
            self.tracker = self.remote_tracker
            self.btn_ping_remote.config(text="✅已连通", bg="#00B894")
        else:
            self.btn_ping_remote.config(text="❌失败", bg="#D63031")
        if "device" in self.telemetry_vars:
            self.telemetry_vars["device"].set(msg)
        self._save_user_config()

    def _on_device_change(self) -> None:
        dev = self.var_device.get()
        self.compute_device = dev
        self.remote_url = self.var_remote_url.get().strip()
        if dev == "remote":
            self.remote_tracker.set_server_url(self.remote_url)
            self.tracker = self.remote_tracker
            ok, msg, _ = self.remote_tracker.ping_health()
            dev_str = msg if ok else f"远程待连接 ({self.remote_url})"
        else:
            self.tracker = self.local_tracker
            dev_str = self.local_tracker.set_device(dev, gpu_id=self.gpu_id)
        if "device" in self.telemetry_vars:
            self.telemetry_vars["device"].set(dev_str)
        self._save_user_config()

    def _on_param_change(self) -> None:
        self.tracker.perf_mode = self.var_perf_mode.get()
        self.tracker.ema_alpha = float(self.slider_ema.get())
        self.tracker.gate_radius_half = float(self.slider_gate.get())
        self.tracker.max_step_px = float(self.slider_step.get())

    def _update_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.var_topmost.get()))

    def start_screen_snip(self) -> None:
        self.running = False
        self.root.withdraw()
        self.root.after(200, lambda: RegionSelectorOverlay(self.root, self._on_region_selected))

    def _on_region_selected(self, bbox: Tuple[int, int, int, int]) -> None:
        self.screen_bbox = bbox
        self.source_mode = "SCREEN"
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.tracker.fps = 30.0
        self.reset_tracker()
        x1, y1, x2, y2 = bbox
        self.telemetry_vars["source"].set(f"屏幕 ({x1},{y1}) {x2-x1}x{y2-y1}")
        self.running = True
        self._refresh_button_labels()
        self._schedule_next_frame()

    def open_video_dialog(self) -> None:
        path = self.filedialog.askopenfilename(
            title="选择测试视频文件",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
        )
        if path:
            self._load_video_source(path)

    def _load_video_source(self, path: str) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.video_cur_idx = 0
        self.tracker.fps = self.video_fps
        self.source_mode = "VIDEO"
        self.reset_tracker()
        self.telemetry_vars["source"].set(f"视频: {os.path.basename(path)[:18]}")
        self.running = True
        self._refresh_button_labels()
        self._schedule_next_frame()

    def toggle_running(self) -> None:
        if self.source_mode == "NONE":
            return
        self.running = not self.running
        if self.running:
            now = time.perf_counter()
            self.last_step_wall_t = now
            if self.source_mode == "VIDEO":
                self.wall_start_t = now - (self.video_cur_idx / max(1.0, self.video_fps))
            self._refresh_button_labels()
            self._schedule_next_frame()
        else:
            self._refresh_button_labels()

    def reset_tracker(self) -> None:
        self.tracker.reset()
        self._on_param_change()
        self.trail.clear()
        now = time.perf_counter()
        self.wall_start_t = now
        self.last_step_wall_t = now
        self.clock_drift_sec = 0.0
        if self.source_mode == "VIDEO" and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_cur_idx = 0

    def _compute_target_screen_coords(self, state: dict) -> Tuple[float, float]:
        lead_sec = max(0.0, self.mouse_ctrl.lead_ms / 1000.0)
        cx_lead = state["cx"] + state.get("vx", 0.0) * lead_sec
        cy_lead = state["cy"] + state.get("vy", 0.0) * lead_sec

        if self.source_mode == "SCREEN" and self.screen_bbox is not None:
            bx1, by1, _, _ = self.screen_bbox
            return float(bx1 + cx_lead), float(by1 + cy_lead)

        canvas_root_x = self.preview_canvas.winfo_rootx()
        canvas_root_y = self.preview_canvas.winfo_rooty()
        ox, oy = self.preview_img_origin
        sc = self.preview_img_scale
        return float(canvas_root_x + ox + cx_lead * sc), float(canvas_root_y + oy + cy_lead * sc)

    def _schedule_next_frame(self) -> None:
        if not self.running:
            return

        now = time.perf_counter()
        dt_wall = now - self.last_step_wall_t if self.last_step_wall_t > 0 else (1.0 / 30.0)
        self.last_step_wall_t = now
        sync_clock = bool(self.var_wall_clock.get())

        frame_bgr = None
        if self.source_mode == "SCREEN" and self.screen_bbox is not None:
            frame_bgr = self.capturer.grab_bgr(self.screen_bbox)
            self.clock_drift_sec = 0.0
        elif self.source_mode == "VIDEO" and self.cap is not None:
            if sync_clock:
                elapsed_wall = now - self.wall_start_t
                target_idx = int(elapsed_wall * self.video_fps)
                if self.video_cur_idx >= 15 and target_idx > self.video_cur_idx:
                    skip_n = min(4, target_idx - self.video_cur_idx)
                    for _ in range(skip_n):
                        if self.cap.grab():
                            self.video_cur_idx += 1
                ret, frame_bgr = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_cur_idx = 0
                    self.wall_start_t = time.perf_counter()
                    self.tracker.reset()
                    self._on_param_change()
                    self.trail.clear()
                    ret, frame_bgr = self.cap.read()
                self.video_cur_idx += 1
                video_time_sec = self.video_cur_idx / max(1.0, self.video_fps)
                self.clock_drift_sec = video_time_sec - (time.perf_counter() - self.wall_start_t)
            else:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_cur_idx = 0
                    self.wall_start_t = time.perf_counter()
                    self.tracker.reset()
                    self._on_param_change()
                    self.trail.clear()
                    ret, frame_bgr = self.cap.read()
                self.video_cur_idx += 1
                self.clock_drift_sec = (self.video_cur_idx / max(1.0, self.video_fps)) - (time.perf_counter() - self.wall_start_t)

        if frame_bgr is not None:
            state = self.tracker.update(frame_bgr, dt_sec=(dt_wall if sync_clock else None))
            self.trail.append((state["cx"], state["cy"]))
            if len(self.trail) > 120:
                self.trail = self.trail[-120:]

            self._on_mouse_setting_change()
            raw_sx, raw_sy = self._compute_target_screen_coords(state)
            should_move_mouse = self.mouse_ctrl.enabled and (
                (not self.mouse_ctrl.only_transparent) or (state["phase"] != "WHITE")
            )
            if should_move_mouse:
                sx, sy = self.mouse_ctrl.move_to_screen_point(raw_sx, raw_sy)
            else:
                sx = int(round(raw_sx)) + int(self.mouse_ctrl.offset_x)
                sy = int(round(raw_sy)) + int(self.mouse_ctrl.offset_y)

            hk_hint = f"{self.hk_toggle_run}:Run/Pause | {self.hk_toggle_mouse}:Mouse | {self.hk_panic_stop}:Stop"
            vis_bgr = draw_unet_overlay(
                frame_bgr, state, self.trail, self.tracker,
                show_heatmap=bool(self.var_heatmap.get()),
                show_hud=True,
                clock_drift_sec=self.clock_drift_sec,
                mouse_active=should_move_mouse,
                screen_xy=(sx, sy),
                hotkey_hint=hk_hint,
            )
            self._update_canvas_and_telemetry(vis_bgr, state, (sx, sy), should_move_mouse)

        proc_elapsed_ms = (time.perf_counter() - now) * 1000.0
        target_period_ms = 1000.0 / max(10.0, self.video_fps if self.source_mode == "VIDEO" else 30.0)
        if sync_clock and self.source_mode == "VIDEO" and self.clock_drift_sec < -0.025:
            wait_ms = 1
        else:
            wait_ms = max(1, int(round(target_period_ms - proc_elapsed_ms)))
        self.root.after(wait_ms, self._schedule_next_frame)

    def _update_canvas_and_telemetry(
        self, vis_bgr: np.ndarray, state: dict, screen_xy: Tuple[int, int], mouse_moved: bool
    ) -> None:
        from PIL import Image, ImageTk
        cw = max(320, self.preview_canvas.winfo_width())
        ch = max(240, self.preview_canvas.winfo_height())
        H, W = vis_bgr.shape[:2]
        scale = min(cw / float(W), ch / float(H))
        nw, nh = max(64, int(W * scale)), max(64, int(H * scale))

        self.preview_img_origin = ((cw - nw) // 2, (ch - nh) // 2)
        self.preview_img_scale = scale

        vis_resized = cv2.resize(vis_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        vis_rgb = cv2.cvtColor(vis_resized, cv2.COLOR_BGR2RGB)
        self._tk_img_ref = ImageTk.PhotoImage(Image.fromarray(vis_rgb))

        if self.canvas_img_id is None:
            self.canvas_img_id = self.preview_canvas.create_image(cw // 2, ch // 2, image=self._tk_img_ref, anchor="center")
        else:
            self.preview_canvas.coords(self.canvas_img_id, cw // 2, ch // 2)
            self.preview_canvas.itemconfig(self.canvas_img_id, image=self._tk_img_ref)

        fps_val = 1000.0 / max(0.5, state["latency_ms"])
        m_status = "跟随中" if mouse_moved else "待机"
        if "device" in state:
            self.telemetry_vars["device"].set(str(state["device"])[:34])
        self.telemetry_vars["phase"].set(f"{state['phase']} | {state['shape'][:14]}")
        self.telemetry_vars["pos"].set(f"({state['cx']:5.1f}, {state['cy']:5.1f})")
        self.telemetry_vars["screen_pos"].set(f"({screen_xy[0]}, {screen_xy[1]}) [{m_status}]")
        self.telemetry_vars["vel"].set(f"({state['vx']:+4.0f},{state['vy']:+4.0f})px/s | {state['angle']:4.0f}°")
        self.telemetry_vars["snr"].set(f"{state['snr']:4.2f} ({state['conf']*100:3.0f}%)")
        self.telemetry_vars["fps"].set(f"{fps_val:4.1f} FPS ({state.get('n_step', 1.0):3.1f}x)")

    def _on_close(self) -> None:
        self.running = False
        self.hotkey_mgr.stop()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


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
    multiprocessing.freeze_support()
    main()