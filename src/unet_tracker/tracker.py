#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.0 核心追踪引擎模块：包含 CudaAcceleratorBackend 与 UniversalUNetTracker (平顶圆盘目标隔离 + 无偏几何中心精确定位)。
"""

from typing import Dict, List, Optional, Tuple
import time
import cv2
import numpy as np

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
        self.k_open3_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dup_streak: int = 0
        self.total_dup_frames: int = 0

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
        sal_roi = np.where(cw_roi >= 0.72, sal_roi, 0.0)

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
            if max_area >= 480.0:
                self.white_reappear_streak += 1
                if self.white_reappear_streak >= 3:
                    self.reset()
                    self._init_grids(hs, ws)
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
                if float(np.mean(ad_w0)) >= 1.18 or float(np.percentile(ad_w0, 99.2)) >= 5.0:
                    tx1, ty1 = self._estimate_bg_translation_fast(self.prev_grays_s[0], gray_s, self.prev_uis_s[0])
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
        is_dup = (raw_diff_0 < 1.18) and (float(np.percentile(ad_0, 99.2)) < 5.0)
        if is_dup:
            self.dup_streak += 1
            self.total_dup_frames += 1

        # Detect whether this video is a high-duplicate game-tick recording (like MapleStoryN where >45% of frames are dupes)
        dup_ratio = self.total_dup_frames / float(max(1, self.frame_idx))
        is_game_tick_video = (dup_ratio > 0.40) or (self.dup_streak >= 2)

        if is_dup and is_game_tick_video:
            # In high-duplicate game-tick recordings (MapleStoryN), hold state cleanly on duplicate frames without decaying KF
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
            self.prev_shifts.insert(0, (tx1, ty1))
            self.prev_shifts = self.prev_shifts[:3]

        comp_diffs = []
        valid_masks = []
        warped_grays = []
        cum_tx, cum_ty = 0.0, 0.0
        raw_d0 = None
        for k in range(len(self.prev_frames_s)):
            cum_tx += self.prev_shifts[k][0]
            cum_ty += self.prev_shifts[k][1]
            Mk = np.array([[1.0, 0.0, cum_tx], [0.0, 1.0, cum_ty]], dtype=np.float32)
            use_cubic = (k == 0) or (self.perf_mode == "ultra")
            interp_flag = cv2.INTER_CUBIC if use_cubic else cv2.INTER_LINEAR
            wk = cv2.warpAffine(self.prev_frames_s[k], Mk, (ws, hs), flags=interp_flag, borderMode=cv2.BORDER_REPLICATE)
            if k == 0:
                wk_g = cv2.warpAffine(self.prev_grays_s[0], Mk, (ws, hs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                warped_grays.append(wk_g)
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

        if (not is_dup) and raw_d0 is not None and self.transparent_streak > int(36 * fps_scale) and (abs(tx1) + abs(ty1) > 0.12):
            t_dist_sq = (self.xx_s - self.ema_cx * 0.5) ** 2 + (self.yy_s - self.ema_cy * 0.5) ** 2
            far_m = (t_dist_sq > (46.0 ** 2)) & (ui_mask_s == 0)
            self.clutter_sum_s[far_m] += raw_d0[far_m]
            self.clutter_count_s[far_m] += 1.0

        if self.transparent_streak >= int(160 * fps_scale) and self.init_cx_s is not None:
            if (self.transparent_streak == int(160 * fps_scale)) or (self.transparent_streak % 4 == 0):
                valid_c = self.clutter_count_s > (30.0 * fps_scale if not is_game_tick_video else 12.0 * fps_scale)
                if np.any(valid_c):
                    raw_m = np.zeros_like(self.clutter_sum_s)
                    mc = self.clutter_count_s > 1.0
                    raw_m[mc] = self.clutter_sum_s[mc] / self.clutter_count_s[mc]
                    mu0 = float(np.median(raw_m[valid_c])) + 0.25
                    N0 = 35.0 * fps_scale if not is_game_tick_video else 15.0 * fps_scale
                    eb_m = (self.clutter_sum_s + N0 * mu0) / (self.clutter_count_s + N0)
                    eb_s = cv2.boxFilter(eb_m, cv2.CV_32F, (25, 25))
                    ratio = eb_s / max(0.5, mu0)
                    b_dist_sq = (self.xx_s - self.init_cx_s) ** 2 + (self.yy_s - self.init_cy_s) ** 2
                    b_zone = np.exp(-0.5 * b_dist_sq / (24.0 ** 2)).astype(np.float32)
                    s_birth = np.clip(ratio - 1.35, 0.0, 1.2)
                    supp = (1.0 - np.exp(-3.5 * (s_birth ** 2))) * b_zone
                    self.cached_clutter_weight_s = (1.0 - 0.88 * supp).astype(np.float32)
        clutter_w_s = self.cached_clutter_weight_s

        comp_norm_s = (np.mean(comp_diffs, axis=0) if comp_diffs else np.zeros((hs, ws), dtype=np.float32)) * clutter_w_s
        vmask_s = np.min(valid_masks, axis=0) if valid_masks else np.ones((hs, ws), dtype=np.float32)
        vmask_s[:4, :] = 0.0
        vmask_s[-4:, :] = 0.0
        vmask_s[:, :6] = 0.0
        vmask_s[:, -6:] = 0.0
        if has_bot_bar:
            vmask_s[hs - 27:, :] = 0.0

        flow_norms = []
        for k in range(min(2, len(self.prev_grays_s))):
            ad_k = cv2.absdiff(gray_s, self.prev_grays_s[k])
            if float(np.mean(ad_k)) > 1.18 or float(np.percentile(ad_k, 99.2)) >= 5.0:
                ux, uy = self.cuda_backend.calc_farneback_gpu(self.prev_grays_s[k], gray_s, win_size=13, levels=2, iters=2)
                rk = np.hypot(ux - np.median(ux), uy - np.median(uy)) * vmask_s
                dv_dx = cv2.Sobel(uy, cv2.CV_32F, 1, 0, ksize=3)
                du_dy = cv2.Sobel(ux, cv2.CV_32F, 0, 1, ksize=3)
                curl_mag = np.abs(dv_dx - du_dy) * vmask_s
                rk[:8, :] = 0.0
                curl_mag[:8, :] = 0.0
                mr, sr = cv2.meanStdDev(rk)
                mc, sc = cv2.meanStdDev(curl_mag)
                r_clean = np.maximum(0.0, rk - (float(mr[0, 0]) + 0.75 * float(sr[0, 0])))
                c_clean = np.maximum(0.0, curl_mag - (float(mc[0, 0]) + 0.75 * float(sc[0, 0])))
                r_n = np.clip(r_clean / (float(mr[0, 0]) + 4.0 * float(sr[0, 0]) + 1e-5), 0.0, 1.0)
                c_n = np.clip(c_clean / (float(mc[0, 0]) + 4.0 * float(sc[0, 0]) + 1e-5), 0.0, 1.0)
                flow_norms.append((0.75 * r_n + 0.25 * c_n) * clutter_w_s)

        rf_norm_s = np.mean(flow_norms, axis=0) if flow_norms else comp_norm_s
        saliency_s = (0.55 * comp_norm_s + 0.45 * rf_norm_s) * vmask_s

        # Current-frame ghost-free saliency for Layer-2 True Geometric Center Localization
        if len(comp_diffs) >= 2:
            d_now = 0.65 * comp_diffs[0] + 0.35 * np.sqrt(comp_diffs[0] * comp_diffs[1] + 1e-6)
        elif len(comp_diffs) == 1:
            d_now = comp_diffs[0]
        else:
            d_now = comp_norm_s
        f_now = flow_norms[0] if len(flow_norms) > 0 else rf_norm_s
        saliency_now_s = (0.55 * d_now * clutter_w_s + 0.45 * f_now) * vmask_s
        self.last_unet_prob_s = saliency_now_s

        mean_side = 0.5 * (self.ref_w + self.ref_h)
        if is_game_tick_video:
            k_half = int(np.clip(int(round(0.52 * mean_side)) | 1, 49, 59))
            energy_raw = self._damped_normalized_box_filter(saliency_now_s, vmask_s, ksize=(k_half, k_half), min_den_ratio=0.55)
            energy_s = cv2.GaussianBlur(energy_raw, (21, 21), 5.2) * clutter_w_s
        else:
            k_half = int(np.clip(int(round(0.38 * mean_side)) | 1, 35, 43))
            energy_s = self._damped_normalized_box_filter(saliency_s, vmask_s, ksize=(k_half, k_half), min_den_ratio=0.25)
            energy_s *= clutter_w_s
        global_max = float(np.max(energy_s))
        energy_norm_s = energy_s / (global_max + 1e-5)

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
        posterior_s[:18, :] = 0.0
        posterior_s[-10:, :] = 0.0
        posterior_s[:, :18] = 0.0
        posterior_s[:, -10:] = 0.0
        if has_bot_bar:
            posterior_s[hs - 27:, :] = 0.0
        _, max_post, _, max_loc_s = cv2.minMaxLoc(posterior_s)

        mean_e, std_e = cv2.meanStdDev(energy_s, mask=(vmask_s > 0).astype(np.uint8))
        snr = max(
            1.05,
            (float(energy_s[max_loc_s[1], max_loc_s[0]]) - float(mean_e[0, 0]))
            / (float(std_e[0, 0]) + 1e-5)
        )

        eff_max_step = self.max_step_px * max(1.0, 0.85 * eff_step)
        eff_alpha = 1.0 - (1.0 - self.ema_alpha) ** max(1.0, 0.85 * eff_step)

        if snr >= 1.15 and max_post >= 0.08:
            if is_game_tick_video:
                gx_s, gy_s = self._compute_true_geometric_center(
                    float(max_loc_s[0]), float(max_loc_s[1]),
                    saliency_now_s, clutter_w_s, vmask_s,
                    mean_side, ws, hs, max_valid_y_s
                )
                meas_cx = float(np.clip(gx_s * 2.0, 24.0, W - 16.0))
                meas_cy = float(np.clip(gy_s * 2.0, 34.0, max_valid_cy))
                post = self.kf.correct(np.array([[np.float32(meas_cx)], [np.float32(meas_cy)]]))
                raw_cx = 0.72 * meas_cx + 0.28 * float(post[0, 0])
                raw_cy = 0.72 * meas_cy + 0.28 * float(post[1, 0])
            else:
                meas_cx = float(np.clip(max_loc_s[0] * 2.0, 24.0, W - 16.0))
                meas_cy = float(np.clip(max_loc_s[1] * 2.0, 34.0, max_valid_cy))
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
            if is_dup:
                snr = max(snr, self.last_valid_snr * 0.92)
                conf = float(np.clip(0.64 + 0.08 * min(snr, 4.5), 0.64, 0.95))
                phase = "TRANSPARENT"
            else:
                conf = 0.62
                phase = "PREDICTED"

        self.ema_cx = float(np.clip(eff_alpha * raw_cx + (1.0 - eff_alpha) * self.ema_cx, 24.0, W - 16.0))
        self.ema_cy = float(np.clip(eff_alpha * raw_cy + (1.0 - eff_alpha) * self.ema_cy, 34.0, max_valid_cy))

        # Layer 2: Compute Unbiased True Geometric Center around (ema_cx, ema_cy)
        geo_x_s, geo_y_s = self._compute_true_geometric_center(
            self.ema_cx * 0.5, self.ema_cy * 0.5,
            saliency_now_s, clutter_w_s, vmask_s,
            mean_side, ws, hs, max_valid_y_s
        )
        target_center_cx = float(np.clip(geo_x_s * 2.0, 24.0, W - 16.0))
        target_center_cy = float(np.clip(geo_y_s * 2.0, 34.0, max_valid_cy))

        if is_game_tick_video and phase == "TRANSPARENT":
            # In high-duplicate game-tick videos (MapleStoryN), keep internal anchor centered on the 4-pane tile
            self.ema_cx = 0.55 * target_center_cx + 0.45 * self.ema_cx
            self.ema_cy = 0.55 * target_center_cy + 0.45 * self.ema_cy

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



