#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 桌面可视化 GUI 主窗口与全屏半透明区域框选组件。
"""

import ctypes
import json
import os
import platform
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .capture import ScreenRegionCapturer
from .hotkeys import SystemGlobalHotkeyManager
from .mouse_controller import ScreenMouseController
from .tracker import UniversalUNetTracker
from .visualizer import draw_unet_overlay


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
    """
    Windows 实时区域截图 & 视频回放追踪 GUI 主窗口 (V2.3 全局热键 + 鼠标自动跟随中心点版)
    """

    CONFIG_FILE = os.path.join(os.getcwd(), "tracker_hotkey_config.json")

    def __init__(self, initial_video: Optional[str] = None):
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
        self.root.title("U-Net 全图形自适应透明目标实时追踪系统 V2.3 (全局热键 + 鼠标自动跟随中心版)")
        self.root.geometry("1280x830")
        self.root.configure(bg="#141820")

        self.tracker = UniversalUNetTracker(fps=30.0, perf_mode="turbo")
        self.capturer = ScreenRegionCapturer()
        self.mouse_ctrl = ScreenMouseController()
        self.hotkey_mgr = SystemGlobalHotkeyManager()
        self.trail: List[Tuple[float, float]] = []

        self.hk_toggle_run = "F8"
        self.hk_toggle_mouse = "F9"
        self.hk_panic_stop = "F10"
        self._load_user_config()

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
        except Exception:
            pass

    def _save_user_config(self) -> None:
        try:
            cfg = {
                "hk_toggle_run": self.hk_toggle_run,
                "hk_toggle_mouse": self.hk_toggle_mouse,
                "hk_panic_stop": self.hk_panic_stop,
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
            right_frame, text="⚡ U-Net 实时追踪 & 鼠标锁定控制面板",
            bg="#1C222D", fg="#00E5FF", font=("Microsoft YaHei", 12, "bold")
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
            right_frame, text="⏱️ 时钟同步与硬件加速 (i5-9400F + RX 550X)",
            bg="#1C222D", fg="#00FFCC", font=("Microsoft YaHei", 9, "bold"), padx=8, pady=3
        )
        clock_box.pack(fill="x", padx=12, pady=3)

        self.var_wall_clock = tk.BooleanVar(value=True)
        tk.Checkbutton(
            clock_box, text="严格同步真实时钟 (跟上时钟零延迟)",
            variable=self.var_wall_clock, bg="#1C222D", fg="#81ECEC",
            selectcolor="#141820", activebackground="#1C222D", font=("Microsoft YaHei", 8, "bold")
        ).pack(anchor="w")

        self.var_perf_mode = tk.StringVar(value="turbo")
        perf_row = tk.Frame(clock_box, bg="#1C222D")
        perf_row.pack(fill="x", pady=1)
        tk.Label(perf_row, text="计算档位:", bg="#1C222D", fg="#DFE6E9", font=("Microsoft YaHei", 8)).pack(side="left")
        ttk.Radiobutton(
            perf_row, text="极速档 (60 FPS)", variable=self.var_perf_mode,
            value="turbo", command=self._on_param_change
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            perf_row, text="高清档 (35 FPS)", variable=self.var_perf_mode,
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

        self.slider_ema = self._add_slider(param_box, "EMA 平滑系数 (α):", 0.20, 0.85, 0.54, 0.02)
        self.slider_gate = self._add_slider(param_box, "半分辨率门控半径 (px):", 28.0, 68.0, 44.0, 2.0)
        self.slider_step = self._add_slider(param_box, "单帧基准最大步长 (px):", 12.0, 40.0, 22.0, 1.0)

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
