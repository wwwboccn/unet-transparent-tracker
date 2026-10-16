#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 全局快捷键监听与按键录制模块。
"""

import ctypes
import platform
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

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


