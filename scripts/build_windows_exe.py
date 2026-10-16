#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net 透明目标追踪系统 V3.0 — Windows 独立可执行文件 (.exe) 自动化打包脚本
支持在 GitHub Actions (windows-latest) 或本地 Windows 10/11 环境下直接运行:
    python scripts/build_windows_exe.py

打包产物 (位于 dist/UNetTracker_V3.0_Portable/ 与 dist/UNetTracker_V3.0_Windows_x64_Portable.zip):
1. UNetTracker_V3.0.exe            : 纯净桌面 GUI 版 (--noconsole，双击直接弹出可视化操作台，无黑框)
2. UNetTracker_V3.0_Server_CLI.exe : 命令行与远程算力服务端版 (--console，支持 --server 启动 8765 端口远程算力或 --video 批量评测)
"""

import os
import shutil
import subprocess
import sys
import zipfile


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    entry_script = os.path.join(repo_root, "unet_universal_tracker_v2.py")
    if not os.path.exists(entry_script):
        entry_script = os.path.join(repo_root, "main.py")

    dist_dir = os.path.join(repo_root, "dist")
    build_dir = os.path.join(repo_root, "build")
    portable_dir = os.path.join(dist_dir, "UNetTracker_V3.0_Portable")

    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(portable_dir, exist_ok=True)

    common_args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--distpath", portable_dir,
        "--workpath", build_dir,
        "--specpath", build_dir,
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=PIL.ImageGrab",
        "--hidden-import=mss",
        "--hidden-import=mss.windows",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=IPython",
        "--exclude-module=notebook",
    ]

    print("==================================================================")
    print("[1/2] 正在编译纯净桌面 GUI 版: UNetTracker_V3.0.exe (--noconsole)...")
    print("==================================================================")
    cmd_gui = common_args + [
        "--noconsole",
        "--name", "UNetTracker_V3.0",
        entry_script,
    ]
    subprocess.check_call(cmd_gui)

    print("==================================================================")
    print("[2/2] 正在编译远程算力服务端 & CLI 版: UNetTracker_V3.0_Server_CLI.exe...")
    print("==================================================================")
    cmd_cli = common_args + [
        "--console",
        "--name", "UNetTracker_V3.0_Server_CLI",
        entry_script,
    ]
    subprocess.check_call(cmd_cli)

    # Copy configuration, README, client SDK, and convenient .bat launchers into portable_dir
    os.makedirs(os.path.join(portable_dir, "config"), exist_ok=True)
    for rel_src, rel_dst in [
        ("README.md", "README.md"),
        ("remote_tracker_client.py", "remote_tracker_client.py"),
        ("config/default_config.json", "config/default_config.json"),
        ("config/default_config.json", "tracker_hotkey_config.json"),
    ]:
        s_path = os.path.join(repo_root, rel_src)
        if os.path.exists(s_path):
            shutil.copy2(s_path, os.path.join(portable_dir, rel_dst))

    with open(os.path.join(portable_dir, "1_启动GUI可视化操作台.bat"), "w", encoding="utf-8") as f:
        f.write("@echo off\nchcp 65001 >nul\nstart \"\" \"%~dp0UNetTracker_V3.0.exe\" --gui\n")

    with open(os.path.join(portable_dir, "2_启动远程算力服务端(8765端口).bat"), "w", encoding="utf-8") as f:
        f.write(
            "@echo off\nchcp 65001 >nul\n"
            "echo 正在启动 U-Net V3.0 远程算力服务端 (监听端口 8765)...\n"
            "\"%~dp0UNetTracker_V3.0_Server_CLI.exe\" --server --host 0.0.0.0 --port 8765 --device auto --mode ultra\n"
            "pause\n"
        )

    # Create portable zip archive
    zip_out = os.path.join(dist_dir, "UNetTracker_V3.0_Windows_x64_Portable.zip")
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(portable_dir):
            for fn in sorted(files):
                abs_p = os.path.join(root, fn)
                rel_p = os.path.relpath(abs_p, dist_dir)
                zf.write(abs_p, arcname=rel_p)

    print("==================================================================")
    print(f"打包完成! 免安装绿色版压缩包: {zip_out}")
    print("==================================================================")


if __name__ == "__main__":
    main()
