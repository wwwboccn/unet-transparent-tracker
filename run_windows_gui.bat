@echo off
chcp 65001 >nul
title U-Net Universal Transparent Object Tracker V2.3
echo ============================================================================
echo   启动 U-Net 全图形自适应透明目标实时追踪系统 V2.3 (Windows GUI)
echo   默认全局热键: [F8] 启动/暂停追踪  ^|  [F9] 开关鼠标跟随  ^|  [F10] 紧急释放
echo ============================================================================
python main.py --gui
pause
