@echo off
chcp 65001 >nul
echo ============================================================================
echo 正在安装依赖并使用 PyInstaller 打包 Windows 免安装可执行文件 (.exe)...
echo ============================================================================
python -m pip install -r requirements.txt pyinstaller
python scripts/build_windows_exe.py
echo.
echo 打包完成！请查看 dist\UNetTracker_V3.7_Portable 文件夹。
pause
