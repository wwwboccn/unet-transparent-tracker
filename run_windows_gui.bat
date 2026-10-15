@echo off
chcp 65001 >nul
python unet_universal_tracker_v2.py --gui --device cuda --gpu-id 0
pause
