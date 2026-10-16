@echo off
chcp 65001 >nul
python unet_universal_tracker_v2.py --server --host 0.0.0.0 --port 8765 --device cuda --mode ultra
pause
