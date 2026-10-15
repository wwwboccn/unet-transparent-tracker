#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目根目录统一启动入口：
- 启动 Windows 图形界面: python main.py --gui
- 处理本地视频并导出结果: python main.py --video input.mp4 --output tracked.mp4
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from unet_tracker.cli import main

if __name__ == "__main__":
    main()
