#!/bin/bash

# 设置 Qt 插件路径（如你的程序中用到）
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

# 检查 DISPLAY=:0 是否可用
if DISPLAY=:0 xset q > /dev/null 2>&1; then
    export DISPLAY=:0
    export XAUTHORITY="/run/user/1000/gdm/Xauthority"
elif DISPLAY=:1 xset q > /dev/null 2>&1; then
    export DISPLA=:1
    export XAUTHORITY="/run/user/1000/gdm/Xauthority"
else
    echo "No valid DISPLAY found"
    exit 1
fi

cd /home/elf/pythonProject
/home/elf/pythonProject/bin/python /home/elf/pythonProject/main.py
