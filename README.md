# elf2_yolo
基于elf2开发板的低光增强与物体检测项目 

资料网址：https://www.elfboard.com/information/detail.html?id=7

下载完全部资料或者按需下载

# 环境准备

## 开发板镜像烧录

打开**01-教程文档**文件夹，打开**ELF2开发板快速启动手册**根据第五章的内容将**04-镜像**中的desktop镜像烧录进开发板；

## 串口通信与文件传输

打开**ELF2开发板快速启动手册**根据第二章的内容进行串口通信和使用SFTP进行文件传输；

## 虚拟机环境搭建

打开**08-开发环境**解压所有压缩包，所有同名文件是一样的，选择内容不为空的合并就是有效的虚拟环境(每个文件夹中都有部分无效，全部整合可以得到完整的)，使用VMware打开虚拟环境即可；

## Pycharm连接ubuntu实现同步与调试

参考CSDN的帖子：https://blog.csdn.net/weixin_72965172/article/details/134448207 ；

## 安装RKNN-Toolkit-Lite2

打开**01-教程文档**，打开**AI开发教程**，打开**基于 RK3588 的AI模型训练到部署**，根据第二章的**2.2.3**在开发板上安装RKNN-Toolkit-Lite2；

# python环境配置

## pyqt安装

```
sudo apt update
sudo apt install python3-pyqt5        # Qt5 运行时 + 绑定
sudo apt install pyqt5-dev-tools      # (可选) pyuic5, pyrcc5, lrelease
```

## GStreamer introspection 包和必要的插件

```
sudo apt update
sudo apt install python3-gi python3-gi-cairo        # PyGObject 运行时
sudo apt install gir1.2-gstreamer-1.0               # Gst 主库 introspection
sudo apt install gir1.2-gst-plugins-base-1.0        # 含 GstVideo typelib
sudo apt install gstreamer1.0-plugins-base \
                 gstreamer1.0-plugins-good \
                 gstreamer1.0-tools                # 常用元素和 gst-launch
```

