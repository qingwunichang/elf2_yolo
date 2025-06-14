import os
os.environ["DISPLAY"] = ":0"
os.environ["XAUTHORITY"] = "/run/user/1000/gdm/Xauthority"   # 若已能正常连接 :0，可删掉

import sys
import time
import numpy as np
from PIL import Image, ImageDraw

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui  import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout

from camera_module.gst_camera import GSTCamera  # 你的摄像头类


IMG_W, IMG_H = 640, 480          # 根据实际分辨率调整


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """把 Pillow Image 转成 Qt 原生 QImage（RGB888）"""
    rgb_img = pil_img.convert("RGB")
    w, h = rgb_img.size
    data = rgb_img.tobytes("raw", "RGB")
    return QImage(data, w, h, 3 * w, QImage.Format_RGB888)


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RKNN 检测 | 初始化…")
        self.resize(IMG_W, IMG_H + 40)

        # 显示区域
        self.label = QLabel(alignment=Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        # 摄像头
        self.cam = GSTCamera("/dev/video11", IMG_W, IMG_H)

        # 计时
        self.frame_cnt = 0
        self.drop_cnt  = 0
        self.t0 = time.time()

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)           # ≈33 FPS

    # ------------------------------------------------------------------
    def draw_boxes(self, frame: np.ndarray) -> Image.Image:
        """示例：在 numpy.uint8 BGR/RGB 图像上画 1 个假框"""
        if frame.ndim == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame[:, :, ::-1])   # 若 frame 是 BGR 改成 RGB
        else:
            img = Image.fromarray(frame)

        draw = ImageDraw.Draw(img)
        box   = (100, 80, 220, 200)
        label = "person 0.95"

        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 12), label, fill="red")
        return img
    # ------------------------------------------------------------------

    def update_frame(self):
        frame = self.cam.read()
        if frame is None:
            self.drop_cnt += 1
            return

        img_pil = self.draw_boxes(frame)
        qimg = pil_to_qimage(img_pil)
        self.label.setPixmap(QPixmap.fromImage(qimg))

        # FPS 统计
        self.frame_cnt += 1
        elapsed = time.time() - self.t0
        if elapsed >= 1.0:
            fps = self.frame_cnt / elapsed
            self.setWindowTitle(f"RKNN 检测 | FPS: {fps:.2f} | 丢帧: {self.drop_cnt}")
            self.frame_cnt = 0
            self.drop_cnt  = 0
            self.t0 = time.time()

    def closeEvent(self, event):
        self.cam.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w   = CameraWidget()
    w.show()
    sys.exit(app.exec_())
