import os
import time
import threading
import queue
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtWidgets, QtGui, QtCore
from rknnlite.api import RKNNLite
from camera_module.gst_camera import GSTCamera
from yolo_module.postprocess import YOLOPostProcessor
import cv2
import threading

# ------------------ 环境变量 ------------------
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
os.environ["DISPLAY"] = ":1"
os.environ["XAUTHORITY"] = "/run/user/1000/gdm/Xauthority"

# ------------------ 配置 ------------------
DEVICE = "/dev/video11"
IMG_SIZE = (640, 640)
GSTCamera_SIZE = (640, 640)
ANCHOR_PATH = "yolo_module/anchors_yolov5.txt"

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

YOLO_MODEL = "onnx_or_rknn/yolov5.rknn"

CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def hisEqulColor2(imga):
    ycrcb = cv2.cvtColor(imga, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))  # 将 tuple 转为 list
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
    channels[0] = clahe.apply(channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def enhance_image(img):
    img_bil = cv2.bilateralFilter(img, 5, 15, 15)
    img_equ = hisEqulColor2(img_bil)
    hsv_image = cv2.cvtColor(img_equ, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = gamma_trans(hsv_image[:, :, 2], 0.55)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

class RKNNStreamApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RKNN 实时检测")

        # 创建两个 QLabel，分别用于显示原始图像和增强后的图像
        self.raw_label = QtWidgets.QLabel(self)
        self.raw_label.setFixedSize(IMG_SIZE[0], IMG_SIZE[1])
        self.raw_label.setAlignment(QtCore.Qt.AlignCenter)

        self.enh_label = QtWidgets.QLabel(self)
        self.enh_label.setFixedSize(IMG_SIZE[0], IMG_SIZE[1])
        self.enh_label.setAlignment(QtCore.Qt.AlignCenter)

        # 创建一个 QLabel 用于显示流水段信息
        self.process_label = QtWidgets.QLabel(self)
        self.process_label.setFixedSize(IMG_SIZE[0], 30)
        self.process_label.setAlignment(QtCore.Qt.AlignCenter)

        # 使用 QHBoxLayout 将两个 QLabel 放在一行
        # 使用 QVBoxLayout 将两个 QLabel 和流水段 QLabel 垂直排列
        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.raw_label)
        hlayout.addWidget(self.enh_label)
        layout.addLayout(hlayout)
        layout.addWidget(self.process_label)
        self.setLayout(layout)

        self.cam = GSTCamera(device=DEVICE, width=GSTCamera_SIZE[0], height=GSTCamera_SIZE[1])

        self.rknn_yolo = RKNNLite()
        self.rknn_yolo.load_rknn(YOLO_MODEL)
        self.rknn_yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        self.processor = YOLOPostProcessor(ANCHOR_PATH, CLASSES, OBJ_THRESH, NMS_THRESH, IMG_SIZE)

        self.frame_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.raw_frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)


        self.frame_count = 0
        self.none_frame_count = 0
        self.start_time = time.time()

        self.raw_window = RawFrameWindow()

        self.worker_thread = threading.Thread(target=self.cnn5_and_sci_worker, daemon=True)
        self.worker_thread.start()

        self.visualization_thread = threading.Thread(target=self.visualization_worker, daemon=True)
        self.visualization_thread.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1)


    def cnn5_and_sci_worker(self):
        while True:
            frame = self.cam.read()
            if frame is None:
                self.none_frame_count += 1
                print(f"Warning: Received None frame. Total none frames: {self.none_frame_count}")
                continue  # 继续读取下一个帧

            timestamp = time.time()  # 创建时间戳
            enhanced = enhance_image(frame)
            # out_t =  time.time()
            # print(f"低光：{(out_t-timestamp)*1000:.2f}ms")

            try:
                self.output_queue.put_nowait((enhanced, timestamp))
                # print("Warning: output_queue is full. Discarding frame.")
            except queue.Full:
                continue

            # 将原始图像放入共享队列
            try:
                self.raw_frame_queue.put_nowait(frame)
            except queue.Full:
                # print("Warning: raw_frame_queue is full. Discarding frame.")
                pass



    def update(self):
        try:
            enh_img, timestamp= self.output_queue.get_nowait()
        except queue.Empty:
            print("Warning: output_queue is empty. Skipping update.")
            return
        # get_time = time.time()
        enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)
        tensor = enh_img.astype(np.uint8).transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.rknn_yolo.inference(inputs=[tensor], data_format='nchw')
        # out_time = time.time()
        # print(f"yolo:{(out_time - get_time) * 1000:.2f}ms")
        try:
            self.detection_queue.put_nowait((outputs, enh_img))
        except queue.Full:
            pass


        self.frame_count += 1

        # 更新原始图像显示
        self.update_raw_frame()
        self.update_enhanced_frame()

        current_time = time.time()
        elapsed = current_time - self.start_time
        latency = current_time - timestamp
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.setWindowTitle(f"RKNN 检测 | FPS: {fps:.2f} | 丢帧: {self.none_frame_count} | 延迟: {latency:.2f}秒")
            self.frame_count = 0
            self.start_time = current_time

    def visualization_worker(self):
        while True:
            try:
                outputs, enh_img = self.detection_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            # o_time = time.time()
            detection_outputs = [np.squeeze(out, axis=0) for out in outputs[:3]]
            boxes, classes, scores = self.processor(detection_outputs)
            display_img = self.processor.draw(enh_img, boxes, scores, classes)
            # draw_time = time.time()
            # print(f"draw:{(draw_time - o_time) * 1000:.2f}ms")
            # 把绘制好的图像送回主线程
            try:
                self.display_queue.put_nowait(display_img)
            except queue.Full:
                pass

    def update_enhanced_frame(self):
        try:
            display_img = self.display_queue.get_nowait()
        except queue.Empty:
            # print("Warning: display_queue is empty. Skipping enhanced frame update.")
            return

        if display_img is None:
            print("Warning: display_img is None. Skipping.")
            return

        # 将绘制后的增强图像转换为 RGB 格式并更新 QLabel
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(display_img.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.enh_label.setPixmap(pixmap)

    def update_raw_frame(self):
        try:
            raw_img = self.raw_frame_queue.get_nowait()
        except queue.Empty:
            print("Warning: raw_frame_queue is empty. Skipping raw frame update.")
            return

        # 如果队列为空，尝试下一次更新
        if raw_img is None:
            print("Warning: raw_img is None. Skipping.")
            return

        # 将原始图像转换为RGB格式并更新显示
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        h, w, ch = raw_img.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(raw_img.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.raw_label.setPixmap(pixmap)


    def closeEvent(self, event):
        self.frame_queue.put(None)
        self.cam.stop()
        self.rknn_cnn5.release()
        self.rknn_sci.release()
        self.rknn_yolo.release()
        event.accept()


class RawFrameWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("原始图像")
        self.resize(GSTCamera_SIZE[0], GSTCamera_SIZE[1])

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = RKNNStreamApp()
    window.show()
    app.exec_()
