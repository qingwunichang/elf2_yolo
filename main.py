import time
import queue
from PyQt5 import QtWidgets, QtGui, QtCore
from camera_module.gst_camera import GSTCamera
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from yolo_module.postprocess import YOLOPostProcessor
import subprocess

app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------ 环境变量 ------------------
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
os.environ["XAUTHORITY"] = "/run/user/1000/gdm/Xauthority"
# os.environ["DISPLAY"] = ":0"

# ------------------ 配置 ------------------
DEVICE = "/dev/video11"
IMG_SIZE = (640, 640)
GSTCamera_SIZE = (640, 640)
ANCHOR_PATH = "yolo_module/anchors_yolov5.txt"

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

YOLO_MODEL = "onnx_or_rknn/yolov5.rknn"
IMGFUSION_MODEL = "onnx_or_rknn/alpha_predictor.rknn"

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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels[0] = clahe.apply(channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def enhance_image(img):
    img_bil = cv2.bilateralFilter(img, 5, 15, 15)
    img_equ = hisEqulColor2(img_bil)
    hsv_image = cv2.cvtColor(img_equ, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = gamma_trans(hsv_image[:, :, 2], 0.6)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

class RKNNStreamApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时检测")
        # 创建一个 QLabel 用于显示作品名称和图标
        self.title_label = QtWidgets.QLabel(self)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)

        # 创建两个 QLabel，分别用于显示原始图像和增强后的图像
        self.raw_label = QtWidgets.QLabel(self)
        self.raw_label.setFixedSize(IMG_SIZE[0], IMG_SIZE[1])
        self.raw_label.setAlignment(QtCore.Qt.AlignCenter)

        self.enh_label = QtWidgets.QLabel(self)
        self.enh_label.setFixedSize(IMG_SIZE[0], IMG_SIZE[1])
        self.enh_label.setAlignment(QtCore.Qt.AlignCenter)

        # 创建一个 QLabel 显示作品名称
        self.title_label = QtWidgets.QLabel(self)
        self.title_label.setText("基于ELF 2的低光条件下的目标检测系统")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setMinimumHeight(30)
        self.title_label.setContentsMargins(0, 0, 0, 0)
        self.title_label.setStyleSheet("font-size: 60px; color: #003366; font-weight: bold;")

        # 创建一个 QLabel 用于显示流水段信息
        self.process_label = QtWidgets.QLabel(self)
        self.process_label.setFixedSize(IMG_SIZE[0], 30)
        self.process_label.setAlignment(QtCore.Qt.AlignCenter)

        # 使用 QHBoxLayout 将两个图像 QLabel 放在一行
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.raw_label)
        hlayout.addWidget(self.enh_label)

        # 垂直布局：标题 -> 流水段信息 -> 图像区域
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.process_label)
        layout.addLayout(hlayout)

        self.setLayout(layout)
        self.showMaximized()

        self.cam = GSTCamera(device=DEVICE, width=GSTCamera_SIZE[0], height=GSTCamera_SIZE[1])

        self.rknn_yolo = RKNNLite()
        self.rknn_yolo.load_rknn(YOLO_MODEL)
        self.rknn_yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        self.rknn_imgfusion = RKNNLite()
        self.rknn_imgfusion.load_rknn(IMGFUSION_MODEL)
        self.rknn_imgfusion.init_runtime(core_mask=RKNNLite.NPU_CORE_1)

        self.processor = YOLOPostProcessor(ANCHOR_PATH, CLASSES, OBJ_THRESH, NMS_THRESH, IMG_SIZE)

        self.frame_queue = queue.Queue(maxsize=1)
        self.enhancement_queue = queue.Queue(maxsize=1)
        self.imgfusion_queue = queue.Queue(maxsize=1)
        self.yolo_queue = queue.Queue(maxsize=1)
        self.draw_queue = queue.Queue(maxsize=1)


        self.frame_count = 0
        self.total_latency = 0.0
        self.latency_count = 0
        self.start_time = time.time()

        self.raw_window = RawFrameWindow()

        self.img_enhancement_thread = threading.Thread(target=self.img_enhancement_worker, daemon=True)
        self.img_enhancement_thread.start()

        self.imgfusion_thread = threading.Thread(target=self.imgfusion_worker, daemon=True)
        self.imgfusion_thread.start()

        self.draw_thread = threading.Thread(target=self.draw_worker, daemon=True)
        self.draw_thread.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.yolo)
        self.timer.start(1)

    def img_enhancement_worker(self):
        while True:
            timestamp = time.time()  # 创建时间戳
            frame = self.cam.read()
            if frame is None:
                # print(f"Warning: Received None frame.")
                continue  # 继续读取下一个帧

            enhanced = enhance_image(frame)

            # 将原始图像和增强图像放入共享队列
            try:
                self.enhancement_queue.put_nowait((frame, enhanced, timestamp))
                # print("Warning: enhancement_queue is full. Discarding frame.")
            except queue.Full:
                continue

            # out_t = time.time()
            # print(f"低光增强：{(out_t - timestamp) * 1000:.2f}ms")

    def imgfusion_worker(self):
        while True:
            # start_time = time.time()
            try:
                origin, enhancement, timestamp = self.enhancement_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
            enhancement = cv2.cvtColor(enhancement, cv2.COLOR_BGR2RGB)

            origin_tensor = origin.transpose(2, 0, 1)
            origin_tensor = np.expand_dims(origin_tensor, 0)
            origin_tensor = origin_tensor.astype(np.uint8)

            outputs = self.rknn_imgfusion.inference(inputs=[origin_tensor], data_format='nchw')
            alpha = outputs[0].squeeze(0)

            fusionimg = origin * alpha + enhancement * (1 - alpha)
            fusionimg = np.clip(fusionimg, 0, 255).astype(np.uint8)

            # 将融合图像放入共享队列
            try:
                self.imgfusion_queue.put_nowait((origin, fusionimg, timestamp))
            except queue.Full:
                pass
            # end_time = time.time()
            # print(f"图像融合:{(end_time - start_time) * 1000:.2f}ms")

    def yolo(self):
        # start_time = time.time()
        try:
            origin, fusionimg, timestamp = self.imgfusion_queue.get_nowait()
        except queue.Empty:
            print("Warning: imgfusion_queue is empty. Skipping yolo.")
            return
        fusionimg_tensor = fusionimg.transpose(2, 0, 1)
        fusionimg_tensor = np.expand_dims(fusionimg_tensor, axis=0)

        outputs = self.rknn_yolo.inference(inputs=[fusionimg_tensor], data_format='nchw')
        try:
            self.yolo_queue.put_nowait((origin, outputs, fusionimg))
        except queue.Full:
            pass
        # end_time = time.time()
        # print(f"yolo检测:{(end_time - start_time) * 1000:.2f}ms")

        self.frame_count += 1

        # 更新图像显示
        self.update_img()

        current_time = time.time()
        elapsed = current_time - self.start_time
        latency = current_time - timestamp
        self.total_latency += latency
        self.latency_count += 1
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            avg_latency = (self.total_latency / self.latency_count) * 1000
            self.setWindowTitle(f"RKNN 检测 | FPS: {fps:.2f} | 延迟: {avg_latency:.2f}ms")
            self.frame_count = 0
            self.total_latency = 0.0
            self.latency_count = 0
            self.start_time = current_time

    def draw_worker(self):
        while True:
            # start_time = time.time()
            try:
                origin, outputs, fusionimg = self.yolo_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            detection_outputs = [np.squeeze(out, axis=0) for out in outputs[:3]]
            boxes, classes, scores = self.processor(detection_outputs)
            display_img = self.processor.draw(fusionimg, boxes, scores, classes)
            # 把绘制好的图像送回主线程
            try:
                self.draw_queue.put_nowait((origin, display_img))
            except queue.Full:

                pass
            # end_time = time.time()
            # print(f"画框:{(end_time - start_time) * 1000:.2f}ms")

    def update_img(self):
        try:
            origin, display_img = self.draw_queue.get_nowait()
        except queue.Empty:
            print("draw_queue is empty")
            # 跳过此时没有新图像的数据
            return

        # 将原始图像转换为RGB格式并更新显示
        h, w, ch = origin.shape
        bytes_per_line = ch * w
        qimage1 = QtGui.QImage(origin.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        qimage2 = QtGui.QImage(display_img.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # 使用 pixmap 更新 UI，确保只有在有新数据时才更新
        pixmap1 = QtGui.QPixmap.fromImage(qimage1)
        pixmap2 = QtGui.QPixmap.fromImage(qimage2)

        if self.raw_label.pixmap() != pixmap1:  # 检查是否需要更新
            self.raw_label.setPixmap(pixmap1)

        if self.enh_label.pixmap() != pixmap2:  # 检查是否需要更新
            self.enh_label.setPixmap(pixmap2)


    def closeEvent(self, event):
        self.frame_queue.put(None)
        self.cam.stop()
        self.rknn_imgfusion.release()
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

# 初始化模型
SCI_YOLO_MODEL = "onnx_or_rknn/SCI_YOLOv5.rknn"
rknn_SCI_YOLO = RKNNLite()
rknn_SCI_YOLO.load_rknn(SCI_YOLO_MODEL)
rknn_SCI_YOLO.init_runtime(core_mask=RKNNLite.NPU_CORE_2)

processor = YOLOPostProcessor(ANCHOR_PATH, CLASSES, OBJ_THRESH, NMS_THRESH, IMG_SIZE)

def add_silent_audio_h264(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出
        "-i", input_path,
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "fast",  # 可选: 影响编码速度和压缩比
        "-pix_fmt", "yuv420p",  # 保证浏览器兼容性
        "-c:a", "aac",
        "-movflags", "+faststart",  # 优化网页加载速度
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_image(file_path):
    # 读取图像
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("无法读取图像文件")

    h0, w0 = image.shape[:2]
    image = cv2.resize(image, IMG_SIZE)


    image_tensor = image.transpose(2, 0, 1)
    input = np.expand_dims(image_tensor, axis=0).astype(np.uint8)  # 添加 batch 维度
    outputs = rknn_SCI_YOLO.inference(inputs=[input], data_format='nchw')

    fusion_tensor = np.array(outputs[3])
    fusion_image = fusion_tensor[0]  # 直接取batch维度
    fusion_image = np.transpose(fusion_image, (1, 2, 0))  # CHW->HWC
    fusion_image = np.clip(fusion_image * 255, 0, 255.0).astype(np.uint8)

    # 处理YOLO检测结果
    detection_outputs = [np.squeeze(out, axis=0) for out in outputs[:3]]
    boxes, classes, scores = processor(detection_outputs)
    display_img = processor.draw(fusion_image, boxes, scores, classes)

    display_img = cv2.resize(display_img, (w0, h0))

    # 生成唯一文件名
    filename = f"{uuid.uuid4().hex}.png"
    output_image_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_image_path, display_img)

    return filename

def process_video(file_path):
    video_capture = cv2.VideoCapture(file_path)
    frame_count = 0
    filename = f"{uuid.uuid4().hex}.mp4"

    # 获取原始宽高
    ret, frame = video_capture.read()
    if not ret:
        raise ValueError("无法读取视频帧")

    h0, w0 = frame.shape[:2]

    # 重新设置 video_capture 位置
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    output_video_path = os.path.join(OUTPUT_FOLDER, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w0, h0))  # ✅ 初始化一次，使用原始大小

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, IMG_SIZE)

        image_tensor = frame.transpose(2, 0, 1)
        input = np.expand_dims(image_tensor, axis=0).astype(np.uint8)  # 添加 batch 维度
        outputs = rknn_SCI_YOLO.inference(inputs=[input], data_format='nchw')

        fusion_tensor = np.array(outputs[3])
        fusion_image = fusion_tensor[0]  # 直接取batch维度
        fusion_image = np.transpose(fusion_image, (1, 2, 0))  # CHW->HWC
        fusion_image = np.clip(fusion_image * 255, 0, 255.0).astype(np.uint8)

        # 处理YOLO检测结果
        detection_outputs = [np.squeeze(out, axis=0) for out in outputs[:3]]
        boxes, classes, scores = processor(detection_outputs)
        display_img = processor.draw(fusion_image, boxes, scores, classes)

        # 恢复原始分辨率后写入
        display_img = cv2.resize(display_img, (w0, h0))
        out.write(display_img)

        frame_count += 1

    video_capture.release()
    out.release()

    final_filename = f"{uuid.uuid4().hex}.mp4"
    final_output_path = os.path.join(OUTPUT_FOLDER, final_filename)

    add_silent_audio_h264(output_video_path, final_output_path)
    os.remove(output_video_path)  # 可选：删除无音轨版本

    return final_filename


@app.route("/process_file", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        if file.filename.lower().endswith(('mp4', 'avi', 'mov')):
            output_video_name = process_video(file_path)
            return jsonify({
                "success": True,
                "file_type": "video",
                "output_video_path": output_video_name
            })
        else:
            output_image_name = process_image(file_path)
            return jsonify({
                "success": True,
                "file_type": "image",
                "output_image_path": output_image_name
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def run_flask():
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    app_ = QtWidgets.QApplication([])
    window = RKNNStreamApp()
    window.show()
    app_.exec_()
