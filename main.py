import os
import sys
import time
import traceback
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox, QComboBox, QProgressBar,
    QLineEdit, QGridLayout, QFrame
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
try:
    # ResNet50 preprocess (tuỳ chọn)
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
except Exception:
    resnet50_preprocess = None

APP_TITLE = "Skin Lesion Classifier"

# Fallback nhãn mặc định theo thứ tự alphabet (flow_from_directory)
DEFAULT_LABELS = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

# ============ TF GPU memory growth (đỡ full VRAM) ============
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# ==================== Utils ====================
def load_image_for_predict(path: str,
                           target_size: Tuple[int, int],
                           mode: str = "rescale",
                           channels: int = 3) -> np.ndarray:
    """
    Đọc ảnh, convert RGB, resize, tiền xử lý theo mode.
    mode: "rescale" (mặc định) hoặc "resnet50"
    """
    im = Image.open(path).convert("RGB")
    im = im.resize(target_size, Image.LANCZOS)
    arr = np.array(im).astype("float32")

    if mode == "resnet50":
        if resnet50_preprocess is None:
            # fallback nếu không có
            arr = arr / 255.0
        else:
            # resnet50_preprocess nhận BGR theo chuẩn keras? Hàm này tự xử lý mean/std & channel
            arr = resnet50_preprocess(arr)
    else:
        # rescale 1/255 (khớp với ImageDataGenerator(rescale=1./255))
        arr = arr / 255.0

    if channels == 1:
        # nếu model là grayscale (hiếm)
        arr = np.mean(arr, axis=-1, keepdims=True)

    # add batch dim
    arr = np.expand_dims(arr, axis=0)
    return arr


def qimage_from_pil(pil_img: Image.Image) -> QImage:
    """Convert PIL -> QImage để show preview."""
    pil_img = pil_img.convert("RGBA")
    w, h = pil_img.size
    data = pil_img.tobytes("raw", "RGBA")
    return QImage(data, w, h, QImage.Format_RGBA8888)


# ==================== Worker Thread ====================
class PredictWorker(QThread):
    finished = pyqtSignal(str, float, float)  # (label, prob, elapsed_ms)
    failed = pyqtSignal(str)

    def __init__(self, model: tf.keras.Model, image_path: str,
                 input_size: Tuple[int, int], labels: List[str],
                 preprocess_mode: str):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.input_size = input_size
        self.labels = labels
        self.preprocess_mode = preprocess_mode

    def run(self):
        try:
            t0 = time.time()
            batch = load_image_for_predict(
                self.image_path,
                target_size=self.input_size,
                mode=self.preprocess_mode,
                channels=3
            )
            preds = self.model.predict(batch, verbose=0)
            # softmax output shape (1, num_classes)
            probs = preds[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            top_label = self.labels[top_idx] if 0 <= top_idx < len(self.labels) else f"Class #{top_idx}"
            elapsed = (time.time() - t0) * 1000.0
            self.finished.emit(top_label, top_prob, elapsed)
        except Exception as e:
            self.failed.emit(f"Predict error: {e}\n{traceback.format_exc()}")


# ==================== Main Window ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(900, 600)

        # state
        self.model: Optional[tf.keras.Model] = None
        self.model_input_size = (224, 224)  # fallback
        self.labels: List[str] = list(DEFAULT_LABELS)
        self.image_path: Optional[str] = None

        # UI
        central = QWidget()
        self.setCentralWidget(central)

        # ---------- Left: Controls ----------
        ctrl_box = QGroupBox("Điều khiển")
        ctrl_layout = QGridLayout(ctrl_box)

        self.btn_load_model = QPushButton("📦 Load Model (.h5/.keras/SavedModel)")
        self.btn_load_model.clicked.connect(self.on_load_model)

        self.le_model_path = QLineEdit()
        self.le_model_path.setReadOnly(True)
        self.le_model_path.setPlaceholderText("Chưa chọn model...")

        self.btn_load_labels = QPushButton("🏷️ Load Labels (.txt)")
        self.btn_load_labels.clicked.connect(self.on_load_labels)

        self.le_labels_info = QLineEdit()
        self.le_labels_info.setReadOnly(True)
        self.le_labels_info.setPlaceholderText("Mặc định 9 lớp ISIC (nếu không load file)")

        self.cbo_preprocess = QComboBox()
        self.cbo_preprocess.addItems(["Rescale 1/255 (khuyến nghị)", "ResNet50 preprocess_input"])
        self.cbo_preprocess.setCurrentIndex(0)

        self.btn_open_image = QPushButton("🖼️ Chọn Ảnh...")
        self.btn_open_image.clicked.connect(self.on_open_image)

        self.btn_predict = QPushButton("🔮 Dự đoán (Top-1)")
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_predict.setEnabled(False)

        self.lbl_input_size = QLabel("Input size: (224, 224) • Channels: 3")

        ctrl_layout.addWidget(self.btn_load_model,   0, 0, 1, 2)
        ctrl_layout.addWidget(self.le_model_path,    1, 0, 1, 2)
        ctrl_layout.addWidget(self.btn_load_labels,  2, 0, 1, 1)
        ctrl_layout.addWidget(self.le_labels_info,   2, 1, 1, 1)
        ctrl_layout.addWidget(QLabel("Tiền xử lý:"), 3, 0, 1, 1)
        ctrl_layout.addWidget(self.cbo_preprocess,   3, 1, 1, 1)
        ctrl_layout.addWidget(self.btn_open_image,   4, 0, 1, 1)
        ctrl_layout.addWidget(self.btn_predict,      4, 1, 1, 1)
        ctrl_layout.addWidget(self.lbl_input_size,   5, 0, 1, 2)

        # ---------- Right: Preview & Result ----------
        right_box = QGroupBox("Ảnh & Kết quả")
        right_layout = QVBoxLayout(right_box)

        self.preview = QLabel("Kéo thả ảnh vào đây hoặc nhấn 'Chọn Ảnh...'")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setAcceptDrops(True)

        # Enable drag&drop
        self.preview.installEventFilter(self)

        # Result area
        self.lbl_pred = QLabel("Bệnh dự đoán: —")
        self.lbl_pred.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.pb_conf = QProgressBar()
        self.pb_conf.setRange(0, 100)
        self.pb_conf.setValue(0)
        self.lbl_conf = QLabel("Độ tin cậy: 0.00%   •   Thời gian: — ms")

        right_layout.addWidget(self.preview)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.lbl_pred)
        right_layout.addWidget(self.pb_conf)
        right_layout.addWidget(self.lbl_conf)
        right_layout.addStretch(1)

        # ---------- Layout root ----------
        root = QHBoxLayout(central)
        root.addWidget(ctrl_box, 1)
        root.addWidget(right_box, 2)

        # Styling nhẹ
        self.setStyleSheet("""
            QGroupBox { font-weight: 600; }
            QPushButton { padding: 8px 10px; }
        """)

    # -------- Drag & Drop handlers on preview --------
    def eventFilter(self, obj, event):
        if obj is self.preview:
            if event.type() == event.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            if event.type() == event.Drop:
                urls = event.mimeData().urls()
                if urls:
                    path = urls[0].toLocalFile()
                    self.set_image(path)
                return True
        return super().eventFilter(obj, event)

    # ---------------- Actions ----------------
    def on_load_model(self):
        path = QFileDialog.getOpenFileName(
            self, "Chọn model (.h5 hoặc .keras)", "", "Keras model (*.h5 *.keras *.hdf5);;All files (*)"
        )[0]

        if not path:
            # Cho phép chọn thư mục SavedModel
            dirpath = QFileDialog.getExistingDirectory(self, "Hoặc chọn thư mục SavedModel (assets/variables)")
            if not dirpath:
                return
            path = dirpath

        try:
            # Nếu model dùng custom_objects thì có thể thêm vào đây
            self.model = load_model(path, compile=False)
            self.le_model_path.setText(path)

            # Lấy input shape
            ishape = self.model.inputs[0].shape  # (None, H, W, C)
            try:
                h = int(ishape[1])
                w = int(ishape[2])
                c = int(ishape[3]) if ishape[3] is not None else 3
            except Exception:
                h, w, c = 224, 224, 3
            self.model_input_size = (w, h)  # (W, H) nhưng ta dùng (H, W) ở load_image -> cẩn thận
            self.lbl_input_size.setText(f"Input size: ({h}, {w}) • Channels: {c}")
            self.btn_predict.setEnabled(True)
            QMessageBox.information(self, "OK", "Model đã được load thành công.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không load được model:\n{e}\n{traceback.format_exc()}")

    def on_load_labels(self):
        path = QFileDialog.getOpenFileName(
            self, "Chọn file labels (.txt - mỗi dòng 1 lớp)", "", "Text files (*.txt);;All files (*)"
        )[0]
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if not lines:
                raise ValueError("File rỗng.")
            self.labels = lines
            self.le_labels_info.setText(f"{os.path.basename(path)} • {len(lines)} lớp")
            QMessageBox.information(self, "OK", f"Đã nạp {len(lines)} nhãn.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được labels:\n{e}")

    def on_open_image(self):
        path = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)"
        )[0]
        if not path:
            return
        self.set_image(path)

    def set_image(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Cảnh báo", "Đường dẫn ảnh không hợp lệ.")
            return
        try:
            self.image_path = path
            pil = Image.open(path).convert("RGB")
            # Fit preview
            max_w, max_h = 720, 420
            pil_thumb = pil.copy()
            pil_thumb.thumbnail((max_w, max_h), Image.LANCZOS)
            qimg = qimage_from_pil(pil_thumb)
            self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.lbl_pred.setText("Bệnh dự đoán: —")
            self.pb_conf.setValue(0)
            self.lbl_conf.setText("Độ tin cậy: 0.00%   •   Thời gian: — ms")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không hiển thị được ảnh:\n{e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # auto-fit preview khi resize
        if self.preview.pixmap():
            self.preview.setPixmap(self.preview.pixmap().scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def on_predict(self):
        if self.model is None:
            QMessageBox.warning(self, "Thiếu model", "Vui lòng load model trước.")
            return
        if not self.image_path:
            QMessageBox.warning(self, "Thiếu ảnh", "Vui lòng chọn ảnh đầu vào.")
            return

        proc = "rescale" if self.cbo_preprocess.currentIndex() == 0 else "resnet50"
        H = self.model_input_size[1] if len(self.model_input_size) == 2 else 224
        W = self.model_input_size[0] if len(self.model_input_size) == 2 else 224

        self.btn_predict.setEnabled(False)
        self.lbl_pred.setText("Đang dự đoán...")
        self.pb_conf.setValue(0)

        self.worker = PredictWorker(
            model=self.model,
            image_path=self.image_path,
            input_size=(H, W),
            labels=self.labels,
            preprocess_mode=proc
        )
        self.worker.finished.connect(self.on_predict_done)
        self.worker.failed.connect(self.on_predict_failed)
        self.worker.start()

    def on_predict_done(self, label: str, prob: float, elapsed_ms: float):
        self.btn_predict.setEnabled(True)
        self.lbl_pred.setText(f"Bệnh dự đoán: {label}")
        self.pb_conf.setValue(int(round(prob * 100)))
        self.lbl_conf.setText(f"Độ tin cậy: {prob*100:.2f}%   •   Thời gian: {elapsed_ms:.1f} ms")

        # Tô màu nhẹ theo độ tin cậy
        if prob >= 0.8:
            self.lbl_pred.setStyleSheet("font-size: 18px; font-weight: 700; color: #0b7a26;")
        elif prob >= 0.5:
            self.lbl_pred.setStyleSheet("font-size: 18px; font-weight: 700; color: #e69500;")
        else:
            self.lbl_pred.setStyleSheet("font-size: 18px; font-weight: 700; color: #b91c1c;")

    def on_predict_failed(self, msg: str):
        self.btn_predict.setEnabled(True)
        QMessageBox.critical(self, "Lỗi dự đoán", msg)


# ==================== main ====================
def main():
    # macOS hiDPI
    if os.name == "posix":
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
