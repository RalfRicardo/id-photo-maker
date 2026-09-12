"""Main Window for ID Photo Maker Desktop Application."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    QFileSystemWatcher,
    QObject,
    QPointF,
    QRectF,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.core.face_aligner import CropResult, FaceAligner, FaceFeatures
from app.core.matting import MattingEngine, decontaminate_color, fast_guided_filter
from app.core.psd_builder import PSDBuilder
from app.core.retouch import FaceRetoucher
from app.ui.components import (
    DARK_STYLESHEET,
    AdjustmentSlidersWidget,
    ColorPickerWidget,
    InteractiveCanvas,
    PresetSelectorWidget,
)


class AnalysisWorker(QThread):
    """
    Background worker thread that runs once per loaded image:
    1. Detects facial landmarks with MediaPipe FaceLandmarker.
    2. Runs CPU-optimized neural AI background removal (BRIA RMBG-1.4 ONNX).
    3. Refines fine hair boundaries via Fast Guided Filter.
    4. Performs Color Decontamination (Spill Suppression) to neutralize background color bleed.
    """

    finished = pyqtSignal(object, object, object, str)  # base_features, base_alpha_mask, clean_rgb, error_msg
    progress = pyqtSignal(str, int)

    def __init__(
        self,
        aligner: FaceAligner,
        matting_engine: MattingEngine,
        rgb_image: np.ndarray,
    ):
        super().__init__()
        self.aligner = aligner
        self.matting_engine = matting_engine
        self.rgb_image = rgb_image
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            self.progress.emit("Đang phát hiện mốc khuôn mặt sinh trắc học...", 20)
            if self._is_cancelled:
                return

            features = self.aligner.detect_features(self.rgb_image)
            if self._is_cancelled:
                return

            self.progress.emit("Đang tách nền AI CPU & viền tóc (RMBG-1.4 + Guided Filter)...", 50)
            alpha_mask = self.matting_engine.extract_alpha_matte(
                self.rgb_image,
                feather_radius=0.0,
                refine_edges=True,
                radius=6,
                eps=1e-4
            )
            if self._is_cancelled:
                return

            self.progress.emit("Đang khử lem màu nền cũ (Color Decontamination)...", 80)
            clean_rgb = self.matting_engine.decontaminate(
                self.rgb_image,
                alpha_mask
            )
            if self._is_cancelled:
                return

            self.progress.emit("Hoàn tất phân tích!", 100)
            self.finished.emit(features, alpha_mask, clean_rgb, "")
        except Exception as ex:
            self.finished.emit(None, None, None, str(ex))


class MainWindow(QMainWindow):
    """Main application window for ID Photo Studio."""

    def __init__(self, specs_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("ID Photo Studio Master — Multi-Layer PSD Engine")
        self.resize(1300, 850)

        # 1. Load Presets Configuration
        self.specs_path = specs_path or self._default_specs_path()
        self.specs_data = self._load_specs()
        self.presets = self.specs_data.get("presets", {})
        self.print_sheets = self.specs_data.get("print_sheets", {})

        # 2. Initialize Engines
        self.aligner = FaceAligner()
        self.matting_engine = MattingEngine()
        self.psd_builder = PSDBuilder()

        # 3. State Variables
        self.current_image_path: Optional[str] = None
        self.original_rgb: Optional[np.ndarray] = None
        self.clean_rgb: Optional[np.ndarray] = None
        self.base_features: Optional[FaceFeatures] = None
        self.base_alpha_mask: Optional[np.ndarray] = None

        # Processed Result State
        self.crop_res: Optional[CropResult] = None
        self.alpha_mask: Optional[np.ndarray] = None
        self.bg_color: str = "#0090FF"

        # Background Watcher & Worker
        self.watch_folder_path: Optional[str] = None
        self.watcher: Optional[QFileSystemWatcher] = None
        self.worker: Optional[AnalysisWorker] = None
        self.manual_brush_delta: Optional[np.ndarray] = None

        # Debounce timer for slider adjustments (15ms debounce for silky smooth 60fps)
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(15)
        self.slider_timer.timeout.connect(self._apply_adjustments_fast)

        # 4. Setup UI
        self.setStyleSheet(DARK_STYLESHEET)
        self._init_ui()

    def _default_specs_path(self) -> str:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(current_dir, "presets", "photo_specs.json")

    def _load_specs(self) -> Dict[str, Any]:
        if os.path.isfile(self.specs_path):
            with open(self.specs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _init_ui(self) -> None:
        # Central Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Left Control Sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(360)
        sidebar_scroll.setMaximumWidth(420)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)

        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(14)

        # 1. Preset Selector
        preset_group = QGroupBox("1. TIÊU CHUẨN ẢNH (PRESET)")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_widget = PresetSelectorWidget(self.presets)
        self.preset_widget.preset_changed.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_widget)
        sidebar_layout.addWidget(preset_group)

        # 2. Background Color Picker
        color_group = QGroupBox("2. MÀU NỀN (BACKGROUND)")
        color_layout = QVBoxLayout(color_group)
        default_color = self.preset_widget.get_current_spec().get("default_bg_color", "#0090FF")
        self.color_widget = ColorPickerWidget(default_color)
        self.color_widget.color_changed.connect(self._on_color_changed)
        color_layout.addWidget(self.color_widget)
        sidebar_layout.addWidget(color_group)

        # 3. Fine-Tune Sliders
        tune_group = QGroupBox("3. CÂN CHỈNH VI MÔ & KHUÔN MẶT")
        tune_layout = QVBoxLayout(tune_group)
        self.sliders_widget = AdjustmentSlidersWidget()
        self.sliders_widget.values_changed.connect(self._on_adjustments_changed)
        tune_layout.addWidget(self.sliders_widget)
        sidebar_layout.addWidget(tune_group)

        # 4. View Toggles
        view_group = QGroupBox("4. HIỂN THỊ XEM TRƯỚC")
        view_layout = QVBoxLayout(view_group)
        self.chk_guides = QCheckBox("Hiện khung căn chuẩn (Guide Lines)")
        self.chk_guides.setChecked(True)
        self.chk_guides.toggled.connect(self._toggle_guides)
        view_layout.addWidget(self.chk_guides)

        self.chk_mask_only = QCheckBox("Xem riêng mặt nạ Alpha Matte")
        self.chk_mask_only.setChecked(False)
        self.chk_mask_only.toggled.connect(self._toggle_mask_view)
        view_layout.addWidget(self.chk_mask_only)
        sidebar_layout.addWidget(view_group)

        sidebar_layout.addStretch()

        # Action Export Buttons in Sidebar
        self.btn_export_psd = QPushButton("💾 Xuất file PSD nhiều lớp (Multi-layer)")
        self.btn_export_psd.setProperty("class", "primary")
        self.btn_export_psd.setFixedHeight(44)
        self.btn_export_psd.clicked.connect(self.export_psd)
        sidebar_layout.addWidget(self.btn_export_psd)

        self.btn_export_sheet = QPushButton("🖨️ Xuất bảng in 10x15 cm (Print Sheet)")
        self.btn_export_sheet.setFixedHeight(36)
        self.btn_export_sheet.clicked.connect(self.export_print_sheet_dialog)
        sidebar_layout.addWidget(self.btn_export_sheet)

        sidebar_scroll.setWidget(sidebar_container)
        main_splitter.addWidget(sidebar_scroll)

        # Right Preview Area
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        # Top Canvas Header Bar
        canvas_bar = QHBoxLayout()
        self.lbl_compliance_badge = QLabel("Chưa có ảnh")
        self.lbl_compliance_badge.setStyleSheet("""
            background-color: #27272A;
            color: #A1A1AA;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
        """)
        canvas_bar.addWidget(self.lbl_compliance_badge)
        canvas_bar.addStretch()

        btn_open = QPushButton("📂 Mở ảnh...")
        btn_open.clicked.connect(self.open_file_dialog)
        canvas_bar.addWidget(btn_open)

        btn_fit = QPushButton("🔍 Vừa khung (Fit)")
        btn_fit.clicked.connect(self._fit_canvas)
        canvas_bar.addWidget(btn_fit)

        btn_watch = QPushButton("👀 Thư mục chụp (Watch Folder)...")
        btn_watch.clicked.connect(self.setup_watch_folder)
        canvas_bar.addWidget(btn_watch)

        right_layout.addLayout(canvas_bar)

        # Dedicated Retouch & Eraser Tool Bar
        tool_frame = QFrame()
        tool_frame.setStyleSheet("""
            QFrame {
                background-color: #18191E;
                border: 1px solid #27272A;
                border-radius: 6px;
            }
        """)
        tool_layout = QHBoxLayout(tool_frame)
        tool_layout.setContentsMargins(8, 6, 8, 6)
        tool_layout.setSpacing(8)

        lbl_tools = QLabel("Công cụ:")
        lbl_tools.setStyleSheet("font-weight: bold; color: #38BDF8; font-size: 12px;")
        tool_layout.addWidget(lbl_tools)

        self.tool_group = QButtonGroup(self)

        self.btn_tool_pan = QPushButton("🖐️ Di chuyển (Pan)")
        self.btn_tool_pan.setCheckable(True)
        self.btn_tool_pan.setChecked(True)
        self.tool_group.addButton(self.btn_tool_pan)
        tool_layout.addWidget(self.btn_tool_pan)

        self.btn_tool_eraser = QPushButton("🧹 Tẩy tóc thừa & viền áo (Eraser)")
        self.btn_tool_eraser.setCheckable(True)
        self.btn_tool_eraser.setStyleSheet("""
            QPushButton:checked {
                background-color: #7F1D1D;
                border: 1px solid #EF4444;
                color: #FFFFFF;
                font-weight: bold;
            }
        """)
        self.tool_group.addButton(self.btn_tool_eraser)
        tool_layout.addWidget(self.btn_tool_eraser)

        self.btn_tool_restore = QPushButton("🖌️ Khôi phục nét (Restore)")
        self.btn_tool_restore.setCheckable(True)
        self.btn_tool_restore.setStyleSheet("""
            QPushButton:checked {
                background-color: #064E3B;
                border: 1px solid #10B981;
                color: #FFFFFF;
                font-weight: bold;
            }
        """)
        self.tool_group.addButton(self.btn_tool_restore)
        tool_layout.addWidget(self.btn_tool_restore)

        self.tool_group.buttonClicked.connect(self._on_tool_changed)

        tool_layout.addSpacing(10)
        lbl_size = QLabel("Cỡ cọ:")
        lbl_size.setStyleSheet("color: #A1A1AA; font-size: 12px;")
        tool_layout.addWidget(lbl_size)

        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(4, 80)
        self.slider_brush_size.setValue(15)
        self.slider_brush_size.setFixedWidth(90)
        self.slider_brush_size.valueChanged.connect(self._on_brush_size_changed)
        tool_layout.addWidget(self.slider_brush_size)

        self.lbl_brush_size = QLabel("15 px")
        self.lbl_brush_size.setStyleSheet("color: #38BDF8; font-weight: bold; min-width: 40px;")
        tool_layout.addWidget(self.lbl_brush_size)

        tool_layout.addStretch()

        self.btn_reset_mask = QPushButton("↺ Khôi phục mặt nạ gốc")
        self.btn_reset_mask.setToolTip("Khôi phục lại mặt nạ bóc tách AI gốc (hủy bỏ nét tẩy)")
        self.btn_reset_mask.clicked.connect(self._reset_mask_to_original)
        tool_layout.addWidget(self.btn_reset_mask)

        right_layout.addWidget(tool_frame)

        # Interactive Canvas
        self.canvas = InteractiveCanvas()
        self.canvas.file_dropped.connect(self.load_image_file)
        self.canvas.brush_painted.connect(self._on_canvas_brush_painted)
        right_layout.addWidget(self.canvas, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #27272A;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #0090FF;
                border-radius: 3px;
            }
        """)
        right_layout.addWidget(self.progress_bar)

        main_splitter.addWidget(right_container)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng. Kéo thả ảnh vào khung hoặc bấm 'Mở ảnh'.")

    # Image Loading & Processing
    def open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh chân dung",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.webp *.bmp *.tiff);;Tất cả (*.*)"
        )
        if file_path:
            self.load_image_file(file_path)

    def load_image_file(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            return

        self.current_image_path = file_path
        # Read image with OpenCV (handles UTF-8 paths safely via binary buffer)
        try:
            stream = open(file_path, "rb")
            bytes_data = bytearray(stream.read())
            stream.close()
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            bgr = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Không thể giải mã định dạng ảnh")
            self.original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi nạp ảnh", f"Không thể đọc file: {e}")
            return

        # Reset state & sliders on new image
        self.sliders_widget.reset_values()
        self.base_features = None
        self.base_alpha_mask = None
        self.clean_rgb = None
        self.manual_brush_delta = None

        # Start initial analysis in worker thread
        self.start_initial_analysis()

    def start_initial_analysis(self) -> None:
        if self.original_rgb is None:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.status_bar.showMessage("Đang phân tích khuôn mặt và tách nền AI...")

        # If previous worker is running, cancel it gracefully without calling terminate()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

        self.worker = AnalysisWorker(
            aligner=self.aligner,
            matting_engine=self.matting_engine,
            rgb_image=self.original_rgb
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.start()

    def _on_worker_progress(self, msg: str, val: int) -> None:
        self.status_bar.showMessage(msg)
        self.progress_bar.setValue(val)

    def _on_analysis_finished(
        self,
        base_features: Optional[FaceFeatures],
        base_alpha_mask: Optional[np.ndarray],
        clean_rgb: Optional[np.ndarray],
        err_msg: str
    ) -> None:
        self.progress_bar.setVisible(False)
        if err_msg:
            self.status_bar.showMessage(f"Lỗi phân tích: {err_msg}")
            QMessageBox.warning(self, "Cảnh báo xử lý", f"Không thể hoàn tất phân tích: {err_msg}")
            return

        self.base_features = base_features
        self.base_alpha_mask = base_alpha_mask
        self.clean_rgb = clean_rgb if clean_rgb is not None else self.original_rgb

        # Immediately run instantaneous fast-path adjustment
        self._apply_adjustments_fast()

    def _apply_adjustments_fast(self) -> None:
        """
        Instantaneous (< 10ms) geometric alignment, affine warp, skin retouch, and feathering.
        Runs purely on CPU via OpenCV without re-running neural network rembg.
        Guarantees silky smooth 60 FPS slider updates without freezing or crashing.
        """
        if self.original_rgb is None or self.base_features is None:
            return

        spec = self.preset_widget.get_current_spec()
        adjustments = self.sliders_widget.get_values()

        target_w = int(spec.get("width_px", 354))
        target_h = int(spec.get("height_px", 472))

        # 1. Fast geometric alignment using cached landmarks and color-decontaminated RGB
        source_rgb = self.clean_rgb if self.clean_rgb is not None else self.original_rgb
        self.crop_res = self.aligner.process(
            rgb_image=source_rgb,
            preset_spec=spec,
            scale_offset=adjustments.get("scale_offset", 0.0),
            x_shift_px=adjustments.get("x_shift_px", 0.0),
            y_shift_px=adjustments.get("y_shift_px", 0.0),
            rotation_offset_deg=adjustments.get("rotation_offset_deg", 0.0),
            detected_features=self.base_features
        )

        # 2. Retouch: skin smoothing (làm mịn da / xóa mụn) & detail sharpening
        cropped_rgb = self.crop_res.cropped_image.copy()
        smooth_pct = adjustments.get("skin_smooth_pct", 0.0)
        if smooth_pct > 0.001:
            cropped_rgb = FaceRetoucher.smooth_skin(cropped_rgb, smooth_pct)

        sharpen_pct = adjustments.get("sharpen_pct", 0.0)
        if sharpen_pct > 0.001:
            cropped_rgb = FaceRetoucher.sharpen(cropped_rgb, sharpen_pct)

        self.crop_res.cropped_image = cropped_rgb

        # 3. Fast warp of the base alpha mask using the exact affine transform
        if self.base_alpha_mask is not None:
            cropped_mask = cv2.warpAffine(
                self.base_alpha_mask,
                self.crop_res.transform_matrix,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            cropped_mask = np.full((target_h, target_w), 255, dtype=np.uint8)

        # 4. Fast hair / edge guided refinement using Fast Guided Filter
        feather = adjustments.get("feather_radius", 1.0)
        if feather > 0.1:
            r = max(2, min(int(round(feather * 3 + 1)), 16))
            gray_crop = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            refined_float = fast_guided_filter(guide=gray_crop, src=cropped_mask, radius=r, eps=1e-4)
            cropped_mask = (refined_float * 255.0).clip(0, 255).astype(np.uint8)

        # 5. Apply manual brush strokes (eraser / restore) if any
        if self.manual_brush_delta is not None and self.manual_brush_delta.shape == (target_h, target_w):
            cropped_mask = np.where(self.manual_brush_delta == -1, 0, cropped_mask)
            cropped_mask = np.where(self.manual_brush_delta == 1, 255, cropped_mask)

        self.alpha_mask = cropped_mask

        # 6. Update preview and compliance UI
        self._update_preview()
        self._update_compliance_badge()

        self.status_bar.showMessage(
            f"✓ {spec.get('name', '')} ({target_w}x{target_h}px) | "
            f"Góc nghiêng đã sửa: {self.crop_res.features.roll_angle_deg:+.1f}° | 300 DPI"
        )

    def _on_tool_changed(self, button: QPushButton) -> None:
        if button == self.btn_tool_eraser:
            self.canvas.set_tool_mode("eraser")
            self.status_bar.showMessage("🧹 Chế độ cọ tẩy: Nhấn và kéo chuột trên ảnh để xóa tóc thừa hoặc viền áo dính nền.")
        elif button == self.btn_tool_restore:
            self.canvas.set_tool_mode("restore")
            self.status_bar.showMessage("🖌️ Chế độ khôi phục: Nhấn và kéo chuột để lấy lại các chi tiết tóc/áo đã bị xóa.")
        else:
            self.canvas.set_tool_mode("pan")
            self.status_bar.showMessage("🖐️ Chế độ di chuyển: Kéo chuột để di chuyển ảnh, lăn chuột để phóng to/thu nhỏ.")

    def _on_brush_size_changed(self, val: int) -> None:
        self.lbl_brush_size.setText(f"{val} px")
        self.canvas.set_brush_radius(val)

    def _on_canvas_brush_painted(self, img_x: float, img_y: float, radius: int, mode: str) -> None:
        if self.alpha_mask is None or self.crop_res is None:
            return

        h, w = self.alpha_mask.shape[:2]
        if self.manual_brush_delta is None or self.manual_brush_delta.shape != (h, w):
            self.manual_brush_delta = np.zeros((h, w), dtype=np.int8)

        # Update manual delta
        val_delta = -1 if mode == "eraser" else 1
        cv2.circle(self.manual_brush_delta, (int(round(img_x)), int(round(img_y))), int(radius), val_delta, -1, lineType=cv2.LINE_AA)

        # Mutate active alpha_mask immediately
        val_mask = 0 if mode == "eraser" else 255
        cv2.circle(self.alpha_mask, (int(round(img_x)), int(round(img_y))), int(radius), val_mask, -1, lineType=cv2.LINE_AA)

        # Instantaneous redraw of preview canvas
        self._update_preview()

    def _reset_mask_to_original(self) -> None:
        if self.crop_res is None:
            return
        self.manual_brush_delta = None
        self._apply_adjustments_fast()
        self.status_bar.showMessage("✓ Đã khôi phục lại mặt nạ bóc tách AI gốc.")

    def _update_preview(self) -> None:
        if self.crop_res is None or self.alpha_mask is None:
            self.canvas.set_image_and_guides(None)
            return

        composite = MattingEngine.apply_matte(
            self.crop_res.cropped_image,
            self.alpha_mask,
            self.bg_color
        )

        # Prepare guidelines coordinates
        h, w = self.crop_res.cropped_image.shape[:2]
        fc = self.crop_res.features_in_crop
        guides = {
            "eye_y": fc.eyes_center[1],
            "crown_y": fc.crown[1],
            "chin_y": fc.chin[1],
            "center_x": w / 2.0
        }

        self.canvas.set_image_and_guides(
            composite_rgb=composite,
            guidelines_data=guides,
            mask_array=self.alpha_mask
        )

    def _update_compliance_badge(self) -> None:
        if self.crop_res is None:
            return

        spec = self.preset_widget.get_current_spec()
        h = self.crop_res.target_height
        fc = self.crop_res.features_in_crop

        face_h = abs(fc.chin[1] - fc.crown[1])
        face_pct = face_h / h

        min_pct = spec.get("min_face_height_pct", 0.70)
        max_pct = spec.get("max_face_height_pct", 0.80)

        if min_pct <= face_pct <= max_pct:
            self.lbl_compliance_badge.setText(
                f"✓ ĐẠT CHUẨN: Tỉ lệ đầu {int(face_pct*100)}% (Chuẩn: {int(min_pct*100)}-{int(max_pct*100)}%)"
            )
            self.lbl_compliance_badge.setStyleSheet("""
                background-color: #064E3B;
                color: #34D399;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
            """)
        else:
            self.lbl_compliance_badge.setText(
                f"⚠️ CHƯA ĐẠT: Tỉ lệ đầu {int(face_pct*100)}% (Khuyến nghị: {int(min_pct*100)}-{int(max_pct*100)}%)"
            )
            self.lbl_compliance_badge.setStyleSheet("""
                background-color: #7C2D12;
                color: #F87171;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
            """)

    # Event Handlers
    def _on_preset_changed(self, preset_id: str, spec: Dict[str, Any]) -> None:
        def_bg = spec.get("default_bg_color", "#0090FF")
        self.color_widget.set_color(def_bg)
        self._apply_adjustments_fast()

    def _on_color_changed(self, hex_color: str) -> None:
        self.bg_color = hex_color
        self._update_preview()

    def _on_adjustments_changed(self) -> None:
        # Trigger single-shot debounced fast update (15ms)
        self.slider_timer.start()

    def _toggle_guides(self, checked: bool) -> None:
        self.canvas.show_guides = checked
        self.canvas.update()

    def _toggle_mask_view(self, checked: bool) -> None:
        self.canvas.show_mask_only = checked
        self._update_preview()

    def _fit_canvas(self) -> None:
        self.canvas.fit_in_view()

    # Watch Folder Mode
    def setup_watch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục đồng bộ chụp ảnh (Watch Folder)")
        if not folder:
            return

        self.watch_folder_path = folder
        if self.watcher is None:
            self.watcher = QFileSystemWatcher(self)
            self.watcher.directoryChanged.connect(self._on_watch_folder_changed)

        self.watcher.addPath(folder)
        self.status_bar.showMessage(f"👀 Đang theo dõi thư mục: {folder}")
        QMessageBox.information(
            self,
            "Chế độ Studio Watch Folder",
            f"Đã bật theo dõi thư mục:\n{folder}\n\nMọi ảnh mới chụp xuất vào thư mục này sẽ được tự động nạp!"
        )

    def _on_watch_folder_changed(self, path: str) -> None:
        if not os.path.isdir(path):
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(valid_exts)
        ]
        if not files:
            return

        latest_file = max(files, key=os.path.getmtime)
        if latest_file != self.current_image_path:
            self.load_image_file(latest_file)

    # Export Functions
    def export_psd(self) -> None:
        if self.crop_res is None or self.alpha_mask is None:
            QMessageBox.information(self, "Chưa có ảnh", "Vui lòng mở ảnh và hoàn tất căn chỉnh trước khi xuất.")
            return

        default_name = "anh_the_id.psd"
        if self.current_image_path:
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
            pid = self.preset_widget.current_preset_id
            default_name = f"{base}_{pid}.psd"

        out_file, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất Adobe Photoshop PSD nhiều lớp",
            default_name,
            "Adobe Photoshop (*.psd)"
        )
        if not out_file:
            return

        fc = self.crop_res.features_in_crop
        guides_info = {
            "eye_y": fc.eyes_center[1],
            "crown_y": fc.crown[1],
            "chin_y": fc.chin[1],
            "center_x": self.crop_res.target_width / 2.0
        }

        try:
            self.psd_builder.export_id_photo(
                rgb_portrait=self.crop_res.cropped_image,
                alpha_mask=self.alpha_mask,
                output_path=out_file,
                bg_color=self.bg_color,
                guidelines_info=guides_info,
                dpi=300
            )
            self.status_bar.showMessage(f"✓ Đã xuất PSD thành công: {out_file}")
            QMessageBox.information(
                self,
                "Xuất PSD thành công",
                f"Đã tạo file PSD chuẩn studio tại:\n{out_file}\n\n"
                "• Lớp 1: Nền (Background)\n"
                "• Lớp 2: Chủ thể (Portrait) + Layer Mask không phá hủy\n"
                "• Lớp 3: Khung căn chuẩn (Guide lines)\n"
                "• Độ phân giải: 300 DPI"
            )
        except Exception as err:
            QMessageBox.critical(self, "Lỗi xuất PSD", f"Không thể lưu file PSD: {err}")

    def export_print_sheet_dialog(self) -> None:
        if self.crop_res is None or self.alpha_mask is None:
            QMessageBox.information(self, "Chưa có ảnh", "Vui lòng nạp ảnh trước khi xuất bảng in.")
            return

        sheet_keys = list(self.print_sheets.keys())
        if not sheet_keys:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy cấu hình bảng in trong photo_specs.json.")
            return

        sheet_id = "10x15_combo" if "10x15_combo" in self.print_sheets else sheet_keys[0]
        sheet_spec = self.print_sheets[sheet_id]

        out_file, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất bảng in 10x15 cm @ 300 DPI",
            "bang_in_10x15.psd",
            "Adobe Photoshop (*.psd)"
        )
        if not out_file:
            return

        # Prepare photo crops for available presets
        adjustments = self.sliders_widget.get_values()
        smooth_pct = adjustments.get("skin_smooth_pct", 0.0)
        sharpen_pct = adjustments.get("sharpen_pct", 0.0)

        photo_crops = {}
        for pid in ("3x4_vn", "4x6_vn", "2x2_us_visa", "35x45_schengen"):
            if pid in self.presets and self.original_rgb is not None:
                if pid == self.preset_widget.current_preset_id and self.crop_res is not None and self.alpha_mask is not None:
                    photo_crops[pid] = (self.crop_res.cropped_image, self.alpha_mask, self.bg_color)
                else:
                    c_res = self.aligner.process(
                        self.original_rgb,
                        self.presets[pid],
                        detected_features=self.base_features
                    )
                    cropped_rgb = c_res.cropped_image.copy()
                    if smooth_pct > 0.001:
                        cropped_rgb = FaceRetoucher.smooth_skin(cropped_rgb, smooth_pct)
                    if sharpen_pct > 0.001:
                        cropped_rgb = FaceRetoucher.sharpen(cropped_rgb, sharpen_pct)

                    if self.base_alpha_mask is not None:
                        c_mask = cv2.warpAffine(
                            self.base_alpha_mask,
                            c_res.transform_matrix,
                            (c_res.target_width, c_res.target_height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                    else:
                        c_mask = cv2.resize(self.alpha_mask, (c_res.target_width, c_res.target_height))
                    photo_crops[pid] = (cropped_rgb, c_mask, self.bg_color)

        try:
            self.psd_builder.export_print_sheet(
                sheet_spec=sheet_spec,
                photo_crops=photo_crops,
                output_path=out_file,
                dpi=300
            )
            QMessageBox.information(
                self,
                "Bảng in hoàn tất",
                f"Đã xuất bảng in 10x15 cm @ 300 DPI:\n{out_file}\n\n"
                "Bảng in đã được xếp sẵn kèm đường cắt căn lề sẵn sàng cho máy in studio!"
            )
        except Exception as err:
            QMessageBox.critical(self, "Lỗi xuất bảng in", f"Không thể lưu bảng in: {err}")

    def closeEvent(self, event: Any) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
        event.accept()
