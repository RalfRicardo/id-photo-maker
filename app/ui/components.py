"""Custom reusable PyQt6 UI components for ID Photo Maker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core.matting import hex_to_rgb, rgb_to_hex


DARK_STYLESHEET = """
QWidget {
    background-color: #121316;
    color: #E4E4E7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #27272A;
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 14px;
    background-color: #18191E;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #00A6FF;
}

QPushButton {
    background-color: #27272A;
    color: #FAFAFA;
    border: 1px solid #3F3F46;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3F3F46;
    border-color: #52525B;
}

QPushButton:pressed {
    background-color: #18181B;
}

QPushButton.primary {
    background-color: #0077EE;
    border-color: #0090FF;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton.primary:hover {
    background-color: #0090FF;
}

QSlider::groove:horizontal {
    border: 1px solid #27272A;
    height: 6px;
    background: #27272A;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #0090FF;
    border: 1px solid #0090FF;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #38BDF8;
}

QComboBox {
    background-color: #27272A;
    border: 1px solid #3F3F46;
    border-radius: 6px;
    padding: 6px 10px;
    color: #FAFAFA;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QScrollBar:vertical {
    border: none;
    background: #18191E;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3F3F46;
    min-height: 20px;
    border-radius: 5px;
}
"""


class PresetSelectorWidget(QWidget):
    """Card and dropdown selector for ID / Visa photo presets."""

    preset_changed = pyqtSignal(str, dict)

    def __init__(self, presets: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.presets = presets
        self.current_preset_id = next(iter(presets.keys())) if presets else "3x4_vn"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Dropdown
        self.combo = QComboBox()
        for pid, spec in self.presets.items():
            self.combo.addItem(f"{spec.get('name', pid)}", pid)

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self.combo)

        # Info Card Box
        self.info_card = QFrame()
        self.info_card.setStyleSheet("""
            QFrame {
                background-color: #1F2026;
                border: 1px solid #2E303A;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        card_layout = QVBoxLayout(self.info_card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(4)

        self.lbl_dims = QLabel()
        self.lbl_dims.setStyleSheet("font-weight: bold; color: #38BDF8; font-size: 13px;")
        self.lbl_specs = QLabel()
        self.lbl_specs.setStyleSheet("color: #A1A1AA; font-size: 12px;")
        self.lbl_purpose = QLabel()
        self.lbl_purpose.setStyleSheet("color: #71717A; font-size: 11px;")

        card_layout.addWidget(self.lbl_dims)
        card_layout.addWidget(self.lbl_specs)
        card_layout.addWidget(self.lbl_purpose)

        layout.addWidget(self.info_card)
        self._update_card()

    def _on_combo_changed(self, index: int) -> None:
        self.current_preset_id = self.combo.itemData(index)
        self._update_card()
        self.preset_changed.emit(self.current_preset_id, self.get_current_spec())

    def _update_card(self) -> None:
        spec = self.get_current_spec()
        w_mm = spec.get("width_mm", 30)
        h_mm = spec.get("height_mm", 40)
        w_px = spec.get("width_px", 354)
        h_px = spec.get("height_px", 472)
        head_min = int(spec.get("min_face_height_pct", 0.7) * 100)
        head_max = int(spec.get("max_face_height_pct", 0.75) * 100)
        eye_pct = int(spec.get("eye_level_ratio_from_bottom", 0.58) * 100)
        purpose = spec.get("purpose", "")

        self.lbl_dims.setText(f"📐 {w_mm:.1f} x {h_mm:.1f} mm ({w_px} x {h_px} px @ 300 DPI)")
        self.lbl_specs.setText(f"👤 Đầu: {head_min}% - {head_max}% | 👁 Mắt: {eye_pct}% từ đáy")
        self.lbl_purpose.setText(f"ℹ️ {purpose}")

    def get_current_spec(self) -> Dict[str, Any]:
        return self.presets.get(self.current_preset_id, {})

    def set_preset(self, preset_id: str) -> None:
        idx = self.combo.findData(preset_id)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)


class ColorPickerWidget(QWidget):
    """Background color picker with ID studio presets and custom RGB dialog."""

    color_changed = pyqtSignal(str)

    SWATCHES = [
        ("#0090FF", "Xanh chuẩn (Official Blue)"),
        ("#FFFFFF", "Trắng (Pure White)"),
        ("#1E60AA", "Xanh đậm (Deep Blue)"),
        ("#E2E8F0", "Xám nhạt (Off-White)"),
        ("#E11D48", "Đỏ thẻ (Red ID)")
    ]

    def __init__(self, default_color: str = "#0090FF", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_color = default_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        swatch_layout = QHBoxLayout()
        swatch_layout.setSpacing(6)

        self.buttons: List[QPushButton] = []
        for hex_code, title in self.SWATCHES:
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setToolTip(f"{title} ({hex_code})")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_code};
                    border: 2px solid #3F3F46;
                    border-radius: 14px;
                }}
                QPushButton:hover {{
                    border: 2px solid #FFFFFF;
                }}
            """)
            btn.clicked.connect(lambda checked, c=hex_code: self.set_color(c))
            swatch_layout.addWidget(btn)
            self.buttons.append(btn)

        # Custom Color Button
        btn_custom = QPushButton("🎨 Tự chọn...")
        btn_custom.setFixedHeight(28)
        btn_custom.clicked.connect(self._choose_custom_color)
        swatch_layout.addWidget(btn_custom)
        swatch_layout.addStretch()

        layout.addLayout(swatch_layout)

        # Active Color Status
        status_layout = QHBoxLayout()
        self.preview_chip = QFrame()
        self.preview_chip.setFixedSize(20, 20)
        self.preview_chip.setStyleSheet(f"border-radius: 4px; background-color: {self.current_color}; border: 1px solid #52525B;")
        self.lbl_hex = QLabel(self.current_color)
        self.lbl_hex.setStyleSheet("font-family: monospace; font-size: 12px; color: #A1A1AA;")

        status_layout.addWidget(self.preview_chip)
        status_layout.addWidget(self.lbl_hex)
        status_layout.addStretch()
        layout.addLayout(status_layout)

    def set_color(self, hex_code: str) -> None:
        self.current_color = hex_code.upper()
        self.preview_chip.setStyleSheet(f"border-radius: 4px; background-color: {self.current_color}; border: 1px solid #52525B;")
        self.lbl_hex.setText(self.current_color)
        self.color_changed.emit(self.current_color)

    def _choose_custom_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.current_color), self, "Chọn màu nền ID")
        if c.isValid():
            self.set_color(c.name())


class AdjustmentSlidersWidget(QWidget):
    """Fine-tuning sliders for manual adjustment of scale, shift, tilt, and feathering."""

    values_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 1. Face Scale slider (-15% to +15%)
        self.slider_scale, self.lbl_scale = self._create_slider_row("Tỉ lệ đầu / Zoom", -15, 15, 0, "%", layout)
        # 2. Vertical Shift (-50px to +50px)
        self.slider_y, self.lbl_y = self._create_slider_row("Vị trí đứng (Y)", -50, 50, 0, " px", layout)
        # 3. Horizontal Shift (-50px to +50px)
        self.slider_x, self.lbl_x = self._create_slider_row("Vị trí ngang (X)", -50, 50, 0, " px", layout)
        # 4. Rotation offset (-10.0 deg to +10.0 deg, step 0.1)
        self.slider_rot, self.lbl_rot = self._create_slider_row("Xoay góc (Tilt)", -100, 100, 0, "°", layout, divisor=10.0)
        # 5. Mask Feather (0 to 50, step 0.1)
        self.slider_feather, self.lbl_feather = self._create_slider_row("Độ mịn viền tóc", 0, 50, 10, " px", layout, divisor=10.0)

        # Retouching Group Divider
        lbl_retouch = QLabel("CHỈNH SỬA KHUÔN MẶT (RETOUCH)")
        lbl_retouch.setStyleSheet("color: #00A6FF; font-weight: bold; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_retouch)

        # 6. Skin Smoothing / Blemish removal (0% to 100%)
        self.slider_smooth, self.lbl_smooth = self._create_slider_row("Làm mịn da & Xóa mụn", 0, 100, 0, "%", layout)
        # 7. Sharpening (0% to 100%)
        self.slider_sharpen, self.lbl_sharpen = self._create_slider_row("Làm nét chi tiết (Sharp)", 0, 100, 0, "%", layout)

        # Reset button
        btn_reset = QPushButton("↺ Đặt lại mặc định (Reset)")
        btn_reset.setFixedHeight(28)
        btn_reset.clicked.connect(self.reset_values)
        layout.addWidget(btn_reset)

    def _create_slider_row(
        self,
        name: str,
        min_v: int,
        max_v: int,
        default_v: int,
        unit: str,
        parent_layout: QVBoxLayout,
        divisor: float = 1.0
    ) -> Tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        lbl_title = QLabel(name)
        lbl_title.setStyleSheet("color: #D4D4D8; font-size: 12px;")

        val_init = default_v / divisor
        lbl_val = QLabel(f"{val_init:+.1f}{unit}" if divisor > 1 else f"{default_v:+d}{unit}")
        lbl_val.setStyleSheet("color: #38BDF8; font-size: 12px; font-weight: bold; min-width: 50px;")
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(lbl_title)
        row.addStretch()
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default_v)

        def on_change(val: int) -> None:
            f_val = val / divisor
            sign = "+" if f_val > 0 and min_v < 0 else ""
            if divisor > 1:
                lbl_val.setText(f"{sign}{f_val:.1f}{unit}")
            else:
                lbl_val.setText(f"{sign}{val}{unit}")
            self.values_changed.emit()

        slider.valueChanged.connect(on_change)
        parent_layout.addWidget(slider)
        return slider, lbl_val

    def get_values(self) -> Dict[str, float]:
        return {
            "scale_offset": self.slider_scale.value() / 100.0,
            "x_shift_px": float(self.slider_x.value()),
            "y_shift_px": float(self.slider_y.value()),
            "rotation_offset_deg": self.slider_rot.value() / 10.0,
            "feather_radius": self.slider_feather.value() / 10.0,
            "skin_smooth_pct": self.slider_smooth.value() / 100.0,
            "sharpen_pct": self.slider_sharpen.value() / 100.0,
        }

    def reset_values(self) -> None:
        self.slider_scale.setValue(0)
        self.slider_x.setValue(0)
        self.slider_y.setValue(0)
        self.slider_rot.setValue(0)
        self.slider_feather.setValue(10)
        self.slider_smooth.setValue(0)
        self.slider_sharpen.setValue(0)
        self.values_changed.emit()


class InteractiveCanvas(QWidget):
    """
    High-performance zoomable/pannable canvas for previewing ID photo,
    toggling biometric guide lines, drag-and-drop import, and interactive eraser brush.
    """

    file_dropped = pyqtSignal(str)
    brush_started = pyqtSignal(float, float, int, str)  # img_x, img_y, radius, mode
    brush_painted = pyqtSignal(float, float, int, str)  # img_x, img_y, radius, mode
    brush_ended = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 500)

        self.pixmap: Optional[QPixmap] = None
        self.show_guides: bool = True
        self.show_mask_only: bool = False
        self.guidelines_data: Optional[Dict[str, Any]] = None

        # Tool state: 'pan', 'eraser', 'restore'
        self.tool_mode: str = "pan"
        self.brush_radius: int = 15
        self._is_brushing: bool = False
        self._mouse_pos: QPointF = QPointF(-100, -100)

        # Transform state
        self.zoom_factor: float = 1.0
        self.pan_offset = QPointF(0, 0)
        self._drag_start = QPointF()
        self._is_panning = False

    def set_tool_mode(self, mode: str) -> None:
        """Set active canvas tool: 'pan', 'eraser', or 'restore'."""
        self.tool_mode = mode
        if mode == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius = max(2, min(radius, 100))
        self.update()

    def set_image_and_guides(
        self,
        composite_rgb: Optional[np.ndarray],
        guidelines_data: Optional[Dict[str, Any]] = None,
        mask_array: Optional[np.ndarray] = None
    ) -> None:
        """Set preview image and guideline landmarks."""
        self.guidelines_data = guidelines_data

        if composite_rgb is None:
            self.pixmap = None
            self.update()
            return

        if self.show_mask_only and mask_array is not None:
            # Render grayscale mask view
            h, w = mask_array.shape[:2]
            qimg = QImage(mask_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = composite_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(composite_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self.pixmap = QPixmap.fromImage(qimg)
        self.update()

    def fit_in_view(self) -> None:
        """Fit image inside canvas area."""
        if not self.pixmap or self.pixmap.isNull():
            return

        canvas_w = self.width() - 40
        canvas_h = self.height() - 40
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()

        scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))
        self.zoom_factor = max(0.2, min(scale, 3.0))
        # Center image
        dest_w = img_w * self.zoom_factor
        dest_h = img_h * self.zoom_factor
        self.pan_offset = QPointF((self.width() - dest_w) / 2.0, (self.height() - dest_h) / 2.0)
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Canvas Background (Dark Studio Slate)
        painter.fillRect(self.rect(), QColor("#121316"))

        if not self.pixmap or self.pixmap.isNull():
            # Empty state / Dropzone UI
            painter.setPen(QPen(QColor("#3F3F46"), 2, Qt.PenStyle.DashLine))
            drop_rect = QRectF(30, 30, self.width() - 60, self.height() - 60)
            painter.drawRoundedRect(drop_rect, 12, 12)

            painter.setPen(QColor("#71717A"))
            f = painter.font()
            f.setPointSize(14)
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "📸 Kéo thả ảnh chân dung vào đây\nhoặc nhấn 'Mở ảnh'")
            return

        # Draw checkerboard pattern behind transparent photos
        img_w = self.pixmap.width() * self.zoom_factor
        img_h = self.pixmap.height() * self.zoom_factor
        img_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), img_w, img_h)

        # Draw subtle drop shadow around photo card
        shadow_rect = img_rect.adjusted(-2, -2, 2, 2)
        painter.setPen(QPen(QColor("#000000"), 3))
        painter.drawRect(shadow_rect)

        # Draw Image
        painter.drawPixmap(img_rect.toRect(), self.pixmap)

        # Draw Guide Lines Overlay if enabled
        if self.show_guides and self.guidelines_data:
            self._draw_guidelines(painter, img_rect)

        # Draw Brush Circle Cursor if in Eraser or Restore mode
        if self.tool_mode in ("eraser", "restore") and self._mouse_pos.x() >= 0:
            screen_r = self.brush_radius * self.zoom_factor
            if self.tool_mode == "eraser":
                painter.setPen(QPen(QColor("#EF4444"), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(239, 68, 68, 35))
            else:
                painter.setPen(QPen(QColor("#10B981"), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(16, 185, 129, 35))
            painter.drawEllipse(self._mouse_pos, screen_r, screen_r)

    def _draw_guidelines(self, painter: QPainter, img_rect: QRectF) -> None:
        """Render precise biometric guide markers over the canvas."""
        scale = self.zoom_factor
        ox = self.pan_offset.x()
        oy = self.pan_offset.y()
        w = img_rect.width()

        eye_y = oy + float(self.guidelines_data.get("eye_y", 0)) * scale
        crown_y = oy + float(self.guidelines_data.get("crown_y", 0)) * scale
        chin_y = oy + float(self.guidelines_data.get("chin_y", 0)) * scale
        center_x = ox + float(self.guidelines_data.get("center_x", 0)) * scale

        font = QFont("-apple-system", 9)
        font.setBold(True)
        painter.setFont(font)

        # 1. Crown line (Top of Head)
        painter.setPen(QPen(QColor("#F59E0B"), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(ox, crown_y), QPointF(ox + w, crown_y))
        painter.drawText(QPointF(ox + 8, crown_y - 4), "ĐỈNH ĐẦU (CROWN)")

        # 2. Eye level line
        painter.setPen(QPen(QColor("#EC4899"), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(ox, eye_y), QPointF(ox + w, eye_y))
        painter.drawText(QPointF(ox + 8, eye_y - 4), "ĐƯỜNG MẮT (EYES)")

        # 3. Chin line
        painter.setPen(QPen(QColor("#06B6D4"), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(ox, chin_y), QPointF(ox + w, chin_y))
        painter.drawText(QPointF(ox + 8, chin_y + 14), "CẰM (CHIN)")

        # 4. Vertical axis
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(center_x, oy), QPointF(center_x, oy + img_rect.height()))

    # Mouse & Gesture Handlers
    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        zoom_step = 1.15 if delta > 0 else 0.85

        # Zoom towards cursor
        mouse_pos = event.position()
        old_zoom = self.zoom_factor
        new_zoom = max(0.2, min(old_zoom * zoom_step, 5.0))
        self.zoom_factor = new_zoom

        # Adjust pan to keep cursor position fixed
        factor = new_zoom / old_zoom
        self.pan_offset = mouse_pos - (mouse_pos - self.pan_offset) * factor
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()
        if self.tool_mode in ("eraser", "restore") and event.button() == Qt.MouseButton.LeftButton:
            self._is_brushing = True
            if self.pixmap and not self.pixmap.isNull():
                scale = self.zoom_factor
                img_x = (self._mouse_pos.x() - self.pan_offset.x()) / scale
                img_y = (self._mouse_pos.y() - self.pan_offset.y()) / scale
                self.brush_started.emit(img_x, img_y, self.brush_radius, self.tool_mode)
        elif event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True
            self._drag_start = event.position() - self.pan_offset

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()
        if self._is_brushing:
            self._apply_brush_at(event.position())
        elif self._is_panning:
            self.pan_offset = event.position() - self._drag_start
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_brushing:
            self._is_brushing = False
            self.brush_ended.emit()
        self._is_panning = False
        self.update()

    def leaveEvent(self, event: Any) -> None:
        self._mouse_pos = QPointF(-100, -100)
        self.update()

    def _apply_brush_at(self, pos: QPointF) -> None:
        if not self.pixmap or self.pixmap.isNull():
            return
        # Convert canvas coordinate to image pixel coordinate
        scale = self.zoom_factor
        img_x = (pos.x() - self.pan_offset.x()) / scale
        img_y = (pos.y() - self.pan_offset.y()) / scale
        self.brush_painted.emit(img_x, img_y, self.brush_radius, self.tool_mode)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
                self.file_dropped.emit(file_path)
                break
        event.acceptProposedAction()
