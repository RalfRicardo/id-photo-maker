"""Adobe Photoshop (.psd) generator preserving layers, alpha masks, and UTF-8 names."""

from __future__ import annotations

import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytoshop
from pytoshop import enums, image_resources, layers, tagged_block, util

from app.core.matting import hex_to_rgb, MattingEngine


class EmptyLayerMask(layers.LayerMask):
    """Custom LayerMask that serializes 0 extra bytes when no mask is present."""

    def total_length(self, header: Any) -> int:
        return 4

    def write(self, fd: Any, header: Any) -> None:
        util.write_value(fd, "I", 0)


def create_resolution_info_block(dpi: int = 300) -> image_resources.GenericImageResourceBlock:
    """
    Create a Photoshop Image Resource Block 0x03ED (ResolutionInfo) setting the DPI.
    Format:
    - hRes: 32-bit fixed point (dpi << 16)
    - hResUnit: 1 (pixels/inch)
    - widthUnit: 1 (inches)
    - vRes: 32-bit fixed point (dpi << 16)
    - vResUnit: 1 (pixels/inch)
    - heightUnit: 1 (inches)
    """
    data = struct.pack(">IHHIHH", dpi << 16, 1, 1, dpi << 16, 1, 1)
    return image_resources.GenericImageResourceBlock(resource_id=0x03ED, data=data)


def create_guidelines_overlay(
    width: int,
    height: int,
    eye_y: float,
    crown_y: float,
    chin_y: float,
    center_x: float,
    preset_name: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an RGBA overlay containing visual guideline markers for ID validation.
    Returns: (R, G, B, Alpha) channels each of shape (H, W).
    """
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Colors (B, G, R, A) for OpenCV
    # 1. Crown line (Top of head target) - Golden Orange
    c_y = int(np.clip(round(crown_y), 0, height - 1))
    cv2.line(rgba, (0, c_y), (width, c_y), (0, 165, 255, 220), 1, cv2.LINE_AA)
    cv2.putText(rgba, "Dinh Dau (Crown)", (10, max(15, c_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255, 240), 1, cv2.LINE_AA)

    # 2. Eye level line - Magenta
    e_y = int(np.clip(round(eye_y), 0, height - 1))
    cv2.line(rgba, (0, e_y), (width, e_y), (255, 0, 255, 220), 1, cv2.LINE_AA)
    cv2.putText(rgba, "Duong Mat (Eye Level)", (10, max(15, e_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255, 240), 1, cv2.LINE_AA)

    # 3. Chin line - Cyan
    ch_y = int(np.clip(round(chin_y), 0, height - 1))
    cv2.line(rgba, (0, ch_y), (width, ch_y), (255, 255, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(rgba, "Cam (Chin)", (10, min(height - 6, ch_y + 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0, 240), 1, cv2.LINE_AA)

    # 4. Vertical center axis - Dotted White
    cx = int(np.clip(round(center_x), 0, width - 1))
    for y in range(0, height, 8):
        cv2.line(rgba, (cx, y), (cx, min(height - 1, y + 4)), (255, 255, 255, 180), 1)

    # Convert to RGB + Alpha
    r = rgba[:, :, 0]
    g = rgba[:, :, 1]
    b = rgba[:, :, 2]
    a = rgba[:, :, 3]

    return r, g, b, a


class PSDBuilder:
    """Builder for professional multi-layer ID photo PSDs."""

    def __init__(self, compression: enums.Compression = enums.Compression.rle):
        # Check if RLE compression is working in this environment
        try:
            from pytoshop import packbits  # noqa: F401
            self.compression = compression
        except (ImportError, NameError):
            self.compression = enums.Compression.raw

    def export_id_photo(
        self,
        rgb_portrait: np.ndarray,
        alpha_mask: np.ndarray,
        output_path: str,
        bg_color: Union[str, Tuple[int, int, int]] = "#0090FF",
        guidelines_info: Optional[Dict[str, Any]] = None,
        dpi: int = 300
    ) -> str:
        """
        Assemble and export standard multi-layer ID Photo PSD:
        - Layer 1 (Bottom): "Nền (Background)" - Solid Color Fill
        - Layer 2: "Chủ thể (Portrait)" - Original RGB image with attached user Layer Mask
        - Layer 3 (Top, Hidden): "Khung căn chuẩn (Guide lines)" - Verification overlay
        """
        h, w = rgb_portrait.shape[:2]
        if isinstance(bg_color, str):
            bg_rgb = hex_to_rgb(bg_color)
        else:
            bg_rgb = bg_color

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # -------------------------------------------------------------
        # 1. Layer 1: "Nền (Background)"
        # -------------------------------------------------------------
        bg_r = np.full((h, w), bg_rgb[0], dtype=np.uint8)
        bg_g = np.full((h, w), bg_rgb[1], dtype=np.uint8)
        bg_b = np.full((h, w), bg_rgb[2], dtype=np.uint8)

        layer_bg = layers.LayerRecord(
            top=0, left=0, bottom=h, right=w,
            name="Nen (Background)",
            blocks=[tagged_block.UnicodeLayerName("Nền (Background)")]
        )
        layer_bg.channels[enums.ChannelId.red] = layers.ChannelImageData(image=bg_r, compression=self.compression)
        layer_bg.channels[enums.ChannelId.green] = layers.ChannelImageData(image=bg_g, compression=self.compression)
        layer_bg.channels[enums.ChannelId.blue] = layers.ChannelImageData(image=bg_b, compression=self.compression)
        layer_bg._mask = EmptyLayerMask()

        # -------------------------------------------------------------
        # 2. Layer 2: "Chủ thể (Portrait)" WITH attached Layer Mask
        # -------------------------------------------------------------
        p_r = np.ascontiguousarray(rgb_portrait[:, :, 0], dtype=np.uint8)
        p_g = np.ascontiguousarray(rgb_portrait[:, :, 1], dtype=np.uint8)
        p_b = np.ascontiguousarray(rgb_portrait[:, :, 2], dtype=np.uint8)
        mask_data = np.ascontiguousarray(alpha_mask, dtype=np.uint8)

        layer_portrait = layers.LayerRecord(
            top=0, left=0, bottom=h, right=w,
            name="Chu the (Portrait)",
            blocks=[tagged_block.UnicodeLayerName("Chủ thể (Portrait)")]
        )
        layer_portrait.channels[enums.ChannelId.red] = layers.ChannelImageData(image=p_r, compression=self.compression)
        layer_portrait.channels[enums.ChannelId.green] = layers.ChannelImageData(image=p_g, compression=self.compression)
        layer_portrait.channels[enums.ChannelId.blue] = layers.ChannelImageData(image=p_b, compression=self.compression)

        # Non-destructive layer mask
        layer_portrait.mask = layers.LayerMask(
            top=0, left=0, bottom=h, right=w,
            default_color=0,
            layer_mask_disabled=False
        )
        layer_portrait.channels[enums.ChannelId.user_layer_mask] = layers.ChannelImageData(
            image=mask_data,
            compression=self.compression
        )

        # -------------------------------------------------------------
        # 3. Layer 3: "Khung căn chuẩn (Guide lines)" (Hidden by default)
        # -------------------------------------------------------------
        if guidelines_info:
            eye_y = float(guidelines_info.get("eye_y", h * 0.42))
            crown_y = float(guidelines_info.get("crown_y", h * 0.12))
            chin_y = float(guidelines_info.get("chin_y", h * 0.85))
            center_x = float(guidelines_info.get("center_x", w / 2.0))
        else:
            eye_y = h * 0.42
            crown_y = h * 0.12
            chin_y = h * 0.85
            center_x = w / 2.0

        g_r, g_g, g_b, g_a = create_guidelines_overlay(w, h, eye_y, crown_y, chin_y, center_x)

        layer_guides = layers.LayerRecord(
            top=0, left=0, bottom=h, right=w,
            visible=False,  # Hidden by default
            name="Khung can chuan (Guide lines)",
            blocks=[tagged_block.UnicodeLayerName("Khung căn chuẩn (Guide lines)")]
        )
        layer_guides.channels[enums.ChannelId.red] = layers.ChannelImageData(image=g_r, compression=self.compression)
        layer_guides.channels[enums.ChannelId.green] = layers.ChannelImageData(image=g_g, compression=self.compression)
        layer_guides.channels[enums.ChannelId.blue] = layers.ChannelImageData(image=g_b, compression=self.compression)
        layer_guides.channels[enums.ChannelId.transparency] = layers.ChannelImageData(image=g_a, compression=self.compression)
        layer_guides._mask = EmptyLayerMask()

        # -------------------------------------------------------------
        # Assemble PsdFile
        # -------------------------------------------------------------
        psd = pytoshop.core.PsdFile(
            num_channels=3,
            height=h,
            width=w,
            depth=8,
            color_mode=enums.ColorMode.rgb
        )

        # Set 300 DPI metadata
        psd.image_resources.blocks.append(create_resolution_info_block(dpi))

        # Layers in bottom-to-top order
        psd.layer_and_mask_info.layer_info.layer_records = [
            layer_bg,
            layer_portrait,
            layer_guides
        ]

        # Composite preview for external viewers
        composite_rgb = MattingEngine.apply_matte(rgb_portrait, alpha_mask, bg_rgb)
        psd.image_data.image = np.ascontiguousarray(
            np.stack([composite_rgb[:, :, 0], composite_rgb[:, :, 1], composite_rgb[:, :, 2]], axis=0)
        )

        with open(output_path, "wb") as f:
            psd.write(f)

        return output_path

    def export_print_sheet(
        self,
        sheet_spec: Dict[str, Any],
        photo_crops: Dict[str, Tuple[np.ndarray, np.ndarray, str]],
        output_path: str,
        dpi: int = 300
    ) -> str:
        """
        Create a print sheet PSD (e.g., 10x15 cm @ 300DPI) tiling multiple photos with cut lines.

        Args:
            sheet_spec: Configuration dictionary from photo_specs.json.
            photo_crops: Dictionary mapping preset_id to (cropped_rgb, alpha_mask, bg_color).
            output_path: Destination file path.
            dpi: Output resolution.
        """
        sheet_w = int(sheet_spec.get("width_px", 1772))
        sheet_h = int(sheet_spec.get("height_px", 1181))
        layout_type = sheet_spec.get("layout_type", "grid")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 1. Base Paper Background (Pure White)
        bg_r = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)
        bg_g = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)
        bg_b = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

        layer_paper = layers.LayerRecord(
            top=0, left=0, bottom=sheet_h, right=sheet_w,
            name="Giay in (Paper)",
            blocks=[tagged_block.UnicodeLayerName("Nền giấy in (Paper Background)")]
        )
        layer_paper.channels[enums.ChannelId.red] = layers.ChannelImageData(image=bg_r, compression=self.compression)
        layer_paper.channels[enums.ChannelId.green] = layers.ChannelImageData(image=bg_g, compression=self.compression)
        layer_paper.channels[enums.ChannelId.blue] = layers.ChannelImageData(image=bg_b, compression=self.compression)
        layer_paper._mask = EmptyLayerMask()

        placed_layers: List[layers.LayerRecord] = [layer_paper]

        # Overlay canvas for cut guidelines
        cut_overlay = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)

        slots = []
        if layout_type == "combo":
            slots = sheet_spec.get("slots", [])
        elif layout_type == "grid":
            preset_id = sheet_spec.get("preset", "3x4_vn")
            rows = int(sheet_spec.get("rows", 2))
            cols = int(sheet_spec.get("cols", 4))
            mx = int(sheet_spec.get("margin_x", 80))
            my = int(sheet_spec.get("margin_y", 70))
            sx = int(sheet_spec.get("spacing_x", 50))
            sy = int(sheet_spec.get("spacing_y", 50))

            # Default sample crop size if not available
            first_crop = next(iter(photo_crops.values()), None)
            pw = first_crop[0].shape[1] if first_crop else 354
            ph = first_crop[0].shape[0] if first_crop else 472

            for r in range(rows):
                for c in range(cols):
                    x = mx + c * (pw + sx)
                    y = my + r * (ph + sy)
                    slots.append({"preset": preset_id, "x": x, "y": y})

        # Place photos into slots
        photo_idx = 1
        for slot in slots:
            pid = slot.get("preset", "3x4_vn")
            x = int(slot["x"])
            y = int(slot["y"])

            crop_tuple = photo_crops.get(pid) or next(iter(photo_crops.values()), None)
            if crop_tuple is None:
                continue

            crop_rgb, crop_alpha, crop_bg = crop_tuple
            comp_img = MattingEngine.apply_matte(crop_rgb, crop_alpha, crop_bg)
            ph, pw = comp_img.shape[:2]

            # Clip if beyond canvas
            if x + pw > sheet_w or y + ph > sheet_h:
                continue

            # Create layer for this photo
            layer_name_ascii = f"Photo {photo_idx} ({pid})"
            layer_name_utf8 = f"Ảnh {photo_idx} ({pid})"

            rec = layers.LayerRecord(
                top=y, left=x, bottom=y + ph, right=x + pw,
                name=layer_name_ascii,
                blocks=[tagged_block.UnicodeLayerName(layer_name_utf8)]
            )
            rec.channels[enums.ChannelId.red] = layers.ChannelImageData(
                image=np.ascontiguousarray(comp_img[:, :, 0]), compression=self.compression
            )
            rec.channels[enums.ChannelId.green] = layers.ChannelImageData(
                image=np.ascontiguousarray(comp_img[:, :, 1]), compression=self.compression
            )
            rec.channels[enums.ChannelId.blue] = layers.ChannelImageData(
                image=np.ascontiguousarray(comp_img[:, :, 2]), compression=self.compression
            )
            rec._mask = EmptyLayerMask()
            placed_layers.append(rec)

            # Draw fine cut guidelines / corner crop marks
            mark_len = 15
            color = (180, 180, 180, 255)  # Subtle gray
            # Top-left corner
            cv2.line(cut_overlay, (x - 6, y), (x - 6 - mark_len, y), color, 1)
            cv2.line(cut_overlay, (x, y - 6), (x, y - 6 - mark_len), color, 1)
            # Top-right corner
            cv2.line(cut_overlay, (x + pw + 6, y), (x + pw + 6 + mark_len, y), color, 1)
            cv2.line(cut_overlay, (x + pw, y - 6), (x + pw, y - 6 - mark_len), color, 1)
            # Bottom-left corner
            cv2.line(cut_overlay, (x - 6, y + ph), (x - 6 - mark_len, y + ph), color, 1)
            cv2.line(cut_overlay, (x, y + ph + 6), (x, y + ph + 6 + mark_len), color, 1)
            # Bottom-right corner
            cv2.line(cut_overlay, (x + pw + 6, y + ph), (x + pw + 6 + mark_len, y + ph), color, 1)
            cv2.line(cut_overlay, (x + pw, y + ph + 6), (x + pw, y + ph + 6 + mark_len), color, 1)

            # Subtle boundary border
            cv2.rectangle(cut_overlay, (x, y), (x + pw, y + ph), (220, 220, 220, 150), 1)

            photo_idx += 1

        # Cut Guidelines Layer
        rec_cut = layers.LayerRecord(
            top=0, left=0, bottom=sheet_h, right=sheet_w,
            name="Duong cat (Cut Guides)",
            blocks=[tagged_block.UnicodeLayerName("Đường cắt căn lề (Cut Guidelines)")]
        )
        rec_cut.channels[enums.ChannelId.red] = layers.ChannelImageData(
            image=np.ascontiguousarray(cut_overlay[:, :, 0]), compression=self.compression
        )
        rec_cut.channels[enums.ChannelId.green] = layers.ChannelImageData(
            image=np.ascontiguousarray(cut_overlay[:, :, 1]), compression=self.compression
        )
        rec_cut.channels[enums.ChannelId.blue] = layers.ChannelImageData(
            image=np.ascontiguousarray(cut_overlay[:, :, 2]), compression=self.compression
        )
        rec_cut.channels[enums.ChannelId.transparency] = layers.ChannelImageData(
            image=np.ascontiguousarray(cut_overlay[:, :, 3]), compression=self.compression
        )
        rec_cut._mask = EmptyLayerMask()
        placed_layers.append(rec_cut)

        # Build Sheet PSD
        psd = pytoshop.core.PsdFile(
            num_channels=3,
            height=sheet_h,
            width=sheet_w,
            depth=8,
            color_mode=enums.ColorMode.rgb
        )
        psd.image_resources.blocks.append(create_resolution_info_block(dpi))
        psd.layer_and_mask_info.layer_info.layer_records = placed_layers

        # Composite preview
        comp_sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
        for layer in placed_layers[1:-1]:
            t, l, b, r = layer.top, layer.left, layer.bottom, layer.right
            lr = layer.channels[enums.ChannelId.red].image
            lg = layer.channels[enums.ChannelId.green].image
            lb = layer.channels[enums.ChannelId.blue].image
            comp_sheet[t:b, l:r] = np.stack([lr, lg, lb], axis=-1)

        psd.image_data.image = np.ascontiguousarray(
            np.stack([comp_sheet[:, :, 0], comp_sheet[:, :, 1], comp_sheet[:, :, 2]], axis=0)
        )

        with open(output_path, "wb") as f:
            psd.write(f)

        return output_path
