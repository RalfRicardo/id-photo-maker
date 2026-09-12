"""Application bootstrap for ID Photo Maker Desktop App & CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Ensure app package is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.core.face_aligner import FaceAligner
from app.core.matting import MattingEngine
from app.core.psd_builder import PSDBuilder


def run_headless(
    input_path: str,
    output_psd: str,
    preset_id: str = "3x4_vn",
    bg_color: str = "#0090FF",
    export_sheet: Optional[str] = None
) -> int:
    """Run batch or automated processing headlessly from the command line."""
    print(f"[CLI] Processing: {input_path}")
    specs_file = os.path.join(current_dir, "presets", "photo_specs.json")
    with open(specs_file, "r", encoding="utf-8") as f:
        specs_data = json.load(f)

    presets = specs_data.get("presets", {})
    if preset_id not in presets:
        print(f"[CLI] Error: Unknown preset '{preset_id}'. Available: {list(presets.keys())}")
        return 1

    spec = presets[preset_id]

    # Load input image
    stream = open(input_path, "rb")
    bytes_data = bytearray(stream.read())
    stream.close()
    bgr = cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[CLI] Error: Could not decode image file '{input_path}'")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    aligner = FaceAligner()
    matting = MattingEngine()
    builder = PSDBuilder()

    print(f"[CLI] Aligning & cropping according to {spec.get('name')}...")
    crop_res = aligner.process(rgb, spec)
    print(f"[CLI] Auto-corrected roll angle: {crop_res.features.roll_angle_deg:+.2f}°")

    print("[CLI] Extracting AI alpha matte (rembg)...")
    alpha_mask = matting.extract_alpha_matte(crop_res.cropped_image)

    print(f"[CLI] Exporting multi-layer PSD: {output_psd}...")
    fc = crop_res.features_in_crop
    guides_info = {
        "eye_y": fc.eyes_center[1],
        "crown_y": fc.crown[1],
        "chin_y": fc.chin[1],
        "center_x": crop_res.target_width / 2.0
    }
    builder.export_id_photo(
        rgb_portrait=crop_res.cropped_image,
        alpha_mask=alpha_mask,
        output_path=output_psd,
        bg_color=bg_color,
        guidelines_info=guides_info,
        dpi=300
    )
    print(f"[CLI] Success! Multi-layer PSD created at: {output_psd}")

    if export_sheet:
        print(f"[CLI] Generating print sheet (10x15 cm) at: {export_sheet}...")
        sheets = specs_data.get("print_sheets", {})
        sheet_spec = sheets.get("10x15_combo", next(iter(sheets.values())))
        # Produce crops for all presets needed by the sheet
        photo_crops = {}
        for pid in ("3x4_vn", "4x6_vn"):
            if pid in presets:
                c_res = aligner.process(rgb, presets[pid])
                c_mask = cv2.resize(alpha_mask, (c_res.target_width, c_res.target_height))
                photo_crops[pid] = (c_res.cropped_image, c_mask, bg_color)

        builder.export_print_sheet(
            sheet_spec=sheet_spec,
            photo_crops=photo_crops,
            output_path=export_sheet,
            dpi=300
        )
        print(f"[CLI] Print sheet exported to: {export_sheet}")

    return 0


def run_gui() -> int:
    """Launch the interactive PyQt6 Desktop Application."""
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow

    # Enable High-DPI scaling
    app = QApplication(sys.argv)
    app.setApplicationName("ID Photo Studio Master")
    app.setOrganizationName("PhotoStudio")

    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="ID Photo Maker - Auto align, matte, and export to multi-layer PSD.")
    parser.add_argument("--input", "-i", type=str, help="Path to input portrait image (JPG/PNG)")
    parser.add_argument("--output", "-o", type=str, help="Destination path for output .psd file")
    parser.add_argument("--preset", "-p", type=str, default="3x4_vn", help="Preset ID: 3x4_vn, 4x6_vn, 2x2_us_visa, 35x45_schengen")
    parser.add_argument("--bg", "-b", type=str, default="#0090FF", help="Background color hex code (e.g. #0090FF or #FFFFFF)")
    parser.add_argument("--sheet", "-s", type=str, default=None, help="Optional destination path to export 10x15 cm print sheet PSD")
    parser.add_argument("--headless", action="store_true", help="Run without graphical interface")

    args = parser.parse_args()

    if args.input and (args.output or args.headless):
        out = args.output or os.path.splitext(args.input)[0] + f"_{args.preset}.psd"
        return run_headless(
            input_path=args.input,
            output_psd=out,
            preset_id=args.preset,
            bg_color=args.bg,
            export_sheet=args.sheet
        )

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
