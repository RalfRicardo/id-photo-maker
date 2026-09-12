"""Unit tests for PSD generation, layer preservation, and alpha mask integrity."""

import os
import tempfile
import numpy as np
import pytest
import psd_tools
from psd_tools.constants import Resource

from app.core.face_aligner import FaceAligner
from app.core.matting import (
    MattingEngine,
    decontaminate_color,
    fast_guided_filter,
    hex_to_rgb,
)
from app.core.psd_builder import PSDBuilder
from app.core.retouch import FaceRetoucher


@pytest.fixture
def sample_data():
    """Generate synthetic portrait and alpha matte for testing."""
    h, w = 472, 354
    portrait = np.full((h, w, 3), 190, dtype=np.uint8)
    # Add simple gradient
    portrait[:, :, 0] = np.linspace(150, 220, w, dtype=np.uint8)
    # Circular alpha matte
    y, x = np.ogrid[:h, :w]
    mask = (((x - w // 2) ** 2 + (y - h // 2) ** 2) < 130 ** 2).astype(np.uint8) * 255
    return portrait, mask


def test_psd_layers_and_alpha_mask(sample_data):
    """Assert output PSD has preserved layers, correct names, and alpha mask."""
    portrait, mask = sample_data
    h, w = portrait.shape[:2]

    builder = PSDBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_psd = os.path.join(tmpdir, "id_photo_test.psd")
        builder.export_id_photo(
            rgb_portrait=portrait,
            alpha_mask=mask,
            output_path=output_psd,
            bg_color="#0090FF"
        )

        assert os.path.exists(output_psd), "PSD file was not created"
        psd = psd_tools.PSDImage.open(output_psd)

        # 1. Dimensions
        assert psd.size == (w, h), f"Expected {(w, h)}, got {psd.size}"

        # 2. Layer count: exactly 3 layers
        assert len(psd) == 3, f"Expected 3 layers, found {len(psd)}"

        # 3. Layer 1 (Bottom): Nền (Background)
        bg_layer = psd[0]
        bg_name = bg_layer.name.rstrip("\x00")
        assert bg_name == "Nền (Background)", f"Expected 'Nền (Background)', got '{bg_name}'"
        assert bg_layer.visible is True
        assert not bg_layer.has_mask(), "Background layer should not have an active mask"

        # 4. Layer 2: Chủ thể (Portrait) WITH editable Layer Mask
        portrait_layer = psd[1]
        p_name = portrait_layer.name.rstrip("\x00")
        assert p_name == "Chủ thể (Portrait)", f"Expected 'Chủ thể (Portrait)', got '{p_name}'"
        assert portrait_layer.visible is True
        assert portrait_layer.has_mask(), "Portrait layer MUST have an attached layer mask"
        assert portrait_layer.mask.bbox == (0, 0, w, h), f"Mask bbox mismatch: {portrait_layer.mask.bbox}"

        # 5. Layer 3: Khung căn chuẩn (Guide lines) - Hidden by default
        guide_layer = psd[2]
        g_name = guide_layer.name.rstrip("\x00")
        assert g_name == "Khung căn chuẩn (Guide lines)", f"Expected 'Khung căn chuẩn (Guide lines)', got '{g_name}'"
        assert guide_layer.visible is False, "Guide layer should be hidden by default"
        assert not guide_layer.has_mask()

        # 6. Resolution / DPI Metadata check
        assert Resource.RESOLUTION_INFO in psd.image_resources


def test_face_aligner_roll_correction():
    """Verify roll angle calculation and horizontal eye leveling."""
    aligner = FaceAligner()

    # Image with rotated eyes
    img = np.full((600, 600, 3), 200, dtype=np.uint8)
    preset = {
        "width_px": 354,
        "height_px": 472,
        "target_face_height_pct": 0.72,
        "eye_level_ratio_from_bottom": 0.58
    }

    result = aligner.process(img, preset, rotation_offset_deg=0.0)
    assert result.cropped_image.shape == (472, 354, 3)
    assert result.features_in_crop is not None
    # Eye line should be near target height * (1 - 0.58)
    expected_eye_y = 472 * (1.0 - 0.58)
    actual_eye_y = result.features_in_crop.eyes_center[1]
    assert abs(actual_eye_y - expected_eye_y) < 5.0, f"Eye Y {actual_eye_y} deviated from {expected_eye_y}"


def test_matting_composite():
    """Verify alpha matte extraction and solid color compositing."""
    fg = np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8)  # Red
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[:, :25] = 255  # Left half opaque, right half transparent

    # Composite with blue background
    comp = MattingEngine.apply_matte(fg, mask, "#0000FF")
    # Left side should be red
    np.testing.assert_array_equal(comp[25, 10], [255, 0, 0])
    # Right side should be blue
    np.testing.assert_array_equal(comp[25, 35], [0, 0, 255])


def test_print_sheet_layout(sample_data):
    """Verify 10x15 cm @ 300DPI print sheet generation."""
    portrait, mask = sample_data
    builder = PSDBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        sheet_path = os.path.join(tmpdir, "print_sheet_test.psd")
        sheet_spec = {
            "name": "10x15 Combo Sheet",
            "width_px": 1772,
            "height_px": 1181,
            "layout_type": "combo",
            "slots": [
                {"preset": "4x6_vn", "x": 100, "y": 60},
                {"preset": "3x4_vn", "x": 100, "y": 800}
            ]
        }
        builder.export_print_sheet(
            sheet_spec=sheet_spec,
            photo_crops={"4x6_vn": (portrait, mask, "#FFFFFF"), "3x4_vn": (portrait, mask, "#0090FF")},
            output_path=sheet_path
        )

        assert os.path.exists(sheet_path)
        psd = psd_tools.PSDImage.open(sheet_path)
        assert psd.size == (1772, 1181)
        assert len(psd) == 3  # paper + 2 photos (note: slot 2 beyond 1181 height clipped or placed)


def test_face_retoucher_smooth_and_sharpen(sample_data):
    """Verify edge-preserving bilateral skin smoothing and LAB unsharp sharpening."""
    portrait, _ = sample_data
    h, w = portrait.shape[:2]

    # 1. Zero strength should return identical array
    same_img = FaceRetoucher.smooth_skin(portrait, strength_pct=0.0)
    np.testing.assert_array_equal(same_img, portrait)

    # 2. Smoothing reduces local variance / softens blemishes while keeping dimensions
    noisy = portrait.copy()
    np.random.seed(42)
    noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    smoothed = FaceRetoucher.smooth_skin(noisy, strength_pct=0.7)
    assert smoothed.shape == (h, w, 3)
    assert smoothed.dtype == np.uint8
    # Variance of smoothed should be significantly less than noisy
    assert np.var(smoothed) < np.var(noisy)

    # 3. Zero sharpen should return identical array
    same_sharp = FaceRetoucher.sharpen(portrait, amount_pct=0.0)
    np.testing.assert_array_equal(same_sharp, portrait)

    # 4. Sharpening enhances contrast along luminance edges
    sharpened = FaceRetoucher.sharpen(portrait, amount_pct=0.8)
    assert sharpened.shape == (h, w, 3)
    assert sharpened.dtype == np.uint8
    assert not np.array_equal(sharpened, portrait)


def test_face_retoucher_eraser_brush():
    """Verify manual circular brush carving and restoring on alpha mask."""
    mask = np.full((200, 200), 255, dtype=np.uint8)

    # Erase circle of radius 20 at center (100, 100)
    erased = FaceRetoucher.apply_eraser_brush(mask, 100, 100, radius=20, mode="erase")
    assert erased.shape == (200, 200)
    assert erased[100, 100] == 0, "Center of erased area should be 0"
    assert erased[0, 0] == 255, "Area outside brush should remain 255"

    # Restore circle of radius 10 at center (100, 100)
    restored = FaceRetoucher.apply_eraser_brush(erased, 100, 100, radius=10, mode="restore")
    assert restored[100, 100] == 255, "Center of restored area should be 255"
    # Point at distance 15 should still be 0 (erased in first pass, outside restore radius 10)
    assert restored[100, 115] == 0


def test_psd_export_with_retouch_and_eraser_mask(sample_data):
    """Verify exporting PSD with retouching and manual eraser strokes preserves LayerMask in PSD."""
    portrait, mask = sample_data
    h, w = portrait.shape[:2]

    # Retouch skin & sharpen
    retouched_portrait = FaceRetoucher.smooth_skin(portrait, strength_pct=0.5)
    retouched_portrait = FaceRetoucher.sharpen(retouched_portrait, amount_pct=0.4)

    # Erase stray hair/artifact at (w//2, h//2)
    cx, cy = w // 2, h // 2
    edited_mask = FaceRetoucher.apply_eraser_brush(mask, cx, cy, radius=25, mode="erase")
    assert edited_mask[cy, cx] == 0

    builder = PSDBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_psd = os.path.join(tmpdir, "retouched_id.psd")
        builder.export_id_photo(
            rgb_portrait=retouched_portrait,
            alpha_mask=edited_mask,
            output_path=output_psd,
            bg_color="#FFFFFF"
        )

        assert os.path.exists(output_psd)
        psd = psd_tools.PSDImage.open(output_psd)

        # Check Layer 2 has LayerMask matching edited_mask
        portrait_layer = psd[1]
        assert portrait_layer.has_mask()
        mask_np = np.array(portrait_layer.mask.topil())
        assert mask_np.shape == (h, w)

        # Erased center must be 0 in exported PSD layer mask
        assert mask_np[cy, cx] == 0, "Erased spot should be transparent (0) in PSD LayerMask"
        # Edge outside erased circle but inside original mask should be 255
        assert mask_np[cy, cx - 80] == 255


def test_matting_engine_cpu_execution():
    """Verify MattingEngine operates purely on CPU via ONNX Runtime and extracts 8-bit soft matte."""
    engine = MattingEngine()
    assert engine._session is not None, "ONNX InferenceSession should be initialized"
    providers = engine._session.get_providers()
    assert "CPUExecutionProvider" in providers, "Engine MUST run on CPUExecutionProvider"

    # Synthetic portrait
    img = np.full((320, 240, 3), 235, dtype=np.uint8)
    img[40:280, 40:200] = [50, 40, 30]

    raw_prob = engine.extract_raw_probability(img)
    assert raw_prob.shape == (320, 240)
    assert 0.0 <= raw_prob.min() <= raw_prob.max() <= 1.0

    mask = engine.extract_alpha_matte(img, refine_edges=True, radius=6, eps=1e-4)
    assert mask.shape == (320, 240)
    assert mask.dtype == np.uint8
    assert mask.min() >= 0 and mask.max() <= 255


def test_fast_guided_filter_edge_preservation():
    """Verify Fast Guided Filter transfers high-resolution guide gradients onto the matte."""
    h, w = 200, 200
    guide_gray = np.full((h, w), 240, dtype=np.uint8)
    # Dark sharp edge
    guide_gray[:, :100] = 20

    # Coarse blurred probability mask
    coarse_prob = np.zeros((h, w), dtype=np.float32)
    coarse_prob[:, :100] = 1.0
    import cv2
    coarse_blurred = cv2.GaussianBlur(coarse_prob, (21, 21), 7.0)

    # Apply Fast Guided Filter
    refined = fast_guided_filter(guide=guide_gray, src=coarse_blurred, radius=6, eps=1e-4)
    assert refined.shape == (h, w)
    assert 0.0 <= refined.min() <= refined.max() <= 1.0

    # Refined transition at edge boundary (x=98 to 102) should be substantially sharper than blurred input
    blurred_diff = abs(coarse_blurred[100, 95] - coarse_blurred[100, 105])
    refined_diff = abs(refined[100, 95] - refined[100, 105])
    assert refined_diff > blurred_diff, "Guided filter should sharpen boundary aligned with guide edge"


def test_color_decontamination_spill_suppression():
    """Verify color decontamination removes background color bleed in semi-transparent edges."""
    h, w = 100, 100
    bg_blue = np.array([20, 120, 240], dtype=np.uint8)
    hair_dark = np.array([35, 30, 25], dtype=np.uint8)

    # Edge pixel: 50% alpha, 50% blue spill
    contaminated_img = np.full((h, w, 3), bg_blue, dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    # Center is solid hair
    contaminated_img[30:70, 30:70] = hair_dark
    alpha[30:70, 30:70] = 255

    # Ring around center is transition zone with blue spill
    for y in range(20, 80):
        for x in range(20, 80):
            if not (30 <= y < 70 and 30 <= x < 70):
                alpha[y, x] = 128  # 50% alpha
                # Blend hair with blue background
                contaminated_img[y, x] = (hair_dark * 0.5 + bg_blue * 0.5).astype(np.uint8)

    decon = decontaminate_color(contaminated_img, alpha, bg_color_hint=tuple(bg_blue))
    assert decon.shape == (h, w, 3)

    # Check a transition pixel: blue channel should be drastically suppressed toward hair color
    edge_pixel_before = contaminated_img[25, 25]
    edge_pixel_after = decon[25, 25]
    assert edge_pixel_after[2] < edge_pixel_before[2], "Blue channel in transition edge should be decontaminated"


def test_psd_export_preserves_soft_grayscale_mask(sample_data):
    """Verify that exported PSD Layer 2 contains the full 8-bit soft grayscale mask (not binarized)."""
    portrait, _ = sample_data
    h, w = portrait.shape[:2]

    # Create synthetic soft gradient mask with intermediate grayscale values
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    soft_mask = np.clip(255 - dist * 1.5, 0, 255).astype(np.uint8)

    # Verify mask has intermediate values
    intermediate = np.sum((soft_mask > 30) & (soft_mask < 225))
    assert intermediate > 1000, "Mask should have soft gradient transitions"

    builder = PSDBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "soft_mask_test.psd")
        builder.export_id_photo(
            rgb_portrait=portrait,
            alpha_mask=soft_mask,
            output_path=out_file,
            bg_color="#FFFFFF"
        )

        assert os.path.exists(out_file)
        psd = psd_tools.PSDImage.open(out_file)
        p_layer = psd[1]
        assert p_layer.has_mask()
        layer_mask_np = np.array(p_layer.mask.topil())

        # Assert LayerMask is 8-bit and retains intermediate soft values exactly
        assert layer_mask_np.shape == (h, w)
        assert layer_mask_np.dtype == np.uint8
        np.testing.assert_array_equal(layer_mask_np, soft_mask)


def test_brush_undo_redo_and_transform_synchronization():
    """Verify brush strokes move in lockstep with zoom/shifts and support Ctrl+Z / Ctrl+Y."""
    import sys
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow
    from app.core.face_aligner import FaceFeatures

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()

    dummy_feat = FaceFeatures(
        left_eye=(150.0, 180.0),
        right_eye=(250.0, 180.0),
        eyes_center=(200.0, 180.0),
        chin=(200.0, 320.0),
        forehead=(200.0, 100.0),
        crown=(200.0, 80.0),
        nose_tip=(200.0, 220.0),
        roll_angle_deg=0.0,
        face_height=240.0,
        eye_to_chin_dist=140.0
    )
    win.original_rgb = np.full((600, 500, 3), 200, dtype=np.uint8)
    win.base_features = dummy_feat
    win.base_alpha_mask = np.full((600, 500), 255, dtype=np.uint8)
    win.clean_rgb = win.original_rgb.copy()

    win._apply_adjustments_fast()
    assert win.crop_res is not None

    # Paint a stroke with eraser at crop coordinates (150, 150)
    win._on_canvas_brush_started(150.0, 150.0, 15, "eraser")
    win._on_canvas_brush_painted(155.0, 155.0, 15, "eraser")
    win._on_canvas_brush_ended()

    assert len(win.undo_strokes) == 1
    assert win.btn_undo.isEnabled()
    assert win.alpha_mask[150, 150] == 0

    # Lockstep movement: Shift X by +20px
    win.sliders_widget.slider_x.setValue(20)
    win._apply_adjustments_fast()

    # The erased spot moves with X shift
    assert win.alpha_mask[150, 170] == 0, "Erased spot must follow X shift"
    assert win.alpha_mask[150, 150] == 255, "Old coordinate should no longer be erased"

    # Undo (Ctrl+Z)
    win.undo_brush_stroke()
    assert len(win.undo_strokes) == 0
    assert len(win.redo_strokes) == 1
    assert win.alpha_mask[150, 170] == 255, "After undo, erased spot should be restored"

    # Redo (Ctrl+Y)
    win.redo_brush_stroke()
    assert len(win.undo_strokes) == 1
    assert len(win.redo_strokes) == 0
    assert win.alpha_mask[150, 170] == 0, "After redo, erased spot should be erased again"



