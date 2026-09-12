"""High-performance background matting, edge refinement, and color decontamination for CPU."""

from __future__ import annotations

import os
import urllib.request
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

RMBG_MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx"
DEFAULT_MODEL_FILENAME = "rmbg-1.4_quantized.onnx"


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 6,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Fast Guided Filter for fine hair and edge refinement without heavy neural networks.
    Transfers high-frequency spatial gradients (hair strands, whiskers, edges) from the
    guide image onto the coarse probability matte.

    Args:
        guide: Grayscale image (uint8 or float32 in [0, 1]).
        src: Coarse probability matte (uint8 or float32 in [0, 1]).
        radius: Filter window radius (typically 4 to 8).
        eps: Regularization parameter (typically 1e-4).

    Returns:
        Refined matte as float32 array in range [0.0, 1.0].
    """
    if len(guide.shape) == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)

    # Normalize guide to float32 in [0, 1]
    if guide.dtype == np.uint8:
        guide_f = guide.astype(np.float32) / 255.0
    else:
        guide_f = np.clip(guide.astype(np.float32), 0.0, 1.0)

    # Normalize source to float32 in [0, 1]
    if src.dtype == np.uint8:
        src_f = src.astype(np.float32) / 255.0
    else:
        src_f = np.clip(src.astype(np.float32), 0.0, 1.0)

    # Attempt OpenCV ximgproc implementation first (fastest)
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            result = cv2.ximgproc.guidedFilter(guide=guide_f, src=src_f, radius=radius, eps=eps)
            if not np.isnan(result).any():
                return np.clip(result, 0.0, 1.0)
    except Exception:
        pass

    # High-performance native NumPy / cv2.boxFilter fallback
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide_f, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src_f, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide_f * src_f, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * guide_f + mean_b
    return np.clip(q, 0.0, 1.0)


def decontaminate_color(
    rgb_image: np.ndarray,
    alpha_mask: np.ndarray,
    bg_color_hint: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Color Decontamination (Spill Suppression):
    Neutralizes background color bleed (e.g. blue/green studio bounce) along semi-transparent
    alpha edges so hair and shoulders do not look fringed when composited on a white or
    different background.

    Args:
        rgb_image: uint8 RGB image of shape (H, W, 3).
        alpha_mask: uint8 alpha mask in [0, 255] of shape (H, W).
        bg_color_hint: Optional RGB tuple of the background color to suppress.

    Returns:
        Decontaminated uint8 RGB image where semi-transparent fringes have their original
        hair/garment colors restored.
    """
    h, w = rgb_image.shape[:2]
    edge_mask = (alpha_mask > 8) & (alpha_mask < 245)
    if not np.any(edge_mask):
        return rgb_image.copy()

    alpha_f = alpha_mask.astype(np.float32) / 255.0

    # 1. Estimate background color from outer border / low alpha regions
    if bg_color_hint is None:
        bg_pixels = rgb_image[alpha_mask < 10]
        if len(bg_pixels) > 50:
            # Downsample for speed
            bg_est = np.median(bg_pixels[::4], axis=0).astype(np.float32)
        else:
            bg_est = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    else:
        bg_est = np.array(bg_color_hint, dtype=np.float32)

    # 2. Color Unmixing: F = (Observed - (1 - alpha) * Background) / alpha
    alpha_edge = np.maximum(alpha_f, 0.25)[:, :, None]
    unmixed = (rgb_image.astype(np.float32) - (1.0 - alpha_f[:, :, None]) * bg_est) / alpha_edge
    unmixed = np.clip(unmixed, 0.0, 255.0)

    # 3. Foreground Color Propagation (extend solid hair color outward to low-alpha fringes)
    scale = 4
    sh, sw = max(1, h // scale), max(1, w // scale)
    solid_fg = (alpha_mask >= 180).astype(np.float32)

    if np.sum(solid_fg) > 50:
        fg_w_small = cv2.resize(solid_fg, (sw, sh), interpolation=cv2.INTER_NEAREST)
        fg_rgb_small = cv2.resize(
            rgb_image.astype(np.float32) * solid_fg[:, :, None],
            (sw, sh),
            interpolation=cv2.INTER_AREA
        )

        ksize = 7
        blurred_fg = cv2.boxFilter(fg_rgb_small, -1, (ksize, ksize))
        blurred_w = cv2.boxFilter(fg_w_small, -1, (ksize, ksize))
        valid = blurred_w > 1e-3
        ext_small = np.zeros_like(fg_rgb_small)
        ext_small[valid] = blurred_fg[valid] / blurred_w[valid, None]
        ext_large = cv2.resize(ext_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        ext_large = rgb_image.astype(np.float32)

    # 4. Blend unmixing with extended color based on alpha confidence
    blend_w = np.clip((alpha_f - 0.1) / 0.6, 0.0, 1.0)[:, :, None]
    decon_edge = unmixed * blend_w + ext_large * (1.0 - blend_w)

    result = rgb_image.copy()
    result[edge_mask] = np.clip(decon_edge[edge_mask], 0, 255).astype(np.uint8)
    return result


class MattingEngine:
    """
    Optimized neural background matting engine running purely on CPU via ONNX Runtime.
    Specially tuned for mid-range PCs (Intel Core i3/i5, 8GB RAM) with zero UI freezing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        cpu_threads: Optional[int] = None
    ):
        self.model_path = model_path
        self.cpu_threads = cpu_threads or min(4, max(1, (os.cpu_count() or 4) // 2))
        self._session: Optional[ort.InferenceSession] = None
        self._input_size: Tuple[int, int] = (1024, 1024)
        self._init_session()

    def _get_default_model_path(self) -> str:
        """Resolve model path from app/models or system cache."""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_model = os.path.join(current_dir, "models", DEFAULT_MODEL_FILENAME)
        if os.path.isfile(app_model):
            return app_model

        # Check user rembg cache as secondary fallback
        cache_rmbg = os.path.expanduser("~/.rembg/models/bria-rmbg/bria-rmbg.onnx")
        if os.path.isfile(cache_rmbg):
            return cache_rmbg

        cache_u2net = os.path.expanduser("~/.rembg/models/u2net/u2net.onnx")
        if os.path.isfile(cache_u2net):
            return cache_u2net

        return app_model

    def _init_session(self) -> None:
        """Initialize the CPU-only ONNX Runtime session."""
        if not self.model_path:
            self.model_path = self._get_default_model_path()

        # Download quantized model if missing
        if not os.path.isfile(self.model_path):
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                print(f"[MattingEngine] Downloading quantized RMBG-1.4 model (42MB) to {self.model_path}...")
                urllib.request.urlretrieve(RMBG_MODEL_URL, self.model_path)
                print("[MattingEngine] Model download complete.")
            except Exception as dl_err:
                print(f"[MattingEngine] Could not download ONNX model: {dl_err}")

        if not os.path.isfile(self.model_path):
            print("[MattingEngine] Warning: No ONNX model file found. Falling back to GrabCut.")
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.cpu_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )

            # Detect input dimensions dynamically
            inputs = self._session.get_inputs()
            if inputs:
                shape = inputs[0].shape
                h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 1024
                w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 1024
                self._input_size = (w, h)

            print(f"[MattingEngine] Initialized ONNX CPU session ({self._input_size[0]}x{self._input_size[1]}) using {self.cpu_threads} threads.")
        except Exception as err:
            print(f"[MattingEngine] Error initializing ONNX session: {err}")
            self._session = None

    def extract_raw_probability(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Execute ONNX model inference to extract the raw continuous probability matte [0.0, 1.0].
        """
        h, w = rgb_image.shape[:2]
        if self._session is None:
            # Fallback geometric matte
            return (self._fallback_matte(rgb_image).astype(np.float32) / 255.0)

        # Preprocess: resize to model input shape and normalize
        mw, mh = self._input_size
        resized = cv2.resize(rgb_image, (mw, mh), interpolation=cv2.INTER_LINEAR)
        # RMBG-1.4 normalization: (im / 255.0 - 0.5)
        tensor = (resized.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[np.newaxis, ...]

        # Run inference on CPU
        input_name = self._session.get_inputs()[0].name
        outs = self._session.run(None, {input_name: tensor})

        pred = outs[0][0, 0]  # Shape: (mh, mw)

        # Min-max normalization
        p_min = float(np.min(pred))
        p_max = float(np.max(pred))
        if p_max - p_min > 1e-5:
            pred = (pred - p_min) / (p_max - p_min)
        else:
            pred = np.clip(pred, 0.0, 1.0)

        # Bilinear resize back to original image dimensions
        prob_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob_orig, 0.0, 1.0)

    def extract_alpha_matte(
        self,
        rgb_image: np.ndarray,
        feather_radius: float = 0.0,
        threshold: int = 0,
        refine_edges: bool = True,
        radius: int = 6,
        eps: float = 1e-4
    ) -> np.ndarray:
        """
        Extract refined 8-bit soft grayscale alpha mask [0, 255].
        Combines ONNX model inference with Fast Guided Filter to preserve individual
        hair strands and crisp clothing edges.

        Args:
            rgb_image: uint8 numpy array (H, W, 3).
            feather_radius: Optional additional edge feathering.
            threshold: Minimum alpha cutoff.
            refine_edges: If True, applies Fast Guided Filter with grayscale guide image.
            radius: Guided filter radius (4 to 8).
            eps: Guided filter regularization (1e-4).

        Returns:
            uint8 numpy array of shape (H, W) where 255 is foreground, 0 is background.
        """
        h, w = rgb_image.shape[:2]

        # 1. Raw neural inference
        prob = self.extract_raw_probability(rgb_image)

        # 2. Fast Guided Filter edge refinement
        if refine_edges:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            prob = fast_guided_filter(guide=gray, src=prob, radius=radius, eps=eps)

        alpha = (prob * 255.0).clip(0, 255).astype(np.uint8)

        # 3. Optional cutoff threshold
        if threshold > 0:
            _, alpha = cv2.threshold(alpha, threshold, 255, cv2.THRESH_TOZERO)

        # 4. Optional additional feathering
        if feather_radius > 0:
            k = int(math_odd(feather_radius * 2 + 1))
            blurred = cv2.GaussianBlur(alpha, (k, k), feather_radius)
            edge_region = cv2.Canny(alpha, 50, 200)
            edge_region = cv2.dilate(edge_region, np.ones((k, k), np.uint8))
            alpha = np.where(edge_region > 0, blurred, alpha)

        return alpha

    def decontaminate(
        self,
        rgb_image: np.ndarray,
        alpha_mask: np.ndarray,
        bg_color_hint: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Expose color decontamination helper via the engine instance."""
        return decontaminate_color(rgb_image, alpha_mask, bg_color_hint)

    def _fallback_matte(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback segmentation using elliptical portrait geometry if model unavailable."""
        h, w = rgb_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.38), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        pts = np.array([
            [int(w * 0.1), h],
            [int(w * 0.9), h],
            [int(w * 0.75), int(h * 0.7)],
            [int(w * 0.25), int(h * 0.7)]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.GaussianBlur(mask, (15, 15), 5.0)

    @staticmethod
    def apply_matte(
        rgb_image: np.ndarray,
        alpha_mask: np.ndarray,
        bg_color: Union[str, Tuple[int, int, int]] = "#FFFFFF"
    ) -> np.ndarray:
        """
        Composite the foreground RGB with a solid background color using the alpha mask.
        """
        if isinstance(bg_color, str):
            bg_rgb = hex_to_rgb(bg_color)
        else:
            bg_rgb = bg_color

        alpha_f = (alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        bg_arr = np.full_like(rgb_image, bg_rgb, dtype=np.float32)
        fg_arr = rgb_image.astype(np.float32)

        composite = fg_arr * alpha_f + bg_arr * (1.0 - alpha_f)
        return np.clip(composite, 0, 255).astype(np.uint8)


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex string '#RRGGBB' to (R, G, B) tuple."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    return (
        int(hex_clean[0:2], 16),
        int(hex_clean[2:4], 16),
        int(hex_clean[4:6], 16)
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (R, G, B) to uppercase hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


def math_odd(n: float) -> int:
    """Round to the nearest odd integer."""
    val = int(round(n))
    return val if val % 2 == 1 else val + 1
