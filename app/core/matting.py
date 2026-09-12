"""Background removal and alpha matte extraction module."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import rembg


class MattingEngine:
    """Performs neural background segmentation and alpha matte extraction."""

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session = None
        self._init_session()

    def _init_session(self) -> None:
        """Initialize the ONNX rembg session."""
        try:
            self._session = rembg.new_session(self.model_name)
        except Exception as err:
            print(f"[MattingEngine] Error initializing session for {self.model_name}: {err}")
            # Try fallback to u2netp if u2net fails
            try:
                self._session = rembg.new_session("u2netp")
                self.model_name = "u2netp"
            except Exception as err2:
                print(f"[MattingEngine] Fallback initialization failed: {err2}")
                self._session = None

    def extract_alpha_matte(
        self,
        rgb_image: np.ndarray,
        feather_radius: float = 1.0,
        threshold: int = 0
    ) -> np.ndarray:
        """
        Extract grayscale alpha matte (0 to 255) from an RGB image.

        Args:
            rgb_image: uint8 numpy array of shape (H, W, 3).
            feather_radius: Optional Gaussian feathering applied to edges.
            threshold: Minimum alpha cutoff threshold (0 = keep soft gradients).

        Returns:
            uint8 numpy array of shape (H, W) where 255 is foreground, 0 is background.
        """
        h, w = rgb_image.shape[:2]

        if self._session is not None:
            try:
                pil_img = Image.fromarray(rgb_image)
                # only_mask=True returns a grayscale PIL Image (L mode)
                mask_pil = rembg.remove(
                    pil_img,
                    session=self._session,
                    only_mask=True,
                    post_process_mask=False
                )
                alpha = np.array(mask_pil, dtype=np.uint8)
            except Exception as e:
                print(f"[MattingEngine] Error during rembg execution: {e}")
                alpha = self._fallback_matte(rgb_image)
        else:
            alpha = self._fallback_matte(rgb_image)

        # Thresholding if requested
        if threshold > 0:
            _, alpha = cv2.threshold(alpha, threshold, 255, cv2.THRESH_TOZERO)

        # Edge feathering / smoothing if radius > 0
        if feather_radius > 0:
            k = int(math_odd(feather_radius * 2 + 1))
            blurred = cv2.GaussianBlur(alpha, (k, k), feather_radius)
            # Only blend near edge transitions to preserve solid core
            edge_region = cv2.Canny(alpha, 50, 200)
            edge_region = cv2.dilate(edge_region, np.ones((k, k), np.uint8))
            alpha = np.where(edge_region > 0, blurred, alpha)

        return alpha

    def _fallback_matte(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback segmentation using GrabCut or elliptical portrait mask."""
        h, w = rgb_image.shape[:2]
        # Generate elliptical foreground mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, int(h * 0.45))
        axes = (int(w * 0.38), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        # Add shoulder taper
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

        Args:
            rgb_image: uint8 array (H, W, 3)
            alpha_mask: uint8 array (H, W) in range [0, 255]
            bg_color: Hex string (e.g., "#0090FF") or RGB tuple (r, g, b).

        Returns:
            Composited RGB uint8 image (H, W, 3).
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
