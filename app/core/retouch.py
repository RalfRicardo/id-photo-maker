"""Facial retouching: skin smoothing, blemish softening, and smart sharpening."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FaceRetoucher:
    """Provides natural skin smoothing, blemish reduction, and detail sharpening."""

    @staticmethod
    def smooth_skin(
        rgb_image: np.ndarray,
        strength_pct: float = 0.0,
        face_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply natural skin smoothing and blemish softening using edge-preserving Bilateral Filter.
        Preserves eyes, lips, eyebrows, and nostrils while softening pores and blemishes.

        Args:
            rgb_image: uint8 numpy array (H, W, 3).
            strength_pct: Float from 0.0 to 1.0 (0% = no effect, 100% = maximum smoothing).
            face_mask: Optional grayscale mask (H, W) restricting smoothing to the face skin area.

        Returns:
            Smoothed uint8 RGB image.
        """
        if strength_pct <= 0.001:
            return rgb_image

        strength = np.clip(strength_pct, 0.0, 1.0)
        h, w = rgb_image.shape[:2]

        # Bilateral filter parameters scaled by strength
        # d: diameter of pixel neighborhood (5 to 11)
        # sigmaColor: color distance (softens blemishes: 20 to 80)
        # sigmaSpace: coordinate space distance (20 to 80)
        d = int(round(5 + strength * 6))
        if d % 2 == 0:
            d += 1
        sigma_color = 20.0 + strength * 60.0
        sigma_space = 20.0 + strength * 60.0

        # Bilateral filter runs in RGB/BGR
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        smoothed_bgr = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        smoothed_rgb = cv2.cvtColor(smoothed_bgr, cv2.COLOR_BGR2RGB)

        # High-pass skin texture preservation to prevent waxy look
        # Add subtle original fine grain back
        diff = cv2.subtract(rgb_image, smoothed_rgb)
        smoothed_rgb = cv2.addWeighted(smoothed_rgb, 1.0, diff, 0.15, 0)

        # If a face skin mask is provided, blend only inside the face region
        if face_mask is not None:
            mask_f = (face_mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * strength
            result = rgb_image.astype(np.float32) * (1.0 - mask_f) + smoothed_rgb.astype(np.float32) * mask_f
            return np.clip(result, 0, 255).astype(np.uint8)

        # Default: blend whole image with strength factor
        result = cv2.addWeighted(rgb_image, 1.0 - strength, smoothed_rgb, strength, 0)
        return result

    @staticmethod
    def sharpen(
        rgb_image: np.ndarray,
        amount_pct: float = 0.0,
        radius: float = 1.0
    ) -> np.ndarray:
        """
        Apply professional unsharp mask (USM) detail sharpening.
        Enhances eyes, eyelashes, hair definition, and suit texture.

        Args:
            rgb_image: uint8 numpy array (H, W, 3).
            amount_pct: Float from 0.0 to 1.0.
            radius: Gaussian blur radius in pixels.

        Returns:
            Sharpened uint8 RGB image.
        """
        if amount_pct <= 0.001:
            return rgb_image

        amount = np.clip(amount_pct, 0.0, 1.5)
        # Convert to LAB to sharpen only the Luminance (L) channel, preventing color fringing
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Unsharp mask on Luminance
        blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=radius, sigmaY=radius)
        sharpened_l = cv2.addWeighted(l_channel, 1.0 + amount, blurred, -amount, 0)

        merged_lab = cv2.merge([sharpened_l, a_channel, b_channel])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def apply_eraser_brush(
        alpha_mask: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
        mode: str = "erase"  # "erase" or "restore"
    ) -> np.ndarray:
        """
        Interactively modify the alpha mask at (center_x, center_y) with a circular brush.
        - "erase": sets alpha to 0 (removes stray hair, background leaks, or shoulder artifacts).
        - "restore": sets alpha to 255 (recovers clipped hair or garment edges).

        Args:
            alpha_mask: Grayscale uint8 array (H, W).
            center_x, center_y: Pixel coordinate center of the brush.
            radius: Radius in pixels.
            mode: 'erase' to carve out (0) or 'restore' to reveal (255).

        Returns:
            Modified uint8 alpha mask.
        """
        val = 0 if mode == "erase" else 255
        h, w = alpha_mask.shape[:2]

        # Draw circle on mask with anti-aliasing
        mask_copy = alpha_mask.copy()
        cv2.circle(mask_copy, (int(center_x), int(center_y)), int(radius), val, -1, lineType=cv2.LINE_AA)
        return mask_copy
