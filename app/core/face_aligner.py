"""Face alignment, rotation correction, and smart cropping module."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceFeatures:
    """Detected facial landmarks and calculated biometric metrics."""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    eyes_center: Tuple[float, float]
    chin: Tuple[float, float]
    forehead: Tuple[float, float]
    crown: Tuple[float, float]
    nose_tip: Tuple[float, float]
    roll_angle_deg: float
    face_height: float
    eye_to_chin_dist: float
    raw_landmarks: Optional[np.ndarray] = None


@dataclass
class CropResult:
    """Result of smart alignment and cropping."""
    cropped_image: np.ndarray  # RGB uint8
    transform_matrix: np.ndarray  # 2x3 affine matrix from original to cropped
    features: FaceFeatures
    features_in_crop: FaceFeatures
    target_width: int
    target_height: int
    scale: float


class FaceAligner:
    """Performs face landmark detection, eye leveling, and standard ID photo cropping."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default model location in app/models
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(current_dir, "models", "face_landmarker.task")

        self.model_path = model_path
        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize MediaPipe FaceLandmarker task."""
        if not os.path.isfile(self.model_path):
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as err:
            print(f"[FaceAligner] Warning: MediaPipe initialization failed: {err}")
            self._detector = None

    def detect_features(self, rgb_image: np.ndarray) -> Optional[FaceFeatures]:
        """Detect face features from an RGB image using MediaPipe FaceLandmarker."""
        h, w = rgb_image.shape[:2]

        if self._detector is not None:
            try:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))
                res = self._detector.detect(mp_img)

                if res.face_landmarks and len(res.face_landmarks) > 0:
                    lms = res.face_landmarks[0]
                    coords = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

                    # Extract primary landmarks
                    # Left eye pupil/center (468 or average of eye corners/contours: 33, 133, 159, 145)
                    if len(coords) > 473:
                        left_eye = (float(coords[468][0]), float(coords[468][1]))
                        right_eye = (float(coords[473][0]), float(coords[473][1]))
                    else:
                        left_eye = (float((coords[33][0] + coords[133][0]) / 2.0),
                                    float((coords[159][1] + coords[145][1]) / 2.0))
                        right_eye = (float((coords[362][0] + coords[263][0]) / 2.0),
                                     float((coords[386][1] + coords[374][1]) / 2.0))

                    chin = (float(coords[152][0]), float(coords[152][1]))
                    forehead = (float(coords[10][0]), float(coords[10][1]))
                    nose_tip = (float(coords[1][0]), float(coords[1][1]))

                    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

                    # Roll angle: angle of vector from left eye to right eye
                    dx = right_eye[0] - left_eye[0]
                    dy = right_eye[1] - left_eye[1]
                    roll_deg = math.degrees(math.atan2(dy, dx))

                    # Crown (top of hair) estimation based on skull-to-hair proportion:
                    # In anthropometry, chin-to-forehead (152 to 10) is ~75-80% of total head height.
                    forehead_to_chin = abs(chin[1] - forehead[1])
                    hair_buffer = forehead_to_chin * 0.28
                    crown_y = max(0.0, forehead[1] - hair_buffer)
                    crown = (forehead[0], crown_y)

                    face_height = abs(chin[1] - crown_y)
                    eye_to_chin = abs(chin[1] - eyes_center[1])

                    return FaceFeatures(
                        left_eye=left_eye,
                        right_eye=right_eye,
                        eyes_center=eyes_center,
                        chin=chin,
                        forehead=forehead,
                        crown=crown,
                        nose_tip=nose_tip,
                        roll_angle_deg=roll_deg,
                        face_height=face_height,
                        eye_to_chin_dist=eye_to_chin,
                        raw_landmarks=coords
                    )
            except Exception as e:
                print(f"[FaceAligner] Error in MediaPipe detection: {e}")

        # Fallback: OpenCV Haar Cascade detector
        return self._detect_features_haar(rgb_image)

    def _detect_features_haar(self, rgb_image: np.ndarray) -> Optional[FaceFeatures]:
        """Fallback face detector using OpenCV Haar Cascades or center framing."""
        h, w = rgb_image.shape[:2]
        faces = []
        try:
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(cascade_file):
                face_cascade = cv2.CascadeClassifier(cascade_file)
                if not face_cascade.empty():
                    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        except Exception:
            faces = []

        if len(faces) == 0:
            # Standard center portrait framing fallback
            fx, fy, fw, fh = int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.55)
        else:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])

        eyes_center = (fx + fw * 0.5, fy + fh * 0.38)
        left_eye = (fx + fw * 0.35, fy + fh * 0.38)
        right_eye = (fx + fw * 0.65, fy + fh * 0.38)
        chin = (fx + fw * 0.5, fy + fh * 0.95)
        forehead = (fx + fw * 0.5, fy + fh * 0.12)
        crown = (fx + fw * 0.5, max(0.0, fy - fh * 0.1))

        return FaceFeatures(
            left_eye=left_eye,
            right_eye=right_eye,
            eyes_center=eyes_center,
            chin=chin,
            forehead=forehead,
            crown=crown,
            nose_tip=(fx + fw * 0.5, fy + fh * 0.55),
            roll_angle_deg=0.0,
            face_height=abs(chin[1] - crown[1]),
            eye_to_chin_dist=abs(chin[1] - eyes_center[1]),
            raw_landmarks=None
        )

    def process(
        self,
        rgb_image: np.ndarray,
        preset_spec: Dict[str, Any],
        scale_offset: float = 0.0,
        x_shift_px: float = 0.0,
        y_shift_px: float = 0.0,
        rotation_offset_deg: float = 0.0,
        detected_features: Optional[FaceFeatures] = None
    ) -> CropResult:
        """
        Execute full alignment pipeline:
        1. Detect facial landmarks (or reuse cached detected_features).
        2. Compute roll angle and rotate image to level the eye line.
        3. Recalculate features and compute optimal scale/crop box according to preset specs.
        4. Apply shifts/offsets and produce final cropped RGB canvas.
        """
        h_orig, w_orig = rgb_image.shape[:2]
        target_w = int(preset_spec.get("width_px", 354))
        target_h = int(preset_spec.get("height_px", 472))
        target_face_pct = float(preset_spec.get("target_face_height_pct", 0.72))
        eye_ratio_from_bottom = float(preset_spec.get("eye_level_ratio_from_bottom", 0.58))

        # 1. Detect initial features
        features = detected_features if detected_features is not None else self.detect_features(rgb_image)
        if features is None:
            raise RuntimeError("No face detected in image")

        # 2. Total rotation angle (auto roll correction + manual offset)
        total_rot = -features.roll_angle_deg + rotation_offset_deg

        # Center of rotation: eye center
        cx, cy = features.eyes_center
        rot_mat = cv2.getRotationMatrix2D((cx, cy), total_rot, 1.0)

        # Transform key landmarks by rot_mat
        def transform_pt(pt: Tuple[float, float]) -> Tuple[float, float]:
            v = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
            res = rot_mat @ v
            return (float(res[0]), float(res[1]))

        rot_left_eye = transform_pt(features.left_eye)
        rot_right_eye = transform_pt(features.right_eye)
        rot_eyes_center = transform_pt(features.eyes_center)
        rot_chin = transform_pt(features.chin)
        rot_forehead = transform_pt(features.forehead)
        rot_crown = transform_pt(features.crown)
        rot_nose = transform_pt(features.nose_tip)

        rot_face_height = abs(rot_chin[1] - rot_crown[1])
        if rot_face_height < 10:
            rot_face_height = max(50.0, h_orig * 0.4)

        # 3. Compute Scale Factor
        # Desired face height in pixels on output canvas
        effective_face_pct = np.clip(target_face_pct + scale_offset, 0.40, 0.90)
        desired_face_h_px = target_h * effective_face_pct
        scale = desired_face_h_px / rot_face_height

        # 4. Compute Crop Box in rotated image coordinates
        crop_w = target_w / scale
        crop_h = target_h / scale

        # Horizontal centering: eyes midpoint placed at target canvas center
        # With optional manual x_shift_px (in canvas pixels -> / scale)
        center_x = rot_eyes_center[0] - (x_shift_px / scale)
        crop_x0 = center_x - crop_w / 2.0

        # Vertical positioning:
        # The eye line should sit at (1 - eye_ratio_from_bottom) * target_h
        target_eye_y_px = target_h * (1.0 - eye_ratio_from_bottom)
        # With optional manual y_shift_px
        crop_y0 = rot_eyes_center[1] - (target_eye_y_px / scale) - (y_shift_px / scale)

        # Build combined 2x3 affine matrix from original image directly to target crop:
        # M_final = S * [rot_mat with translation to crop origin]
        # [x_target] = [scale * rot_mat[0,0], scale * rot_mat[0,1], scale * (rot_mat[0,2] - crop_x0)]
        # [y_target] = [scale * rot_mat[1,0], scale * rot_mat[1,1], scale * (rot_mat[1,2] - crop_y0)]
        trans_rot = rot_mat.copy()
        trans_rot[0, 2] -= crop_x0
        trans_rot[1, 2] -= crop_y0

        final_affine = np.zeros((2, 3), dtype=np.float32)
        final_affine[0, :2] = trans_rot[0, :2] * scale
        final_affine[0, 2] = trans_rot[0, 2] * scale
        final_affine[1, :2] = trans_rot[1, :2] * scale
        final_affine[1, 2] = trans_rot[1, 2] * scale

        # Perform high-quality affine warp directly to target size
        cropped_canvas = cv2.warpAffine(
            rgb_image,
            final_affine,
            (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Compute landmarks mapped into target crop coordinates
        def map_to_crop(pt: Tuple[float, float]) -> Tuple[float, float]:
            v = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
            res = final_affine @ v
            return (float(res[0]), float(res[1]))

        features_in_crop = FaceFeatures(
            left_eye=map_to_crop(features.left_eye),
            right_eye=map_to_crop(features.right_eye),
            eyes_center=map_to_crop(features.eyes_center),
            chin=map_to_crop(features.chin),
            forehead=map_to_crop(features.forehead),
            crown=map_to_crop(features.crown),
            nose_tip=map_to_crop(features.nose_tip),
            roll_angle_deg=0.0,
            face_height=abs(map_to_crop(features.chin)[1] - map_to_crop(features.crown)[1]),
            eye_to_chin_dist=abs(map_to_crop(features.chin)[1] - map_to_crop(features.eyes_center)[1]),
            raw_landmarks=None
        )

        return CropResult(
            cropped_image=cropped_canvas,
            transform_matrix=final_affine,
            features=features,
            features_in_crop=features_in_crop,
            target_width=target_w,
            target_height=target_h,
            scale=scale
        )
