"""
Video Processing Pipeline
Handles object segmentation, tracking, and replacement.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .segmentation import SegmentationService
from .compositor import Compositor

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """State for a tracked object."""
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[int, int]
    confidence: float  # Detection confidence for weighted EMA
    # Smoothed values (EMA)
    smooth_bbox: tuple[float, float, float, float]
    smooth_centroid: tuple[float, float]
    # Velocity for prediction (dx, dy per frame)
    velocity: tuple[float, float] = (0.0, 0.0)


class ProcessingPipeline:
    """
    Main video processing pipeline.
    Coordinates segmentation, tracking, and compositing.
    """

    def __init__(self):
        self.segmentation = SegmentationService()
        self.compositor = Compositor()

        self.target_point: Optional[tuple[int, int]] = None
        self.selected_ad: Optional[str] = None
        self.tracked_object: Optional[TrackedObject] = None
        self.is_processing = False

        # Direct mask mode (from YOLO)
        self._use_direct_mask = False
        self._direct_mask: Optional[np.ndarray] = None
        self._direct_bbox: Optional[tuple[int, int, int, int]] = None

        # Tracking parameters
        self.ema_alpha_base = 0.15  # Base smoothing factor (lower = smoother)
        self.ema_alpha_min = 0.05  # Minimum alpha for low-confidence detections
        self.ema_alpha_max = 0.35  # Maximum alpha for high-confidence detections
        self.velocity_alpha = 0.1  # Velocity smoothing
        self.segment_interval = 3  # Re-segment every N frames (reduced to minimize lag)
        self._last_segment_frame = 0

    def set_target_point(self, x: int, y: int):
        """Set the target point for object selection."""
        logger.info(f"Target point set: ({x}, {y})")
        self.target_point = (x, y)
        self.tracked_object = None  # Reset tracking
        self.is_processing = True
        self._use_direct_mask = False
        self._last_segment_frame = 0

    def set_ad_asset(self, ad_id: str):
        """Set the ad asset to overlay."""
        logger.info(f"Ad asset selected: {ad_id}")
        self.selected_ad = ad_id
        self.compositor.load_ad_asset(ad_id)

    def reset(self):
        """Reset the pipeline state."""
        logger.info("Pipeline reset")
        self.target_point = None
        self.tracked_object = None
        self.is_processing = False
        self._last_segment_frame = 0
        self._use_direct_mask = False
        self._direct_mask = None
        self._direct_bbox = None

    def set_mask_directly(self, mask: np.ndarray, bbox: tuple[int, int, int, int]):
        """
        Set mask directly from YOLO segmentation.
        This bypasses the segmentation service.
        """
        logger.info(f"Setting direct mask with bbox: {bbox}")
        self._use_direct_mask = True
        self._direct_mask = mask
        self._direct_bbox = bbox
        self.is_processing = True

        # Also set target_point to the center for fallback
        x1, y1, x2, y2 = bbox
        self.target_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Initialize tracking with this mask
        self._update_tracking(mask, is_new=True)

    def set_bbox_directly(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, int]):
        """
        Set tracking from bbox only (when YOLO doesn't provide segmentation mask).
        Creates a rectangular mask from the bounding box.
        """
        logger.info(f"Setting bbox directly: {bbox}")
        self._use_direct_mask = True
        self._direct_bbox = bbox
        self.is_processing = True

        # Create rectangular mask
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1
        self._direct_mask = mask

        # Set target_point to center
        self.target_point = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Initialize tracking
        self._update_tracking(mask, is_new=True)

    def update_mask_from_detection(
        self,
        mask: Optional[np.ndarray],
        bbox: tuple[int, int, int, int],
        confidence: float = 1.0,
    ):
        """
        Update the tracked mask from a new YOLO detection.
        Called when YOLO provides a new mask for the tracked object.

        Args:
            mask: Segmentation mask (or None for bbox-only)
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence for weighted EMA
        """
        if not self._use_direct_mask:
            return

        self._direct_bbox = bbox

        if mask is not None:
            self._direct_mask = mask
            self._update_tracking(mask, is_new=False, confidence=confidence)
        elif self.tracked_object is not None:
            # No mask provided, create a simple rectangular mask from bbox
            # Get frame dimensions from existing mask
            h, w = self.tracked_object.mask.shape[:2]
            rect_mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            rect_mask[y1:y2, x1:x2] = 1
            self._direct_mask = rect_mask
            self._update_tracking(rect_mask, is_new=False, confidence=confidence)

    async def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        occlusion_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process a single video frame.
        Returns the processed frame with object replacement.

        Args:
            frame: Input BGR frame
            frame_num: Frame number for tracking
            occlusion_mask: Optional mask of occluding objects (hands/body)
        """
        if not self.is_processing or not self.selected_ad:
            return frame

        # Need either target_point or direct mask to process
        if not self._use_direct_mask and not self.target_point:
            return frame

        try:
            if self._use_direct_mask:
                # Use YOLO-provided mask directly
                # The mask is updated via update_mask_from_detection()
                pass  # tracked_object is already set
            else:
                # Determine if we need to run segmentation
                should_segment = (
                    self.tracked_object is None or
                    (frame_num - self._last_segment_frame) >= self.segment_interval
                )

                if should_segment:
                    # Run segmentation on this frame
                    mask = await self.segmentation.segment(frame, self.target_point)
                    if mask is not None:
                        self._update_tracking(mask, is_new=self.tracked_object is None)
                        self._last_segment_frame = frame_num
                elif self.tracked_object is not None:
                    # Use simple tracking between segmentation frames
                    mask = self._track_between_frames(frame)
                    if mask is not None:
                        self._update_tracking(mask, is_new=False)

            # Composite the result
            if self.tracked_object is not None:
                frame = self.compositor.composite(
                    frame,
                    self.tracked_object.mask,
                    self.tracked_object.smooth_bbox,
                    self.tracked_object.smooth_centroid,
                    occlusion_mask=occlusion_mask,
                )

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()

        return frame

    def _update_tracking(self, mask: np.ndarray, is_new: bool, confidence: float = 1.0):
        """
        Update tracked object state with new mask.

        Args:
            mask: Binary segmentation mask
            is_new: Whether this is a new object (initialize) or update
            confidence: Detection confidence (0-1) for weighted EMA
        """
        # Compute bounding box
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        # Compute centroid
        centroid = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

        if is_new or self.tracked_object is None:
            # Initialize tracking
            self.tracked_object = TrackedObject(
                mask=mask,
                bbox=bbox,
                centroid=centroid,
                confidence=confidence,
                smooth_bbox=tuple(float(v) for v in bbox),
                smooth_centroid=tuple(float(v) for v in centroid),
                velocity=(0.0, 0.0),
            )
        else:
            # Compute velocity (change in centroid position)
            old_cx, old_cy = self.tracked_object.smooth_centroid
            new_cx, new_cy = centroid
            raw_velocity = (float(new_cx - old_cx), float(new_cy - old_cy))

            # Smooth the velocity using EMA
            old_vx, old_vy = self.tracked_object.velocity
            v_alpha = self.velocity_alpha
            smooth_velocity = (
                v_alpha * raw_velocity[0] + (1 - v_alpha) * old_vx,
                v_alpha * raw_velocity[1] + (1 - v_alpha) * old_vy,
            )

            # Apply velocity prediction to reduce lag
            # Add a fraction of velocity to help track fast-moving objects
            prediction_factor = 0.3
            predicted_bbox = (
                bbox[0] + smooth_velocity[0] * prediction_factor,
                bbox[1] + smooth_velocity[1] * prediction_factor,
                bbox[2],  # width doesn't change with motion
                bbox[3],  # height doesn't change with motion
            )
            predicted_centroid = (
                centroid[0] + smooth_velocity[0] * prediction_factor,
                centroid[1] + smooth_velocity[1] * prediction_factor,
            )

            # Confidence-weighted EMA: higher confidence = trust detection more (higher alpha)
            # Scale alpha linearly between min and max based on confidence
            alpha = self.ema_alpha_min + (self.ema_alpha_max - self.ema_alpha_min) * confidence
            alpha = np.clip(alpha, self.ema_alpha_min, self.ema_alpha_max)

            smooth_bbox = tuple(
                alpha * new + (1 - alpha) * old
                for new, old in zip(predicted_bbox, self.tracked_object.smooth_bbox)
            )
            smooth_centroid = tuple(
                alpha * new + (1 - alpha) * old
                for new, old in zip(predicted_centroid, self.tracked_object.smooth_centroid)
            )

            self.tracked_object = TrackedObject(
                mask=mask,
                bbox=bbox,
                centroid=centroid,
                confidence=confidence,
                smooth_bbox=smooth_bbox,
                smooth_centroid=smooth_centroid,
                velocity=smooth_velocity,
            )

    def _track_between_frames(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple tracking between segmentation frames.
        Uses the previous mask shifted to follow motion.
        For MVP, we just return the previous mask (assumes small motion).
        A more advanced approach would use optical flow.
        """
        if self.tracked_object is None:
            return None
        return self.tracked_object.mask
