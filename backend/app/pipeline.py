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
    # Smoothed values (EMA)
    smooth_bbox: tuple[float, float, float, float]
    smooth_centroid: tuple[float, float]


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

        # Tracking parameters
        self.ema_alpha = 0.3  # Smoothing factor (lower = smoother)
        self.segment_interval = 10  # Re-segment every N frames
        self._last_segment_frame = 0

    def set_target_point(self, x: int, y: int):
        """Set the target point for object selection."""
        logger.info(f"Target point set: ({x}, {y})")
        self.target_point = (x, y)
        self.tracked_object = None  # Reset tracking
        self.is_processing = True
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

        # Initialize tracking with this mask
        self._update_tracking(mask, is_new=True)

    async def process_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Process a single video frame.
        Returns the processed frame with object replacement.
        """
        if not self.is_processing or not self.target_point or not self.selected_ad:
            return frame

        try:
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
                )

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")

        return frame

    def _update_tracking(self, mask: np.ndarray, is_new: bool):
        """Update tracked object state with new mask."""
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
                smooth_bbox=tuple(float(v) for v in bbox),
                smooth_centroid=tuple(float(v) for v in centroid),
            )
        else:
            # Update with EMA smoothing
            alpha = self.ema_alpha
            smooth_bbox = tuple(
                alpha * new + (1 - alpha) * old
                for new, old in zip(bbox, self.tracked_object.smooth_bbox)
            )
            smooth_centroid = tuple(
                alpha * new + (1 - alpha) * old
                for new, old in zip(centroid, self.tracked_object.smooth_centroid)
            )

            self.tracked_object = TrackedObject(
                mask=mask,
                bbox=bbox,
                centroid=centroid,
                smooth_bbox=smooth_bbox,
                smooth_centroid=smooth_centroid,
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
