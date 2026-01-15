"""
Segmentation Service
Handles object segmentation using SAM 2 via Modal.
"""
import asyncio
import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Flag to use local fallback (GrabCut) instead of Modal SAM 2
USE_LOCAL_FALLBACK = True  # Set to False when Modal is configured


class SegmentationService:
    """
    Service for object segmentation.
    Uses SAM 2 via Modal for production, or local GrabCut for development.
    """

    def __init__(self):
        self._modal_client = None
        self._initialized = False

    async def segment(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Segment an object at the given point in the frame.
        Returns a binary mask (H, W) where 1 = object, 0 = background.
        """
        if USE_LOCAL_FALLBACK:
            return self._segment_local(frame, point)
        else:
            return await self._segment_modal(frame, point)

    def _segment_local(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Local segmentation using GrabCut.
        This is a fallback for development without GPU.
        """
        try:
            h, w = frame.shape[:2]
            x, y = point

            # Create initial mask
            mask = np.zeros((h, w), np.uint8)

            # Initialize rectangle around the click point
            # Use a reasonable default size
            rect_size = min(w, h) // 3
            x1 = max(0, x - rect_size // 2)
            y1 = max(0, y - rect_size // 2)
            x2 = min(w, x + rect_size // 2)
            y2 = min(h, y + rect_size // 2)

            rect = (x1, y1, x2 - x1, y2 - y1)

            # Run GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(
                frame,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,  # iterations
                cv2.GC_INIT_WITH_RECT,
            )

            # Convert mask to binary
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                1,
                0,
            ).astype(np.uint8)

            return binary_mask

        except Exception as e:
            logger.error(f"Local segmentation failed: {e}")
            return None

    async def _segment_modal(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Segment using SAM 2 via Modal.
        """
        try:
            # Lazy import modal
            import modal

            if self._modal_client is None:
                # Get reference to deployed Modal function
                self._modal_client = modal.Cls.lookup(
                    "dystopian-ads", "SAM2Inference"
                )
                self._initialized = True

            # Encode frame as JPEG bytes
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            frame_bytes = buffer.getvalue()

            # Call Modal function
            result = await asyncio.to_thread(
                self._modal_client().segment.remote,
                frame_bytes,
                point[0],
                point[1],
            )

            # Decode mask from bytes
            mask = np.frombuffer(result["mask"], dtype=np.uint8)
            mask = mask.reshape(result["shape"])

            logger.info(f"SAM 2 segmentation score: {result['score']:.3f}")
            return mask

        except Exception as e:
            logger.error(f"Modal segmentation failed: {e}")
            # Fall back to local
            return self._segment_local(frame, point)
