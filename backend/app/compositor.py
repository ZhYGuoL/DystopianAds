"""
Frame Compositor
Handles background inpainting and ad asset overlay.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Path to ad assets
ASSETS_DIR = Path(__file__).parent.parent / "assets"


class Compositor:
    """
    Composites the ad asset onto the video frame.
    Handles background inpainting and overlay with proper scaling.
    """

    def __init__(self):
        self.ad_asset: Optional[np.ndarray] = None
        self.ad_asset_id: Optional[str] = None

    def load_ad_asset(self, ad_id: str):
        """Load an ad asset image."""
        if ad_id == self.ad_asset_id and self.ad_asset is not None:
            return

        # Try to load from assets directory
        asset_path = ASSETS_DIR / f"{ad_id}.png"

        if asset_path.exists():
            img = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.ad_asset = img
                self.ad_asset_id = ad_id
                logger.info(f"Loaded ad asset: {ad_id}")
                return

        # Generate a placeholder if asset not found
        logger.warning(f"Asset not found: {ad_id}, using placeholder")
        self.ad_asset = self._generate_placeholder(ad_id)
        self.ad_asset_id = ad_id

    def _generate_placeholder(self, ad_id: str) -> np.ndarray:
        """Generate a placeholder ad asset."""
        # Create a colored cylinder-like shape with transparency
        size = 200
        img = np.zeros((size * 2, size, 4), dtype=np.uint8)

        # Define colors for different brands
        colors = {
            "coke": (34, 34, 200),     # Red (BGR)
            "pepsi": (200, 100, 50),    # Blue
            "sprite": (100, 200, 100),  # Green
            "fanta": (50, 150, 255),    # Orange
        }
        color = colors.get(ad_id, (128, 128, 128))

        # Draw a simple can shape
        cv2.rectangle(img, (20, 20), (size - 20, size * 2 - 20), (*color, 255), -1)
        cv2.rectangle(img, (20, 20), (size - 20, size * 2 - 20), (255, 255, 255, 255), 2)

        # Add brand name
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ad_id.upper()
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = (size - text_size[0]) // 2
        text_y = size + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.8, (255, 255, 255, 255), 2)

        return img

    def composite(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[float, float, float, float],
        centroid: tuple[float, float],
    ) -> np.ndarray:
        """
        Composite the ad asset onto the frame.

        1. Inpaint the masked region (remove original object)
        2. Overlay the ad asset at the correct position and scale
        """
        if self.ad_asset is None:
            return frame

        result = frame.copy()

        # Step 1: Inpaint the masked region
        mask_uint8 = (mask * 255).astype(np.uint8)
        # Dilate mask slightly for better inpainting
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=2)

        # Use OpenCV inpainting (Telea algorithm)
        result = cv2.inpaint(result, dilated_mask, 5, cv2.INPAINT_TELEA)

        # Step 2: Overlay ad asset
        x, y, w, h = [int(v) for v in bbox]

        # Scale ad asset to match bounding box
        if self.ad_asset.shape[2] == 4:
            # Has alpha channel
            asset_rgba = self.ad_asset
        else:
            # Add alpha channel
            asset_rgba = cv2.cvtColor(self.ad_asset, cv2.COLOR_BGR2BGRA)
            asset_rgba[:, :, 3] = 255

        # Resize asset to fit bounding box
        # Maintain aspect ratio
        asset_h, asset_w = asset_rgba.shape[:2]
        aspect = asset_w / asset_h

        if w / h > aspect:
            # Bounding box is wider, fit to height
            new_h = h
            new_w = int(h * aspect)
        else:
            # Bounding box is taller, fit to width
            new_w = w
            new_h = int(w / aspect)

        # Ensure minimum size
        new_w = max(new_w, 10)
        new_h = max(new_h, 10)

        try:
            scaled_asset = cv2.resize(asset_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return result

        # Position asset at centroid
        cx, cy = [int(v) for v in centroid]
        overlay_x = cx - new_w // 2
        overlay_y = cy - new_h // 2

        # Clip to frame bounds
        frame_h, frame_w = result.shape[:2]
        src_x1 = max(0, -overlay_x)
        src_y1 = max(0, -overlay_y)
        src_x2 = min(new_w, frame_w - overlay_x)
        src_y2 = min(new_h, frame_h - overlay_y)

        dst_x1 = max(0, overlay_x)
        dst_y1 = max(0, overlay_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return result

        # Alpha blending
        asset_region = scaled_asset[src_y1:src_y2, src_x1:src_x2]
        alpha = asset_region[:, :, 3:4] / 255.0
        bgr = asset_region[:, :, :3]

        frame_region = result[dst_y1:dst_y2, dst_x1:dst_x2]

        # Blend: result = alpha * foreground + (1 - alpha) * background
        blended = (alpha * bgr + (1 - alpha) * frame_region).astype(np.uint8)
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        return result
