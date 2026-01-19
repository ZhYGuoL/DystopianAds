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
        # Lighting adaptation settings
        self.adapt_lighting = True
        self.lighting_strength = 0.4  # How much to adapt (0=none, 1=full)

    def _analyze_scene_lighting(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict:
        """
        Analyze the lighting conditions around the masked region.
        Returns brightness and color temperature adjustments.
        """
        x, y, w, h = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]

        # Expand bbox to sample surrounding area
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_frame, x + w + margin)
        y2 = min(h_frame, y + h + margin)

        # Create inverse mask (surrounding area only)
        region = frame[y1:y2, x1:x2]
        region_mask = mask[y1:y2, x1:x2]
        surrounding_mask = (region_mask == 0)

        if not np.any(surrounding_mask):
            # No surrounding pixels, return neutral
            return {"brightness": 1.0, "color_shift": np.array([0, 0, 0], dtype=np.float32)}

        # Sample surrounding pixels
        surrounding_pixels = region[surrounding_mask]

        # Calculate average brightness (using luminance formula)
        avg_color = np.mean(surrounding_pixels, axis=0)  # BGR
        brightness = 0.299 * avg_color[2] + 0.587 * avg_color[1] + 0.114 * avg_color[0]
        brightness_factor = brightness / 128.0  # Normalize to ~1.0 for normal lighting

        # Calculate color temperature bias
        # Compare blue vs red to estimate warm/cool lighting
        color_shift = avg_color - np.array([128, 128, 128], dtype=np.float32)

        return {
            "brightness": np.clip(brightness_factor, 0.5, 1.5),
            "color_shift": color_shift * 0.3,  # Reduce intensity of color shift
        }

    def _apply_lighting_adjustment(
        self,
        asset: np.ndarray,
        lighting: dict,
    ) -> np.ndarray:
        """
        Apply lighting adjustments to the ad asset.
        """
        result = asset.astype(np.float32)

        # Separate alpha channel if present
        if result.shape[2] == 4:
            bgr = result[:, :, :3]
            alpha = result[:, :, 3:4]
        else:
            bgr = result
            alpha = None

        # Apply brightness adjustment
        brightness = lighting["brightness"]
        strength = self.lighting_strength
        adjusted_brightness = 1.0 + (brightness - 1.0) * strength
        bgr = bgr * adjusted_brightness

        # Apply color shift
        color_shift = lighting["color_shift"] * strength
        bgr = bgr + color_shift

        # Clip to valid range
        bgr = np.clip(bgr, 0, 255)

        if alpha is not None:
            result = np.concatenate([bgr, alpha], axis=2)
        else:
            result = bgr

        return result.astype(np.uint8)

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
        occlusion_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Composite the ad asset onto the frame.

        1. Inpaint the masked region (remove original object)
        2. Overlay the ad asset at the correct position and scale
        3. Apply occlusion mask to show hands/fingers in front of ad

        Args:
            frame: Input BGR frame
            mask: Binary mask of object to replace
            bbox: Bounding box (x, y, w, h) - smoothed
            centroid: Center point (x, y) - smoothed
            occlusion_mask: Optional binary mask where 1 = occluding pixels (hands/body)
        """
        if self.ad_asset is None:
            return frame

        # Keep a copy of the original frame for occlusion handling
        original_frame = frame.copy()
        result = frame.copy()

        # Step 1: Inpaint the masked region with improved quality
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Create dilated mask for inpainting (slightly larger to cover edges)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=2)

        # Create edge mask for blending
        edge_mask = cv2.dilate(mask_uint8, kernel, iterations=3) - cv2.erode(mask_uint8, kernel, iterations=1)

        # Use Navier-Stokes inpainting with larger radius for better quality on large areas
        # NS method is better for larger regions, Telea is faster for small regions
        inpaint_radius = 7  # Larger radius for smoother results
        result = cv2.inpaint(result, dilated_mask, inpaint_radius, cv2.INPAINT_NS)

        # Apply slight blur at the inpainting boundary for smoother blending
        if np.any(edge_mask):
            # Create a soft edge mask for blending
            edge_float = edge_mask.astype(np.float32) / 255.0
            edge_blurred = cv2.GaussianBlur(edge_float, (7, 7), 0)

            # Blend original edge pixels with inpainted result for smoother transition
            edge_alpha = edge_blurred[:, :, np.newaxis]
            blurred_result = cv2.GaussianBlur(result, (3, 3), 0)
            result = (edge_alpha * blurred_result + (1 - edge_alpha) * result).astype(np.uint8)

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

        # Step 2.5: Apply lighting adaptation to match scene
        if self.adapt_lighting:
            lighting = self._analyze_scene_lighting(original_frame, mask, bbox)
            scaled_asset = self._apply_lighting_adjustment(scaled_asset, lighting)

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

        # Step 3: Apply occlusion mask (restore original pixels where hands/body are)
        if occlusion_mask is not None:
            # Only apply occlusion within the region affected by the ad
            # Expand the occlusion mask to ensure clean edges
            kernel = np.ones((3, 3), np.uint8)
            occlusion_dilated = cv2.dilate(occlusion_mask, kernel, iterations=1)

            # Also check intersection with the object mask - only occlude where
            # the person overlaps with the replaced object region
            intersection = occlusion_dilated & (mask > 0)

            if np.any(intersection):
                # Feather the edge for smoother blending
                intersection_float = intersection.astype(np.float32)
                intersection_blurred = cv2.GaussianBlur(intersection_float, (5, 5), 0)
                occlusion_alpha = intersection_blurred[:, :, np.newaxis]

                # Blend original frame pixels where there's occlusion
                result = (occlusion_alpha * original_frame +
                         (1 - occlusion_alpha) * result).astype(np.uint8)

        return result
