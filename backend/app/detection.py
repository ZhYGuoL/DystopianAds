"""
YOLO Object Detection Service
Detects objects that can be replaced with ads (bottles, cups, cans, etc.)
"""
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Target classes for ad replacement (COCO class IDs)
# Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
TARGET_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "bowl",
    45: "banana",
    46: "apple",
    47: "sandwich",
    49: "orange",
    65: "remote",
    67: "cell phone",
    73: "book",
    75: "vase",
    76: "scissors",
}

# Classes used for occlusion (hands/body parts)
OCCLUSION_CLASSES = {
    0: "person",  # Person includes hands/arms
}


@dataclass
class Detection:
    """A detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    mask: Optional[np.ndarray] = None  # Segmentation mask if available
    track_id: Optional[int] = None  # Persistent track ID from ByteTrack
    contour: Optional[list[tuple[int, int]]] = None  # Simplified contour points for visualization


class DetectionService:
    """
    YOLO-based object detection service.
    Detects bottles, cups, and other objects suitable for ad replacement.   
    Supports ByteTrack for persistent object tracking.
    """

    def __init__(self, model_name: str = "yolo26n-seg.pt", use_tracking: bool = True, imgsz: int = 960):
        """
        Initialize YOLO model.

        Args:
            model_name: Full model name - use '-seg' suffix for segmentation masks
                       Recommended for real-time: yolo26m-seg.pt or yolo26l-seg.pt
                       For max accuracy (slower): yolo26x-seg.pt
            use_tracking: Enable ByteTrack tracking for persistent IDs
            imgsz: Input image size (higher = more accurate, slower).
                   640=fast, 960=balanced, 1280=accurate
        """
        self.model = None
        self.model_name = model_name
        self.use_tracking = use_tracking
        self.imgsz = imgsz  # Higher resolution for better accuracy
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            # yolo26n.pt: largest and most accurate model with segmentation
            logger.info(f"Loading YOLO model: {self.model_name}")

            self.model = YOLO(self.model_name)
            logger.info(f"YOLO model loaded. Classes: {len(self.model.names)} total")
            logger.info(f"Target classes: {list(TARGET_CLASSES.values())}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes_only: bool = True,
    ) -> list[Detection]:
        """
        Detect objects in a frame with optional ByteTrack tracking.

        Args:
            frame: BGR image as numpy array
            confidence_threshold: Minimum confidence score
            target_classes_only: Only return objects suitable for ad replacement

        Returns:
            List of Detection objects with track_id if tracking is enabled
        """
        if self.model is None:
            return []

        try:
            # Run inference with or without tracking
            # Using higher resolution (imgsz) for better accuracy
            if self.use_tracking:
                # ByteTrack tracking - persist=True maintains IDs across frames
                results = self.model.track(
                    frame,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence_threshold,
                    imgsz=self.imgsz,
                )[0]
            else:
                results = self.model(frame, verbose=False, conf=confidence_threshold, imgsz=self.imgsz)[0]

            detections = []

            # Process detections
            if results.boxes is not None:
                boxes = results.boxes

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    # Filter by confidence (redundant if conf passed to model, but safe)
                    if confidence < confidence_threshold:
                        continue

                    # Filter by target classes if requested
                    if target_classes_only and class_id not in TARGET_CLASSES:
                        continue

                    # Get class name
                    if class_id in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[class_id]
                    else:
                        class_name = self.model.names.get(class_id, f"class_{class_id}")

                    # Get bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Get track ID if tracking is enabled
                    track_id = None
                    if self.use_tracking and boxes.id is not None:
                        try:
                            track_id = int(boxes.id[i])
                        except (IndexError, TypeError):
                            pass

                    # Get segmentation mask if available
                    mask = None
                    contour = None
                    if results.masks is not None and i < len(results.masks):
                        mask_data = results.masks[i].data.cpu().numpy()[0]
                        # Resize mask to frame size
                        import cv2
                        mask = cv2.resize(
                            mask_data,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)

                        # Extract contour points for visualization
                        contours, _ = cv2.findContours(
                            (mask * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            # Get the largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            # Simplify contour to reduce points (epsilon = 0.3% of perimeter for more detail)
                            epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
                            # Convert to list of (x, y) tuples
                            contour = [(int(pt[0][0]), int(pt[0][1])) for pt in simplified]

                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        center=center,
                        mask=mask,
                        track_id=track_id,
                        contour=contour,
                    ))

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_all(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> list[Detection]:
        """
        Detect ALL objects (not just target classes).
        Useful for debugging or showing all detections.
        """
        return self.detect(frame, confidence_threshold, target_classes_only=False)

    def get_occlusion_mask_from_detections(
        self,
        detections: list["Detection"],
        frame_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Get a combined occlusion mask from detected persons in the detection list.
        This is more efficient than running another YOLO inference.

        Args:
            detections: List of Detection objects from detect()
            frame_shape: (height, width) of the frame

        Returns:
            Binary mask where 1 = occluding region (person), 0 = background
        """
        h, w = frame_shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            # Only process occlusion classes (person)
            if det.class_id not in OCCLUSION_CLASSES:
                continue

            if det.mask is not None:
                # Add to combined mask
                combined_mask = np.maximum(combined_mask, det.mask)

        return combined_mask if np.any(combined_mask) else None

    def get_occlusion_mask(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Get a combined occlusion mask from all detected persons.
        This is used to handle hand/finger occlusion - where body parts
        should appear in front of the replaced ad.

        Note: Prefer get_occlusion_mask_from_detections() if you already
        have detection results, as it avoids duplicate inference.

        Returns:
            Binary mask where 1 = occluding region (person), 0 = background
        """
        if self.model is None:
            return None

        try:
            results = self.model(frame, verbose=False)[0]

            h, w = frame.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            if results.boxes is not None and results.masks is not None:
                boxes = results.boxes

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    # Only process occlusion classes (person)
                    if class_id not in OCCLUSION_CLASSES:
                        continue
                    if confidence < confidence_threshold:
                        continue

                    # Get segmentation mask
                    if i < len(results.masks):
                        import cv2
                        mask_data = results.masks[i].data.cpu().numpy()[0]
                        mask = cv2.resize(
                            mask_data,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)

                        # Add to combined mask
                        combined_mask = np.maximum(combined_mask, mask)

            return combined_mask if np.any(combined_mask) else None

        except Exception as e:
            logger.error(f"Occlusion mask error: {e}")
            return None


def find_detection_at_point(
    detections: list[Detection],
    point: tuple[int, int],
) -> Optional[Detection]:
    """
    Find the detection that contains the given point.
    If multiple detections contain the point, return the smallest one.
    """
    x, y = point
    matching = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            matching.append((det, area))

    if not matching:
        return None

    # Return smallest matching detection
    matching.sort(key=lambda x: x[1])
    return matching[0][0]
