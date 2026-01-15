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


@dataclass
class Detection:
    """A detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    mask: Optional[np.ndarray] = None  # Segmentation mask if available


class DetectionService:
    """
    YOLO-based object detection service.
    Detects bottles, cups, and other objects suitable for ad replacement.
    """

    def __init__(self, model_size: str = "n"):
        """
        Initialize YOLO model.

        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
        """
        self.model = None
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            # Use YOLOv8 with segmentation for better masks
            # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg
            model_name = f"yolov8{self.model_size}-seg"
            logger.info(f"Loading YOLO model: {model_name}")

            self.model = YOLO(model_name)
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
        Detect objects in a frame.

        Args:
            frame: BGR image as numpy array
            confidence_threshold: Minimum confidence score
            target_classes_only: Only return objects suitable for ad replacement

        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]

            detections = []

            # Process detections
            if results.boxes is not None:
                boxes = results.boxes

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    # Filter by confidence
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

                    # Get segmentation mask if available
                    mask = None
                    if results.masks is not None and i < len(results.masks):
                        mask_data = results.masks[i].data.cpu().numpy()[0]
                        # Resize mask to frame size
                        import cv2
                        mask = cv2.resize(
                            mask_data,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)

                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        center=center,
                        mask=mask,
                    ))

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
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
