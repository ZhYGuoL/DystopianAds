"""
DystopianAds Backend - FastAPI + WebSocket + YOLO Detection
"""
import asyncio
import json
import logging
import io
import time
from typing import Optional
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import cv2

from .pipeline import ProcessingPipeline
from .detection import DetectionService, Detection, find_detection_at_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DystopianAds API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline = ProcessingPipeline()
detector: Optional[DetectionService] = None
viewer_connections: list[WebSocket] = []
frame_count = 0
current_detections: list[Detection] = []
selected_detection: Optional[Detection] = None
selected_track_id: Optional[int] = None  # ByteTrack ID for persistent tracking
selected_class_name: Optional[str] = None  # Fallback: track by class name
last_frame_shape: tuple[int, int, int] = (480, 640, 3)  # Default shape, updated on each frame
current_occlusion_mask: Optional[np.ndarray] = None  # Current person/hand occlusion mask

# Frame buffer for viewer - stores (timestamp, frame, detections) tuples with 1 second delay
frame_buffer: deque = deque(maxlen=100)


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global detector
    logger.info("Initializing YOLO detector with segmentation model...")
    # Using yolo26m-seg for good accuracy + real-time performance
    # Options: yolo26s-seg (fast), yolo26m-seg (balanced), yolo26l-seg (accurate), yolo26x-seg (best)
    # imgsz: 640 (fast), 960 (balanced), 1280 (accurate)
    detector = DetectionService(
        model_name="yolo26n.pt",
        use_tracking=True,
        imgsz=640,
    )
    logger.info("YOLO detector ready (yolo26n, ByteTrack, 640px - optimized for speed)")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "dystopian-ads", "yolo": detector is not None}


@app.websocket("/ws/capture")
async def capture_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for the capture window.
    Receives frames and control messages.
    """
    global frame_count, current_detections, selected_detection, selected_track_id, selected_class_name, last_frame_shape, current_occlusion_mask

    await websocket.accept()
    logger.info("Capture client connected")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                # Binary frame data (JPEG)
                frame_bytes = message["bytes"]
                frame_count += 1

                # Decode frame
                img = Image.open(io.BytesIO(frame_bytes))
                frame = np.array(img)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = frame[:, :, ::-1].copy()  # RGB to BGR

                # Store frame shape for use in click handling
                last_frame_shape = frame.shape

                # Run YOLO detection on every frame for responsive bounding boxes
                if detector:  # Every frame
                    # Use lower confidence and detect ALL objects for debugging
                    detect_start = time.time()
                    current_detections = detector.detect(
                        frame,
                        confidence_threshold=0.25,  # Lower threshold
                        target_classes_only=False,  # Show ALL detections
                    )
                    detect_time = (time.time() - detect_start) * 1000  # Convert to ms
                    if current_detections:
                        logger.info(f"Frame {frame_count} | Inference: {detect_time:.1f}ms | Detected: {[(d.class_name, d.confidence) for d in current_detections]}")
                    else:
                        logger.info(f"Frame {frame_count} | Inference: {detect_time:.1f}ms | No detections")

                    # Get occlusion mask from existing detections (more efficient than re-running YOLO)
                    if pipeline.is_processing:
                        current_occlusion_mask = detector.get_occlusion_mask_from_detections(
                            current_detections,
                            frame.shape[:2],
                        )

                    # Update tracking - prioritize track_id, fallback to class_name
                    if selected_track_id is not None or selected_class_name:
                        best_match = None

                        # First try to match by track_id (most reliable)
                        if selected_track_id is not None:
                            for d in current_detections:
                                if d.track_id == selected_track_id:
                                    best_match = d
                                    break

                        # Fallback to class_name matching if track_id not found
                        if best_match is None and selected_class_name:
                            matching = [d for d in current_detections if d.class_name == selected_class_name]
                            if matching:
                                best_match = max(matching, key=lambda d: d.confidence)
                                # Update track_id if we found a match (re-acquire tracking)
                                if best_match.track_id is not None:
                                    selected_track_id = best_match.track_id
                                    logger.info(f"Re-acquired track_id: {selected_track_id}")

                        if best_match:
                            selected_detection = best_match
                            # Update pipeline with new mask/bbox and confidence for weighted EMA
                            if best_match.mask is not None:
                                pipeline.update_mask_from_detection(
                                    best_match.mask,
                                    best_match.bbox,
                                    confidence=best_match.confidence,
                                )
                            else:
                                # No mask, update tracking with bbox-based approach
                                pipeline.update_mask_from_detection(
                                    None,
                                    best_match.bbox,
                                    confidence=best_match.confidence,
                                )

                # Send detections to capture client for visualization on EVERY frame
                # (detections are refreshed every 2 frames, but we send on all frames for smooth UI updates)
                # Convert numpy types to Python native types for JSON serialization
                det_data = [
                    {
                        "id": i,
                        "class": d.class_name,
                        "confidence": round(float(d.confidence), 2),
                        "bbox": [int(x) for x in d.bbox],
                        "center": [int(x) for x in d.center],
                        "track_id": d.track_id,  # ByteTrack persistent ID
                        "contour": d.contour,  # Segmentation contour points [(x,y), ...]
                    }
                    for i, d in enumerate(current_detections)
                ]
                await websocket.send_json({
                    "type": "detections",
                    "data": det_data,
                })

                # Add frame to buffer with current detections for delayed viewer processing
                frame_buffer.append({
                    'timestamp': time.time(),
                    'frame': frame.copy(),
                    'detections': list(current_detections),
                    'frame_num': frame_count,
                })

                # Process buffered frames (with 1 second delay)
                asyncio.create_task(process_buffered_frames())

            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    logger.info(f"Control message: {data}")

                    if data.get("type") == "click":
                        x, y = data["x"], data["y"]

                        # Find detection at click point
                        clicked_det = find_detection_at_point(current_detections, (x, y))

                        if clicked_det:
                            selected_detection = clicked_det
                            selected_track_id = clicked_det.track_id  # ByteTrack ID for persistent tracking
                            selected_class_name = clicked_det.class_name  # Fallback tracking by class
                            logger.info(f"Selected: {clicked_det.class_name} (track_id={clicked_det.track_id}) at {clicked_det.bbox}")

                            # Use detection mask or bbox for pipeline
                            if clicked_det.mask is not None:
                                pipeline.set_mask_directly(clicked_det.mask, clicked_det.bbox)
                            else:
                                # No segmentation mask, use bbox directly
                                pipeline.set_bbox_directly(clicked_det.bbox, last_frame_shape)

                            if "adId" in data:
                                pipeline.set_ad_asset(data["adId"])

                            await websocket.send_json({
                                "type": "selected",
                                "detection": {
                                    "class": clicked_det.class_name,
                                    "bbox": [int(x) for x in clicked_det.bbox],
                                }
                            })
                        else:
                            # No detection at point, use point directly
                            selected_track_id = None  # Clear track ID
                            selected_class_name = None  # Clear class tracking
                            pipeline.set_target_point(x, y)
                            if "adId" in data:
                                pipeline.set_ad_asset(data["adId"])

                    elif data.get("type") == "select_ad":
                        pipeline.set_ad_asset(data["adId"])

                    elif data.get("type") == "reset":
                        pipeline.reset()
                        selected_detection = None
                        selected_track_id = None
                        selected_class_name = None

                    await broadcast_status()

                except json.JSONDecodeError:
                    logger.error("Invalid JSON message")

    except WebSocketDisconnect:
        logger.info("Capture client disconnected")
    except Exception as e:
        logger.error(f"Capture error: {e}")


@app.websocket("/ws/viewer")
async def viewer_websocket(websocket: WebSocket):
    """WebSocket endpoint for the viewer window."""
    await websocket.accept()
    viewer_connections.append(websocket)
    logger.info(f"Viewer connected. Total: {len(viewer_connections)}")

    await websocket.send_json({
        "type": "status",
        "message": "Connected - waiting for capture",
    })

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Viewer error: {e}")
    finally:
        if websocket in viewer_connections:
            viewer_connections.remove(websocket)
        logger.info(f"Viewer disconnected. Total: {len(viewer_connections)}")


async def process_buffered_frames():
    """Process frames from buffer with 1-second delay and draw detection boxes."""
    global selected_detection, current_occlusion_mask

    current_time = time.time()
    delay_threshold = 1.0  # 1 second delay

    # Check if any frames are ready (1+ seconds old)
    if not frame_buffer:
        return

    # Get the oldest frame
    oldest = frame_buffer[0]
    if current_time - oldest['timestamp'] >= delay_threshold:
        # Frame is old enough, process it
        frame_data = frame_buffer.popleft()

        try:
            # Process frame through pipeline
            processed = await pipeline.process_frame(
                frame_data['frame'],
                frame_data['frame_num'],
                occlusion_mask=current_occlusion_mask,
            )

            # Draw ALL detection boxes with bounding boxes
            for detection in frame_data['detections']:
                x1, y1, x2, y2 = detection.bbox
                # Yellow for all detections
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # Draw label
                label = f"{detection.class_name} {int(detection.confidence*100)}%"
                cv2.putText(
                    processed,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            # Draw selected detection in green
            if selected_detection:
                x1, y1, x2, y2 = selected_detection.bbox
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    processed,
                    f"Replacing: {selected_detection.class_name}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Convert to RGB and encode
            if len(processed.shape) == 3 and processed.shape[2] == 3:
                processed = processed[:, :, ::-1]  # BGR to RGB

            output_img = Image.fromarray(processed)
            output_buffer = io.BytesIO()
            output_img.save(output_buffer, format="JPEG", quality=85)
            output_bytes = output_buffer.getvalue()

            # Broadcast to viewers
            disconnected = []
            for viewer in viewer_connections:
                try:
                    await viewer.send_bytes(output_bytes)
                except Exception:
                    disconnected.append(viewer)

            for viewer in disconnected:
                if viewer in viewer_connections:
                    viewer_connections.remove(viewer)

        except Exception as e:
            logger.error(f"Buffered frame processing error: {e}")


async def process_and_broadcast(frame: np.ndarray, frame_num: int):
    """Process a frame and broadcast to all viewers."""
    global selected_detection, current_occlusion_mask

    try:
        # Process frame through pipeline with occlusion handling
        processed = await pipeline.process_frame(
            frame,
            frame_num,
            occlusion_mask=current_occlusion_mask,
        )

        # Draw selected detection box on output
        if selected_detection:
            x1, y1, x2, y2 = selected_detection.bbox
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                processed,
                f"Replacing: {selected_detection.class_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Convert to RGB and encode
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed = processed[:, :, ::-1]  # BGR to RGB

        output_img = Image.fromarray(processed)
        output_buffer = io.BytesIO()
        output_img.save(output_buffer, format="JPEG", quality=85)
        output_bytes = output_buffer.getvalue()

        # Broadcast to viewers
        disconnected = []
        for viewer in viewer_connections:
            try:
                await viewer.send_bytes(output_bytes)
            except Exception:
                disconnected.append(viewer)

        for viewer in disconnected:
            if viewer in viewer_connections:
                viewer_connections.remove(viewer)

    except Exception as e:
        logger.error(f"Processing error: {e}")


async def broadcast_status():
    """Broadcast status to all viewers."""
    status_msg = {
        "type": "status",
        "message": f"Processing: {pipeline.is_processing}",
        "selected_ad": pipeline.selected_ad,
    }

    disconnected = []
    for viewer in viewer_connections:
        try:
            await viewer.send_json(status_msg)
        except Exception:
            disconnected.append(viewer)

    for viewer in disconnected:
        if viewer in viewer_connections:
            viewer_connections.remove(viewer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
