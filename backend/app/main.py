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
from concurrent.futures import ThreadPoolExecutor

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

# Frame buffer for viewer - stores (timestamp, frame, detections) tuples with guaranteed constant delay
frame_buffer: deque = deque(maxlen=150)  # Larger buffer for constant delay window
buffer_processor_task: Optional[asyncio.Task] = None
BUFFER_DELAY_MS = 200  # 200ms constant delay to guarantee encoding completes
last_viewer_frame_num = 0

# Thread pool for non-blocking YOLO inference
detection_executor: Optional[ThreadPoolExecutor] = None
# Thread pool for frame encoding
encode_executor: Optional[ThreadPoolExecutor] = None


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global detector, buffer_processor_task, detection_executor, encode_executor
    logger.info("Initializing YOLO detector with segmentation model...")
    # Using yolo26m-seg for good accuracy + real-time performance
    # Options: yolo26n-seg (fast), yolo26m-seg (balanced), yolo26l-seg (accurate), yolo26x-seg (best)
    # imgsz: 640 (fast), 960 (balanced), 1280 (accurate)
    detector = DetectionService(
        model_name="yolo26n-seg.pt",
        use_tracking=True,
        imgsz=480,
    )
    logger.info("YOLO detector ready (yolo26n-seg, ByteTrack, 480px - optimized for speed+segmentation)")

    # Initialize thread pools for parallel processing
    detection_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo-")
    encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="encode-")

    # Start background buffer processor task
    buffer_processor_task = asyncio.create_task(buffer_processor_loop())


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

                # Run YOLO detection every 3rd frame for faster viewer throughput
                if detector and frame_count % 3 == 0:
                    # Run detection in thread pool to avoid blocking event loop
                    detect_start = time.time()
                    loop = asyncio.get_event_loop()
                    current_detections = await loop.run_in_executor(
                        detection_executor,
                        detector.detect,
                        frame,
                        0.25,  # confidence_threshold
                        False,  # target_classes_only
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

                # Always add frame to buffer - the processor handles backlog
                frame_buffer.append({
                    'timestamp': time.time(),
                    'frame': frame.copy(),
                    'detections': list(current_detections),
                    'frame_num': frame_count,
                })

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


async def buffer_processor_loop():
    """Background task to process buffered frames at steady rate with constant delay."""
    global last_viewer_frame_num
    logger.info("Buffer processor task started")
    while True:
        try:
            current_time = time.time()
            delay_threshold = BUFFER_DELAY_MS / 1000.0  # Convert to seconds

            # Aggressive frame dropping if buffer exceeds safe size
            # This prevents lag accumulation by maintaining steady throughput
            if len(frame_buffer) > 50:
                # Drop frames older than delay_threshold to catch up quickly
                dropped_count = 0
                while len(frame_buffer) > 40:
                    oldest = frame_buffer[0]
                    if current_time - oldest['timestamp'] > delay_threshold * 1.5:  # 1.5x delay window
                        dropped = frame_buffer.popleft()
                        dropped_count += 1
                    else:
                        break
                if dropped_count > 0:
                    logger.warning(f"[Buffer Drop] Dropped {dropped_count} old frames - buffer at {len(frame_buffer)}")

            # Process all frames that are old enough (maintain constant delay)
            processed_count = 0
            while frame_buffer:
                oldest = frame_buffer[0]
                if current_time - oldest['timestamp'] >= delay_threshold:
                    frame_data = frame_buffer.popleft()
                    await process_buffered_frame(frame_data, current_time)
                    last_viewer_frame_num = frame_data['frame_num']
                    current_time = time.time()  # Update time after processing
                    processed_count += 1
                else:
                    break  # Oldest frame not ready yet, wait for delay window

            # Sleep briefly to yield to other tasks
            await asyncio.sleep(0.005)  # 5ms - allows 200 buffer processes/sec

        except Exception as e:
            logger.error(f"Buffer processor loop error: {e}")


def encode_frame_to_jpeg(frame_rgb: np.ndarray) -> bytes:
    """Encode frame to JPEG bytes (blocking, meant for thread pool)."""
    # Scale down frame for faster encoding if needed
    h, w = frame_rgb.shape[:2]
    if w > 960:  # Only scale if large
        scale = 960 / w
        new_h = int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (960, new_h), interpolation=cv2.INTER_LINEAR)

    output_img = Image.fromarray(frame_rgb)
    output_buffer = io.BytesIO()
    output_img.save(output_buffer, format="JPEG", quality=70)  # Reduced quality for speed
    return output_buffer.getvalue()


async def process_buffered_frame(frame_data: dict, process_time: float):
    """Process a single buffered frame and draw detections with contours."""
    global selected_detection, current_occlusion_mask

    try:
        proc_start = time.time()

        # Use frame directly without heavy pipeline processing for viewer speed
        # (pipeline processing adds object replacement which viewer doesn't need)
        t_copy = time.time()
        processed = frame_data['frame'].copy()
        copy_time = (time.time() - t_copy) * 1000

        # Draw ALL detections with segmentation contours
        contour_count = 0
        bbox_count = 0
        t_draw = time.time()
        for detection in frame_data['detections']:
            # Draw segmentation contour if available, otherwise use bbox
            if detection.contour and len(detection.contour) > 2:
                # Convert contour to numpy array format for cv2.polylines
                contour_points = np.array(detection.contour, dtype=np.int32)
                # Yellow for all detections
                cv2.polylines(processed, [contour_points], isClosed=True, color=(0, 255, 255), thickness=2)
                contour_count += 1
            else:
                # Fallback to bounding box
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 2)
                bbox_count += 1

            # Draw label
            x1, y1, x2, y2 = detection.bbox
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

        # Draw selected detection in green with thicker contour
        if selected_detection:
            if selected_detection.contour and len(selected_detection.contour) > 2:
                contour_points = np.array(selected_detection.contour, dtype=np.int32)
                cv2.polylines(processed, [contour_points], isClosed=True, color=(0, 255, 0), thickness=3)
            else:
                x1, y1, x2, y2 = selected_detection.bbox
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 3)

            x1, y1, x2, y2 = selected_detection.bbox
            cv2.putText(
                processed,
                f"Replacing: {selected_detection.class_name}",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        draw_time = (time.time() - t_draw) * 1000

        # Convert to RGB
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed = processed[:, :, ::-1]  # BGR to RGB

        # Encode JPEG in thread pool to avoid blocking event loop
        t_encode = time.time()
        loop = asyncio.get_event_loop()
        output_bytes = await loop.run_in_executor(
            encode_executor,
            encode_frame_to_jpeg,
            processed,
        )
        encode_time = (time.time() - t_encode) * 1000

        proc_duration = (time.time() - proc_start) * 1000
        age_ms = (process_time - frame_data['timestamp']) * 1000
        logger.info(f"[Viewer] Frame {frame_data['frame_num']} | Age: {age_ms:.0f}ms | Total: {proc_duration:.1f}ms [copy:{copy_time:.1f} draw:{draw_time:.1f} encode:{encode_time:.1f}] | Contours: {contour_count} | Buffer: {len(frame_buffer)}")

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
