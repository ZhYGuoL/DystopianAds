"""
DystopianAds Backend - FastAPI + WebSocket + YOLO Detection
"""
import asyncio
import json
import logging
import io
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


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global detector
    logger.info("Initializing YOLO detector...")
    detector = DetectionService(model_size="n")  # Use nano for speed
    logger.info("YOLO detector ready")


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
    global frame_count, current_detections, selected_detection

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

                # Run YOLO detection every few frames
                if detector and frame_count % 3 == 0:  # Every 3rd frame
                    # Use lower confidence and detect ALL objects for debugging
                    current_detections = detector.detect(
                        frame,
                        confidence_threshold=0.25,  # Lower threshold
                        target_classes_only=False,  # Show ALL detections
                    )
                    if current_detections:
                        logger.info(f"Detected: {[(d.class_name, d.confidence) for d in current_detections]}")

                    # Send detections to capture client for visualization
                    # Convert numpy types to Python native types for JSON serialization
                    det_data = [
                        {
                            "id": i,
                            "class": d.class_name,
                            "confidence": round(float(d.confidence), 2),
                            "bbox": [int(x) for x in d.bbox],
                            "center": [int(x) for x in d.center],
                        }
                        for i, d in enumerate(current_detections)
                    ]
                    await websocket.send_json({
                        "type": "detections",
                        "data": det_data,
                    })

                # Process and broadcast to viewers
                asyncio.create_task(process_and_broadcast(frame, frame_count))

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
                            logger.info(f"Selected: {clicked_det.class_name} at {clicked_det.bbox}")

                            # Use detection mask or center point for pipeline
                            if clicked_det.mask is not None:
                                pipeline.set_mask_directly(clicked_det.mask, clicked_det.bbox)
                            else:
                                pipeline.set_target_point(clicked_det.center[0], clicked_det.center[1])

                            if "adId" in data:
                                pipeline.set_ad_asset(data["adId"])

                            await websocket.send_json({
                                "type": "selected",
                                "detection": {
                                    "class": clicked_det.class_name,
                                    "bbox": clicked_det.bbox,
                                }
                            })
                        else:
                            # No detection at point, use point directly
                            pipeline.set_target_point(x, y)
                            if "adId" in data:
                                pipeline.set_ad_asset(data["adId"])

                    elif data.get("type") == "select_ad":
                        pipeline.set_ad_asset(data["adId"])

                    elif data.get("type") == "reset":
                        pipeline.reset()
                        selected_detection = None

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


async def process_and_broadcast(frame: np.ndarray, frame_num: int):
    """Process a frame and broadcast to all viewers."""
    global selected_detection

    try:
        # Process frame through pipeline
        processed = await pipeline.process_frame(frame, frame_num)

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
