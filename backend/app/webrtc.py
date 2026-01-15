"""
WebRTC Video Track Processing
"""
import asyncio
import logging
from typing import Optional

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

from .pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


class VideoTransformTrack(VideoStreamTrack):
    """
    A video track that transforms incoming frames using the processing pipeline.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, pipeline: ProcessingPipeline):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self._frame_count = 0

    async def recv(self) -> VideoFrame:
        """Receive and transform a video frame."""
        frame = await self.track.recv()
        self._frame_count += 1

        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Process frame through pipeline
        try:
            processed = await self.pipeline.process_frame(img, self._frame_count)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            processed = img

        # Convert back to VideoFrame
        new_frame = VideoFrame.from_ndarray(processed, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame
