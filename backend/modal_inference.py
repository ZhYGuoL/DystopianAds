"""
Modal SAM 2 Inference Service
Deploy with: modal deploy modal_inference.py
"""
import modal

app = modal.App("dystopian-ads")

# Define the container image with all dependencies
sam2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "numpy",
        "pillow",
        "opencv-python-headless",
    )
    .run_commands(
        # Install SAM 2 from GitHub
        "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
    )
)


@app.cls(gpu="T4", image=sam2_image, container_idle_timeout=300)
class SAM2Inference:
    """
    SAM 2 inference class that runs on Modal GPU.
    """

    @modal.enter()
    def load_model(self):
        """Load SAM 2 model when container starts."""
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Use the smallest model for speed
        # Available: sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base_plus, sam2_hiera_large
        model_cfg = "sam2_hiera_tiny.yaml"
        checkpoint = "sam2_hiera_tiny.pt"

        # Download checkpoint if needed
        import urllib.request
        import os

        checkpoint_path = f"/tmp/{checkpoint}"
        if not os.path.exists(checkpoint_path):
            url = f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/{checkpoint}"
            print(f"Downloading {checkpoint}...")
            urllib.request.urlretrieve(url, checkpoint_path)

        # Build model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device
        print(f"SAM 2 loaded on {device}")

    @modal.method()
    def segment(self, frame_bytes: bytes, point_x: int, point_y: int) -> dict:
        """
        Segment an object at the given point.

        Args:
            frame_bytes: JPEG-encoded image bytes
            point_x: X coordinate of the click point
            point_y: Y coordinate of the click point

        Returns:
            dict with mask bytes, shape, and confidence score
        """
        import numpy as np
        from PIL import Image
        import io

        # Decode frame
        image = np.array(Image.open(io.BytesIO(frame_bytes)))

        # Set image for prediction
        self.predictor.set_image(image)

        # Run prediction with point prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([[point_x, point_y]]),
            point_labels=np.array([1]),  # 1 = foreground
            multimask_output=True,
        )

        # Get the best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])

        return {
            "mask": best_mask.tobytes(),
            "shape": best_mask.shape,
            "score": best_score,
        }

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        return {
            "status": "healthy",
            "device": self.device,
        }


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the SAM 2 inference."""
    import numpy as np
    from PIL import Image
    import io

    # Create a test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(test_img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    frame_bytes = buffer.getvalue()

    # Run inference
    sam2 = SAM2Inference()
    result = sam2.segment.remote(frame_bytes, 320, 240)

    print(f"Mask shape: {result['shape']}")
    print(f"Score: {result['score']:.3f}")
