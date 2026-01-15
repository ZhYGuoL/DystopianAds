# DystopianAds

Live video object replacement demo - replace objects in your webcam feed with branded ad assets in real-time.

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- (Optional) Modal account for GPU-accelerated SAM 2 segmentation

### Development Setup

```bash
# 1. Install frontend dependencies
cd frontend && npm install && cd ..

# 2. Set up Python backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_assets.py
cd ..
```

### Running the Demo

**Terminal 1 - Backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## How to Use

1. **Select an ad asset** from the left panel (Coca-Cola, Pepsi, etc.)
2. **Click "Connect to Server"** to establish WebRTC connection
3. **Click on an object** in the video that you want to replace (e.g., a can)
4. Watch as the system tracks and replaces the object in real-time
5. Use **"Reset Selection"** to clear and try a different object

## Architecture

```
Browser (React + Vite)          Backend (Python + FastAPI)
┌─────────────────────┐         ┌─────────────────────────┐
│  Webcam Capture     │◄───────►│  WebRTC Server (aiortc) │
│  Click Handler      │ WebRTC  │  Processing Pipeline    │
│  Canvas Display     │         │  ├─ Segmentation        │
│  Ad Selector UI     │         │  ├─ Inpainting          │
└─────────────────────┘         │  └─ Compositing         │
                                └─────────────────────────┘
                                          │
                                          ▼ (Optional)
                                ┌─────────────────────────┐
                                │  Modal (GPU)            │
                                │  SAM 2 Inference        │
                                └─────────────────────────┘
```

## Cloud GPU Setup (Optional)

For production-quality segmentation using SAM 2:

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate
modal setup

# 3. Deploy the inference service
cd backend
modal deploy modal_inference.py

# 4. Enable Modal in the code
# Edit backend/app/segmentation.py
# Set USE_LOCAL_FALLBACK = False
```

## Tech Stack

- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS
- **Backend:** Python, FastAPI, aiortc (WebRTC)
- **Segmentation:** OpenCV GrabCut (local) / SAM 2 via Modal (GPU)
- **Compositing:** OpenCV inpainting + alpha blending

## Project Structure

```
DystopianAds/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamCapture.tsx
│   │   │   └── AdSelector.tsx
│   │   ├── hooks/
│   │   │   └── useWebRTC.ts
│   │   └── App.tsx
│   └── public/assets/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI server
│   │   ├── webrtc.py         # Video track processing
│   │   ├── pipeline.py       # Processing pipeline
│   │   ├── segmentation.py   # Object segmentation
│   │   └── compositor.py     # Frame compositing
│   ├── modal_inference.py    # Modal SAM 2 service
│   └── assets/               # Ad asset images
├── docker-compose.yml
└── Makefile
```

## Development Notes

### Local Fallback Mode
By default, the app uses OpenCV's GrabCut for segmentation (no GPU required). This works for testing but produces lower-quality masks than SAM 2.

### Adding Custom Ad Assets
1. Add PNG images (with transparency) to `backend/assets/`
2. Update `AD_ASSETS` array in `frontend/src/App.tsx`
3. Restart both servers

### Improving Tracking
The current tracking is simple (reuse previous mask between segmentation frames). For better results:
- Implement optical flow tracking
- Reduce `segment_interval` in `pipeline.py`
- Use SAM 2 video segmentation mode

## License

MIT
