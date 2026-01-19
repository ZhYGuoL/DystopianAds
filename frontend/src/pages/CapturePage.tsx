import { useState, useRef, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'

interface AdAsset {
  id: string
  name: string
  color: string
}

interface Detection {
  id: number
  class: string
  confidence: number
  bbox: [number, number, number, number] // x1, y1, x2, y2
  center: [number, number]
  track_id: number | null
  contour: [number, number][] | null // Segmentation contour points
}

const AD_ASSETS: AdAsset[] = [
  { id: 'coke', name: 'Coca-Cola', color: 'bg-red-600' },
  { id: 'pepsi', name: 'Pepsi', color: 'bg-blue-600' },
  { id: 'sprite', name: 'Sprite', color: 'bg-green-600' },
  { id: 'fanta', name: 'Fanta', color: 'bg-orange-500' },
]

export default function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const [isConnected, setIsConnected] = useState(false)
  const [selectedAd, setSelectedAd] = useState<AdAsset | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null)
  const [status, setStatus] = useState('Disconnected')
  const [fps, setFps] = useState(0)
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 })

  const frameCountRef = useRef(0)
  const lastFpsTimeRef = useRef(Date.now())

  // Initialize webcam
  useEffect(() => {
    async function setupWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            setVideoDimensions({
              width: videoRef.current!.videoWidth,
              height: videoRef.current!.videoHeight,
            })
          }
        }
      } catch (err) {
        console.error('Webcam error:', err)
        setStatus('Webcam access denied')
      }
    }
    setupWebcam()

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  // Connect to WebSocket
  const connect = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/capture`)

    ws.onopen = () => {
      setIsConnected(true)
      setStatus('Connected - YOLO detecting objects...')

      const sendFrame = () => {
        if (ws.readyState !== WebSocket.OPEN) return
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')!

        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        ctx.drawImage(video, 0, 0)

        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              ws.send(blob)
              frameCountRef.current++

              const now = Date.now()
              if (now - lastFpsTimeRef.current >= 1000) {
                setFps(frameCountRef.current)
                frameCountRef.current = 0
                lastFpsTimeRef.current = now
              }
            }
          },
          'image/jpeg',
          0.8
        )

        setTimeout(() => requestAnimationFrame(sendFrame), 33)
      }

      const video = videoRef.current
      if (video && video.readyState >= 2) {
        sendFrame()
      } else if (video) {
        video.addEventListener('loadeddata', sendFrame, { once: true })
      }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'detections') {
          setDetections(data.data)
          if (data.data.length > 0) {
            setStatus(`Detected ${data.data.length} object(s) - click to select`)
          } else {
            setStatus('No objects detected - try a bottle or cup')
          }
        } else if (data.type === 'selected') {
          setStatus(`Selected: ${data.detection.class} - replacing with ${selectedAd?.name || 'ad'}`)
        }
      } catch {
        // Not JSON, ignore
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      setStatus('Disconnected')
      setDetections([])
    }

    ws.onerror = () => {
      setStatus('Connection error')
    }

    wsRef.current = ws
  }, [selectedAd])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
    setDetections([])
  }, [])

  // Handle click on video or detection box
  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!selectedAd) {
      alert('Please select an ad asset first')
      return
    }

    const container = containerRef.current
    if (!container) return

    const rect = container.getBoundingClientRect()
    const scaleX = videoDimensions.width / rect.width
    const scaleY = videoDimensions.height / rect.height

    const x = Math.round((e.clientX - rect.left) * scaleX)
    const y = Math.round((e.clientY - rect.top) * scaleY)

    // Find if clicked on a detection
    const clicked = detections.find(d => {
      const [x1, y1, x2, y2] = d.bbox
      return x >= x1 && x <= x2 && y >= y1 && y <= y2
    })

    if (clicked) {
      setSelectedDetection(clicked)
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'click',
        x,
        y,
        adId: selectedAd.id,
      }))
    }
  }, [selectedAd, detections, videoDimensions])

  const handleAdSelect = useCallback((ad: AdAsset) => {
    setSelectedAd(ad)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'select_ad',
        adId: ad.id,
      }))
    }
  }, [])

  const handleReset = useCallback(() => {
    setSelectedDetection(null)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }))
      setStatus('Selection reset')
    }
  }, [])

  // Calculate box position for overlay (fallback when no contour)
  const getBoxStyle = (bbox: [number, number, number, number]) => {
    const [x1, y1, x2, y2] = bbox
    const container = containerRef.current
    if (!container) return {}

    const rect = container.getBoundingClientRect()
    const scaleX = rect.width / videoDimensions.width
    const scaleY = rect.height / videoDimensions.height

    return {
      left: `${x1 * scaleX}px`,
      top: `${y1 * scaleY}px`,
      width: `${(x2 - x1) * scaleX}px`,
      height: `${(y2 - y1) * scaleY}px`,
    }
  }

  // Convert contour points to scaled SVG polygon points string
  const getContourPoints = (contour: [number, number][]) => {
    const container = containerRef.current
    if (!container) return ''

    const rect = container.getBoundingClientRect()
    const scaleX = rect.width / videoDimensions.width
    const scaleY = rect.height / videoDimensions.height

    return contour.map(([x, y]) => `${x * scaleX},${y * scaleY}`).join(' ')
  }

  // Get label position for contour (use first point or bbox top-left)
  const getLabelPosition = (det: Detection) => {
    const container = containerRef.current
    if (!container) return { left: 0, top: 0 }

    const rect = container.getBoundingClientRect()
    const scaleX = rect.width / videoDimensions.width
    const scaleY = rect.height / videoDimensions.height

    if (det.contour && det.contour.length > 0) {
      // Find topmost point for label
      const topPoint = det.contour.reduce((min, pt) => pt[1] < min[1] ? pt : min, det.contour[0])
      return {
        left: topPoint[0] * scaleX,
        top: topPoint[1] * scaleY - 24,
      }
    }
    // Fallback to bbox
    return {
      left: det.bbox[0] * scaleX,
      top: det.bbox[1] * scaleY - 24,
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white">Capture Window</h1>
            <p className="text-gray-400">YOLO detects objects - click one to replace</p>
          </div>
          <Link to="/" className="text-blue-400 hover:text-blue-300">← Back</Link>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Video with detection overlay */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg p-4">
              <div
                ref={containerRef}
                className="relative cursor-crosshair"
                onClick={handleClick}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full rounded-lg bg-black"
                />

                {/* Detection contours/boxes - SVG overlay */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ overflow: 'visible' }}
                >
                  {detections.map((det) => {
                    const isSelected = selectedDetection?.id === det.id
                    const strokeColor = isSelected ? '#22c55e' : '#facc15' // green-500 / yellow-400
                    const fillColor = isSelected ? 'rgba(34, 197, 94, 0.2)' : 'rgba(250, 204, 21, 0.1)'

                    if (det.contour && det.contour.length > 2) {
                      // Draw contour polygon
                      return (
                        <polygon
                          key={det.id}
                          points={getContourPoints(det.contour)}
                          fill={fillColor}
                          stroke={strokeColor}
                          strokeWidth="2"
                        />
                      )
                    } else {
                      // Fallback to rectangle
                      const style = getBoxStyle(det.bbox)
                      return (
                        <rect
                          key={det.id}
                          x={parseFloat(style.left as string) || 0}
                          y={parseFloat(style.top as string) || 0}
                          width={parseFloat(style.width as string) || 0}
                          height={parseFloat(style.height as string) || 0}
                          fill={fillColor}
                          stroke={strokeColor}
                          strokeWidth="2"
                        />
                      )
                    }
                  })}
                </svg>

                {/* Detection labels */}
                {detections.map((det) => {
                  const pos = getLabelPosition(det)
                  const isSelected = selectedDetection?.id === det.id
                  return (
                    <div
                      key={`label-${det.id}`}
                      className={`absolute text-xs px-1 rounded pointer-events-none ${
                        isSelected
                          ? 'bg-green-500 text-white'
                          : 'bg-yellow-400 text-black'
                      }`}
                      style={{ left: pos.left, top: pos.top }}
                    >
                      {det.class} {Math.round(det.confidence * 100)}%
                      {det.track_id !== null && <span className="ml-1 opacity-70">#{det.track_id}</span>}
                    </div>
                  )
                })}

                {/* Status overlay */}
                <div className="absolute top-2 left-2 bg-black/80 text-white text-sm px-3 py-2 rounded max-w-xs">
                  {status}
                </div>

                {/* FPS counter */}
                {isConnected && (
                  <div className="absolute top-2 right-2 bg-black/80 text-sm px-3 py-2 rounded">
                    <div className="text-green-400">{fps} FPS</div>
                    <div className="text-gray-400">{detections.length} objects</div>
                  </div>
                )}

                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Connection */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-3">Connection</h2>
              {!isConnected ? (
                <button
                  onClick={connect}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded transition"
                >
                  Connect
                </button>
              ) : (
                <button
                  onClick={disconnect}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded transition"
                >
                  Disconnect
                </button>
              )}
            </div>

            {/* Ad Selection */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-3">Replace With</h2>
              <div className="grid grid-cols-2 gap-2">
                {AD_ASSETS.map((ad) => (
                  <button
                    key={ad.id}
                    onClick={() => handleAdSelect(ad)}
                    className={`${ad.color} ${
                      selectedAd?.id === ad.id ? 'ring-2 ring-white' : ''
                    } text-white font-medium py-3 px-2 rounded text-sm transition hover:opacity-90`}
                  >
                    {ad.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Reset */}
            {selectedDetection && (
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-300 mb-2">
                  Selected: <span className="text-green-400">{selectedDetection.class}</span>
                </div>
                <button
                  onClick={handleReset}
                  className="w-full bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded transition"
                >
                  Reset Selection
                </button>
              </div>
            )}

            {/* Instructions */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-3">How it works</h2>
              <ol className="text-gray-300 text-sm space-y-2 list-decimal list-inside">
                <li>Click Connect to start</li>
                <li>YOLO detects bottles, cups, etc.</li>
                <li>Select an ad (Coke, Pepsi...)</li>
                <li>Click a yellow box to replace</li>
                <li>View result in Viewer tab</li>
              </ol>
            </div>

            {/* Detection classes */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white mb-2">Detects</h2>
              <div className="flex flex-wrap gap-2">
                {['bottle', 'cup', 'bowl', 'banana', 'apple'].map((cls) => (
                  <span key={cls} className="bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded">
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
