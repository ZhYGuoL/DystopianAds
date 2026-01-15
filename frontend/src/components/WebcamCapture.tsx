import { useRef, useEffect, useState, useCallback } from 'react'

interface WebcamCaptureProps {
  onVideoClick: (x: number, y: number) => void
  processedFrame: string | null
  isConnected: boolean
  targetPoint: { x: number; y: number } | null
}

export default function WebcamCapture({
  onVideoClick,
  processedFrame,
  isConnected,
  targetPoint,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hasWebcam, setHasWebcam] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 })

  // Initialize webcam
  useEffect(() => {
    async function setupWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
          audio: false,
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            setVideoDimensions({
              width: videoRef.current!.videoWidth,
              height: videoRef.current!.videoHeight,
            })
            setHasWebcam(true)
          }
        }
      } catch (err) {
        console.error('Failed to access webcam:', err)
        setError('Failed to access webcam. Please allow camera permissions.')
      }
    }

    setupWebcam()

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
        tracks.forEach(track => track.stop())
      }
    }
  }, [])

  // Handle click on video/canvas
  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const container = containerRef.current
    const video = videoRef.current
    if (!container || !video) return

    const rect = container.getBoundingClientRect()
    const scaleX = videoDimensions.width / rect.width
    const scaleY = videoDimensions.height / rect.height

    const x = Math.round((e.clientX - rect.left) * scaleX)
    const y = Math.round((e.clientY - rect.top) * scaleY)

    onVideoClick(x, y)
  }, [onVideoClick, videoDimensions])

  if (error) {
    return (
      <div className="bg-red-900/50 border border-red-500 rounded-lg p-8 text-center">
        <p className="text-red-300">{error}</p>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="relative cursor-crosshair"
      onClick={handleClick}
      style={{ aspectRatio: `${videoDimensions.width}/${videoDimensions.height}` }}
    >
      {/* Video element - shown when not connected or no processed frame */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`w-full h-full rounded-lg bg-gray-900 object-cover ${processedFrame ? 'hidden' : ''}`}
      />

      {/* Canvas for processed frames */}
      {processedFrame && (
        <img
          src={processedFrame}
          alt="Processed video"
          className="w-full h-full rounded-lg bg-gray-900 object-cover"
        />
      )}

      {/* Target point indicator overlay */}
      {targetPoint && (
        <div
          className="absolute w-6 h-6 border-3 border-green-500 rounded-full pointer-events-none"
          style={{
            left: `${(targetPoint.x / videoDimensions.width) * 100}%`,
            top: `${(targetPoint.y / videoDimensions.height) * 100}%`,
            transform: 'translate(-50%, -50%)',
            boxShadow: '0 0 10px rgba(0, 255, 0, 0.5)',
            border: '3px solid #00ff00',
          }}
        >
          <div
            className="absolute w-2 h-2 bg-green-500 rounded-full"
            style={{
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          />
        </div>
      )}

      {/* Connection status overlay */}
      {isConnected && (
        <div className="absolute top-4 right-4 bg-green-600/80 text-white text-xs px-2 py-1 rounded">
          LIVE
        </div>
      )}

      {/* Loading indicator */}
      {!hasWebcam && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded-lg">
          <div className="text-gray-300">Loading webcam...</div>
        </div>
      )}

      {/* Hidden canvas ref for future use */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
