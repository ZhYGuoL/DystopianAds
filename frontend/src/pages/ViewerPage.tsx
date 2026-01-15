import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'

export default function ViewerPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [status, setStatus] = useState('Disconnected')
  const [fps, setFps] = useState(0)
  const [frameCount, setFrameCount] = useState(0)

  const imgRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const frameCountRef = useRef(0)
  const lastFpsTimeRef = useRef(Date.now())

  useEffect(() => {
    // Auto-connect on mount
    connect()

    return () => {
      wsRef.current?.close()
    }
  }, [])

  const connect = () => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/viewer`)

    ws.onopen = () => {
      setIsConnected(true)
      setStatus('Connected - waiting for frames')
    }

    ws.onmessage = (event) => {
      // Receive processed frame as blob
      if (event.data instanceof Blob) {
        const url = URL.createObjectURL(event.data)
        if (imgRef.current) {
          // Revoke previous URL to prevent memory leak
          if (imgRef.current.src.startsWith('blob:')) {
            URL.revokeObjectURL(imgRef.current.src)
          }
          imgRef.current.src = url
        }

        frameCountRef.current++
        setFrameCount((c) => c + 1)

        // Calculate FPS
        const now = Date.now()
        if (now - lastFpsTimeRef.current >= 1000) {
          setFps(frameCountRef.current)
          frameCountRef.current = 0
          lastFpsTimeRef.current = now
        }

        setStatus('Receiving processed frames')
      } else {
        // JSON message (status updates)
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'status') {
            setStatus(data.message || 'Processing...')
          }
        } catch {
          // Ignore
        }
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      setStatus('Disconnected')
    }

    ws.onerror = () => {
      setStatus('Connection error')
    }

    wsRef.current = ws
  }

  const disconnect = () => {
    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white">Viewer Window</h1>
            <p className="text-gray-400">Processed video output</p>
          </div>
          <Link to="/" className="text-blue-400 hover:text-blue-300">← Back</Link>
        </div>

        {/* Video display */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
            <img
              ref={imgRef}
              alt="Processed video"
              className="w-full h-full object-contain"
            />

            {/* Status overlay */}
            <div className="absolute top-4 left-4 bg-black/70 text-white text-sm px-3 py-2 rounded">
              {status}
            </div>

            {/* Stats overlay */}
            <div className="absolute top-4 right-4 bg-black/70 text-sm px-3 py-2 rounded">
              <div className={isConnected ? 'text-green-400' : 'text-red-400'}>
                {isConnected ? '● Connected' : '○ Disconnected'}
              </div>
              <div className="text-gray-300">
                {fps} FPS | {frameCount} frames
              </div>
            </div>

            {/* Waiting message */}
            {isConnected && frameCount === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-gray-400 text-xl animate-pulse">
                  Waiting for frames from Capture window...
                </div>
              </div>
            )}

            {/* Not connected message */}
            {!isConnected && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-gray-400 text-xl mb-4">Not connected</div>
                  <button
                    onClick={connect}
                    className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded transition"
                  >
                    Reconnect
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="mt-6 flex gap-4">
          {isConnected ? (
            <button
              onClick={disconnect}
              className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded transition"
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={connect}
              className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded transition"
            >
              Connect
            </button>
          )}

          <div className="flex-1 bg-gray-800 rounded-lg px-4 py-2 text-gray-300">
            <span className="text-gray-500">Tip:</span> Open the Capture window in another tab to start streaming
          </div>
        </div>
      </div>
    </div>
  )
}
