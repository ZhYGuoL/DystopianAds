import { useState, useCallback, useRef, useEffect } from 'react'

interface UseWebRTCReturn {
  isConnected: boolean
  isProcessing: boolean
  processedFrame: string | null
  connect: () => Promise<void>
  disconnect: () => void
  sendClick: (x: number, y: number) => void
  sendAdSelection: (adId: string) => void
  resetSelection: () => void
}

export function useWebRTC(): UseWebRTCReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedFrame, setProcessedFrame] = useState<string | null>(null)

  const peerConnection = useRef<RTCPeerConnection | null>(null)
  const dataChannel = useRef<RTCDataChannel | null>(null)
  const localStream = useRef<MediaStream | null>(null)
  const frameCapture = useRef<number | null>(null)
  const remoteVideo = useRef<HTMLVideoElement | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameCapture.current) {
        cancelAnimationFrame(frameCapture.current)
      }
    }
  }, [])

  const disconnect = useCallback(() => {
    if (frameCapture.current) {
      cancelAnimationFrame(frameCapture.current)
      frameCapture.current = null
    }

    if (dataChannel.current) {
      dataChannel.current.close()
      dataChannel.current = null
    }

    if (peerConnection.current) {
      peerConnection.current.close()
      peerConnection.current = null
    }

    if (localStream.current) {
      localStream.current.getTracks().forEach((track) => track.stop())
      localStream.current = null
    }

    if (remoteVideo.current) {
      remoteVideo.current.srcObject = null
      remoteVideo.current = null
    }

    setIsConnected(false)
    setIsProcessing(false)
    setProcessedFrame(null)
  }, [])

  const connect = useCallback(async () => {
    try {
      // Get webcam stream
      localStream.current = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      })

      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      })
      peerConnection.current = pc

      // Add local video track
      localStream.current.getTracks().forEach((track) => {
        pc.addTrack(track, localStream.current!)
      })

      // Create data channel for messages (click coordinates, ad selection, etc.)
      const dc = pc.createDataChannel('messages', { ordered: true })
      dataChannel.current = dc

      dc.onopen = () => {
        console.log('Data channel opened')
      }

      dc.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'status') {
            setIsProcessing(data.processing)
          }
        } catch {
          console.error('Failed to parse data channel message')
        }
      }

      // Handle incoming video track (processed frames)
      pc.ontrack = (event) => {
        console.log('Received remote track:', event.track.kind)
        const remoteStream = event.streams[0]

        // Create a video element to capture frames
        const video = document.createElement('video')
        video.srcObject = remoteStream
        video.autoplay = true
        video.playsInline = true
        video.muted = true
        remoteVideo.current = video

        // Need to play video to receive frames
        video.play().catch(console.error)

        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!

        const captureFrame = () => {
          if (video.readyState >= 2 && video.videoWidth > 0) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            ctx.drawImage(video, 0, 0)
            setProcessedFrame(canvas.toDataURL('image/jpeg', 0.7))
          }
          // Keep capturing while pc exists
          if (peerConnection.current) {
            frameCapture.current = requestAnimationFrame(captureFrame)
          }
        }

        video.onloadeddata = () => {
          console.log('Remote video loaded, starting capture')
          captureFrame()
        }
      }

      // ICE candidate handling
      pc.onicecandidate = async (event) => {
        if (event.candidate === null) {
          // ICE gathering complete, send offer to server
          const offer = pc.localDescription!
          console.log('Sending offer to server...')

          try {
            const response = await fetch('/api/offer', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
              }),
            })

            if (!response.ok) {
              throw new Error(`Server returned ${response.status}`)
            }

            const answer = await response.json()
            console.log('Received answer from server')
            await pc.setRemoteDescription(new RTCSessionDescription(answer))
            setIsConnected(true)
          } catch (err) {
            console.error('Failed to connect to server:', err)
            disconnect()
          }
        }
      }

      pc.onconnectionstatechange = () => {
        console.log('Connection state:', pc.connectionState)
        if (pc.connectionState === 'connected') {
          setIsConnected(true)
        } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
          setIsConnected(false)
          setIsProcessing(false)
        }
      }

      pc.oniceconnectionstatechange = () => {
        console.log('ICE connection state:', pc.iceConnectionState)
      }

      // Create and set local offer
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      console.log('Local offer created, waiting for ICE gathering...')

    } catch (err) {
      console.error('Failed to connect:', err)
      disconnect()
    }
  }, [disconnect])

  const sendClick = useCallback((x: number, y: number) => {
    if (dataChannel.current?.readyState === 'open') {
      dataChannel.current.send(JSON.stringify({
        type: 'click',
        x,
        y,
      }))
      setIsProcessing(true)
    }
  }, [])

  const sendAdSelection = useCallback((adId: string) => {
    if (dataChannel.current?.readyState === 'open') {
      dataChannel.current.send(JSON.stringify({
        type: 'select_ad',
        adId,
      }))
    }
  }, [])

  const resetSelection = useCallback(() => {
    if (dataChannel.current?.readyState === 'open') {
      dataChannel.current.send(JSON.stringify({
        type: 'reset',
      }))
      setIsProcessing(false)
    }
  }, [])

  return {
    isConnected,
    isProcessing,
    processedFrame,
    connect,
    disconnect,
    sendClick,
    sendAdSelection,
    resetSelection,
  }
}
