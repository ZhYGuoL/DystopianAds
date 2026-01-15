import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import CapturePage from './pages/CapturePage'
import ViewerPage from './pages/ViewerPage'

function HomePage() {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-8">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-white mb-4">DystopianAds</h1>
        <p className="text-gray-400 mb-12 text-lg">Live Object Replacement Demo</p>

        <div className="flex gap-8 justify-center">
          <Link
            to="/capture"
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-6 px-12 rounded-xl text-xl transition transform hover:scale-105"
          >
            Open Capture Window
          </Link>
          <Link
            to="/viewer"
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-6 px-12 rounded-xl text-xl transition transform hover:scale-105"
          >
            Open Viewer Window
          </Link>
        </div>

        <div className="mt-16 bg-gray-800 rounded-lg p-8 max-w-2xl mx-auto text-left">
          <h2 className="text-xl font-semibold text-white mb-4">How to Use</h2>
          <ol className="list-decimal list-inside text-gray-300 space-y-3">
            <li>Open <strong>Capture Window</strong> in one browser tab</li>
            <li>Open <strong>Viewer Window</strong> in another browser tab</li>
            <li>In Capture: Select an ad asset (e.g., Coca-Cola)</li>
            <li>In Capture: Click on an object to replace (e.g., a can)</li>
            <li>Watch the Viewer window show the replaced object</li>
          </ol>
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/capture" element={<CapturePage />} />
        <Route path="/viewer" element={<ViewerPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
