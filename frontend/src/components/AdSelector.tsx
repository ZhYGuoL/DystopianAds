export interface AdAsset {
  id: string
  name: string
  imageUrl: string
}

interface AdSelectorProps {
  assets: AdAsset[]
  selectedId: string | null
  onSelect: (asset: AdAsset) => void
}

export default function AdSelector({ assets, selectedId, onSelect }: AdSelectorProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {assets.map((asset) => (
        <button
          key={asset.id}
          onClick={() => onSelect(asset)}
          className={`
            relative p-3 rounded-lg border-2 transition-all
            ${selectedId === asset.id
              ? 'border-blue-500 bg-blue-500/20'
              : 'border-gray-600 bg-gray-700/50 hover:border-gray-500 hover:bg-gray-700'
            }
          `}
        >
          <div className="aspect-square bg-gray-800 rounded-md mb-2 flex items-center justify-center overflow-hidden">
            <img
              src={asset.imageUrl}
              alt={asset.name}
              className="w-full h-full object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
            <span className="text-4xl absolute">{getEmoji(asset.id)}</span>
          </div>
          <p className="text-sm text-white font-medium truncate">{asset.name}</p>

          {selectedId === asset.id && (
            <div className="absolute top-2 right-2 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
          )}
        </button>
      ))}
    </div>
  )
}

function getEmoji(id: string): string {
  const emojis: Record<string, string> = {
    coke: '🥤',
    pepsi: '🥤',
    sprite: '🥤',
    fanta: '🥤',
  }
  return emojis[id] || '📦'
}
