import type { PhotoRegion } from '@afilmory/builder'
import { clsxm } from '@afilmory/utils'
import type { CSSProperties } from 'react'
import { useEffect, useMemo, useState } from 'react'

interface PhotoRegionsOverlayProps {
  regions: PhotoRegion[]
  photoWidth?: number
  photoHeight?: number
  orientation?: number
  accentColor?: string
  showAllBoxes?: boolean
  interactive?: boolean
}

interface RegionBounds {
  left: number
  top: number
  width: number
  height: number
}

interface RegionViewModel {
  id: string
  name: string
  bounds: RegionBounds
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

export const PhotoRegionsOverlay = ({
  regions,
  photoWidth,
  photoHeight,
  orientation,
  accentColor,
  showAllBoxes = false,
  interactive = true,
}: PhotoRegionsOverlayProps) => {
  const [hoveredRegionId, setHoveredRegionId] = useState<string | null>(null)

  useEffect(() => {
    if (!interactive) {
      setHoveredRegionId(null)
    }
  }, [interactive])

  const regionViews = useMemo<RegionViewModel[]>(() => {
    return regions
      .map((region, index) => {
        const bounds = getRegionBounds(region, photoWidth, photoHeight, orientation)
        if (!bounds) {
          return null
        }

        return {
          id: `${region.name}-${index}`,
          name: region.name,
          bounds,
        }
      })
      .filter((region): region is RegionViewModel => region !== null)
  }, [orientation, photoHeight, photoWidth, regions])

  if (regionViews.length === 0) {
    return null
  }

  const accentStyle = accentColor ? ({ '--color-accent': accentColor } as CSSProperties) : undefined

  return (
    <div
      className="pointer-events-none absolute inset-0 z-20"
      style={accentStyle}
      onMouseLeave={() => setHoveredRegionId(null)}
    >
      {regionViews.map((region) => {
        const isHovered = hoveredRegionId === region.id
        const isActive = isHovered
        const isMuted = showAllBoxes && !isActive
        const showBox = showAllBoxes || isHovered
        const showLabel = showAllBoxes || isHovered
        const regionCenterX = region.bounds.left + region.bounds.width / 2
        const labelAnchor = regionCenterX < 0.18 ? 'left' : regionCenterX > 0.82 ? 'right' : 'center'
        const topGap = region.bounds.top
        const bottomGap = 1 - (region.bounds.top + region.bounds.height)
        const verticalPlacement = topGap >= 0.1 ? 'above' : bottomGap >= 0.1 ? 'below' : 'inside'

        return (
          <div
            key={region.id}
            className="absolute"
            style={{
              left: `${region.bounds.left * 100}%`,
              top: `${region.bounds.top * 100}%`,
              width: `${region.bounds.width * 100}%`,
              height: `${region.bounds.height * 100}%`,
            }}
          >
            {interactive && (
              <div
                className="pointer-events-auto absolute inset-0 cursor-pointer"
                onPointerEnter={() => setHoveredRegionId(region.id)}
                onPointerLeave={() => {
                  if (!showAllBoxes) {
                    setHoveredRegionId(current => (current === region.id ? null : current))
                  }
                }}
              />
            )}

            <div
              className={clsxm(
                'pointer-events-none absolute inset-0 transition-opacity duration-200 ease-out',
                showBox ? (isMuted ? 'opacity-40' : 'opacity-100') : 'opacity-0',
              )}
            >
              <div
                className={clsxm(
                  'absolute inset-0 border bg-transparent',
                  isActive ? 'border-accent/100' : 'border-white/95',
                )}
                style={{
                  borderWidth: '1.5px',
                  boxShadow: isActive
                    ? '0 0 0 1px rgb(0 0 0 / 0.5), 0 0 18px color-mix(in srgb, var(--color-accent) 45%, transparent)'
                    : '0 0 0 1px rgb(0 0 0 / 0.45), inset 0 0 0 1px rgb(0 0 0 / 0.25)',
                }}
              />
            </div>

            <div
              className={clsxm(
                'pointer-events-none absolute z-30 max-w-[min(18rem,62vw)] transition-all duration-200 ease-out',
                labelAnchor === 'left' ? 'left-0' : labelAnchor === 'right' ? 'right-0' : 'left-1/2 -translate-x-1/2',
                verticalPlacement === 'below'
                  ? 'top-full mt-2'
                  : verticalPlacement === 'inside'
                    ? 'top-2'
                    : 'bottom-full mb-2',
                showLabel
                  ? labelAnchor === 'center'
                    ? '-translate-x-1/2 translate-y-0 opacity-100'
                    : 'translate-y-0 opacity-100'
                  : verticalPlacement === 'below'
                    ? labelAnchor === 'center'
                      ? '-translate-x-1/2 -translate-y-1 opacity-0'
                      : '-translate-y-1 opacity-0'
                    : labelAnchor === 'center'
                      ? '-translate-x-1/2 translate-y-1 opacity-0'
                      : 'translate-y-1 opacity-0',
              )}
            >
              <div
                className={clsxm(
                  'bg-material-ultra-thick rounded-full border px-3 py-1.5 text-xs font-medium text-white shadow-context-menu backdrop-blur-2xl transition-[background-color,border-color,opacity] duration-200 ease-out',
                  isActive ? 'border-accent/20' : 'border-white/12',
                )}
              >
                <span className="flex items-center gap-2">
                  <span
                    className={clsxm('block size-1.5 shrink-0 rounded-full', isActive ? 'bg-accent' : 'bg-white/55')}
                  />
                  <span className="block truncate text-white/92">{region.name}</span>
                </span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function getRegionBounds(
  region: PhotoRegion,
  fallbackWidth?: number,
  fallbackHeight?: number,
  orientation?: number,
): RegionBounds | null {
  const { area } = region
  if (!area) {
    return null
  }

  const rawBounds
    = area.unit.toLowerCase() === 'normalized'
      ? normalizeBounds({
          left: area.x - area.width / 2,
          top: area.y - area.height / 2,
          width: area.width,
          height: area.height,
        })
      : getPixelBounds(region, fallbackWidth, fallbackHeight)

  if (!rawBounds) {
    return null
  }

  return applyOrientation(rawBounds, orientation)
}

function getPixelBounds(region: PhotoRegion, fallbackWidth?: number, fallbackHeight?: number): RegionBounds | null {
  const { area } = region
  if (!area) {
    return null
  }

  const unit = area.unit.toLowerCase()
  if (unit === 'normalized') {
    return null
  }

  const referenceWidth = region.appliedToDimensions?.width ?? fallbackWidth
  const referenceHeight = region.appliedToDimensions?.height ?? fallbackHeight
  if (!referenceWidth || !referenceHeight) {
    return null
  }

  return normalizeBounds({
    left: (area.x - area.width / 2) / referenceWidth,
    top: (area.y - area.height / 2) / referenceHeight,
    width: area.width / referenceWidth,
    height: area.height / referenceHeight,
  })
}

function applyOrientation(bounds: RegionBounds, orientation?: number): RegionBounds | null {
  switch (orientation) {
    case 3:
      return normalizeBounds({
        left: 1 - (bounds.left + bounds.width),
        top: 1 - (bounds.top + bounds.height),
        width: bounds.width,
        height: bounds.height,
      })
    case 6:
      return normalizeBounds({
        left: 1 - (bounds.top + bounds.height),
        top: bounds.left,
        width: bounds.height,
        height: bounds.width,
      })
    case 8:
      return normalizeBounds({
        left: bounds.top,
        top: 1 - (bounds.left + bounds.width),
        width: bounds.height,
        height: bounds.width,
      })
    default:
      return bounds
  }
}

function normalizeBounds(bounds: RegionBounds): RegionBounds | null {
  const left = clamp01(bounds.left)
  const top = clamp01(bounds.top)
  const right = clamp01(bounds.left + bounds.width)
  const bottom = clamp01(bounds.top + bounds.height)

  const width = right - left
  const height = bottom - top
  if (width <= 0 || height <= 0) {
    return null
  }

  return {
    left,
    top,
    width,
    height,
  }
}
