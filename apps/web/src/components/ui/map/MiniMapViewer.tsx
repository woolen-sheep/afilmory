import 'maplibre-gl/dist/maplibre-gl.css'

import type { FC } from 'react'
import { useCallback, useEffect, useRef } from 'react'
import Map, { Marker } from 'react-map-gl/maplibre'

interface MiniMapViewerProps {
  latitude: number
  longitude: number
  className?: string
  zoom?: number
}

export const MiniMapViewer: FC<MiniMapViewerProps> = ({
  latitude,
  longitude,
  className = '',
  zoom = 12,
}) => {
  const mapRef = useRef<any>(null)

  const onMapLoad = useCallback(() => {
    // Map has loaded, can add additional customizations here if needed
  }, [])

  useEffect(() => {
    // If coordinates change, fly to the new location
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [longitude, latitude],
        zoom,
        duration: 1000,
      })
    }
  }, [latitude, longitude, zoom])

  return (
    <div className={`overflow-hidden rounded-md ${className}`}>
      <Map
        ref={mapRef}
        initialViewState={{
          longitude,
          latitude,
          zoom,
        }}
        style={{ width: '100%', height: '100%' }}
        mapStyle="https://tiles.openfreemap.org/styles/liberty"
        onLoad={onMapLoad}
        attributionControl={false}
        logoPosition="bottom-left"
        dragPan={true}
        dragRotate={false}
        scrollZoom={true}
        doubleClickZoom={true}
        keyboard={false}
        minZoom={1}
        maxZoom={18}
      >
        <Marker longitude={longitude} latitude={latitude} anchor="bottom">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-red-500 text-white shadow-lg">
            <i className="i-mingcute-location-fill text-sm" />
          </div>
        </Marker>
      </Map>
    </div>
  )
}
