import type { PickedExif } from '@afilmory/builder'
import type { FC } from 'react'

import { injectConfig } from '~/config'
import { ExifPanel } from '~/modules/metadata/ExifPanel'
import type { PhotoManifest } from '~/types/photo'

import { InspectorPanel } from './InspectorPanel'

export interface PhotoInspectorProps {
  currentPhoto: PhotoManifest
  exifData: PickedExif | null
  activeRegionId?: string | null
  visible?: boolean
  onClose?: () => void
  onActiveRegionChange?: (regionId: string | null) => void
}

const CloudInspector: FC<PhotoInspectorProps> = props => <InspectorPanel {...props} />

const LegacyInspector: FC<PhotoInspectorProps> = ({ currentPhoto, exifData, ...rest }) => (
  <ExifPanel currentPhoto={currentPhoto} exifData={exifData} {...rest} />
)

export const PhotoInspector: FC<PhotoInspectorProps> = injectConfig.useCloud ? CloudInspector : LegacyInspector
