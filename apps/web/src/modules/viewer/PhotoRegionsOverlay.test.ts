import assert from 'node:assert/strict'

import type { PhotoRegion } from '@afilmory/builder'
import { it } from 'vitest'

import { getRenderablePhotoRegions } from './photo-region-bounds'

it('getRenderablePhotoRegions excludes metadata-only and zero-area regions from viewer controls', () => {
  const regions: PhotoRegion[] = [
    {
      name: '',
      type: 'Focus',
      area: null,
      appliedToDimensions: { width: 6000, height: 4000, unit: 'pixel' },
    },
    {
      name: 'Collapsed',
      type: 'Object',
      area: { x: 0.5, y: 0.5, width: 0, height: 0.2, unit: 'normalized' },
      appliedToDimensions: null,
    },
    {
      name: 'Northern Rough-winged Swallow',
      type: 'Object',
      area: { x: 0.47, y: 0.39, width: 0.25, height: 0.31, unit: 'normalized' },
      appliedToDimensions: null,
    },
  ]

  assert.deepEqual(
    getRenderablePhotoRegions(regions).map(region => region.name),
    ['Northern Rough-winged Swallow'],
  )
})

it('getRenderablePhotoRegions uses the photo dimensions when pixel regions omit their own dimensions', () => {
  const region: PhotoRegion = {
    name: 'Subject',
    type: 'Object',
    area: { x: 2000, y: 1500, width: 1000, height: 800, unit: 'pixel' },
    appliedToDimensions: null,
  }

  assert.deepEqual(getRenderablePhotoRegions([region]), [])
  assert.deepEqual(getRenderablePhotoRegions([region], 6000, 4000), [region])
})
