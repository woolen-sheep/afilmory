import type { PhotoRegion, PhotoRegionArea, PhotoRegionDimensions, PhotoXmpMetadata } from '@afilmory/typing'
import { XMLParser } from 'fast-xml-parser'

const EMPTY_XMP: PhotoXmpMetadata = {
  keywords: [],
  regions: [],
}

const xmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '',
  trimValues: true,
  parseTagValue: false,
})

const OPEN_TAGS = ['<x:xmpmeta', '<xmpmeta']
const CLOSE_TAGS = ['</x:xmpmeta>', '</xmpmeta>']

export function extractEmbeddedXmpPacket(buffer: Buffer): string | null {
  const binaryText = buffer.toString('latin1')

  let startIndex = -1
  for (const openTag of OPEN_TAGS) {
    startIndex = binaryText.indexOf(openTag)
    if (startIndex >= 0) {
      break
    }
  }

  if (startIndex < 0) {
    return null
  }

  let endIndex = -1
  let closingTagLength = 0
  for (const closeTag of CLOSE_TAGS) {
    const candidate = binaryText.indexOf(closeTag, startIndex)
    if (candidate !== -1) {
      endIndex = candidate
      closingTagLength = closeTag.length
      break
    }
  }

  if (endIndex < 0) {
    return null
  }

  return buffer
    .subarray(startIndex, endIndex + closingTagLength)
    .toString('utf8')
    .trim()
}

export function extractEmbeddedXmpMetadata(buffer: Buffer): PhotoXmpMetadata {
  const packet = extractEmbeddedXmpPacket(buffer)
  if (!packet) {
    return EMPTY_XMP
  }

  return parseXmpPacket(packet)
}

export function parseXmpPacket(packet: string): PhotoXmpMetadata {
  try {
    const parsed = xmlParser.parse(packet) as unknown
    const description = findFirstObjectByKey(parsed, 'rdf:Description')
    if (!description) {
      return EMPTY_XMP
    }

    return {
      keywords: mergeUniqueStrings(
        readBagValues(description['dc:subject']),
        readBagValues(description['lr:weightedFlatSubject']),
        readBagValues(description['lr:hierarchicalSubject']),
      ),
      regions: parseRegions(description['mwg-rs:Regions']),
    }
  } catch {
    return EMPTY_XMP
  }
}

function parseRegions(input: unknown): PhotoRegion[] {
  const regionRoot = asRecord(input)
  if (!regionRoot) {
    return []
  }

  const appliedToDimensions = parseDimensions(regionRoot['mwg-rs:AppliedToDimensions'])
  const regionList = asRecord(regionRoot['mwg-rs:RegionList'])
  const sequence = asRecord(regionList?.['rdf:Seq'])
  const entries = toArray(sequence?.['rdf:li'])

  return entries
    .map((entry) => parseRegion(entry, appliedToDimensions))
    .filter((region): region is PhotoRegion => region !== null)
}

function parseRegion(input: unknown, appliedToDimensions: PhotoRegionDimensions | null): PhotoRegion | null {
  const regionEntry = asRecord(input)
  const region = asRecord(regionEntry?.['rdf:Description']) ?? regionEntry
  if (!region) {
    return null
  }

  const name = normalizeText(region['mwg-rs:Name'])
  const type = normalizeText(region['mwg-rs:Type'])
  const area = parseArea(region['mwg-rs:Area'])

  if (!name && !type && !area) {
    return null
  }

  return {
    name: name ?? '',
    ...(type ? { type } : {}),
    area,
    appliedToDimensions,
  }
}

function parseDimensions(input: unknown): PhotoRegionDimensions | null {
  const value = asRecord(input)
  if (!value) {
    return null
  }

  const width = parseNumber(value['stDim:w'])
  const height = parseNumber(value['stDim:h'])
  const unit = normalizeText(value['stDim:unit'])
  if (width === null || height === null || !unit) {
    return null
  }

  return {
    width,
    height,
    unit,
  }
}

function parseArea(input: unknown): PhotoRegionArea | null {
  const value = asRecord(input)
  if (!value) {
    return null
  }

  const x = parseNumber(value['stArea:x'])
  const y = parseNumber(value['stArea:y'])
  const width = parseNumber(value['stArea:w'])
  const height = parseNumber(value['stArea:h'])
  const unit = normalizeText(value['stArea:unit'])
  if (x === null || y === null || width === null || height === null || !unit) {
    return null
  }

  return {
    x,
    y,
    width,
    height,
    unit,
  }
}

function readBagValues(input: unknown): string[] {
  const subject = asRecord(input)
  const bag = asRecord(subject?.['rdf:Bag'])
  const entries = toArray(bag?.['rdf:li'])

  return entries.map((entry) => normalizeText(entry)).filter((value): value is string => value !== null)
}

function findFirstObjectByKey(input: unknown, key: string): Record<string, unknown> | null {
  if (Array.isArray(input)) {
    for (const item of input) {
      const found = findFirstObjectByKey(item, key)
      if (found) {
        return found
      }
    }
    return null
  }

  const record = asRecord(input)
  if (!record) {
    return null
  }

  const direct = asRecord(record[key])
  if (direct) {
    return direct
  }

  for (const value of Object.values(record)) {
    const found = findFirstObjectByKey(value, key)
    if (found) {
      return found
    }
  }

  return null
}

function mergeUniqueStrings(...groups: string[][]): string[] {
  const seen = new Set<string>()
  const merged: string[] = []

  for (const group of groups) {
    for (const value of group) {
      const normalized = value.trim()
      if (!normalized || seen.has(normalized)) {
        continue
      }

      seen.add(normalized)
      merged.push(normalized)
    }
  }

  return merged
}

function normalizeText(input: unknown): string | null {
  if (typeof input !== 'string') {
    return null
  }

  const normalized = input.trim()
  return normalized.length > 0 ? normalized : null
}

function parseNumber(input: unknown): number | null {
  const value = typeof input === 'number' ? input : typeof input === 'string' ? Number.parseFloat(input) : Number.NaN
  return Number.isFinite(value) ? value : null
}

function asRecord(input: unknown): Record<string, unknown> | null {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    return null
  }

  return input as Record<string, unknown>
}

function toArray<T>(input: T | T[] | undefined): T[] {
  if (input === undefined) {
    return []
  }

  return Array.isArray(input) ? input : [input]
}
