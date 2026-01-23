import type { Logger } from '../../../logger/index.js'

export type OgVendorKind = 'cloudflare-middleware'

export interface OgVendorBuildContext {
  repoRoot: string
  logger: Logger
}

export abstract class OgVendor {
  abstract readonly type: OgVendorKind
  abstract build(context: OgVendorBuildContext): Promise<void>
}
