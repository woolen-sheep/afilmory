import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import nunjucks from 'nunjucks'

import type { Logger } from '../../../logger/index.js'
import type { OgVendorBuildContext } from './types.js'
import { OgVendor } from './types.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const templatesDir = path.join(__dirname, 'templates')
const nunjucksEnv = nunjucks.configure(templatesDir, { autoescape: false })

const DEFAULT_SITE_ORIGIN = 'https://example.com'

export interface CloudflareMiddlewareVendorConfig {
  type: 'cloudflare-middleware'
  storageURL: string
  siteConfigPath?: string
}

function escapeRegexLiteral(value: string): string {
  return value.replaceAll(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function normalizeUrlToOrigin(value: string | undefined | null): string | null {
  if (!value) return null

  const trimmed = value.trim()
  if (!trimmed) return null

  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`

  try {
    const parsed = new URL(withProtocol)
    return parsed.origin
  } catch {
    return null
  }
}

function resolveSiteConfigPath(siteConfigPath: string | undefined, repoRoot: string): string {
  if (!siteConfigPath) return path.resolve(repoRoot, 'config.json')
  return path.isAbsolute(siteConfigPath) ? siteConfigPath : path.resolve(repoRoot, siteConfigPath)
}

async function loadSiteUrl(
  siteConfigPath: string | undefined,
  repoRoot: string,
  logger: Logger,
): Promise<string | null> {
  const target = resolveSiteConfigPath(siteConfigPath, repoRoot)

  try {
    const raw = await readFile(target, 'utf8')
    const parsed = JSON.parse(raw) as Partial<{ url?: string }>
    const normalized = normalizeUrlToOrigin(parsed.url)

    if (!normalized) {
      logger.main.info(`OG image plugin: missing or invalid site.url in ${target}, using fallback origin.`)
      return null
    }

    return normalized
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    logger.main.info(`OG image plugin: using fallback origin (cannot read ${target}: ${message}).`)
    return null
  }
}

function renderCloudflareMiddlewareTemplate(patternHost: string, ogBase: string): string {
  return nunjucksEnv.render('cloudflare-middleware.njk', {
    patternHost,
    ogBase,
  })
}

export class CloudflareMiddlewareVendor extends OgVendor {
  readonly type = 'cloudflare-middleware' as const

  constructor(private readonly options: CloudflareMiddlewareVendorConfig) {
    super()
  }

  private normalizeStorageOrigin(): string {
    const normalized = normalizeUrlToOrigin(this.options.storageURL)
    if (!normalized) {
      throw new Error('CloudflareMiddleware vendor requires a valid storageURL (e.g., https://cdn.example.com)')
    }
    return normalized
  }

  private async resolveSiteOrigin(logger: Logger, repoRoot: string): Promise<string> {
    const loaded = await loadSiteUrl(this.options.siteConfigPath, repoRoot, logger)
    return loaded ?? DEFAULT_SITE_ORIGIN
  }

  private renderTemplate(siteOrigin: string, storageOrigin: string): string {
    const siteHost = normalizeUrlToOrigin(siteOrigin) ?? DEFAULT_SITE_ORIGIN

    const patternHost = (() => {
      try {
        return escapeRegexLiteral(new URL(siteHost).host)
      } catch {
        return escapeRegexLiteral(new URL(DEFAULT_SITE_ORIGIN).host)
      }
    })()

    const ogBase = `${storageOrigin}/.afilmory/og-images`

    return renderCloudflareMiddlewareTemplate(patternHost, ogBase)
  }

  async build(context: OgVendorBuildContext): Promise<void> {
    const siteOrigin = await this.resolveSiteOrigin(context.logger, context.repoRoot)
    const storageOrigin = this.normalizeStorageOrigin()

    const content = this.renderTemplate(siteOrigin, storageOrigin)

    const functionsDir = path.join(context.repoRoot, 'functions')
    const target = path.join(functionsDir, '_middleware.ts')

    await mkdir(functionsDir, { recursive: true })
    await writeFile(target, content, 'utf8')

    context.logger.main.info(`OG image vendor (CloudflareMiddleware): wrote ${target}`)
  }
}
