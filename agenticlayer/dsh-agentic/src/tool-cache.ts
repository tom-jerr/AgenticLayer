import { createHash } from 'node:crypto'

export function canonical(value: unknown): string {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'number' && Number.isFinite(value) && !Object.is(value, -0)) return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (typeof value === 'object' && value !== null && Object.getPrototypeOf(value) === Object.prototype) {
    const object = value as Record<string, unknown>
    return `{${Object.keys(object).sort().map(key => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`
  }
  throw new TypeError('cache input must be lossless JSON')
}

export function digest(value: unknown): string {
  return createHash('sha256').update(canonical(value)).digest('hex')
}

export interface CacheIdentity {
  tool_id: string
  tool_implementation_sha256: string
  schema_sha256: string
  normalized_args: unknown
  workspace_snapshot_sha256: string
  data_version: string
  permission_scope_sha256: string
  isolation_domain: string
}

export function cacheKey(identity: CacheIdentity): string {
  return digest([identity.tool_id, identity.tool_implementation_sha256, identity.schema_sha256,
    identity.normalized_args, identity.workspace_snapshot_sha256, identity.data_version,
    identity.permission_scope_sha256, identity.isolation_domain])
}

interface Entry { readonly bytes: number; readonly json: string }

/** Stores detached successful values, never promises or execution identities. */
export class ToolResultCache {
  private readonly entries = new Map<string, Entry>()
  private bytes = 0
  constructor(private readonly budget = 64 * 1024 * 1024, private readonly maxEntry = 1024 * 1024) {}

  get(identity: CacheIdentity): unknown | undefined {
    const key = cacheKey(identity)
    const entry = this.entries.get(key)
    if (entry === undefined) return undefined
    this.entries.delete(key)
    this.entries.set(key, entry)
    return JSON.parse(entry.json)
  }

  put(identity: CacheIdentity, value: unknown): void {
    const key = cacheKey(identity)
    const json = canonical(value)
    const bytes = Buffer.byteLength(json, 'utf8')
    if (bytes > this.maxEntry || bytes > this.budget) return
    const old = this.entries.get(key)
    if (old !== undefined) { this.bytes -= old.bytes; this.entries.delete(key) }
    while (this.bytes + bytes > this.budget) {
      const first = this.entries.entries().next().value as [string, Entry] | undefined
      if (first === undefined) break
      this.entries.delete(first[0])
      this.bytes -= first[1].bytes
    }
    this.entries.set(key, { bytes, json })
    this.bytes += bytes
  }
}
