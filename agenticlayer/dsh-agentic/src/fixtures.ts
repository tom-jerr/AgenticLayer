import { createHash } from 'node:crypto'
import { readFile, realpath, open } from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool, parameterSchemaSpecToJsonSchema } from '@deepseek-ai/dsh-tools'
import { delegationDepthOf } from '@deepseek-ai/dsh-subagent'
import { digest, type CacheIdentity } from './tool-cache.js'

interface FixtureConfig {
  root: string
  files: Record<string, string>
  marker: string
  dataVersion: string
  permissionScope: string
  isolationDomain: string
  stateFile?: string
}

const hashBytes = (data: Uint8Array) => createHash('sha256').update(data).digest('hex')
const implementation = hashBytes(readFileSync(new URL(import.meta.url)))
const readParameters = { path: { type: 'string', required: true }, delay_ms: { type: 'integer' } } as const
const sumParameters = { values: { type: 'array', required: true, items: { type: 'number' } } } as const
const readDescription = 'Read a named immutable fixture snapshot file.'
const sumDescription = 'Add independent finite numeric inputs.'

export function registerFixtures(ctx: Context): { identity: (name: string, args: unknown, permission: unknown) => Promise<CacheIdentity | undefined> } {
  const raw = process.env.AGENTIC_FIXTURES
  if (!raw) throw new Error('AGENTIC_FIXTURES must specify the immutable workspace snapshot')
  const config = JSON.parse(raw) as FixtureConfig
  if (!isAbsolute(config.root) || !config.files || !config.marker || !config.dataVersion || !config.permissionScope || !config.isolationDomain) {
    throw new Error('invalid fixture configuration')
  }
  const root = resolve(config.root)
  function lexical(path: string): string {
    if (isAbsolute(path) || path === '' || path.split(/[\\/]/).some(part => part === '..' || part === '.')) throw new Error('fixture path must be workspace relative')
    const full = resolve(root, path)
    const rel = relative(root, full)
    if (!rel || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new Error('fixture path escapes workspace')
    return full
  }
  async function current(): Promise<Pick<FixtureConfig, 'files' | 'dataVersion' | 'permissionScope'>> {
    if (!config.stateFile) return config
    if (!isAbsolute(config.stateFile) || await realpath(config.stateFile) !== config.stateFile) throw new Error('invalid fixture state file')
    const state = JSON.parse(await readFile(config.stateFile, 'utf8')) as Pick<FixtureConfig, 'files' | 'dataVersion' | 'permissionScope'>
    if (!state.files || !state.dataVersion || !state.permissionScope) throw new Error('invalid fixture state')
    return state
  }
  async function file(path: string, files: Record<string, string>): Promise<Buffer> {
    const expected = files[path]
    if (!expected || !/^[0-9a-f]{64}$/.test(expected)) throw new Error('path is not in the approved fixture snapshot')
    const full = lexical(path)
    if (await realpath(full) !== full || await realpath(root) !== root) throw new Error('fixture symlinks are forbidden')
    const bytes = await readFile(full)
    if (hashBytes(bytes) !== expected) throw new Error('fixture snapshot changed')
    return bytes
  }
  async function snapshot() {
    const state = await current()
    for (const path of Object.keys(state.files).sort()) await file(path, state.files)
    if (digest(state) !== digest(await current())) throw new Error('fixture state changed during snapshot')
    return state
  }
  ctx.tools.register(defineTool({
    name: 'fixture_read', description: readDescription, parameters: readParameters,
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      if (args.delay_ms !== undefined) {
        if (!Number.isSafeInteger(args.delay_ms) || args.delay_ms < 0 || args.delay_ms > 2000) throw new Error('invalid fixture delay')
        await delay(args.delay_ms, undefined, { signal: exec.signal })
      }
      exec.signal.throwIfAborted()
      const state = await current()
      const bytes = await file(args.path, state.files)
      if (digest(state) !== digest(await current())) throw new Error('fixture state changed during read')
      return bytes.toString('utf8')
    },
  }))
  ctx.tools.register(defineTool({
    name: 'fixture_sum', description: sumDescription, parameters: sumParameters,
    output: { schema: { type: 'number' }, render: (_args, value) => [{ type: 'text', text: String(value) }] },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      exec.signal.throwIfAborted()
      const result = args.values.reduce((sum, value) => sum + value, 0)
      if (!Number.isFinite(result)) throw new Error('non-finite sum')
      return result
    },
  }))
  ctx.tools.register(defineTool({
    name: 'fixture_write_marker', description: 'Create the configured marker with the decimal integer on one line; never overwrite it.',
    parameters: { value: { type: 'integer', required: true } },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    async execute(args, exec) {
      if (!exec.agent || delegationDepthOf(exec.agent) !== 0) throw new Error('only the root agent may write the marker')
      exec.signal.throwIfAborted()
      const target = lexical(config.marker)
      if (await realpath(dirname(target)) !== dirname(target) || await realpath(root) !== root) throw new Error('fixture symlinks are forbidden')
      const handle = await open(target, 'wx', 0o600)
      try { await handle.writeFile(`${args.value}\n`, { encoding: 'utf8', signal: exec.signal }) } finally { await handle.close() }
      return config.marker
    },
  }))
  return {
    async identity(name, args, permission) {
      if (name !== 'fixture_read' && name !== 'fixture_sum') return undefined
      const state = await snapshot()
      return { tool_id: name, tool_implementation_sha256: implementation,
        schema_sha256: digest({ name, description: name === 'fixture_read' ? readDescription : sumDescription,
          parameters: parameterSchemaSpecToJsonSchema(name === 'fixture_read' ? readParameters : sumParameters) }), normalized_args: args,
        workspace_snapshot_sha256: digest(state.files), data_version: state.dataVersion,
        permission_scope_sha256: digest([state.permissionScope, permission]), isolation_domain: config.isolationDomain }
    },
  }
}
