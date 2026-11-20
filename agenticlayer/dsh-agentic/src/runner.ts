import { randomBytes, randomUUID } from 'node:crypto'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs'
import { DeepSeekHarness } from '@deepseek-ai/dsh-sdk-client'

export interface RunnerOptions {
  cwd: string
  dshHome: string
  dshBin: string
  isolationDomain: string
  mode: 'native' | 'controlled'
  serial?: boolean
  token?: string
  runId?: string
  files: Record<string, string>
  marker: string
  dataVersion: string
  permissionScope: string
  stateFile?: string
  toolCache?: boolean
  delegation?: boolean
  fixedSummary?: string
}

/** Owns the real SDK subprocess; callers must await harness.close(). */
export function createHarness(options: RunnerOptions): DeepSeekHarness {
  if (!options.isolationDomain) throw new Error('isolationDomain is required')
  if (options.mode === 'controlled' && !options.token) throw new Error('controlled mode requires the controller bearer token')
  const token = options.token ?? randomBytes(32).toString('hex')
  const root = fileURLToPath(new URL('..', import.meta.url))
  const cliRequire = createRequire(resolve(options.dshBin))
  const baseRequire = createRequire(cliRequire.resolve('@deepseek-ai/dsh-base'))
  const pluginRequire = createRequire(resolve(root, 'lib/index.js'))
  for (const name of ['@deepseek-ai/cordis', '@deepseek-ai/dsh-tools', '@deepseek-ai/dsh-session',
    '@deepseek-ai/dsh-subagent', '@deepseek-ai/dsh-llm', '@deepseek-ai/dsh-compaction-basic']) {
    if (realpathSync(baseRequire.resolve(name)) !== realpathSync(pluginRequire.resolve(name))) {
      throw new Error(`plugin and DSH must share the same runtime module: ${name}`)
    }
  }
  const home = resolve(options.dshHome)
  mkdirSync(home, { recursive: true, mode: 0o700 })
  const patch = resolve(home, `agentic-profile-${randomUUID()}.patch.yml`)
  // Cordis resolves entry names while composing patches, before config expressions.
  const template = readFileSync(resolve(root, 'profile.patch.yml'), 'utf8')
  const rendered = template.replace(/"__AGENTIC_(COMPACTION|PLUGIN)_MODULE__"/g, (_marker, kind: string) =>
    JSON.stringify(resolve(root, kind === 'COMPACTION' ? 'lib/compaction.js' : 'lib/index.js')))
  writeFileSync(patch, rendered, { flag: 'wx', mode: 0o600 })
  const env: NodeJS.ProcessEnv = {}
  for (const key of ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR', 'NODE_EXTRA_CA_CERTS']) {
    if (process.env[key] !== undefined) env[key] = process.env[key]
  }
  Object.assign(env, {
    AGENTIC_LOCAL_TOKEN: token,
    AGENTIC_RUN_ID: options.runId ?? randomUUID(),
    AGENTIC_ISOLATION_DOMAIN: options.isolationDomain,
    AGENTIC_MODE: options.mode,
    AGENTIC_SERIAL: options.serial ? '1' : '0',
    AGENTIC_URL: 'http://127.0.0.1:30081',
    AGENTIC_MODEL_BASE_URL: options.mode === 'native' ? 'http://127.0.0.1:30080/v1' : 'http://127.0.0.1:30081/v1',
    AGENTIC_FIXTURES: JSON.stringify({ root: resolve(options.cwd), files: options.files, marker: options.marker,
      dataVersion: options.dataVersion, permissionScope: options.permissionScope, isolationDomain: options.isolationDomain,
      ...(options.stateFile === undefined ? {} : { stateFile: resolve(options.stateFile) }) }),
    AGENTIC_TOOL_CACHE: options.toolCache ? 'on' : 'off',
    AGENTIC_DELEGATION: options.delegation ? '1' : '0',
    ...(options.fixedSummary === undefined ? {} : { AGENTIC_FIXED_SUMMARY: options.fixedSummary }),
  })
  return new DeepSeekHarness({ profile: 'sdk', patches: [patch],
    dshHome: home, cwd: resolve(options.cwd), dshBin: resolve(options.dshBin), env,
    provider: 'local-sglang', model: 'qwen35-9b-nvfp4', maxTokens: 512 })
}
