import { AsyncLocalStorage } from 'node:async_hooks'
import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { CONTEXT_WINDOW_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-compaction'
import type {} from '@deepseek-ai/dsh-llm-pi-ai'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { SessionId } from '@deepseek-ai/dsh-session'
import { delegationDepthOf } from '@deepseek-ai/dsh-subagent'
import type { SubagentActivation } from '@deepseek-ai/dsh-subagent'
import type { Session } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-tools'
import type { ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import { Controller, type Branch, type Permit } from './controller.js'
import { ToolResultCache, type CacheIdentity } from './tool-cache.js'
import { registerFixtures } from './fixtures.js'
import { digest } from './tool-cache.js'

export const name = 'agentic-layer'
export const inject = ['llm', 'tools', 'sessions', 'subagents', 'compaction']
interface Binding { session: string; branch: string; version: string; rid: string }
interface View { ids: string[]; hashes: string[]; envelope: string }
interface State {
  branch: Branch
  view?: View
  turn: string
  lastEndedVersion?: string
  lastEndedView?: View
  isChild?: boolean
  suspended?: boolean
  actions: Map<string, string>
  ordinals: Map<string, number>
  modelAction?: string
  nextOrdinal: number
  pending: Set<Promise<void>>
  failure?: unknown
}

const local = new AsyncLocalStorage<Binding>()
const json = (value: unknown): unknown => JSON.parse(JSON.stringify(value))
const id = () => randomUUID()
function permissionScope(session: Session): unknown {
  return session.snapshotEvents().filter(event =>
    ['permission/preset', 'sandbox/mode', 'approval/policy'].includes(String(event.type)))
    .map(event => [event.type, event.data])
}

function messageView(options: GenerateOptions): View {
  const messages = options.messages.map((message, index) => {
    const hash = digest(json(message))
    return { id: message.id === undefined ? `aux-${index}-${hash}` : String(message.id), hash }
  })
  return { ids: messages.map(message => message.id), hashes: messages.map(message => message.hash),
    envelope: digest(json({ provider: options.provider, model: options.model, tools: options.tools ?? [],
      toolHistory: options.toolHistory ?? null, system: options.system ?? null })) }
}

function surfaceChange(old: View | undefined, view: View): 'append' | 'replace' | 'remove' | undefined {
  if (old === undefined) return 'append'
  const same = (a: number, b: number) => old.ids[a] === view.ids[b] && old.hashes[a] === view.hashes[b]
  if (old.envelope !== view.envelope) return 'replace'
  if (old.ids.length <= view.ids.length && old.ids.every((_value, index) => same(index, index))) {
    return old.ids.length === view.ids.length ? undefined : 'append'
  }
  let cursor = 0
  for (let index = 0; index < old.ids.length && cursor < view.ids.length; index++) {
    if (same(index, cursor)) cursor++
  }
  return cursor === view.ids.length && view.ids.length < old.ids.length ? 'remove' : 'replace'
}

function requirePermit(permit: Permit, branch: Branch): void {
  if (branch.currentVersion !== permit.version_id || BigInt(Math.floor(permit.expires_mono_ns)) <= process.hrtime.bigint() + 1024n) {
    throw new Error('agentic permit expired or version changed before dispatch')
  }
}

interface Retention { view: View; dropped: string[]; last: number }

/** AKV/ActKV message-boundary drop: keep the task anchor and the newest turn,
 *  drop the oldest middle messages whole. The kept view stays template-legal
 *  because a turn's user→assistant→tool block is removed as one unit. */
function retention(messages: GenerateOptions['messages'], view: View, threshold: number): Retention | undefined {
  if (view.ids.length <= threshold) return undefined
  let last = view.ids.length - 1
  for (let index = view.ids.length - 1; index > 0; index--) {
    const role = messages[index].role
    if (role === 'user' || role === 'system') { last = index; break }
  }
  // A user rendered right after a tool block gets no turn header, so the kept
  // tail would re-render differently; only drop at a clean turn boundary.
  if (messages[last - 1]?.role === 'tool') return undefined
  const dropped = view.ids.slice(1, last)
  if (dropped.length === 0) return undefined
  return {
    view: { ids: [view.ids[0], ...view.ids.slice(last)], hashes: [view.hashes[0], ...view.hashes.slice(last)], envelope: view.envelope },
    dropped,
    last,
  }
}

export interface Config { mode?: 'controlled' | 'native'; isolationDomain?: string; dropAfterMessages?: number }

export function apply(ctx: Context, config: Config = {}): void {
  const controlled = config.mode !== 'native'
  const dropAfter = config.dropAfterMessages ?? 32
  const token = process.env.AGENTIC_LOCAL_TOKEN ?? ''
  const runId = process.env.AGENTIC_RUN_ID ?? ''
  const domain = config.isolationDomain ?? process.env.AGENTIC_ISOLATION_DOMAIN
  if (controlled && !domain) throw new Error('AGENTIC_ISOLATION_DOMAIN is required')
  const controller = controlled ? new Controller(process.env.AGENTIC_URL ?? 'http://127.0.0.1:30081', token, runId) : undefined
  const states = new Map<string, State>()
  const fixtures = registerFixtures(ctx)
  const cache = controlled && process.env.AGENTIC_TOOL_CACHE === 'on' ? new ToolResultCache() : undefined
  const pendingCache = new Map<symbol, { identity: CacheIdentity; name: string; args: unknown; result: unknown }>()
  function state(session: string, branch = session): State {
    const key = `${session}\0${branch}`
    let entry = states.get(key)
    if (!entry) {
      if (!controller) throw new Error('native mode has no controller')
      entry = { branch: controller.branch(session, branch), turn: '0', actions: new Map(), ordinals: new Map(), nextOrdinal: 0, pending: new Set() }
      states.set(key, entry)
    }
    return entry
  }
  function enqueue(entry: State, work: Promise<unknown>): void {
    const tracked = work.then(() => {}, error => { entry.failure = error })
    entry.pending.add(tracked)
    void tracked.then(() => { entry.pending.delete(tracked) })
  }
  async function barrier(entry: State): Promise<void> {
    while (entry.pending.size) await Promise.all(entry.pending)
    if (entry.failure !== undefined) throw entry.failure
  }

  ctx.on('agent/request', async ({ signal }, next) => {
    const proposal = await next()
    signal.throwIfAborted()
    return proposal.provider === 'local-sglang' ? { ...proposal, temperature: 0 } : proposal
  })

  const delegation = process.env.AGENTIC_DELEGATION === '1'
  const serialChildren = process.env.AGENTIC_SERIAL === '1'
  if (delegation) ctx.tools.register(defineTool({
    name: 'delegate_all',
    description: 'Run all independent read-only child tasks and wait for every result.',
    parameters: {
      children: { type: 'array', required: true, items: { type: 'object', additionalProperties: false,
        properties: { name: { type: 'string', required: true }, prompt: { type: 'string', required: true } } } },
    },
    output: {
      schema: { type: 'array', items: { type: 'json' } },
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const parent = exec.agent
      if (!parent) throw new Error('delegate_all requires an agent')
      if (delegationDepthOf(parent) >= 1) throw new Error('delegate_all maximum depth is one')
      if (args.children.length < 1 || args.children.length > 8) throw new Error('delegate_all needs 1..8 children')
      const names = new Set<string>()
      for (const child of args.children) {
        if (!/^[a-z][a-z0-9-]{0,31}$/.test(child.name) || child.name === 'lead' || names.has(child.name) || !child.prompt) {
          throw new Error('delegate_all child names must be unique valid names and prompts must be nonempty')
        }
        names.add(child.name)
      }
      const parentState = controlled ? state(String(parent.session.id)) : undefined
      if (parentState) {
        await barrier(parentState)
        if (!parentState.lastEndedVersion) throw new Error('delegation needs a completed parent turn')
      }
      const children = args.children.map(child => ({ ...child, childId: SessionId(`child-${randomUUID()}`), action: id(), registered: false, started: false }))
      try {
        if (parentState) {
          for (const child of children) {
            const childSession = String(child.childId)
            const ack = await parentState.branch.event('fork', {
              parent_branch_id: parentState.branch.branch, parent_version: parentState.lastEndedVersion,
              child_branch_id: childSession, parent_session_id: String(parent.session.id), child_session_id: childSession,
            }, exec.signal)
            if (!ack.child_version_id) throw new Error('fork acknowledgement lacks child version')
            const childState = state(childSession)
            childState.branch.seed(ack.child_version_id)
            childState.view = parentState.lastEndedView
            childState.isChild = true
            child.registered = true
          }
        }
        const results = await Promise.allSettled(children.map(async (child, ordinal) => {
          let permit: Permit | undefined
          if (parentState) {
            permit = await parentState.branch.permit({ action_id: child.action, rid: id(), turn_id: parentState.turn,
              call_id: String(exec.callId), ordinal, action_kind: 'child', exclusive_child: serialChildren }, exec.signal)
            requirePermit(permit, parentState.branch)
            await parentState.branch.event('start', { action_id: child.action }, exec.signal)
            child.started = true
          }
          let receipt: SubagentActivation | undefined
          let canceled: Promise<unknown> | undefined
          let disposing: Promise<void> | undefined
          let status = 'error'
          const cancel = () => {
            if (parentState && !canceled) canceled = parentState.branch.event('cancel', { action_id: child.action })
            if (receipt && !disposing) disposing = receipt.dispose()
            void canceled?.catch(() => {})
            void disposing?.catch(() => {})
          }
          exec.signal.addEventListener('abort', cancel, { once: true })
          try {
            if (parentState && permit) requirePermit(permit, parentState.branch)
            exec.signal.throwIfAborted()
            receipt = await ctx.subagents.startActivation({ provider: 'fork', childId: child.childId,
              name: child.name, label: child.name, parent, prompt: [{ type: 'text', text: child.prompt }],
              maxDepth: 1, delivery: 'caller', signal: exec.signal })
            if (exec.signal.aborted) cancel()
            const result = await receipt.result
            if (result.stopReason !== 'completed') throw new Error(result.diagnostic ?? `child ended: ${result.stopReason}`)
            status = 'completed'
            return { name: child.name, output: JSON.parse(JSON.stringify(result.output)) }
          } finally {
            exec.signal.removeEventListener('abort', cancel)
            if (exec.signal.aborted) cancel()
            if (receipt && !disposing) disposing = receipt.dispose()
            try {
              await disposing
              await canceled
            } finally {
              if (parentState) await parentState.branch.event('finish', { action_id: child.action,
                status: exec.signal.aborted ? 'canceled' : status })
            }
          }
        }))
        if (parentState) {
          const completed = children.filter(child => child.started).map(child => child.action)
          if (completed.length) await parentState.branch.event('join', { children: completed })
        }
        const failed = results.some(result => result.status === 'rejected')
        const outcomes = results.map((result, index) => result.status === 'fulfilled' ? result.value : {
          name: children[index]!.name, error: result.reason instanceof Error ? result.reason.message : String(result.reason),
        })
        if (failed) throw new Error(`delegate_all failed: ${JSON.stringify(outcomes)}`)
        return outcomes
      } finally {
        if (parentState) {
          await Promise.all(children.filter(child => child.registered).map(async child => {
            const childState = state(String(child.childId))
            await barrier(childState)
            await childState.branch.event(exec.signal.aborted ? 'cancel' : 'end', {})
          }))
        }
      }
    },
  }))

  ctx.on('agent/created', ({ agent }) => {
    const allow = delegationDepthOf(agent) === 0
      ? ['fixture_read', 'fixture_sum', 'fixture_write_marker', ...(delegation ? ['delegate_all'] : [])]
      : ['fixture_read', 'fixture_sum']
    agent.ctx.tools.restrict({ allow })
  })

  if (controlled) {
    const overflowRetried = new WeakSet<object>()
    const overflowAgents = new WeakMap<object, object>()
    ctx.on('agent/request-error', async ({ agent, failure, signal }, next) => {
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted || overflowRetried.has(agent)) return next()
      overflowAgents.set(agent.session, agent)
      const generation = agent.session.surface.replaceGeneration
      try {
        await ctx.compaction.compactIfNeeded(agent, 'context-overflow', signal)
      } catch {
        // A committed prune/replacement remains valid even if later summary work failed.
      }
      if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next()
      overflowRetried.add(agent)
      return { kind: 'retry' }
    })
    ctx.on('session/event', (session, event) => {
      const sessionId = String(session.id)
      const entry = state(sessionId)
      if (event.type === 'assistant/message') {
        const agent = overflowAgents.get(session)
        if (agent) overflowRetried.delete(agent)
      }
      if (event.type === 'turn/start') {
        entry.turn = String(event.data.turn)
        entry.ordinals.clear()
        entry.nextOrdinal = 0
        if (entry.suspended) {
          entry.suspended = false
          const preceding = [...entry.pending]
          enqueue(entry, Promise.all(preceding).then(() => entry.branch.event('resume', {})))
        }
      } else if (event.type === 'tool/call') {
        entry.ordinals.set(String(event.data.callId), entry.nextOrdinal++)
      } else if (event.type === 'assistant/message' && entry.modelAction) {
        enqueue(entry, entry.branch.event('commit', { action_id: entry.modelAction }))
        entry.modelAction = undefined
      } else if (event.type === 'tool/result') {
        const action = entry.actions.get(String(event.data.message.toolCallId))
        if (action) enqueue(entry, entry.branch.event('commit', { action_id: action }))
      } else if (event.type === 'turn/end' && entry.branch.currentVersion !== null) {
        const turn = String(event.data.turn)
        const previous = entry.view
        const messages = session.deriveMessages()
        const view: View | undefined = previous && {
          ids: messages.map(message => String(message.id)),
          hashes: messages.map(message => digest(json(message))),
          envelope: previous.envelope,
        }
        const preceding = [...entry.pending]
        if (!entry.isChild) entry.suspended = true
        enqueue(entry, (async () => {
          await Promise.all(preceding)
          if (view) {
            const kept = retention(messages, view, dropAfter)
            const surface = kept?.view ?? view
            const change = surfaceChange(previous, surface)
            if (change) {
              await entry.branch.event(kept ? 'repos' : change, {
                message_ids: surface.ids,
                ...(kept ? { dropped_message_ids: kept.dropped } : {}),
                surface_hash: digest([surface.hashes, surface.envelope]), isolation_domain: domain,
              })
              entry.view = surface
            }
          }
          const ack = await entry.branch.event('commit', { turn_id: turn, turn_end: true })
          entry.lastEndedVersion = ack.version_id
          entry.lastEndedView = view
          if (!entry.isChild) await entry.branch.event('suspend', {})
        })())
      }
    })
    const disposal = new Set<Promise<unknown>>()
    ctx.on('session/disposed', session => {
      for (const entry of states.values()) {
        if (entry.branch.session !== String(session.id) || entry.isChild) continue
        const work = (async () => {
          await barrier(entry)
          if (entry.branch.currentVersion !== null) await entry.branch.event('end', {})
        })()
        disposal.add(work)
        void work.then(() => disposal.delete(work), () => disposal.delete(work))
      }
    })
    ctx.effect(() => async () => { await Promise.allSettled(disposal) })


    ctx.on('llm-pi-ai/request-headers', (request, headers, next) => {
      if (request.provider !== 'local-sglang') return next()
      const binding = local.getStore()
      if (!binding || String(request.sessionId) !== binding.session) throw new Error('agentic request missing matching dispatch binding')
      headers.set('X-Agentic-Session', binding.session)
      headers.set('X-Agentic-Branch', binding.branch)
      headers.set('X-Agentic-Version', binding.version)
      headers.set('X-Agentic-Request', binding.rid)
      return next()
    })

    ctx.on('llm/stream', (options, next) => {
      if (options.provider !== 'local-sglang') return next()
      return (async function* (): AsyncGenerator<StreamChunk> {
        if (!options.sessionId) throw new Error('agentic model request requires session identity')
        const session = String(options.sessionId)
        const branchId = options.purpose ? `${session}/aux/${options.purpose}` : session
        const entry = state(session, branchId)
        await barrier(entry)
        const view = messageView(options)
        const kept = retention(options.messages, view, dropAfter)
        const surface = kept?.view ?? view
        const change = surfaceChange(entry.view, surface)
        if (change) {
          // repos rotates the retained prefix; a rewritten surface that cannot
          // align cold-rebuilds (the middleware validates the subsequence).
          await entry.branch.event(kept ? 'repos' : change, {
            message_ids: surface.ids,
            ...(kept ? { dropped_message_ids: kept.dropped } : {}),
            surface_hash: digest([surface.hashes, surface.envelope]), isolation_domain: domain,
          }, options.signal)
          entry.view = surface
        }
        const action = id(), rid = id()
        const effective = kept
          ? { ...options, messages: [options.messages[0], ...options.messages.slice(kept.last)] }
          : options
        const permit = await entry.branch.permit({ action_id: action, rid, turn_id: entry.turn,
          call_id: action, ordinal: 0, action_kind: options.purpose ? 'summary' : 'model' }, options.signal)
        requirePermit(permit, entry.branch)
        await entry.branch.event('start', { action_id: action }, options.signal)
        if (!options.purpose) entry.modelAction = action
        let status = 'error'
        try {
          requirePermit(permit, entry.branch)
          // The drop sends the kept view downstream; the emit-time options are
          // frozen, so the derived copy is forwarded as the next handler's own
          // options argument (the type omits it but dispatch forwards args).
          const forward = next as unknown as (options: GenerateOptions) => AsyncIterable<StreamChunk>
          const iterator = forward(effective)[Symbol.asyncIterator]()
          try {
            for (;;) {
              const step = await local.run({ session, branch: branchId, version: permit.version_id, rid }, () => iterator.next())
              if (step.done) break
              if (step.value.type === 'finish') status = ['stop', 'tool-calls'].includes(step.value.reason.kind)
                ? 'completed' : step.value.reason.kind === 'aborted' ? 'canceled' : 'error'
              yield step.value
            }
          } finally { if (iterator.return) await iterator.return() }
        } catch (error) {
          if (options.signal?.aborted) status = 'canceled'
          throw error
        } finally {
          if (status === 'canceled') await entry.branch.event('cancel', { action_id: action })
          await entry.branch.event('finish', { action_id: action, status })
        }
      })()
    })

    ctx.on('tools/result', (exec, result) => {
      const candidate = pendingCache.get(exec.token)
      pendingCache.delete(exec.token)
      if (!cache || !candidate || result.isError || exec.signal.aborted || !exec.agent
        || result.additionalContexts?.length || result.concludesTurn
        || digest(json(result)) !== digest(json(candidate.result))) return undefined
      const entry = state(String(exec.agent.session.id))
      enqueue(entry, (async () => {
        try {
          const permission = permissionScope(exec.agent!.session)
          const current = await fixtures.identity(candidate.name, candidate.args, permission)
          if (current && digest(permission) === digest(permissionScope(exec.agent!.session))
            && digest(current) === digest(candidate.identity) && !exec.signal.aborted) cache.put(current, candidate.result)
        } catch {
          // Changed or unavailable snapshots never publish a result.
        }
      })())
      return undefined
    })

    ctx.on('tools/execute', async (exec, next) => {
      if (!exec.agent) throw new Error('agentic tool dispatch requires an agent')
      const entry = state(String(exec.agent.session.id))
      await barrier(entry)
      const action = id(), call = String(exec.callId)
      entry.actions.set(call, action)
      const start = process.hrtime.bigint()
      const permit = await entry.branch.permit({ action_id: action, rid: id(), turn_id: entry.turn, call_id: call,
        ordinal: entry.ordinals.get(call) ?? 0, action_kind: 'tool',
        exclusive: !['delegate_all', 'fixture_read', 'fixture_sum'].includes(exec.name) }, exec.signal)
      requirePermit(permit, entry.branch)
      await entry.branch.event('start', { action_id: action }, exec.signal)
      let canceling: Promise<unknown> | undefined
      const cancel = () => {
        canceling ??= entry.branch.event('cancel', { action_id: action })
        void canceling.catch(() => {})
      }
      exec.signal.addEventListener('abort', cancel, { once: true })
      let status = 'error'
      let cacheHit = false
      try {
        requirePermit(permit, entry.branch)
        let identity: CacheIdentity | undefined
        if (cache) {
          try {
            const permission = permissionScope(exec.agent.session)
            identity = await fixtures.identity(exec.name, exec.arguments, permission)
            if (digest(permission) !== digest(permissionScope(exec.agent.session))) identity = undefined
          } catch { identity = undefined }
        }
        exec.signal.throwIfAborted()
        const cached = identity && cache?.get(identity) as ToolExecutionResult | undefined
        const result = cached ?? await next()
        if (!cached && identity && !result.isError && !exec.signal.aborted) {
          pendingCache.set(exec.token, { identity, name: exec.name, args: exec.arguments, result: json(result) })
        }
        cacheHit = cached !== undefined
        status = exec.signal.aborted ? 'canceled' : result.isError ? 'error' : 'completed'
        return result
      } finally {
        exec.signal.removeEventListener('abort', cancel)
        if (exec.signal.aborted) { status = 'canceled'; cancel() }
        await canceling
        await entry.branch.event('finish', { action_id: action, status,
          cache_hit: cacheHit, wait_ms: Number(process.hrtime.bigint() - start) / 1e6 })
      }
    })
  }
}
