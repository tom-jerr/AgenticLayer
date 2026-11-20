import { randomUUID } from 'node:crypto'

export interface Permit { action_id: string; version_id: string; expires_mono_ns: number }
export interface Ack { event_id: string; version_id: string; status: string; permit: Permit | null; child_version_id?: string }

/** A branch owns its sequence and CAS version; unknown outcomes poison it. */
export class Branch {
  private seq = 0
  private version: string | null = null
  private tail: Promise<void> = Promise.resolve()
  private failed: unknown
  constructor(readonly client: Controller, readonly session: string, readonly branch: string) {}
  get currentVersion(): string | null { return this.version }
  seed(version: string): void {
    if (this.seq !== 0 || this.version !== null || !version) throw new Error('invalid child branch initialization')
    this.version = version
  }

  event(kind: string, payload: Record<string, unknown>, signal?: AbortSignal): Promise<Ack> {
    const { promise: result, resolve: settle, reject: fail } = Promise.withResolvers<Ack>()
    const work = this.tail.then(async () => {
      if (this.failed !== undefined) throw this.failed
      signal?.throwIfAborted()
      const eventId = randomUUID()
      const request = { run_id: this.client.runId, session_id: this.session, branch_id: this.branch,
        seq: ++this.seq, event_id: eventId, expected_version: this.version, kind, payload }
      let requestError: unknown
      const response = fetch(`${this.client.url}/agentic/v1/events`, {
        method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${this.client.token}` },
        body: JSON.stringify(request), signal,
      }).then(async reply => {
        if (!reply.ok) throw new Error(`agentic event ${kind} failed (${reply.status}): ${await reply.text()}`)
        const ack = await reply.json() as Ack
        if (ack.event_id !== eventId || typeof ack.version_id !== 'string') throw new Error('invalid agentic event acknowledgement')
        return ack
      }).catch(error => { requestError = error; throw error })
      // Grant waits do not hold the sequencing lane. The status endpoint proves
      // acceptance before the next seq is issued, even on independent sockets.
      if (kind === 'ready') {
        void response.then(settle, fail)
        const failureDeadline = Date.now() + 2000
        for (;;) {
          if (requestError !== undefined && Date.now() >= failureDeadline) throw requestError
          const status = await fetch(`${this.client.url}/agentic/v1/events/${encodeURIComponent(this.client.runId)}/${eventId}`, {
            headers: { authorization: `Bearer ${this.client.token}` },
          })
          if (status.ok) {
            const accepted = await status.json() as Ack
            if (accepted.event_id !== eventId || typeof accepted.version_id !== 'string') throw new Error('invalid accepted event')
            this.version = accepted.version_id
            return
          }
          if (status.status !== 404) throw new Error(`agentic event lookup failed (${status.status})`)
          await new Promise(resolve => setTimeout(resolve, 5))
        }
      }
      const ack = await response
      this.version = ack.version_id
      settle(ack)
    }).catch(error => { this.failed = error; fail(error) })
    this.tail = work
    return result
  }
  async permit(payload: Record<string, unknown>, signal?: AbortSignal): Promise<Permit> {
    const ack = await this.event('ready', payload, signal)
    const permit = ack.permit
    if (!permit || permit.action_id !== payload.action_id || permit.version_id !== ack.version_id
      || !Number.isFinite(permit.expires_mono_ns)
      || BigInt(Math.floor(permit.expires_mono_ns)) <= process.hrtime.bigint() + 1024n) {
      throw new Error('missing, mismatched, or expired agentic permit')
    }
    signal?.throwIfAborted()
    return permit
  }
}

export class Controller {
  private readonly branches = new Map<string, Branch>()
  constructor(readonly url: string, readonly token: string, readonly runId: string) {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)) {
      throw new Error('agentic controller must be loopback HTTP')
    }
    if (!token || !runId) throw new Error('agentic controller token and run id are required')
  }
  branch(session: string, branch = session): Branch {
    const key = `${session}\0${branch}`
    let value = this.branches.get(key)
    if (value === undefined) { value = new Branch(this, session, branch); this.branches.set(key, value) }
    return value
  }
}
