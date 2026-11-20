#!/usr/bin/env node
// The driver never substitutes tool results or issues model tool calls itself.
// Native session logs remain under each condition's dsh-home/sessions.
import { createHash, randomUUID } from 'node:crypto'
import { appendFileSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const usage = `Usage: node experiments/dsh_bench.mjs --run-dir DIR --suite smoke|e1|e2|allof|allof-pair|drop [--mode native|controlled] [--token TOKEN] [--dsh-bin FILE] [--runner FILE] [--events FILE] [--repeats N]
Uses an already running SGLang (30080) or controller (30081); starts no server.
Each invocation creates a new directory under DIR/dsh-bench. --events optionally
names the controller's JSONL event trace; controlled mode needs --token or
AGENTIC_LOCAL_TOKEN. E1 and allof-pair --repeats N run N fixed serial/parallel
pairs in fresh workspaces (default 1; controlled mode required for pairs).
Exit 0 only if all requested, observable gates pass.`

function options(argv) {
  const result = {}
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--help') { console.log(usage); process.exit(0) }
    const key = argv[i]
    if (!['--run-dir', '--suite', '--mode', '--token', '--dsh-bin', '--runner', '--events', '--repeats'].includes(key)
        || result[key] !== undefined || !argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`Invalid argument: ${key}\n${usage}`)
    result[key] = argv[++i]
  }
  if (!result['--run-dir'] || !['smoke', 'e1', 'e2', 'allof', 'allof-pair', 'drop'].includes(result['--suite'])) throw new Error(usage)
  result['--mode'] ??= 'native'
  if (!['native', 'controlled'].includes(result['--mode'])) throw new Error('Invalid mode')
  if (result['--repeats'] !== undefined) {
    if (!['e1', 'allof-pair'].includes(result['--suite'])) throw new Error('--repeats is only available for E1 and allof-pair')
    if (!/^[1-9]\d*$/.test(result['--repeats']) || !Number.isSafeInteger(Number(result['--repeats']))) {
      throw new Error('--repeats requires a positive integer')
    }
  }
  result['--repeats'] = Number(result['--repeats'] ?? 1)
  if ((result['--repeats'] > 1 || result['--suite'] === 'allof-pair') && result['--mode'] !== 'controlled') throw new Error('Paired SDK benchmarks require controlled mode')
  result['--token'] ??= process.env.AGENTIC_LOCAL_TOKEN
  if (result['--mode'] === 'controlled' && !result['--token']) throw new Error('Controlled mode requires --token or AGENTIC_LOCAL_TOKEN')
  return result
}

const hash = value => createHash('sha256').update(value).digest('hex')
const monotonic = () => process.hrtime.bigint().toString()
const json = value => JSON.stringify(value, null, 2) + '\n'
const assertGate = (gates, name, pass, detail) => gates.push({ name, pass: Boolean(pass), detail })
const errorDetail = error => error instanceof Error ? (error.stack ?? error.message) : String(error)
const parseJSON = text => { try { return JSON.parse(text) } catch { return null } }
const parseArgs = call => parseJSON(call.arguments)
const canonical = value => Array.isArray(value) ? value.map(canonical)
  : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])])) : value
const exact = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b))
const callsOf = result => result.events.filter(e => e.type === 'tool/call').map(e => ({ ...e.data, args: parseArgs(e.data) }))
const resultsOf = result => result.events.filter(e => e.type === 'tool/result').map(e => e.data)
const resultText = message => (message?.content ?? []).filter(c => c.type === 'text').map(c => c.text).join('')

function checkPairing(result, gates, prefix) {
  const calls = callsOf(result), results = resultsOf(result)
  const ids = calls.map(c => c.callId)
  const resultIds = results.map(r => r.message?.toolCallId)
  assertGate(gates, `${prefix}.native_call_result_order`, ids.length > 0 && new Set(ids).size === ids.length && exact(ids, resultIds), { calls: ids, results: resultIds })
  assertGate(gates, `${prefix}.tools_succeeded`, results.length === calls.length && results.every(r => !r.message?.isError && !r.error), results.filter(r => r.message?.isError || r.error))
  return calls
}

async function fixture(base, name) {
  const dir = join(base, name), workspace = join(dir, 'workspace'), home = join(dir, 'dsh-home')
  await mkdir(workspace, { recursive: true, mode: 0o700 })
  await mkdir(home, { recursive: true, mode: 0o700 })
  const data = { 'a.txt': '17\n', 'b.txt': '25\n' }
  for (const [path, content] of Object.entries(data)) await writeFile(join(workspace, path), content, { flag: 'wx', mode: 0o600 })
  const state = { files: Object.fromEntries(Object.entries(data).map(([path, content]) => [path, hash(content)])), dataVersion: 'v1', permissionScope: 'read-v1' }
  const stateFile = join(dir, 'fixture-state.json')
  await writeFile(stateFile, json(state), { flag: 'wx', mode: 0o600 })
  return { dir, workspace, home, state, stateFile, marker: join(workspace, 'commit.marker') }
}

async function runTurn(session, prompt, fixture, label) {
  const path = join(fixture.dir, `${label}.notifications.jsonl`)
  await writeFile(path, '', { flag: 'wx', mode: 0o600 })
  const started = monotonic()
  let timer
  const operation = session.run(prompt, { onNotification: notification => {
    appendFileSync(path, JSON.stringify({ mono_ns: monotonic(), notification }) + '\n')
  } })
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Turn ${label} exceeded 900 seconds; closing harness`)), 900_000) })
  let result
  try { result = await Promise.race([operation, timeout]) } finally { clearTimeout(timer) }
  const finished = monotonic()
  await writeFile(join(fixture.dir, `${label}.result.json`), json({ started_mono_ns: started, finished_mono_ns: finished, ...result }), { flag: 'wx', mode: 0o600 })
  return { ...result, active_turn_ms: Number(BigInt(finished) - BigInt(started)) / 1e6 }
}

async function markerGate(f, gates) {
  let content = null
  try { content = await readFile(f.marker, 'utf8') } catch (e) { if (e.code !== 'ENOENT') throw e }
  assertGate(gates, 'marker.exact', content === '42\n', { expected: '42\n', actual: content })
}

function taskWorkload(secondDelay, firstDelay = 1000) {
  const expected = [
    ['fixture_read', { path: 'a.txt', delay_ms: firstDelay }],
    ['fixture_read', { path: 'b.txt', delay_ms: secondDelay }],
    ['fixture_sum', { values: [17, 25] }],
    ['fixture_write_marker', { value: 42 }],
  ]
  const prompt = 'Begin with precisely two fixture_read calls in the same assistant step. ' +
    `The first call's exact arguments are ${JSON.stringify(expected[0][1])}; ` +
    `the second call's exact arguments are ${JSON.stringify(expected[1][1])}. ` +
    'After both results, pass the two integers in the same order as values to fixture_sum. ' +
    'After the sum returns, pass that integer as value to fixture_write_marker once, then answer only with the integer.'
  return { prompt, expected }
}

async function taskCase(createHarness, config, base, name, serial, secondDelay = 0, warmHarness = false) {
  const f = await fixture(base, name), gates = []
  const harness = createHarness({ cwd: f.workspace, dshHome: f.home, dshBin: config.bin, isolationDomain: name,
    mode: config.mode, serial, token: config.token, runId: `${config.id}-${name}`, files: f.state.files,
    marker: 'commit.marker', dataVersion: f.state.dataVersion, permissionScope: f.state.permissionScope, stateFile: f.stateFile, toolCache: false })
  try {
    if (warmHarness) await harness.start()
    const { prompt, expected } = taskWorkload(secondDelay)
    const result = await runTurn(harness.session(), prompt, f, 'task')
    const calls = checkPairing(result, gates, 'task')
    const actual = calls.map(c => [c.name, c.args])
    assertGate(gates, 'task.expected_sequence', exact(actual, expected), { expected, actual })
    assertGate(gates, 'task.parallel_batch', calls.length >= 2 && calls[0].turn === calls[1].turn && calls[0].step === calls[1].step, calls.slice(0, 2))
    assertGate(gates, 'task.dependencies', calls.length === 4 && calls[2].turn === calls[0].turn && calls[2].step > calls[1].step && calls[3].turn === calls[2].turn && calls[3].step > calls[2].step, calls.map(c => ({ name: c.name, turn: c.turn, step: c.step })))
    const results = resultsOf(result)
    assertGate(gates, 'task.read_values', resultText(results[0]?.message).trim() === '17' && resultText(results[1]?.message).trim() === '25', results.slice(0, 2))
    assertGate(gates, 'task.sum_value', resultText(results[2]?.message).trim() === '42', results[2])
    assertGate(gates, 'task.answer', result.finishReason === 'completed' && result.finalResponse.trim() === '42', { finishReason: result.finishReason, answer: result.finalResponse })
    await markerGate(f, gates)
    return { gates, sequence: actual, calls, sessionId: result.sessionId, active_turn_ms: result.active_turn_ms }
  } finally { await harness.close() }
}

const cacheReadDelayMs = 300

async function cacheCase(createHarness, config, base, name, cache, serial) {
  const f = await fixture(base, name), gates = [], turns = []
  const harness = createHarness({ cwd: f.workspace, dshHome: f.home, dshBin: config.bin, isolationDomain: name,
    mode: config.mode, serial, token: config.token, runId: `${config.id}-${name}`, files: f.state.files,
    marker: 'commit.marker', dataVersion: f.state.dataVersion, permissionScope: f.state.permissionScope, stateFile: f.stateFile, toolCache: cache })
  let label
  let commit = []
  try {
    const args = { path: 'a.txt', delay_ms: cacheReadDelayMs }
    // Give the model no prior answer to memorize. The SDK subprocess, workspace,
    // authorization and plugin's result cache are shared across these fresh sessions.
    const prompt = `Call fixture_read exactly once for the bare filename ${args.path} with delay_ms ${args.delay_ms}. ` +
      'The path value must contain only the filename, without quote characters. ' +
      'Do not call any other tool. After the tool returns, answer with only the decimal integer from this call.'
    for (let i = 0; i < 4; i++) {
      label = `read-${i + 1}`
      if (i === 2) {
        await writeFile(join(f.workspace, 'a.txt'), '18\n')
        f.state.files['a.txt'] = hash('18\n'); f.state.dataVersion = 'v2'
        await writeFile(f.stateFile, json(f.state))
      }
      if (i === 3) { f.state.permissionScope = 'read-v2'; await writeFile(f.stateFile, json(f.state)) }
      const result = await runTurn(harness.session(), prompt, f, label)
      const calls = checkPairing(result, gates, label)
      const expected = i < 2 ? '17' : '18'
      const observed = resultText(resultsOf(result)[0]?.message)
      assertGate(gates, `${label}.sequence`, exact(calls.map(c => [c.name, c.args]), [['fixture_read', args]]), calls)
      assertGate(gates, `${label}.value`, result.finishReason === 'completed' && result.finalResponse === expected && observed === `${expected}\n`, { expected, toolResult: observed, answer: result.finalResponse, finishReason: result.finishReason })
      turns.push({ label, sessionId: result.sessionId, calls, dataVersion: f.state.dataVersion, permissionScope: f.state.permissionScope, fileHash: f.state.files['a.txt'] })
    }
    label = 'commit'
    await writeFile(join(f.workspace, 'a.txt'), '17\n')
    f.state.files['a.txt'] = hash('17\n'); f.state.dataVersion = 'v3'
    await writeFile(f.stateFile, json(f.state))
    const workload = taskWorkload(0, 0)
    const result = await runTurn(harness.session(), `${workload.prompt} Include both fields in each read call, including delay_ms 0.`, f, label)
    const calls = checkPairing(result, gates, 'commit')
    commit = calls
    assertGate(gates, 'commit.sequence', exact(calls.map(c => [c.name, c.args]), workload.expected), calls)
    assertGate(gates, 'commit.answer', result.finishReason === 'completed' && result.finalResponse.trim() === '42' && resultText(resultsOf(result)[0]?.message).trim() === '17' && resultText(resultsOf(result)[1]?.message).trim() === '25' && resultText(resultsOf(result)[2]?.message).trim() === '42', { answer: result.finalResponse, finishReason: result.finishReason })
    assertGate(gates, 'commit.dependencies', calls.length === 4 && calls[2].step > Math.max(calls[0].step, calls[1].step) && calls[3].step > calls[2].step, calls.map(c => ({ name: c.name, turn: c.turn, step: c.step })))
    assertGate(gates, 'commit.parallel_batch', calls.length >= 2 && calls[0].turn === calls[1].turn && calls[0].step === calls[1].step, calls.slice(0, 2))
    await markerGate(f, gates)
  } catch (error) {
    assertGate(gates, `${label ?? 'cache'}.execution`, false, errorDetail(error))
  } finally { await harness.close() }
  return { gates, turns, commit }
}

async function allofCase(createHarness, config, base, name, serial = false, paired = false) {
  const f = await fixture(base, name), gates = []
  const harness = createHarness({ cwd: f.workspace, dshHome: f.home, dshBin: config.bin, isolationDomain: name,
    mode: config.mode, serial, token: config.token, runId: `${config.id}-${name}`, files: f.state.files,
    marker: 'commit.marker', dataVersion: f.state.dataVersion, permissionScope: f.state.permissionScope, stateFile: f.stateFile, toolCache: false, delegation: true })
  try {
    const session = harness.session()
    const priming = await runTurn(session, 'Answer exactly READY. Do not call tools.', f, 'prime')
    assertGate(gates, 'allof.prime', priming.finishReason === 'completed' && priming.finalResponse.trim() === 'READY' && callsOf(priming).length === 0, { answer: priming.finalResponse })
    const fastDelay = paired ? 650 : 0, slowDelay = paired ? 1000 : 2000
    const prompt = `Call delegate_all exactly once with two children in this order: child named fast should read b.txt with fixture_read delay_ms ${fastDelay} and return the integer; child named slow should read a.txt with fixture_read delay_ms ${slowDelay} and return the integer. Wait for both children. Then call fixture_sum with [17,25], call fixture_write_marker once with value 42, and answer exactly 42.`
    const result = await runTurn(session, prompt, f, 'delegate')
    const calls = checkPairing(result, gates, 'delegate')
    const names = calls.map(c => c.name)
    assertGate(gates, 'allof.sequence', exact(names, ['delegate_all', 'fixture_sum', 'fixture_write_marker']) && calls[0]?.args?.children?.length === 2 && exact(calls[0].args.children.map(c => c.name), ['fast', 'slow']) && exact(calls[1]?.args, { values: [17, 25] }) && exact(calls[2]?.args, { value: 42 }), calls)
    const notifications = result.notifications
    const start = notifications.filter(n => n.method === 'subagent.started' && n.params.parentSessionId === result.sessionId)
    const children = start.map(n => n.params.childSessionId)
    const ends = children.map(id => notifications.findIndex(n => n.method === 'subagent.finished' && n.params.childSessionId === id))
    const delegateId = calls[0]?.callId
    const parentResult = notifications.findIndex(n => n.method === 'session.event' && n.params.sessionId === result.sessionId && n.params.event?.type === 'tool/result' && n.params.event.data?.message?.toolCallId === delegateId)
    assertGate(gates, 'allof.two_children_finished_before_result', children.length === 2 && new Set(children).size === 2 && ends.every(i => i >= 0 && i < parentResult), { children, ends, parentResult })
    const childReads = children.map(id => {
      const read = notifications.find(n => n.method === 'session.event' && n.params.sessionId === id && n.params.event?.type === 'tool/call' && n.params.event.data?.name === 'fixture_read')
      const finished = notifications.find(n => n.method === 'subagent.finished' && n.params.childSessionId === id)
      return { id, args: read ? parseArgs(read.params.event.data) : null, status: finished?.params.status }
    })
    const fast = childReads.find(c => exact(c.args, { path: 'b.txt', delay_ms: fastDelay }))
    const slow = childReads.find(c => exact(c.args, { path: 'a.txt', delay_ms: slowDelay }))
    const startIndex = child => notifications.findIndex(n => n.method === 'subagent.started' && n.params.childSessionId === child?.id)
    const finishIndex = child => notifications.findIndex(n => n.method === 'subagent.finished' && n.params.childSessionId === child?.id)
    const overlap = Math.max(startIndex(slow), startIndex(fast)) < Math.min(finishIndex(slow), finishIndex(fast))
    assertGate(gates, 'allof.child_work_and_order', Boolean(fast && slow && fast.id !== slow.id && fast.status === 'ok' && slow.status === 'ok'
      && (paired ? overlap !== serial : finishIndex(fast) < finishIndex(slow))), { childReads, overlap })
    assertGate(gates, 'allof.answer', result.finishReason === 'completed' && result.finalResponse.trim() === '42' && resultText(resultsOf(result)[1]?.message).trim() === '42', { answer: result.finalResponse, finishReason: result.finishReason })
    await markerGate(f, gates)
    // Induce a genuine snapshot failure without telling the model which call
    // fails. The approved path and recorded digest stay unchanged.
    if (paired) await writeFile(join(f.workspace, 'a.txt'), 'changed after snapshot\n')
    const errorPrompt = paired
      ? 'Call delegate_all exactly once with two children, in order: child named good must call fixture_read with {"path":"b.txt","delay_ms":2000} and return the tool result; child named bad must call fixture_read with {"path":"a.txt","delay_ms":0} and return the tool result. Both children must issue the tool call even if they already know a value. Wait for both and report their results. Do not write a marker or call another tool.'
      : 'Call delegate_all exactly once with two children: child named good should read b.txt with fixture_read delay_ms 2000 and return the value; child named bad should read missing.txt with fixture_read delay_ms 0 and return the tool error without recovery. Wait for both, and report any error. Do not write a marker or call any other tool.'
    const failed = await runTurn(session, errorPrompt, f, 'delegate-error')
    const failedCalls = callsOf(failed)
    const failedNotifications = failed.notifications
    const failedChildren = failedNotifications.filter(n => n.method === 'subagent.started' && n.params.parentSessionId === failed.sessionId).map(n => n.params.childSessionId)
    const failedEnds = failedChildren.map(id => failedNotifications.findIndex(n => n.method === 'subagent.finished' && n.params.childSessionId === id))
    const failureId = failedCalls.find(c => c.name === 'delegate_all')?.callId
    const errorIndex = failedNotifications.findIndex(n => n.method === 'session.event' && n.params.sessionId === failed.sessionId && n.params.event?.type === 'tool/result' && n.params.event.data?.message?.toolCallId === failureId)
    const outcome = resultsOf(failed).find(r => r.message?.toolCallId === failureId)
    const drained = failedChildren.length === 2 && failedEnds.every(i => i >= 0 && i < errorIndex)
    if (paired) {
      // A read failure handled by the child is not a failed activation. The
      // parent must receive the report and the sibling after both children end.
      const readErrors = failedNotifications.filter(n => n.method === 'session.event' && failedChildren.includes(n.params.sessionId)
        && n.params.event?.type === 'tool/result' && n.params.event.data?.message?.isError)
      const returned = parseJSON(resultText(outcome?.message))
      const ordered = Array.isArray(returned) && exact(returned.map(child => child?.name), ['good', 'bad'])
        && returned.every(child => Array.isArray(child.output))
      const goodReport = ordered ? resultText({ content: returned[0].output }) : ''
      const badReport = ordered ? resultText({ content: returned[1].output }) : ''
      const readCalls = failedNotifications.filter(n => n.method === 'session.event' && failedChildren.includes(n.params.sessionId)
        && n.params.event?.type === 'tool/call' && n.params.event.data?.name === 'fixture_read')
      const badRead = readCalls.find(n => exact(parseArgs(n.params.event.data), { path: 'a.txt', delay_ms: 0 }))
      const goodRead = readCalls.find(n => exact(parseArgs(n.params.event.data), { path: 'b.txt', delay_ms: 2000 }))
      assertGate(gates, 'allof.handled_read_error', drained && failedCalls.length === 1 && readCalls.length === 2
        && Boolean(badRead && goodRead && badRead.params.sessionId !== goodRead.params.sessionId)
        && readErrors.length === 1 && badRead?.params.event.data.callId === readErrors[0].params.event.data.message.toolCallId
        && /fixture snapshot changed/.test(resultText(readErrors[0].params.event.data.message))
        && ordered && /\b25\b/.test(goodReport)
        && /\bsnapshot\b/i.test(badReport) && /\b(changed|modified)\b/i.test(badReport),
      { failedChildren, failedEnds, errorIndex, readCalls, readErrors, outcome })
    } else {
      assertGate(gates, 'allof.failure_drain', drained && Boolean(outcome?.message?.isError)
        && /delegate_all failed/.test(JSON.stringify(outcome)), { failedCalls, failedChildren, failedEnds, errorIndex, outcome, note: 'A child handling its read error and completing normally is not a failed activation.' })
    }
    await markerGate(f, gates)
    return { gates, calls, sessionId: result.sessionId, active_turn_ms: result.active_turn_ms }
  } finally { await harness.close() }
}

// Only an actual controller trace can establish physical execution, timing, or cache hits.
// Accept the controller's INFO log as well as raw JSONL event envelopes.
async function traceFor(path, runId, names) {
  const rows = new Map(names.map(name => [`${runId}-${name}`, []]))
  if (path) {
    for (const line of (await readFile(path, 'utf8')).split('\n')) {
      const match = line.match(/(?:^|:)agentic_layer:event (\{.*\})$/)
      const text = match?.[1] ?? (line.trim().startsWith('{') ? line.trim() : null)
      if (!text) continue
      const row = JSON.parse(text)
      const envelope = row.event ?? row
      rows.get(envelope.run_id)?.push({ envelope, mono_ns: Number(row.mono_ns ?? envelope.mono_ns) })
    }
  }
  return new Map(names.map(name => [name, traceActions(rows.get(`${runId}-${name}`))]))
}
function traceActions(rows) {
  const actions = new Map()
  for (const row of rows) {
    const e = row.envelope, p = e.payload ?? {}, id = p.action_id
    if (!id) continue
    const a = actions.get(id) ?? { id }
    if (e.kind === 'ready') Object.assign(a, { sessionId: e.session_id, turnId: p.turn_id, callId: p.call_id, rid: p.rid, kind: p.action_kind })
    if (e.kind === 'start') a.start = row.mono_ns
    if (e.kind === 'finish') {
      Object.assign(a, { finish: row.mono_ns, status: p.status, cacheHit: p.cache_hit, wait_ms: p.wait_ms })
      // The real SDK hook skips next() only when it returns the cached result.
      if (a.kind === 'tool' && a.status === 'completed' && typeof a.cacheHit === 'boolean') a.invocations = Number(!a.cacheHit)
      if (Number.isFinite(a.start) && Number.isFinite(a.finish) && a.start > 0 && a.finish > a.start) {
        a.elapsed_ms = (a.finish - a.start) / 1e6
      }
    }
    actions.set(id, a)
  }
  return [...actions.values()]
}

function toolTrace(actions, call, sessionId) {
  if (!call?.callId || !sessionId) return null
  const matching = actions.filter(a => a.kind === 'tool' && a.sessionId === sessionId
    && a.callId === call.callId && a.turnId === String(call.turn))
  return matching.length === 1 ? matching[0] : null
}

async function collectCase(report, name, run) {
  try {
    report.cases[name] = await run()
  } catch (error) {
    const gates = []
    assertGate(gates, `${name}.execution`, false, errorDetail(error))
    report.cases[name] = { gates }
  }
}

async function dropCase(createHarness, config, base, name) {
  const f = await fixture(base, name), gates = []
  const harness = createHarness({ cwd: f.workspace, dshHome: f.home, dshBin: config.bin, isolationDomain: name,
    mode: config.mode, serial: false, token: config.token, runId: `${config.id}-${name}`, files: f.state.files,
    marker: 'commit.marker', dataVersion: f.state.dataVersion, permissionScope: f.state.permissionScope,
    stateFile: f.stateFile, toolCache: false, dropAfterMessages: 3 })
  try {
    const session = harness.session()
    // Phase 1 is tool-free so the phase-2 drop boundary is always template-clean:
    // a tool-call turn may merge its final answer into the tool message, which
    // leaves the retained tail re-rendered differently and (correctly) skips the drop.
    const phase1 = await runTurn(session,
      'Answer only with the integer 17. Do not call any tool.',
      f, 'phase1')
    assertGate(gates, 'drop.phase1_answer', phase1.finishReason === 'completed' && phase1.finalResponse.trim() === '17',
      { finishReason: phase1.finishReason, answer: phase1.finalResponse })
    const phase2 = await runTurn(session,
      'Call fixture_read exactly once for the bare filename b.txt with delay_ms 0. ' +
      'Then pass the two integers 17 and 25 in that order as values to fixture_sum. ' +
      'After the sum returns, pass that integer as value to fixture_write_marker once, then answer only with the integer.',
      f, 'phase2')
    const calls = callsOf(phase2)
    assertGate(gates, 'drop.phase2_sequence', exact(calls.map(c => [c.name, c.args]), [
      ['fixture_read', { path: 'b.txt', delay_ms: 0 }],
      ['fixture_sum', { values: [17, 25] }],
      ['fixture_write_marker', { value: 42 }],
    ]), calls.map(c => [c.name, c.args]))
    assertGate(gates, 'drop.phase2_answer', phase2.finishReason === 'completed' && phase2.finalResponse.trim() === '42',
      { finishReason: phase2.finishReason, answer: phase2.finalResponse })
    await markerGate(f, gates)
    return { gates, sessionId: phase2.sessionId }
  } finally { await harness.close() }
}

async function reposTrace(path, runId) {
  if (!path) return []
  const rows = []
  for (const line of (await readFile(path, 'utf8')).split('\n')) {
    const match = line.match(/(?:^|:)agentic_layer:repos materialised (\{.*\})$/)
    if (!match) continue
    try {
      const row = JSON.parse(match[1])
      if (row.run_id === runId) rows.push(row)
    } catch { /* not a repos row */ }
  }
  return rows
}

async function runE1(createHarness, config, report, repeats) {
  const pairs = []
  report.repeats = repeats
  for (let i = 0; i < repeats; i++) {
    const sample = `repeat-${i + 1}`
    const names = repeats === 1 ? { serial: 'serial', parallel: 'parallel' }
      : { serial: `${sample}-serial`, parallel: `${sample}-parallel` }
    pairs.push({ sample, names })
    // Keep the same workload, but alternate the order to balance warmup.
    for (const mode of (i % 2 ? ['parallel', 'serial'] : ['serial', 'parallel'])) {
      const name = names[mode]
      await collectCase(report, name, () => taskCase(createHarness, config, report.output, name, mode === 'serial', 650, true))
    }
  }
  const traces = await traceFor(report.events, config.id, Object.keys(report.cases))
  report.read_batch_ms = {}
  for (const { sample, names } of pairs) {
    const prefix = repeats === 1 ? 'e1' : `e1.${sample}`
    const serial = report.cases[names.serial].sequence, parallel = report.cases[names.parallel].sequence
    assertGate(report.gates, `${prefix}.same_workload`, serial && parallel && exact(serial, parallel), { serial, parallel })
    const span = repeats === 1 ? report.read_batch_ms : (report.read_batch_ms[sample] = {})
    for (const [mode, name] of Object.entries(names)) {
      const c = report.cases[name], actions = traces.get(name)
      const [first, second] = c.calls ?? []
      const a = toolTrace(actions, first, c.sessionId), b = toolTrace(actions, second, c.sessionId)
      const timed = [a, b].every(x => x?.status === 'completed' && x.elapsed_ms > 0)
      const valid = timed && (mode === 'serial' ? b.start >= a.finish : b.start < a.finish && b.finish < a.finish)
      assertGate(report.gates, `${prefix}.${mode}.physical_reads`, valid, { first: a, second: b })
      if (valid) span[mode] = (Math.max(a.finish, b.finish) - Math.min(a.start, b.start)) / 1e6
    }
  }
}

async function runE2(createHarness, config, report) {
  if (config.mode !== 'controlled') throw new Error('E2 cache on/off requires controlled mode; native mode never enables result cache')
  const conditions = [['serial-off', true, false], ['parallel-off', false, false], ['serial-on', true, true], ['parallel-on', false, true]]
  for (const [name, serial, cache] of conditions) {
    await collectCase(report, name, () => cacheCase(createHarness, config, report.output, name, cache, serial))
  }
  const signatures = c => [...(c.turns ?? []).flatMap(t => t.calls), ...(c.commit ?? [])].map(c => [c.name, c.args])
  const workloads = Object.fromEntries(conditions.map(([name]) => [name, signatures(report.cases[name])]))
  const complete = Object.values(report.cases).every(c => c.turns?.length === 4 && c.commit?.length)
  assertGate(report.gates, 'e2.same_workload', complete && Object.values(workloads).every(work => exact(work, workloads['serial-off'])), workloads)
  const traces = await traceFor(report.events, config.id, Object.keys(workloads))
  const minimumMissMs = cacheReadDelayMs - 5 // Allow timer rounding without counting a short hit as a physical miss.
  const cases = {}
  report.cache_reads = { scope: 'Four fresh single-read SDK sessions sharing one plugin process per condition; excludes the commit and is not a full E2 pass.',
    requested_delay_ms: cacheReadDelayMs, minimum_miss_ms: minimumMissMs, cases }
  for (const [name, , cache] of conditions) {
    const expected = cache ? [false, true, false, false] : [false, false, false, false]
    const c = report.cases[name], actions = traces.get(name)
    const observed = expected.map((_, i) => {
      const turn = c.turns?.[i]
      return toolTrace(actions, turn?.calls[0], turn?.sessionId)
    })
    const physical = observed.every((a, i) => a?.status === 'completed' && a.cacheHit === expected[i])
    assertGate(report.gates, `e2.${name}.physical_cache`, physical, { observed, expected, note: 'Ingress finish.cache_hit distinguishes the SDK hook returning cached data from calling the real tool; SDK tool/result alone does not.' })
    const timed = observed.every((a, i) => a?.elapsed_ms > 0 && Number.isFinite(a.wait_ms) && a.wait_ms > 0
      && (expected[i] || (a.elapsed_ms >= minimumMissMs && a.wait_ms >= minimumMissMs)))
    assertGate(report.gates, `e2.${name}.physical_read_timing`, timed, { observed, minimum_miss_ms: minimumMissMs })
    const semantics = c.turns?.length === 4 && c.gates.filter(g => g.name.startsWith('read-')).every(g => g.pass)
    cases[name] = {
      pass: Boolean(semantics && physical && timed), invocations: observed.map(a => a?.invocations ?? null),
      repeated_read_ms: observed[1]?.elapsed_ms ?? null,
    }
  }
  for (const mode of ['serial', 'parallel']) {
    const off = cases[`${mode}-off`], on = cases[`${mode}-on`]
    const saved = off.repeated_read_ms !== null && on.repeated_read_ms !== null ? off.repeated_read_ms - on.repeated_read_ms : null
    assertGate(report.gates, `e2.${mode}.repeated_read_savings`, off.pass && on.pass && saved > 0,
      { off, on, saved_ms: saved, scope: report.cache_reads.scope })
  }
}

async function main() {
  const opt = options(process.argv.slice(2)), run = resolve(opt['--run-dir'])
  const id = `bench-${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID()}`
  const base = join(run, 'dsh-bench', id)
  await mkdir(base, { recursive: true, mode: 0o700 })
  const config = { id, mode: opt['--mode'], token: opt['--token'], bin: resolve(opt['--dsh-bin'] ?? join(run, 'src/dsh/apps/cli/lib/bin.js')) }
  const runner = resolve(opt['--runner'] ?? join(run, 'src/agenticlayer/dsh-agentic/lib/runner.js'))
  const events = opt['--events'] ? resolve(opt['--events']) : null
  const report = { id, suite: opt['--suite'], mode: config.mode, runner, dshBin: config.bin, output: base, events, cases: {}, gates: [] }
  try {
    const { createHarness } = await import(pathToFileURL(runner).href)
    if (typeof createHarness !== 'function') throw new Error('Compiled runner does not export createHarness')
    if (opt['--suite'] === 'smoke') {
      report.cases.smoke = await taskCase(createHarness, config, base, 'smoke', false)
    } else if (opt['--suite'] === 'e1') {
      await runE1(createHarness, config, report, opt['--repeats'])
    } else if (opt['--suite'] === 'e2') {
      await runE2(createHarness, config, report)
    } else if (opt['--suite'] === 'allof-pair') {
      for (let i = 0; i < opt['--repeats']; i++) {
        for (const serial of (i % 2 ? [false, true] : [true, false])) {
          const name = `repeat-${i+1}-${serial ? 'serial' : 'parallel'}`
          await collectCase(report, name, () => allofCase(createHarness, config, base, name, serial, true))
        }
      }
    } else if (opt['--suite'] === 'drop') {
      const name = 'drop'
      report.cases[name] = await dropCase(createHarness, config, base, name)
      const materialised = await reposTrace(report.events, `${config.id}-${name}`)
      assertGate(report.gates, 'drop.repos_materialised', materialised.length >= 1 && materialised[0].kept_tokens > 0,
        { materialised, note: 'controller log must show the engine repos op completed for this run' })
    } else report.cases.allof = await allofCase(createHarness, config, base, 'allof')
  } catch (error) { report.error = errorDetail(error) }
  report.pass = !report.error && report.gates.every(g => g.pass) && Object.values(report.cases).every(c => c.gates.every(g => g.pass))
  await writeFile(join(base, 'report.json'), json(report), { flag: 'wx', mode: 0o600 })
  console.log(json({ output: base, pass: report.pass, error: report.error, failed: [...report.gates, ...Object.values(report.cases).flatMap(c => c.gates)].filter(g => !g.pass) }))
  if (!report.pass) process.exitCode = 1
}

main().catch(error => { console.error(error.stack ?? error); process.exitCode = 1 })
