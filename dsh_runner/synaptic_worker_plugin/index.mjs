/**
 * Orchestrator-free native DSH worker session bridge for Synaptic-PA ContextCells.
 *
 * Exposes loopback-only `/synaptic-worker/api/*` endpoints to manage per-Cell DSH `Agent`
 * sessions (`synaptic-dsh-worker` preset), registers `peek_peer_cell` and `Shared Trunk T_0`
 * system prompt sections on each worker, enforces read-only/sandbox tool guards, and streams
 * real-time `session/event` updates (`assistant/message`, `tool/result`, compaction) to Python's
 * local callback server so `HybridSynapticBus` operates identically to the Python worker loop.
 */

const WORKER_PRESET = 'synaptic-dsh-worker'
const WORK_TOOLS = new Set([
  'read',
  'write',
  'edit',
  'read_image',
  'glob',
  'grep',
  'bash',
  'pwsh',
  'job_output',
  'job_list',
  'job_kill',
  'skill',
  'todo_write',
  'subagent',
  'subagent_fork',
  'list_subagent_models',
  'send_message',
  'list_agents',
  'interrupt_agent',
  'peek_peer_cell',
])
const ALWAYS_DENY_TOOLS = ['web_search', 'web_fetch', 'ask_question', 'present_artifact']
const DEFAULT_DENY_TOOLS = ['subagent_fork', 'list_agents']
const READONLY_DENY_TOOLS = ['write', 'edit', 'subagent', 'subagent_fork', 'send_message', 'interrupt_agent']
const BLOCKED_CMD_RE =
  /(?:^|[;&|({!`\n]|\b(?:do|then|else|elif|if|while|until|time|exec|nohup|command)\s)\s*(?:(?:env(?:\s+(?:-\S+(?:\s+\S+)?|[A-Za-z_][A-Za-z0-9_]*=\S*))*\s+|timeout(?:\s+-\S+)*\s+\S+\s+)*)(?:\S*\/)?(?:curl|wget)\b/
const BLOCKED_GIT_GH_RE =
  /\bgit\b[^\n;&|`]*\b(?:clone|fetch|pull|ls-remote|push)\b[^\n;&|`]*(?:github\.com|githubusercontent\.com)/i
const KILL_WAIT_MS = 10_000
const DRAIN_LIMIT_MS = 10_000

export const name = 'synaptic-dsh-worker-bridge'
export const inject = [
  'agents',
  'sessionController',
  'sessionPersistence',
  'workspaceRegistry',
  'permissionPresets',
  'tools',
  'systemPrompt',
  'skills',
  'webServer',
  'llm',
  'subprocess',
]

function deepFreeze(obj) {
  if (obj === null || typeof obj !== 'object' || Object.isFrozen(obj)) return obj
  Object.freeze(obj)
  for (const val of Object.values(obj)) deepFreeze(val)
  return obj
}

function makeMessage(requestId, text, mode) {
  const id = `synaptic:${requestId}`
  const source =
    typeof mode === 'object' && mode !== null && 'inject' in mode
      ? { kind: 'synaptic-pa', form: 'notice', summary: String(mode.inject || 'Synaptic update') }
      : { kind: 'synaptic-pa', form: 'relay' }
  return deepFreeze(
    structuredClone({
      id,
      role: 'user',
      content: [{ type: 'text', text: String(text ?? '') }],
      source,
    }),
  )
}

function until(work, signal, ms) {
  if (signal?.aborted) return Promise.resolve(false)
  let stop
  const stopped = new Promise(resolve => {
    stop = () => resolve(false)
  })
  signal?.addEventListener('abort', stop, { once: true })
  const timer = ms === undefined ? undefined : setTimeout(stop, ms)
  timer?.unref?.()
  return Promise.race([work, stopped]).finally(() => {
    signal?.removeEventListener('abort', stop)
    if (timer !== undefined) clearTimeout(timer)
  })
}

const runningStatus = status => status === 'running' || status === 'stopping'

function ownedJobs(jobs, sessionId) {
  if (!jobs) return []
  return jobs.list(sessionId).filter(j => j.owner === sessionId && runningStatus(j.status))
}

function jobsEnd(ctx, sessionId, signal) {
  const jobs = ctx.get('jobs')
  if (!jobs || !ownedJobs(jobs, sessionId).length) return Promise.resolve(false)
  return new Promise(resolve => {
    let settled = false
    const done = val => {
      if (settled) return
      settled = true
      unsubscribe()
      signal?.removeEventListener('abort', aborted)
      resolve(val)
    }
    const aborted = () => done(false)
    const unsubscribe = jobs.events.subscribe({ owner: sessionId }, event => {
      if ((event.type === 'settled' || event.type === 'removed') && !ownedJobs(jobs, sessionId).length) {
        done(true)
      }
    })
    signal?.addEventListener('abort', aborted, { once: true })
    if (signal?.aborted) aborted()
  })
}

const activeSubagentJoins = new WeakMap()
function childrenEnd(subagents, agent) {
  let join = activeSubagentJoins.get(agent)
  if (!join) {
    join = subagents.waitForChildren(agent).finally(() => activeSubagentJoins.delete(agent))
    activeSubagentJoins.set(agent, join)
  }
  return join
}

async function untilQuiet(ctx, agent, signal) {
  for (;;) {
    const subagents = ctx.get('subagents')
    const children = subagents ? await until(childrenEnd(subagents, agent), signal) : false
    const jobs = await jobsEnd(ctx, agent.id, signal)
    if (signal?.aborted || agent.status !== 'idle') return false
    if (!children && !jobs) return true
  }
}

async function killJobs(ctx, sessionId, reason) {
  const jobs = ctx.get('jobs')
  const live = ownedJobs(jobs, sessionId)
  const waits = live.map(j => jobs.wait(j.id, KILL_WAIT_MS, sessionId).catch(() => undefined))
  for (const j of live) {
    try {
      jobs.kill(j.id, sessionId, reason)
    } catch {}
  }
  await Promise.all(waits)
}

async function stopSubagents(ctx, agent, keepInbox = false) {
  const subagents = ctx.get('subagents')
  if (!subagents) return
  const children = await subagents.listChildren(agent.id).catch(() => [])
  const live = children.map(c => c.id).filter(id => ctx.agents.get(id))
  if (!live.length) return
  try {
    await until(subagents.drainChildren(agent, live), undefined, DRAIN_LIMIT_MS)
  } finally {
    agent.cancel({ kind: 'user' }, { keepInbox })
    await until(agent.whenIdle(), undefined, DRAIN_LIMIT_MS)
  }
}

async function stopBackground(ctx, sessionId, reason, keepInbox = false) {
  await killJobs(ctx, sessionId, reason).catch(() => {})
  const agent = ctx.agents.get(sessionId)
  if (agent) await stopSubagents(ctx, agent, keepInbox).catch(() => {})
}

async function readJsonBody(req, maxBytes = 512 * 1024) {
  let size = 0
  const chunks = []
  for await (const chunk of req) {
    const buf = Buffer.from(chunk)
    size += buf.length
    if (size > maxBytes) throw new Error(`Request body exceeds ${maxBytes} bytes`)
    chunks.push(buf)
  }
  const raw = Buffer.concat(chunks).toString('utf8')
  return raw ? JSON.parse(raw) : {}
}

function sendJson(res, status, payload) {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  })
  res.end(JSON.stringify(payload))
}

function sameLoopback(req, port) {
  const host = req.headers.host
  if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) return false
  return req.headers['x-synaptic-worker'] === '1'
}

async function postCallback(callbackUrl, routePath, payload, signal) {
  if (!callbackUrl) return null
  const url = `${callbackUrl.replace(/\/$/, '')}${routePath}`
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json; charset=utf-8' },
    body: JSON.stringify(payload),
    ...(signal ? { signal } : {}),
  })
  if (!resp.ok) {
    const errTxt = await resp.text().catch(() => '')
    throw new Error(`Callback ${routePath} HTTP ${resp.status}: ${errTxt}`)
  }
  return resp.json()
}

function extractTextFromBlocks(blocks) {
  if (!Array.isArray(blocks)) return ''
  const parts = []
  for (const b of blocks) {
    if (!b || typeof b !== 'object') continue
    if (b.type === 'text' && typeof b.text === 'string') parts.push(b.text)
  }
  return parts.join('\n').trim()
}

function extractToolCallsFromBlocks(blocks) {
  if (!Array.isArray(blocks)) return []
  const calls = []
  for (const b of blocks) {
    if (!b || typeof b !== 'object') continue
    if (b.type === 'tool-call' || b.type === 'tool_use' || b.type === 'toolCall' || b.type === 'tool_call') {
      let parsedArgs = b.input ?? b.arguments ?? {}
      if (typeof parsedArgs === 'string') {
        try {
          parsedArgs = JSON.parse(parsedArgs)
        } catch {
          parsedArgs = { raw: parsedArgs }
        }
      }
      calls.push({
        id: String(b.id ?? b.toolCallId ?? b.toolUseId ?? ''),
        name: String(b.name ?? b.toolName ?? ''),
        args: parsedArgs && typeof parsedArgs === 'object' ? parsedArgs : {},
      })
    }
  }
  return calls
}

function extractReasoningFromBlocks(blocks) {
  if (!Array.isArray(blocks)) return ''
  const parts = []
  for (const b of blocks) {
    if (!b || typeof b !== 'object') continue
    if ((b.type === 'reasoning' || b.type === 'thinking') && typeof (b.text ?? b.thinking) === 'string') {
      parts.push(String(b.text ?? b.thinking))
    }
  }
  return parts.join('\n').trim()
}

function extractLastAssistantText(session) {
  try {
    const msgs = session.deriveMessages()
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i]
      if (m && m.role === 'assistant') {
        const txt = extractTextFromBlocks(m.content)
        if (txt) return txt
      }
    }
  } catch {}
  return ''
}

export async function apply(ctx) {
  const sessionSpecs = new Map()
  const bindings = new Map()
  const seenDeliveries = new Map()
  const modelDisposers = new Map()
  const callRegistry = new Map()
  const lastTurnError = new Map()
  let callbackChain = Promise.resolve()

  function findOwnerSpec(session) {
    if (!session) return undefined
    if (sessionSpecs.has(session.id)) return sessionSpecs.get(session.id)
    let parent = session.header?.parentSession
    for (let depth = 0; parent !== undefined && depth < 8; depth++) {
      if (sessionSpecs.has(parent)) return sessionSpecs.get(parent)
      parent = ctx.agents.get(parent)?.session?.header?.parentSession
    }
    return undefined
  }

  function buildGuard(spec, isSubagent = false) {
    return execution => {
      const toolName = String(execution?.name ?? '')
      const args = execution?.arguments && typeof execution.arguments === 'object' ? execution.arguments : {}
      const denyTools = Array.isArray(spec.denyTools) ? spec.denyTools : DEFAULT_DENY_TOOLS
      const customNames = new Set(
        Array.isArray(spec.customTools) ? spec.customTools.map(t => String(t?.name || '')).filter(Boolean) : [],
      )

      if (ALWAYS_DENY_TOOLS.includes(toolName)) {
        return `Error: "${toolName}" is disabled by local workspace sandbox policy.`
      }
      if (!spec.enablePeekPeerCell && toolName === 'peek_peer_cell') {
        return 'Error: "peek_peer_cell" is disabled in this session.'
      }
      if (denyTools.includes(toolName)) {
        return `Error: "${toolName}" is disabled in this session configuration.`
      }
      if (isSubagent && !WORK_TOOLS.has(toolName) && !(spec.allowSubagentCustomTools && customNames.has(toolName))) {
        return `Tool "${toolName}" is not permitted inside a worker subagent.`
      }
      if (spec.readOnly) {
        if (READONLY_DENY_TOOLS.includes(toolName)) {
          return `Error: "${toolName}" is disabled during read-only alignment / subscriber mode.`
        }
        if (toolName === 'bash' && !spec.allowBashInAlignment) {
          return 'Error: `bash` is disabled during non-inspection alignment.'
        }
      }
      if (toolName === 'bash' || toolName === 'pwsh') {
        const cmd = String(args.command ?? '')
        if (BLOCKED_CMD_RE.test(cmd)) {
          return 'Error: Sandbox policy blocked command (`curl` / `wget` disallowed).'
        }
        if (BLOCKED_GIT_GH_RE.test(cmd)) {
          return 'Error: Sandbox policy blocked outbound network access to upstream GitHub repositories.'
        }
      }
      const forbidden = Array.isArray(spec.forbiddenSubstrings) ? spec.forbiddenSubstrings : []
      if (forbidden.length > 0) {
        const rawJson = JSON.stringify(args)
        for (const pat of forbidden) {
          if (pat && rawJson.includes(pat)) {
            return `Error: access to benchmark-internal path "${pat}" is blocked by sandbox policy.`
          }
        }
      }
      return undefined
    }
  }

  function releaseAgent(agent) {
    const binding = bindings.get(agent)
    if (binding) {
      bindings.delete(agent)
      for (const dispose of binding.disposers.splice(0).reverse()) {
        try {
          dispose()
        } catch {}
      }
    }
    const md = modelDisposers.get(agent)
    if (md) {
      modelDisposers.delete(agent)
      try {
        md()
      } catch {}
    }
  }

  function decorateAgent(agent) {
    const ownSpec = sessionSpecs.get(agent.id)
    const ownerSpec = ownSpec ?? findOwnerSpec(agent.session)
    if (!ownerSpec) return

    const isSubagent = !ownSpec
    const enablePeek = Boolean(ownerSpec.enablePeekPeerCell ?? true)
    const denyTools = Array.isArray(ownerSpec.denyTools) ? ownerSpec.denyTools : DEFAULT_DENY_TOOLS
    const customTools = Array.isArray(ownerSpec.customTools) ? ownerSpec.customTools : []
    const sig = JSON.stringify({
      ro: Boolean(ownerSpec.readOnly),
      bashRo: Boolean(ownerSpec.allowBashInAlignment),
      peek: enablePeek,
      deny: denyTools,
      custom: customTools.map(t => String(t?.name || '')),
      subCustom: Boolean(ownerSpec.allowSubagentCustomTools),
      sub: isSubagent,
    })

    // Re-decorate cleanly if spec flags changed
    if (bindings.has(agent)) {
      const prev = bindings.get(agent)
      if (prev.sig === sig) {
        return
      }
      releaseAgent(agent)
    }

    const disposers = []

    if (!isSubagent) {
      disposers.push(
        agent.ctx.systemPrompt.section({
          name: 'synaptic-pa-cell-trunk',
          order: 9999,
          interpolate: false,
          text: () => {
            const cur = sessionSpecs.get(agent.id) ?? ownerSpec
            return [cur.trunkPrompt, cur.modePrompt].filter(Boolean).join('\n\n')
          },
        }),
      )

      if (enablePeek) {
        disposers.push(
          agent.ctx.tools.register({
            name: 'peek_peer_cell',
            description:
              'Zero-side-effect pull of another Context Cell\'s live working slice (`cell_id` or `"ALL"`), optionally narrowed to `rel_path`.',
            parameters: {
              type: 'object',
              properties: {
                cell_id: {
                  type: 'string',
                  description: "Peer cell ID such as 'cell_1', or 'ALL' to inspect all sibling cells.",
                },
                rel_path: {
                  type: 'string',
                  description: 'Optional relative file path (e.g. `src/foo.py`) to inspect in the peer worktree.',
                },
              },
              required: ['cell_id'],
              additionalProperties: false,
            },
            output: {
              schema: {
                type: 'object',
                properties: { slice: { type: 'string' } },
                required: ['slice'],
                additionalProperties: false,
              },
              render: (_args, value) => [{ type: 'text', text: String(value?.slice ?? '') }],
            },
            execute: async (args, execution) => {
              const cur = sessionSpecs.get(agent.id) ?? ownerSpec
              const res = await postCallback(
                cur.callbackUrl,
                '/peek_peer_cell',
                {
                  caller_cell_id: cur.cellId,
                  session_id: agent.id,
                  cell_id: String(args?.cell_id ?? 'ALL'),
                  rel_path: args?.rel_path ? String(args.rel_path) : '',
                },
                execution?.signal,
              )
              return { slice: String(res?.slice ?? 'No peer slice available.') }
            },
          }),
        )
      }
    }

    if ((!isSubagent || ownerSpec.allowSubagentCustomTools) && customTools.length > 0) {
      for (const ct of customTools) {
        const tName = String(ct?.name || '').trim()
        if (!tName) continue
        disposers.push(
          agent.ctx.tools.register({
            name: tName,
            description: String(ct?.description || `Execute ${tName}`),
            parameters:
              ct?.parameters && typeof ct.parameters === 'object'
                ? ct.parameters
                : { type: 'object', properties: {}, additionalProperties: true },
            output: {
              schema: {
                type: 'object',
                properties: { result: { type: 'string' } },
                required: ['result'],
                additionalProperties: false,
              },
              render: (_args, value) => [{ type: 'text', text: String(value?.result ?? '') }],
            },
            execute: async (args, execution) => {
              const cur = sessionSpecs.get(agent.id) ?? findOwnerSpec(agent.session) ?? ownerSpec
              const res = await postCallback(
                cur.callbackUrl,
                '/custom_tool',
                {
                  caller_cell_id: cur.cellId,
                  session_id: agent.id,
                  tool_name: tName,
                  args: args && typeof args === 'object' ? args : {},
                },
                execution?.signal,
              )
              return { result: String(res?.result ?? '') }
            },
          }),
        )
      }
    }

    const restrictSet = new Set([...ALWAYS_DENY_TOOLS, ...denyTools])
    if (!enablePeek) restrictSet.add('peek_peer_cell')
    for (const toolName of restrictSet) {
      try {
        disposers.push(agent.ctx.tools.restrict({ deny: [toolName] }))
      } catch {}
    }

    if (ownerSpec.readOnly) {
      const toDeny = [...READONLY_DENY_TOOLS]
      if (!ownerSpec.allowBashInAlignment) toDeny.push('bash')
      for (const toolName of toDeny) {
        try {
          disposers.push(agent.ctx.tools.restrict({ deny: [toolName] }))
        } catch {}
      }
    }

    disposers.push(
      agent.ctx.tools.guard(execution => {
        const cur = sessionSpecs.get(agent.id) ?? findOwnerSpec(agent.session) ?? ownerSpec
        return buildGuard(cur, isSubagent)(execution)
      }),
    )

    bindings.set(agent, {
      disposers,
      sig,
      readOnly: Boolean(ownerSpec.readOnly),
      subagent: isSubagent,
    })
  }

  async function seedSessionModel(agent, candidates) {
    if (!Array.isArray(candidates) || !candidates.length) return null
    if (agent.session.requestHeader()) return null

    let selected = null
    for (const cand of candidates) {
      if (!cand || !cand.provider || !cand.model) continue
      try {
        const val = await ctx.llm.resolveCallConfig({
          provider: String(cand.provider),
          model: String(cand.model),
          ...(cand.reasoningEffort ? { reasoningEffort: String(cand.reasoningEffort) } : {}),
        })
        selected = {
          provider: val.provider,
          model: val.model,
          ...(val.reasoningEffort ? { reasoningEffort: val.reasoningEffort } : {}),
        }
        break
      } catch {}
    }
    if (!selected) return null

    try {
      agent.session.append('model/selection', selected)
    } catch {
      return selected
    }
    let initial = true
    let assembled
    const innerDisposers = [
      agent.ctx.on(
        'system-prompt/assemble',
        async (_assembly, _context, next) => {
          const sel = initial ? selected : undefined
          const result = await next()
          assembled = sel
          return sel
            ? { ...result, variables: { ...result.variables, provider: sel.provider, model: sel.model } }
            : result
        },
        { prepend: true },
      ),
      agent.ctx.on(
        'agent/request',
        async (_payload, next) => {
          const result = await next()
          if (!assembled) return result
          const { reasoningEffort: _inh, ...rest } = result
          return { ...rest, ...assembled }
        },
        { prepend: true },
      ),
      ctx.on('session/event', (session, event) => {
        if (session === agent.session && (event.type === 'request/header' || event.type === 'model/selection')) {
          initial = false
        }
      }),
    ]
    modelDisposers.set(agent, () => innerDisposers.forEach(d => d()))
    await ctx.sessionPersistence.flush()
    return selected
  }

  const rootDisposers = [
    ctx.on('agent/created', ({ agent }) => {
      decorateAgent(agent)
    }),
    ctx.on('agent/disposed', ({ agent }) => {
      releaseAgent(agent)
    }),
    ctx.inject(['agentPresets'], child => {
      child.on('agent-preset/selected', (sessionId, preset) => {
        if (!sessionSpecs.has(sessionId) || preset === WORKER_PRESET) return
        const agent = ctx.agents.get(sessionId)
        if (!agent) return
        const presets = child.get('agentPresets')
        presets?.select?.(agent, WORKER_PRESET)?.catch?.(() => {})
      })
    }),
    ctx.on('session/event', (session, event) => {
      if (event.type === 'turn/end') {
        const rKind = event.data?.reason?.kind
        if (rKind === 'error') {
          lastTurnError.set(session.id, event.data?.reason)
        } else if (rKind === 'completed') {
          lastTurnError.delete(session.id)
        }
      }
      const ownerSpec = findOwnerSpec(session)
      if (!ownerSpec || !ownerSpec.callbackUrl) return
      const isSubagent = session.id !== ownerSpec.sessionId

      let payload = null
      if (event.type === 'user/message') {
        const msgId = event.data?.id
        if (typeof msgId === 'string' && msgId.startsWith('synaptic:')) {
          let set = seenDeliveries.get(session.id)
          if (!set) {
            set = new Set()
            seenDeliveries.set(session.id, set)
          }
          set.add(msgId)
        }
        return
      } else if (event.type === 'tool/call') {
        const callId = String(event.data?.callId ?? '')
        let parsedArgs = event.data?.arguments ?? {}
        if (typeof parsedArgs === 'string') {
          try {
            parsedArgs = JSON.parse(parsedArgs)
          } catch {
            parsedArgs = { raw: parsedArgs }
          }
        }
        const callMeta = {
          name: String(event.data?.name ?? ''),
          args: parsedArgs && typeof parsedArgs === 'object' ? parsedArgs : {},
        }
        if (callId) callRegistry.set(callId, callMeta)
        payload = {
          kind: 'tool_call',
          cell_id: ownerSpec.cellId,
          session_id: session.id,
          is_subagent: isSubagent,
          seq: event.seq,
          tool_call_id: callId,
          tool_name: callMeta.name,
          tool_args: callMeta.args,
        }
      } else if (event.type === 'assistant/message') {
        const content = event.data?.message?.content ?? []
        const text = extractTextFromBlocks(content)
        const reasoning = extractReasoningFromBlocks(content)
        const toolCalls = extractToolCallsFromBlocks(content)
        for (const tc of toolCalls) {
          if (tc.id) callRegistry.set(tc.id, { name: tc.name, args: tc.args })
        }
        payload = {
          kind: 'assistant_message',
          cell_id: ownerSpec.cellId,
          session_id: session.id,
          is_subagent: isSubagent,
          seq: event.seq,
          text,
          reasoning_excerpt: reasoning.slice(0, 600),
          tool_calls: toolCalls,
        }
      } else if (event.type === 'tool/result') {
        const msg = event.data?.message ?? {}
        const rawText = extractTextFromBlocks(msg.content)
        const callId = String(msg.toolCallId ?? msg.toolUseId ?? msg.source?.callId ?? '')
        const prior = callRegistry.get(callId)
        payload = {
          kind: 'tool_result',
          cell_id: ownerSpec.cellId,
          session_id: session.id,
          is_subagent: isSubagent,
          seq: event.seq,
          tool_call_id: callId,
          tool_name: String(msg.toolName ?? event.data?.toolName ?? prior?.name ?? ''),
          tool_args: prior?.args ?? {},
          is_error: Boolean(msg.isError || event.data?.error),
          output_excerpt: rawText.slice(0, 8000),
        }
      } else if (
        String(event.type || '').startsWith('compaction') ||
        (event.surfaceOp && typeof event.surfaceOp === 'object' && event.surfaceOp.op === 'replace' && event.type === 'user/message')
      ) {
        payload = {
          kind: 'compaction',
          cell_id: ownerSpec.cellId,
          session_id: session.id,
          is_subagent: isSubagent,
          seq: event.seq,
          event_type: event.type,
        }
      }

      if (payload) {
        callbackChain = callbackChain
          .then(() => postCallback(ownerSpec.callbackUrl, '/session_event', payload))
          .catch(() => {})
      }
    }),
  ]

  rootDisposers.push(
    ctx.webServer.register({
      kind: 'prefix',
      path: '/synaptic-worker',
      handler: async (req, res) => {
        const url = new URL(req.url ?? '/', 'http://127.0.0.1')
        if (!url.pathname.startsWith('/synaptic-worker/api/')) {
          res.writeHead(404)
          res.end()
          return
        }
        if (!sameLoopback(req, ctx.webServer.port)) {
          sendJson(res, 403, { error: 'Loopback X-Synaptic-Worker header required' })
          return
        }
        const subRoute = `${req.method ?? 'GET'} ${url.pathname.slice('/synaptic-worker/api/'.length)}`
        try {
          if (subRoute === 'GET health') {
            sendJson(res, 200, {
              ok: true,
              port: ctx.webServer.port,
              activeSessions: sessionSpecs.size,
            })
            return
          }
          if (subRoute === 'GET models') {
            const catalog = await ctx.sessionController.modelCatalog()
            sendJson(res, 200, catalog)
            return
          }
          if (subRoute === 'GET session/status') {
            const sessionId = String(url.searchParams.get('sessionId') || '').trim()
            if (!sessionId) throw new Error('`sessionId` query parameter is required')
            const agent = ctx.agents.get(sessionId)
            if (!agent) {
              sendJson(res, 200, { ok: true, exists: false, status: 'missing', busy: false, pendingInbox: 0 })
              return
            }
            const pendingInbox = agent.inbox.nextTurn.length + agent.inbox.nextStep.length
            sendJson(res, 200, {
              ok: true,
              exists: true,
              status: agent.status,
              busy: agent.status !== 'idle' || pendingInbox > 0,
              pendingInbox,
              seq: agent.session.seq,
            })
            return
          }
          if (subRoute === 'POST session/ensure') {
            const body = await readJsonBody(req)
            const sessionId = String(body.sessionId || '').trim()
            const cwd = String(body.cwd || '').trim()
            if (!sessionId || !cwd) throw new Error('`sessionId` and `cwd` are required')

            const prev = sessionSpecs.get(sessionId) || {}
            const spec = {
              sessionId,
              cellId: String(body.cellId || prev.cellId || sessionId),
              cwd,
              title: String(body.title || prev.title || sessionId),
              readOnly: Boolean(body.readOnly ?? prev.readOnly ?? false),
              allowBashInAlignment: Boolean(body.allowBashInAlignment ?? prev.allowBashInAlignment ?? true),
              enablePeekPeerCell: Boolean(body.enablePeekPeerCell ?? prev.enablePeekPeerCell ?? true),
              denyTools: Array.isArray(body.denyTools)
                ? body.denyTools.map(x => String(x))
                : (prev.denyTools ?? [...DEFAULT_DENY_TOOLS]),
              customTools: Array.isArray(body.customTools) ? body.customTools : (prev.customTools ?? []),
              allowSubagentCustomTools: Boolean(
                body.allowSubagentCustomTools ?? prev.allowSubagentCustomTools ?? false,
              ),
              trunkPrompt: String(body.trunkPrompt ?? prev.trunkPrompt ?? ''),
              modePrompt: String(body.modePrompt ?? prev.modePrompt ?? ''),
              callbackUrl: String(body.callbackUrl ?? prev.callbackUrl ?? ''),
              forbiddenSubstrings: Array.isArray(body.forbiddenSubstrings)
                ? body.forbiddenSubstrings
                : (prev.forbiddenSubstrings ?? []),
            }
            sessionSpecs.set(sessionId, spec)

            let resolved = await ctx.sessionController.resolveAgent(sessionId)
            if ('error' in resolved && resolved.error.code === 'session/not-found') {
              const workspace = await ctx.workspaceRegistry.create(cwd)
              await ctx.sessionController.create({
                sessionId,
                workspaceId: workspace.id,
                agentPreset: WORKER_PRESET,
              })
              resolved = await ctx.sessionController.resolveAgent(sessionId)
            }
            if ('error' in resolved) {
              throw new Error(`Failed to resolve agent ${sessionId}: ${JSON.stringify(resolved.error)}`)
            }
            const agent = resolved.agent
            const selectedModel = await seedSessionModel(agent, body.modelCandidates || [])
            decorateAgent(agent)
            ctx.permissionPresets.set(agent.session, 'danger-full-access')
            await ctx.sessionController.rename({ sessionId, title: spec.title })
            await ctx.sessionPersistence.flush()

            sendJson(res, 200, {
              ok: true,
              sessionId,
              status: agent.status,
              selectedModel,
            })
            return
          }
          if (subRoute === 'POST session/send') {
            const body = await readJsonBody(req)
            const sessionId = String(body.sessionId || '').trim()
            const requestId = String(body.requestId || '').trim()
            const text = String(body.text ?? '')
            const mode = body.mode ?? 'followup'
            if (!sessionId || !requestId) throw new Error('`sessionId` and `requestId` are required')

            const resolved = await ctx.sessionController.resolveAgent(sessionId)
            if ('error' in resolved) {
              throw new Error(`Session not found: ${sessionId}`)
            }
            const agent = resolved.agent
            const msgId = `synaptic:${requestId}`
            let seen = seenDeliveries.get(sessionId)
            if (!seen) {
              seen = new Set()
              seenDeliveries.set(sessionId, seen)
            }
            const inInbox = [...agent.inbox.nextTurn, ...agent.inbox.nextStep].some(m => m.id === msgId)
            let delivered = false
            let effectiveMode = mode
            if (!seen.has(msgId) && !inInbox) {
              if (mode === 'steer_if_busy') {
                effectiveMode = agent.status !== 'idle' ? 'steer' : 'followup'
              }
              const msg = makeMessage(requestId, text, effectiveMode)
              if (typeof effectiveMode === 'object' && effectiveMode !== null && 'inject' in effectiveMode) {
                agent.inject(msg)
                seen.add(msgId)
              } else if (effectiveMode === 'steer') {
                agent.steer(msg)
              } else {
                agent.followup(msg)
              }
              delivered = true
            }
            await ctx.sessionPersistence.flush()
            sendJson(res, 200, { ok: true, delivered, effectiveMode, status: agent.status })
            return
          }
          if (subRoute === 'POST session/wait_quiet') {
            const body = await readJsonBody(req)
            const sessionId = String(body.sessionId || '').trim()
            const timeoutMs = Math.max(1000, Number(body.timeoutMs || 600_000))
            const resolved = await ctx.sessionController.resolveAgent(sessionId)
            if ('error' in resolved) {
              throw new Error(`Session not found: ${sessionId}`)
            }
            const agent = resolved.agent
            const limitCtrl = new AbortController()
            const timer = setTimeout(() => limitCtrl.abort(), timeoutMs)
            timer.unref?.()

            let timedOut = false
            try {
              while (!limitCtrl.signal.aborted) {
                if (agent.status !== 'idle') {
                  await until(agent.whenIdle(), limitCtrl.signal)
                  continue
                }
                const quiet = await untilQuiet(ctx, agent, limitCtrl.signal)
                if (limitCtrl.signal.aborted) break
                const pendingInbox = agent.inbox.nextTurn.length + agent.inbox.nextStep.length
                if (quiet && agent.status === 'idle' && pendingInbox === 0) {
                  break
                }
                await new Promise(r => setTimeout(r, 30))
              }
              if (limitCtrl.signal.aborted) {
                timedOut = true
                await stopBackground(ctx, sessionId, 'Turn wait timeout reached', false)
                if (agent.status !== 'idle') {
                  agent.cancel({ kind: 'user' }, { keepInbox: false })
                  await until(agent.whenIdle(), undefined, 8000)
                }
              }
            } finally {
              clearTimeout(timer)
            }

            await callbackChain
            await ctx.sessionPersistence.flush()
            const turnErr = lastTurnError.get(sessionId)
            if (turnErr) {
              lastTurnError.delete(sessionId)
              throw new Error(`DSH session ${sessionId} turn aborted with provider error: ${JSON.stringify(turnErr)}`)
            }
            const finalText = extractLastAssistantText(agent.session)
            sendJson(res, 200, {
              ok: true,
              quiet: !timedOut,
              timedOut,
              status: agent.status,
              finalText,
              seq: agent.session.seq,
            })
            return
          }
          if (subRoute === 'POST session/cancel') {
            const body = await readJsonBody(req)
            const sessionId = String(body.sessionId || '').trim()
            const reason = String(body.reason || 'cancelled by Synaptic-PA')
            const keepInbox = Boolean(body.keepInbox ?? false)
            const agent = ctx.agents.get(sessionId)
            await stopBackground(ctx, sessionId, reason, keepInbox)
            if (agent && agent.status !== 'idle') {
              agent.cancel({ kind: 'user' }, { keepInbox })
              await until(agent.whenIdle(), undefined, 8000)
            }
            await callbackChain
            await ctx.sessionPersistence.flush()
            sendJson(res, 200, { ok: true, status: agent?.status ?? 'missing' })
            return
          }
          if (subRoute === 'POST shutdown') {
            for (const sid of sessionSpecs.keys()) {
              await stopBackground(ctx, sid, 'Synaptic-PA host shutdown', true)
              const ag = ctx.agents.get(sid)
              if (ag && ag.status !== 'idle') {
                ag.cancel({ kind: 'user' }, { keepInbox: true })
                await until(ag.whenIdle(), undefined, 5000)
              }
            }
            await callbackChain
            await ctx.sessionPersistence.flush()
            sendJson(res, 200, { ok: true })
            return
          }
          sendJson(res, 404, { error: `Unknown route: ${subRoute}` })
        } catch (err) {
          sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) })
        }
      },
    }),
  )

  ctx.effect(() => async () => {
    for (const sid of sessionSpecs.keys()) {
      await stopBackground(ctx, sid, 'Plugin unloading', true).catch(() => {})
    }
    for (const agent of [...bindings.keys()]) releaseAgent(agent)
    for (const d of rootDisposers.splice(0).reverse()) {
      try {
        if (typeof d === 'function') d()
        else if (d && typeof d.dispose === 'function') await d.dispose()
      } catch {}
    }
  })
}
