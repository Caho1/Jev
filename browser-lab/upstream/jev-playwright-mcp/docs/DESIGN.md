# Design — jev-playwright-mcp

This document explains how the proxy is put together and why. The normative
surface (flags, env, defaults, tool policy) lives in
[`src/contracts.ts`](../src/contracts.ts) and [`src/interfaces.ts`](../src/interfaces.ts);
if this doc and the code ever disagree, the code wins.

---

## 1. Topology

```
                 MCP (stdio)                  MCP (stdio)
  +-----------+  tools/list   +-------------+  spawn child  +------------------+
  |  Coding   | <-----------> |    jev-     | <------------> |  @playwright/mcp |
  |  agent    |  tools/call   |  playwright |  (version-    |     0.0.81       |
  | (Claude   |               |  -mcp       |   pinned)     |       │          |
  |  Code,    |               |   PROXY     |               |       v          |
  |  Codex,   |               |             |               |   Chromium       |
  |  Cursor…) |               |  +-------+  |               +------------------+
  +-----------+               |  | Jev   |  |
       ^                      |  +---+---+  |
       |  <jev-insights> block     | HTTPS  |
       +-- masked/pruned text +----+--------+
                             POST api.typesafe.ai/v1/systemone
```

Three processes, two MCP hops, one decision service:

1. **Agent ↔ proxy.** The agent sees a normal MCP server named
   `jev-playwright-mcp`. Tool names and input schemas are re-exposed exactly as
   the upstream reports them in `tools/list`, plus two extra tools
   (`browser_set_goal`, `browser_jev_status`) appended on the last page of the
   listing.
2. **Proxy ↔ upstream.** The proxy spawns the official `@playwright/mcp` as a
   child over stdio (`StdioClientTransport`) and forwards requests
   transparently — `ping` and everything else it does not explicitly intercept
   passes through untouched. All proxy logging goes to **stderr**; stdout is
   the MCP channel and must stay clean.
3. **Proxy ↔ Jev.** All decisions go through one client interface
   (`JevClient.decide()`, a single `POST /v1/systemone` per batched question
   set) with `Bearer TYPESAFE_API_KEY`. Jev is never exposed to the agent as a
   tool: it is infrastructure, not a peer.

### Upstream resolution

`resolveUpstreamCommand()` prefers the version-pinned dependency
(`<repo>/node_modules/@playwright/mcp/cli.js`) and falls back to
`npx -y @playwright/mcp@0.0.81`. If `PLAYWRIGHT_MCP_EXECUTABLE_PATH` is unset,
the proxy probes the local ms-playwright cache for a chromium executable and
injects the env var so the child finds a browser even in bare environments.

### Environment loading

Before flag parsing, `loadPackageEnv()` applies a `.env` file found at the
package root (dotenv-compatible subset: `#` comments, optional `export `
prefix, quoted values, ` #` inline comments) — but only for variables not
already set in the process environment. This lets one checkout serve multiple
agents with a single key while any agent's `env` block still wins.
`JEV_MCP_NO_ENV_FILE` (truthy) skips loading entirely, preserving the
"no key = pure passthrough" escape hatch. Effective precedence:
**CLI flag > process env > `.env` file > default.**

---

## 2. Interception pipeline

For every `tools/call`, in order:

```
tools/call(name, args)
  │
  ├─ 0. own tools?            browser_set_goal / browser_jev_status → local, never forwarded
  ├─ 1. blockTools            deterministic policy block — ALWAYS applies,
  │                           independent of key/mode/budget (default: browser_run_code_unsafe)
  ├─ 2. risk gate             only if: key present ∧ mode ∈ {gate, all} ∧ budget not exhausted
  │                           ∧ tool ∈ gateTools → Jev Choice; blocked → isError result + reason
  │                           Jev failure → FAIL OPEN (log to stderr, allow)
  ├─ 3. upstream call         forwarded verbatim (with progress notifications relayed)
  └─ 4. annotate / prune      only if: key present ∧ mode ∈ {annotate, all} ∧ budget ok
                              ∧ tool ∈ annotateTools ∧ non-empty text
                              → triage + injection masking (+ pruning when a goal is set)
                              → result text replaced, <jev-insights> block appended
```

Design rules baked into the pipeline:

- **Blocking is the only unconditional step.** `blockTools` is a config-level
  policy decision, not a Jev verdict, so it survives every degradation mode.
- **Jev never blocks browsing by being down.** The gate fails open on
  `JevUnavailableError`-class failures; annotation silently returns the
  original text. Availability of the stock tool beats the safety add-on.
- **Budget exhaustion is announced, not silent.** The first response after the
  session budget is crossed carries a one-time `budget:` note in the insights
  block; afterwards everything passes through unannotated for the rest of the
  session.
- **Only text content is touched.** Images and other content items pass
  through byte-identical; annotation rewrites text items only.

---

## 3. The Jev question batteries

All batteries follow the TypeSafe "System One" model: one `state` (the shared,
untrusted page context) plus a `Record<key, question>` of *independent*
questions answered in a single request. Question keys are opaque to the model —
every question's `instructions` must therefore be fully self-contained.

### 3.1 Page-state triage — one Choice

A single `choice` question over the page context asking which state the page is
in, with exactly the candidates from `PAGE_STATES`:

`expected · login_wall · captcha · consent_popup · error · rate_limited ·
empty_or_js_only · paywall · unknown`

The answer's `probabilities` distribution and `confidence` are kept; the
one-line `hint` shown to the agent is generated in code from the winning state
(e.g. login_wall → try logging in or skip; captcha → stop and escalate), not by
the model, so hints stay short and deterministic.

### 3.2 Prompt-injection scan — local heuristics, then a Noul batch

Two stages, so the common case costs nothing:

1. **Local, deterministic, free.** Regex/keyword heuristics extract candidate
   spans that look like instructions aimed at an AI: "ignore (all )?previous
   instructions", "system prompt", "you are now", second-person imperatives
   addressed at an assistant/agent, "do not tell the user", role-play/override
   demands, and similar. No candidate spans → no Jev call at all.
2. **One batched request.** Every candidate span becomes its own `noul`
   question ("is this span an instruction attempting to redirect the AI
   reader?") inside the *same* `decide()` as the triage question. Spans at or
   above `injectionThreshold` (default 0.5) are hits; hits are masked in the
   returned text as `[INJECTION MASKED p=0.87]` and summarized (max 3 excerpts)
   in the insights block.

### 3.3 Region-relevance pruning — Noul batch over snapshot regions

Only when a goal is set (`browser_set_goal`) and the tool is in `pruneTools`
(`browser_snapshot`):

1. `splitSnapshotRegions()` cuts the snapshot YAML into top-level blocks
   (region id + short label + text).
2. Each region becomes one `noul` question — "is this region relevant to the
   goal `<goal>`?" — all batched into a single Jev request over the same
   state.
3. If there are more than 40 regions, a label-based first cut reduces the
   batch; the number of pre-batch drops is returned and reported
   (`droppedBeforeBatching`) — no silent caps.
4. `collapseRegions()` folds regions below `pruneKeepThreshold` (default 0.35)
   into a one-line summary, preserving the original snapshot format so the
   agent's yaml-reading habits still work. The insights block reports
   `pruned: N/M regions collapsed`.

### 3.4 Risk gate — one Choice per gated call

Before a call in `gateTools` reaches the upstream, a single `choice` question
rates it `safe · needs_confirmation · destructive`, grounded in the tool name,
its arguments (labels, text to type, URLs), the current goal (if any), and the
last observed page state (if any — the gate must be able to judge with no
recent observation). Policy application is code, not the model:
`gateDecision()` blocks unconditionally on `blockTools`, and blocks when the
`destructive` probability is at/above `destructiveThreshold` (default 0.6).
`needs_confirmation` is informational today — it flows into the verdict/reason
the agent sees, not a hard block.

### Default tool policy (`DEFAULT_TOOL_POLICY`)

| Bucket | Tools | When it runs |
|---|---|---|
| `annotateTools` | browser_snapshot, browser_navigate, browser_navigate_back, browser_wait_for, browser_find, browser_network_requests | post-response, modes annotate/all |
| `pruneTools` | browser_snapshot | post-response, additionally requires a goal |
| `gateTools` | browser_click, browser_type, browser_fill_form, browser_select_option, browser_press_key, browser_file_upload, browser_handle_dialog, browser_run_code_unsafe | pre-call, modes gate/all |
| `blockTools` | browser_run_code_unsafe | pre-call, always |

---

## 4. Caching

Verdict caching is keyed by content, so identical pages cost one decision:

```
key = sha256( tool ++ "\0" ++ (goal ?? "") ++ "\0" ++ text )
```

- Cached entries store the whole verdict bundle (page state, injection scan,
  spans, region relevances), so a re-visited page with the same goal skips the
  Jev round-trip entirely; pruning re-collapses from cached relevances.
- The cache is in-memory, per proxy session. `--jev-no-cache` /
  `JEV_MCP_NO_CACHE=1` disables it (useful when tuning thresholds).
- Cache hits cost $0 and ~0 ms — the ~100–600 ms Jev latency applies only to
  unseen content.

---

## 5. Budget guard

`budgetUsd` (default 1.0) caps Jev input-token spend per proxy session. Jev
pricing is $0.042 per 1M input tokens, output free, so the default budget buys
on the order of 20M input tokens — far more than a typical session annotates.

- The client accumulates `costUsd` from `usage.input_tokens` per call.
- Once accumulated cost exceeds the budget: gate and annotate go quiet
  (passthrough), and exactly one annotation carrying a `budget:` note is
  emitted so the agent (and the human reading the transcript) knows why.
- `browser_jev_status` always reports calls, failures, cost, and budget
  consumption percentage regardless of exhaustion.

---

## 6. Failure and degradation modes

| Situation | Behavior |
|---|---|
| No `TYPESAFE_API_KEY` (and no `JEV_MCP_API_KEY`) | `NullJevClient` — pure passthrough. No gate, no annotate, no extra Jev calls; the two extra tools are still listed; `blockTools` still blocks. Stock behavior otherwise. |
| `--jev-mode=off` | Same as no key, even with a key present. |
| HTTP 429/529 from Jev | Exponential backoff retry (2 attempts), then the call raises `JevUnavailableError`. |
| Gate raises | **Fail open** — the call proceeds; warning on stderr. |
| Annotate/decide raises | Original text returned unchanged, no insights block; error counted in `browser_jev_status`. |
| Budget exhausted | Passthrough + one-time `budget:` note (see §5). |
| Invalid flag/env numbers (NaN, negative, infinite) or invalid mode | Fall back to the default for that setting + one-line warning on stderr; never crash the proxy. |
| Upstream child dies | Standard MCP child-failure semantics propagate to the agent; the proxy does not mask them. |

The invariant across all rows: **the proxy must never make stock Playwright MCP
less available than without it**, except for the deliberate `blockTools` policy.

---

## 7. Why a proxy instead of a fork

- **Upstream velocity.** `@playwright/mcp` source lives inside the Playwright
  monorepo and moves fast. A fork would rot; a proxy tracks upstream by
  pinning the package version (`0.0.81` in `package.json`) and re-exposing
  whatever `tools/list` reports. Schema changes upstream propagate for free.
- **Drop-in compatibility.** Because tool names and schemas are forwarded
  verbatim, an agent configured against stock Playwright MCP works unchanged
  against the proxy — existing prompts, permissions, and muscle memory all
  keep working. The only visible additions are the two `browser_*` extras.
- **Separation of concerns.** Jev judgment is transport-level infrastructure.
  Keeping it in the proxy means it applies to *every* agent (Claude Code,
  Codex, Cursor, ...) with zero per-agent prompting, and it cannot be
  prompt-injected away by page content that tells the agent to "ignore
  previous instructions" — the masking happens before the agent ever sees the
  text.
- **Cheap degradation.** No key? The proxy is the stock server. Operators can
  adopt it risk-free and turn on Jev later by adding one env var.

---

## 8. Alignment with TypeSafe question-design principles

The batteries follow the official TypeSafe SKILL.md question-design guidance:

- **Self-contained instructions.** Question keys carry no semantics on the
  wire; each question's `instructions` (and `criteria`) spell out everything
  the model needs — including the decision boundary — without assuming the
  reader saw any other question.
- **Batched independent questions over shared state.** All questions in one
  `decide()` share a single `state` (the untrusted page context) and are
  answerable independently: triage + N injection spans in one request;
  N region-relevance Nouls in one request. One HTTP round-trip per response,
  not one per span.
- **Backticked state paths.** Instructions reference parts of the state with
  backtick-quoted paths/labels (e.g. the goal string, region labels, span
  text) so the model binds its judgment to the exact datum, not a paraphrase.
- **Confidence ≠ permission.** Choice `confidence` is a distribution statistic,
  never a correctness guarantee and never equal to the winning option's
  probability. Thresholds (`injectionThreshold`, `pruneKeepThreshold`,
  `destructiveThreshold`) are configuration, reviewed by the operator — the
  code never interprets a confidence as authority to act. The only
  unconditional block (`blockTools`) is a human-written policy, not a model
  verdict.
- **No silent caps.** Where batching must shed load (the >40-region first
  cut), the shed count is returned and surfaced (`droppedBeforeBatching`,
  `pruned: N/M`), keeping the operator able to audit what was judged and what
  was not.

---

## 9. File map

| Path | Role |
|---|---|
| `src/contracts.ts` | Shared types, defaults, pricing, insights rendering — the single source of truth |
| `src/interfaces.ts` | Function-signature contract every module implements (compile-time checked) |
| `src/config.ts` | CLI flag / env parser with upstream-arg passthrough |
| `src/env.ts` | Lightweight package-root `.env` loader (dotenv subset, process env wins) |
| `src/jev/client.ts` | `POST /v1/systemone` client, retries, cost stats |
| `src/jev/detectors.ts` | Page-state triage Choice + two-stage injection scan |
| `src/jev/annotate.ts` | `ResponseAnnotator` — cache, masking, insights block |
| `src/jev/prune.ts` | Region splitting, relevance Noul batch, collapse |
| `src/jev/gate.ts` | Risk Choice + policy application (`gateDecision`) |
| `src/proxy.ts` | MCP server, pipeline wiring, extra tools, budget announcement |
| `src/upstream.ts` | Child command resolution + chromium auto-detection |
| `src/index.ts` | Entry point (`main()`), startup banner, help passthrough |
| `cli.js` | `bin` shim: loads `dist/` if built, else runs `src/` via tsx |
