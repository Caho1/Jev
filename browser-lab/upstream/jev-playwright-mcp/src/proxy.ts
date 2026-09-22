/**
 * proxy.ts — MCP 프록시 코어.
 *
 *  [downstream agent] ←stdio/InMemory→ [OUR Server] ←stdio child→ [@playwright/mcp]
 *
 *  - tools/list: 업스트림 도구 + 자체 도구 2개(browser_set_goal / browser_jev_status) 병합.
 *  - tools/call: blockTools 정책 차단 → Jev 위험 게이트(선택) → 업스트림 호출 →
 *    text 콘텐츠에 한해 Jev 어노테이션(선택). 이미지/리소스는 무조건 원본.
 *  - 그 외 모든 요청(ping, prompts/*, resources/*, completion/*, ...)은 투명 포워딩.
 *  - Jev 비활성(키 없음 / mode=off)이면 바이트 단위 순수 passthrough.
 *  - 예산 초과 시 이후 Jev 스킵 + 최초 1회 경고 어노테이션.
 *  - 로그는 stderr에만 (stdout은 MCP 채널).
 *
 *  jev/* 모듈(client/annotate/gate)은 정적 import — 통합 단계부터 모든
 *  모듈이 함께 컴파일·검증된다 (서명 드리프트는 interfaces.ts 검증이 잡음).
 */
import process from "node:process";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { RequestOptions } from "@modelcontextprotocol/sdk/shared/protocol.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import {
  CallToolRequestSchema,
  CompleteRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  PingRequestSchema,
  ReadResourceRequestSchema,
  ResultSchema,
  SetLevelRequestSchema,
  type CallToolResult,
  type ContentBlock,
  type ListToolsResult,
  type ServerCapabilities,
  type Tool,
} from "@modelcontextprotocol/sdk/types.js";
import {
  INSIGHTS_CLOSE,
  neutralizeInsightMarkers,
  renderInsightsBlock,
  renderRiskLine,
  type AnnotatedContent,
  type GateDecision,
  type GoalState,
  type JevClient,
  type JevMcpConfig,
  type RiskVerdict,
} from "./contracts.js";
import { ResponseAnnotator } from "./jev/annotate.js";
import { createJevClient } from "./jev/client.js";
import { assessRisk, gateDecision, type GatePageContext, type ToolCallContext } from "./jev/gate.js";
import { detectChromiumExecutable, resolveUpstreamCommand, stripJevSecrets } from "./upstream.js";

export const PROXY_NAME = "jev-playwright-mcp";
export const PROXY_VERSION = "0.1.0";

/** tools/call 포워딩 제한시간 — 브라우저 조작이 길어질 수 있어 여유 있게. */
const CALL_FORWARD_TIMEOUT_MS = 10 * 60_000;
/** 그 외 요청 제한시간 (SDK 기본값과 동일). */
const FORWARD_TIMEOUT_MS = 60_000;
/** 게이트 직전 관측으로 넘길 text 길이 상한. */
const OBSERVATION_MAX_CHARS = 8_000;

function log(msg: string): void {
  process.stderr.write(`[${PROXY_NAME}] ${msg}\n`);
}

// ─── 주입 가능한 의존성 (테스트/프로그램 실행용) ───────────────────────────

/** 업스트림 자식 프로세스 실행 정의 (기본: resolveUpstreamCommand + chromium env 주입). */
export interface UpstreamSpawn {
  command: string;
  args: string[];
  env?: Record<string, string>;
}

/** 응답 후처리 훅 — 기본 구현은 jev/annotate.js의 ResponseAnnotator. */
export type AnnotateFn = (
  tool: string,
  args: unknown,
  text: string,
  goal: GoalState | null,
) => Promise<AnnotatedContent>;

/** 호출 전 게이트 훅 — 기본 구현은 jev/gate.js의 assessRisk + gateDecision. */
export type GateFn = (call: ToolCallContext) => Promise<GateDecision>;

export interface ProxyDeps {
  /** JevClient 주입 (기본: jev/client.js createJevClient). */
  client?: JevClient;
  /** 업스트림 실행 오버라이드 (테스트용 stub). */
  upstream?: UpstreamSpawn;
  /** 다운스트림 transport (기본: StdioServerTransport; 테스트는 InMemoryTransport). */
  serverTransport?: Transport;
  /** 어노테이터 주입. */
  annotate?: AnnotateFn;
  /** 게이트 주입. */
  gate?: GateFn;
}

export interface ProxySession {
  server: Server;
  upstream: Client;
  client: JevClient;
  setGoal(goal: string): GoalState;
  getGoal(): GoalState | null;
  /** 세션 종료 대기 (양측 transport 중 하나가 닫히면 resolve). */
  readonly closed: Promise<void>;
  /** 멱등 종료 — 자식 프로세스 종료 포함. */
  close(): Promise<void>;
}

// ─── 자체 도구 정의 ────────────────────────────────────────────────────────

const SET_GOAL_TOOL: Tool = {
  name: "browser_set_goal",
  title: "Set session goal (Jev pruning context)",
  description:
    "Set the session goal. When Jev annotation is active, browser_snapshot regions irrelevant " +
    "to this goal are collapsed to one-line summaries, saving agent context tokens. " +
    "Call again to replace the goal.",
  inputSchema: {
    type: "object",
    properties: {
      goal: {
        type: "string",
        minLength: 1,
        description: "What this browser session is trying to accomplish.",
      },
    },
    required: ["goal"],
    additionalProperties: false,
  },
};

const JEV_STATUS_TOOL: Tool = {
  name: "browser_jev_status",
  title: "Jev proxy status",
  description:
    "Report jev-playwright-mcp status: mode, enabled, session goal, Jev usage stats, " +
    "budget consumption, and the active tool policy.",
  inputSchema: { type: "object", properties: {} },
};

/** tools/list 병합 시 끝에 붙는 자체 도구. */
export const EXTRA_TOOLS: readonly Tool[] = [SET_GOAL_TOOL, JEV_STATUS_TOOL];

// ─── 헬퍼 ─────────────────────────────────────────────────────────────────

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * 다운스트림 요청 params에서 링크 로컬 progressToken 을 제거한다.
 * progressToken 은 다운스트림↔프록시 연결에서만 유효한 불투명 값 —
 * onprogress 를 달지 않는 포워딩에서는 SDK 가 토큰을 교체해 주지 않으므로
 * 그대로 upstream 으로 흘려보내면 잘못된 요청으로 progress 가 오라우팅된다.
 */
function stripLinkScopedProgressToken<T>(params: T): T {
  if (typeof params !== "object" || params === null) return params;
  const p = params as Record<string, unknown>;
  const meta = p["_meta"];
  if (typeof meta !== "object" || meta === null || !("progressToken" in meta)) return params;
  const { progressToken: _dropped, ...restMeta } = meta as Record<string, unknown>;
  void _dropped;
  const clone = { ...p };
  if (Object.keys(restMeta).length > 0) clone["_meta"] = restMeta;
  else delete clone["_meta"];
  return clone as T;
}

/** 진행 알림 params 의 progressToken (없으면 undefined). */
function progressTokenOf(params: unknown): string | number | undefined {
  const meta = (params as { _meta?: { progressToken?: unknown } } | undefined)?._meta;
  const token = meta?.progressToken;
  return typeof token === "string" || typeof token === "number" ? token : undefined;
}

/** 이미 렌더링된 insights 블록에 risk 한 줄을 병합해 넣는다. */
function mergeRiskLine(block: string, risk: RiskVerdict): string {
  const closeIdx = block.lastIndexOf(INSIGHTS_CLOSE);
  if (closeIdx === -1) return block;
  return `${block.slice(0, closeIdx)}${renderRiskLine(risk)}\n${block.slice(closeIdx)}`;
}

// ─── 업스트림 실행 환경 구성 ───────────────────────────────────────────────

function buildDefaultUpstreamSpawn(config: JevMcpConfig): UpstreamSpawn {
  const { command, args } = resolveUpstreamCommand();
  // SDK 기본 env는 안전한 최소 집합만 상속한다(PLAYWRIGHT_* 유실).
  // 스톡 동작 보존을 위해 부모 env를 통째로 넘기되, 이 프록시 전용 결제
  // 크리덴셜(TYPESAFE_API_KEY/JEV_MCP_API_KEY)은 제외한다 — 업스트림 자식과
  // 그 Chromium 손자들(신뢰할 수 없는 웹 콘텐츠 렌더)이 읽을 이유가 없다.
  const fullEnv: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) fullEnv[key] = value;
  }
  const env = stripJevSecrets(fullEnv);
  if (!env.PLAYWRIGHT_MCP_EXECUTABLE_PATH) {
    const exe = detectChromiumExecutable();
    if (exe) {
      env.PLAYWRIGHT_MCP_EXECUTABLE_PATH = exe;
      log(`injected PLAYWRIGHT_MCP_EXECUTABLE_PATH=${exe}`);
    }
  }
  return { command, args: [...args, ...config.upstreamArgs], env };
}

// ─── 세션 구성 ─────────────────────────────────────────────────────────────

export async function createProxySession(
  config: JevMcpConfig,
  deps: ProxyDeps = {},
): Promise<ProxySession> {
  const spawn = deps.upstream ?? buildDefaultUpstreamSpawn(config);

  // 1) 업스트림 자식 + initialize 핸드셰이크.
  const upstreamTransport = new StdioClientTransport({
    command: spawn.command,
    args: spawn.args,
    env: spawn.env,
  });
  const upstream = new Client({ name: PROXY_NAME, version: PROXY_VERSION });
  await upstream.connect(upstreamTransport);
  log(
    `upstream connected: ${spawn.command} ${spawn.args.join(" ") || "(no args)"}`,
  );

  // upstream.connect 성공 이후의 wiring + 다운스트림 연결 전체를 try로 감싼다 —
  // 도중 무엇이든 실패하면 이미 살아 있는 업스트림 자식을 종료하고 다시 던진다
  // (SDK Client.connect 는 initialize 실패에만 스스로 정리되고, 여기는 그 이후다).
  try {

    // 2) 업스트림 capabilities를 미러링 + tools는 항상(자체 도구 때문에).
    const upstreamCaps = upstream.getServerCapabilities() ?? {};
    const capabilities: ServerCapabilities = {
      ...upstreamCaps,
      tools: { listChanged: true },
    };

    const server = new Server({ name: PROXY_NAME, version: PROXY_VERSION }, {
      capabilities,
      instructions:
        "Playwright MCP (official) proxied by jev-playwright-mcp. " +
        "Snapshot/navigation responses may carry a <jev-insights> block with page-state " +
        "triage, prompt-injection masking notes, and goal-based pruning. " +
        "Use browser_set_goal to focus pruning; browser_jev_status for diagnostics.",
    });

    // 3) 세션 상태 (module-level per-connection).
    //    deps.client 미지정 시 실제 createJevClient(키 없으면 NullJevClient).
    const client = deps.client ?? createJevClient(config);
    let goal: GoalState | null = null;
    let lastObservation: GatePageContext | null = null;
    let budgetAnnounced = false;
    let annotator: ResponseAnnotator | undefined;

    const jevActive = (): boolean => client.enabled && config.mode !== "off";
    const overBudget = (): boolean =>
      jevActive() && client.stats().costUsd > config.budgetUsd;

    const annotateFn: AnnotateFn =
      deps.annotate ??
      (async (tool, args, text, goalArg) => {
        if (annotator === undefined) annotator = new ResponseAnnotator(client, config);
        return annotator.annotate(tool, args, text, goalArg);
      });

    const gateFn: GateFn =
      deps.gate ??
      (async (call) => gateDecision(await assessRisk(client, call, config), call.tool, config));

    // ── tools/list: 업스트림 병합 + 자체 도구 ──
    server.setRequestHandler(ListToolsRequestSchema, async (request, extra) => {
      // 다운스트림 progressToken 은 링크 로컬 값 — 업스트림으로 흘려보내지 않는다.
      const result: ListToolsResult = await upstream.listTools(
        stripLinkScopedProgressToken(request.params),
        {
          signal: extra.signal,
          timeout: FORWARD_TIMEOUT_MS,
          resetTimeoutOnProgress: true,
        },
      );
      const tools = [...result.tools];
      // 페이지가 더 있으면 마지막 페이지에만 붙인다 (중복 방지).
      if (!result.nextCursor) tools.push(...EXTRA_TOOLS);
      return { ...result, tools };
    });

    // ── tools/call: 차단 → 게이트 → 업스트림 → 어노테이션 ──
    server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
      const name = request.params.name;
      const args = (request.params.arguments ?? {}) as Record<string, unknown>;

      // 자체 도구는 로컬 처리 (절대 업스트림으로 포워딩하지 않는다).
      if (name === SET_GOAL_TOOL.name) return handleSetGoal(args);
      if (name === JEV_STATUS_TOOL.name) return handleStatus();

      // 1) 정책 차단 — Jev와 무관하게 항상 적용.
      if (config.toolPolicy.blockTools.includes(name)) {
        return blockedResult(name, config.toolPolicy.blockTools);
      }

      const useJev = jevActive();
      const budgetGone = overBudget();

      // 2) 위험 게이트 — 예산 소진에도 계속 돈다(게이트는 gated 호출당 작은 질문
      //    1개뿐). 예산 끊김이 파괴 행동 차단을 영구히 해제하는 fail-open 트리거가
      //    되지 않게 하는 방어. 예산은 어노테이션/프루닝만 끈다.
      let gateVerdict: RiskVerdict | null = null;
      if (useJev && (config.mode === "gate" || config.mode === "all") &&
          config.toolPolicy.gateTools.includes(name)) {
        let decision: GateDecision;
        try {
          decision = await gateFn({
            tool: name,
            args,
            goal,
            recentPage: lastObservation ?? undefined,
          });
        } catch (err) {
          // Jev 장애가 브라우징을 막아서는 안 된다 — fail-open.
          log(`WARN risk gate failed for ${name}, failing open (${errText(err)})`);
          decision = { allow: true, verdict: null, blockedMessage: null };
        }
        if (!decision.allow) {
          const msg =
            decision.blockedMessage ??
            (decision.verdict
              ? `Blocked by Jev risk gate: ${decision.verdict.level} ` +
                `(p=${decision.verdict.probability.toFixed(2)}) — ${decision.verdict.reason}`
              : "Blocked by Jev risk gate.");
          log(`gate blocked ${name}: ${msg}`);
          return textErrorResult(msg);
        }
        // 허용된 호출이라도 verdict 는 결과에 경고 어노테이션으로 살린다
        // (needs_confirmation / 미달 destructive 가 safe 와 구분되지 않으면
        //  게이트의 중간 등급이 조용히 사라진다).
        if (decision.verdict && decision.verdict.level !== "safe") {
          gateVerdict = decision.verdict;
        }
      }

      // 3) 업스트림 호출. 다운스트림이 준 progressToken(있다면)을 기억해
      //    진행 알림에 다시 붙여 중계한다 — SDK 가 업스트림 요청에는 자기
      //    토큰을 박으므로 onprogress 로 돌아온 params 에는 토큰이 없다.
      const downstreamToken = progressTokenOf(request.params);
      const result = (await upstream.callTool(request.params, undefined, {
        signal: extra.signal,
        timeout: CALL_FORWARD_TIMEOUT_MS,
        resetTimeoutOnProgress: true,
        onprogress: (p) => {
          if (downstreamToken === undefined) return; // 토큰 없는 요청은 중계 불가.
          void extra.sendNotification({
            method: "notifications/progress",
            params: { ...p, progressToken: downstreamToken },
          });
        },
      })) as CallToolResult;

      return postProcess(name, args, result, useJev, budgetGone, gateVerdict);

      // ── 로컬 도구 핸들러 ──

      function handleSetGoal(toolArgs: Record<string, unknown>): CallToolResult {
        const raw = toolArgs.goal;
        const goalText = typeof raw === "string" ? raw.trim() : "";
        if (!goalText) {
          return textErrorResult(
            "browser_set_goal requires a non-empty 'goal' string.",
          );
        }
        goal = { goal: goalText, setAt: new Date().toISOString() };
        log(`goal set: ${goalText}`);
        // 실제 활성 모드를 반영해 답한다 — 꺼져 있는 프루닝을 '된다'고 약속하면
        // 에이전트의 컨텍스트 예산 계획이 왜곡된다.
        const pruningActive =
          client.enabled && (config.mode === "annotate" || config.mode === "all");
        const effect = pruningActive
          ? `Jev will now prune ${config.toolPolicy.pruneTools.join(", ") || "(no tools)"} ` +
            `responses down to regions relevant to this goal. ` +
            `Set a new goal anytime to replace it.`
          : client.enabled
            ? `Jev mode is "${config.mode}" — annotation/pruning is not active in this mode, ` +
              `so no pruning will occur (requires mode annotate or all). The goal is stored ` +
              `and takes effect if the session later runs in an annotating mode.`
            : `Jev is disabled (no API key) — pure passthrough, so no pruning will occur. ` +
              `The goal is stored for when a key is configured.`;
        return {
          content: [{ type: "text", text: `Goal set: ${goalText}\n${effect}` }],
        };
      }

      function handleStatus(): CallToolResult {
        const stats = client.stats();
        const payload = {
          mode: config.mode,
          enabled: client.enabled,
          goal,
          stats,
          budgetUsd: config.budgetUsd,
          budgetConsumedPercent:
            config.budgetUsd > 0
              ? Math.round((stats.costUsd / config.budgetUsd) * 10_000) / 100
              : null,
          budgetExhausted: overBudget(),
          // 저급 모드 노출: 예산 소진 시 어노테이션만 꺼지고 게이트는 계속 돈다.
          gateActive: jevActive() && (config.mode === "gate" || config.mode === "all"),
          annotateActive: jevActive() && !overBudget() &&
            (config.mode === "annotate" || config.mode === "all"),
          // no silent caps: 배치 한도 초과로 심사 없이 사전 컷된 지역 누적 수.
          pruneDroppedBeforeBatching: annotator?.pruneStats.droppedBeforeBatching ?? 0,
          toolPolicy: config.toolPolicy,
        };
        return { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }] };
      }
    });

    /** 결과 후처리: 어노테이션 / 게이트 경고 / 예산 경고 1회 / 직전 관측 기록. */
    async function postProcess(
      name: string,
      args: unknown,
      result: CallToolResult,
      useJev: boolean,
      budgetGone: boolean,
      gateVerdict: RiskVerdict | null,
    ): Promise<CallToolResult> {
      const content = result.content;
      if (!content || content.length === 0) return result;

      // Jev 가 활성한 세션에서는 untrusted 텍스트의 <jev-insights> 마커를 무력화한다
      // (위조 어노테이션 블록이 진짜 블록과 구분 불가로 섞이는 것을 차단).
      let workingContent = content;
      if (useJev) {
        workingContent = content.map((block) =>
          block.type === "text"
            ? { ...block, text: neutralizeInsightMarkers(block.text) }
            : block,
        );
      }

      // 직전 관측(게이트 근거) — 어노테이션 이전(마커 무력화 후) 텍스트 기준.
      const firstText = workingContent.find((b): b is Extract<ContentBlock, { type: "text" }> =>
        b.type === "text");
      if (firstText) {
        lastObservation = {
          tool: name,
          text: firstText.text.slice(0, OBSERVATION_MAX_CHARS),
        };
      }

      const annotateMode =
        useJev && !budgetGone && (config.mode === "annotate" || config.mode === "all");

      if (annotateMode) {
        const newContent: ContentBlock[] = [];
        let riskMerged = false;
        for (const block of workingContent) {
          if (block.type === "text" && block.text.trim() !== "") {
            let text = block.text;
            try {
              const annotated = await annotateFn(name, args, block.text, goal);
              text = annotated.insightsBlock
                ? `${annotated.text}\n\n${annotated.insightsBlock}`
                : annotated.text;
            } catch (err) {
              log(`WARN annotation failed for ${name}, keeping original (${errText(err)})`);
            }
            // 게이트가 허용했지만 risky 로 본 호출 — 첫 어노테이션 블록에 경고 병합.
            if (gateVerdict && !riskMerged && text.includes(INSIGHTS_CLOSE)) {
              text = mergeRiskLine(text, gateVerdict);
              riskMerged = true;
            }
            newContent.push(text === block.text ? block : { ...block, text });
          } else {
            newContent.push(block); // 이미지/리소스 등은 무조건 원본.
          }
        }
        // 어노테이션이 블록을 못 만든 경로(예: Jev 장애)여도 게이트 경고는 붙인다.
        if (gateVerdict && !riskMerged) {
          const idx = newContent.findIndex((b) => b.type === "text");
          if (idx >= 0 && newContent[idx]?.type === "text") {
            const orig = newContent[idx] as Extract<ContentBlock, { type: "text" }>;
            newContent[idx] = {
              ...orig,
              text: `${orig.text}\n\n${renderInsightsBlock({ risk: gateVerdict })}`,
            };
          }
        }
        return { ...result, content: newContent };
      }

      // 어노테이션 미실행 경로(gate 모드 / 예산 소진): 게이트 경고 + 예산 노트(1회)를
      // 첫 text 블록에 붙인다.
      const tail: { risk?: RiskVerdict; budgetNote?: string } = {};
      if (gateVerdict) tail.risk = gateVerdict;
      if (useJev && budgetGone && !budgetAnnounced) {
        tail.budgetNote =
          `budget: Jev budget exhausted ($${client.stats().costUsd.toFixed(4)} > ` +
          `$${config.budgetUsd}) — annotation/pruning passthrough for the rest of this ` +
          `session (the risk gate stays active)`;
      }
      if (tail.risk !== undefined || tail.budgetNote !== undefined) {
        const idx = workingContent.findIndex((b) => b.type === "text");
        if (idx >= 0 && workingContent[idx]?.type === "text") {
          if (tail.budgetNote !== undefined) budgetAnnounced = true;
          const note = renderInsightsBlock(tail);
          const newContent = workingContent.map((block, i) =>
            i === idx && block.type === "text"
              ? { ...block, text: `${block.text}\n\n${note}` }
              : block,
          );
          return { ...result, content: newContent };
        }
        // text 블록이 없으면 다음 기회에 붙인다.
      }

      return { ...result, content: workingContent };
    }

    // ── 그 외 표준 메서드: 타입화된 투명 포워딩 ──
    // SDK는 핸들러 등록 시점에 capability를 검사하므로, 업스트림이 광고한
    // capability에 대해서만 등록한다 (광고 없는 메서드는 스펙상 올 수 없다).
    // 이 경로들은 onprogress 를 달지 않으므로 SDK 가 progressToken 을 교체해
    // 주지 않는다 — 다운스트림 링크 로컬 토큰이 새어나가지 않게 여기서 뗀다.
    const fwdOpts = (extra: { signal?: AbortSignal }): RequestOptions => ({
      signal: extra.signal,
      timeout: FORWARD_TIMEOUT_MS,
      resetTimeoutOnProgress: true,
    });

    // ping은 capability 불필요 — 업스트림까지 실제 건강검진.
    server.setRequestHandler(PingRequestSchema, async (_req, extra) =>
      upstream.ping(fwdOpts(extra)));
    if (upstreamCaps.prompts) {
      server.setRequestHandler(ListPromptsRequestSchema, async (req, extra) =>
        upstream.listPrompts(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
      server.setRequestHandler(GetPromptRequestSchema, async (req, extra) =>
        upstream.getPrompt(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
    }
    if (upstreamCaps.resources) {
      server.setRequestHandler(ListResourcesRequestSchema, async (req, extra) =>
        upstream.listResources(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
      server.setRequestHandler(ReadResourceRequestSchema, async (req, extra) =>
        upstream.readResource(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
      server.setRequestHandler(ListResourceTemplatesRequestSchema, async (req, extra) =>
        upstream.listResourceTemplates(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
    }
    if (upstreamCaps.completions) {
      server.setRequestHandler(CompleteRequestSchema, async (req, extra) =>
        upstream.complete(stripLinkScopedProgressToken(req.params), fwdOpts(extra)));
    }
    if (upstreamCaps.logging) {
      server.setRequestHandler(SetLevelRequestSchema, async (req, extra) =>
        upstream.setLoggingLevel(req.params.level, fwdOpts(extra)));
    }

    // ── 미등록 메서드(커스텀/확장)용 원시 폴백 포워딩 ──
    // SDK 1.30 의 request() 는 resultSchema 로 undefined 를 받지 못한다(파싱
    // 단계에서 crash) — ResultSchema(loose passthrough)로 원시 결과를 보존해 받는다.
    const rawUpstream = upstream as unknown as {
      request(
        req: { method: string; params?: unknown },
        resultSchema: typeof ResultSchema,
        options?: RequestOptions,
      ): Promise<unknown>;
    };
    server.fallbackRequestHandler = async (request, extra) => {
      return (await rawUpstream.request(
        { method: request.method, params: stripLinkScopedProgressToken(request.params) },
        ResultSchema,
        fwdOpts(extra),
      )) as never;
    };

    // 업스트림 → 다운스트림 알림 전달 (tools/list_changed 등).
    upstream.fallbackNotificationHandler = async (notification) => {
      try {
        await server.notification(notification as never);
      } catch (err) {
        log(
          `WARN failed to forward notification ${String(notification.method)} (${errText(err)})`,
        );
      }
    };

    // ── 생명주기 ──
    let resolveClosed!: () => void;
    const closed = new Promise<void>((resolve) => {
      resolveClosed = resolve;
    });
    let closeInitiated = false;

    server.onclose = () => {
      resolveClosed();
      void upstream.close().catch(() => {});
    };
    upstream.onclose = () => {
      // 업스트림 자식이 죽으면 다운스트림도 정리 (EOF 전달).
      void server.close().catch(() => {});
      resolveClosed();
    };
    upstream.onerror = (err) => {
      log(`upstream error: ${errText(err)}`);
    };

    const close = async (): Promise<void> => {
      if (closeInitiated) return;
      closeInitiated = true;
      try {
        await server.close();
      } catch {
        // already closed
      }
      try {
        await upstream.close(); // StdioClientTransport.close()가 자식을 종료한다.
      } catch {
        // already closed
      }
      resolveClosed();
    };

    // 4) 다운스트림 연결.
    const transport = deps.serverTransport ?? new StdioServerTransport();
    await server.connect(transport);

    return {
      server,
      upstream,
      client,
      setGoal(g: string): GoalState {
        goal = { goal: g, setAt: new Date().toISOString() };
        return goal;
      },
      getGoal: () => goal,
      closed,
      close,
    };
  } catch (err) {
    // 다운스트림 연결/와이어링 실패 — 업스트림 자식을 남겨두지 않는다.
    void upstream.close().catch(() => {});
    throw err;
  }
}

// ─── 헬퍼 ──────────────────────────────────────────────────────────────────

function textErrorResult(message: string): CallToolResult {
  return { content: [{ type: "text", text: message }], isError: true };
}

function blockedResult(name: string, blockTools: string[]): CallToolResult {
  return textErrorResult(
    `Tool "${name}" is blocked by jev-playwright-mcp policy (blockTools: ` +
      `${blockTools.join(", ")}).\n` +
      `This tool class is considered unsafe to auto-forward and is disabled by default. ` +
      `To override, remove "${name}" from the proxy's blockTools tool policy ` +
      `(CLI/env configuration of jev-playwright-mcp) and restart the session.`,
  );
}

// ─── 실행 엔트리 (stdio 모드) ──────────────────────────────────────────────

/**
 * stdio 프록시 실행 — SIGINT/SIGTERM에서 자식까지 정리 후 종료.
 * 세션 양측 중 하나가 닫히면 resolve된다.
 */
export async function runProxy(
  config: JevMcpConfig,
  deps: ProxyDeps = {},
): Promise<void> {
  const transport = deps.serverTransport ?? new StdioServerTransport();
  const session = await createProxySession(config, { ...deps, serverTransport: transport });

  let shuttingDown = false;
  const onSignal = (signal: NodeJS.Signals): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    log(`received ${signal} — shutting down`);
    void session
      .close()
      .catch(() => {})
      .finally(() => process.exit(0));
  };
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);

  try {
    await session.closed;
  } finally {
    process.removeListener("SIGINT", onSignal);
    process.removeListener("SIGTERM", onSignal);
    await session.close().catch(() => {});
  }
}
