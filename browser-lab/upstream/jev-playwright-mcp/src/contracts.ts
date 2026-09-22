/**
 * jev-playwright-mcp — 공유 계약 (모든 모듈이 import하는 단일 인터페이스 원천).
 *
 * 설계 원칙 (jev_browser_blueprint + 방향A "Jev를 인프라 안에"):
 *  - Jev는 MCP 서버 내부의 판정 계층이다. Jev를 코딩 에이전트에게 도구로 노출하지 않는다.
 *  - 페이지 콘텐츠는 untrusted다. 프록시가 에이전트 컨텍스트로 가기 전에 판정한다.
 *  - TYPESAFE_API_KEY가 없으면 모든 기능이 순수 passthrough로 강등된다 (스톡 동작 보장).
 *  - Confidence는 분포 통계일 뿐 정답 보증이 아니다 — 임계값은 설정값이다.
 */

// ─── Jev API 와이어 계약 ([S2] docs.typesafe.ai/api) ────────────────────

export type JevQuestion =
  | { type: "noul"; instructions: string; criteria?: { true: string; false: string } }
  | { type: "choice"; instructions: string; criteria: Record<string, string | null> }
  | { type: "score"; instructions: string; criteria: string[] };

export interface JevRequest {
  state: unknown;
  model: string;
  questions: Record<string, JevQuestion>;
}

export type JevAnswer =
  | { type: "noul"; noul: number }
  | {
      type: "choice";
      choice: string;
      probabilities: Record<string, number>;
      confidence: number;
    }
  | {
      type: "score";
      score: number;
      legend: Record<string, string>;
      probabilities: Record<string, number>;
      confidence: number;
    };

export interface JevUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface JevResponse {
  model: string;
  answers: Record<string, JevAnswer>;
  usage: JevUsage;
}

/** 호출 1회의 정규화된 기록 (저널/통계용). */
export interface JevCallRecord {
  ok: boolean;
  costUsd: number;
  latencyMs: number;
  inputTokens: number;
  model: string | null;
  error?: string;
}

/**
 * Jev 클라이언트 계약.
 *  - decide(): 단일 요청(질문 배치). state+questions를 그대로 POST /v1/systemone.
 *  - enabled=false면 NullJevClient — decide()는 호출 금지(호출 전에 반의적으로 검사).
 *  - 공식 typesafe-sdk-js DEFAULT_RETRY_POLICY 정렬: HTTP 408/429/5xx 및 네트워크/
 *    타임아웃 오류는 재시도(최대 2회 = 3 attempts). 'retry-after-ms'(ms, 우선) 또는
 *    'Retry-After'(초 또는 HTTP-date) 헤더를 존중하되 60s로 상한. 백오프는
 *    500ms 초기 · 5s 상한 · ±25% 지터.
 */
export interface JevClient {
  readonly enabled: boolean;
  decide(req: JevRequest): Promise<JevResponse>;
  /** 누적 호출 통계 (browser_jev_status 도구가 소비). */
  stats(): { calls: number; failures: number; costUsd: number; inputTokens: number };
}

export class JevUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JevUnavailableError";
  }
}

// ─── 판정 결과 타입 ───────────────────────────────────────────────────────

export const PAGE_STATES = [
  "expected",
  "login_wall",
  "captcha",
  "consent_popup",
  "error",
  "rate_limited",
  "empty_or_js_only",
  "paywall",
  "unknown",
] as const;
export type PageState = (typeof PAGE_STATES)[number];

export interface PageVerdict {
  state: PageState;
  /** Choice 확률 분포 (state별). */
  probabilities: Record<string, number>;
  confidence: number;
  /** 에이전트에게 주는 한 줄 행동 힌트. */
  hint: string;
}

/** 페이지 본문에서 추출된, 에이전트를 겨냥한 지시 후보 스팬. */
export interface InjectionHit {
  /**
   * 트리거 구절의 짧은 발췌 (≤40자, 태그 문자 제거) — 마스킹된 페이로드를
   * 그대로 에이전트 컨텍스트로 되돌려 보내지 않기 위해 크게 축약한다.
   */
  excerpt: string;
  probability: number;
  /** 마스킹된 원문 스팬의 길이 (감사용 — 본문 자체는 노출하지 않는다). */
  length: number;
}

export interface InjectionScan {
  present: boolean;
  hits: InjectionHit[];
  /** 1단계 휴리스틱이 심사에 올린 후보 스팬 수 (스캔 커버리지 공개용). */
  candidatesExamined: number;
}

export interface RegionRelevance {
  regionId: string;
  label: string;
  /** P(relevant | goal). */
  probability: number;
  kept: boolean;
}

export type RiskLevel = "safe" | "needs_confirmation" | "destructive";

export interface RiskVerdict {
  level: RiskLevel;
  probability: number;
  confidence: number;
  reason: string;
}

// ─── 인터셉션 흐름 계약 ──────────────────────────────────────────────────

/** 프록시가 가로채는 도구 분류. 모두 설정으로 덮어쓸 수 있다. */
export interface ToolPolicy {
  /** 응답 후처리: 페이지 상태 트리아지 + 인젝션 스캔 + 어노테이션. */
  annotateTools: string[];
  /** 목적(goal) 설정 시 응답 프루닝 대상. */
  pruneTools: string[];
  /** 호출 전 위험 게이트 대상. */
  gateTools: string[];
  /** 호출 자체를 차단 (기본: run_code_unsafe). */
  blockTools: string[];
}

export const DEFAULT_TOOL_POLICY: ToolPolicy = {
  // 페이지 발 텍스트를 돌려주는 도구는 모두 인젝션 스캔 대상이어야 한다.
  // browser_evaluate / browser_console_messages / browser_network_request 는
  // 페이지가 제어한 문자열(스토리지·콘솔·응답 본문)을 그대로 반환하고,
  // browser_click / browser_hover 의 ack 은 스냅샷 YAML을 포함한다 —
  // 스냅샷 경로만 스캔하면 같은 페이로드가 더 쉬운 경로로 새어든다.
  annotateTools: [
    "browser_snapshot",
    "browser_navigate",
    "browser_navigate_back",
    "browser_wait_for",
    "browser_find",
    "browser_network_requests",
    "browser_network_request",
    "browser_evaluate",
    "browser_console_messages",
    "browser_click",
    "browser_hover",
  ],
  pruneTools: ["browser_snapshot"],
  gateTools: [
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
    "browser_file_upload",
    "browser_handle_dialog",
    "browser_run_code_unsafe",
  ],
  blockTools: ["browser_run_code_unsafe"],
};

/** 게이트 결과: 차단 시 구조화된 오류로 반환. */
export interface GateDecision {
  allow: boolean;
  verdict: RiskVerdict | null;
  blockedMessage: string | null;
}

/** 응답 후처리 결과: 어노테이션 블록 + (있다면) 수정된 본문. */
export interface AnnotatedContent {
  /** 인젝션 히트가 제거/마스킹된 본문 (없으면 원문 그대로). */
  text: string;
  /** 어노테이션 블록 (이미지 등 비텍스트 항목은 그대로 둔다). */
  insightsBlock: string | null;
}

// ─── 설정 계약 ────────────────────────────────────────────────────────────

export type JevMode = "off" | "annotate" | "gate" | "all";

export interface JevMcpConfig {
  mode: JevMode;
  /** 이 세션의 Jev 호출 예산(USD). 초과 시 passthrough + 경고 1회. */
  budgetUsd: number;
  jevModel: string;
  /** 인젝션 판정 임계값 (P 이상이면 히트). */
  injectionThreshold: number;
  /** 프루닝: 이 확률 미만 지역은 접는다. */
  pruneKeepThreshold: number;
  /** 게이트: destructive 판정 확률 임계값. */
  destructiveThreshold: number;
  /** 동일 콘텐츠 해시 캐시 (true 권장). */
  cacheVerdicts: boolean;
  /** Jev HTTP 요청 타임아웃 (ms). 기본 10s, env JEV_MCP_TIMEOUT_MS로 덮어쓰기. */
  requestTimeoutMs: number;
  toolPolicy: ToolPolicy;
  /** 업스트림에게 줄 추가 인자 (-- 이후 또는 미확인 플래그 폴백). */
  upstreamArgs: string[];
  /** API 키 (env TYPESAFE_API_KEY). 없으면 모든 기능 off. */
  apiKey: string | null;
}

export const DEFAULT_CONFIG: Omit<JevMcpConfig, "upstreamArgs" | "apiKey"> = {
  mode: "all",
  budgetUsd: 1.0,
  jevModel: "jev-latest",
  injectionThreshold: 0.5,
  pruneKeepThreshold: 0.35,
  destructiveThreshold: 0.6,
  cacheVerdicts: true,
  requestTimeoutMs: 10_000,
  toolPolicy: DEFAULT_TOOL_POLICY,
};

/** 공식 요금: input $0.042/MTok, output 무료 ([S2] 참고 — 2026-09 기준). */
export const JEV_PRICE_PER_MTOK_INPUT = 0.042;

export function costOfUsage(usage: JevUsage): number {
  return (usage.input_tokens / 1_000_000) * JEV_PRICE_PER_MTOK_INPUT;
}

// ─── 어노테이션 출력 형식 (단일 원천) ─────────────────────────────────────

export const INSIGHTS_OPEN = "<jev-insights>";
export const INSIGHTS_CLOSE = "</jev-insights>";

export interface InsightLines {
  pageState?: PageVerdict;
  injection?: InjectionScan;
  relevance?: RegionRelevance[];
  /** 배치 한도 초과로 심사 없이 사전 컷된 지역 수 (no silent caps). */
  neverJudged?: number;
  risk?: RiskVerdict;
  budgetNote?: string;
}

/** risk 한 줄 렌더링 — renderInsightsBlock과 게이트 경고 병합이 같은 형식을 쓴다. */
export function renderRiskLine(risk: RiskVerdict): string {
  return `risk: ${risk.level} (p=${risk.probability.toFixed(2)}) — ${risk.reason}`;
}

/** 어노테이션 블록 렌더링 — 항상 짧게. 에이전트 컨텍스트 토큰을 아껴야 한다. */
export function renderInsightsBlock(lines: InsightLines): string {
  const out: string[] = [INSIGHTS_OPEN];
  if (lines.pageState) {
    out.push(
      `page_state: ${lines.pageState.state} (confidence=${lines.pageState.confidence.toFixed(2)})`,
    );
    out.push(`hint: ${lines.pageState.hint}`);
  }
  if (lines.injection) {
    // 커버리지 공개: 휴리스틱 게이트 스캔이었다는 사실 자체를 항상 노출한다
    // (스캔이 '깨끗함'으로 오독되지 않도록).
    out.push(
      `injection: ${lines.injection.hits.length} flagged span(s) masked ` +
        `(heuristic-gated scan of ${lines.injection.candidatesExamined} candidate span(s))`,
    );
    for (const h of lines.injection.hits.slice(0, 3)) {
      // 마스킹된 스팬 본문을 되돌려 주지 않는다 — 짧은 트리거 발췌 + 길이만.
      out.push(`  - [p=${h.probability.toFixed(2)}] ${h.excerpt} (${h.length} chars masked)`);
    }
  }
  if (lines.relevance && lines.relevance.length > 0) {
    const dropped = lines.relevance.filter((r) => !r.kept).length;
    const neverJudged = lines.neverJudged ?? 0;
    out.push(
      `pruned: ${dropped}/${lines.relevance.length} regions collapsed (goal-based)` +
        (neverJudged > 0 ? `, ${neverJudged} never judged (over batch cap)` : ""),
    );
  }
  if (lines.risk) {
    out.push(renderRiskLine(lines.risk));
  }
  if (lines.budgetNote) out.push(lines.budgetNote);
  out.push(INSIGHTS_CLOSE);
  return out.join("\n");
}

/**
 * untrusted 응답 텍스트에서 <jev-insights> 마커 문자열을 무력화한다.
 * 페이지가 진짜 블록과 구분 불가한 위조 블록을 심어 어노테이션 신뢰 채널을
 * 도용하는 것을 막는다 (대소문자·공백 관용, 멱등).
 */
export function neutralizeInsightMarkers(text: string): string {
  return text.replace(
    /<\s*(\/?)\s*jev-insights\s*>/gi,
    (_match, slash: string) => `<${slash}jev-insights[page-data]>`,
  );
}

// ─── 목적(goal) 상태 ──────────────────────────────────────────────────────

/** browser_set_goal 도구가 저장하는 세션 목적. 프루닝의 기준. */
export interface GoalState {
  goal: string;
  setAt: string;
}
