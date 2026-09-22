/**
 * 모듈 간 함수 서명 계약 — 병렬 구현 에이전트들이 반드시 준수하는 export 목록.
 * 내부 구현은 자유지만, 이 서명은 통합 단계에서 그대로 wiring된다.
 *
 * 통합자 노트 (2026-09): 본 파일은 '계약 문서' — 아래 export declare 는 런타임
 * 구현이 아니라 서명 원천이다(런타임에 아무것도 export 하지 않는다).
 * 파일 하단의 contractChecks 가 실제 구현 모듈(jev/detectors, jev/annotate,
 * jev/prune, jev/gate) 과의 구조적 호환을 컴파일 타임에 강제한다 —
 * 어느 쪽이든 서명이 드리프트하면 `npm run typecheck` 가 실패한다.
 */
import type {
  AnnotatedContent,
  GateDecision,
  GoalState,
  InjectionScan,
  JevClient,
  JevMcpConfig,
  PageVerdict,
  RegionRelevance,
  RiskVerdict,
} from "./contracts.js";
import type * as Detectors from "./jev/detectors.js";
import type * as Annotate from "./jev/annotate.js";
import type * as Prune from "./jev/prune.js";
import type * as Gate from "./jev/gate.js";

// ─── jev/client.ts ────────────────────────────────────────────────────────

// export function createJevClient(config: JevMcpConfig): JevClient
//   - enabled = !!config.apiKey
//   - POST https://api.typesafe.ai/v1/systemone, Bearer apiKey
//   - 429/529: 지수 백오프 재시도 (기본 2회)
//   - 401/422/기타: JevUnavailableError
//   - stats()에 호출수/실패/비용(input_tokens 기준)/누적 inputTokens 반영

// ─── config.ts ────────────────────────────────────────────────────────────

// export function parseConfig(argv: string[], env: NodeJS.ProcessEnv): JevMcpConfig
//   - --jev-mode=off|annotate|gate|all            (env JEV_MCP_MODE)
//   - --jev-budget-usd=<n>       default 1.0      (env JEV_MCP_BUDGET_USD)
//   - --jev-model=<id>           default jev-latest (env JEV_MCP_MODEL)
//   - --jev-injection-threshold=<p>   default 0.5 (env JEV_MCP_INJECTION_THRESHOLD)
//   - --jev-prune-keep-threshold=<p>  default 0.35 (env JEV_MCP_PRUNE_KEEP_THRESHOLD)
//   - --jev-destructive-threshold=<p> default 0.6 (env JEV_MCP_DESTRUCTIVE_THRESHOLD)
//   - --jev-no-cache                                (env JEV_MCP_NO_CACHE=1)
//   -                                            requestTimeoutMs default 10000
//                                               (env JEV_MCP_TIMEOUT_MS, Jev HTTP 타임아웃)
//   - `--` 이후 인자 전부 + 알 수 없는 플래그 → upstreamArgs로 전달
//   - apiKey는 env TYPESAFE_API_KEY (우선) 또는 JEV_MCP_API_KEY
//   - 모든 숫자 검증: 빈 값/NaN/음수면 기본값으로 강등하고 stderr에 1행 경고

// ─── jev/detectors.ts ─────────────────────────────────────────────────────

/** 판정 대상 페이지 맥락 — 응답 text에서 추출한 것. untrusted 취급. */
export interface PageContext {
  tool: string;
  url?: string;
  title?: string;
  text: string;
}

/** 페이지 상태 트리아지 ( Choice 1문항 + 힌트는 코드 생성 ). 모델은 config.jevModel. */
export declare function triagePageState(
  client: JevClient,
  ctx: PageContext,
  config: JevMcpConfig,
): Promise<PageVerdict>;

/**
 * 인젝션 스캔: 2단계.
 *  1) 로컬 휴리스틱으로 지시 후보 스팬 추출 (무료, 결정론적)
 *  2) 각 스팬에 Noul 1문항 — 전부 하나의 Jev 요청으로 배치 (병렬 평가)
 * 휴리스틱 예: "ignore (all )?previous instructions", "system prompt",
 * "you are now", "AI/assistant/agent 를 2인칭으로 지시하는 문장",
 * "do not tell the user", role-play/override 요구 등. 스팬 없으면 Jev 미호출.
 */
export declare function scanInjection(
  client: JevClient,
  ctx: PageContext,
  config: JevMcpConfig,
): Promise<InjectionScan>;

// ─── jev/annotate.ts ──────────────────────────────────────────────────────

/**
 * 응답 후처리 통합기.
 *  - tool이 annotateTools에 없으면 원문 그대로 반환 (insightsBlock=null)
 *  - 캐시: (tool, sha256(text), goal?) 키로 판정 재사용 (cacheVerdicts && enabled)
 *  - 인젝션 히트는 본문에서 마스킹 [INJECTION MASKED p=0.87] 후 반환
 *  - goal이 설정되고 tool이 pruneTools에 있으면 프루닝 후 반환
 *  - insightsBlock은 renderInsightsBlock()으로 생성 (contracts.ts)
 */
export declare class ResponseAnnotator {
  constructor(client: JevClient, config: JevMcpConfig);
  annotate(
    tool: string,
    args: unknown,
    text: string,
    goal: GoalState | null,
  ): Promise<AnnotatedContent>;
}

// ─── jev/prune.ts ─────────────────────────────────────────────────────────

/** 스냅샷 yaml을 최상위 블록 단위 지역으로 분할. */
export interface SnapshotRegion {
  id: string;
  label: string;
  text: string;
}

export declare function splitSnapshotRegions(snapshotYaml: string): SnapshotRegion[];

/**
 * 지역별 관련성 — 각 지역을 Noul 1문항으로, 전부 하나의 Jev 요청에 배치.
 * 지역이 많으면(>40) 라벨 기반 1차 컷 후 배치 (no silent caps: 컷 수 반환).
 */
export declare function scoreRelevance(
  client: JevClient,
  goal: GoalState,
  regions: SnapshotRegion[],
  config: JevMcpConfig,
): Promise<{ verdicts: RegionRelevance[]; droppedBeforeBatching: number }>;

/** 낮은 지역을 한 줄 요약으로 접어 원문 형식을 유지한 채 재구성. */
export declare function collapseRegions(
  snapshotYaml: string,
  regions: SnapshotRegion[],
  verdicts: RegionRelevance[],
  keepThreshold: number,
): string;

// ─── jev/gate.ts ──────────────────────────────────────────────────────────

export interface ToolCallContext {
  tool: string;
  args: Record<string, unknown>;
  goal: GoalState | null;
  /** 직전 관측(있다면) — 없어도 판정 가능해야 한다. */
  recentPage?: PageContext;
}

/**
 * 위험 판정 Choice: safe / needs_confirmation / destructive.
 * 근거 입력: tool, args(라벨·텍스트·url), goal, 직전 페이지 상태.
 * 모델은 config.jevModel (DEFAULT_CONFIG 하드코딩 금지).
 */
export declare function assessRisk(
  client: JevClient,
  call: ToolCallContext,
  config: JevMcpConfig,
): Promise<RiskVerdict>;

/** 정책 적용: blockTools → 무조건 차단, destructive 확률 임계 초과 → 차단. */
export declare function gateDecision(
  verdict: RiskVerdict | null,
  tool: string,
  config: JevMcpConfig,
): GateDecision;

// ─── proxy.ts / index.ts / upstream.ts ────────────────────────────────────

// export async function runProxy(config: JevMcpConfig): Promise<void>
//   - 자식으로 @playwright/mcp cli.js 실행 (StdioClientTransport)
//   - upstream tools/list를 반영해 동일 이름/스키마로 재노출 +
//     자체 도구 2개 추가:
//       browser_set_goal   { goal: string }        → GoalState 저장
//       browser_jev_status {}                      → stats/모드/예산 소진율
//   - tools/call: blockTools/gate → 차단 시 isError + 사유 메시지
//     나머지는 upstream 그대로 호출, text 콘텐츠에 annotate 적용
//   - client.disabled이거나 mode 반영해 각 단계 조건부 실행
//   - 예산 초과: 이후 passthrough + 최초 1회 어노테이션 경고
//   - ping/기타 요청은 투명 포워딩, 로그는 stderr (stdout은 MCP 채널)

// export function resolveUpstreamCommand(): { command: string; args: string[] }
//   - 우선: <pkg>/node_modules/@playwright/mcp/cli.js (버전 고정)
//   - 폴백: npx -y @playwright/mcp@0.0.81
//   - PLAYWRIGHT_MCP_EXECUTABLE_PATH 미설정 시 ms-playwright 캐시에서
//     chromium 실행 파일 자동탐지하여 env 주입 (없으면 미설정으로 방관)

// ─── 컴파일 타임 계약 준수 검증 (통합자 추가) ──────────────────────────────
// 런타임 비용 없음(타입 전용 import). 주석(annotation) 형식이므로 실제 구현이
// 아래 declare 서명과 호환되지 않게 드리프트하면 `true` 를 `never` 타입에
// 할당할 수 없어 tsc 가 실패한다 (as 단언이면 never 로의 캐스트가 통과하므로
// 반드시 `: 타입 = true` 형태여야 한다).

type Implements<Actual, Contract> = Actual extends Contract ? true : never;

const detectorsContract: [
  Implements<typeof Detectors.triagePageState, typeof triagePageState>,
  Implements<typeof Detectors.scanInjection, typeof scanInjection>,
] = [true, true];

const annotatorContract: Implements<
  typeof Annotate.ResponseAnnotator,
  typeof ResponseAnnotator
> = true;

const pruneContract: [
  Implements<typeof Prune.splitSnapshotRegions, typeof splitSnapshotRegions>,
  Implements<typeof Prune.scoreRelevance, typeof scoreRelevance>,
  Implements<typeof Prune.collapseRegions, typeof collapseRegions>,
] = [true, true, true];

const gateContract: [
  Implements<typeof Gate.assessRisk, typeof assessRisk>,
  Implements<typeof Gate.gateDecision, typeof gateDecision>,
] = [true, true];

void [detectorsContract, annotatorContract, pruneContract, gateContract];
