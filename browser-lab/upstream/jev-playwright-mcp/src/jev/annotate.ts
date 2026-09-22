/**
 * jev/annotate.ts — 응답 후처리 통합기 (ResponseAnnotator).
 *
 * 흐름 (annotateTools 대상 + client.enabled + mode ∈ {annotate, all}):
 *  1. 빠른 경로: 비활성/비대상 도구/빈 텍스트 → 원문 그대로 (insightsBlock=null).
 *     단 활성 세션에서는 untrusted 텍스트의 <jev-insights> 마커를 무력화해 반환.
 *  2. 판정 캐시: sha256(tool + goal + text) 키로 {pageState, injection, relevance} 재사용.
 *  3. 판정: 트리아지(Choice 1문항) + 인젝션(스팬별 Noul)을 '하나의 요청'으로 묶어 전송.
 *  4. 마스킹 → (goal + pruneTools면) 프루닝 → renderInsightsBlock().
 *  - 예산 경고는 프록시가 단일 담당한다(입구 검사 + 최초 1회 note) — 여기서
 *    중복 검사/중복 경고를 하지 않는다.
 *  - Jev 장애/프루닝 장애는 절대 응답을 깨지 않는다 — passthrough로 강등.
 *  - prune 모듈(jev/prune.js)은 지연 로딩: 없으면 프루닝만 건너뛴다.
 *    테스트/특수 배포에서는 생성자 opts.prune으로 주입한다.
 */
import { createHash } from "node:crypto";
import { neutralizeInsightMarkers, renderInsightsBlock } from "../contracts.js";
import type {
  AnnotatedContent,
  GoalState,
  InjectionScan,
  JevClient,
  JevMcpConfig,
  PageVerdict,
  RegionRelevance,
} from "../contracts.js";
import {
  TRIAGE_KEY,
  injectionFromAnswers,
  maskInjectionSpans,
  pageVerdictFromAnswer,
  planDetection,
} from "./detectors.js";
import type { PageContext } from "./detectors.js";

/** 프루닝 지역 — interfaces.ts의 SnapshotRegion과 구조 동일 (하드 의존 회피). */
export interface PruneRegion {
  id: string;
  label: string;
  text: string;
}

/** jev/prune.ts 계약 (interfaces.ts 서명 그대로). 통합 시 실제 모듈 또는 stub 주입. */
export interface PruneFunctions {
  splitSnapshotRegions(snapshotYaml: string): PruneRegion[];
  scoreRelevance(
    client: JevClient,
    goal: GoalState,
    regions: PruneRegion[],
    config: JevMcpConfig,
  ): Promise<{ verdicts: RegionRelevance[]; droppedBeforeBatching: number }>;
  collapseRegions(
    snapshotYaml: string,
    regions: PruneRegion[],
    verdicts: RegionRelevance[],
    keepThreshold: number,
  ): string;
}

export interface ResponseAnnotatorOptions {
  /** 프루닝 함수 주입 (테스트/번들 빌드용). 없으면 ./prune.js를 지연 로딩한다. */
  prune?: PruneFunctions;
}

/** 판정 캐시 엔트리 — 마스킹 재적용을 위해 스팬 원문도 보관. */
interface VerdictEntry {
  pageState: PageVerdict | undefined;
  scan: InjectionScan;
  spans: Array<{ span: string; probability: number }>;
  relevance?: RegionRelevance[];
  /** 배치 한도 초과 사전 컷 수 (no silent caps — 재사용 시에도 보존). */
  droppedBeforeBatching?: number;
}

export class ResponseAnnotator {
  private readonly client: JevClient;
  private readonly config: JevMcpConfig;
  private readonly opts?: ResponseAnnotatorOptions;
  private readonly cache = new Map<string, VerdictEntry>();
  private resolvedPrune?: PruneFunctions | null;
  /** 디버그/상태용 오류 카운터 (응답에는 영향 없음). */
  readonly errors = { decide: 0, prune: 0 };
  /** 누적: 배치 한도 초과로 심사 없이 사전 컷된 지역 수 (browser_jev_status 공개용). */
  readonly pruneStats = { droppedBeforeBatching: 0 };

  constructor(client: JevClient, config: JevMcpConfig, opts?: ResponseAnnotatorOptions) {
    this.client = client;
    this.config = config;
    this.opts = opts;
  }

  async annotate(
    tool: string,
    args: unknown,
    text: string,
    goal: GoalState | null,
  ): Promise<AnnotatedContent> {
    const cfg = this.config;

    // 1. 빠른 경로 — 스톡 동작 보장 (Jev 비활성/mode off는 바이트 단위 원문).
    if (!this.client.enabled) return { text, insightsBlock: null };
    if (cfg.mode !== "annotate" && cfg.mode !== "all") return { text, insightsBlock: null };

    // 어노테이션이 활성인 세션에서는 untrusted 텍스트의 <jev-insights> 마커를
    // 무력화한다 — 페이지가 위조 블록으로 신뢰 채널을 도용하는 것을 차단.
    const safeText = neutralizeInsightMarkers(text);

    if (!cfg.toolPolicy.annotateTools.includes(tool)) return { text: safeText, insightsBlock: null };
    if (safeText.trim().length === 0) return { text: safeText, insightsBlock: null };

    // 2. 예산 가드는 프록시가 단일 담당한다(입구 1회 검사 + 최초 1회 경고).
    //    여기서 중복 검사하지 않는다 — 두 번째 emitter 가 다른 문구의 경고를
    //    이중으로 붙이는 것을 방지.

    // 3. 판정 캐시.
    const key = createHash("sha256")
      .update(`${tool}\0${goal?.goal ?? ""}\0${safeText}`)
      .digest("hex");
    let verdict: VerdictEntry | null = cfg.cacheVerdicts ? (this.cache.get(key) ?? null) : null;

    // 4. 판정 — 트리아지 + 인젝션을 한 요청으로.
    if (!verdict) {
      verdict = await this.detect(tool, args, safeText);
      if (!verdict) return { text: safeText, insightsBlock: null }; // Jev unavailable → passthrough
      if (cfg.cacheVerdicts) this.cache.set(key, verdict);
    }

    // 5. 마스킹 → 프루닝 → insights 블록.
    const masked = maskInjectionSpans(safeText, verdict.spans);
    const {
      text: finalText,
      relevance,
      droppedBeforeBatching,
    } = await this.applyPrune(tool, masked, goal, verdict.relevance, verdict.droppedBeforeBatching);
    verdict.relevance = relevance; // 캐시 엔트리에 반영 (동일 참조)
    verdict.droppedBeforeBatching = droppedBeforeBatching;

    return {
      text: finalText,
      insightsBlock: renderInsightsBlock({
        pageState: verdict.pageState,
        injection: verdict.scan,
        relevance,
        neverJudged: droppedBeforeBatching,
      }),
    };
  }

  /** 단일 decide(): page_state(Choice) + inj_N(Noul) 묶음 요청. */
  private async detect(tool: string, args: unknown, text: string): Promise<VerdictEntry | null> {
    const ctx = buildPageContext(tool, args, text);
    const plan = planDetection(ctx, { triage: true, injection: true });
    try {
      const res = await this.client.decide({
        state: plan.state,
        model: this.config.jevModel,
        questions: plan.questions,
      });
      const pageState = pageVerdictFromAnswer(res.answers[TRIAGE_KEY]);
      const { scan, spans } = injectionFromAnswers(
        plan.ids,
        plan.candidates,
        res.answers,
        this.config.injectionThreshold,
      );
      return { pageState, scan, spans };
    } catch {
      this.errors.decide += 1;
      return null;
    }
  }

  /** goal 설정 + pruneTools 대상일 때만. 캐시된 relevance면 collapse만 재실행. */
  private async applyPrune(
    tool: string,
    masked: string,
    goal: GoalState | null,
    cachedRelevance: RegionRelevance[] | undefined,
    cachedDropped: number | undefined,
  ): Promise<{
    text: string;
    relevance?: RegionRelevance[];
    droppedBeforeBatching: number;
  }> {
    if (goal === null || !this.config.toolPolicy.pruneTools.includes(tool)) {
      return { text: masked, relevance: undefined, droppedBeforeBatching: 0 };
    }
    const prune = await this.resolvePrune();
    if (!prune) return { text: masked, relevance: undefined, droppedBeforeBatching: 0 };
    try {
      const regions = prune.splitSnapshotRegions(masked);
      let relevance = cachedRelevance;
      let dropped = cachedDropped ?? 0;
      if (!relevance) {
        // no silent caps: 사전 컷 수를 받아 insights 블록과 상태에 반드시 노출한다.
        const scored = await prune.scoreRelevance(this.client, goal, regions, this.config);
        relevance = scored.verdicts;
        dropped = scored.droppedBeforeBatching;
        this.pruneStats.droppedBeforeBatching += dropped;
      }
      const collapsed = prune.collapseRegions(
        masked,
        regions,
        relevance,
        this.config.pruneKeepThreshold,
      );
      return { text: collapsed, relevance, droppedBeforeBatching: dropped };
    } catch {
      this.errors.prune += 1;
      return { text: masked, relevance: cachedRelevance, droppedBeforeBatching: 0 };
    }
  }

  /**
   * opts.prune 우선 → 없으면 실제 jev/prune.js 모듈을 literal dynamic import
   * (지연 로딩이되 tsc 가 해석·검증함 — 통합자 wiring). 로드 실패 시 1회만
   * null 로 캐시해 프루닝을 건너뛴다 (응답은 깨지지 않음).
   */
  private async resolvePrune(): Promise<PruneFunctions | null> {
    if (this.opts?.prune) return this.opts.prune;
    if (this.resolvedPrune !== undefined) return this.resolvedPrune;
    try {
      this.resolvedPrune = await import("./prune.js");
    } catch {
      this.resolvedPrune = null;
    }
    return this.resolvedPrune;
  }
}

/** 도구 인자/본문에서 url·title을 기회적으로 추출 (없어도 동작). */
function extractPageMeta(args: unknown, text: string): { url?: string; title?: string } {
  const meta: { url?: string; title?: string } = {};
  if (args && typeof args === "object") {
    const a = args as Record<string, unknown>;
    if (typeof a["url"] === "string" && a["url"].length > 0) meta.url = a["url"];
  }
  if (!meta.url) {
    const m = /(?:^|\n)\s*[-•*]?\s*Page URL:\s*(\S+)/.exec(text);
    if (m && m[1]) meta.url = m[1];
  }
  const t = /(?:^|\n)\s*[-•*]?\s*Page Title:\s*([^\n]+)/.exec(text);
  if (t && t[1] && t[1].trim().length > 0) meta.title = t[1].trim();
  return meta;
}

function buildPageContext(tool: string, args: unknown, text: string): PageContext {
  const { url, title } = extractPageMeta(args, text);
  return { tool, ...(url !== undefined ? { url } : {}), ...(title !== undefined ? { title } : {}), text };
}
