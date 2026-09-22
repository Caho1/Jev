/**
 * prune.test.ts — splitSnapshotRegions / scoreRelevance / collapseRegions 단위 테스트.
 * MOCK JevClient 만 사용 (네트워크 없음, 브라우저 없음, API 키 불필요).
 */
import { describe, expect, it } from "vitest";
import type {
  JevAnswer,
  JevClient,
  JevMcpConfig,
  JevRequest,
  JevResponse,
} from "../src/contracts.js";
import { DEFAULT_CONFIG } from "../src/contracts.js";
import {
  collapseRegions,
  scoreRelevance,
  splitSnapshotRegions,
  type SnapshotRegion,
} from "../src/jev/prune.js";

function makeConfig(over: Partial<JevMcpConfig> = {}): JevMcpConfig {
  return { ...DEFAULT_CONFIG, apiKey: "test-key", upstreamArgs: [], ...over };
}

/** 스크립트된 응답을 주는 mock — 네트워크 호출 없음. */
class MockJev implements JevClient {
  readonly enabled = true;
  readonly requests: JevRequest[] = [];

  constructor(private respond: (req: JevRequest) => JevResponse) {}

  async decide(req: JevRequest): Promise<JevResponse> {
    this.requests.push(req);
    return this.respond(req);
  }

  stats() {
    return { calls: this.requests.length, failures: 0, costUsd: 0, inputTokens: 0 };
  }
}

/** 지역 라벨 → noul 확률 응답기 (noul 배치 1회 호출 전제). */
function noulResponder(probByLabel: (label: string) => number) {
  return (req: JevRequest): JevResponse => {
    const state = req.state as {
      region_labels_and_excerpts: { id: string; label: string }[];
    };
    const answers: Record<string, JevAnswer> = {};
    for (const r of state.region_labels_and_excerpts) {
      answers[r.id] = { type: "noul", noul: probByLabel(r.label) };
    }
    return {
      model: "jev-latest",
      answers,
      usage: { input_tokens: 100, output_tokens: 0 },
    };
  };
}

// ─── fixture: 투자자 관계 페이지 스냅샷 (~56 lines, nav/alert/main/footer) ───
const SNAPSHOT = [
  '- banner "GlobalCorp" [ref=e1]:',
  '  - heading "GlobalCorp" [level=1] [ref=e2]',
  '  - navigation "Primary" [ref=e3]:',
  '    - link "Home" [ref=e4]',
  '    - link "Products" [ref=e5]',
  '    - link "Pricing" [ref=e6]',
  '    - link "Blog" [ref=e7]',
  '    - link "Contact sales" [ref=e8]',
  '    - link "Careers" [ref=e9]',
  '    - textbox "Search site" [ref=e10]',
  '    - button "Search" [ref=e11]',
  "- alert [ref=e12]:",
  "  - text: Cookie preferences updated.",
  "- main [ref=e20]:",
  '  - heading "Investor Relations" [level=1] [ref=e21]',
  "  - paragraph [ref=e22]: GlobalCorp (NASDAQ: GC) publishes quarterly and annual filings for shareholders and analysts.",
  '  - region "Annual reports" [ref=e23]:',
  '    - heading "Annual reports" [level=2] [ref=e24]',
  "    - list [ref=e25]:",
  "      - listitem [ref=e26]:",
  '        - link "2025 Annual Report (PDF)" [ref=e27]',
  "      - listitem [ref=e28]:",
  '        - link "2025 Annual Report (HTML)" [ref=e29]',
  "      - listitem [ref=e30]:",
  '        - link "2024 Annual Report (PDF)" [ref=e31]',
  "      - listitem [ref=e32]:",
  '        - link "2023 Annual Report (PDF)" [ref=e33]',
  "      - listitem [ref=e34]:",
  '        - link "2022 Annual Report (PDF)" [ref=e35]',
  '    - button "Download 2025 annual report" [ref=e36]',
  '  - region "Quarterly results" [ref=e37]:',
  '    - heading "Quarterly results" [level=2] [ref=e38]',
  "    - table [ref=e39]:",
  "      - row [ref=e40]:",
  '        - columnheader "Quarter" [ref=e41]',
  '        - columnheader "Revenue" [ref=e42]',
  "      - row [ref=e43]:",
  '        - cell "Q1 2025" [ref=e44]',
  '        - cell "$1.21B" [ref=e45]',
  "      - row [ref=e46]:",
  '        - cell "Q2 2025" [ref=e47]',
  '        - cell "$1.34B" [ref=e48]',
  "      - row [ref=e49]:",
  '        - cell "Q3 2025" [ref=e50]',
  '        - cell "$1.42B" [ref=e51]',
  '    - link "Download quarterly data (CSV)" [ref=e52]',
  '  - region "SEC filings" [ref=e53]:',
  '    - heading "SEC filings" [level=2] [ref=e54]',
  '    - link "10-K 2025" [ref=e55]',
  '    - link "10-Q Q2 2025" [ref=e56]',
  "- contentinfo [ref=e80]:",
  '  - heading "Site footer" [level=2] [ref=e81]',
  '  - link "Privacy policy" [ref=e82]',
  '  - link "Terms of use" [ref=e83]',
  '  - link "Sitemap" [ref=e84]',
  '  - link "Accessibility" [ref=e85]',
  "  - paragraph [ref=e86]: (c) 2025 GlobalCorp. All rights reserved.",
].join("\n") + "\n";

const GOAL = { goal: "download 2025 annual report", setAt: "2026-09-17T00:00:00.000Z" };

// 라벨 기반 스크립트: 본문/보고서 계열은 0.92, banner/alert/footer 는 0.04.
const RELEVANT = /investor|annual|report|quarterly|download|filing/i;
const probByLabel = (label: string) => (RELEVANT.test(label) ? 0.92 : 0.04);

describe("splitSnapshotRegions", () => {
  it("splits into top-level blocks with sequential ids and meaningful labels", () => {
    const regions = splitSnapshotRegions(SNAPSHOT);
    expect(regions.map((r) => r.id)).toEqual(["r1", "r2", "r3", "r4"]);
    expect(regions[0]!.label).toBe("GlobalCorp"); // 첫 heading
    expect(regions[1]!.label).toBe("alert"); // heading 없음 → role
    expect(regions[2]!.label).toBe("Investor Relations");
    expect(regions[3]!.label).toBe("Site footer");
    expect(regions[0]!.text.startsWith('- banner "GlobalCorp" [ref=e1]:')).toBe(true);
    expect(regions[2]!.text).toContain('        - link "2025 Annual Report (PDF)" [ref=e27]');
  });

  it("returns [] for empty input", () => {
    expect(splitSnapshotRegions("")).toEqual([]);
    expect(splitSnapshotRegions("   \n  \n")).toEqual([]);
  });
});

describe("scoreRelevance", () => {
  it("batches all regions into ONE Jev request and flags kept by threshold", async () => {
    const client = new MockJev(noulResponder(probByLabel));
    const regions = splitSnapshotRegions(SNAPSHOT);
    const { verdicts, droppedBeforeBatching } = await scoreRelevance(
      client,
      GOAL,
      regions,
      makeConfig(),
    );

    expect(droppedBeforeBatching).toBe(0);
    expect(client.requests.length).toBe(1);
    expect(verdicts).toHaveLength(4);

    const req = client.requests[0]!;
    expect(req.model).toBe("jev-latest");
    expect(Object.keys(req.questions)).toEqual(["r1", "r2", "r3", "r4"]);
    for (const [id, q] of Object.entries(req.questions)) {
      expect(q.type).toBe("noul");
      expect(q.instructions).toContain("download 2025 annual report");
      expect(q.instructions).toContain("untrusted");
      // 질문 key는 모델에게 전송되지 않는다 — 각 문항이 스스로 지역을 특정해야 한다.
      expect(q.instructions).toContain(`'${id}'`);
      expect(q.instructions).toContain("'region_labels_and_excerpts'");
    }
    // 각 문항이 자기 지역의 라벨까지 밝힌다 (r3 = main 블록).
    expect(req.questions["r3"]!.instructions).toContain("Investor Relations");
    const state = req.state as {
      goal: string;
      region_labels_and_excerpts: { id: string; label: string; excerpt: string }[];
    };
    expect(state.goal).toBe("download 2025 annual report");
    expect(state.region_labels_and_excerpts).toHaveLength(4);
    for (const rae of state.region_labels_and_excerpts) {
      expect(rae.excerpt.length).toBeGreaterThan(0);
      expect(rae.excerpt.length).toBeLessThanOrEqual(241);
    }

    const main = verdicts.find((v) => v.regionId === "r3")!;
    expect(main.probability).toBe(0.92);
    expect(main.kept).toBe(true);
    for (const id of ["r1", "r2", "r4"]) {
      const v = verdicts.find((x) => x.regionId === id)!;
      expect(v.probability).toBe(0.04);
      expect(v.kept).toBe(false);
    }
  });

  it("returns empty verdicts and makes ZERO Jev calls when no regions", async () => {
    const client = new MockJev(noulResponder(probByLabel));
    const res = await scoreRelevance(client, GOAL, [], makeConfig());
    expect(res.verdicts).toEqual([]);
    expect(res.droppedBeforeBatching).toBe(0);
    expect(client.requests.length).toBe(0);
  });

  it("pre-cuts beyond 40 regions by naive goal-keyword label matching (no silent caps)", async () => {
    const lines: string[] = [];
    for (let i = 0; i < 45; i++) {
      lines.push(`- generic "Widget section ${i}" [ref=e${i + 1}]:`);
    }
    for (let i = 0; i < 10; i++) {
      lines.push(`- region "Report archive ${i}" [ref=e${100 + i}]:`);
    }
    const yaml = `${lines.join("\n")}\n`;
    const regions = splitSnapshotRegions(yaml);
    expect(regions).toHaveLength(55);

    const client = new MockJev(noulResponder(probByLabel));
    const { verdicts, droppedBeforeBatching } = await scoreRelevance(
      client,
      GOAL,
      regions,
      makeConfig(),
    );

    expect(droppedBeforeBatching).toBe(45);
    expect(client.requests.length).toBe(1);
    const sentIds = Object.keys(client.requests[0]!.questions);
    expect(sentIds).toHaveLength(10);
    expect(sentIds.every((id) => Number(id.slice(1)) > 45)).toBe(true);

    expect(verdicts).toHaveLength(55);
    const widget = verdicts.find((v) => v.regionId === "r1")!;
    expect(widget.label).toBe("Widget section 0");
    expect(widget.probability).toBe(0);
    expect(widget.kept).toBe(false);
    const report = verdicts.find((v) => v.regionId === "r50")!;
    expect(report.probability).toBe(0.92); // "Report archive" → mock 0.92
    expect(report.kept).toBe(true);
  });

  it("caps the batch at 40 even when keyword matching yields more (matched branch)", async () => {
    // 라벨이 전부 goal 키워드에 걸려도 요청 1회의 상한은 유지되어야 한다.
    const lines: string[] = [];
    for (let i = 0; i < 50; i++) {
      lines.push(`- region "Report archive ${i}" [ref=e${i + 1}]:`);
    }
    const yaml = `${lines.join("\n")}\n`;
    const regions = splitSnapshotRegions(yaml);
    expect(regions).toHaveLength(50);

    const client = new MockJev(noulResponder(probByLabel));
    const { verdicts, droppedBeforeBatching } = await scoreRelevance(
      client,
      GOAL,
      regions,
      makeConfig(),
    );

    expect(client.requests.length).toBe(1);
    expect(Object.keys(client.requests[0]!.questions)).toHaveLength(40);
    expect(droppedBeforeBatching).toBe(10); // 초과분은 사전 컷 수로 노출 (no silent caps)
    expect(verdicts).toHaveLength(50);
  });

  it("keeps everything (fail-open) when the client is disabled", async () => {
    const client = new MockJev(noulResponder(probByLabel));
    (client as unknown as { enabled: boolean }).enabled = false;
    const regions = splitSnapshotRegions(SNAPSHOT);
    const { verdicts, droppedBeforeBatching } = await scoreRelevance(
      client,
      GOAL,
      regions,
      makeConfig(),
    );
    expect(client.requests.length).toBe(0);
    expect(droppedBeforeBatching).toBe(0);
    expect(verdicts.every((v) => v.kept && v.probability === 1)).toBe(true);
  });

  it("treats a missing/malformed Jev answer as p=1 (never drops content on judge failure)", async () => {
    const broken = new MockJev(() => ({
      model: "jev-latest",
      answers: {},
      usage: { input_tokens: 1, output_tokens: 0 },
    }));
    const regions = splitSnapshotRegions(SNAPSHOT);
    const { verdicts } = await scoreRelevance(broken, GOAL, regions, makeConfig());
    expect(verdicts.every((v) => v.probability === 1 && v.kept)).toBe(true);
  });
});

describe("collapseRegions", () => {
  it("keeps relevant blocks verbatim and collapses low regions to one-line markers", async () => {
    const client = new MockJev(noulResponder(probByLabel));
    const regions = splitSnapshotRegions(SNAPSHOT);
    const { verdicts } = await scoreRelevance(client, GOAL, regions, makeConfig());
    const collapsed = collapseRegions(SNAPSHOT, regions, verdicts, makeConfig().pruneKeepThreshold);

    // main(r3) 블록은 원문 그대로.
    const main = regions.find((r) => r.id === "r3")!;
    expect(collapsed).toContain(main.text);
    expect(collapsed).toContain('        - link "2025 Annual Report (PDF)" [ref=e27]'); // 들여쓰기 보존

    // banner/alert/footer 는 한 줄 마커로.
    expect(collapsed).toContain('# [jev-pruned r1 "GlobalCorp" p=0.04]');
    expect(collapsed).toContain('# [jev-pruned r2 "alert" p=0.04]');
    expect(collapsed).toContain('# [jev-pruned r4 "Site footer" p=0.04]');
    expect(collapsed).not.toContain('- link "Home" [ref=e4]'); // 접힌 블록 내용은 제거
    expect(collapsed).not.toContain('- link "Privacy policy" [ref=e82]');

    // 구조 보존: 전체 줄 수 = (banner 1) + (alert 1) + main 원문 줄 수 + (footer 1).
    const mainLineCount = main.text.split("\n").length;
    expect(collapsed.split("\n")).toHaveLength(mainLineCount + 3);
    const markerCount = collapsed.split("\n").filter((l) => l.includes("# [jev-pruned")).length;
    expect(markerCount).toBe(3);
  });

  it("returns the original yaml untouched when no verdicts exist", () => {
    const regions = splitSnapshotRegions(SNAPSHOT);
    const collapsed = collapseRegions(SNAPSHOT, regions, [], 0.35);
    expect(collapsed).toBe(SNAPSHOT);
  });

  it("respects the caller-provided threshold over verdict.kept", async () => {
    const client = new MockJev(noulResponder(probByLabel));
    const regions = splitSnapshotRegions(SNAPSHOT);
    const { verdicts } = await scoreRelevance(client, GOAL, regions, makeConfig());
    // threshold=0 → 아무도 접히지 않음 (kept 는 threshold 파라미터로 재계산).
    const none = collapseRegions(SNAPSHOT, regions, verdicts, 0);
    expect(none).toBe(SNAPSHOT);
    // threshold=1 → 0.92 도 미달 → 전부 마커.
    const all = collapseRegions(SNAPSHOT, regions, verdicts, 1);
    expect(all.split("\n").filter((l) => l.includes("# [jev-pruned"))).toHaveLength(4);
    expect(all).not.toContain('        - link "2025 Annual Report (PDF)" [ref=e27]');
  });

  it("collapses pre-cut regions too (p=0.00 markers)", async () => {
    const lines: string[] = [];
    for (let i = 0; i < 45; i++) lines.push(`- generic "Widget section ${i}" [ref=e${i + 1}]:`);
    for (let i = 0; i < 10; i++) lines.push(`- region "Report archive ${i}" [ref=e${100 + i}]:`);
    const yaml = `${lines.join("\n")}\n`;
    const regions: SnapshotRegion[] = splitSnapshotRegions(yaml);
    const client = new MockJev(noulResponder(probByLabel));
    const { verdicts } = await scoreRelevance(client, GOAL, regions, makeConfig());
    const collapsed = collapseRegions(yaml, regions, verdicts, makeConfig().pruneKeepThreshold);

    expect(collapsed).toContain('# [jev-pruned r1 "Widget section 0" p=0.00]');
    expect(collapsed).toContain('- region "Report archive 0" [ref=e100]:'); // 남은 지역 원문
    const collapsedLines = collapsed.split("\n");
    expect(collapsedLines.filter((l) => l.includes("# [jev-pruned"))).toHaveLength(45);
    expect(collapsedLines.filter((l) => l.startsWith("- region"))).toHaveLength(10);
    // 55줄 + trailing newline element.
    expect(collapsedLines).toHaveLength(56);
  });
});
