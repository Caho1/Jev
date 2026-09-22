/**
 * scripts/smoke-live.ts — 실제 Jev API로 전체 파이프라인을 돌리는 라이브 스모크.
 *
 *  실행: npx tsx scripts/smoke-live.ts   (TYPESAFE_API_KEY 필요 — .env 자동 로드)
 *  스텁 없음: 진짜 jev 판정 + 진짜 @playwright/mcp + 진짜 chromium.
 *
 *  흐름: browser_set_goal → browser_navigate(file:// fixture) → browser_snapshot
 *        → 실제 어노테이션/마스킹/프루닝 출력 → browser_jev_status(실제 비용).
 *  fixture 페이지에 인젝션 유사 문단이 있어 실제 마스킹이 관측돼야 한다.
 */
import process from "node:process";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { detectChromiumExecutable } from "../src/upstream.js";
import { loadPackageEnv } from "../src/env.js";

const CLI = fileURLToPath(new URL("../cli.js", import.meta.url));
const FIXTURE_URL = new URL("../tests/fixtures/sample-report.html", import.meta.url).href;
const GOAL = "Collect the quarterly report links from this page";
const TIMEOUT = 120_000;

function info(msg: string): void {
  process.stdout.write(`[smoke-live] ${msg}\n`);
}

async function main(): Promise<void> {
  // 이 스크립트 프로세스에서도 .env 를 로드(프록시 cli.js는 자체 로드함).
  loadPackageEnv(process.env, process.cwd());
  const env = { ...process.env } as Record<string, string>;
  const key = env["TYPESAFE_API_KEY"] ?? "";
  if (!key) {
    info("TYPESAFE_API_KEY 없음 — .env 를 확인하라. 종료.");
    process.exitCode = 1;
    return;
  }
  const chromium = detectChromiumExecutable();
  info(`chromium: ${chromium ?? "(auto — upstream 기본 탐지)"}`);

  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [CLI, "--jev-mode=all", "--headless", "--allow-unrestricted-file-access"],
    env,
    stderr: "inherit",
  });
  const client = new Client({ name: "smoke-live", version: "0.0.0" });
  await client.connect(transport);

  const textOf = (r: unknown): string => {
    const c = (r as { content?: Array<{ type: string; text?: string }> }).content ?? [];
    return c.filter((b) => b.type === "text").map((b) => b.text ?? "").join("\n");
  };

  try {
    const goal = await client.callTool(
      { name: "browser_set_goal", arguments: { goal: GOAL } },
      undefined,
      { timeout: TIMEOUT },
    );
    info(`set_goal → ${textOf(goal).split("\n")[0]}`);

    const nav = await client.callTool(
      { name: "browser_navigate", arguments: { url: FIXTURE_URL } },
      undefined,
      { timeout: TIMEOUT },
    );
    const navText = textOf(nav);
    info("── navigate 응답 (실제 Jev 판정 포함) ──");
    info(navText.slice(0, 1200));

    const snap = await client.callTool(
      { name: "browser_snapshot", arguments: {} },
      undefined,
      { timeout: TIMEOUT },
    );
    const snapText = textOf(snap);
    const insights = snapText.match(/<jev-insights>[\s\S]*?<\/jev-insights>/);
    info("── snapshot: jev-insights 블록 ──");
    info(insights ? insights[0] : "(없음!)");
    info(`── snapshot 길이: ${snapText.length}자, 마스킹 마커: ${(snapText.match(/INJECTION MASKED/g) ?? []).length}개, 프루닝 마커: ${(snapText.match(/jev-pruned/g) ?? []).length}개 ──`);

    const status = await client.callTool(
      { name: "browser_jev_status", arguments: {} },
      undefined,
      { timeout: TIMEOUT },
    );
    info("── jev_status (실제 비용) ──");
    info(textOf(status));

    const okInsights = Boolean(insights);
    process.stdout.write(
      `\n[smoke-live] 결과: ${okInsights ? "PASS — 라이브 Jev 판정 관측됨" : "FAIL — insights 없음"}\n`,
    );
    process.exitCode = okInsights ? 0 : 1;
  } finally {
    await client.close().catch(() => {});
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`[smoke-live] 오류: ${String(err)}\n`);
  process.exitCode = 1;
});
