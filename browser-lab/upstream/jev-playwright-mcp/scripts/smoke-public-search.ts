/**
 * scripts/smoke-public-search.ts — 실제 공개 웹사이트(네이버) 라이브 테스트.
 *  검색창에 "아아아" 입력 → 제출 → 결과 페이지 관측. 전 과정 프록시+실제 Jev.
 */
import process from "node:process";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { loadPackageEnv } from "../src/env.js";

const CLI = fileURLToPath(new URL("../cli.js", import.meta.url));
const SITE = "https://www.naver.com";
const QUERY = "아아아";
const TIMEOUT = 120_000;

function info(msg: string): void {
  process.stdout.write(`[public] ${msg}\n`);
}
function textOf(r: unknown): string {
  const c = (r as { content?: Array<{ type: string; text?: string }> }).content ?? [];
  return c.filter((b) => b.type === "text").map((b) => b.text ?? "").join("\n");
}

async function main(): Promise<void> {
  loadPackageEnv(process.env, process.cwd());
  if (!process.env["TYPESAFE_API_KEY"]) {
    info("TYPESAFE_API_KEY 없음 — 종료.");
    process.exitCode = 1;
    return;
  }

  const pruneKeep = process.env["PRUNE_KEEP"] ?? "0.35";
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [CLI, "--jev-mode=all", "--headless", `--jev-prune-keep-threshold=${pruneKeep}`],
    env: { ...process.env } as Record<string, string>,
    stderr: "inherit",
  });
  const client = new Client({ name: "public-search", version: "0.0.0" });
  await client.connect(transport);

  try {
    await client.callTool(
      { name: "browser_set_goal", arguments: { goal: `Search Naver for "${QUERY}"` } },
      undefined,
      { timeout: TIMEOUT },
    );

    info(`${SITE} 로 이동…`);
    const nav = await client.callTool(
      { name: "browser_navigate", arguments: { url: SITE } },
      undefined,
      { timeout: TIMEOUT },
    );
    const navText = textOf(nav);
    const navInsights = navText.match(/<jev-insights>[\s\S]*?<\/jev-insights>/);
    info(`── 네이버 진입 insights ──\n${navInsights ? navInsights[0] : "(없음)"}`);

    info("스냅샷 촬영 + 검색창 ref 탐색…");
    const snap = await client.callTool(
      { name: "browser_snapshot", arguments: {} },
      undefined,
      { timeout: TIMEOUT },
    );
    const snapText = textOf(snap);
    const { writeFileSync } = await import("node:fs");
    writeFileSync("/tmp/naver-snap.txt", snapText);
    const snapInsights = snapText.match(/<jev-insights>[\s\S]*?<\/jev-insights>/);
    info(`── 스냅샷 insights (원본 ${snapText.length}자) ──\n${snapInsights ? snapInsights[0] : "(없음)"}`);

    // 검색창 textbox ref 찾기 — '검색' 라벨/플레이스홀더 우선, 아니면 첫 textbox.
    const boxLine =
      snapText.split("\n").find((l) => /textbox|searchbox|combobox/.test(l) && l.includes("검색")) ??
      snapText.split("\n").find((l) => /textbox|searchbox|combobox/.test(l) && l.includes("Search")) ??
      snapText.split("\n").find((l) => /textbox|searchbox|combobox/.test(l));
    if (!boxLine) {
      info("검색창(textbox)을 못 찾음 — 스냅샷 앞부분:\n" + snapText.slice(0, 1500));
      process.exitCode = 1;
      return;
    }
    const ref = boxLine.match(/\[ref=(e\d+)\]/)?.[1];
    info(`검색창 발견: ${boxLine.trim()} → ref=${ref}`);

    info(`"${QUERY}" 입력+제출 (browser_type — 게이트 심사 대상)…`);
    const typed = await client.callTool(
      {
        name: "browser_type",
        arguments: { element: "Naver search box", target: ref, text: QUERY, submit: true },
      },
      undefined,
      { timeout: TIMEOUT },
    );
    const typedText = textOf(typed);
    const typeInsights = typedText.match(/<jev-insights>[\s\S]*?<\/jev-insights>/);
    info(`── 입력 결과 (isError=${Boolean(typed.isError)}) ──`);
    info(typedText.slice(0, 900));
    if (typeInsights) info(`── type 응답 insights ──\n${typeInsights[0]}`);

    // 결과 페이지 확인
    const result = await client.callTool(
      { name: "browser_snapshot", arguments: {} },
      undefined,
      { timeout: TIMEOUT },
    );
    const resultText = textOf(result);
    const resultInsights = resultText.match(/<jev-insights>[\s\S]*?<\/jev-insights>/);
    const queryEcho = resultText.includes(QUERY) || typedText.includes(QUERY);
    info(`── 검색 결과 페이지 insights ──\n${resultInsights ? resultInsights[0] : "(없음)"}`);
    info(`결과 스냅샷에 "${QUERY}" 존재: ${queryEcho}`);

    const status = await client.callTool(
      { name: "browser_jev_status", arguments: {} },
      undefined,
      { timeout: TIMEOUT },
    );
    const s = JSON.parse(textOf(status));
    info(`── jev_status: calls=${s.stats.calls} cost=$${s.stats.costUsd.toFixed(6)} failures=${s.stats.failures}`);

    const pass = queryEcho && !typed.isError;
    info(`결과: ${pass ? "PASS — 실제 사이트 검색창 입력 성공" : "PARTIAL/FAIL — 위 로그 확인"}`);
    process.exitCode = pass ? 0 : 1;
  } finally {
    await client.close().catch(() => {});
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`[public] 오류: ${String(err)}\n`);
  process.exitCode = 1;
});
