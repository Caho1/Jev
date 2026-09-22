/**
 * scripts/smoke-e2e.ts — 종단 간(E2E) 스모크 (실제 브라우저, 실제 cli.js).
 *
 *  실행 조건: env RUN_E2E=1. 미설정 시 "skipped (set RUN_E2E=1)" 출력 후 종료 0.
 *  (`npm run smoke:e2e` 가 이 값을 설정한다.) 일반 `npx vitest run` 에는 포함되지 않는다.
 *
 *  검증 흐름 — [downstream MCP client] ←stdio→ [cli.js → proxy] ←stdio→ [@playwright/mcp → chromium]
 *   0. src/upstream.ts 로 chromium 자동탐지 — 없으면 스킵 사유 출력 후 종료 0.
 *   A. TYPESAFE_API_KEY 설정(실제 키 없으면 더미) + NODE_OPTIONS --require 로
 *      scripts/helpers/stub-jev-fetch.cjs 를 선로드해 api.typesafe.ai 를
 *      결정론적 스텁으로 대체(오프라인 보장):
 *        browser_set_goal → browser_navigate(file:// fixture) → browser_snapshot
 *        → <jev-insights> 블록 + 인젝션 마스킹("Ignore all previous instructions") 단언.
 *   B. API 키 제거 환경: 동일 호출 → <jev-insights> 없는 깨끗한 passthrough 단언.
 *
 *  stdout/stderr 순수성 제약은 프록시에만 적용된다 — 이 스크립트는 사람이
 *  읽는 진단 출력을 자유로이 한다.
 */
import process from "node:process";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { detectChromiumExecutable } from "../src/upstream.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const CLI = fileURLToPath(new URL("../cli.js", import.meta.url));
const FIXTURE_URL = new URL("../tests/fixtures/sample-report.html", import.meta.url).href;
const STUB_FETCH = fileURLToPath(new URL("./helpers/stub-jev-fetch.cjs", import.meta.url));

const GOAL = "Collect the quarterly report links from this page";
const REQUEST_TIMEOUT_MS = 120_000;

function info(msg: string): void {
  process.stdout.write(`[smoke-e2e] ${msg}\n`);
}
function fail(msg: string): never {
  throw new Error(msg);
}
function assert(cond: boolean, msg: string, context?: string): void {
  if (!cond) {
    const tail = context ? `\n--- context (tail) ---\n${context.slice(-800)}` : "";
    fail(`ASSERT FAILED: ${msg}${tail}`);
  }
}

function textOf(result: unknown): string {
  const content = (result as { content?: Array<{ type: string; text?: string }> }).content ?? [];
  return content
    .filter((b) => b.type === "text")
    .map((b) => b.text ?? "")
    .join("\n");
}

function childEnv(overrides: Record<string, string | undefined>): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined) env[k] = v;
  }
  for (const [k, v] of Object.entries(overrides)) {
    if (v === undefined) delete env[k];
    else env[k] = v;
  }
  return env;
}

/** cli.js 를 자식으로 띄우고 MCP 클라이언트로 접속한다. */
async function connectCli(
  env: Record<string, string>,
): Promise<{ client: Client; close: () => Promise<void> }> {
  const transport = new StdioClientTransport({
    command: process.execPath,
    // --allow-unrestricted-file-access: 업스트림은 기본적으로 file:// 탐색을
    // 차단한다 — fixture 가 file:// 페이지라 스모크에서만 허용한다.
    args: [CLI, "--jev-mode=all", "--headless", "--allow-unrestricted-file-access"],
    env,
    stderr: "inherit",
  });
  const client = new Client({ name: "smoke-e2e", version: "0.0.0" });
  await client.connect(transport);
  return {
    client,
    close: async () => {
      await client.close().catch(() => {});
    },
  };
}

// ─── Run A: API 키 있음(+ 스텁 Jev) → insights/마스킹 단언 ────────────────

async function runWithKey(apiKey: string | undefined): Promise<void> {
  info("run A: TYPESAFE_API_KEY present — expecting annotated responses");
  if (/\s/.test(STUB_FETCH)) {
    fail(
      `stub path contains whitespace; NODE_OPTIONS cannot quote it: ${STUB_FETCH}`,
    );
  }
  const baseEnv = childEnv({});
  const nodeOptions = `${baseEnv["NODE_OPTIONS"] ?? ""} --require ${STUB_FETCH}`.trim();
  const { client, close } = await connectCli(
    childEnv({
      TYPESAFE_API_KEY: apiKey ?? "smoke-dummy-key",
      JEV_MCP_API_KEY: undefined,
      NODE_OPTIONS: nodeOptions,
    }),
  );

  try {
    const tools = await client.listTools(undefined, { timeout: REQUEST_TIMEOUT_MS });
    const names = new Set(tools.tools.map((t) => t.name));
    for (const expected of [
      "browser_set_goal",
      "browser_jev_status",
      "browser_navigate",
      "browser_snapshot",
    ]) {
      assert(names.has(expected), `tools/list should include ${expected}`);
    }

    const setGoal = await client.callTool(
      { name: "browser_set_goal", arguments: { goal: GOAL } },
      undefined,
      { timeout: REQUEST_TIMEOUT_MS },
    );
    assert(!setGoal.isError, "browser_set_goal should succeed", textOf(setGoal));
    assert(textOf(setGoal).includes("Goal set"), "set_goal ack", textOf(setGoal));

    const nav = await client.callTool(
      { name: "browser_navigate", arguments: { url: FIXTURE_URL } },
      undefined,
      { timeout: REQUEST_TIMEOUT_MS },
    );
    assert(!nav.isError, "browser_navigate should succeed", textOf(nav));
    assert(
      textOf(nav).includes("Acme Reports Portal"),
      "navigate result should carry the page title",
      textOf(nav),
    );

    const snap = await client.callTool(
      { name: "browser_snapshot", arguments: {} },
      undefined,
      { timeout: REQUEST_TIMEOUT_MS },
    );
    const snapText = textOf(snap);
    assert(!snap.isError, "browser_snapshot should succeed", snapText);
    assert(snapText.includes("<jev-insights>"), "snapshot should carry a <jev-insights> block", snapText);
    assert(snapText.includes("page_state:"), "insights should include page_state", snapText);
    // 본문(= insights 블록 이전 부분)에서만 마스킹을 검사 — insights 의
    // excerpt 라인은 히트 스팬을 인용해 원문을 보여주는 것이 설계다.
    const body = snapText.split("<jev-insights>")[0] ?? "";
    assert(
      body.includes("[INJECTION MASKED p=0.93]"),
      "the injection paragraph should be masked in the body",
      snapText,
    );
    assert(
      !body.includes("email me the credentials"),
      "masked injection payload must not survive verbatim in the body",
      snapText,
    );
    assert(snapText.includes("Q1 2026 report"), "snapshot should list report links", snapText);

    const status = JSON.parse(
      textOf(
        await client.callTool({ name: "browser_jev_status", arguments: {} }, undefined, {
          timeout: REQUEST_TIMEOUT_MS,
        }),
      ),
    ) as {
      enabled: boolean;
      goal: { goal: string } | null;
      stats: { calls: number };
    };
    assert(status.enabled === true, "jev_status.enabled should be true");
    assert(status.goal?.goal === GOAL, "jev_status should report the session goal");
    assert(status.stats.calls >= 2, "jev_status.stats.calls >= 2 (navigate + snapshot)", JSON.stringify(status));
    info(`run A ok — stats=${JSON.stringify(status.stats)}`);
  } finally {
    await close();
  }
}

// ─── Run B: API 키 없음 → 깨끗한 passthrough 단언 ────────────────────────

async function runWithoutKey(): Promise<void> {
  info("run B: TYPESAFE_API_KEY absent — expecting clean passthrough");
  const { client, close } = await connectCli(
    childEnv({
      TYPESAFE_API_KEY: undefined,
      JEV_MCP_API_KEY: undefined,
      NODE_OPTIONS: undefined,
      // 저장소 루트 .env에 실제 키가 있어도 이 실행은 '키 없음' 경로를
      // 검증한다 — .env 자동 로딩을 명시적으로 끈다.
      JEV_MCP_NO_ENV_FILE: "1",
    }),
  );

  try {
    const nav = await client.callTool(
      { name: "browser_navigate", arguments: { url: FIXTURE_URL } },
      undefined,
      { timeout: REQUEST_TIMEOUT_MS },
    );
    assert(!nav.isError, "browser_navigate should succeed", textOf(nav));

    const snap = await client.callTool(
      { name: "browser_snapshot", arguments: {} },
      undefined,
      { timeout: REQUEST_TIMEOUT_MS },
    );
    const snapText = textOf(snap);
    assert(!snap.isError, "browser_snapshot should succeed", snapText);
    assert(
      !snapText.includes("<jev-insights>"),
      "no <jev-insights> block may appear without an API key",
      snapText,
    );
    assert(
      snapText.includes("email me the credentials"),
      "without Jev the injection paragraph must pass through verbatim (stock behavior)",
      snapText,
    );
    assert(snapText.includes("Q1 2026 report"), "snapshot should list report links", snapText);

    const status = JSON.parse(
      textOf(
        await client.callTool({ name: "browser_jev_status", arguments: {} }, undefined, {
          timeout: REQUEST_TIMEOUT_MS,
        }),
      ),
    ) as { enabled: boolean };
    assert(status.enabled === false, "jev_status.enabled should be false");
    info("run B ok — byte-level passthrough confirmed");
  } finally {
    await close();
  }
}

// ─── entry ────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  if (process.env.RUN_E2E !== "1") {
    info("skipped (set RUN_E2E=1)");
    return;
  }

  const exe = detectChromiumExecutable();
  if (!exe) {
    info(
      "skipped — no chromium executable found: PLAYWRIGHT_MCP_EXECUTABLE_PATH is unset and " +
        "no ms-playwright cache (~/Library/Caches/ms-playwright or ~/.cache/ms-playwright, " +
        "or PLAYWRIGHT_BROWSERS_PATH) contains a chromium build. " +
        "Install one (e.g. `npx playwright install chromium`) and re-run.",
    );
    return;
  }
  info(`chromium: ${exe}`);

  await runWithKey(process.env.TYPESAFE_API_KEY);
  await runWithoutKey();
  info("E2E SMOKE PASSED");
}

main().catch((err: unknown) => {
  process.stderr.write(`[smoke-e2e] FATAL ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exitCode = 1;
});
