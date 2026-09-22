/**
 * index.ts — 진입점.
 *
 *  - main(): argv/env 파싱(config.js) → JevClient 생성(jev/client.js) →
 *    stderr 스타트업 배너 1회 → runProxy.
 *  - startProxy(config, deps?): 프로그램/테스트용 (InMemoryTransport 주입 가능).
 *
 *  통합자 노트: config.js / jev/client.js 는 정적 import — 모든 모듈이
 *  함께 컴파일·검증된다 (stdout 은 절대 오염하지 않는다).
 */
import { spawn } from "node:child_process";
import { readFileSync, realpathSync } from "node:fs";
import process from "node:process";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  createProxySession,
  PROXY_NAME,
  PROXY_VERSION,
  runProxy,
  type ProxyDeps,
  type ProxySession,
} from "./proxy.js";
import type { JevMcpConfig } from "./contracts.js";
import { parseConfig } from "./config.js";
import { createJevClient } from "./jev/client.js";
import { resolveUpstreamCommand, stripJevSecrets } from "./upstream.js";
import { loadPackageEnv } from "./env.js";

export { runProxy, createProxySession } from "./proxy.js";
export type { ProxyDeps, ProxySession } from "./proxy.js";
export { resolveUpstreamCommand, detectChromiumExecutable } from "./upstream.js";
export { applyEnvFileLines, findPackageRoot, loadPackageEnv } from "./env.js";

/** 이 모듈의 디렉터리(src/ 또는 dist/) — .env 탐색 기점. */
const MODULE_DIR = dirname(fileURLToPath(import.meta.url));

function log(msg: string): void {
  process.stderr.write(`[${PROXY_NAME}] ${msg}\n`);
}

/**
 * 프로그램 실행 — 테스트/라이브러리 import와 달리 실제 stdio 프록시를 띄운다.
 * deps.serverTransport 미지정 시 StdioServerTransport.
 */
export async function startProxy(
  config: JevMcpConfig,
  deps: ProxyDeps = {},
): Promise<ProxySession> {
  return createProxySession(config, deps);
}

function errText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * --help/-h/--version 는 업스트림 CLI 에게 그대로 위임한다(stdio inherit).
 * 이 플래그들은 업스트림이 즉시 종료하는 출력형 플래그라 MCP 세션으로
 * 돌리면 도움말이 stdio 채널로 새어나가 "Connection closed" FATAL 이 되므로
 * 프록시 기동 전에 여기서 가로챈다. jev 플래그는 parseConfig 가 이미 소비했고
 * 나머지(순서 보존)는 upstreamArgs 로 전달된다.
 */
const PASSTHROUGH_CLI_FLAGS = new Set(["--help", "-h", "--version"]);

async function runUpstreamCliOnce(upstreamArgs: string[]): Promise<void> {
  const { command, args } = resolveUpstreamCommand();
  // 결제 크리덴셜(TYPESAFE_API_KEY/JEV_MCP_API_KEY)은 이 프록시 전용 —
  // 도움말 출력용 1회성 자식에게도 굳이 물려주지 않는다.
  const childEnv: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) childEnv[key] = value;
  }
  const child = spawn(command, [...args, ...upstreamArgs], {
    stdio: "inherit",
    env: stripJevSecrets(childEnv),
  });
  const code = await new Promise<number | null>((resolvePromise) => {
    child.once("error", (err: NodeJS.ErrnoException) => {
      process.stderr.write(`[${PROXY_NAME}] failed to run upstream CLI: ${err.message}\n`);
      resolvePromise(1);
    });
    child.once("exit", (exitCode) => resolvePromise(exitCode));
  });
  process.exitCode = code ?? 1;
}

/**
 * CLI 진입: (선택) 패키지 루트 .env 적용 → 설정 파싱 → 클라이언트 생성 →
 * 배너 → runProxy.
 *  - .env는 프로세스 env를 절대 덮어쓰지 않는다(dotenv 관례).
 *  - JEV_MCP_NO_ENV_FILE=1이면 .env 로딩 자체를 건너뛴다 — '키 없음 =
 *    순수 passthrough' 스톡 동작을 강제할 때 사용.
 */
export async function main(
  argv: string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const envFile = loadPackageEnv(env, MODULE_DIR);
  if (envFile) log(`loaded env file ${envFile.file} (${envFile.applied} var(s) applied)`);

  const config = parseConfig(argv, env);

  // 도움말/버전 — 프록시 없이 업스트림 CLI 로 직접 응답하고 종료.
  if (argv.some((token) => PASSTHROUGH_CLI_FLAGS.has(token))) {
    await runUpstreamCliOnce(config.upstreamArgs);
    return;
  }

  const client = createJevClient(config); // 키 없으면 NullJevClient (pure passthrough)

  const upstream = resolveUpstreamCommand();
  // 스타트업 배너 — stderr 1회 (stdout은 MCP 채널).
  log(
    `starting ${PROXY_NAME} v${PROXY_VERSION} | mode=${config.mode} | ` +
      `jev=${client.enabled ? "enabled" : "disabled (no API key — pure passthrough)"} | ` +
      `budget=$${config.budgetUsd} | upstream=${upstream.command} ${upstream.args.join(" ")}`,
  );

  await runProxy(config, { client });
}

/** 이 모듈이 프로세스 엔트리로 실행됐는지 판정. */
function invokedAsEntry(): boolean {
  if (process.env.JEV_MCP_RUN_MAIN === "1") return true; // cli.js shim이 설정
  const argv1 = process.argv[1];
  if (!argv1 || !isAbsolute(argv1)) return false;
  try {
    const entry = realpathSync(resolve(argv1));
    return import.meta.url === pathToFileURL(entry).href;
  } catch {
    return false;
  }
}

if (invokedAsEntry()) {
  void main().catch((err: unknown) => {
    log(`FATAL ${errText(err)}`);
    process.exitCode = 1;
  });
}
