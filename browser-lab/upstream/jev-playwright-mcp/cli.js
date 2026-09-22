#!/usr/bin/env node
/**
 * cli.js — jev-playwright-mcp 실행 shim (package "bin" 엔트리).
 *
 *  1) ./dist/index.js가 빌드되어 있으면 같은 프로세스에서 로드.
 *  2) 아니면 tsx로 ./src/index.ts를 자식 프로세스 실행
 *     (`node --import tsx src/index.ts`). in-process module.register() 대신
 *     자식 실행을 쓰는 이유: runtime register는 SIGINT/SIGTERM 디스폴트
 *     처리를 복구시키는 Node/tsx 조합 문제가 있어 clean shutdown이 깨진다.
 *
 *  argv는 그대로 전달된다: node cli.js --jev-mode=annotate --headless ...
 *  stdout은 MCP 채널이므로 이 shim은 절대 stdout에 쓰지 않는다.
 */
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import os from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const distEntry = join(root, "dist", "index.js");
const srcEntry = join(root, "src", "index.ts");

function fail(message) {
  process.stderr.write(`[jev-playwright-mcp] ${message}\n`);
  process.exit(1);
}

if (existsSync(distEntry)) {
  process.env.JEV_MCP_RUN_MAIN = "1";
  await import(pathToFileURL(distEntry).href);
} else if (existsSync(srcEntry)) {
  if (!existsSync(join(root, "node_modules", "tsx"))) {
    fail(
      "dist/ is not built and tsx is not installed. " +
        "Run `npm run build`, or `npm install` to get the tsx dev dependency.",
    );
  }
  const child = spawn(
    process.execPath,
    ["--import", "tsx", srcEntry, ...process.argv.slice(2)],
    { stdio: "inherit", env: process.env, cwd: root },
  );
  // 신호 전달 — clean shutdown은 실제 프록시(자식)가 담당한다.
  const forward = (signal) => {
    try {
      child.kill(signal);
    } catch {
      // child already gone
    }
  };
  process.on("SIGINT", () => forward("SIGINT"));
  process.on("SIGTERM", () => forward("SIGTERM"));
  child.on("error", (err) => {
    fail(`failed to run tsx: ${err instanceof Error ? err.message : String(err)}`);
  });
  child.on("exit", (code, signal) => {
    // 자식이 신호로 죽었으면 관례 코드(128+signum)로, 정상 종료면 그 코드로.
    // Node 는 신호를 이름('SIGINT' 등)으로 전달하므로 번호로 매핑한다
    // (SIGINT→130, SIGTERM→143, ...). 모르는 신호는 1로 간주.
    const signum = signal ? (os.constants.signals[signal] ?? 1) : 0;
    process.exit(code ?? (signal ? 128 + signum : 0));
  });
} else {
  fail(`no entry point: expected ${distEntry} or ${srcEntry}`);
}
