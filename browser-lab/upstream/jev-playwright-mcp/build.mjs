#!/usr/bin/env node
/**
 * build.mjs — dist/ 프로덕션 빌드 (tsc 방출, 추가 의존성 없음).
 *
 *  - tsconfig.build.json(module nodenext) 기준으로 src/**→ dist/** 컴파일.
 *    소스의 `.js` import 접미사가 그대로 유지되므로 dist/index.js 는
 *    순수 node 로 바로 실행 가능하고 cli.js 가 in-process 로 로드한다.
 *  - 실행: npm run build → node dist/index.js (또는 cli.js — dist 우선).
 */
import { spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const tscBin = path.join(root, "node_modules", "typescript", "bin", "tsc");

if (!existsSync(tscBin)) {
  console.error("[jev-playwright-mcp] typescript is not installed — run `npm install` first.");
  process.exit(1);
}

rmSync(path.join(root, "dist"), { recursive: true, force: true });

const result = spawnSync(process.execPath, [tscBin, "-p", path.join(root, "tsconfig.build.json")], {
  stdio: "inherit",
  cwd: root,
});
if (result.error) {
  console.error(`[jev-playwright-mcp] failed to run tsc: ${result.error.message}`);
  process.exit(1);
}
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
if (!existsSync(path.join(root, "dist", "index.js"))) {
  console.error("[jev-playwright-mcp] build finished but dist/index.js was not produced.");
  process.exit(1);
}
console.log("[jev-playwright-mcp] build ok → dist/index.js");
