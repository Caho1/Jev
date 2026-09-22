/**
 * upstream.ts — 업스트림(@playwright/mcp) 실행 파일 해결.
 *
 *  - resolveUpstreamCommand(): 버전 고정 dep(cli.js) 우선, npx 폴백.
 *  - detectChromiumExecutable(): PLAYWRIGHT_MCP_EXECUTABLE_PATH 미설정 시
 *    ms-playwright 캐시에서 chromium 실행 파일을 찾아 env 주입용 경로 반환.
 */
import { existsSync, readdirSync, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

/** @playwright/mcp 버전 고정값 — package.json dependency와 일치해야 한다. */
export const PINNED_PLAYWRIGHT_MCP_VERSION = "0.0.81";

/**
 * 이 프록시 전용 시크릿 — 업스트림 자식(@playwright/mcp + 그 Chromium 손자들)이
 * 필요로 하지 않는다. 스톡 env 보존(PLAYWRIGHT_*)은 하되 결제 크리덴셜만 제거해
 * 자식 환경으로 흘러들 어가는 노출 면적을 줄인다.
 */
export const JEV_ONLY_SECRET_ENV_KEYS: readonly string[] = [
  "TYPESAFE_API_KEY",
  "JEV_MCP_API_KEY",
];

/** env 사본에서 Jev 전용 시크릿을 제거한다 (원본은 불변). */
export function stripJevSecrets(env: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = { ...env };
  for (const key of JEV_ONLY_SECRET_ENV_KEYS) delete out[key];
  return out;
}

/** package root 기준 상대 경로로, 시작 디렉터리에서 위로 올라가며 첫 번째 존재 경로 반환. */
function findFileUpward(relPaths: string[], startDirs: (string | null)[]): string | null {
  for (const start of startDirs) {
    if (!start) continue;
    let dir = path.resolve(start);
    // eslint-disable-next-line no-constant-condition
    while (true) {
      for (const rel of relPaths) {
        const candidate = path.join(dir, rel);
        if (existsSync(candidate)) return candidate;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  return null;
}

/**
 * 업스트림 MCP 서버 실행 명령.
 *  - 우선: node <pkg>/node_modules/@playwright/mcp/cli.js (버전 고정 dep)
 *  - 폴백: npx -y @playwright/mcp@0.0.81 (dep 미설치 환경)
 */
export function resolveUpstreamCommand(): { command: string; args: string[] } {
  const here = path.dirname(fileURLToPath(import.meta.url)); // src/ or dist/
  const cli = findFileUpward(
    [path.join("node_modules", "@playwright", "mcp", "cli.js")],
    [here, process.cwd()],
  );
  if (cli) {
    return { command: process.execPath, args: [cli] };
  }
  return {
    command: "npx",
    args: ["-y", `@playwright/mcp@${PINNED_PLAYWRIGHT_MCP_VERSION}`],
  };
}

/** 일반 파일이며 실행 비트가 있으면 true. */
function isExecutableFile(p: string): boolean {
  try {
    const st = statSync(p);
    return st.isFile() && (st.mode & 0o111) !== 0;
  } catch {
    return false;
  }
}

/** "chromium-1187" → 1187. 파싱 실패 시 0. */
function versionOf(dirName: string): number {
  const m = /(\d+)\s*$/.exec(dirName);
  return m ? Number.parseInt(m[1] ?? "0", 10) : 0;
}

/** 버전 디렉터리명들을 최신 순으로 정렬. */
function byVersionDesc(a: string, b: string): number {
  return versionOf(b) - versionOf(a);
}

/** ms-playwright 캐시 루트 후보 (환경변수 우선, 그 다음 플랫폼 기본값). */
function playwrightCacheDirs(): string[] {
  const dirs: string[] = [];
  const fromEnv = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (fromEnv && fromEnv.trim() !== "") dirs.push(path.resolve(fromEnv.trim()));
  const home = os.homedir();
  if (process.platform === "darwin") {
    dirs.push(path.join(home, "Library", "Caches", "ms-playwright"));
  } else {
    // linux 및 기타 유닉스 계열 기본값.
    dirs.push(path.join(home, ".cache", "ms-playwright"));
  }
  return dirs;
}

/** 정식 chromium 빌드 후보 (최신 버전 우선, 그다음 headless shell). */
function collectChromiumCandidates(base: string): string[] {
  const out: string[] = [];
  let top;
  try {
    top = readdirSync(base, { withFileTypes: true });
  } catch {
    return out;
  }

  const chromiumDirs = top
    .filter((e) => e.isDirectory() && /^chromium-\d+/.test(e.name))
    .map((e) => e.name)
    .sort(byVersionDesc);
  const shellDirs = top
    .filter((e) => e.isDirectory() && /^chromium_headless_shell-\d+/.test(e.name))
    .map((e) => e.name)
    .sort(byVersionDesc);

  // 정식 chromium이 headless-only shell보다 우선 (headed 기본 동작 보존).
  for (const dirName of chromiumDirs) {
    const vdir = path.join(base, dirName);
    let mids;
    try {
      mids = readdirSync(vdir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const mid of mids
      .filter((m) => m.isDirectory())
      .map((m) => m.name)
      .sort((a, b) => b.localeCompare(a))) {
      if (process.platform === "darwin" && mid === "chrome-mac") {
        // <base>/chromium-<v>/chrome-mac/Google Chrome for Testing.app/Contents/MacOS/*
        const macBinDir = path.join(
          vdir,
          mid,
          "Google Chrome for Testing.app",
          "Contents",
          "MacOS",
        );
        try {
          for (const bin of readdirSync(macBinDir)) {
            const p = path.join(macBinDir, bin);
            if (isExecutableFile(p)) out.push(p);
          }
        } catch {
          // app 번들 없음 — skip
        }
      } else if (mid.startsWith("chrome-linux")) {
        const exe = path.join(vdir, mid, "chrome");
        if (isExecutableFile(exe)) out.push(exe);
      }
    }
  }

  for (const dirName of shellDirs) {
    const vdir = path.join(base, dirName);
    let mids;
    try {
      mids = readdirSync(vdir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const mid of mids
      .filter((m) => m.isDirectory() && m.name.startsWith("chrome-headless-shell"))
      .map((m) => m.name)
      .sort((a, b) => b.localeCompare(a))) {
      const exe = path.join(vdir, mid, "chrome-headless-shell");
      if (isExecutableFile(exe)) out.push(exe);
    }
  }

  return out;
}

/**
 * chromium 실행 파일 자동탐지.
 *  1) env PLAYWRIGHT_MCP_EXECUTABLE_PATH (설정되면 그대로 신뢰)
 *  2) ms-playwright 캐시 스캔 (darwin: ~/Library/Caches/ms-playwright,
 *     linux: ~/.cache/ms-playwright, env PLAYWRIGHT_BROWSERS_PATH 우선)
 * 없으면 null (미설정으로 방관 — 업스트림 기본 탐색에 맡김).
 */
export function detectChromiumExecutable(): string | null {
  const fromEnv = process.env.PLAYWRIGHT_MCP_EXECUTABLE_PATH;
  if (fromEnv && fromEnv.trim() !== "") return fromEnv.trim();

  for (const base of playwrightCacheDirs()) {
    const candidates = collectChromiumCandidates(base);
    if (candidates.length > 0) return candidates[0] ?? null;
  }
  return null;
}
