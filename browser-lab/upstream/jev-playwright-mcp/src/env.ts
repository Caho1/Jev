/**
 * env.ts — 경량 .env 로더 (추가 의존성 없음, dotenv 호환 부분집합).
 *
 *  - applyEnvFileLines(): KEY=VALUE 줄 배열을 env에 반영. 이미 설정된(비어있지
 *    않은) 변수는 절대 덮어쓰지 않는다 — 실제 프로세스 env가 항상 우선.
 *  - loadPackageEnv(): startDir에서 위로 올라 package.json이 있는 디렉터리
 *    (패키지 루트)를 찾아 그 아래 .env를 적용. JEV_MCP_NO_ENV_FILE가
 *    truthy(1/true/yes/on)면 완전 건너뛴다(스톡 동작 보장용 탈출구).
 *  - stdout에는 절대 쓰지 않는다(프록시의 stdout은 MCP 채널).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const TRUTHY = new Set(["1", "true", "yes", "on"]);

const KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

function isTruthy(raw: string | undefined): boolean {
  return raw !== undefined && TRUTHY.has(raw.trim().toLowerCase());
}

/** 큰따옴표 값의 이스케이프(\\", \\\\, \n) 풀기. */
function unescapeDoubleQuoted(body: string): string {
  return body.replace(/\\["\\n]/g, (seq) => {
    if (seq === '\\"') return '"';
    if (seq === "\\\\") return "\\";
    return "\n";
  });
}

/** 따옴표로 감싸져 있으면 벗긴 값, 아니면 인라인 주석 제거한 값. */
function unwrapValue(raw: string): string {
  const first = raw[0] ?? "";
  if (first === '"' || first === "'") {
    // 닫는 따옴표를 찾아 그 안쪽을 값으로 쓰고, 닫는 따옴표 뒤는 전부 주석으로
    // 취급한다 — `KEY="value" # comment` 가 따옴표를 값에 남기지 않도록.
    // (큰따옴표는 \", \\ 이스케이프를 건너뛴다.)
    let i = 1;
    while (i < raw.length) {
      if (first === '"' && raw[i] === "\\" && i + 1 < raw.length) {
        i += 2;
        continue;
      }
      if (raw[i] === first) break;
      i += 1;
    }
    if (i < raw.length && raw[i] === first) {
      const body = raw.slice(1, i);
      return first === '"' ? unescapeDoubleQuoted(body) : body;
    }
    // 닫는 따옴표가 없으면 아래 공통 경로(인라인 주석 제거)로 폴백.
  }
  // 따옴표 밖의 ` #` 이후는 인라인 주석으로 본다(dotenv 관례).
  const commentAt = raw.indexOf(" #");
  return (commentAt === -1 ? raw : raw.slice(0, commentAt)).trimEnd();
}

/**
 * .env 형식 줄들을 env에 적용하고, 반영된 키 개수를 반환한다.
 *  - `#` 주석/빈 줄 무시, 선택적 `export ` prefix 허용.
 *  - 값이 따옴표로 감싸져 있으면 벗기고, 아니면 끝의 ` #` 인라인 주석 제거.
 *  - 이미 값이 있는(env[key]가 undefined도 아니고 ""도 아님) 키는 건드리지
 *    않는다 — parseConfig가 빈 문자열을 '미설정'으로 취급하므로 빈 값은
 *    .env 값으로 채울 수 있다.
 */
export function applyEnvFileLines(lines: readonly string[], env: NodeJS.ProcessEnv): number {
  let applied = 0;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;

    let body = line;
    if (body.startsWith("export ") || body.startsWith("export\t")) {
      body = body.slice("export".length).trimStart();
    }

    const eq = body.indexOf("=");
    if (eq <= 0) continue; // '=' 없음 또는 빈 키 — 조용히 건너뜀
    const key = body.slice(0, eq).trim();
    if (!KEY_PATTERN.test(key)) continue;

    const existing = env[key];
    if (existing !== undefined && existing !== "") continue; // 프로세스 env 우선

    env[key] = unwrapValue(body.slice(eq + 1).trim());
    applied += 1;
  }
  return applied;
}

/** startDir에서 위로 올라 package.json이 있는 첫 디렉터리(패키지 루트) 반환. */
export function findPackageRoot(startDir: string): string {
  let dir = path.resolve(startDir);
  for (;;) {
    if (existsSync(path.join(dir, "package.json"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) return path.resolve(startDir, ".."); // 루트 도달 — 추정치
    dir = parent;
  }
}

/**
 * 패키지 루트의 .env를 env에 적용한다.
 *  - 파일이 없거나 읽을 수 없으면 null (조용히).
 *  - JEV_MCP_NO_ENV_FILE가 truthy면 아무 것도 하지 않고 null.
 *  - 반영 결과 { file, applied } 반환 — 호출자는 stderr 로그 등에 쓸 수 있다.
 */
export function loadPackageEnv(
  env: NodeJS.ProcessEnv,
  startDir: string,
): { file: string; applied: number } | null {
  if (isTruthy(env["JEV_MCP_NO_ENV_FILE"])) return null;
  const file = path.join(findPackageRoot(startDir), ".env");
  if (!existsSync(file)) return null;
  let lines: string[];
  try {
    lines = readFileSync(file, "utf8").split(/\r?\n/);
  } catch {
    return null;
  }
  return { file, applied: applyEnvFileLines(lines, env) };
}
