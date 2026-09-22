/**
 * CLI/env → JevMcpConfig 파서 (interfaces.ts 주석 계약 구현).
 *  - 인식된 --jev-* 플래그만 소비하고, "--" 이후 인자 + 미확인 플래그는
 *    순서를 보존해 upstreamArgs로 전달한다.
 *  - 무효 숫자(빈 값/NaN/음수/무한대)·무효 모드는 기본값 강등 + stderr 1행 경고.
 *  - 우선순위: CLI 플래그 > env > 기본값. 무효 플래그 값은 env 무시 기본값.
 */
import {
  DEFAULT_CONFIG,
  DEFAULT_TOOL_POLICY,
  type JevMcpConfig,
  type JevMode,
  type ToolPolicy,
} from "./contracts.js";

const VALID_MODES = new Set<string>(["off", "annotate", "gate", "all"]);

/** 값을 갖는 jev 플래그 (boolean 플래그 --jev-no-cache 제외). */
const VALUE_FLAGS = new Set<string>([
  "--jev-mode",
  "--jev-budget-usd",
  "--jev-model",
  "--jev-injection-threshold",
  "--jev-prune-keep-threshold",
  "--jev-destructive-threshold",
]);

const TRUTHY = new Set(["1", "true", "yes", "on"]);
const FALSY = new Set(["0", "false", "no", "off"]);

function warn(message: string): void {
  process.stderr.write(`[jev-mcp] warning: ${message}\n`);
}

/** 숫자 파싱 — 빈 문자열/NaN/음수/무한대면 기본값 강등 + 경고 1행. */
function parseNumber(raw: string, def: number, source: string): number {
  if (raw.trim() === "") {
    warn(`empty number value for ${source}; falling back to default ${def}`);
    return def;
  }
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) {
    warn(`invalid number '${raw}' for ${source}; falling back to default ${def}`);
    return def;
  }
  return n;
}

function parseMode(raw: string, def: JevMode, source: string): JevMode {
  if (VALID_MODES.has(raw)) return raw as JevMode;
  warn(
    `invalid mode '${raw}' for ${source}; expected off|annotate|gate|all, falling back to default '${def}'`,
  );
  return def;
}

/** `--flag=value` 분리. 값 플래그 토큰이 아니면 null. */
function splitFlag(token: string): { name: string; value: string | null } | null {
  if (!token.startsWith("--")) return null;
  const eq = token.indexOf("=");
  if (eq === -1) return { name: token, value: null };
  return { name: token.slice(0, eq), value: token.slice(eq + 1) };
}

/** 다음 토큰을 플래그 값으로 소비해도 되는가 (플래그처럼 보이면 거부, 숫자는 예외). */
function looksLikeValue(token: string): boolean {
  if (!token.startsWith("-")) return true;
  return Number.isFinite(Number(token));
}

/** 비어있지 않은 첫 값 (좌우 공백 trim). */
function firstNonEmpty(...values: Array<string | undefined>): string | null {
  for (const v of values) {
    if (v !== undefined && v.trim() !== "") return v.trim();
  }
  return null;
}

function resolveNumber(
  flagRaw: string | undefined,
  envRaw: string | undefined,
  def: number,
  flagName: string,
  envName: string,
): number {
  if (flagRaw !== undefined) return parseNumber(flagRaw, def, `flag ${flagName}`);
  if (envRaw !== undefined) return parseNumber(envRaw, def, `env ${envName}`);
  return def;
}

function resolveMode(flagRaw: string | undefined, envRaw: string | undefined): JevMode {
  const def = DEFAULT_CONFIG.mode;
  if (flagRaw !== undefined) return parseMode(flagRaw, def, "flag --jev-mode");
  if (envRaw !== undefined) return parseMode(envRaw, def, "env JEV_MCP_MODE");
  return def;
}

function resolveModel(flagRaw: string | undefined, envRaw: string | undefined): string {
  const def = DEFAULT_CONFIG.jevModel;
  if (flagRaw !== undefined) {
    if (flagRaw.trim() === "") {
      warn(`empty value for flag --jev-model; falling back to default '${def}'`);
      return def;
    }
    return flagRaw;
  }
  if (envRaw !== undefined) return envRaw;
  return def;
}

export function parseConfig(argv: string[], env: NodeJS.ProcessEnv): JevMcpConfig {
  // 1) 토큰 스캔 — 인식된 플래그만 소비, 나머지는 순서 보존.
  const raw: Record<string, string> = {};
  let noCacheFlag: boolean | null = null;
  const upstreamArgs: string[] = [];
  let passthrough = false;

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === undefined) continue;

    if (passthrough) {
      upstreamArgs.push(token);
      continue;
    }
    if (token === "--") {
      passthrough = true;
      continue;
    }

    const flag = splitFlag(token);
    if (flag !== null && VALUE_FLAGS.has(flag.name)) {
      let value = flag.value;
      if (value === null) {
        const next = argv[i + 1];
        if (next !== undefined && looksLikeValue(next)) {
          value = next;
          i += 1;
        }
      }
      if (value === null) {
        warn(`${flag.name} requires a value; flag ignored`);
      } else {
        raw[flag.name] = value;
      }
      continue;
    }
    if (flag !== null && flag.name === "--jev-no-cache") {
      if (flag.value === null) {
        noCacheFlag = true;
      } else {
        const v = flag.value.toLowerCase();
        noCacheFlag = FALSY.has(v) ? false : true; // 명시적 부정값 외엔 활성으로 본다
      }
      continue;
    }
    upstreamArgs.push(token);
  }

  // 2) env 원문 (빈 문자열은 미설정 취급).
  const envOf = (name: string): string | undefined => {
    const v = env[name];
    return v !== undefined && v !== "" ? v : undefined;
  };

  const envNoCache = envOf("JEV_MCP_NO_CACHE");
  const envCacheOff = envNoCache !== undefined && TRUTHY.has(envNoCache.toLowerCase());
  const cacheDisabled = noCacheFlag ?? envCacheOff;
  const cacheVerdicts = cacheDisabled ? false : DEFAULT_CONFIG.cacheVerdicts;

  // 3) toolPolicy는 방어적 복사본 (호출자가 변형해도 기본 정책 오염 방지).
  const toolPolicy: ToolPolicy = {
    annotateTools: [...DEFAULT_TOOL_POLICY.annotateTools],
    pruneTools: [...DEFAULT_TOOL_POLICY.pruneTools],
    gateTools: [...DEFAULT_TOOL_POLICY.gateTools],
    blockTools: [...DEFAULT_TOOL_POLICY.blockTools],
  };

  return {
    mode: resolveMode(raw["--jev-mode"], envOf("JEV_MCP_MODE")),
    budgetUsd: resolveNumber(
      raw["--jev-budget-usd"],
      envOf("JEV_MCP_BUDGET_USD"),
      DEFAULT_CONFIG.budgetUsd,
      "--jev-budget-usd",
      "JEV_MCP_BUDGET_USD",
    ),
    jevModel: resolveModel(raw["--jev-model"], envOf("JEV_MCP_MODEL")),
    injectionThreshold: resolveNumber(
      raw["--jev-injection-threshold"],
      envOf("JEV_MCP_INJECTION_THRESHOLD"),
      DEFAULT_CONFIG.injectionThreshold,
      "--jev-injection-threshold",
      "JEV_MCP_INJECTION_THRESHOLD",
    ),
    pruneKeepThreshold: resolveNumber(
      raw["--jev-prune-keep-threshold"],
      envOf("JEV_MCP_PRUNE_KEEP_THRESHOLD"),
      DEFAULT_CONFIG.pruneKeepThreshold,
      "--jev-prune-keep-threshold",
      "JEV_MCP_PRUNE_KEEP_THRESHOLD",
    ),
    destructiveThreshold: resolveNumber(
      raw["--jev-destructive-threshold"],
      envOf("JEV_MCP_DESTRUCTIVE_THRESHOLD"),
      DEFAULT_CONFIG.destructiveThreshold,
      "--jev-destructive-threshold",
      "JEV_MCP_DESTRUCTIVE_THRESHOLD",
    ),
    cacheVerdicts,
    requestTimeoutMs: resolveNumber(
      undefined,
      envOf("JEV_MCP_TIMEOUT_MS"),
      DEFAULT_CONFIG.requestTimeoutMs,
      "(no flag)",
      "JEV_MCP_TIMEOUT_MS",
    ),
    toolPolicy,
    upstreamArgs,
    apiKey: firstNonEmpty(env["TYPESAFE_API_KEY"], env["JEV_MCP_API_KEY"]),
  };
}
