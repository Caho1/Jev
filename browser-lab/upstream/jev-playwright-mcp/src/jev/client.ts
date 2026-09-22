/**
 * Jev HTTP 클라이언트 — POST https://api.typesafe.ai/v1/systemone.
 *  - enabled = !!config.apiKey. 비활성이면 decide()는 JevUnavailableError (호출자은 .enabled 선검사).
 *  - 재시도(공식 typesafe-sdk-js DEFAULT_RETRY_POLICY 정렬, 최대 2회 = 3 attempts):
 *      HTTP 408/429/모든 5xx + 네트워크/연결 오류 + 타임아웃.
 *      'retry-after-ms' 헤더(ms, 우선) 또는 'Retry-After'(초|HTTP-date)를 존중,
 *      60s 상한. 헤더가 없으면 지수 백오프 500ms 초기 · 5s 상한 · ±25% 지터.
 *  - 401/422/그 외 4xx: 재시도 없이 JevUnavailableError (서버 field-level detail 포함).
 *  - stats(): 누적 calls/failures/costUsd/inputTokens (cost는 input tokens 기준).
 */
import {
  JevUnavailableError,
  costOfUsage,
  type JevClient,
  type JevMcpConfig,
  type JevRequest,
  type JevResponse,
} from "../contracts.js";

const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
const DEFAULT_REQUEST_TIMEOUT_MS = 10_000;
const MAX_RETRIES = 2;
const BASE_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 5_000;
/** 지터 폭 — 백오프 지연의 ±25%. */
const JITTER_FRACTION = 0.25;
/** Retry-After 헤더로 기다릴 수 있는 상한. */
const RETRY_AFTER_CAP_MS = 60_000;

/** passthrough 모드용 null 객체 — API 키가 없을 때 createJevClient가 그대로 반환. */
export const NullJevClient: JevClient = {
  enabled: false,
  async decide(): Promise<JevResponse> {
    throw new JevUnavailableError("Jev is disabled (no API key) — check .enabled before calling decide()");
  },
  stats() {
    return { calls: 0, failures: 0, costUsd: 0, inputTokens: 0 };
  },
};

/** API 응답 오류 — 재시도 가능 여부를 구분하기 위한 내부 타입. */
class ApiError extends JevUnavailableError {
  readonly status: number;
  readonly retryable: boolean;
  /** Retry-After 헤더에서 파생된 대기 시간(ms) — 없으면 null. */
  readonly retryAfterMs: number | null;
  constructor(message: string, status: number, retryable: boolean, retryAfterMs: number | null = null) {
    super(message);
    this.status = status;
    this.retryable = retryable;
    this.retryAfterMs = retryAfterMs;
  }
}

/** 네트워크/연결/타임아웃 오류 — 언제나 재시도 가능. */
class TransportError extends JevUnavailableError {
  readonly retryable = true;
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** 지수 백오프(500ms 초기 · 5s 상한) + ±25% 지터. */
const backoffDelayMs = (retryIndex: number): number => {
  const base = Math.min(BASE_BACKOFF_MS * 2 ** retryIndex, MAX_BACKOFF_MS);
  const spread = base * JITTER_FRACTION;
  return base - spread + Math.random() * spread * 2;
};

/** 헤더에서 대기 시간 파싱 — 'retry-after-ms'(ms, 우선) 또는 'Retry-After'(초|HTTP-date), 60s 상한. */
function retryAfterMsFromHeaders(headers: Headers | undefined): number | null {
  const rawMs = headers?.get?.("retry-after-ms");
  if (rawMs !== null && rawMs !== undefined && rawMs.trim() !== "") {
    const ms = Number(rawMs);
    if (Number.isFinite(ms) && ms >= 0) return Math.min(ms, RETRY_AFTER_CAP_MS);
  }
  const raw = headers?.get?.("retry-after") ?? headers?.get?.("Retry-After");
  if (raw === null || raw === undefined || raw.trim() === "") return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1000, RETRY_AFTER_CAP_MS);
  }
  // HTTP-date 형태 — 파싱 실패 시 null (백오프로 폴백).
  const at = Date.parse(raw);
  if (!Number.isFinite(at)) return null;
  return Math.min(Math.max(at - Date.now(), 0), RETRY_AFTER_CAP_MS);
}

const isFiniteNumber = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

const isNumberRecord = (v: unknown): boolean => {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return false;
  return Object.values(v).every(isFiniteNumber);
};

function describeNetworkError(err: unknown, timeoutMs: number): string {
  if (err instanceof Error && err.name === "AbortError") {
    return `request timed out after ${timeoutMs}ms`;
  }
  return err instanceof Error ? err.message : String(err);
}

function summarizeBody(bodyText: string, maxLen = 160): string {
  const trimmed = bodyText.trim();
  return trimmed.length > maxLen ? `${trimmed.slice(0, maxLen)}…` : trimmed;
}

/** 서버 오류 본문 정형화 — FastAPI 스타일 field-level detail 우선. */
function formatDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail === "" ? null : detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry): string => {
        if (typeof entry === "object" && entry !== null) {
          const e = entry as Record<string, unknown>;
          const loc = Array.isArray(e["loc"]) ? e["loc"].map(String).join(".") : null;
          const msg = typeof e["msg"] === "string" ? e["msg"] : null;
          if (loc !== null && msg !== null) return `${loc}: ${msg}`;
          return msg ?? loc ?? "";
        }
        return String(entry);
      })
      .filter((part) => part !== "");
    return parts.length > 0 ? parts.join("; ") : null;
  }
  if (typeof detail === "object" && detail !== null) {
    const e = detail as Record<string, unknown>;
    if (typeof e["message"] === "string" && e["message"] !== "") return e["message"];
  }
  return null;
}

function formatApiError(status: number, bodyText: string): string {
  const prefix = `Jev API ${status}`;
  let data: unknown;
  try {
    data = JSON.parse(bodyText);
  } catch {
    return `${prefix}: ${summarizeBody(bodyText)}`;
  }
  if (typeof data === "object" && data !== null) {
    const obj = data as Record<string, unknown>;
    const formatted = formatDetail(obj["detail"] ?? obj["error"] ?? obj["message"]);
    if (formatted !== null) return `${prefix}: ${formatted}`;
  }
  return `${prefix}: ${summarizeBody(bodyText)}`;
}

function validateAnswer(id: string, answer: unknown): string | null {
  if (typeof answer !== "object" || answer === null || Array.isArray(answer)) {
    return `answers.${id} is not an object`;
  }
  const a = answer as Record<string, unknown>;
  switch (a["type"]) {
    case "noul":
      return isFiniteNumber(a["noul"])
        ? null
        : `answers.${id}.noul is not a finite number`;
    case "choice":
      if (typeof a["choice"] !== "string") return `answers.${id}.choice is not a string`;
      break;
    case "score":
      if (!isFiniteNumber(a["score"])) return `answers.${id}.score is not a finite number`;
      if (typeof a["legend"] !== "object" || a["legend"] === null || Array.isArray(a["legend"])) {
        return `answers.${id}.legend is not a record`;
      }
      break;
    default:
      return `answers.${id}.type must be one of noul|choice|score`;
  }
  // choice/score 공통 필드
  if (!isNumberRecord(a["probabilities"])) return `answers.${id}.probabilities is not a number record`;
  if (!isFiniteNumber(a["confidence"])) return `answers.${id}.confidence is not a finite number`;
  return null;
}

function validateUsage(usage: unknown): string | null {
  if (typeof usage !== "object" || usage === null || Array.isArray(usage)) {
    return "usage is not an object";
  }
  const u = usage as Record<string, unknown>;
  for (const key of ["input_tokens", "output_tokens"] as const) {
    const v = u[key];
    if (!isFiniteNumber(v) || v < 0) return `usage.${key} is not a non-negative number`;
  }
  return null;
}

function validateResponse(data: unknown): string | null {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return "body is not an object";
  }
  const obj = data as Record<string, unknown>;
  if (typeof obj["model"] !== "string" || obj["model"] === "") {
    return "model is missing or not a string";
  }
  const answers = obj["answers"];
  if (typeof answers !== "object" || answers === null || Array.isArray(answers)) {
    return "answers is not a record";
  }
  for (const [id, answer] of Object.entries(answers)) {
    const reason = validateAnswer(id, answer);
    if (reason !== null) return reason;
  }
  return validateUsage(obj["usage"]);
}

function parseAndValidate(bodyText: string): JevResponse {
  let data: unknown;
  try {
    data = JSON.parse(bodyText);
  } catch {
    throw new JevUnavailableError("Malformed Jev response: body is not valid JSON");
  }
  const reason = validateResponse(data);
  if (reason !== null) throw new JevUnavailableError(`Malformed Jev response: ${reason}`);
  return data as JevResponse;
}

/** 공식 재시도 정책: 408(요청 타임아웃), 429(레이트 리밋), 모든 5xx. */
const isRetryableStatus = (status: number): boolean =>
  status === 408 || status === 429 || status >= 500;

/** 1회 POST 시도 (타임아웃 config.requestTimeoutMs, 상태코드 분류). */
async function postOnce(
  req: JevRequest,
  apiKey: string,
  timeoutMs: number,
): Promise<JevResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(JEV_ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
      signal: controller.signal,
    });
  } catch (err) {
    // 네트워크/연결 오류와 타임아웃은 재시도 대상이다.
    throw new TransportError(`Jev request failed: ${describeNetworkError(err, timeoutMs)}`);
  } finally {
    clearTimeout(timer);
  }

  const bodyText = await response.text();
  if (isRetryableStatus(response.status)) {
    throw new ApiError(
      `Jev API ${response.status} (timeout/rate limited/server error): ${summarizeBody(bodyText)}`,
      response.status,
      true,
      retryAfterMsFromHeaders(response.headers),
    );
  }
  if (!response.ok) {
    throw new ApiError(formatApiError(response.status, bodyText), response.status, false);
  }
  return parseAndValidate(bodyText);
}

export function createJevClient(config: JevMcpConfig): JevClient {
  if (!config.apiKey) return NullJevClient;
  const apiKey = config.apiKey;
  const timeoutMs = config.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;

  let calls = 0;
  let failures = 0;
  let costUsd = 0;
  let inputTokens = 0;

  return {
    enabled: true,
    async decide(req: JevRequest): Promise<JevResponse> {
      calls += 1;
      for (let attempt = 0; ; attempt++) {
        try {
          const response = await postOnce(req, apiKey, timeoutMs);
          costUsd += costOfUsage(response.usage);
          inputTokens += response.usage.input_tokens;
          return response;
        } catch (err) {
          const error =
            err instanceof JevUnavailableError
              ? err
              : new JevUnavailableError(`Jev client error: ${describeNetworkError(err, timeoutMs)}`);
          const retryable =
            (err instanceof ApiError && err.retryable) || err instanceof TransportError;
          const canRetry = retryable && attempt < MAX_RETRIES;
          if (!canRetry) {
            failures += 1;
            throw error;
          }
          const headerDelay =
            err instanceof ApiError && err.retryAfterMs !== null ? err.retryAfterMs : null;
          await sleep(headerDelay ?? backoffDelayMs(attempt));
        }
      }
    },
    stats() {
      return { calls, failures, costUsd, inputTokens };
    },
  };
}
