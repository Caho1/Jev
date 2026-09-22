/**
 * Jev 클라이언트 단위 테스트 — global fetch를 mock해 네트워크 없이 검증.
 * (재시도 대기는 실제 타이머로 돈다: 최대 ~1.7s/test, 타임아웃 30s 내 여유)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NullJevClient, createJevClient } from "../src/jev/client.js";
import {
  DEFAULT_CONFIG,
  JevUnavailableError,
  costOfUsage,
  type JevMcpConfig,
  type JevRequest,
  type JevResponse,
} from "../src/contracts.js";

const ENDPOINT = "https://api.typesafe.ai/v1/systemone";

const enabledConfig = (): JevMcpConfig => ({
  ...DEFAULT_CONFIG,
  upstreamArgs: [],
  apiKey: "test-key",
});

const disabledConfig = (): JevMcpConfig => ({
  ...DEFAULT_CONFIG,
  upstreamArgs: [],
  apiKey: null,
});

const request: JevRequest = {
  state: { url: "https://example.com/login", title: "Sign in", text: "Please sign in to continue" },
  model: "jev-latest",
  questions: {
    q1: {
      type: "choice",
      instructions: "Classify the page state.",
      criteria: { expected: null, login_wall: "page blocks content behind a login form" },
    },
  },
};

const okResponse: JevResponse = {
  model: "jev-latest",
  answers: {
    q1: {
      type: "choice",
      choice: "login_wall",
      probabilities: { login_wall: 0.91, expected: 0.09 },
      confidence: 0.91,
    },
  },
  usage: { input_tokens: 312, output_tokens: 0 },
};

/** fetch가 쓰는 최소 응답 형태 (status/ok/text/headers). */
const mockResponse = (body: unknown, status = 200, headers?: Headers) => ({
  ok: status >= 200 && status < 300,
  status,
  headers,
  text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
});

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const captureError = async (p: Promise<unknown>): Promise<unknown> => p.catch((e) => e);

describe("createJevClient", () => {
  it("posts to /v1/systemone with bearer auth and parses the response", async () => {
    fetchMock.mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    const result = await client.decide(request);

    expect(result).toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      { method: string; headers: Record<string, string>; body: string },
    ];
    expect(url).toBe(ENDPOINT);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer test-key");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual(request);
  });

  it("records stats: calls, cost from input tokens, cumulative inputTokens", async () => {
    fetchMock.mockResolvedValue(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await client.decide(request);
    await client.decide(request);

    const stats = client.stats();
    expect(stats.calls).toBe(2);
    expect(stats.failures).toBe(0);
    expect(stats.inputTokens).toBe(624);
    // 312 tokens * 2 = 624 → 624/1M * $0.042
    expect(stats.costUsd).toBeCloseTo(0.000026208, 12);
  });

  it("computes cost with the official input-only pricing", () => {
    expect(costOfUsage({ input_tokens: 312, output_tokens: 7 })).toBeCloseTo(0.000013104, 12);
    expect(costOfUsage({ input_tokens: 1_000_000, output_tokens: 999_999 })).toBeCloseTo(0.042, 12);
    expect(costOfUsage({ input_tokens: 0, output_tokens: 0 })).toBe(0);
  });

  it("retries a 429 once and succeeds on the second attempt", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse({ detail: "rate limited" }, 429))
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    const result = await client.decide(request);

    expect(result.answers.q1).toMatchObject({ type: "choice", choice: "login_wall" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(client.stats()).toMatchObject({ calls: 1, failures: 0, inputTokens: 312 });
  });

  it("retries a 408 (request timeout) and succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse({ detail: "request timeout" }, 408))
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await expect(client.decide(request)).resolves.toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries a 500 (server error) and succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse({ detail: "internal error" }, 500))
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await expect(client.decide(request)).resolves.toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("retries a 529 (overloaded) and succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse("overloaded", 529))
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await expect(client.decide(request)).resolves.toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("prefers the retry-after-ms header over computed backoff", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockResponse({ detail: "slow down" }, 429, new Headers({ "retry-after-ms": "1" })),
      )
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await expect(client.decide(request)).resolves.toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("respects a seconds-form Retry-After header", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockResponse({ detail: "slow down" }, 429, new Headers({ "Retry-After": "0" })),
      )
      .mockResolvedValueOnce(mockResponse(okResponse));
    const client = createJevClient(enabledConfig());

    await expect(client.decide(request)).resolves.toEqual(okResponse);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("gives up after 2 retries on persistent 429 and counts one failure", async () => {
    fetchMock.mockResolvedValue(mockResponse({ detail: "rate limited" }, 429));
    const client = createJevClient(enabledConfig());

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    expect((err as Error).message).toMatch(/429/);
    expect(fetchMock).toHaveBeenCalledTimes(3); // 초기 1회 + 재시도 2회
    const stats = client.stats();
    expect(stats.calls).toBe(1);
    expect(stats.failures).toBe(1);
    expect(stats.costUsd).toBe(0);
    expect(stats.inputTokens).toBe(0);
  });

  it("does not retry on 401 and surfaces the server detail", async () => {
    fetchMock.mockResolvedValueOnce(mockResponse({ detail: "Invalid API key" }, 401));
    const client = createJevClient(enabledConfig());

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    expect((err as Error).message).toContain("401");
    expect((err as Error).message).toContain("Invalid API key");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(client.stats().failures).toBe(1);
  });

  it("does not retry on 422 and formats field-level validation detail", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse(
        {
          detail: [
            { loc: ["body", "questions", "q1"], msg: "Field required", type: "missing" },
            { loc: ["body", "state"], msg: "Input should be a valid string", type: "string_type" },
          ],
        },
        422,
      ),
    );
    const client = createJevClient(enabledConfig());

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    const message = (err as Error).message;
    expect(message).toContain("422");
    expect(message).toContain("body.questions.q1: Field required");
    expect(message).toContain("body.state: Input should be a valid string");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects malformed 200 responses", async () => {
    const badBodies = [
      "not json at all",
      JSON.stringify({ model: "jev-latest", answers: {} }), // usage 누락
      JSON.stringify({
        model: "jev-latest",
        answers: { q1: { type: "choice" } }, // choice 필드 누락
        usage: { input_tokens: 10, output_tokens: 0 },
      }),
      JSON.stringify({
        model: "jev-latest",
        answers: { q1: { type: "noul", noul: "high" } }, // noul이 숫자 아님
        usage: { input_tokens: 10, output_tokens: 0 },
      }),
    ];
    for (const body of badBodies) {
      fetchMock.mockResolvedValueOnce(mockResponse(body, 200));
      const client = createJevClient(enabledConfig());
      const err = await captureError(client.decide(request));
      expect(err).toBeInstanceOf(JevUnavailableError);
      expect((err as Error).message).toMatch(/Malformed Jev response/);
    }
  });

  it("retries network failures and gives up after 2 retries (3 attempts)", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const client = createJevClient(enabledConfig());

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    expect((err as Error).message).toContain("fetch failed");
    expect(fetchMock).toHaveBeenCalledTimes(3); // 네트워크 오류도 재시도 대상
    expect(client.stats().failures).toBe(1);
  });

  it("retries timeouts and reports the configured requestTimeoutMs", async () => {
    // signal abort를 거부로 연결해 타임아웃을 흉내낸다 (실제 타이머 사용).
    fetchMock.mockImplementation(
      (_url: unknown, init: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          init.signal.addEventListener("abort", () => {
            reject(new DOMException("This operation was aborted", "AbortError"));
          });
        }),
    );
    const client = createJevClient({ ...enabledConfig(), requestTimeoutMs: 25 });

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    expect((err as Error).message).toContain("timed out after 25ms");
    expect(fetchMock).toHaveBeenCalledTimes(3); // 타임아웃도 재시도 대상
    expect(client.stats().failures).toBe(1);
  });

  it("throws JevUnavailableError when disabled and never touches the network", async () => {
    const client = createJevClient(disabledConfig());
    expect(client.enabled).toBe(false);

    const err = await captureError(client.decide(request));

    expect(err).toBeInstanceOf(JevUnavailableError);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(client.stats()).toEqual({ calls: 0, failures: 0, costUsd: 0, inputTokens: 0 });
  });
});

describe("NullJevClient", () => {
  it("is a disabled passthrough client", async () => {
    expect(NullJevClient.enabled).toBe(false);
    const err = await captureError(NullJevClient.decide(request));
    expect(err).toBeInstanceOf(JevUnavailableError);
    expect(NullJevClient.stats()).toEqual({ calls: 0, failures: 0, costUsd: 0, inputTokens: 0 });
  });
});
