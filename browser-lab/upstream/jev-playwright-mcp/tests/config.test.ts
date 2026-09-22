/**
 * parseConfig 단위 테스트 — 플래그 형태, env 폴백, upstream 전달, 무효값 경고.
 */
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { parseConfig } from "../src/config.js";
import { DEFAULT_CONFIG, DEFAULT_TOOL_POLICY } from "../src/contracts.js";

let stderrSpy: MockInstance<typeof process.stderr.write>;

beforeEach(() => {
  stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
});

afterEach(() => {
  stderrSpy.mockRestore();
});

/** 지금까지 기록된 stderr 출력을 문자열 배열로. */
const stderrLines = (): string[] =>
  stderrSpy.mock.calls.map((call) => String(call[0])).filter((line) => line.length > 0);

describe("parseConfig", () => {
  it("returns defaults when no args and no env are given", () => {
    const cfg = parseConfig([], {});
    expect(cfg.mode).toBe("all");
    expect(cfg.budgetUsd).toBe(1.0);
    expect(cfg.jevModel).toBe("jev-latest");
    expect(cfg.injectionThreshold).toBe(0.5);
    expect(cfg.pruneKeepThreshold).toBe(0.35);
    expect(cfg.destructiveThreshold).toBe(0.6);
    expect(cfg.cacheVerdicts).toBe(true);
    expect(cfg.requestTimeoutMs).toBe(10_000);
    expect(cfg.toolPolicy).toEqual(DEFAULT_TOOL_POLICY);
    expect(cfg.upstreamArgs).toEqual([]);
    expect(cfg.apiKey).toBeNull();
    expect(stderrLines()).toEqual([]);
  });

  it("supports both --flag=value and --flag value forms", () => {
    const cfg = parseConfig(
      [
        "--jev-mode=gate",
        "--jev-budget-usd",
        "2.5",
        "--jev-model",
        "jev-prod",
        "--jev-injection-threshold=0.7",
        "--jev-prune-keep-threshold",
        "0.2",
        "--jev-destructive-threshold=0.9",
        "--jev-no-cache",
      ],
      {},
    );
    expect(cfg.mode).toBe("gate");
    expect(cfg.budgetUsd).toBe(2.5);
    expect(cfg.jevModel).toBe("jev-prod");
    expect(cfg.injectionThreshold).toBe(0.7);
    expect(cfg.pruneKeepThreshold).toBe(0.2);
    expect(cfg.destructiveThreshold).toBe(0.9);
    expect(cfg.cacheVerdicts).toBe(false);
    expect(cfg.upstreamArgs).toEqual([]);
    expect(stderrLines()).toEqual([]);
  });

  it("falls back to env mirrors when flags are absent", () => {
    const cfg = parseConfig(
      [],
      {
        JEV_MCP_MODE: "annotate",
        JEV_MCP_BUDGET_USD: "0.25",
        JEV_MCP_MODEL: "jev-env",
        JEV_MCP_INJECTION_THRESHOLD: "0.8",
        JEV_MCP_PRUNE_KEEP_THRESHOLD: "0.1",
        JEV_MCP_DESTRUCTIVE_THRESHOLD: "0.75",
        JEV_MCP_NO_CACHE: "1",
      },
    );
    expect(cfg.mode).toBe("annotate");
    expect(cfg.budgetUsd).toBe(0.25);
    expect(cfg.jevModel).toBe("jev-env");
    expect(cfg.injectionThreshold).toBe(0.8);
    expect(cfg.pruneKeepThreshold).toBe(0.1);
    expect(cfg.destructiveThreshold).toBe(0.75);
    expect(cfg.cacheVerdicts).toBe(false);
    expect(stderrLines()).toEqual([]);
  });

  it("flags take precedence over env", () => {
    const cfg = parseConfig(["--jev-mode=off"], { JEV_MCP_MODE: "annotate" });
    expect(cfg.mode).toBe("off");
  });

  it("treats empty env strings as unset", () => {
    const cfg = parseConfig([], { JEV_MCP_MODE: "", JEV_MCP_BUDGET_USD: "" });
    expect(cfg.mode).toBe(DEFAULT_CONFIG.mode);
    expect(cfg.budgetUsd).toBe(DEFAULT_CONFIG.budgetUsd);
  });

  it("forwards everything after '--' to upstreamArgs, preserving order", () => {
    const cfg = parseConfig(
      ["--jev-mode=gate", "--", "--jev-mode=off", "--headless", "positional.txt"],
      {},
    );
    expect(cfg.mode).toBe("gate");
    expect(cfg.upstreamArgs).toEqual(["--jev-mode=off", "--headless", "positional.txt"]);
  });

  it("forwards unrecognized flags and non-flag tokens in order", () => {
    const cfg = parseConfig(["--headless", "--jev-model=m2", "--port", "8931", "note.md"], {});
    expect(cfg.jevModel).toBe("m2");
    expect(cfg.upstreamArgs).toEqual(["--headless", "--port", "8931", "note.md"]);
  });

  it("degrades invalid numbers to defaults with exactly one stderr line each", () => {
    const cfg = parseConfig(["--jev-budget-usd=abc", "--jev-injection-threshold=-1"], {
      JEV_MCP_PRUNE_KEEP_THRESHOLD: "not-a-number",
    });
    expect(cfg.budgetUsd).toBe(DEFAULT_CONFIG.budgetUsd);
    expect(cfg.injectionThreshold).toBe(DEFAULT_CONFIG.injectionThreshold);
    expect(cfg.pruneKeepThreshold).toBe(DEFAULT_CONFIG.pruneKeepThreshold);

    expect(stderrSpy).toHaveBeenCalledTimes(3);
    const lines = stderrLines();
    expect(lines[0]).toContain("--jev-budget-usd");
    expect(lines[0]).toContain("abc");
    expect(lines[1]).toContain("--jev-injection-threshold");
    expect(lines[2]).toContain("JEV_MCP_PRUNE_KEEP_THRESHOLD");
    for (const line of lines) {
      expect(line.startsWith("[jev-mcp] warning:")).toBe(true);
      expect(line.endsWith("\n")).toBe(true);
      expect(line.split("\n").filter(Boolean)).toHaveLength(1);
    }
  });

  it("degrades an invalid mode to the default with a warning", () => {
    const cfg = parseConfig(["--jev-mode=speed"], {});
    expect(cfg.mode).toBe("all");
    expect(stderrSpy).toHaveBeenCalledTimes(1);
    expect(stderrLines()[0]).toContain("speed");
  });

  it("degrades an empty mode value to the default with a warning", () => {
    const cfg = parseConfig(["--jev-mode="], {});
    expect(cfg.mode).toBe("all");
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });

  it("consumes a space-form negative number and rejects it as invalid", () => {
    const cfg = parseConfig(["--jev-budget-usd", "-5"], {});
    expect(cfg.budgetUsd).toBe(DEFAULT_CONFIG.budgetUsd);
    expect(cfg.upstreamArgs).toEqual([]);
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });

  it("degrades empty numeric flag values to defaults with one warning each", () => {
    // `--flag=` 와 `--flag ""` 모두 Number("")===0 이 아니라 무효로 강등된다
    // (threshold 0 이 보안 동작을 바꾸는 일이 없게).
    const cfg = parseConfig(
      ["--jev-budget-usd=", "--jev-injection-threshold=", "--jev-destructive-threshold", ""],
      {},
    );
    expect(cfg.budgetUsd).toBe(DEFAULT_CONFIG.budgetUsd);
    expect(cfg.injectionThreshold).toBe(DEFAULT_CONFIG.injectionThreshold);
    expect(cfg.destructiveThreshold).toBe(DEFAULT_CONFIG.destructiveThreshold);
    expect(stderrSpy).toHaveBeenCalledTimes(3);
    for (const line of stderrLines()) {
      expect(line).toContain("empty number value");
    }
  });

  it("still accepts an explicit zero for numeric flags", () => {
    const cfg = parseConfig(["--jev-budget-usd=0"], {});
    expect(cfg.budgetUsd).toBe(0);
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it("parses JEV_MCP_TIMEOUT_MS for the Jev HTTP timeout (10s default)", () => {
    expect(parseConfig([], {}).requestTimeoutMs).toBe(10_000);
    expect(parseConfig([], { JEV_MCP_TIMEOUT_MS: "2500" }).requestTimeoutMs).toBe(2500);
    // 빈 문자열 env는 미설정 취급 (경고 없음).
    expect(parseConfig([], { JEV_MCP_TIMEOUT_MS: "" }).requestTimeoutMs).toBe(10_000);
    expect(stderrSpy).not.toHaveBeenCalled();
    // 무효값은 기본값 강등 + 경고 1행.
    expect(parseConfig([], { JEV_MCP_TIMEOUT_MS: "abc" }).requestTimeoutMs).toBe(10_000);
    expect(stderrSpy).toHaveBeenCalledTimes(1);
    expect(stderrLines()[0]).toContain("JEV_MCP_TIMEOUT_MS");
  });

  it("warns on a value flag missing its value and leaves the next flag intact", () => {
    const cfg = parseConfig(["--jev-model", "--headless"], {});
    expect(cfg.jevModel).toBe(DEFAULT_CONFIG.jevModel);
    expect(cfg.upstreamArgs).toEqual(["--headless"]);
    expect(stderrSpy).toHaveBeenCalledTimes(1);
    expect(stderrLines()[0]).toContain("--jev-model");
  });

  it("resolves the API key with TYPESAFE_API_KEY priority, treating blanks as unset", () => {
    expect(parseConfig([], { TYPESAFE_API_KEY: "ts-key" }).apiKey).toBe("ts-key");
    expect(parseConfig([], { JEV_MCP_API_KEY: "jev-key" }).apiKey).toBe("jev-key");
    expect(
      parseConfig([], { TYPESAFE_API_KEY: "ts-key", JEV_MCP_API_KEY: "jev-key" }).apiKey,
    ).toBe("ts-key");
    expect(parseConfig([], { TYPESAFE_API_KEY: "", JEV_MCP_API_KEY: "jev-key" }).apiKey).toBe(
      "jev-key",
    );
    expect(parseConfig([], { TYPESAFE_API_KEY: "   ", JEV_MCP_API_KEY: " " }).apiKey).toBeNull();
  });

  it("interprets JEV_MCP_NO_CACHE truthiness and flag precedence", () => {
    expect(parseConfig([], { JEV_MCP_NO_CACHE: "0" }).cacheVerdicts).toBe(true);
    expect(parseConfig([], { JEV_MCP_NO_CACHE: "false" }).cacheVerdicts).toBe(true);
    expect(parseConfig([], { JEV_MCP_NO_CACHE: "true" }).cacheVerdicts).toBe(false);
    // 플래그가 env를 이긴다: env가 꺼짐이어도 플래그 명시가 우선
    expect(parseConfig(["--jev-no-cache"], { JEV_MCP_NO_CACHE: "0" }).cacheVerdicts).toBe(false);
    expect(parseConfig(["--jev-no-cache=0"], {}).cacheVerdicts).toBe(true);
  });

  it("returns a defensive copy of the default tool policy", () => {
    const cfg = parseConfig([], {});
    expect(cfg.toolPolicy).not.toBe(DEFAULT_TOOL_POLICY);
    expect(cfg.toolPolicy.gateTools).not.toBe(DEFAULT_TOOL_POLICY.gateTools);

    cfg.toolPolicy.blockTools.push("browser_evil");
    expect(DEFAULT_TOOL_POLICY.blockTools).not.toContain("browser_evil");
  });
});
