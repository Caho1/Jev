/**
 * upstream.test.ts — 업스트림 실행 환경 관련 단위 테스트.
 *  - stripJevSecrets: Jev 전용 결제 크리덴셜이 업스트림 자식 env로 흘러가지 않게 한다.
 */
import { describe, expect, it } from "vitest";
import { PINNED_PLAYWRIGHT_MCP_VERSION, stripJevSecrets } from "../src/upstream.js";

describe("stripJevSecrets", () => {
  it("removes TYPESAFE_API_KEY / JEV_MCP_API_KEY while forwarding everything else", () => {
    const env = {
      PATH: "/usr/bin",
      PLAYWRIGHT_MCP_EXECUTABLE_PATH: "/chromium",
      TYPESAFE_API_KEY: "tsk-secret",
      JEV_MCP_API_KEY: "jev-secret",
      HOME: "/home/user",
    };
    const child = stripJevSecrets(env);
    expect(child).toEqual({
      PATH: "/usr/bin",
      PLAYWRIGHT_MCP_EXECUTABLE_PATH: "/chromium",
      HOME: "/home/user",
    });
    // 원본은 불변.
    expect(env["TYPESAFE_API_KEY"]).toBe("tsk-secret");
  });

  it("is a no-op when no jev secrets are present", () => {
    const env = { PATH: "/usr/bin" };
    expect(stripJevSecrets(env)).toEqual(env);
  });
});

describe("PINNED_PLAYWRIGHT_MCP_VERSION", () => {
  it("matches the package.json dependency pin", () => {
    expect(PINNED_PLAYWRIGHT_MCP_VERSION).toBe("0.0.81");
  });
});
