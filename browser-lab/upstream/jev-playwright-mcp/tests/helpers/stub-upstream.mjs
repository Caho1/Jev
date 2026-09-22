#!/usr/bin/env node
/**
 * stub-upstream.mjs — 테스트용 최소 upstream MCP 서버 (stdio).
 *  - browser_snapshot        → 고정 yaml text 반환
 *  - browser_click           → ok text 반환
 *  - browser_run_code_unsafe → ok (프록시 blockTools 정책이 차단해야 정상)
 *
 *  tests/proxy.test.ts가 `node <this file>`로 spawn한다. stdout이 MCP 채널.
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

/** 테스트가 기대하는 고정 스냅샷 — tests/proxy.test.ts의 CANNED_SNAPSHOT과 동일. */
export const CANNED_SNAPSHOT = [
  "- Page URL: https://example.test/login",
  "- Page Title: Sign in",
  "- Page snapshot",
  "  - heading \"Sign in\" [level=1]",
  "  - textbox \"Email\"",
  "  - textbox \"Password\"",
  "  - button \"Sign in\"",
  "  - button \"Create account\"",
].join("\n");

const server = new Server(
  { name: "stub-upstream", version: "0.0.1" },
  { capabilities: { tools: {} } },
);

/** 마지막으로 받은 tools/list params._meta (progressToken 누출 검증용 echo). */
let lastListMeta = null;

server.setRequestHandler(ListToolsRequestSchema, async (request) => {
  lastListMeta = request.params?._meta ?? null;
  return {
    tools: [
      {
        name: "browser_snapshot",
        description: "stub: canned accessibility snapshot",
        inputSchema: { type: "object", properties: {} },
      },
      {
        name: "browser_click",
        description: "stub: click an element",
        inputSchema: {
          type: "object",
          properties: {
            element: { type: "string" },
            ref: { type: "string" },
          },
          required: ["element", "ref"],
        },
      },
      {
        name: "browser_run_code_unsafe",
        description: "stub: arbitrary code execution (blocked by policy)",
        inputSchema: {
          type: "object",
          properties: { code: { type: "string" } },
          required: ["code"],
        },
      },
      {
        name: "browser_progress",
        description: "stub: emit progress notifications then finish",
        inputSchema: { type: "object", properties: {} },
      },
    ],
    // 결과 _meta는 zod passthrough — 프록시가 progressToken을 걷어냈는지 검증용.
    _meta: { echo_list_meta: JSON.stringify(lastListMeta) },
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
  const name = request.params.name;
  const args = request.params.arguments ?? {};
  switch (name) {
    case "browser_snapshot":
      return { content: [{ type: "text", text: CANNED_SNAPSHOT }] };
    case "browser_click":
      return {
        content: [{
          type: "text",
          text: `Clicked [ref=${String(args.ref ?? "?")}] [element=${String(args.element ?? "?")}]`,
        }],
      };
    case "browser_run_code_unsafe":
      return { content: [{ type: "text", text: "code executed" }] };
    case "browser_progress": {
      // 실제 업스트림처럼: 요청이 준 progressToken(프록시 Client의 것)을
      // 그대로 돌려붙여 진행 알림을 보낸다.
      const token = request.params?._meta?.progressToken ?? 1;
      await extra.sendNotification({
        method: "notifications/progress",
        params: { progress: 1, total: 2, progressToken: token },
      });
      await extra.sendNotification({
        method: "notifications/progress",
        params: { progress: 2, total: 2, progressToken: token },
      });
      return { content: [{ type: "text", text: "progress done" }] };
    }
    default:
      return {
        content: [{ type: "text", text: `unknown tool: ${String(name)}` }],
        isError: true,
      };
  }
});

// 커스텀 메서드 폴백: 받은 params를 그대로 echo (progressToken 누출 검증용).
server.fallbackRequestHandler = async (request) => {
  return { echo: request.params ?? null };
};

await server.connect(new StdioServerTransport());
