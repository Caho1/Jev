/**
 * env.test.ts — .env 로더 단위 테스트 (파일시스템은 tmpdir만 사용, 네트워크 없음).
 */
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  applyEnvFileLines,
  findPackageRoot,
  loadPackageEnv,
} from "../src/env.js";

const createdDirs: string[] = [];
afterEach(() => {
  while (true) {
    const dir = createdDirs.pop();
    if (dir === undefined) break;
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeTempDir(): string {
  const dir = mkdtempSync(path.join(tmpdir(), "jev-env-test-"));
  createdDirs.push(dir);
  return dir;
}

describe("applyEnvFileLines", () => {
  it("applies KEY=VALUE lines and returns the number applied", () => {
    const env: NodeJS.ProcessEnv = {};
    const applied = applyEnvFileLines(
      ["TYPESAFE_API_KEY=abc123", "JEV_MCP_MODE=all", "# comment", "", "   "],
      env,
    );
    expect(applied).toBe(2);
    expect(env["TYPESAFE_API_KEY"]).toBe("abc123");
    expect(env["JEV_MCP_MODE"]).toBe("all");
  });

  it("never overrides variables that already carry a non-empty value", () => {
    const env: NodeJS.ProcessEnv = { TYPESAFE_API_KEY: "from-shell" };
    const applied = applyEnvFileLines(["TYPESAFE_API_KEY=from-file"], env);
    expect(applied).toBe(0);
    expect(env["TYPESAFE_API_KEY"]).toBe("from-shell");
  });

  it("fills empty-string variables (parseConfig treats '' as unset)", () => {
    const env: NodeJS.ProcessEnv = { TYPESAFE_API_KEY: "" };
    const applied = applyEnvFileLines(["TYPESAFE_API_KEY=from-file"], env);
    expect(applied).toBe(1);
    expect(env["TYPESAFE_API_KEY"]).toBe("from-file");
  });

  it("handles export prefixes, quote unwrapping, escapes and inline comments", () => {
    const env: NodeJS.ProcessEnv = {};
    applyEnvFileLines(
      [
        "export JEV_MCP_MODEL=jev-latest",
        'QUOTED="hello world"',
        'ESCAPED="line\\nbreak \\"quoted\\""',
        "SINGLE='not # a comment'",
        "COMMENTED=value # trailing note",
        "SPACED =  padded  ",
      ],
      env,
    );
    expect(env["JEV_MCP_MODEL"]).toBe("jev-latest");
    expect(env["QUOTED"]).toBe("hello world");
    expect(env["ESCAPED"]).toBe('line\nbreak "quoted"');
    expect(env["SINGLE"]).toBe("not # a comment");
    expect(env["COMMENTED"]).toBe("value");
    expect(env["SPACED"]).toBe("padded");
  });

  it("unquotes values followed by inline comments (dotenv idiom)", () => {
    // `KEY="value" # comment` — 주석을 먼저 떼고 따옴표를 벗겨야 한다.
    // 따옴표를 값에 남기면 Bearer 헤더가 깨져 조용한 401 이 된다.
    const env: NodeJS.ProcessEnv = {};
    applyEnvFileLines(
      [
        'QUOTED_WITH_COMMENT="hello" # note',
        "SINGLE_WITH_COMMENT='v' # c",
        'TYPESAFE_API_KEY="tsk_abc123" # production key',
        'ESCAPED_WITH_COMMENT="line\\nbreak" # note',
      ],
      env,
    );
    expect(env["QUOTED_WITH_COMMENT"]).toBe("hello");
    expect(env["SINGLE_WITH_COMMENT"]).toBe("v");
    expect(env["TYPESAFE_API_KEY"]).toBe("tsk_abc123");
    expect(env["ESCAPED_WITH_COMMENT"]).toBe("line\nbreak");
  });

  it("keeps the fallback path for values without a closing quote", () => {
    const env: NodeJS.ProcessEnv = {};
    applyEnvFileLines(['UNCLOSED="value # not a comment marker'], env);
    // 닫는 따옴표가 없으면 종전처럼 인라인 주석 제거 경로를 따른다.
    expect(env["UNCLOSED"]).toBe('"value');
  });

  it("silently skips malformed lines and invalid key names", () => {
    const env: NodeJS.ProcessEnv = {};
    const applied = applyEnvFileLines(
      ["NO_EQUALS_SIGN", "=NO_KEY", "1BAD=key", "A B=key", "GOOD=1"],
      env,
    );
    expect(applied).toBe(1);
    expect(env["GOOD"]).toBe("1");
  });
});

describe("loadPackageEnv", () => {
  it("finds the package root by walking up to package.json", () => {
    const root = makeTempDir();
    writeFileSync(path.join(root, "package.json"), "{}");
    const nested = path.join(root, "a", "b");
    mkdirSync(nested, { recursive: true });
    expect(findPackageRoot(nested)).toBe(root);
  });

  it("applies .env from the package root without touching preset vars", () => {
    const root = makeTempDir();
    writeFileSync(path.join(root, "package.json"), "{}");
    writeFileSync(
      path.join(root, ".env"),
      ["JEV_ENV_TEST_A=from-file", "JEV_ENV_TEST_PRESET=from-file"].join("\n"),
    );
    const env: NodeJS.ProcessEnv = { JEV_ENV_TEST_PRESET: "from-process" };
    const result = loadPackageEnv(env, path.join(root, "src"));
    expect(result?.file).toBe(path.join(root, ".env"));
    expect(result?.applied).toBe(1);
    expect(env["JEV_ENV_TEST_A"]).toBe("from-file");
    expect(env["JEV_ENV_TEST_PRESET"]).toBe("from-process");
  });

  it("returns null when .env is absent", () => {
    const root = makeTempDir();
    writeFileSync(path.join(root, "package.json"), "{}");
    expect(loadPackageEnv({}, path.join(root, "src"))).toBeNull();
  });

  it("is disabled entirely by JEV_MCP_NO_ENV_FILE", () => {
    const root = makeTempDir();
    writeFileSync(path.join(root, "package.json"), "{}");
    writeFileSync(path.join(root, ".env"), "JEV_ENV_TEST_B=from-file");
    for (const value of ["1", "true", "yes", "on", "TRUE"]) {
      const env: NodeJS.ProcessEnv = { JEV_MCP_NO_ENV_FILE: value };
      expect(loadPackageEnv(env, path.join(root, "src"))).toBeNull();
      expect(env["JEV_ENV_TEST_B"]).toBeUndefined();
    }
  });
});
