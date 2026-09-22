/**
 * stub-jev-fetch.cjs — E2E 스모크 전용 fetch 스텁 (프로덕션 코드 무변경).
 *
 *  사용법: 자식 프로세스 env 의 NODE_OPTIONS 에 `--require <this file>` 을
 *  넣어 선로드한다. https://api.typesafe.ai/v1/systemone 로 향하는 POST 만
 *  결정론적인 가짜 JevResponse 로 응답하고, 그 외 모든 fetch 는 실제
 *  global fetch 로 통과시킨다(업스트림/브라우저 동작은 영향 없음).
 *
 *  응답 규칙 (src/jev/client.ts 검증기가 요구하는 형태):
 *   - "page_state" choice → expected (p=0.90)
 *   - "risk"      choice → safe
 *   - 기타 choice  → 첫 criteria 키
 *   - inj_* noul   → 스팬이 "ignore all previous instructions" 포함 시 0.93,
 *                    아니면 0.05
 *   - r* noul(프루닝) → 0.90 (전부 keep — 스냅샷 원문 보존)
 */
"use strict";

const JEV_URL = "https://api.typesafe.ai/v1/systemone";
const realFetch = globalThis.fetch.bind(globalThis);

function urlOf(input) {
  if (typeof input === "string") return input;
  if (input && typeof input.url === "string") return input.url; // Request
  try {
    return String(input); // URL
  } catch {
    return "";
  }
}

function answerFor(id, question, state) {
  if (question.type === "noul") {
    // 프루닝 판정(r* 질문)은 전부 keep — 스냅샷 원문을 그대로 보존한다.
    const regions = Array.isArray(state && state.region_labels_and_excerpts);
    if (regions) return { type: "noul", noul: 0.9 };
    // 인젝션 후보(inj_*)는 스팬 내용 기반.
    const spans = Array.isArray(state && state.candidate_spans) ? state.candidate_spans : [];
    const entry = spans.find((s) => s && s.id === id);
    const span = String((entry && entry.span) || "");
    const noul = /ignore all previous instructions/i.test(span) ? 0.93 : 0.05;
    return { type: "noul", noul };
  }
  if (question.type === "choice") {
    const choice =
      id === "page_state" ? "expected" : id === "risk" ? "safe" : Object.keys(question.criteria)[0];
    return {
      type: "choice",
      choice,
      probabilities: { [choice]: 0.9, other: 0.1 },
      confidence: 0.9,
    };
  }
  return { type: "score", score: 1, legend: {}, probabilities: {}, confidence: 0.9 };
}

function stubResponse(req) {
  const state = (req && typeof req.state === "object" && req.state) || {};
  const answers = {};
  for (const [id, question] of Object.entries((req && req.questions) || {})) {
    answers[id] = answerFor(id, question, state);
  }
  const body = JSON.stringify({
    model: (req && req.model) || "stub-jev",
    answers,
    usage: { input_tokens: 1234, output_tokens: 10 },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

globalThis.fetch = function stubFetch(input, init) {
  if (urlOf(input) === JEV_URL) {
    let req = null;
    try {
      req = JSON.parse(String((init && init.body) || "{}"));
    } catch {
      req = {};
    }
    return Promise.resolve(stubResponse(req));
  }
  return realFetch(input, init);
};
