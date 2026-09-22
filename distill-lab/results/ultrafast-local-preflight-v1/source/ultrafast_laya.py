"""Loopback TypeSafe-compatible Laya server for jev-ultrafast, using frozen local weights."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evaluate_computer_replay import LocalBackend

LAB = Path(__file__).resolve().parents[1]


def normalize_question(question):
    if question.get("type") != "choice":
        raise ValueError("ultrafast only supports choice questions")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
        raise ValueError("Expected 1–255 observed choices")
    if any(not isinstance(key, str) or not key for key in criteria):
        raise ValueError("Choice IDs must be nonempty strings")
    instructions = question.get("instructions", "")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, separators=(",", ":"))
    return {"type": "choice", "criteria": criteria, "instructions": instructions}


class UltrafastLaya(LocalBackend):
    def evaluate(self, request):
        import torch
        from typed_laya import encode_question, model_batch

        questions = request.get("questions")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 4:
            raise ValueError("Expected 1–4 questions")
        if not isinstance(request.get("state"), dict):
            raise ValueError("Expected structured browser state")
        prepared, answers, lengths, singleton = [], {}, {}, []
        # Validate every head before inference, preserving every option and rejecting overflow.
        for name, raw in questions.items():
            question = normalize_question(raw)
            if len(question["criteria"]) == 1:
                key = next(iter(question["criteria"]))
                answers[name] = {"type": "choice", "choice": key, "confidence": 1.0,
                                 "probabilities": {key: 1.0}}
                singleton.append(name)
                continue
            item, details = encode_question(self.tokenizer, request["state"], question, 8192)
            prepared.append((name, item, details))
            lengths[name] = details["input_tokens"]
        started = time.perf_counter()
        with torch.inference_mode():
            for name, item, details in prepared:
                batch = model_batch([item], self.tokenizer, self.device)
                logits, _ = self.model(**batch)
                values = logits[0, :len(details["labels"] )].float().cpu().double()
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite local logits")
                probabilities = torch.softmax(values, dim=-1).tolist()
                labels = details["labels"]
                best = max(range(len(labels)), key=probabilities.__getitem__)
                answers[name] = {"type": "choice", "choice": labels[best],
                                 "confidence": probabilities[best],
                                 "probabilities": dict(zip(labels, probabilities))}
        return {"model": self.identity["variant"], "answers": answers,
                "usage": {"local": True, "forward_passes": len(prepared),
                          "singleton_heads": singleton, "input_tokens_by_head": lengths,
                          "inference_ms": (time.perf_counter() - started) * 1000,
                          "shared_state_encoding": False}}


def make_server(backend, port=8767):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "not found"})
            self.send_json(200, {"ready": True, "backend": backend.identity})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.send_json(404, {"error": "not found"})
            # This service is for the local agent, never cross-origin web-page requests.
            if self.headers.get("Origin") is not None:
                return self.send_json(403, {"error": "browser-origin requests forbidden"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                with lock:
                    result = backend.evaluate(request)
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": type(exc).__name__, "detail": str(exc)})
            except Exception as exc:
                self.send_json(500, {"error": type(exc).__name__})

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["laya_base", "laya_computer"], default="laya_base")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--run", type=Path, default=LAB / "results/computer-use-grounding-cuda-v1")
    parser.add_argument("--weights", type=Path, default=LAB / "checkpoints/banking77-lora-v1/base")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    backend = UltrafastLaya(args.variant, args.device, args.run, args.weights)
    server = make_server(backend, args.port)
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/v1", **backend.identity}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
