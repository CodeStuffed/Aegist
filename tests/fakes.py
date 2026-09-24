"""
Stand-ins for the two model servers, speaking their real HTTP APIs so the
actual runner code (requests / anthropic SDK) is exercised end to end.

Replies are canned: a "scripted brain" looks at which persona's system
prompt it received and answers with plausible JSON. It knows nothing - it
exists to prove the plumbing, not the reasoning.

Run a fake Ollama by hand for a CLI demo:
    python -m tests.fakes --port 11500
    OLLAMA_HOST=http://127.0.0.1:11500 python cli.py ask --backend local "..."
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MONEY_WORDS = ("pricing", "price", "subscription", "revenue", "charge", "sell", "startup",
               "saas", "market", "customers", "$")


def scripted_reply(system: str, prompt: str) -> str:
    """Canned JSON reply for whichever council role `system` belongs to."""
    claim = prompt.split("CLAIM:", 1)[-1].strip().splitlines()[0] if "CLAIM:" in prompt else prompt
    if system.startswith("You are the Believer"):
        return json.dumps({
            "position": f"The strongest case for '{claim}' rests on a clear mechanism.",
            "key_points": ["Mechanism is plausible", "Comparable cases exist"],
            "confidence": "Medium",
            "what_would_change_my_mind": "Evidence the mechanism fails in practice.",
        })
    if system.startswith("You are the Skeptic"):
        return json.dumps({
            "position": "The claim leans on an unstated assumption about adoption.",
            "key_points": ["Adoption rate assumed, not shown", "Cost side ignored"],
            "confidence": "High",
            "what_would_change_my_mind": "Adoption data from a comparable rollout.",
        })
    if system.startswith("You are the Investor"):
        return json.dumps({
            "position": "Addressable market looks real but unit economics are unproven.",
            "key_points": ["TAM guess: ~$200M (a guess)", "CAC unknown"],
            "confidence": "Low",
            "what_would_change_my_mind": "Payback period under 12 months.",
        })
    if system.startswith("You are the Judge"):
        return json.dumps({
            "verdict": "No - as stated this doesn't hold up yet.",
            "reasoning": "The Skeptic's adoption objection went unanswered by the Believer.",
            "confidence": "High",
            "unresolved": "Actual adoption rate.",
            "stance": "no",
            "relied_on": ["skeptic", "believer"],
        })
    if "is_money_idea" in system:
        money = any(w in claim.lower() for w in MONEY_WORDS)
        return json.dumps({
            "is_money_idea": money,
            "reason": "Mentions pricing/revenue." if money else "No business model involved.",
            "topic": " ".join(claim.lower().split()[:4]),
        })
    return json.dumps({"echo": prompt[:80]})


class ScriptedBackend:
    """Returns the queued replies in order; records every prompt."""

    NAME = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, prompt, *, role="panel", temperature=None, json_mode=True):
        self.calls.append({"system": system, "prompt": prompt, "role": role, "temperature": temperature})
        return {"text": self.replies.pop(0), "backend": self.NAME, "model": "fake-1"}

    def model_for(self, role="panel"):
        return "fake-1"


class _Handler(BaseHTTPRequestHandler):
    server: "_FakeServer"

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")


class _FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler, port: int = 0):
        super().__init__(("127.0.0.1", port), handler)
        self.requests: list[dict] = []
        self.reply_override = None  # callable(system, prompt) -> str | None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def reply(self, system: str, prompt: str) -> str:
        if self.reply_override is not None:
            text = self.reply_override(system, prompt)
            if text is not None:
                return text
        return scripted_reply(system, prompt)

    def start(self) -> "_FakeServer":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


class _OllamaHandler(_Handler):
    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": m} for m in self.server.pulled]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        body = self._json_body()
        self.server.requests.append(body)
        if body["model"] not in self.server.pulled:
            return self._send(404, {"error": f"model '{body['model']}' not found"})
        msgs = {m["role"]: m["content"] for m in body["messages"]}
        text = self.server.reply(msgs.get("system", ""), msgs.get("user", ""))
        self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": text},
                         "done": True, "done_reason": "stop"})


class FakeOllama(_FakeServer):
    def __init__(self, pulled=("qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:32b"), port: int = 0):
        super().__init__(_OllamaHandler, port)
        self.pulled = set(pulled)


class _AnthropicHandler(_Handler):
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"data": [], "has_more": False, "first_id": None, "last_id": None})
        else:
            self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "nope"}})

    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            return self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "nope"}})
        body = self._json_body()
        self.server.requests.append(body)
        if self.server.response_override is not None:
            return self._send(200, self.server.response_override(body))
        prompt = body["messages"][-1]["content"]
        text = self.server.reply(body.get("system", ""), prompt if isinstance(prompt, str) else "")
        self._send(200, message_body(body["model"], [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ]))


def message_body(model: str, content: list, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_fake", "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


class FakeAnthropic(_FakeServer):
    def __init__(self):
        super().__init__(_AnthropicHandler)
        self.response_override = None  # callable(request_body) -> full message body


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scripted fake Ollama server.")
    parser.add_argument("--port", type=int, default=11500)
    args = parser.parse_args()
    server = FakeOllama(port=args.port)
    print(f"fake Ollama listening on {server.url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
