"""AnthropicLLM against a local fake endpoint: checks the request we send and how we parse the reply.

No network and no key - this verifies our use of the SDK (params accepted, thinking/tool_use blocks
round-trip unchanged), not the model's behaviour.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from opsagent.llm import AnthropicLLM

REPLY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "summary", "signature": "SIG=="},
        {"type": "text", "text": "Checking logs."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_logs", "input": {"service": "web"}},
    ],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 3},
}


@pytest.fixture
def fake_api(monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            data = json.dumps(REPLY).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("request-id", "req_test")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield seen
    srv.shutdown()


def test_request_shape_and_response_normalisation(fake_api):
    llm = AnthropicLLM(effort="low")
    tools = [{"name": "get_logs", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    resp = asyncio.run(llm.complete("sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], tools))

    body = fake_api["body"]
    assert fake_api["path"].startswith("/v1/messages")
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "low"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body and "budget_tokens" not in json.dumps(body)
    assert body["tools"] == tools and body["system"] == "sys"

    assert resp.stop_reason == "tool_use" and resp.request_id == "req_test"
    assert resp.usage == {"input_tokens": 12, "output_tokens": 7, "cache_read_tokens": 5, "cache_write_tokens": 3}
    # blocks come back as minimal dicts, thinking signature intact for replay
    assert resp.content == REPLY["content"]
