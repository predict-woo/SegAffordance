"""Minimal client for `codex app-server` — use Codex like an LLM API.

Speaks the app-server JSON-RPC protocol over stdio
(https://learn.chatgpt.com/docs/app-server.md): initialize handshake, one
thread per client, then one `turn/start` per request with text + optional
image input, returning the final agentMessage text.

Library use:
    with CodexClient() as c:
        text = c.describe("What part is highlighted?", image="frame.jpg")

CLI smoke test (run on the pod):
    python tools/codex_client.py --image sample_viz/opdreal_1.jpg \
        --prompt "One imperative sentence for the green-highlighted part."
"""

import argparse
import json
import queue
import subprocess
import threading


class CodexError(RuntimeError):
    pass


class CodexClient:
    def __init__(self, model: str | None = None, effort: str | None = None,
                 cwd: str | None = None,
                 codex_bin: str = "codex", timeout: float = 240.0):
        self.model = model
        self.effort = effort
        self.cwd = cwd
        self.timeout = timeout
        self._id = 0
        self._proc = subprocess.Popen(
            [codex_bin, "app-server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._msgs: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._request("initialize", {
            "clientInfo": {"name": "segaffordance", "title": "SegAffordance",
                           "version": "0.1.0"}
        })
        self._notify("initialized", {})
        params = {}
        if self.model:
            params["model"] = self.model
        if self.effort:
            params["effort"] = self.effort
        if self.cwd:
            params["cwd"] = self.cwd
        self._thread_params = params
        self.last_usage: dict | None = None
        self.new_thread()

    def new_thread(self):
        """Start a fresh conversation thread (persistent server process).

        Use between describe() calls to avoid context accumulating across
        independent requests."""
        params = dict(self._thread_params)
        try:
            res = self._request("thread/start", params)
        except CodexError:
            # Retry without `effort` only if the server is alive and merely
            # rejected the param — never on a dead process.
            if "effort" not in params or self._proc.poll() is not None:
                raise
            # Older app-server builds reject the effort param; fall back to
            # the config.toml default.
            params.pop("effort")
            self._thread_params = params
            res = self._request("thread/start", params)
        self.thread_id = res["thread"]["id"]

    # -- transport ---------------------------------------------------------
    def _read_loop(self):
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._msgs.put(json.loads(line))
            except json.JSONDecodeError:
                pass
        self._msgs.put(None)  # EOF

    def _send(self, obj: dict):
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise CodexError(f"app-server pipe closed: {e}")

    def _notify(self, method: str, params: dict):
        self._send({"method": method, "params": params})

    def _next_msg(self):
        try:
            msg = self._msgs.get(timeout=self.timeout)
        except queue.Empty:
            raise CodexError(f"timed out after {self.timeout}s waiting for app-server")
        if msg is None:
            raise CodexError("app-server exited unexpectedly")
        return msg

    def _request(self, method: str, params: dict) -> dict:
        self._id += 1
        rid = self._id
        self._send({"method": method, "id": rid, "params": params})
        while True:
            msg = self._next_msg()
            if msg.get("id") == rid and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise CodexError(f"{method}: {msg['error']}")
                return msg["result"]
            self._handle_async(msg)

    def _handle_async(self, msg: dict):
        # Server->client requests (e.g. command approvals) must be answered
        # or the turn hangs. We never approve anything: this client is for
        # text/vision generation only.
        if "id" in msg and "method" in msg:
            if "pproval" in msg["method"]:
                self._send({"id": msg["id"], "result": {"decision": "denied"}})
            else:
                self._send({"id": msg["id"], "result": {}})
        # Notifications are handled by the turn loop in describe(); anything
        # arriving here (between requests) is dropped.

    # -- API ---------------------------------------------------------------
    def describe(self, prompt: str, image: str | None = None,
                 images: list[str] | None = None) -> str:
        """One turn: prompt (+ optional local images) -> final agent text."""
        inp = [{"type": "text", "text": prompt}]
        for path in ([image] if image else []) + (images or []):
            inp.append({"type": "localImage", "path": path})
        self._id += 1
        rid = self._id
        self._send({"method": "turn/start", "id": rid,
                    "params": {"threadId": self.thread_id, "input": inp}})

        last_agent_text = None
        started = False
        while True:
            msg = self._next_msg()
            if msg.get("id") == rid and "error" in msg:
                raise CodexError(f"turn/start: {msg['error']}")
            if msg.get("id") == rid and "result" in msg and not started:
                started = True  # accepted; keep streaming notifications
                continue
            method = msg.get("method", "")
            params = msg.get("params", {})
            if method == "thread/tokenUsage/updated":
                self.last_usage = params.get("tokenUsage", {}).get("last")
                continue
            if method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    last_agent_text = item.get("text")
            elif method == "turn/completed":
                status = params.get("turn", {}).get("status")
                if status not in (None, "completed"):
                    raise CodexError(f"turn ended with status={status!r}")
                if last_agent_text is None:
                    raise CodexError("turn completed without an agentMessage")
                return last_agent_text
            elif method == "turn/failed":
                raise CodexError(f"turn failed: {params}")
            else:
                self._handle_async(msg)

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    with CodexClient(model=args.model) as c:
        print(c.describe(args.prompt, image=args.image))


if __name__ == "__main__":
    main()
