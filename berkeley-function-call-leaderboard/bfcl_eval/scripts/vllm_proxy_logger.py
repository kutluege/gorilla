"""
Simple OpenAI-compatible proxy logger for vLLM.

It logs incoming request payloads and forwards them upstream unchanged.
Useful for tracing exactly what is sent to /v1/chat/completions.

Example:
python bfcl_eval/scripts/vllm_proxy_logger.py --listen-port 18000 --upstream http://127.0.0.1:8000 --log-file tmp/vllm_proxy.log.jsonl
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _make_handler(upstream: str, log_file: Path, max_body_chars: int):
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _log(self, payload: dict):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def _forward(self):
            start = time.time()
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
            body_text = body_bytes.decode("utf-8", errors="replace")
            body_json = _safe_json_loads(body_text)

            upstream_url = f"{upstream.rstrip('/')}{self.path}"
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in {"host", "content-length", "connection"}
            }

            self._log(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "request",
                    "method": self.command,
                    "path": self.path,
                    "headers": headers,
                    "body_json": body_json,
                    "body_raw": body_text if body_json is None else None,
                }
            )

            req = urllib.request.Request(
                url=upstream_url,
                data=body_bytes if self.command in {"POST", "PUT", "PATCH"} else None,
                headers=headers,
                method=self.command,
            )

            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    resp_body = resp.read()
                    resp_text = resp_body.decode("utf-8", errors="replace")
                    resp_json = _safe_json_loads(resp_text)
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() in {"transfer-encoding", "connection"}:
                            continue
                        if k.lower() == "content-length":
                            continue
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    self.wfile.flush()
                    self._log(
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "event": "response",
                            "method": self.command,
                            "path": self.path,
                            "status_code": resp.status,
                            "response_json": resp_json,
                            "response_raw": (
                                resp_text[:max_body_chars]
                                if resp_json is None
                                else None
                            ),
                            "response_truncated": len(resp_text) > max_body_chars,
                        }
                    )
            except urllib.error.HTTPError as e:
                err_body = e.read()
                err_text = err_body.decode("utf-8", errors="replace")
                err_json = _safe_json_loads(err_text)
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() in {"transfer-encoding", "connection", "content-length"}:
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                self.wfile.flush()
                self._log(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "response_error",
                        "method": self.command,
                        "path": self.path,
                        "status_code": e.code,
                        "response_json": err_json,
                        "response_raw": (
                            err_text[:max_body_chars] if err_json is None else None
                        ),
                        "response_truncated": len(err_text) > max_body_chars,
                    }
                )
            except Exception as e:
                msg = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                self.wfile.flush()
                self._log(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "event": "proxy_exception",
                        "method": self.command,
                        "path": self.path,
                        "error": str(e),
                    }
                )
            finally:
                elapsed_ms = int((time.time() - start) * 1000)
                self._log(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "method": self.command,
                        "path": self.path,
                        "event": "response_done",
                        "latency_ms": elapsed_ms,
                    }
                )

        def do_POST(self):
            self._forward()

        def do_GET(self):
            self._forward()

        def do_PUT(self):
            self._forward()

        def do_PATCH(self):
            self._forward()

        def log_message(self, fmt, *args):
            # Keep stdout clean; all structured logs go to log_file.
            return

    return ProxyHandler


def main():
    parser = argparse.ArgumentParser(description="vLLM proxy logger")
    parser.add_argument(
        "--listen-host",
        default="127.0.0.1",
        help="Host to bind proxy server.",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=18000,
        help="Port to bind proxy server.",
    )
    parser.add_argument(
        "--upstream",
        default="http://127.0.0.1:8000",
        help="Upstream vLLM/OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--log-file",
        default="tmp/vllm_proxy.log.jsonl",
        help="JSONL log file path.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=20000,
        help="Max characters to keep for raw response text when not JSON.",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    handler_cls = _make_handler(args.upstream, log_file, args.max_body_chars)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler_cls)

    print(
        f"Proxy listening on http://{args.listen_host}:{args.listen_port} "
        f"-> {args.upstream}; logging to {log_file}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
