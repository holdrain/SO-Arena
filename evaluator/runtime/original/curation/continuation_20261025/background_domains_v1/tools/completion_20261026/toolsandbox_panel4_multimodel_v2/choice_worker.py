"""Model-selectable isolated worker for the pinned native Choice protocol.

This versioned copy reuses the established parent/child JSONL protocol and
selects JevAny-4B, JevAny-27B, or Kev-4B out of band. It is separate from the
canonical shared worker and loads no task-specific planning logic.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
from pathlib import Path
import time
from typing import Any

ROOT = Path("/__so_arena_runtime__")
MODEL_PYTHON = ROOT / "evaluation/venv/bin/python"
BOOTSTRAP = Path(__file__).resolve().with_name("worker_bootstrap.py")


class MultiModelChoiceWorker:
    def __init__(self, out: Path, *, model: str, enable_model: bool, max_context: int,
                 startup_timeout_seconds: float = 300.0) -> None:
        if model not in {"jevany4b", "jevany27b", "kev4b"}:
            raise ValueError(f"unsupported model {model!r}")
        if not MODEL_PYTHON.is_file():
            raise FileNotFoundError(f"evaluation worker Python is missing: {MODEL_PYTHON}")
        self.stderr_path = out / "choice_worker.stderr.log"
        self.stderr_stream = self.stderr_path.open("w", encoding="utf-8")
        command = [str(MODEL_PYTHON), str(BOOTSTRAP), "--model", model, "--out", str(out.resolve()), "--max-context", str(max_context)]
        if enable_model:
            command.append("--enable-model")
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr_stream, text=False, bufsize=0)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self._read_buffer = bytearray()
        try:
            self.ready = self._read_response(timeout_seconds=self.startup_timeout_seconds)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise
        if self.ready.get("status") != "ready" or self.ready.get("model") != model:
            self.close()
            raise RuntimeError(f"multimodel Choice worker did not become ready for {model}: {self.ready}")

    def _read_response(self, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        assert self.process.stdout is not None
        fd = self.process.stdout.fileno()
        os.set_blocking(fd, False)
        deadline = time.monotonic() + timeout_seconds
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while b"\n" not in self._read_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(f"Choice worker did not respond within {timeout_seconds}s; see {self.stderr_path}")
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise RuntimeError(f"Choice worker exited before responding (status={self.process.poll()}); see {self.stderr_path}")
                self._read_buffer.extend(chunk)
        line, remainder = self._read_buffer.split(b"\n", 1)
        self._read_buffer = bytearray(remainder)
        if self._read_buffer:
            raise RuntimeError("Choice worker emitted unsolicited bytes beyond one JSONL response")
        response = json.loads(line.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("Choice worker returned a non-object response")
        if response.get("status") in {"error", "unsupported_conversion"}:
            raise RuntimeError(f"Choice worker failed: {response.get('error')}")
        return response

    def _request(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"Choice worker exited with status {self.process.returncode}; see {self.stderr_path}")
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.process.stdin.flush()
        response = self._read_response(timeout_seconds=30.0)
        if response.get("status") != "ok":
            raise RuntimeError(f"Choice worker failed: {response.get('error')}")
        return response

    def measure(self, request: dict[str, Any]) -> dict[str, int]:
        return self._request({"op": "measure", "request": request})["token_counts"]

    def gold_flip(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request({"op": "gold_flip", "request": request})

    def begin_episode(self, episode_key: str) -> None:
        self._request({"op": "begin_episode", "episode_key": episode_key})

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request({"op": "choose", "request": request})["answer"]

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(b'{"op":"shutdown"}\n')
                process.stdin.flush()
            except Exception:
                process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if getattr(self, "stderr_stream", None) is not None and not self.stderr_stream.closed:
            self.stderr_stream.close()
