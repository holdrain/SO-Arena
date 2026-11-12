"""Per-request deadline transport for one already-owned ChoiceWorker child.

Wrap this around an instantiated shared ChoiceWorker after its ready message.
All later JSONL I/O goes through this transport. A timed-out child is terminated
and reaped by PID; the wrapper never reloads, retries, or kills a process group.
"""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any


class WorkerDeadlineExceeded(TimeoutError):
    def __init__(self, reason: str, *, elapsed_seconds: float, child_pid: int) -> None:
        self.reason = reason
        self.elapsed_seconds = elapsed_seconds
        self.child_pid = child_pid
        super().__init__(f"Choice worker {reason} after {elapsed_seconds:.3f}s; owned child pid={child_pid} was stopped")


class WorkerUnavailable(RuntimeError):
    pass


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class DeadlineJsonlChild:
    """Deadline-bounded one-line JSON protocol over a subprocess it owns."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        timeout_seconds: float = 30.0,
        total_deadline_monotonic: float | None = None,
        error_log: Path,
        extra_owned_streams: tuple[Any, ...] = (),
        terminate_grace_seconds: float = 0.25,
        reap_grace_seconds: float = 1.0,
    ) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("owned child must have stdin/stdout pipes")
        if timeout_seconds <= 0 or terminate_grace_seconds < 0 or reap_grace_seconds <= 0:
            raise ValueError("deadline and cleanup bounds must be positive")
        self.process = process
        self.timeout_seconds = float(timeout_seconds)
        self.total_deadline_monotonic = total_deadline_monotonic
        self.error_log = error_log
        self.extra_owned_streams = extra_owned_streams
        self.terminate_grace_seconds = float(terminate_grace_seconds)
        self.reap_grace_seconds = float(reap_grace_seconds)
        self.stdin_fd = process.stdin.fileno()
        self.stdout_fd = process.stdout.fileno()
        os.set_blocking(self.stdin_fd, False)
        os.set_blocking(self.stdout_fd, False)
        self._read_buffer = bytearray()
        self.failed = False
        self.closed = False
        self.active_handlers = 0
        self.timeout_records: list[dict[str, Any]] = []

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def _append_error(self, record: dict[str, Any]) -> None:
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        with self.error_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.timeout_records.append(record)

    def _stop_owned_child(self) -> None:
        """Stop exactly the Popen child and reap it; never use killpg."""
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=self.reap_grace_seconds)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(f"could not reap owned Choice child pid={self.pid} after kill") from error
        self._close_pipes()

    def _close_pipes(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr, *self.extra_owned_streams):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _deadline(self, start: float) -> tuple[float, str]:
        request_deadline = start + self.timeout_seconds
        if self.total_deadline_monotonic is None:
            return request_deadline, "request_deadline"
        if self.total_deadline_monotonic <= request_deadline:
            return self.total_deadline_monotonic, "total_deadline"
        return request_deadline, "request_deadline"

    def _write_until(self, payload: memoryview, deadline: float, selector: selectors.BaseSelector) -> None:
        selector.register(self.stdin_fd, selectors.EVENT_WRITE)
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline expired while writing request")
            try:
                count = os.write(self.stdin_fd, payload[offset:])
            except BlockingIOError:
                if not selector.select(remaining):
                    raise TimeoutError("deadline expired while waiting to write request")
                continue
            except BrokenPipeError as error:
                raise WorkerUnavailable(f"owned worker pid={self.pid} closed its request pipe") from error
            if count <= 0:
                raise WorkerUnavailable(f"owned worker pid={self.pid} accepted zero request bytes")
            offset += count
            if offset < len(payload):
                # The pipe may accept another chunk immediately; the loop is still deadline-checked.
                continue
        selector.unregister(self.stdin_fd)

    def _read_line_until(self, deadline: float, selector: selectors.BaseSelector) -> bytes:
        selector.register(self.stdout_fd, selectors.EVENT_READ)
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._read_buffer[:newline])
                del self._read_buffer[: newline + 1]
                if time.monotonic() >= deadline:
                    raise TimeoutError("response arrived at or after deadline")
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("deadline expired while waiting for response")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError("deadline expired while waiting for response")
            try:
                chunk = os.read(self.stdout_fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                status = self.process.poll()
                raise WorkerUnavailable(f"owned worker pid={self.pid} closed response pipe (status={status})")
            self._read_buffer.extend(chunk)
        # Unreachable.

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.closed or self.failed or self.process.poll() is not None:
            raise WorkerUnavailable(f"owned worker pid={self.pid} is no longer available; retry/reload is disabled")
        if b"\n" in self._read_buffer:
            self._stop_owned_child()
            self.failed = True
            raise WorkerUnavailable("unsolicited or misaligned JSONL response before request")
        start = time.monotonic()
        deadline, deadline_reason = self._deadline(start)
        if deadline <= start:
            elapsed = time.monotonic() - start
            self._append_error({
                "status": "technical_timeout",
                "reason": "total_deadline_before_request",
                "request_sent": False,
                "pid": self.pid,
                "request_sha256": canonical_sha(message),
                "message": message,
                "elapsed_seconds": elapsed,
            })
            self.failed = True
            self._stop_owned_child()
            raise WorkerDeadlineExceeded("total_deadline", elapsed_seconds=elapsed, child_pid=self.pid)

        self.active_handlers += 1
        try:
            payload = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            with selectors.DefaultSelector() as selector:
                try:
                    self._write_until(memoryview(payload), deadline, selector)
                    line = self._read_line_until(deadline, selector)
                except TimeoutError as error:
                    elapsed = time.monotonic() - start
                    self._append_error({
                        "status": "technical_timeout",
                        "reason": deadline_reason,
                        "request_sent": True,
                        "pid": self.pid,
                        "request_sha256": canonical_sha(message),
                        "message": message,
                        "elapsed_seconds": elapsed,
                        "deadline_seconds": max(0.0, deadline - start),
                        "detail": str(error),
                    })
                    self.failed = True
                    self._stop_owned_child()
                    raise WorkerDeadlineExceeded(deadline_reason, elapsed_seconds=elapsed, child_pid=self.pid) from error
                except Exception:
                    self.failed = True
                    self._stop_owned_child()
                    raise
            try:
                response = json.loads(line.decode("utf-8"))
            except Exception as error:
                self.failed = True
                self._stop_owned_child()
                raise WorkerUnavailable(f"owned worker pid={self.pid} returned invalid JSONL") from error
            if not isinstance(response, dict):
                self.failed = True
                self._stop_owned_child()
                raise WorkerUnavailable(f"owned worker pid={self.pid} returned a non-object response")
            return response
        finally:
            self.active_handlers -= 1

    def close(self) -> None:
        if self.closed:
            return
        try:
            if not self.failed and self.process.poll() is None:
                try:
                    self.request({"op": "shutdown"})
                except Exception:
                    self.failed = True
            if self.process.poll() is None and not self.failed:
                try:
                    self.process.wait(timeout=min(1.0, max(self.timeout_seconds, 0.05)))
                except subprocess.TimeoutExpired:
                    self._stop_owned_child()
            elif self.process.poll() is None:
                self._stop_owned_child()
            else:
                self._close_pipes()
        finally:
            self._close_pipes()
            self.closed = True


class DeadlineChoiceWorker:
    """Drop-in proxy for the shared ChoiceWorker after its ready handshake."""

    scripted_test = False

    def __init__(
        self,
        choice_worker: Any,
        *,
        error_log: Path,
        timeout_seconds: float = 30.0,
        total_deadline_monotonic: float | None = None,
    ) -> None:
        self.worker = choice_worker
        self.ready = choice_worker.ready
        worker_globals = getattr(choice_worker.__class__.__init__, "__globals__", {})
        self._unsupported_conversion_type = worker_globals.get("UnsupportedConversion", RuntimeError)
        self.transport = DeadlineJsonlChild(
            choice_worker.process,
            timeout_seconds=timeout_seconds,
            total_deadline_monotonic=total_deadline_monotonic,
            error_log=error_log,
            extra_owned_streams=(getattr(choice_worker, "stderr_stream", None),),
        )
        self._stderr_stream = getattr(choice_worker, "stderr_stream", None)

    def begin_episode(self, episode_key: str) -> None:
        self.transport.request({"op": "begin_episode", "episode_key": episode_key})

    def measure(self, request: dict[str, Any]) -> dict[str, int]:
        response = self.transport.request({"op": "measure", "request": request})
        if response.get("status") == "unsupported_conversion":
            raise self._unsupported_conversion_type(str(response.get("error", "strict codec rejected request")))
        if response.get("status") == "error":
            raise RuntimeError(f"Choice worker failed: {response.get('error')}")
        if "token_counts" not in response:
            raise WorkerUnavailable("Choice worker measure response lacks token_counts")
        return response["token_counts"]

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.transport.request({"op": "choose", "request": request})
        if response.get("status") == "unsupported_conversion":
            raise self._unsupported_conversion_type(str(response.get("error", "strict codec rejected request")))
        if response.get("status") == "error":
            raise RuntimeError(f"Choice worker failed: {response.get('error')}")
        if "answer" not in response:
            raise WorkerUnavailable("Choice worker prediction response lacks answer")
        return response["answer"]

    def close(self) -> None:
        try:
            self.transport.close()
        finally:
            if self._stderr_stream is not None and not self._stderr_stream.closed:
                self._stderr_stream.close()
