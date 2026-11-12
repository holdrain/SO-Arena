"""Bounded process owner for the existing JevAny27B/Kev4B Choice bootstrap.

The worker stays in operations/multimodel_choice/worker_bootstrap.py. This
small adapter only owns its JSONL child process, applies the same bounded ready
handshake as Base151 v1, and gives DeadlineChoiceWorker the runner's conversion
exception class. It is reusable by any compatible one-question Choice driver.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any

ROOT = Path("/__so_arena_runtime__")
MODEL_PYTHON = ROOT / "evaluation/venv/bin/python"
BOOTSTRAP = Path(__file__).resolve().parent / "worker_bootstrap_gpu_override.py"
SUPPORTED_MODELS = {"jevany27b", "kev4b"}
DEFAULT_READY_TIMEOUT_SECONDS = 120.0
# DeadlineChoiceWorker discovers this by inspecting OwnedChoiceProcess.__init__ globals.
UnsupportedConversion: type[Exception] = RuntimeError


class OwnedChoiceProcess:
    """The attributes DeadlineChoiceWorker needs from a ChoiceWorker owner."""
    def __init__(self, process: subprocess.Popen[str], ready: dict[str, Any],
                 stderr_stream: Any, stderr_path: Path, model: str) -> None:
        self.process = process
        self.ready = ready
        self.stderr_stream = stderr_stream
        self.stderr_path = stderr_path
        self.model = model


def _stop_and_close(process: subprocess.Popen[str], stderr_stream: Any) -> None:
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        except ProcessLookupError:
            pass
    for stream in (process.stdin, process.stdout, process.stderr, stderr_stream):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass


def start_multimodel_choice_worker(
    model: str,
    out: Path,
    *,
    enable_model: bool,
    max_context: int = 8192,
    unsupported_conversion_type: type[Exception] = RuntimeError,
    ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
    gpu_cards: list[tuple[str, str]] | None = None,
) -> OwnedChoiceProcess:
    """Start the existing model bootstrap with a bounded, validated ready line."""
    global UnsupportedConversion
    UnsupportedConversion = unsupported_conversion_type
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported multi-model backend: {model!r}")
    if max_context != 8192:
        raise ValueError("this frozen panel supports strict 8192-token context only")
    if ready_timeout_seconds <= 0:
        raise ValueError("ready timeout must be positive")
    if not MODEL_PYTHON.is_file() or not BOOTSTRAP.is_file():
        raise FileNotFoundError("evaluation worker Python or model bootstrap is missing")

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stderr_path = out / "choice_worker.stderr.log"
    stderr_stream = stderr_path.open("w", encoding="utf-8")
    command = [
        str(MODEL_PYTHON), str(BOOTSTRAP), "--model", model,
        "--out", str(out), "--max-context", str(max_context),
    ]
    if enable_model:
        command.append("--enable-model")
    env = os.environ.copy()
    if not enable_model:
        # Tokenizer-only gates are explicitly CPU-side and never acquire a GPU.
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif model == "jevany27b":
        if gpu_cards is None or len(gpu_cards) != 2:
            raise ValueError("JevAny27B model mode requires the two coordinator-approved PCI/UUID pairs")
        expected_uuids = [uuid for _, uuid in gpu_cards]
        if env.get("CUDA_VISIBLE_DEVICES", "").split(",") != expected_uuids:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must match the explicit JevAny27B PCI/UUID order")
        for pci, uuid in gpu_cards:
            command.extend(["--gpu-card", f"{pci}={uuid}"])
    elif gpu_cards is not None:
        raise ValueError("GPU card overrides are supported only for JevAny27B")
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr_stream, text=True, encoding="utf-8", bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("multi-model Choice worker did not create protocol pipes")
        stdout_fd = process.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        deadline = time.monotonic() + ready_timeout_seconds
        buffer = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(stdout_fd, selectors.EVENT_READ)
            while b"\n" not in buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"multi-model Choice worker ready handshake exceeded {ready_timeout_seconds}s")
                if not selector.select(remaining):
                    raise TimeoutError(f"multi-model Choice worker ready handshake exceeded {ready_timeout_seconds}s")
                chunk = os.read(stdout_fd, 65536)
                if not chunk:
                    raise RuntimeError(f"multi-model worker exited before ready (status={process.poll()}); see {stderr_path}")
                buffer.extend(chunk)
        line, _, trailing = buffer.partition(b"\n")
        if trailing:
            raise RuntimeError("multi-model Choice worker emitted bytes after ready before a request")
        ready = json.loads(line.decode("utf-8"))
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            raise RuntimeError(f"multi-model Choice worker failed its ready handshake: {ready!r}")
        if ready.get("model") != model or ready.get("context_limit_tokens") != max_context:
            raise RuntimeError(f"worker ready metadata mismatches requested model/context: {ready!r}")
        if ready.get("model_loaded") is not bool(enable_model):
            raise RuntimeError(f"worker model_loaded mismatches requested mode: {ready!r}")
        if enable_model and not isinstance(ready.get("gpu_guard"), dict):
            raise RuntimeError("model-enabled worker did not return its live GPU ownership guard")
        if not isinstance(ready.get("effective_option_limit"), int) or ready["effective_option_limit"] < 1:
            raise RuntimeError("worker ready metadata lacks a positive effective option limit")
        if not isinstance(ready.get("native_api_option_limit"), int) or ready["native_api_option_limit"] < 1:
            raise RuntimeError("worker ready metadata lacks a positive native API option limit")
        return OwnedChoiceProcess(process, ready, stderr_stream, stderr_path, model)
    except Exception:
        if process is not None:
            _stop_and_close(process, stderr_stream)
        elif not stderr_stream.closed:
            stderr_stream.close()
        raise
