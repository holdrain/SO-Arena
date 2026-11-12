"""StartLux v1.1 4B Instant / AutoThink: the documented server, `startlux-decide serve --model DIR --launch-vllm
--gpus <card> --no-images --max-model-len 16384 --quantization fp8` (model card README.md:62-66;
StartLux-Decision docs/quantization.md:11-14), on vLLM 0.19.1 with the bidirectional plugin. The launcher's release
flags turn prefix caching and chunked prefill off on 0.19.1 (startlux_decide/launch.py:26-28). AutoThink in auto mode
adds `--think-mode auto --gate 0.6` and the same in every request (config startlux-4b-autothink-auto). One request at
a time over HTTP to /v1/systemone (server.py:4).
Environment: JEV_CARD (physical GPU index, equal to CUDA_VISIBLE_DEVICES; the launcher sets CUDA_VISIBLE_DEVICES for
its vLLM server itself, launch.py:139), JEV_PORT_SLOT / JEV_COPIES_PER_CARD (several servers on one card, reliability
runs only: each gets its share of the launcher's default --gpu-memory-utilization 0.88, cli.py:38, which only sizes
its KV cache), JEV_LOG_DIR (log folder, default under TMPDIR)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request


class Adapter:
    def __init__(self, config):
        card = int(os.environ.get("JEV_CARD", "0"))
        slot = int(os.environ.get("JEV_PORT_SLOT", "0"))
        copies = int(os.environ.get("JEV_COPIES_PER_CARD", "1"))
        assert 0 <= slot < 5, "port slots 0-4 (18000 + 10 * card + 2 * slot)"
        self.port = 18000 + 10 * card + 2 * slot
        log_dir = os.environ.get("JEV_LOG_DIR") or os.path.join(tempfile.gettempdir(), f"startlux_{card}")
        os.makedirs(log_dir, exist_ok=True)
        exe = os.path.join(os.path.dirname(sys.executable), "startlux-decide")
        # the launcher sets CUDA_VISIBLE_DEVICES=<--gpus> for each vLLM server itself (startlux_decide/launch.py:139),
        # so --gpus must be the PHYSICAL card, not an index into our CUDA_VISIBLE_DEVICES
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(card), "JEV_CARD must equal CUDA_VISIBLE_DEVICES"
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        cmd = [exe, "serve", "--model", config["model"], "--launch-vllm", "--gpus", str(card), "--no-images",
               "--max-model-len", str(config.get("max_model_len", 16384)), *(["--quantization", config.get("quantization", "fp8")] if config.get("quantization", "fp8") else []),
               "--port", str(self.port),
               "--vllm-base-port", str(self.port + 1), "--vllm-log-dir", log_dir] + config.get("extra_args", [])
        if copies > 1:
            cmd += ["--gpu-memory-utilization", f"{0.88 / copies:.2f}"]
        self.extra = config.get("request_extra", {})
        self.log = open(os.path.join(log_dir, "decision_server.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.port}"
        t = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"StartLux server exited ({self.proc.returncode}); see {log_dir}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    health = json.load(r)
                if health.get("status") in ("ok", "ready") or health.get("ready"):
                    break
            except Exception:  # noqa: BLE001
                pass
            if time.perf_counter() - t > 1800:
                raise TimeoutError("StartLux server not ready after 30 min")
            time.sleep(2)
        self.info = {"cmd": cmd, "health": health, "startup_s": round(time.perf_counter() - t, 1), "log_dir": log_dir}

    def predict(self, request):
        body = dict(request, **self.extra)
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.load(r)
        (name, q), = request["questions"].items()
        a = res["answers"][name]
        out = {"p": [float(a["probabilities"][k]) for k in q["criteria"]], "usage": res.get("usage")}
        for k in ("timings", "jev_context", "passes", "rounds"):
            if k in res:
                out[k] = res[k]
        return out

    def _post(self, request):
        body = dict(request, **self.extra)
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    def predict_noul(self, request):
        (name, q), = request["questions"].items()
        res = self._post(request)
        return {"p_true": float(res["answers"][name]["noul"]), "usage": res.get("usage")}

    def predict_null(self, request):
        """Robustness B, native null: serve's default none_check adds an internal "None of the above" and reports
        none_probability; choice stays a listed option (docs/api.md:55-60)."""
        (name, q), = request["questions"].items()
        res = self._post(request)
        a = res["answers"][name]
        return {"p": [float(a["probabilities"][k]) for k in q["criteria"]], "p_null": float(a["none_probability"]),
                "predicted": a.get("choice"), "usage": res.get("usage")}

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
