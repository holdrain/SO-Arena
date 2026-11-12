"""H2O-Lightning-4B: the card's one shipped command, `bash serve.sh` (model card README.md:164-172,429-437): one
engine on the h2oai vLLM build (0.30.0+cu129 with its per-request-attention overlay), two-way reading, TRITON_ATTN,
prefill CUDA graphs, step graphs at 512, plus the documented opt-in VLLM_ARGS="--quantization fp8", started under
`taskset -c <cores>` on whole physical cores (README.md:690-691). serve.sh warms every path and only then reports
ready; /health answers 503 until then (README.md:171-172,437).
Environment: JEV_CARD (the physical GPU index, equal to CUDA_VISIBLE_DEVICES; it sets the ports), JEV_CACHE (cache
root, default ~/.cache/so-arena; serve.sh keys its compile cache per configuration under JEV_CACHE/home-h2o because
VLLM_CACHE_ROOT is unset, serve.sh:111-122), JEV_LOG_DIR (server log folder, default under TMPDIR), H2O_TASKSET_CORES
(overrides config["taskset_cores"][card]). The vLLM build's bin folder goes first on PATH: config["env_bin"], default
the folder of the running Python. PYTHONDONTWRITEBYTECODE=1 keeps the shim from writing __pycache__ into the model
folder."""
import json
import os
import signal
import subprocess
import time
import urllib.error
import sys
import tempfile
import urllib.request

CACHE = os.environ.get("JEV_CACHE") or os.path.expanduser("~/.cache/so-arena")


class Adapter:
    def __init__(self, config):
        card = int(os.environ["JEV_CARD"])
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(card)
        self.port = 18200 + 10 * card
        cores = os.environ.get("H2O_TASKSET_CORES") or config["taskset_cores"][str(card)]
        log_dir = os.environ.get("JEV_LOG_DIR") or os.path.join(tempfile.gettempdir(), f"h2o_{card}")
        os.makedirs(log_dir, exist_ok=True)
        env = dict(os.environ)
        env.pop("VLLM_CACHE_ROOT", None)
        env.update(HOME=os.path.join(CACHE, "home-h2o"), PYTHONDONTWRITEBYTECODE="1",
                   PATH=config.get("env_bin", os.path.dirname(sys.executable)) + ":" + env["PATH"],
                   PORT=str(self.port), VLLM_PORT=str(self.port + 1), VLLM_ARGS=config["vllm_args"])
        os.makedirs(env["HOME"], exist_ok=True)
        cmd = ["taskset", "-c", cores, "bash", os.path.join(config["model"], "serve.sh")]
        self.log = open(os.path.join(log_dir, "serve.log"), "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=log_dir)
        self.url = f"http://127.0.0.1:{self.port}"
        t = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"serve.sh exited ({self.proc.returncode}); see {log_dir}/serve.log")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        health = json.load(r)
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            if time.perf_counter() - t > 3600:
                raise TimeoutError("H2O server not ready after 60 min")
            time.sleep(2)
        self.info = {"cmd": cmd, "vllm_args": config["vllm_args"], "taskset": cores, "health": health,
                     "startup_s": round(time.perf_counter() - t, 1), "log_dir": log_dir}

    def predict(self, request):
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(request, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.load(r)
        (name, q), = request["questions"].items()
        a = res["answers"][name]
        out = {"p": [float(a["probabilities"][k]) for k in q["criteria"]], "usage": res.get("usage"),
               "attention_mode": res.get("attention_mode")}
        return out

    def predict_noul(self, request):
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(request, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.load(r)
        (name, q), = request["questions"].items()
        return {"p_true": float(res["answers"][name]["noul"]), "attention_mode": res.get("attention_mode")}

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.log.close()
