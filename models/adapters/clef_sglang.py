"""Clef-27B / Clef-flash-9B: the card's documented serving path, SGLang `sglang serve --model-path <folder>` in the
image lmsysorg/sglang:dev-clef (model card README.md:93-117), answering POST /v1/systemone. The image runs under
unprivileged Apptainer (--nv, --cleanenv). On a host driver older than CUDA 13 the image's CUDA 13 runs through the
user-space forward-compatibility library, given in JEV_CUDA13_COMPAT and put first on LD_LIBRARY_PATH (as in the
paper's runs). SGLang's prefix (radix) cache is flushed after every request, outside the clock (every state is
distinct). One server per card, per-card ports.
Environment: JEV_CARD (physical GPU index, equal to CUDA_VISIBLE_DEVICES), CLEF_SGLANG_SIF (the Apptainer image built
from lmsysorg/sglang:dev-clef), APPTAINER (the apptainer executable, default "apptainer"), JEV_CACHE (cache root,
default ~/.cache/so-arena), JEV_APPTAINER_BIND (bind list, default: the model folder and every cache folder passed in,
each at its own path), JEV_LOG_DIR (server log folder, default under TMPDIR)."""
import json
import os
import signal
import subprocess
import time
import urllib.error
import tempfile
import urllib.request

APPTAINER = os.environ.get("APPTAINER", "apptainer")
CACHE = os.environ.get("JEV_CACHE") or os.path.expanduser("~/.cache/so-arena")
CACHE_VARS = ("HF_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")


class Adapter:
    def __init__(self, config):
        card = int(os.environ["JEV_CARD"])
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(card)
        self.port = 18400 + 10 * card
        self.name = config["served_model"]
        log_dir = os.environ.get("JEV_LOG_DIR") or os.path.join(tempfile.gettempdir(), f"clef_{card}")
        os.makedirs(log_dir, exist_ok=True)
        home = os.path.join(CACHE, "home-clef")
        os.makedirs(home, exist_ok=True)
        env = dict(os.environ)
        env.update(APPTAINERENV_CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                   APPTAINER_CACHEDIR=os.path.join(CACHE, "apptainer", "cache"),
                   APPTAINER_TMPDIR=os.path.join(CACHE, "apptainer", "tmp"))
        if os.environ.get("JEV_CUDA13_COMPAT"):
            env["APPTAINERENV_LD_LIBRARY_PATH"] = os.environ["JEV_CUDA13_COMPAT"]
        caches = [os.environ[k] for k in CACHE_VARS if os.environ.get(k)]
        for k in CACHE_VARS:
            if os.environ.get(k):
                env[f"APPTAINERENV_{k}"] = os.environ[k]
        bind = os.environ.get("JEV_APPTAINER_BIND") or ",".join(
            f"{d}:{d}" for d in dict.fromkeys([config["model"], CACHE] + caches))
        # --cleanenv: the host's CUDA_HOME / NVCC_CCBIN / CXX (system CUDA 12.6, g++-11) must not leak into the image,
        # which brings its own CUDA 13 toolkit; only the variables above are passed in
        cmd = [APPTAINER, "exec", "--nv", "--cleanenv", "--home", home, "--bind", bind, os.environ["CLEF_SGLANG_SIF"],
               "sglang", "serve", "--model-path", config["model"], "--served-model-name", self.name,
               "--host", "127.0.0.1", "--port", str(self.port)]
        self.log = open(os.path.join(log_dir, "sglang.log"), "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.url = f"http://127.0.0.1:{self.port}"
        t = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"sglang exited ({self.proc.returncode}); see {log_dir}/sglang.log")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.perf_counter() - t > 3600:
                raise TimeoutError("SGLang not ready after 60 min")
            time.sleep(3)
        self.info = {"cmd": cmd, "startup_s": round(time.perf_counter() - t, 1), "log_dir": log_dir,
                     "compat": os.environ.get("JEV_CUDA13_COMPAT")}

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    def predict(self, request):
        res = self._post("/v1/systemone", dict(request, model=self.name))
        (name, q), = request["questions"].items()
        return {"p": [float(res["answers"][name]["probabilities"][k]) for k in q["criteria"]], "usage": res.get("usage")}

    def predict_noul(self, request):
        res = self._post("/v1/systemone", dict(request, model=self.name))
        (name, q), = request["questions"].items()
        return {"p_true": float(res["answers"][name]["noul"])}

    def after_request(self):
        try:
            urllib.request.urlopen(urllib.request.Request(self.url + "/flush_cache", data=b"", method="POST"), timeout=60).read()
        except Exception:  # noqa: BLE001  flushing is best effort; a failure is visible in the server log
            pass

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
