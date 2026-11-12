"""KnowLine-31B-Gen1: the documented two-process serving path, "equivalent manual launch, with the exact flags of
our run" (model card INFERENCE.md:106-114): SGLang 0.5.21 `sglang.launch_server --served-model-name m --tp 1
--mem-fraction-static 0.85 --enable-fp32-lm-head` (BF16; no FP8: "We did not test FP8 at load", :83-84), and the
front end `knowline_server.py --backend sglang --temperature 1 --workers 16 --temperatures temperatures.json
--box-prompt gemma --box-coords yxyx`, read twice on by default (:50-61). SGLang's prefix cache is flushed after every
request, outside the clock. backend "hf" = the documented fallback (:116-127): the front end alone, transformers.
Per-card ports. PYTHONDONTWRITEBYTECODE keeps the front end from writing __pycache__ into the model folder.
Environment: JEV_CARD (physical GPU index, equal to CUDA_VISIBLE_DEVICES), JEV_CUDA13_COMPAT (on a host driver older
than CUDA 13: the user-space forward-compatibility library, put first on LD_LIBRARY_PATH, as in the paper's runs),
JEV_CACHE (cache root, default ~/.cache/so-arena), JEV_LOG_DIR (log folder, default under TMPDIR)."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

CACHE = os.environ.get("JEV_CACHE") or os.path.expanduser("~/.cache/so-arena")


def _wait(url, proc, what, log, limit=3600):
    t = time.perf_counter()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited ({proc.returncode}); see {log}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return round(time.perf_counter() - t, 1)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.perf_counter() - t > limit:
            raise TimeoutError(f"{what} not ready after {limit} s")
        time.sleep(3)


def _cuda_home(pip_cuda):
    """A CUDA_HOME for SGLang's JIT builds made of symlinks to the pip CUDA 13 toolkit: the JIT links with
    -L<CUDA_HOME>/lib64 -lcudart, but the pip wheels ship lib/ and only versioned names (libcudart.so.13)."""
    home = os.path.join(CACHE, "cuda13_home")
    os.makedirs(os.path.join(home, "lib64"), exist_ok=True)
    for d in ("bin", "include", "nvvm", "cccl"):
        if os.path.exists(os.path.join(pip_cuda, d)) and not os.path.lexists(os.path.join(home, d)):
            os.symlink(os.path.join(pip_cuda, d), os.path.join(home, d))
    lib = os.path.join(pip_cuda, "lib")
    for fn in os.listdir(lib):
        dst = os.path.join(home, "lib64", fn)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(lib, fn), dst)
        if ".so." in fn:
            plain = os.path.join(home, "lib64", fn.split(".so.")[0] + ".so")
            if not os.path.lexists(plain):
                os.symlink(os.path.join(lib, fn), plain)
    return home


class Adapter:
    def __init__(self, config):
        card = int(os.environ["JEV_CARD"])
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(card)
        self.backend = config.get("backend", "sglang")
        folder = config["model"]
        log_dir = os.environ.get("JEV_LOG_DIR") or os.path.join(tempfile.gettempdir(), f"knowline_{card}")
        os.makedirs(log_dir, exist_ok=True)
        self.engine_port, self.front_port = 18600 + 10 * card, 18601 + 10 * card
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        self.procs, self.logs, self.info = [], [], {"backend": self.backend}
        if self.backend == "sglang":
            if os.environ.get("JEV_CUDA13_COMPAT"):
                env["LD_LIBRARY_PATH"] = os.environ["JEV_CUDA13_COMPAT"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            # SGLang's DeepGEMM JIT needs nvcc >= 12.9 (deep_jit backend.hpp:72); the env carries CUDA 13's own nvcc
            # (nvidia-cuda-nvcc, a dependency of sglang 0.5.21), so CUDA_HOME points there instead of the system 12.6
            import sysconfig
            env["CUDA_HOME"] = _cuda_home(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))
            env["PATH"] = os.path.join(env["CUDA_HOME"], "bin") + ":" + env["PATH"]
            # SGLang's JIT cache goes to ~/.cache/sglang regardless of XDG_CACHE_HOME: keep it off the root disk
            env["HOME"] = os.path.join(CACHE, "home-knowline")
            os.makedirs(env["HOME"], exist_ok=True)
            eng = [sys.executable, "-m", "sglang.launch_server", "--model-path", folder, "--served-model-name", "m",
                   "--tp", "1", "--mem-fraction-static", "0.85", "--enable-fp32-lm-head",
                   "--host", "127.0.0.1", "--port", str(self.engine_port)]
            self._start(eng, env, os.path.join(log_dir, "sglang.log"))
            self.info["engine_cmd"] = eng
            self.info["engine_startup_s"] = _wait(f"http://127.0.0.1:{self.engine_port}/health", self.procs[-1],
                                                  "sglang", log_dir)
            front = [sys.executable, os.path.join(folder, "knowline_server.py"), "--model", folder, "--backend", "sglang",
                     "--url", f"http://127.0.0.1:{self.engine_port}", "--served-model-name", "m", "--temperature", "1",
                     "--workers", "16", "--port", str(self.front_port),
                     "--temperatures", os.path.join(folder, "temperatures.json"),
                     "--box-prompt", "gemma", "--box-coords", "yxyx"]
        else:
            front = [sys.executable, os.path.join(folder, "knowline_server.py"), "--model", folder, "--backend", "hf",
                     "--port", str(self.front_port), "--temperatures", os.path.join(folder, "temperatures.json")]
        self._start(front, env, os.path.join(log_dir, "front.log"))
        self.info["front_cmd"] = front
        self.info["front_startup_s"] = _wait(f"http://127.0.0.1:{self.front_port}/health", self.procs[-1], "front end",
                                             log_dir)

    def _start(self, cmd, env, log):
        f = open(log, "w")
        self.logs.append(f)
        self.procs.append(subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))

    def _post(self, request):
        body = dict(request, model="m")
        req = urllib.request.Request(f"http://127.0.0.1:{self.front_port}/v1/systemone",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    def predict(self, request):
        res = self._post(request)
        (name, q), = request["questions"].items()
        return {"p": [float(res["answers"][name]["probabilities"][k]) for k in q["criteria"]], "usage": res.get("usage")}

    def predict_noul(self, request):
        res = self._post(request)
        (name, q), = request["questions"].items()
        return {"p_true": float(res["answers"][name]["noul"])}

    def after_request(self):
        if self.backend == "sglang":
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{self.engine_port}/flush_cache",
                                                              data=b"", method="POST"), timeout=60).read()
            except Exception:  # noqa: BLE001  best effort; a failure shows in the engine log
                pass

    def close(self):
        for p in reversed(self.procs):
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=120)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for f in self.logs:
            f.close()
