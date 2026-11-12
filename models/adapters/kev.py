"""Kev 4B / 27B: kev.serve's NVIDIA serving defaults, in process (kev/serve.py:339-350): BF16, the LoRA merged at
load (LoadOptions default), CUDA graphs, fused Qwen3.5 kernels when flash-linear-attention 0.5.2 is installed. The
request goes through kev.serve's own Server (model thread, admission, graph capture when idle) exactly as the HTTP
endpoint sends it, minus HTTP. State-prefix cache off (KEV_PREFIX_CACHE=0): every state is distinct. Probabilities
are read before the response's rounding (kev/api.py:155) via Server.probs, the step answer() wraps."""
import os


class Adapter:
    def __init__(self, config):
        os.environ.update(config["env"])
        from dataclasses import replace

        import torch
        from kev import serve
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import LoadOptions, fused_available

        # the option logic of kev.serve.main(), line for line (serve.py:339-347)
        opts = LoadOptions.from_env()
        if opts.backend is None:
            opts = replace(opts, backend="auto")
        dev, ck = serve._serving_device("cuda", opts, config["run"])
        if dev != "cpu" and opts.dtype is None:
            opts = replace(opts, dtype=torch.bfloat16)
        cuda_optimizations = dev == "cuda" and torch.version.hip is None
        if cuda_optimizations and opts.cuda_graphs is None:
            opts = replace(opts, cuda_graphs=True)
        if cuda_optimizations and opts.fused is None:
            opts = replace(opts, fused=fused_available())
        tok, model = ck.load(dev, opts)
        self.server = serve.Server(ck, tok, model, dev)
        self._req, self._rec, self._prepare = SystemOneRequest, to_record, serve.prepare
        assert serve.PREFIX_CACHE_SIZE == 0, "state-prefix cache must be off"
        self.info = {"device": dev, "backend": model.backend, "dtype": str(model.dtype), "opts": repr(opts),
                     "fused": opts.fused, "cuda_graphs": opts.cuda_graphs, "prefix_cache": serve.PREFIX_CACHE_SIZE}

    def predict(self, request):
        req = self._req.model_validate(request)
        rec, meta = self._rec(self._prepare(req))
        ps, m = self.server.probs(rec)
        (name, q), = request["questions"].items()
        (keys, p), = [(mm["keys"], pp) for mm, pp in zip(meta, ps)]
        by_key = dict(zip(keys, p))
        return {"p": [float(by_key[k]) for k in q["criteria"]], "input_tokens": m["tokens"],
                "model_latency_ms": m["latency_ms"]}

    def predict_noul(self, request):
        """Robustness B: P(true), unrounded, through the same Server.probs path (kev/api.py:153: noul = p[1])."""
        req = self._req.model_validate(request)
        rec, meta = self._rec(self._prepare(req))
        ps, m = self.server.probs(rec)
        self.server.wait_idle()
        return {"p_true": float(ps[0][1]), "input_tokens": m["tokens"]}

    def after_request(self):
        """Outside the clock: pending CUDA-graph captures run here, between requests, as on an idle server
        (kev/serve.py:150-156,196-200)."""
        self.server.wait_idle()

    def close(self):
        g = getattr(self.server.model, "graphs", None)
        self.info["cuda_graphs_stats"] = g.stats() if g is not None else None
