"""JevAny 4B / 27B: the documented serving fast path (JevAny docs/DEPLOYMENT.md:306-309,337-346,386-388): BF16
with the LoRA merged at load (JEVANY_MERGE_BF16=1), SDPA, no torch.compile, direct CUDA graphs capped at 2,048
tokens, set through the documented environment variables (jevany/checkpoint.py:186-198). Prefix cache 0: every state
is distinct. The request is the full HTTP body, `model(request)` (runtime.py:262-264)."""
import os


class Adapter:
    def __init__(self, config):
        os.environ.update(config["env"])
        from jevany.runtime import JevModel

        if config.get("cpu_merge"):
            _load_merged_on_cpu()
        self.model = JevModel.from_pretrained(config["checkpoint"], device="cuda", dtype="bf16")
        self.info = {"describe": _jsonable(self.model.describe()), "cpu_merge": bool(config.get("cpu_merge"))}
        self._shown = False

    def predict(self, request):
        res = self.model(request)
        (name, q), = request["questions"].items()
        probs = res["answers"][name]["probabilities"]   # unrounded floats by option key (jevany/api.py:243-244)
        return {"p": [float(probs[k]) for k in q["criteria"]], "input_tokens": res["usage"]["input_tokens"],
                "model_latency_ms": res.get("latency_ms")}

    def predict_noul(self, request):
        """Robustness B: P(true) from runtime.probs, the step model(request) wraps, before api.to_answers rounds a noul
        answer to 2 decimals (jevany/api.py:233-241)."""
        from jevany.api import SystemOneRequest, to_record
        record, meta = to_record(SystemOneRequest.model_validate(request))
        rt = self.model.runtime
        with rt.lock:
            dists, _ = rt.probs(record)
        return {"p_true": float(dists[0][1])}   # to_answers: noul = p[1]

    def close(self):
        self.info["describe_end"] = _jsonable(self.model.describe())


def _load_merged_on_cpu():
    """JevAny-27B on one 80 GB card (config "cpu_merge"). JevAny's documented BF16 merge loads the backbone in fp32
    on the target device, merges the LoRA, then casts to BF16 (jevany/checkpoint.py:262,274-277): ~108 GB for 27B, more
    than one H100. This wraps Checkpoint.load to run that same code on the CPU (needs about 110 GB of RAM), then moves the merged
    BF16 model to the GPU and captures the CUDA graphs there with JevAny's own RowGraphs, as Checkpoint.load does on a
    GPU (checkpoint.py:290-294). The weights and the fast path are the same; only where the merge runs differs."""
    from dataclasses import replace

    import torch
    from jevany import checkpoint as ckmod
    from jevany.cudagraphs import RowGraphs

    original = ckmod.Checkpoint.load

    def load(self, device, opts=ckmod.LoadOptions()):
        if not str(device).startswith("cuda"):
            return original(self, device, opts)
        tok, m = original(self, "cpu", replace(opts, cuda_graphs=False))
        m.to(device)
        m.device = torch.device(device) if isinstance(m.device, torch.device) else device
        torch.cuda.empty_cache()
        if opts.cuda_graphs:
            graphs = m.cuda_graphs = RowGraphs(m, max_tokens=opts.cuda_graph_max_tokens).capture()
            m.inference_acceleration["cuda_graphs"] = graphs.stats
        m.inference_acceleration["merge_device"] = "cpu"
        return tok, m

    ckmod.Checkpoint.load = load


def _jsonable(x):
    import json
    return json.loads(json.dumps(x, default=str))
