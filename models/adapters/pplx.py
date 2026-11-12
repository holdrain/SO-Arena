"""Perplexity Decider v1.2 27B and Decider Flash v1.2 4B: `Decider.from_pretrained(<model folder>, device="cuda",
fast=...)` (model card README.md "Fast path"; source/src/pplx_decider/model.py:166-187). Flash: fast=True, i.e.
FastOptions() = whole-layer compile, CUDA graphs over buckets 2k-32k, last-layer pruning (model.py:53-59), no FP8
(its card does not mention it). 27B: FastOptions(fp8=True), the card's FP8 matmuls. warmup() compiles every bucket
before use (model.py:217-220). One request per decide(); the decision record is already Decider's schema (README
"Request schema")."""


class Adapter:
    def __init__(self, config):
        from pplx_decider import Decider
        from pplx_decider.model import FastOptions

        fast = FastOptions(**config["fast_options"]) if config.get("fast_options") else True
        self.decider = Decider.from_pretrained(config["model"], device="cuda", fast=fast)
        import time
        t = time.perf_counter()
        self.decider.warmup()
        self.info = {"fast": repr(getattr(self.decider.fast, "options", fast)), "warmup_s": round(time.perf_counter() - t, 1),
                     "temperature": getattr(self.decider, "temperature", None)}

    @staticmethod
    def result(res, request, kind):
        (name, q), = request["questions"].items()
        a = res["answers"][name]
        if kind == "noul":
            return {"p_true": float(a["noul"]), "usage": res.get("usage")}
        return {"p": [float(a["probabilities"][k]) for k in q["criteria"]], "usage": res.get("usage")}

    def predict_noul(self, request):
        return self.result(self.decider.decide(request), request, "noul")

    def predict_many(self, requests, kind):
        return [self.result(r, q, kind) for r, q in zip(self.decider.decide(list(requests)), requests)]

    def predict(self, request):
        res = self.decider.decide(request)
        (name, q), = request["questions"].items()
        probs = res["answers"][name]["probabilities"]
        return {"p": [float(probs[k]) for k in q["criteria"]], "usage": res.get("usage")}
