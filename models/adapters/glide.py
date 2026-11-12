"""GLiDE 1.2: in process, `Glide.from_pretrained(<model folder>)` with its defaults, then `model.decide(state,
questions)`, one request per call (model card README.md:88-127; glide/model.py:209-268): BF16 and PyTorch SDPA
(README.md:328), flash-linear-attention installed as validated (README.md:79-81), a persistent TRITON_CACHE_DIR
(README.md:305-310). The glide/ package is imported from the model folder (README.md:84). The card's warm-up
("replay a few hundred representative requests", README.md:308-310) is the untimed warm-up pass over all tasks."""
import sys


class Adapter:
    def __init__(self, config):
        sys.dont_write_bytecode = True   # glide/ is imported from the verified model folder: write no __pycache__ there
        sys.path.insert(0, config["model"])
        from glide import Glide
        import glide

        self.model = Glide.from_pretrained(config["model"])
        self.info = {"glide_file": glide.__file__, "name": getattr(self.model, "name", None)}

    def predict(self, request):
        (name, q), = request["questions"].items()
        res = self.model.decide(request["state"], request["questions"])
        probs = res["answers"][name]["probabilities"]
        return {"p": [float(probs[k]) for k in q["criteria"]], "usage": res.get("usage")}

    def predict_noul(self, request):
        (name, q), = request["questions"].items()
        res = self.model.decide(request["state"], request["questions"])
        return {"p_true": float(res["answers"][name]["noul"])}

    def predict_many(self, requests, kind):
        """Robustness A/B batching: GLiDE's decide_batch (glide/model.py:270-272); a RequestError comes back in place."""
        out = []
        for req, res in zip(requests, self.model.decide_batch([{"state": r["state"], "questions": r["questions"]} for r in requests])):
            if isinstance(res, Exception):
                raise res
            (name, q), = req["questions"].items()
            a = res["answers"][name]
            out.append({"p_true": float(a["noul"])} if kind == "noul" else {"p": [float(a["probabilities"][k]) for k in q["criteria"]]})
        return out
