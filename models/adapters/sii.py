"""SII-Decision-4B-LoRA: the documented engine, `sii_decision.infer.DecisionEngine(model, precision='bf16',
gpus=1)` (SII-Decision README.md:35-48), thinking off by design (prompts.py: enable_thinking=False), one row per
predict() call, the upstream default max_input_tokens=6000 (no truncation: over-limit rows raise). The decision
record maps to SII's native row: state, question = instructions, options = criteria in order (prompts.py:22-23).
The paper's latency and reliability runs ("SII-4B + fla") add flash-linear-attention to this documented setup."""


class Adapter:
    def __init__(self, config):
        from sii_decision.infer import DecisionEngine

        self.engine = DecisionEngine(config["model"], precision="bf16", gpus=1)
        # which linear-attention kernels the loaded layers hold: fla's, or transformers' torch fallback (as
        # documented); the paper's runs (sii-decision-4b-fla) add flash-linear-attention
        fns = {f"{f.__module__}.{f.__name__}" for m in self.engine.model.modules()
               for f in [getattr(m, "chunk_gated_delta_rule", None)] if f is not None}
        self.info = {"precision": self.engine.precision, "attn": "sdpa (infer.py:33)",
                     "linear_attn_chunk_kernel": sorted(fns)}

    YES_NO = [{"key": "yes", "desc": "Yes"}, {"key": "no", "desc": "No"}]   # the yes/no question (null-awareness set only)

    @classmethod
    def row(cls, request):
        (name, q), = request["questions"].items()
        options = cls.YES_NO if q["type"] == "noul" else [{"key": k, "desc": d} for k, d in q["criteria"].items()]
        return {"id": "x", "type": q["type"], "state": request["state"], "question": q["instructions"],
                "options": options}

    @staticmethod
    def result(res, kind):
        if kind == "noul":
            return {"p_true": float(res["probs"][0]), "input_tokens": res["input_tokens"]}   # options: yes, no
        return {"p": [float(x) for x in res["probs"]], "input_tokens": res["input_tokens"]}

    def predict(self, request):
        (res,) = self.engine.predict([self.row(request)])
        return self.result(res, "choice")

    def predict_noul(self, request):
        (res,) = self.engine.predict([self.row(request)])
        return self.result(res, "noul")

    def predict_many(self, requests, kind):
        """Robustness A/B batching: SII's own batched predict(rows) (right padding, last real token: infer.py:58-69)."""
        return [self.result(r, kind) for r in self.engine.predict([self.row(q) for q in requests])]
