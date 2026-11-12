"""Sieve-31B-v3: the documented API, `m = load_sieve(<model folder>, device="cuda")` then `decide(m, state,
questions)` (Sieve README.md:100-116), with every default: BF16, the released head.pt policy (two option orders of the
same request read in one pass; "permute" at its default), calibrated temperatures, no max_tokens (nothing is
truncated; an unanswerable request raises ValueError). One request per call."""


class Adapter:
    def __init__(self, config):
        import sieve
        from sieve import decide, load_sieve

        self.decide = decide
        self.m = load_sieve(config["model"], device="cuda")
        self.info = {"sieve_file": sieve.__file__, "model_type": type(self.m).__name__}

    def predict(self, request):
        (name, q), = request["questions"].items()
        answers = self.decide(self.m, request["state"], request["questions"])
        probs = answers[name]["probabilities"]
        return {"p": [float(probs[k]) for k in q["criteria"]]}

    def predict_noul(self, request):
        (name, q), = request["questions"].items()
        answers = self.decide(self.m, request["state"], request["questions"])
        return {"p_true": float(answers[name]["noul"])}
