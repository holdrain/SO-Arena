"""Jeva-4B: `Jeva.from_pretrained(<model folder>, local_files_only=True, truncation="error")` with the documented
defaults (merge_lora=True, cache="hybrid": jeva/inference.py:48-50; README.md:233), and
`predict(state, instructions, "choice", options, allow_null=False)` (inference.py:275): options are the criteria
descriptions in order. (system_one with a criteria mapping would instead render "key: description",
inference.py:151.)
Option cache: "warm" = the default hybrid cache, cleared at the start of each timed pass and kept within it; "cold" =
model.clear_cache() before every request (Jeva's own evaluation protocol, README.md:175-176); both outside the clock.
Config "option_format": "key_desc" renders options as "key: description" (not used for the paper's Jeva-4B rows).
Optional "max_state_length" / "max_option_length" override the model's config.yaml."""


class Adapter:
    def __init__(self, config):
        from jeva import Jeva

        lengths = {k: int(config[k]) for k in ("max_state_length", "max_option_length") if k in config}
        self.model = Jeva.from_pretrained(config["model"], device="cuda", local_files_only=True, truncation="error",
                                          **lengths)
        assert config.get("option_format", "desc") in ("desc", "key_desc")
        self.key_desc = config.get("option_format") == "key_desc"
        assert config["cache_mode"] in ("warm", "cold")
        self.cold = config["cache_mode"] == "cold"
        self.info = {"cache": self.model.cache, "cache_mode": config["cache_mode"], "truncation": self.model.truncation,
                     "lora_merged": getattr(self.model, "lora_merged", None),
                     "max_state_length": getattr(self.model, "max_state_length", None),
                     "max_option_length": getattr(self.model, "max_option_length", None),
                     "option_format": "key_desc" if self.key_desc else "desc"}

    def before_request(self):
        """Outside the clock. Cold mode: an empty option cache for every request (Jeva's evaluation protocol)."""
        if self.cold:
            self.model.clear_cache()

    def before_timed_pass(self):
        """Outside the clock. Warm mode: every timed pass is the same stream, file order from an empty option cache,
        never cleared during the pass (TIMING_PROTOCOL.md §2)."""
        self.model.clear_cache()

    def _options(self, q):
        return [f"{k}: {d}" if self.key_desc else str(d) for k, d in q["criteria"].items()]

    def predict_noul(self, request):
        """Robustness B: a Noul read exactly as Jeva maps one (inference.py:164-165: labels/options false,true; positive 1)."""
        (name, q), = request["questions"].items()
        a = self.model.predict(request["state"], q["instructions"], "noul", ["false", "true"], allow_null=False,
                               positive_index=1)
        return {"p_true": float(a.noul)}

    def predict_null(self, request):
        """Robustness B, native null: Choice with allow_null=True (README.md:110-117); p_null = null_probability."""
        (name, q), = request["questions"].items()
        a = self.model.predict(request["state"], q["instructions"], "choice", self._options(q), allow_null=True)
        keys = list(q["criteria"])
        p = [float(x) for x in a.candidate_probabilities]
        best = max(range(len(p)), key=p.__getitem__)
        return {"p": p, "p_null": float(a.null_probability),
                "predicted": None if a.null_probability >= p[best] else keys[best]}

    def predict(self, request):
        (name, q), = request["questions"].items()
        a = self.model.predict(request["state"], q["instructions"], "choice", self._options(q), allow_null=False)
        if a.usage.truncated_tokens:
            raise ValueError("Jeva truncated the input")
        u = a.usage
        return {"p": [float(x) for x in a.candidate_probabilities],
                "usage": {k: getattr(u, k) for k in dir(u) if not k.startswith("_") and isinstance(getattr(u, k), (int, float, bool))}}
