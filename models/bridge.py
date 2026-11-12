"""Connect the paper's per-model adapters (models/adapters/) to SO-Arena's adapter interface, so that
latency/run_latency.py and reliability/run_variants.py can drive them:

  python3 latency/run_latency.py --adapter bridge:create --adapter-path models --model laya ...

`--model` names a configuration in models/configs/ (e.g. laya, jeva-4b-f78e14e-cold) or is a path to a JSON
configuration. In a configuration, ${SO_ARENA_MODELS} (and any other ${VAR}) is replaced from the environment;
SO_ARENA_MODELS is the folder that holds the downloaded model folders (models/README.md). For server-based models,
JEV_CARD defaults to CUDA_VISIBLE_DEVICES when that names one card.

The adapters keep the interface of the harness that produced the paper's results: `Adapter(config)`, then
`predict(request) -> {"p": [...]}` for a choice question, `predict_noul(request) -> {"p_true": ...}` for the yes/no
question, `predict_null(request) -> {"p": [...], "p_null": ...}` for a model's own null output, and the optional
hooks before_request(), after_request(), before_timed_pass() and close(). This module only converts those results to
the decision protocol's answer format; it does not change requests, settings or outputs."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOOKS = ("before_request", "after_request", "before_timed_pass", "warmup", "close")


def load_config(name):
    path = name if name.endswith(".json") else os.path.join(HERE, "configs", f"{name}.json")
    if "SO_ARENA_MODELS" not in os.environ:
        raise SystemExit("set SO_ARENA_MODELS to the folder that holds the model folders")

    def expand(v):
        if isinstance(v, str):
            out = os.path.expanduser(os.path.expandvars(v))
            if "${" in out:
                raise SystemExit(f"unset variable in {path}: {v}")
            return out
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v
    with open(path) as f:
        return expand(json.load(f))


def load_adapter_class(name):
    """the adapter module by file path, under a private name (adapter files share names with model packages)"""
    spec = importlib.util.spec_from_file_location(f"so_arena_adapter_{name}", os.path.join(HERE, "adapters", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Adapter


class Bridge:
    def __init__(self, model, context_limit=None):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis.isdigit():
            os.environ.setdefault("JEV_CARD", vis)
        self.config = load_config(model)
        self.adapter = load_adapter_class(self.config.get("adapter", model))(self.config)
        self.option_limit = 255

    def __getattr__(self, name):
        if name in HOOKS or name == "info":
            return getattr(self.__dict__["adapter"], name)   # AttributeError if the adapter has no such hook
        if name == "predict_null" and hasattr(self.__dict__["adapter"], "predict_null"):
            return self._predict_null
        raise AttributeError(name)

    @staticmethod
    def _refusal(exc):
        return type(exc).__name__ == "Refusal"

    def predict(self, request):
        (qname, q), = request["questions"].items()
        try:
            if q["type"] == "noul":
                return {"answers": {qname: {"noul": self.adapter.predict_noul(request)["p_true"]}}}
            p = self.adapter.predict(request)["p"]
        except Exception as exc:  # noqa: BLE001
            if self._refusal(exc):
                return {"answers": {qname: {"type": "refusal"}}}
            raise
        return {"answers": {qname: {"probabilities": dict(zip(q["criteria"], p))}}}

    def _predict_null(self, request):
        (qname, q), = request["questions"].items()
        r = self.adapter.predict_null(request)
        return {"answers": {qname: {"probabilities": dict(zip(q["criteria"], r["p"])), "null_probability": r["p_null"]}}}


def create(*, model, context_limit=None):
    return Bridge(model, context_limit)
