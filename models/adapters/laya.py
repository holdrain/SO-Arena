"""Laya with every option read in full: the laya package at commit b9fe351 with the 48-token option cut removed
(patches/laya-uncapped.patch; the adapter checks the patched laya/common.py by its SHA-256), max_len=8192 and
head_max_len=2048 per call, and the documented TileLang fast path (laya README.md:965-975: `laya.load(..., fast=True)`
or `agent.accelerate()`). The fast path is sized from cfg["max_len"] when it is built (laya/agent.py:889), so
cfg["max_len"] is raised to 8192 BEFORE accelerate(). accelerate(strict=True) raises instead of silently keeping the
stock forward (agent.py:867-901). A truncated state or collapsed options are errors. Probabilities are the public
response's (4 decimals), renormalised to sum 1."""
import hashlib


class Adapter:
    def __init__(self, config):
        import laya
        import laya.common

        expected = config["uncapped_common_sha256"]
        got = hashlib.sha256(open(laya.common.__file__, "rb").read()).hexdigest()
        if got != expected:
            raise RuntimeError(f"laya.common is not the uncapped copy: {laya.common.__file__} sha256 {got}")
        self.max_len, self.head_max_len = int(config["max_len"]), int(config["head_max_len"])
        self.agent = laya.load(config["model"], device="cuda")
        self.agent.cfg["max_len"] = self.max_len
        if "head_max_len" in self.agent.cfg:
            self.agent.cfg["head_max_len"] = self.head_max_len
        if not self.agent.accelerate(strict=True):
            raise RuntimeError("Laya fast path did not enable")
        fast = self.agent._fast
        self.info = {"laya_file": laya.__file__, "common_sha256": got, "device": str(self.agent.device),
                     "dtype": str(getattr(self.agent, "dtype", None)), "fast_path": type(fast).__name__,
                     "fast_max_len": getattr(fast, "max_len", None), "cfg_max_len": self.agent.cfg.get("max_len"),
                     "cfg_head_max_len": self.agent.cfg.get("head_max_len")}

    def predict(self, request):
        (name, q), = request["questions"].items()
        res = self.agent.system_one(request["state"], {name: q}, max_len=self.max_len, head_max_len=self.head_max_len)
        usage = res["usage"]
        if usage.get("truncated") or usage.get("options"):
            raise ValueError(f"Laya truncated the state or collapsed options: {usage}")
        probs = res["answers"][name]["probabilities"]
        p = [float(probs[k]) for k in q["criteria"]]
        s = sum(p)
        return {"p": [x / s for x in p], "p_raw_sum": s, "input_tokens": usage.get("input_tokens")}

    def predict_noul(self, request):
        (name, q), = request["questions"].items()
        res = self.agent.system_one(request["state"], {name: q}, max_len=self.max_len, head_max_len=self.head_max_len)
        if res["usage"].get("truncated"):
            raise ValueError(f"Laya truncated the state: {res['usage']}")
        return {"p_true": float(res["answers"][name]["noul"])}   # public answer, 4 decimals
