"""System One decision APIs (reliability runs only): TypeSafe Jev 1.14 (POST /v1/systemone, model jev-1.14.0) and
OpenAI Decisions (POST /v1/decisions, gpt-6-luna). The request mapping, the answer checks and the TypeSafe rounding
repair follow the benchmark's API client used for the leaderboard runs, so the numbers line up with them:
- TypeSafe gets the decision record unchanged plus "model"; the reply must name the pinned version.
- OpenAI: a string state is the input as is, a JSON state one user message with an input_json part; choice criteria
  become choices [{value, description}]; noul becomes predicate (criteria, if any, appended to the instructions).
- TypeSafe returns two decimals: a choice distribution whose sum is off 1 by that rounding alone (at most
  min(0.025, 0.005 x options)) is divided by its sum; the raw values stay in the record ("api").
The key comes from the environment (TYPESAFE_API_KEY, OPENAI_API_KEY) or config["credentials_file"] (mode 600,
NAME=value lines); it is never printed or logged, and error text is redacted. Retries: HTTP 408/409/429/5xx and
network errors, honouring Retry-After, up to max_retries. HTTP 401/403/404 or an exhausted balance stop the process at
once (os._exit), so the file in progress stays .part and a rerun starts it again."""
import json
import math
import os
import random
import stat
import sys
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

ENDPOINTS = {"typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
             "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY")}
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


def dump(x):
    return json.dumps(x, ensure_ascii=False, allow_nan=False)


def load_key(config, name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    path = os.path.expanduser(config["credentials_file"])
    if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"{path} must have mode 600")
    for line in open(path):
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == name:
            value = v.strip().strip("\"'")
    if not value:
        raise SystemExit(f"{name} is missing from {path}")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Refusal(ValueError):
    """The API answered the question with a refusal (OpenAI: {"type": "refusal"}): a successful call, no probabilities."""


class Adapter:
    def __init__(self, config):
        self.provider, self.model = config["provider"], config["model"]
        self.url, name = ENDPOINTS[self.provider]
        self._key = load_key(config, name)
        self.timeout = float(config.get("timeout_seconds", 120))
        self.retries = int(config.get("max_retries", 4))
        self.opener = urllib.request.build_opener(NoRedirect())
        self.info = {"provider": self.provider, "url": self.url, "model": self.model, "timeout_seconds": self.timeout,
                     "max_retries": self.retries, "mapping": "leaderboard API client"}

    # --- HTTP ---
    def _redact(self, text):
        return str(text).replace(self._key, "[redacted]")[:500]

    def _detail(self, error):
        try:
            body = json.loads(error.read(16384).decode("utf-8", "replace"))
            err = body.get("error", body.get("detail", {})) if isinstance(body, dict) else {}
            if isinstance(err, str):
                return {"message": self._redact(err)}
            return {k: self._redact(err[k]) for k in ("type", "code", "message") if isinstance(err, dict) and err.get(k)}
        except (ValueError, OSError):
            return {}
        finally:
            error.close()

    def _stop(self, why):
        print(f"decision_api: stopping, {why}", file=sys.stderr, flush=True)
        os._exit(3)

    def _post(self, payload):
        body = dump(payload).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url, data=body, method="POST",
                                         headers={"Authorization": "Bearer " + self._key,
                                                  "Content-Type": "application/json"})
            t = time.perf_counter()
            try:
                with self.opener.open(req, timeout=self.timeout) as r:
                    res = json.load(r)
                    return res, {"request_id": r.headers.get("x-request-id"), "attempts": attempt + 1,
                                 "seconds": round(time.perf_counter() - t, 3)}
            except urllib.error.HTTPError as e:
                status, after = e.code, e.headers.get("Retry-After")
                detail = self._detail(e)
                if status in (401, 403, 404) or detail.get("code") in ("credit_balance_exhausted", "insufficient_quota") \
                        or detail.get("type") == "insufficient_quota":
                    self._stop(f"HTTP {status} {detail}")
                if status not in RETRY_STATUS or attempt == self.retries:
                    raise RuntimeError(f"HTTP {status} {detail}") from None
                try:
                    wait = float(after)
                except (TypeError, ValueError):
                    try:
                        wait = parsedate_to_datetime(after).timestamp() - time.time()
                    except (TypeError, ValueError, OverflowError):
                        wait = min(30, 2 ** attempt + random.random())
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"network error: {type(e).__name__}") from None
                wait = min(30, 2 ** attempt + random.random())
            time.sleep(max(0.0, min(wait, 60.0)))

    # --- request and answer, as the leaderboard client's build_request / decode / distribution ---
    def _payload(self, request):
        if self.provider == "typesafe":
            return dict(request, model=self.model)
        state = request["state"]
        if isinstance(state, str):
            evidence = state
        elif isinstance(state, dict) or (isinstance(state, list) and state and all(isinstance(x, dict) for x in state)):
            evidence = [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
        else:
            evidence = dump(state)
        questions = []
        for name, q in request["questions"].items():
            instr = q["instructions"] if isinstance(q["instructions"], str) else dump(q["instructions"])
            kind, crit = q["type"], q.get("criteria")
            target = {"name": name, "type": "predicate" if kind == "noul" else kind, "instructions": instr}
            if kind == "noul" and crit:
                target["instructions"] += "\n\nDecision criteria (true and false):\n" + dump(crit)
            elif kind == "choice":
                target["choices"] = [dict(value=k, **({"description": v if isinstance(v, str) else dump(v)}
                                                      if v is not None else {})) for k, v in crit.items()]
            elif kind != "noul":
                raise ValueError(f"unsupported question type {kind!r}")
            questions.append(target)
        return {"model": self.model, "input": evidence, "questions": questions}

    def _answer(self, request, expected_type):
        (name, q), = request["questions"].items()
        res, api = self._post(self._payload(request))
        if not isinstance(res.get("model"), str) or not isinstance(res.get("usage"), dict):
            raise ValueError("response lacks model or usage")
        if self.provider == "typesafe" and res["model"] != self.model:
            raise ValueError(f"model {res['model']!r}, not the pinned {self.model!r}")
        api["model"] = res["model"]
        if self.provider == "openai":
            names = [a.get("name") for a in res["answers"]]
            if len(set(names)) != len(names):
                raise ValueError("duplicate answer names")
            answers = {a.get("name"): a for a in res["answers"]}
        else:
            answers = res["answers"]
        if set(answers) != {name}:
            raise ValueError("incomplete or unexpected answers")
        a = answers[name]
        if a.get("type") == "refusal":
            raise Refusal("refusal")
        want = "predicate" if self.provider == "openai" and expected_type == "noul" else expected_type
        if a.get("type") != want:
            raise ValueError(f"answer type {a.get('type')!r}, expected {want!r}")
        return q, a, res["usage"], api

    @staticmethod
    def _prob(x):
        if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("invalid probability")
        return float(x)

    def predict(self, request):
        q, a, usage, api = self._answer(request, "choice")
        keys = list(q["criteria"])
        if self.provider == "openai":
            by_key = {}
            for entry in a["probabilities"]:
                if not isinstance(entry.get("value"), str) or entry["value"] in by_key:
                    raise ValueError("bad or duplicate choice value")
                by_key[entry["value"]] = entry["probability"]
        else:
            by_key = a["probabilities"]
            if not isinstance(by_key, dict):
                raise ValueError("expected a probability map")
        if set(by_key) != set(keys):
            raise ValueError("missing or unexpected options")
        p = [self._prob(by_key[k]) for k in keys]
        total = sum(p)
        if abs(total - 1) > 1e-5:
            rounded = all(abs(x * 100 - round(x * 100)) < 1e-8 for x in p)
            if self.provider != "typesafe" or not rounded or total <= 0 or \
                    abs(total - 1) > min(0.025, len(p) * 0.005 + 1e-8):
                raise ValueError(f"probabilities sum to {total}")
            api.update(raw_probabilities=p, raw_sum=total)
            p = [x / total for x in p]
        return {"p": p, "usage": usage, "api": api}

    def predict_noul(self, request):
        _, a, usage, api = self._answer(request, "noul")
        return {"p_true": self._prob(a["probability"] if self.provider == "openai" else a["noul"]), "usage": usage,
                "api": api}
