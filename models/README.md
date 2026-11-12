# Model adapters of the paper's latency and reliability runs

These are the adapters that produced the paper's latency and reliability results, one per inference path. Only
paths and machine settings were turned into environment variables. Requests, settings and the reading of outputs are
unchanged (see "Changes from the runs" below).

| Path | Content |
|---|---|
| `adapters/*.py` | one adapter per inference path: `Adapter(config)`, `predict`, `predict_noul`, optional `predict_null` and hooks |
| `configs/*.json` | the configuration of every run in the paper (model folder, settings) |
| `bridge.py` | connects an adapter to SO-Arena's adapter interface (`--adapter bridge:create --adapter-path models --model <config>`) |
| `envs/*.txt` | `pip freeze` of each environment the runs used (Python 3.12.16) |
| `patches/laya-uncapped.patch` | Laya without its 48-token option cut |

## Running

```bash
export SO_ARENA_MODELS=/path/to/models       # holds the model folders named in configs/*.json
export JEV_CACHE=/path/to/cache              # optional; default ~/.cache/so-arena
# latency (one copy per GPU; latency/README.md), e.g. Laya on GPU 0:
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 python3 latency/run_latency.py --data "$SO_ARENA_DATA" --ids run/shard0.txt \
    --adapter bridge:create --adapter-path models --model laya --out run/shard0.jsonl
# reliability (reliability/README.md):
CUDA_VISIBLE_DEVICES=0 python3 reliability/run_variants.py --variants variants --adapter bridge:create \
    --adapter-path models --model laya --out preds/Laya
```

Notes on the setup:

- **Python environment.** Run each model with the Python of its environment (`envs/<env>.txt`). The model's own
  package must be installed there from the source given in that file.
- **`SO_ARENA_MODELS`** holds each Hugging Face repository at its revision (table below), in the folder named in
  its configuration.
- **Base models.** Kev loads its base model from the Hugging Face cache (`HF_HUB_OFFLINE=1`), so download
  `Qwen/Qwen3.5-4B-Base` @ 1001bb4 there. SII's repository holds full weights. JevAny reads its base model from
  `$SO_ARENA_MODELS/Qwen3.5-4B` or `$SO_ARENA_MODELS/Qwen3.8-27B`.
- **Server-based models.** H2O, StartLux, Clef and KnowLine start their own server. `JEV_CARD` is the physical GPU
  index; it must equal `CUDA_VISIBLE_DEVICES` and sets the ports. `bridge.py` fills it in from
  `CUDA_VISIBLE_DEVICES`. Server logs go to `JEV_LOG_DIR` (default: a folder under `TMPDIR`).
- **Clef.** It runs in an Apptainer image built from `lmsysorg/sglang:dev-clef`, given in `CLEF_SGLANG_SIF`
  (`APPTAINER` names the apptainer binary, default `apptainer`).
  - The paper's host driver supports CUDA 12.7. There, the image's CUDA 13, and KnowLine's SGLang 0.5.21, ran
    through NVIDIA's user-space CUDA 13 forward-compatibility library (`cuda-compat-13-0`), given in
    `JEV_CUDA13_COMPAT`. Leave it unset on a host driver with CUDA 13 support.
- **APIs.** The keys come from `TYPESAFE_API_KEY` / `OPENAI_API_KEY`, or from the file in `credentials_file`
  (mode 600, `NAME=value` lines). The keys are never logged.

## Models

| Paper name | Config | Adapter | Release (Hugging Face @ revision) | Inference path and settings | Env |
|---|---|---|---|---|---|
| Laya | `laya` | `laya.py` | `convaiinnovations/laya` @ 7b928d828b7b0e022f929d9bd2e44165aa270148 | laya @ b9fe351 with `patches/laya-uncapped.patch` (options read in full; the adapter checks the patched file's SHA-256); TileLang fast path `accelerate(strict=True)` after raising `cfg["max_len"]` to 8,192; per call `max_len=8192, head_max_len=2048`; a truncated state is an error | `laya` |
| Kev-4B | `kev-4b` | `kev.py` | `jaredpalmer/kev-4b` @ 6cfce5c2fa4b4bd64026336ab649c5ca78857d52 on `Qwen/Qwen3.5-4B-Base` @ 1001bb4d826a52d1f399e183466143f4da7b741b | kev.serve NVIDIA defaults in process: BF16, adapter merged, fused kernels, CUDA graphs; `KEV_PREFIX_CACHE=0`; probabilities before the response's rounding | `kev` |
| Kev-27B | `kev-27b` | `kev.py` | `jaredpalmer/kev-27b` @ af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf | as Kev-4B | `kev` |
| JevAny-4B | `jevany-4b` | `jevany.py` | `SimpleJev/JevAny-Qwen3.5-4B-LoRA` @ 1c7aa9bab14ac347aeb917c0bcd757838a8a78ce on `Qwen/Qwen3.5-4B` @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | documented serving path: LoRA merged in BF16, SDPA, no compile, CUDA graphs up to 2,048 tokens, prefix cache 0 (environment variables in the config) | `main` |
| JevAny-27B | `jevany-27b` | `jevany.py` | `SimpleJev/JevAny-Qwen3.8-27B-LoRA` @ 09c9e9102d5b8cc7d56558d25da1202a761b6c0d on `Qwen/Qwen3.8-27B` @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | as JevAny-4B; the documented merge runs on the CPU (`cpu_merge`, about 110 GB of RAM), then the model moves to one GPU | `main` |
| H2O Lightning | `h2o-lightning-4b` | `h2o.py` | `h2oai/h2o-lightning-4b` @ 08e1da56b5c92a0ff623053af1e8f9fe072338bf | the card's `serve.sh` on the h2oai vLLM 0.30.0+cu129 build: two-way reading, TRITON_ATTN, prefill CUDA graphs, `VLLM_ARGS="--quantization fp8"`; started under `taskset` (config `taskset_cores` per card, or `H2O_TASKSET_CORES`) | `h2o` |
| StartLux Instant | `startlux-4b-instant` | `startlux.py` | `startlux-models/StartLux-Decision-v1.1-4B-Instant` @ 416088f77cfce09633906a3675c7cd2e5dabcf10 | `startlux-decide serve --launch-vllm --no-images --max-model-len 16384 --quantization fp8` on vLLM 0.19.1 with the bidirectional plugin; own null output = the server's default none check | `startlux` |
| StartLux Auto | `startlux-4b-autothink-auto` | `startlux.py` | `startlux-models/StartLux-Decision-v1.1-4B-AutoThink` @ 23dd4c513ef387705bb13fd8caadc2d5c863082f | as Instant, plus `--think-mode auto --gate 0.6` and the same in every request (thought budget 4,096 at its default) | `startlux` |
| Decider Flash | `pplx-decider-flash-4b` | `pplx.py` | `perplexity-ai/pplx-decider-flash-v1.2` @ 9ff7cf9838e59df461c276b7699d56b244d2af58 | `Decider.from_pretrained(..., fast=True)` (layer compile, CUDA graphs, buckets 2k–32k, last-layer pruning), BF16, `warmup()` at load | `pplx` |
| Decider-27B | `pplx-decider-27b` | `pplx.py` | `perplexity-ai/pplx-decider-v1.2` @ cce0f3f80033e9d89d80831d5281fd72124aacce | as Flash with `FastOptions(fp8=True)` | `pplx` |
| SII-4B + fla | `sii-decision-4b-fla` | `sii.py` | `jylin001206/SII-Decision-4B-LoRA` @ 5c77219871d297bd375c033eb63ad2cd010af657 | `DecisionEngine(precision='bf16', gpus=1)`, one row per call, thinking off; flash-linear-attention 0.5.2 **added** to the documented setup (the `sii-fla` env). Yes/no question as options yes / no | `sii-fla` |
| Clef-Flash-9B | `clef-flash-9b` | `clef_sglang.py` | `Cloudflare/clef-flash` @ f0b7d3cd65296abfe459e6c228b618f5d8b26d2f | `sglang serve` in the `lmsysorg/sglang:dev-clef` image (Apptainer), `/v1/systemone`; prefix cache flushed after every request | `clef` |
| Clef-27B | `clef-27b` | `clef_sglang.py` | `Cloudflare/clef` @ a30162b99d68a481d8bd75d38df069a0cf83df40 | as Clef-Flash-9B | `clef` |
| Jeva-4B | `jeva-4b-f78e14e` (warm), `jeva-4b-f78e14e-cold` | `jeva.py` | `LIQIIIII/Jeva-4B` @ f78e14e2e9c55e1b6aa1d6a3256ae78b763ac6d1 | `Jeva.from_pretrained(..., truncation="error")`, `predict()` with the option descriptions; warm: option cache cleared at the start of each timed pass; cold: before every request; own null output = `allow_null=True` | `jeva` |
| GLiDE | `glide` | `glide.py` | `fastino/GLiDE` @ d199b6d7e04523d87986b8b68cd62272ed1cefc0 | in process `Glide.decide`, BF16, SDPA, flash-linear-attention, persistent `TRITON_CACHE_DIR` | `glide` |
| KnowLine-31B | `knowline-31b` | `knowline.py` | `PelaAI/KnowLine-31B-Gen1` @ 8f052a8130e599b53e5324c34f0f5ff90696e1c6 | SGLang 0.5.21 `--tp 1 --mem-fraction-static 0.85 --enable-fp32-lm-head` (BF16) plus the card's `knowline_server.py` front end (read twice, `temperatures.json`); prefix cache flushed after every request | `knowline` |
| Sieve-31B-v3 | `sieve-31b-v3` | `sieve.py` | `sthanika-ai/Sieve-31B-v3` @ 44aa5178ec39ddec3dc2da4c2035ab8be5631dae | `load_sieve` + `decide`, BF16, eager, the released two-order policy, all defaults | `main` |
| Jev API 1.14 | `typesafe-jev-1.14` | `decision_api.py` | API `jev-1.14.0` (TypeSafe) | reliability only; request unchanged plus `model`; 2-decimal distributions renormalised | — |
| OpenAI Decisions | `openai-decisions` | `decision_api.py` | API `gpt-6-luna` (OpenAI decisions) | reliability only; request mapped to choices / predicate; a refusal is recorded as a failed request | — |

## How the paper's runs drove them

- **Latency** (`latency/run_latency.py`): 3 timed passes after the warm-up pass, two copies per model, each pinned
  to 24 cores. The hooks ran outside the clock:
  - `after_request`: Kev (graph capture when idle), Clef and KnowLine (prefix-cache flush);
  - `before_request`: Jeva cold;
  - `before_timed_pass`: Jeva.
- **Reliability**: one request at a time per model copy, in the latency settings. The APIs had 4 requests in flight
  (`--concurrency 4`). The hooks ran for every open-weight model except four groups:
  - Laya, Kev-4B, Kev-27B, JevAny-4B, JevAny-27B and Sieve-31B-v3 ran without them. Of these, only Kev has a hook
    (`after_request`). Kev's yes/no read still waits for graph capture inside `predict_noul`, as in the runs.
  - The 8 copy-(b) requests rerun after the label correction ran with the hooks for every model.

  To reproduce exactly, run `reliability/run_variants.py` with `--no-hooks` for those six models.

## Changes from the runs

- **Every adapter:** the module docstrings were rewritten (internal file references removed); comments were
  reworded.
- **Code identical to the runs** (the syntax trees match once docstrings are stripped): `glide.py`, `jeva.py`,
  `jevany.py`, `kev.py`, `laya.py`, `pplx.py`, `sieve.py`, `sii.py`.
- **`h2o.py`:**
  - The vLLM build's `bin` folder is `config["env_bin"]` (default: the folder of the running Python). It was the
    environment's `bin` folder by absolute path.
  - `JEV_CACHE` and the temporary folder have defaults.
  - `H2O_TASKSET_CORES` can override `config["taskset_cores"]`.
- **`startlux.py`, `knowline.py`:**
  - The temporary folder comes from `tempfile.gettempdir()`, which is `TMPDIR` when that is set.
  - `JEV_CACHE` has a default.
  - `knowline.py` prepends `JEV_CUDA13_COMPAT` to `LD_LIBRARY_PATH` only when it is set; before, it was required.
- **`clef_sglang.py`:**
  - The apptainer binary is `APPTAINER` and the image is `CLEF_SGLANG_SIF`; before, both were fixed paths.
  - The bind list is the model folder and the cache folders (or `JEV_APPTAINER_BIND`). The runs bound the whole
    data disk, which held these same folders; the model folders contain no symbolic links.
  - `JEV_CUDA13_COMPAT` and the cache variables are passed into the container when set; before, they were required.
- **`decision_api.py`:** only the descriptive `info["mapping"]` string, which is metadata.
- **Configs:**
  - Dropped: absolute model paths (now `${SO_ARENA_MODELS}/<folder>`), environment names, notes, and batching
    settings of an earlier pass that the paper does not use.
  - The credentials path is now `~/.config/so-arena/decision-api.env`.
- **Not included:** the in-process Clef fallback (`clef_inprocess.py`), which no paper result used.
- **The bridge:** it converts `{"p": [...]}` to the protocol's probability map inside `predict`, which is inside the
  latency clock. That is one dictionary over the options. The paper's harness took the list as it was, with no such
  step.
