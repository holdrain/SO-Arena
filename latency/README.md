# Latency

The timing protocol and code behind the paper's latency results (the latency figure and the appendix *Latency
Measurement*). Standard library only, Python ≥ 3.8.

| File | Content |
|---|---|
| `latency_ids.txt` | the 1,109 timed task ids, in the order they were sent |
| `split_shards.py` | splits the ids into equal-time shards, one per GPU copy |
| `run_latency.py` | the timer: one warm-up pass, then timed passes, one request per call |
| `summarize_latency.py` | the latency per task and the mean, p90, p99 and max over tasks |

## Tasks

The 1,109 tasks are every single-question choice task of seven static domains, across all tiers. Their texts come
from the SO-Arena data (`./jev fetch-data`); this folder ships only the ids.

| Domain | Tasks |
|---|---:|
| Math | 200 |
| Scientific Reasoning | 200 |
| Knowledge & Commonsense | 200 |
| Safety | 190 |
| Sentiment & Emotion Understanding | 150 |
| Task Completion Verification | 122 |
| Intent & Routing | 47 |

Each request is the task's decision record (`state` / `questions`), as the evaluator accepts for static tasks.

## Timing protocol

1. **Hardware.** NVIDIA H100 80 GB, one model per GPU, no other process on the card. Each card was checked before
   and after every pass (outside the clock); a pass with another user's process on the card is unclean, and
   `summarize_latency.py` refuses it. GPU clocks were not locked, and the CPU frequency governor was `schedutil`.
2. **One request per call** (batch size 1), sent in file order. A server-based model gets one request at a time.
3. **The clock** (`time.perf_counter` in the harness process) covers the time from handing the request to the
   model's documented entry point (`adapter.predict`) until the option probabilities are back on the host.
   - Inside: request building, tokenization, every forward pass and read-out the model does for that one request
     (several passes for Jeva's options, StartLux's group rounds and Sieve's two option orders), GPU
     synchronisation, and for server-based models the local HTTP round trip.
   - Outside: model load; the model's documented warm-up (`adapter.warmup`); writing the record; cache flushes
     between requests (`adapter.before_request` / `after_request`); clearing an option cache
     (`adapter.before_timed_pass`).
4. **Warm-up.** The model's documented warm-up, then one untimed pass over the tasks in the timed order, in the
   same process. CUDA graph capture, compilation and kernel autotuning have then seen every shape.
5. **Timed passes.** Three passes over the tasks follow in the same process. A task's latency is the **mean of its
   three timings**.
6. **Two copies.** Each model ran as two copies at the same time, one per GPU, each on its own half of the tasks
   (`split_shards.py`, balanced on the model's measured times from an earlier untimed pass). Each copy was pinned
   (`taskset`) to its own 24 physical CPU cores and their SMT siblings, on the GPUs' NUMA node. Never two different
   models at once.
7. **Reported:** per model, the mean, 90th and 99th percentiles (linear interpolation) and the maximum of the 1,109
   task latencies. Every task of every pass is kept; no task failed or exceeded a model's limits in the reported
   runs.

**Closed APIs** cannot be timed on this hardware. The paper places them at their end-to-end times per request from
the evaluation runs, which include the network and the provider's queue, with four requests in flight. For each API
these are the mean over the 1,109 tasks that returned an answer (OpenAI Decisions: 1,106).

## Commands

```bash
# 1. untimed pass on each GPU, for the balancing times (any split, e.g. by request length)
python3 latency/split_shards.py --ids latency/latency_ids.txt --n 2 --prefix run0/shard --data "$SO_ARENA_DATA"
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 python3 latency/run_latency.py --data "$SO_ARENA_DATA" \
    --ids run0/shard0.txt --adapter my_adapter:create --model my-model --out run0/shard0.jsonl --timed-passes 0 &
CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 python3 latency/run_latency.py ... --ids run0/shard1.txt --out run0/shard1.jsonl --timed-passes 0 &
wait
# 2. timed run: shards balanced on the measured times, warm-up pass + 3 timed passes per copy
python3 latency/split_shards.py --ids latency/latency_ids.txt --n 2 --prefix run/shard --measured run0/shard*.jsonl
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 python3 latency/run_latency.py --data "$SO_ARENA_DATA" \
    --ids run/shard0.txt --adapter my_adapter:create --model my-model --out run/shard0.jsonl &
CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 python3 latency/run_latency.py ... --ids run/shard1.txt --out run/shard1.jsonl &
wait
# 3. summary
python3 latency/summarize_latency.py --ids latency/latency_ids.txt --name my-model "run/shard*.jsonl" --out run/summary.json
```

Choose the CPU core ranges for your machine. Use cores on the GPUs' NUMA node, and give each copy its own cores.

- The adapter interface is the evaluator's (`evaluator/MODEL_ADAPTER.md`). `predict(request)` must return the option
  probabilities as `{"answers": {"q": {"probabilities": {key: p, ...}}}}`.
- The optional hooks are `warmup(requests)`, `before_request()`, `after_request()` and `before_timed_pass()`.
- Each record holds `id`, `pass` (`warmup`, `timed1`…`timed3`, or `full`), `wall_ms`, `status` and the
  probabilities `p`. The `.meta.json` next to it holds load and warm-up times and the per-pass GPU checks.

## Settings per model

Each model ran its developer's documented fast path, at the developer's settings for precision, fused kernels and
CUDA graphs. FP8 was used only where the developer's documentation specifies it. The reliability runs used the same
settings.

| Model | Release (Hugging Face @ revision) | Settings |
|---|---|---|
| Laya | `convaiinnovations/laya` @ 7b928d8 | TileLang fast path; 8,192-token input window (2,048 for the head); options read in full, without the package's cut at 48 tokens |
| Kev-4B | `jaredpalmer/kev-4b` @ 6cfce5c on `Qwen/Qwen3.5-4B-Base` @ 1001bb4 | as Kev-27B, with the adapter merged |
| JevAny-4B | `SimpleJev/JevAny-Qwen3.5-4B-LoRA` @ 1c7aa9b on `Qwen/Qwen3.5-4B` @ 851bf6e | as JevAny-27B |
| H2O Lightning | `h2oai/h2o-lightning-4b` @ 08e1da5 | the developer's vLLM 0.30.0 build and `serve.sh` defaults (two-way reading, Triton attention, prefill CUDA graphs) with the documented FP8 option |
| StartLux Instant | `startlux-models/StartLux-Decision-v1.1-4B-Instant` @ 416088f | the documented server (`startlux-decide serve`) on vLLM 0.19.1 with its bidirectional plugin, CUDA graphs and FP8; more than 26 options in the documented group rounds |
| Decider Flash | `perplexity-ai/pplx-decider-flash-v1.2` @ 9ff7cf9 | documented fast path (`fast=True`: layer compilation, CUDA graphs, length buckets), BF16, no FP8; `warmup()` |
| SII-4B + fla | `jylin001206/SII-Decision-4B-LoRA` @ 5c77219 | `DecisionEngine`, BF16, one GPU, thinking off; flash-linear-attention **added** to the documented setup |
| Clef-Flash-9B | `Cloudflare/clef-flash` @ f0b7d3c | as Clef-27B |
| Jeva-4B | `LIQIIIII/Jeva-4B` @ f78e14e | `predict()` with the documented defaults. Warm: option cache cleared at the start of each timed pass and kept within it. Cold: cleared before every request (outside the clock) |
| Kev-27B | `jaredpalmer/kev-27b` @ af0e6d5 | documented serving defaults in process: BF16, fused kernels, CUDA graphs; prefix cache off; graph capture between requests outside the clock |
| JevAny-27B | `SimpleJev/JevAny-Qwen3.8-27B-LoRA` @ 09c9e91 on `Qwen/Qwen3.8-27B` @ 1d4bf0f | documented serving path: adapter merged in BF16, SDPA attention, flash-linear-attention, CUDA graphs up to 2,048 tokens; prefix cache off. For 27B the documented merge ran on the CPU before the model moved to one GPU |
| Clef-27B | `Cloudflare/clef` @ a30162b | the documented SGLang build (`dev-clef` image, `/v1/systemone` endpoint); prefix cache flushed after every request, outside the clock |
| GLiDE | `fastino/GLiDE` @ d199b6d | in process (`Glide.decide`), BF16, SDPA attention, flash-linear-attention; documented `warmup()` with a persistent Triton cache |
| Decider-27B | `perplexity-ai/pplx-decider-v1.2` @ cce0f3f | documented fast path (layer compilation, CUDA graphs, length buckets) with FP8 matrix multiplication; `warmup()` |
| KnowLine-31B | `PelaAI/KnowLine-31B-Gen1` @ 8f052a8 | SGLang 0.5.21 (`--tp 1 --mem-fraction-static 0.85 --enable-fp32-lm-head`), BF16, the documented read-twice front end; prefix cache flushed after every request, outside the clock |
| Sieve-31B-v3 | `sthanika-ai/Sieve-31B-v3` @ 44aa517 | in process (`load_sieve`), BF16, eager; the released policy, which scores each request in two option orders in one pass |
| StartLux Auto | `startlux-models/StartLux-Decision-v1.1-4B-AutoThink` @ 23dd4c5 | as StartLux Instant, in auto mode: gate 0.6, at most 4,096 thinking tokens |

| API | Model id | Settings |
|---|---|---|
| Jev API 1.14 | `jev-1.14.0` | TypeSafe decision API |
| OpenAI Decisions | `gpt-6-luna` | OpenAI decision API |
| GPT-6 Sol | `gpt-6-sol` | Responses API, medium reasoning effort, at most 25,000 output tokens |
| GPT-6 Luna | `gpt-6-luna` | as GPT-6 Sol |

`../VERIFY.md` shows that `summarize_latency.py` on the stored timed records reproduces the paper's latencies.

**Adapter bridge.** With `models/bridge.py`, the conversion of the adapter's probability list into the decision
response happens inside `adapter.predict`, so inside the clock; the paper's harness had no such step. It costs
microseconds per request.
