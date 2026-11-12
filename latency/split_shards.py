"""Split the task ids into shards of about equal total time, one per GPU copy of the model, so that the copies finish
together. Greedy longest-first assignment on each task's measured time from an earlier untimed pass of the same model
(run_latency.py --timed-passes 0); without one, the request length in characters stands in for the time. Each shard
keeps the tasks in file order.

Usage: python3 split_shards.py --ids latency_ids.txt --n 2 --prefix run/shard [--measured full0.jsonl full1.jsonl]
       [--data SO_ARENA_DATA]   (needed only without --measured)
Writes <prefix>0.txt ... <prefix>{n-1}.txt. Standard library only."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_latency import load_requests, read_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--measured", nargs="*", default=[], help="records of an untimed pass (pass 'full')")
    ap.add_argument("--data", help="SO-Arena data folder, for the length estimate")
    ap.add_argument("--protected")
    a = ap.parse_args()
    ids = read_ids(a.ids)
    measured = {}
    for path in a.measured:
        for line in open(path):
            r = json.loads(line)
            if r.get("pass", "full") in ("full", "timed1") and r.get("status") == "ok":
                measured[r["id"]] = r["wall_ms"]
    if measured:
        missing = [i for i in ids if i not in measured]
        if missing:
            raise SystemExit(f"{len(missing)} ids have no measured time (first: {missing[0]})")
        cost = {i: measured[i] for i in ids}
    else:
        if not a.data:
            raise SystemExit("give --measured records or --data for the length estimate")
        reqs = load_requests(a.data, a.protected, ids)
        cost = {i: len(json.dumps(reqs[i], ensure_ascii=False)) for i in ids}
    order = {i: k for k, i in enumerate(ids)}
    shards, totals = [[] for _ in range(a.n)], [0.0] * a.n
    for i in sorted(ids, key=lambda i: -cost[i]):
        k = min(range(a.n), key=totals.__getitem__)
        shards[k].append(i)
        totals[k] += cost[i]
    for k, s in enumerate(shards):
        with open(f"{a.prefix}{k}.txt", "w") as f:
            f.write("\n".join(sorted(s, key=order.__getitem__)) + "\n")
    print(f"{len(ids)} ids into {a.n} shards by {'measured time' if measured else 'request length'}: "
          f"sizes {[len(s) for s in shards]}, totals {[round(t) for t in totals]}")


if __name__ == "__main__":
    main()
