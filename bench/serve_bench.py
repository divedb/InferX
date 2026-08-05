#!/usr/bin/env python3
"""Serving benchmark against any OpenAI-compatible endpoint.

The M10 client: it measures inferx-serve, vLLM, and SGLang through the same
socket-level code path, so the only difference between two rows is the engine
under them. `scripts/bench_serve.sh` starts each engine in turn and calls this.

What it measures, per scenario:

  latency    one request at a time. TTFT, inter-token latency (p50/p99), and
             decode tok/s = (tokens-1) / (total - TTFT). This is the batch-1
             number the rest of the repo quotes.
  throughput N requests in flight at once, `--concurrency` of them running at
             any moment. Output tok/s over the whole run is the throughput
             number; TTFT and ITL percentiles say what the queueing cost of
             that throughput was.
  prefill    a long prompt with max_tokens=1, so TTFT is (almost) all prefill:
             prefill tok/s = prompt_tokens / TTFT.

Fairness rules, all of them client-side so no engine can be tuned into them:

  * Greedy everywhere (temperature 0). Both engines then generate the same
    text for the same prompt, so they stop at the same place and the token
    counts being divided by are the same counts.
  * Prompts are unique per request by default -- each carries a random tag --
    so a prefix cache cannot answer one engine's second request from the
    first. `--shared-prefix` deliberately inverts this to measure cache hits.
  * Token counts come from the server's own `usage`, not from counting SSE
    chunks, so a chunking difference cannot show up as a throughput difference.
    (Chunk arrival times are still what ITL is computed from -- that is what a
    streaming client actually experiences.)
  * Raw `http.client` on a per-worker keep-alive connection: no connection
    setup inside a measured request, and no dependency beyond the stdlib, so
    the same interpreter can drive every engine.

Usage:
  serve_bench.py --endpoint http://127.0.0.1:8000 --model Qwen2.5-3B-Instruct \\
                 --label inferx-bf16 --json out.json
"""

import argparse
import json
import random
import statistics
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from urllib.parse import urlparse


# ---------------------------------------------------------------- HTTP client


class StreamClient:
    """One keep-alive connection to an OpenAI-compatible server.

    Not thread-safe by design: the load generator gives each worker its own,
    which is also what keeps a slow response from serializing the others.
    """

    def __init__(self, endpoint, timeout=600.0):
        url = urlparse(endpoint)
        self._host = url.hostname
        self._port = url.port or 80
        self._timeout = timeout
        self._conn = None

    def _connect(self):
        if self._conn is None:
            self._conn = HTTPConnection(self._host, self._port,
                                        timeout=self._timeout)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def stream_completion(self, payload):
        """POST /v1/completions and yield (arrival_time, text, usage) per chunk.

        Times are read as close to the socket as the interpreter allows: the
        readline() returns, then the clock, then the JSON parse.
        """
        body = json.dumps(payload).encode()
        conn = self._connect()
        try:
            conn.request("POST", "/v1/completions", body=body, headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            })
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read(512).decode(errors="replace")
                raise RuntimeError(f"HTTP {resp.status}: {detail}")

            while True:
                line = resp.readline()
                now = time.perf_counter()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    resp.read()  # drain, so the connection stays reusable
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                yield now, text, chunk.get("usage")
        except Exception:
            # A half-read response poisons keep-alive; drop the connection so
            # the next request starts clean rather than inheriting the mess.
            self.close()
            raise


# ------------------------------------------------------------------ one request


class Result:
    """Timings of a single streamed request."""

    __slots__ = ("start", "ttft", "end", "chunk_times", "prompt_tokens",
                 "completion_tokens", "text_len")

    @property
    def latency(self):
        return self.end - self.start

    @property
    def itls(self):
        """Gaps between successive token-bearing chunks, in seconds."""
        return [b - a for a, b in zip(self.chunk_times, self.chunk_times[1:])]

    @property
    def decode_rate(self):
        """Tokens per second once generation is under way (TTFT excluded)."""
        decode_time = self.end - self.start - self.ttft
        if decode_time <= 0 or self.completion_tokens < 2:
            return 0.0
        return (self.completion_tokens - 1) / decode_time


def run_one(client, model, prompt, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    r = Result()
    r.chunk_times = []
    r.prompt_tokens = 0
    r.completion_tokens = 0
    r.text_len = 0
    r.ttft = None
    r.start = time.perf_counter()

    for now, text, usage in client.stream_completion(payload):
        if text:
            if r.ttft is None:
                r.ttft = now - r.start
            r.chunk_times.append(now)
            r.text_len += len(text)
        if usage:
            r.prompt_tokens = usage.get("prompt_tokens", 0) or r.prompt_tokens
            r.completion_tokens = (usage.get("completion_tokens", 0)
                                   or r.completion_tokens)

    r.end = time.perf_counter()
    if r.ttft is None:  # nothing generated: charge the whole thing to TTFT
        r.ttft = r.end - r.start
    if r.completion_tokens == 0:  # server sent no usage; fall back to chunks
        r.completion_tokens = len(r.chunk_times)
    return r


# -------------------------------------------------------------- load generator


def make_prompt(rng, base, shared_prefix):
    """A prompt that no other request in this run will repeat.

    The tag goes first: a prefix cache keys on the prompt's leading tokens, so
    a unique *suffix* would still hand the engine a cache hit for everything
    before it. With --shared-prefix the tag moves to the end, which is the
    opposite experiment -- every request then shares one long prefix.
    """
    tag = "".join(rng.choices(string.ascii_lowercase, k=12))
    if shared_prefix:
        return base + f" (note {tag})"
    return f"Note {tag}. " + base


def run_load(endpoint, model, prompts, max_tokens, concurrency):
    """Run every prompt with `concurrency` of them in flight; time the whole run.

    Returns (results, wall_seconds). Workers hold a connection for the run, and
    the queue is drained by whichever worker is free, so a slow request costs
    the run its own latency and not a whole slot's worth of idleness.
    """
    clients = [StreamClient(endpoint) for _ in range(concurrency)]
    local = threading.local()
    counter = iter(range(concurrency))
    lock = threading.Lock()

    def worker(prompt):
        if not hasattr(local, "client"):
            with lock:
                local.client = clients[next(counter)]
        return run_one(local.client, model, prompt, max_tokens)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.perf_counter()
        results = list(pool.map(worker, prompts))
        wall = time.perf_counter() - t0

    for c in clients:
        c.close()
    return results, wall


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(results, wall, concurrency):
    ttfts = [r.ttft for r in results]
    itls = [x for r in results for x in r.itls]
    out_tokens = sum(r.completion_tokens for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "wall_s": wall,
        "output_tokens": out_tokens,
        "output_tok_s": out_tokens / wall if wall > 0 else 0.0,
        "req_s": len(results) / wall if wall > 0 else 0.0,
        "mean_output_tokens": out_tokens / len(results) if results else 0.0,
        "prompt_tokens": results[0].prompt_tokens if results else 0,
        "ttft_ms_mean": 1000 * statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_ms_p50": 1000 * pct(ttfts, 0.50),
        "ttft_ms_p99": 1000 * pct(ttfts, 0.99),
        "itl_ms_p50": 1000 * pct(itls, 0.50),
        "itl_ms_p99": 1000 * pct(itls, 0.99),
        "per_request_decode_tok_s":
            statistics.fmean([r.decode_rate for r in results]) if results else 0.0,
    }


# ------------------------------------------------------------------- scenarios


DECODE_PROMPT = (
    "Write a detailed technical explanation of how a modern GPU executes a "
    "matrix multiplication, covering memory hierarchy, tensor cores, and "
    "occupancy. Be thorough and precise."
)


def scenario_latency(args, rng):
    """Batch 1, sequentially: the TTFT / ITL / decode tok/s the README quotes."""
    prompts = [make_prompt(rng, DECODE_PROMPT, args.shared_prefix)
               for _ in range(args.warmup + args.iters)]
    warm, _ = run_load(args.endpoint, args.model, prompts[:args.warmup],
                       args.decode_tokens, 1)
    del warm
    results, wall = run_load(args.endpoint, args.model, prompts[args.warmup:],
                             args.decode_tokens, 1)
    s = summarize(results, wall, 1)
    # At concurrency 1 the honest headline is the per-request decode rate; the
    # aggregate includes each request's prefill, which is not decode.
    s["scenario"] = "latency"
    return s


def scenario_throughput(args, rng, concurrency):
    """`concurrency` requests in flight: output tok/s, and what it costs TTFT."""
    n = max(concurrency * args.rounds, concurrency)
    warm_prompts = [make_prompt(rng, DECODE_PROMPT, args.shared_prefix)
                    for _ in range(concurrency)]
    run_load(args.endpoint, args.model, warm_prompts, args.decode_tokens,
             concurrency)
    prompts = [make_prompt(rng, DECODE_PROMPT, args.shared_prefix)
               for _ in range(n)]
    results, wall = run_load(args.endpoint, args.model, prompts,
                             args.decode_tokens, concurrency)
    s = summarize(results, wall, concurrency)
    s["scenario"] = "throughput"
    return s


PREFILL_UNIT = ("Paris is the capital of France. Berlin is the capital of "
                "Germany. Rome is the capital of Italy. Madrid is the capital "
                "of Spain. ")


def calibrate_unit_tokens(args):
    """Tokens per copy of PREFILL_UNIT, measured rather than assumed.

    Asking the server how many tokens a known prompt became is the only way to
    hit a target length without shipping a tokenizer in the client -- and it
    keeps the sizing honest across engines whose tokenizers disagree.
    """
    client = StreamClient(args.endpoint)
    probe = run_one(client, args.model, PREFILL_UNIT * 16, 1)
    client.close()
    if probe.prompt_tokens <= 0:
        return 28.0  # server reported nothing; fall back to a measured constant
    return probe.prompt_tokens / 16.0


def scenario_prefill(args, rng, prefill_len, unit_tokens):
    """A long prompt with max_tokens=1: TTFT is prefill, so tok/s falls out.

    The prompt is built from repeated sentences, sized from the calibrated
    tokens-per-unit, and the tok/s denominator is the count the server itself
    reports -- not the target, and not the client's guess at it.
    """
    repeats = max(1, round(prefill_len / unit_tokens))
    base = PREFILL_UNIT * repeats

    prompts = [make_prompt(rng, base, args.shared_prefix)
               for _ in range(args.warmup + args.iters)]
    run_load(args.endpoint, args.model, prompts[:args.warmup], 1, 1)
    results, wall = run_load(args.endpoint, args.model, prompts[args.warmup:],
                             1, 1)

    ttfts = [r.ttft for r in results]
    ptok = results[0].prompt_tokens
    best = min(ttfts)
    return {
        "scenario": "prefill",
        "concurrency": 1,
        "requests": len(results),
        "target_prompt_tokens": prefill_len,
        "prompt_tokens": ptok,
        "ttft_ms_mean": 1000 * statistics.fmean(ttfts),
        "ttft_ms_p50": 1000 * pct(ttfts, 0.50),
        "ttft_ms_best": 1000 * best,
        "prefill_tok_s": ptok / best if best > 0 else 0.0,
        "prefill_tok_s_mean": ptok / statistics.fmean(ttfts) if ttfts else 0.0,
    }


# ------------------------------------------------------------------------ main


def wait_ready(endpoint, model, timeout_s):
    """Block until the endpoint answers a real completion, or give up."""
    deadline = time.time() + timeout_s
    client = StreamClient(endpoint, timeout=30.0)
    last = None
    while time.time() < deadline:
        try:
            run_one(client, model, "ping", 1)
            client.close()
            return True
        except Exception as e:  # connection refused, model still loading, ...
            last = e
            time.sleep(2.0)
    client.close()
    print(f"endpoint {endpoint} never became ready: {last}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="http://host:port")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--label", default="", help="row label in the output")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--concurrency", default="1,2,4,8",
                    help="comma-separated levels for the throughput sweep")
    ap.add_argument("--prefill-lens", default="512,2048",
                    help="comma-separated prompt sizes for the prefill scenario")
    ap.add_argument("--rounds", type=int, default=4,
                    help="requests per worker in the throughput sweep")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0xC0FFEE,
                    help="fixes the per-request prompt tags across engines")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="reuse one prompt prefix (measures prefix caching)")
    ap.add_argument("--scenarios", default="latency,throughput,prefill")
    ap.add_argument("--json", default="", help="write raw results here")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="seconds to wait for the endpoint to come up")
    args = ap.parse_args()

    label = args.label or args.endpoint
    if args.wait > 0 and not wait_ready(args.endpoint, args.model, args.wait):
        return 1

    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out = {"label": label, "endpoint": args.endpoint, "model": args.model,
           "decode_tokens": args.decode_tokens,
           "shared_prefix": args.shared_prefix, "rows": []}

    # One RNG seeded per run means engine A and engine B see byte-identical
    # prompts in the same order.
    rng = random.Random(args.seed)

    if "latency" in wanted:
        row = scenario_latency(args, rng)
        out["rows"].append(row)
        print(f"{label:14s} latency    | TTFT {row['ttft_ms_p50']:7.2f} ms"
              f" | ITL p50 {row['itl_ms_p50']:6.2f} p99 {row['itl_ms_p99']:6.2f} ms"
              f" | decode {row['per_request_decode_tok_s']:6.1f} tok/s"
              f" | out {row['mean_output_tokens']:.0f} tok", flush=True)

    if "throughput" in wanted:
        for c in [int(x) for x in args.concurrency.split(",") if x.strip()]:
            row = scenario_throughput(args, rng, c)
            out["rows"].append(row)
            print(f"{label:14s} conc {c:<3d}  | {row['output_tok_s']:7.1f} tok/s"
                  f" | {row['req_s']:5.2f} req/s"
                  f" | TTFT p50 {row['ttft_ms_p50']:7.2f} p99 {row['ttft_ms_p99']:7.2f} ms"
                  f" | ITL p99 {row['itl_ms_p99']:6.2f} ms", flush=True)

    if "prefill" in wanted:
        unit_tokens = calibrate_unit_tokens(args)
        for n in [int(x) for x in args.prefill_lens.split(",") if x.strip()]:
            row = scenario_prefill(args, rng, n, unit_tokens)
            out["rows"].append(row)
            print(f"{label:14s} prefill    | {row['prompt_tokens']:5d} tok"
                  f" | TTFT {row['ttft_ms_best']:7.2f} ms"
                  f" | {row['prefill_tok_s']:8.1f} tok/s", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
