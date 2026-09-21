#!/usr/bin/env python3
"""Plot KV-cache statistics captured by `inferx run-bench`.

Usage:
    python plot_kvstats.py <run-bench log> [...] [--out FILE.png]

The benchmark emits one machine-readable `kvstats,...` CSV line after every
request (see apps/cli/run_bench.cc). This script parses those snapshots and
renders a per-request view of cache behavior: hit/miss outcome and reuse,
block lifecycle, data movement between tiers, transfer volume, and tier
occupancy against the configured budgets.

Multiple logs are rendered side by side (e.g. a default-budget run next to
a small-budget tiering run).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIB = 1 << 20

FIELDS = [
    "request",          # 1-based request index
    "prompt",           # which prompt the request used
    "hit",              # 1 when the prefix lookup hit
    "cached_tokens",    # prompt tokens served from the cache
    "computed_tokens",  # prompt tokens recomputed
    "lookups",
    "hits",
    "committed",
    "retired",
    "evicted",
    "stores",
    "store_bytes",
    "loads",
    "load_bytes",
    "gpu_allocations",
    "gpu_reuses",
    "offloads",
    "offload_bytes",
    "promotions",
    "promotion_bytes",
    "gpu_in_use",
    "host_in_use",
    "gpu_budget",
    "host_budget",
]


@dataclass
class Run:
    label: str
    rows: list[dict] = field(default_factory=list)

    @property
    def prompts(self) -> list[str]:
        # Per-request outcome labels: "A1 miss", "B2 hit", ...
        names = ["A", "B", "C", "D"]
        return [
            f"{names[r['prompt'] % len(names)]}{r['request']}\n{'hit' if r['hit'] else 'miss'}"
            for r in self.rows
        ]


def parse(path: str) -> Run:
    run = Run(label=path.rsplit("/", 1)[-1].replace(".log", ""))
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("kvstats,"):
                continue
            values = line.split(",")[1:]
            row = {name: int(value) for name, value in zip(FIELDS, values)}
            run.rows.append(row)
    if not run.rows:
        raise SystemExit(f"no kvstats lines found in {path}")
    return run


def deltas(rows: list[dict], key: str) -> list[int]:
    """Per-request increments of a cumulative counter."""
    out, prev = [], 0
    for row in rows:
        out.append(row[key] - prev)
        prev = row[key]
    return out


def plot(run: Run, ax: dict, color_a: str, color_b: str) -> None:
    rows = run.rows
    x = range(len(rows))
    labels = run.prompts
    budgets = rows[-1]

    # (1) Prompt tokens: cached vs recomputed, colored by hit/miss.
    axis = ax["reuse"]
    cached = [r["cached_tokens"] for r in rows]
    computed = [r["computed_tokens"] for r in rows]
    axis.bar(x, cached, label="cached (loaded)", color=color_a)
    axis.bar(x, computed, bottom=cached, label="recomputed", color="#c0c0c0")
    for i, r in enumerate(rows):
        kind = "hit" if r["hit"] else "miss"
        axis.text(i, (r["cached_tokens"] + r["computed_tokens"]) + 1, kind,
                  ha="center", fontsize=8,
                  color=color_a if r["hit"] else "#606060")
    axis.set_title("Prompt tokens per request", fontsize=10)
    axis.set_ylabel("tokens")
    axis.legend(fontsize=8)
    axis.set_xticks(x, labels, fontsize=8)

    # (2) Cumulative block lifecycle.
    axis = ax["blocks"]
    axis.plot(x, [r["committed"] for r in rows], "o-", color=color_a, label="committed")
    axis.plot(x, [r["retired"] for r in rows], "s--", color="#808080", label="retired")
    axis.plot(x, [r["evicted"] for r in rows], "d-", color=color_b, label="evicted")
    axis.set_title("Blocks (cumulative)", fontsize=10)
    axis.set_ylabel("blocks")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    axis.set_xticks(x, labels, fontsize=8)

    # (3) CPU/GPU movement: cumulative bytes both directions.
    axis = ax["movement"]
    axis.plot(x, [r["offload_bytes"] / MIB for r in rows], "o-", color=color_b,
              label="GPU→CPU offload")
    axis.plot(x, [r["promotion_bytes"] / MIB for r in rows], "^--", color=color_a,
              label="CPU→GPU promotion")
    axis.set_title("Tier movement (cumulative)", fontsize=10)
    axis.set_ylabel("MiB")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    axis.set_xticks(x, labels, fontsize=8)

    # (4) Transfer volume per request (delta of cumulative byte counters).
    axis = ax["transfers"]
    width = 0.38
    store_d = [b / MIB for b in deltas(rows, "store_bytes")]
    load_d = [b / MIB for b in deltas(rows, "load_bytes")]
    axis.bar([i - width / 2 for i in x], store_d, width, label="stores", color=color_b)
    axis.bar([i + width / 2 for i in x], load_d, width, label="loads", color=color_a)
    axis.set_title("Transferred bytes per request", fontsize=10)
    axis.set_ylabel("MiB")
    axis.legend(fontsize=8)
    axis.set_xticks(list(x), labels, fontsize=8)

    # (5) Device slot sourcing.
    axis = ax["slots"]
    alloc_d = deltas(rows, "gpu_allocations")
    reuse_d = deltas(rows, "gpu_reuses")
    axis.bar([i - width / 2 for i in x], alloc_d, width, label="new allocations",
             color=color_b)
    axis.bar([i + width / 2 for i in x], reuse_d, width, label="free-list reuses",
             color=color_a)
    axis.set_title("Device slots per request", fontsize=10)
    axis.set_ylabel("slots")
    axis.legend(fontsize=8)
    axis.set_xticks(list(x), labels, fontsize=8)

    # (6) Tier occupancy against budgets. Budgets far above the observed
    # occupancy are annotated instead of drawn, so the curves stay readable.
    axis = ax["memory"]
    gpu = [r["gpu_in_use"] / MIB for r in rows]
    host = [r["host_in_use"] / MIB for r in rows]
    axis.plot(x, gpu, "o-", color=color_b, label="device in use")
    axis.plot(x, host, "^--", color=color_a, label="host in use")
    peak = max(max(gpu), max(host)) or 1.0
    axis.set_ylim(0, peak * 1.35)
    for value, color, name in (
        (budgets["gpu_budget"] / MIB, color_b, "device"),
        (budgets["host_budget"] / MIB, color_a, "host"),
    ):
        if value <= peak * 1.25:
            axis.axhline(value, color=color, ls=":", lw=1,
                         label=f"{name} budget {value:.1f} MiB")
        else:
            axis.text(0.99, 0.95, f"{name} budget {value:.0f} MiB (off scale)",
                      transform=axis.transAxes, ha="right", va="top", fontsize=7,
                      color=color)
    axis.set_title("Tier occupancy", fontsize=10)
    axis.set_ylabel("MiB")
    axis.legend(fontsize=7, loc="lower right")
    axis.grid(alpha=0.3)
    axis.set_xticks(x, labels, fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="run-bench output logs")
    parser.add_argument("--out", default="kvstats.png", help="output PNG path")
    args = parser.parse_args()

    runs = [parse(path) for path in args.logs]
    keys = ["reuse", "blocks", "movement", "transfers", "slots", "memory"]
    figure, axes = plt.subplots(
        len(keys), len(runs), figsize=(6.4 * len(runs), 12.5), squeeze=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.22})
    palette = [("#1f77b4", "#d62728"), ("#2ca02c", "#9467bd")]
    for column, run in enumerate(runs):
        color_a, color_b = palette[column % len(palette)]
        ax = {key: axes[row][column] for row, key in enumerate(keys)}
        plot(run, ax, color_a, color_b)
        ax["reuse"].set_title(
            f"{run.label}: prompt tokens per request", fontsize=10, fontweight="bold")
    figure.suptitle(
        "Qwen3-0.6B KV-cache statistics during run-bench (per request)",
        fontsize=12, fontweight="bold", y=0.995)
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
