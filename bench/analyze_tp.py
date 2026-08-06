#!/usr/bin/env python3
"""Compare matched TP benchmark JSON and enforce the scrape overhead budget."""

import argparse
import json
import math
from pathlib import Path


def row_key(row):
    return (row["scenario"], row.get("concurrency", 1),
            row.get("target_prompt_tokens", 0))


def performance(row):
    if row["scenario"] == "prefill":
        return row["prefill_tok_s"]
    if row["scenario"] == "latency":
        return row["per_request_decode_tok_s"]
    return row["output_tok_s"]


def load(path):
    document = json.loads(Path(path).read_text())
    rows = {row_key(row): row for row in document["rows"]}
    if len(rows) != len(document["rows"]):
        raise ValueError(f"duplicate scenario rows in {path}")
    return document, rows


def compare(tp1_path, tp2_path, scraped_path, sampled_path, budget_percent):
    documents = {}
    rows = {}
    for name, path in (("tp1", tp1_path), ("tp2", tp2_path),
                       ("scraped", scraped_path), ("sampled", sampled_path)):
        documents[name], rows[name] = load(path)

    expected = set(rows["tp2"])
    if not expected:
        raise ValueError("TP=2 benchmark contains no scenario rows")
    for name in ("tp1", "scraped", "sampled"):
        if set(rows[name]) != expected:
            raise ValueError(f"{name} scenario matrix does not match tp2")

    comparisons = []
    scrape_ratios = []
    output_counts_match = True
    for key in sorted(expected):
        baseline = performance(rows["tp2"][key])
        if baseline <= 0:
            raise ValueError(f"non-positive TP=2 performance for {key}")
        values = {name: performance(rows[name][key]) for name in rows}
        scrape_ratio = values["scraped"] / baseline
        scrape_ratios.append(scrape_ratio)
        counts = [rows[name][key].get("mean_output_tokens") for name in rows]
        if all(value is not None for value in counts):
            output_counts_match &= len(set(counts)) == 1
        if values["tp1"] <= 0:
            raise ValueError(f"non-positive TP=1 performance for {key}")
        comparisons.append({
            "scenario": key[0],
            "concurrency": key[1],
            "target_prompt_tokens": key[2],
            "tp2_over_tp1": values["tp2"] / values["tp1"],
            "scraped_over_tp2": scrape_ratio,
            "sampled_over_tp2": values["sampled"] / baseline,
        })

    # A geometric mean gives prompt, decode, and saturation rows equal weight
    # without letting the largest tokens/s number dominate the result.
    aggregate_scraped_ratio = math.exp(
        sum(math.log(value) for value in scrape_ratios) / len(scrape_ratios))
    overhead_percent = max(0.0, (1.0 - aggregate_scraped_ratio) * 100.0)
    return {
        "models_match": len({doc["model"] for doc in documents.values()}) == 1,
        "output_counts_match": output_counts_match,
        "rows": comparisons,
        "aggregate_scraped_over_tp2": aggregate_scraped_ratio,
        "scrape_overhead_percent": overhead_percent,
        "overhead_budget_percent": budget_percent,
        "overhead_budget_pass": overhead_percent <= budget_percent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp1", required=True)
    parser.add_argument("--tp2", required=True)
    parser.add_argument("--scraped", required=True)
    parser.add_argument("--sampled", required=True)
    parser.add_argument("--overhead-budget-percent", type=float, default=1.0)
    parser.add_argument("--json")
    args = parser.parse_args()
    try:
        result = compare(args.tp1, args.tp2, args.scraped, args.sampled,
                         args.overhead_budget_percent)
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print("scenario                 TP2/TP1  scraped/TP2  sampled/TP2")
    for row in result["rows"]:
        label = row["scenario"]
        if row["scenario"] == "throughput":
            label += f" c={row['concurrency']}"
        elif row["scenario"] == "prefill":
            label += f" n={row['target_prompt_tokens']}"
        print(f"{label:24} {row['tp2_over_tp1']:8.3f}"
              f" {row['scraped_over_tp2']:12.3f}"
              f" {row['sampled_over_tp2']:12.3f}")
    print(f"aggregate scrape overhead: {result['scrape_overhead_percent']:.2f}% "
          f"(budget {result['overhead_budget_percent']:.2f}%)")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    if not result["models_match"]:
        print("FAIL: benchmark model names differ")
        return 2
    if not result["output_counts_match"]:
        print("FAIL: matched runs generated different output token counts")
        return 2
    if not result["overhead_budget_pass"]:
        print("FAIL: scrape overhead exceeds budget")
        return 3
    print("PASS: scenario matrix, model identity, and scrape budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
