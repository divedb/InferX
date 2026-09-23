"""Reusable, model-agnostic benchmark infrastructure for InferX.

The package owns everything a benchmark needs except the model: workload
suites (raw token IDs), engine orchestration (offline suite driver and the
vLLM in-process worker), fail-closed validation, metric computation,
cross-tag analysis, serving-benchmark plumbing, and provenance capture.
Model-specific presets live outside this package (e.g. benchmarks/qwen3/)
and register a ModelConfig here.
"""
