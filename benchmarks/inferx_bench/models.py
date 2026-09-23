"""Model registry: the only place benchmark code learns model specifics.

A ModelConfig carries everything generic runners need to launch and verify
a model on both engines. Adding a model to the benchmark suite means adding
one entry here (plus, optionally, a model directory with its own workload
and entry-point wrappers); no benchmark logic is copied.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Registry record for one locally runnable checkpoint."""

    name: str                    # Registry key, e.g. 'qwen3-0.6b'.
    path: str                    # Checkpoint directory relative to repo root.
    dtype: str = "bfloat16"      # Execution dtype for both engines.
    max_model_len: int = 2048    # Context cap passed to vLLM.
    vocab_size: int = 0          # Token-ID upper bound for validation; 0 skips it.
    layers: int = 0              # Cached layers, for KV-byte accounting.
    kv_heads: int = 0            # KV heads (GQA) or 0 when not applicable.
    head_dim: int = 0            # Per-head dimension.
    kv_entries_per_token: int = 2  # K and V; 1 for a single latent entry.
    seed: int = 0
    gpu_memory_utilization: float = 0.7
    # Extra engine flags that this checkpoint requires, e.g. quantization
    # hints. Keys: 'vllm' and 'inferx', each a list of CLI tokens.
    engine_flags: dict = field(default_factory=dict)

    def kv_bytes(self, kv_blocks: int, block_size: int) -> int:
        """Bytes of paged KV for `kv_blocks` blocks, both K and V included."""
        per_token = self.kv_entries_per_token * self.kv_heads * self.head_dim * 2
        return kv_blocks * block_size * self.layers * per_token

    def weights(self, root: Path):
        """Sorted safetensors paths of the checkpoint (provenance hashes)."""
        return sorted((root / self.path).glob("*.safetensors"))


MODELS: dict = {
    "qwen3-0.6b": ModelConfig(
        name="qwen3-0.6b",
        path="models/Qwen3-0.6B",
        vocab_size=151936,
        layers=28,
        kv_heads=8,
        head_dim=128,
    ),
}


def get_model(name: str) -> ModelConfig:
    try:
        return MODELS[name]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise SystemExit(f"unknown model '{name}' (known: {known}); "
                         "add a ModelConfig in benchmarks/inferx_bench/models.py")
