#!/usr/bin/env python3
"""Print HF output and MoE routing margins for one token-id prefix."""

import argparse
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--bf16-router", action="store_true")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, trust_remote_code=False,
        device_map="cuda",
    ).eval()

    captured = []
    hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DeepseekV2TopkRouter":
            if args.bf16_router:
                def bf16_forward(self, hidden_states):
                    hidden_states = hidden_states.view(-1, self.hidden_dim)
                    router_logits = F.linear(hidden_states.to(torch.bfloat16),
                                             self.weight.to(torch.bfloat16))
                    scores = router_logits.softmax(dim=-1, dtype=torch.float32)
                    topk_weights, topk_indices = torch.topk(
                        scores, k=self.top_k, dim=-1, sorted=False)
                    topk_weights = topk_weights * self.routed_scaling_factor
                    return router_logits, topk_weights, topk_indices
                module.forward = types.MethodType(bf16_forward, module)
            def save(_module, _inputs, output, layer=name):
                router_logits, topk_weights, topk_indices = output
                captured.append((layer, router_logits[-1].float().cpu(),
                                 topk_weights[-1].float().cpu(),
                                 topk_indices[-1].cpu()))
            hooks.append(module.register_forward_hook(save))

    ids = torch.tensor([args.ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float().cpu()

    for hook in hooks:
        hook.remove()

    values, indices = torch.topk(logits, 10)
    print("output_top10", list(zip(indices.tolist(), values.tolist())))
    print("output_top1_top2_margin", float(values[0] - values[1]))

    rows = []
    for layer, router_logits, weights, experts in captured:
        probs = router_logits.softmax(dim=-1)
        top7, top7_ids = torch.topk(probs, 7)
        rows.append((float(top7[5] - top7[6]), layer, top7_ids.tolist(),
                     top7.tolist(), experts.tolist(), weights.tolist()))
    rows.sort()
    print("smallest_router_6th_7th_margins")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
