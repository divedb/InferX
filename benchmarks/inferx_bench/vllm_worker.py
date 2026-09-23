"""vLLM in-process suite worker: real incremental outputs, exact timestamps.

Uses the incremental LLMEngine API so first-token and inter-token timings
are observed on real engine steps (never derived by subtracting separate
runs). Model, dtype, context cap and KV bytes arrive as arguments; the
suite supplies prompts, sampling and capacity. One unmeasured warmup pass
(repeat -1) precedes the measured repeats, matching the InferX bench.
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

from workload import load_suite


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--suite', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--max-model-len', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    p.add_argument('--kv-bytes', type=int, required=True,
                   help='Explicit paged-KV allocation; parity with the other engine')
    p.add_argument('--profile-step', type=int, default=-1)
    p.add_argument('--step-timings', type=int, default=0)
    args = p.parse_args()

    suite = load_suite(args.suite)
    llm = LLM(model=args.model, dtype=args.dtype, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, max_num_seqs=args.batch,
              max_num_batched_tokens=suite['token_budget'],
              block_size=suite['block_size'],
              kv_cache_memory_bytes=args.kv_bytes,
              enable_prefix_caching=False, enable_chunked_prefill=True,
              generation_config='vllm', skip_tokenizer_init=True)
    engine = llm.llm_engine
    uid = 0
    for c in suite['cases']:
        if c['batch'] != args.batch:
            continue
        params = SamplingParams(**suite['sampling'], max_tokens=c['output_len'],
                                detokenize=False, output_kind=RequestOutputKind.DELTA)
        for repeat in range(-1, args.repeats):
            count = c['concurrency']
            outputs = [[] for _ in range(count)]
            arrivals = [[] for _ in range(count)]
            ids = {str(uid + i): i for i in range(count)}
            uid += count
            start = time.perf_counter()
            step = 0
            for rid, prompt in zip(ids, c['prompts']):
                engine.add_request(rid, {'prompt_token_ids': prompt}, params)
            while engine.has_unfinished_requests():
                if repeat == 0 and step == args.profile_step:
                    engine.collective_rpc(
                        lambda worker: __import__('torch').cuda.cudart().cudaProfilerStart())
                before_step = time.perf_counter()
                results = engine.step()
                after_step = time.perf_counter()
                if repeat == 0 and step == args.profile_step:
                    engine.collective_rpc(
                        lambda worker: __import__('torch').cuda.cudart().cudaProfilerStop())
                if args.step_timings and repeat >= 0:
                    print(f'STEP,{c["id"]},{repeat},{step},'
                          f'{(after_step - before_step) * 1000:.6f}', flush=True)
                step += 1
                ms = (time.perf_counter() - start) * 1000
                for r in results:
                    i = ids[r.request_id]
                    tokens = list(r.outputs[0].token_ids)
                    outputs[i].extend(tokens)
                    arrivals[i].extend([ms] * len(tokens))
            elapsed = (time.perf_counter() - start) * 1000
            assert all(len(o) == c['output_len'] for o in outputs)
            if repeat >= 0:
                print(json.dumps(dict(case=c['id'], repeat=repeat, engine='vllm',
                                      e2e_ms=elapsed, arrivals_ms=arrivals,
                                      outputs=outputs)), flush=True)


if __name__ == '__main__':
    main()
