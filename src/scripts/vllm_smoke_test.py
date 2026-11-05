#!/usr/bin/env python
"""vLLM smoke test: does vLLM load the model and generate on this GPU?

Run on the GPU VM (inside the venv) and cap the runtime so a hang cannot burn paid time:

    timeout 300 python src/vllm_smoke_test.py
    timeout 300 python src/vllm_smoke_test.py --model Qwen/Qwen3-1.7B-Base --gpu-mem 0.5

Exit codes: 0 = OK, 1 = check failed, 2 = vLLM missing/import error, 3 = model load error,
            4 = generation error. (`timeout` itself returns 124 if it kills the run.)
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--gpu-mem", type=float, default=0.4,
                    help="fraction of GPU memory vLLM may use (leave room for a trainer on the same GPU)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--expect", default="Paris", help="substring expected in the first answer ('' to skip)")
    args = ap.parse_args()

    # 1) torch must still see the GPU (vLLM installs can replace torch with a mismatched build)
    import torch
    print(f"torch {torch.__version__} | cuda build {torch.version.cuda} | available {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: torch cannot see the GPU. The vLLM install probably replaced torch with a build that "
              "does not match the driver (re-run bootstrap pinned to the cu128 index).")
        return 1
    print("device:", torch.cuda.get_device_name(0))

    # 2) import vLLM
    try:
        import vllm
        from vllm import LLM, SamplingParams
        print("vllm", vllm.__version__)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import vllm: {type(e).__name__}: {e}")
        return 2

    # 3) load the model
    t0 = time.time()
    try:
        llm = LLM(args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: vLLM could not load {args.model}: {type(e).__name__}: {e}")
        print("Likely cause: this vLLM version does not support the model architecture. "
              "Stop here and fall back to transformers generation for RL.")
        return 3
    print(f"load seconds: {time.time() - t0:.0f}")

    # 4) generate
    try:
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        t1 = time.time()
        outs = llm.generate(["The capital of France is"] * args.n_prompts, params)
        dt = time.time() - t1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: generation error: {type(e).__name__}: {e}")
        return 4

    text = outs[0].outputs[0].text
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print("first output:", repr(text))
    print(f"{args.n_prompts} prompts, {ntok} tokens in {dt:.1f}s ({ntok / max(dt, 1e-9):.0f} tok/s)")
    print(f"peak torch mem GB (this process): {torch.cuda.max_memory_allocated() / 1e9:.1f}")

    if args.expect and args.expect not in text:
        print(f"FAIL: expected {args.expect!r} in the output")
        return 1

    print("OK: vLLM works with this model on this GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
