---
title: "Updates in transformers that enabled gpt-oss" 
thumbnail: assets/transformers/thumbnail.png
authors:
- user: ariG23498
- user: sergiopaniego
- user: reach-vb
- user: pcuenq
- user: ArthurZ
- user: SaylorTwift
- user: cyrilvallez
---

# Updates in transformers that enabled gpt-oss

OpenAI recently released their [GPT-OSS series of models](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)
on the Hugging Face Hub. While the models themselves drew significant attention from the open-source community,
it was almost inevitable that the quieter updates in the `transformers` library (which made these releases possible) went largely unnoticed.

In this blog post, we won't be diving into how GPT-OSS itself works or how to fine-tune it
([here is a cookbook if you wanted to take a look](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)),
instead show the concrete capabilities in `transformers` that power this release.

> Note: While these updates were motivated by GPT-OSS, they are **generic** and benefit many other large models.

For this release, we enabled:

- [Kernels from the Hub](#kernels-from-the-hub)
- [MXFP4 Quantization and Kernels](#mxfp4-quantization-and-kernels)
- [Tensor Parallelism in `transformers`](#tensor-parallelism-in-transformers)
- [Expert Parallelism](#expert-parallelism-in-transformers)
- [Dynamic Sliding Window Layer & Cache](#dynamic-sliding-window-layer--cache)
- [Load larger models faster](#load-larger-models-faster)
- [Continuous Batching & Paged Attention](#continuous-batching--paged-attention)

## Kernels from the Hub

A kernel is a small, ***specialized*** program that runs directly on an accelerator to perform operations like matrix multiplications,
activations, or normalizations. In eager PyTorch, each operation launches its own kernel in sequence, simple, but inefficient due
to memory transfers and launch overhead. PyTorch 2.0 improved this with `torch.compile`, where backends like **`TorchInductor`**
fuse and optimize kernels automatically, often yielding 2–10× speedups.

But what if you need something beyond the default fused kernels, like a new operator or a hardware-tuned optimization?
Writing custom GPU kernels is powerful but notoriously difficult. That’s where **Kernels from the Hub** comes in: you
can now build kernels with the [kernel-builder](https://huggingface.co/blog/kernel-builder) and share or consume them
directly via the [kernels package](https://huggingface.co/blog/hello-hf-kernels).

For `transformers`, this is a big shift. Historically, the library avoided native kernels as it had to depend on too
many separate PyPI packages. With Kernels from the Hub, there’s only **one lightweight dependency (`kernels`)**, and the
whole system is community-centric. This means transformers can finally tap into optimized, shareable kernels without
sacrificing simplicity or portability.

### Custom Kernels for GPT-OSS

[GPT-OSS](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py),
a Mixture of Experts (MoE) model, is a real adapter of Kernels from the Hub. It leverages two customized kernels:

- `@use_kernel_forward_from_hub("RMSNorm")`
- `@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`

Behind the scenes, these decorators simply point to community-contributed kernels. For example, `RMSNorm` comes
from [`liger_kernels`](https://huggingface.co/kernels-community/liger_kernels), while the `MegaBlocksMoeMLP` kernel
comes from [`megablocks`](https://huggingface.co/kernels-community/megablocks). Depending on your device (CUDA or ROCm)
and whether you’re training or running inference, the right kernel is pulled in automatically.

This design is both **specific and general**: the MoE kernel is tailored to GPT-OSS, but the RMSNorm kernel is already
being reused across multiple models.

Want to confirm kernels are being used? Just enable `INFO` logging and load GPT-OSS with `use_kernels=True`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(level=logging.INFO)

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    use_kernels=True,
)
```

Running a quick generation yields log messages like

```shell
INFO:root:Using layer `LigerRMSNorm` from repo `kernels-community/liger_kernels`
INFO:root:Using layer `MegaBlocksMoeMLP` from repo `kernels-community/megablocks`
```

In Figure 1, we show how well the kernels work with larger batch sizes. While for lower batch sizes it might seem that
it is not worth while to use kernels, you can see how well it does for larger batch sizes. 

| ![benchmark with and without kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-kernels-with-without.png) |
| :--: |
| Figure 1: Benchmarking results of custom kernels |

> Note: [Here is the benchmarking script](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-kernels-with-without.py)

## MXFP4 Quantization and Kernels

### Why quantize at all

Large language models are memory-hungry. Quantization reduces memory footprint by storing weights (and sometimes activations)
in lower-precision formats. For reference, FP32 uses 32 bits per number and BF16 uses 16. By reducing bit width, we trade
some precision for smaller models and faster memory movement.

If you want a visual primer on quantization trade-offs, Maarten Grootendorst’s article is excellent:
[*A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### What is MXFP4

| ![explanation of mxfp4 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/mxfp4.png) |
| :--: |
| Figure 2: The E2M1 format used in the MXFP4 format |

**MXFP4** is a 4-bit floating format with E2M1 layout: 1 sign bit, 2 exponent bits, and 1 mantissa bit, as shown in Figure 2.
On its own, E2M1 is very coarse. MXFP4 compensates with **blockwise scaling**:

- Vectors are grouped into blocks of 32 elements.
- Each block stores a shared scale that restores dynamic range when dequantizing.
- Inside each block, 4-bit values represent numbers relative to that scale.

This blockwise scheme lets MXFP4 keep range while using very few bits. In practice, GPT-OSS 20B fits in roughly `16 GB`
of VRAM and GPT-OSS 120B fits in roughly `80 GB` when MXFP4 is active, which is the difference between “cannot load” and
“can run on a single high-end card.” The catch is that matrix multiplies now have to respect block scales. Doing this
efficiently at scale requires dedicated kernels.

### MXFP4 in `transformers`

`transformers` has native MXFP4 support:

- **Quantizer logic:** [`quantizers/quantizer_mxfp4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_mxfp4.py)
- **Integration hooks:** [`integrations/mxfp4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/mxfp4.py)

You can inspect a model’s configuration to see whether MXFP4 is enabled:

```python
from transformers import GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

# {'modules_to_not_convert': [...], 'quant_method': 'mxfp4'}
```

If `quant_method` is `"mxfp4"`, the model will use the MXFP4 pathway when possible.

> Note: You can now **save** the model directly to the Hub in MXFP4 format, thanks to [this PR](https://github.com/huggingface/transformers/pull/40176)!

### Requirements and fallbacks

To run MXFP4 on GPU you typically need:

1. `accelerate`, `kernels`, and `triton>=3.4` installed.
2. NVIDIA GPU with compute capability ≥ 7.5 (Ampere or newer), or a compatible backend.

If these constraints are not met, `transformers` falls back to a higher-precision path (for example BF16), which uses about 4× the memory of MXFP4.

The [snippet](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/memory-requirements-quantized-vs-dequantized.py) loads
GPT-OSS twice on CUDA: once with `Mxfp4Config(dequantize=True)` (heavier) and once in the default quantized path (lighter). Figure 3 shows the amount
of used VRAM after each load so you can visualize the savings.

| ![memory used with quantized vs dequantized models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/quantization.png) |
| :--: |
| Figure 3: Memory requirements for the quantized and de-quantized models |

### Kernels for MXFP4

Efficient MXFP4 requires kernels that understand 32-element blocks and their scales during GEMMs and fused ops.
This is where **Kernels from the Hub** comes in again. `transformers` automatically pulls in the MXFP4-aware
Triton kernels from the community repository when you load a model that needs them. The repository will appear
in your local cache and will be used during the forward pass.

Quick sanity check with the Hugging Face cache CLI:

```
hf cache scan
```

Sample output:

```
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G
```

This indicates the MXFP4 kernels were fetched and are available for execution.

Let's run some benchmarks and see how well the MXFP4 kernels perform. In Figure 4, we see that the MXFP4 kernels are even better than the custom
MoE and RMSNorm kernels for larger batches.

| ![benchmark mxfp4 kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-mxfp4.png) |
| :--: |
| Figure 4: MXFP4 kernel benchmark |

> Note: [Here is the benchmarking script](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-mxfp4-kernels.py)

## Tensor Parallelism in `transformers`

Tensor Parallelism (TP) splits **tensors inside a layer** across multiple GPUs (or the hardware accelerator in question).
Each GPU multiplies its shard in parallel, and collectives such as all-gather or all-reduce combine the partial results.
This reduces per-GPU memory and keeps all GPUs working on the **same layer**, which improves throughput as sequence length
or batch size grows. TP is communication-intensive and generally works best on a **single machine with fast intra-node links**.

### What this enables in `transformers`

`transformers` implements TP directly in `from_pretrained`. You can start with the predefined plan:

```python
# run with: torchrun --nproc-per-node 4 tp_gpt_oss.py
import os
import torch
from torch import distributed as dist
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast

def initialize_process():
    # torchrun exports: LOCAL_RANK
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

def run_inference():
    model_id = "openai/gpt-oss-120b"
    tok = PreTrainedTokenizerFast.from_pretrained(model_id)

    # built in TP
    model = GptOssForCausalLM.from_pretrained(
        model_id,
        tp_plan="auto",
        torch_dtype="auto",
    ).eval()

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain KV caching briefly."},
    ]
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128)
        torch.cuda.synchronize(device)

    # keep output from rank 0 only
    dist.barrier()
    if dist.get_rank() == 0:
        print(tok.decode(out[0][inputs["input_ids"].shape[-1]:]))

def main():
    initialize_process()
    try:
        run_inference()
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

If you don’t have the infrastructure to run the above, here is what you can do to run it!

```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/tp_gpt_oss.py && \
  torchrun --nproc-per-node=4 tp_gpt_oss.py"
```

> Note: `hf jobs` is available for all Hugging Face PRO users.

Under the hood, `tp_plan="auto"` selects a predefined sharding recipe for each layer and wires the necessary
collectives. You can inspect the active plan with `print(model._tp_plan)` if you want to verify what is being sharded.

### **When to reach for TP**

Use TP when the model is too large for one GPU **and** you want **parallel compute**, not only memory placement.
TP tends to scale throughput with more GPUs, especially for long sequences or larger batches. For a deeper systems
view and how TP interacts with sequence or pipeline parallelism, the
[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) is a great overview.

> Note: If you are curious about how TP differs from `device_map="auto"` (memory placement), this short [Stack Overflow answer](https://stackoverflow.com/questions/78852192/choose-available-gpu-devices-with-device-map) explains the distinction and when to use each.

If you want to know more about TP, here are the two resources that are must reads:

- [`transformers` guide](https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi): Tensor parallelism, supported models, plans, and extension points.
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): background on TP and its relationship to other parallelism modes.


## Expert Parallelism in `transformers`

Expert Parallelism (EP) shards **experts inside MoE layers** across GPUs. Each token is routed to one or a few experts,
so only those experts run their feed-forward pass. Since experts are independent MLPs, we can place different
experts on different ranks and exchange only the hidden states for the routed tokens. This keeps the matrix multiplies
intact on each rank and replaces tensor slicing with routing and collectives.

Run with multiple processes using `torchrun`. EP is enabled via the distributed configuration and works
with GPT-OSS MoE layers out of the box in transformers.

```python
# run with: torchrun --nproc-per-node 4 ep_gpt_oss.py
import os
import torch
from torch import distributed as dist
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast
from transformers.distributed import DistributedConfig

def initialize_process():
    # torchrun exports: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)

def run_inference():
    model_id = "openai/gpt-oss-20b"
    tok = PreTrainedTokenizerFast.from_pretrained(model_id)
    
    model = GptOssForCausalLM.from_pretrained(
        model_id,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        dtype="auto",
    ).eval()

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain KV caching briefly."},
    ]
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    )

    # Place inputs on *this process's* GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128)
        torch.cuda.synchronize(device)

    # keep output from rank 0 only
    dist.barrier(
        device_ids=[int(os.environ["LOCAL_RANK"])]
    )
    if dist.get_rank() == 0:
        print(tok.decode(out[0][inputs["input_ids"].shape[-1]:]))

def main():
    initialize_process()
    try:
        run_inference()
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Here is how you would run using `hf jobs`
```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/ep_gpt_oss.py && \
  torchrun --nproc-per-node=4 ep_gpt_oss.py"
```

## Dynamic Sliding Window Layer & Cache

`transformers` now has a
[**`DynamicSlidingWindowLayer`**](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L165)
and a **config‑aware [`DynamicCache`](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L959)**.
If the model config declares sliding‑window or hybrid attention, the cache **stops growing past the window** for those layers;
if you don’t pass the config, behavior stays as before (full, ever‑growing KV).

This provides us with:

- **Much lower KV‑cache memory** for models with sliding or hybrid attention (e.g. GPT‑OSS).
Cache growth plateaus once the window is reached (e.g., 4K for Mistral; 128 for GPT‑OSS sliding layers), instead of
scaling linearly with total generated tokens. ([GitHub](https://github.com/huggingface/transformers/pull/40039),
[Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/mistral))
- **Speed/latency wins** on long prompts/long generations: smaller KV tensors mean lighter attention reads/writes
and less memory bandwidth pressure, especially after the window is hit. (This is the central motivation behind
sliding‑window/hybrid LLMs.) ([AI21](https://www.ai21.com/blog/rise-of-hybrid-llms/),
[vLLM Blog](https://blog.vllm.ai/2025/08/05/gpt-oss.html))

### How to use it

The optimized cache is set by default, that means **you would not have to make any changes** to your existing code.

If you want to create the `DynamicCache` explicitly here is how you would do it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	torch_dtype="auto",
	device_map="auto",
).eval()

messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(device)

cache = DynamicCache(config=model.config) # create the cache with the model's config

generated = model.generate(
	**inputs,
	max_new_tokens=500,
	past_key_values=cache
)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

Figure 5 showcases how much of a difference it makes for us to use the Dynamic KV Cache with sliding
window attention.

| ![sliding window cache](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-cache.png) |
| :--: |
| Figure 5: The memory analysis of dynamic cache with sliding window attention |

## Load larger models faster

When you load a large model into your GPU, PyTorch needs to **reserve GPU memory for each layer’s weights**. Each of these requests (per layer) takes time,
and for multi-billion-parameter models it can mean **thousands of tiny memory allocations**, adding up to a long wait before the model is ready.
Instead of asking the GPU for new memory every single time, it can **hold on to a big chunk once** and then hand out slices from it quickly.

The allocator holds on to this big chunk and hands out slices. The catch is that the allocator only gets fast *after* you’ve given it some memory to work with.
If you don’t “stock the pantry” first, you still end up doing many slow trips to the market. The PR (🎉 [#36380](https://github.com/huggingface/transformers/pull/36380))
teaches `transformers` to **pre-stock the pantry** before it starts copying model weights.

It:
- Looks at the `device_map` (where each layer will live).
- **Pre-allocates a big enough block on each GPU**.
- Then, as layers are copied in, they just slot neatly into this pre-reserved space.

This results in huge speedups in practice. The PR’s benchmarks showed model load times dropping
from ~42 seconds to ~6 seconds for an 8B model — about **7× faster**.

You have to make no changes to your existing code, as this is default behaviour in `transformers`. If you use **`device_map="auto"`**
or provide your own device map, your model will now load faster automatically. If you’re running with **Tensor Parallel (`tp_plan="auto"`)
and `torchrun`** you also benefit from companion changes that make multi-GPU loading smarter.

## Continuous Batching & Paged Attention

A typical autoregressive generation process looks like Figure 6. You input the prefill tokens, and the transformer
predicts each new token one after the other until it predicts the EOS (End of Sequence) token.

| ![prefilling](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/prefill-tokens.png) |
| :--: |
| Figure 6: Autoregressive token generation |

Let’s see what the generation process looks like when we pass a **batch** of inputs. In Figure 7 you notice
that some generations finish off early than the others. This mismatch of length underutilizes the GPUs.

| ![static batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/static-batching.png) |
| :--: |
| Figure 7: Static batching of sequences |

This type of batching sequences is called *static batching*. While this is simple and easy to understand, it
inherently comes with inefficiencies. Only after each sentence is completely generated can we move on to the next batch.

To bypass this issue, we use **dynamic batching** (also known as *continuous batching*). Instead of waiting
for all the generation to finish, we schedule incoming requests to the completed generations. That way,
as soon as a generation in a batch in complete, we prefill the batch with the next request. The process
looks like in Figure 8

| ![continuous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-batching.png) |
| :--: |
| Figure 8: Continuous Batching of sequences |

Transformers supports continuous batching and here is how to use it:

```bash
import argparse
import time

import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DISPLAYED_SAMPLES = 3

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", "-n", type=int, default=None)
    parser.add_argument("--max-batch-tokens", "-b", type=int, default=None)
    parser.add_argument(
        "--attn", type=str, default="paged_attention|kernels-community/flash-attn", help="Attention implementation"
    )
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
    )
    model = model.cuda().eval()

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
    simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        use_cuda_graph=False,  # Not supported for simple version
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
    )

    # Warmup iterations
    _ = model.generate_batch(
        inputs=simple_batch_inputs[: min(5, args.samples)],
        generation_config=generation_config,
        slice_inputs=True,
    )

    # Actual batch generation
    print("--- Running CB Generation Example ---")
    start_time = time.time()
    batch_outputs = model.generate_batch(
        inputs=simple_batch_inputs,
        generation_config=generation_config,
        slice_inputs=True,
    )
    end_time = time.time()
    print("Done with batch generation.")

    # Decode outputs
    token_count = 0
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=True)
        # Try to decode the output
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=True)
            token_count += len(batch_outputs[request].generated_tokens[1:])
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            continue

        # Display sample if asked
        if i < DISPLAYED_SAMPLES:
            print("-" * 20)
            print(f"{request} Input:  {input_text}")
            if len(output_text) > 0:
                print(f"{request} Output: {output_text}")
            else:
                print(f"[WARN] {request} Output was empty!")

    # Compute stats and maybe print them
    gen_time = end_time - start_time
    tok_per_sec = token_count / gen_time
    print("-" * 20)
    print("--- Finished CB Generation Example ---\n")
    print(f"CB generation took: {gen_time:.2f} seconds for {token_count} tokens. {tok_per_sec:.2f}tok/s")
```

## Conclusion

`transformers` moves quickly and it is community-first. The library evolves at the pace of the field because
contributors shape it in the open.

That velocity enables day-zero integrations like the GPT-OSS series. As the stack becomes increasingly
[PyTorch-first](https://x.com/LysandreJik/status/1933201171130593530), it sheds bloat and doubles down
on the PyTorch paths that matter in practice. The result is a cleaner core that still unlocks new capabilities
through community kernels, quantization, and parallelism plans, while also
[standardizing model definitions](https://huggingface.co/blog/transformers-model-definition)so that architectures
supported in transformers seamlessly extend across the wider ecosystem.

The direction is constant: serve the needs of the community. This post is a snapshot meant to put the key ideas
in one place, not a rolling changelog. It will not be updated often. For the latest details, check the
[docs](https://huggingface.co/docs) and [release notes](https://huggingface.co/changelog), and keep sharing
feedback so the next steps reflect what you need.