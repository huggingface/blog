---
title: "SOTA OCR with Core ML and dots.ocr"
thumbnail: /blog/assets/dots-ocr-ne/dots_header.png
authors:
- user: FL33TW00D-HF
- user: pcuenq
---

Every year our hardware is a little more powerful, our models a little smarter for each parameter. In 2025, it is more feasible than ever to run truly competitive models on-device. [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr), a 3B parameter OCR model from RedNote, surpasses Gemini 2.5 Pro in [OmniDocBench](https://github.com/opendatalab/OmniDocBench), making OCR a truly no compromises on-device use case. Running models on-device is certainly appealing to developers: no smuggling API keys, zero cost, and no network required. However, if we want these models to run on-device, we need to be mindful of the limited compute and power budgets.

Enter the Neural Engine, Apple's custom AI accelerator that has shipped with every Apple device since 2017. This accelerator is designed for high performance whilst sipping battery power. Some of our testing has found the Neural Engine to be **12x more power efficient than CPU**, and **4x more power efficient than GPU**.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dots-ocr-ne/joules_cu.png" width="512px" alt="Compute unit energy" />

Whilst this all sounds very appealing, unfortunately the Neural Engine is only accessible through [Core ML](https://developer.apple.com/documentation/coreml), Apple's _closed source_ ML framework. Furthermore, even just converting a model from PyTorch to Core ML can present some challenges, and without a preconverted model or some knowledge of the sharp edges it can be arduous for developers. Luckily, Apple also offers [MLX](https://developer.apple.com/documentation/machinelearning/mlx), a more modern and flexible ML framework that targets the GPU (not the Neural Engine), and can be used in conjunction with Core ML.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dots-ocr-ne/NE_Header.png" width="512px" alt="NE Header" />

In this three part series, we will provide a _reasoning trace_ of how we converted `dots.ocr` to run on-device, using a
combination of `CoreML` and `MLX`. This process should be applicable to many other models, and we hope that this will help
highlight the ideas and tools needed for developers looking to run their own models on-device.

To follow along, clone [the repo](https://github.com/FL33TW00D/dots.ocr.ne). You'll need `uv` and `hf` installed to run
the setup command:

```bash
./boostrap.sh
```

If you just want to skip ahead and use the converted model, you can download it [here](https://huggingface.co/FL33TW00D/dots.ocr.ne).

## Conversion

Converting from PyTorch to CoreML is a two step process:
1. Capturing your PyTorch execution graph (via [`torch.jit.trace`](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html) or, the more modern approach of [`torch.export`](https://docs.pytorch.org/docs/stable/export.html)).
2. Compiling this converted graph to an `.mlpackage` using [`coremltools`](https://github.com/apple/coremltools).

Whilst we do have a few knobs we can tweak for step 2, most of our control is in step 1, the graph we feed to `coremltools`.

Following the programmers litany of [`make it work, make it right, make it fast`](https://wiki.c2.com/?MakeItWorkMakeItRightMakeItFast), we will first focus on getting the
conversion working on GPU, in FLOAT32, and with static shapes. Once we have this working, we can dial down the precision and try and
move to the Neural Engine.

## Dots.OCR

Dots.OCR consists of two key components: A 1.2B parameter vision encoder trained from scratch, based on the [NaViT
architecture](https://arxiv.org/pdf/2307.06304), and a Qwen2.5-1.5B backbone. We will be using `CoreML` to run the vision encoder, and `MLX` to run the LM backbone. 

## Step 0: Understand and simplify the model

In order to convert a model, it's best to understand the structure and function before getting started. Looking at the
original vision modelling file
[here](https://github.com/FL33TW00D/dots.ocr.ne/blob/12c57588a97584b1c95e811475791e298e04162d/modeling_dots_vision.py),
we can see that the vision encoder is similar to the QwenVL family. Like many vision encoders, the vision encoder for `dots` works on a patch basis, in this case `14x14` patches. The `dots` vision encoder is capable of processing videos and batches of images. This gives us an opportunity to simplify by only processing a single image at a time. This approach is frequent in on-device apps, where we convert a model that provides the essential functions and iterate if we want to process multiple images.

When kicking off the conversion process, it's best to start with a minimal viable model. This means removing any bells
and whistles that are not strictly necessary for the model to function. In our case, dots has many different attention implementations available for both the vision encoder and the LM backbone. CoreML has lots of infrastructure oriented around the [`scaled_dot_product_attention` operator](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS18.transformers.scaled_dot_product_attention), which they introduced in iOS 18. We can simplify the model by removing all of the other attention implementations and just focusing on simple `sdpa` ([not the memory efficient variant](https://github.com/FL33TW00D/dots.ocr.ne/commit/d51b0478aacf569146cfd6f6a7542850fe61b11c)) for now, [commit here](https://github.com/FL33TW00D/dots.ocr.ne/commit/2cde03299efb7b64c567a3c197ef4a0259768849).

Once we've done this, we see a scary warning message when we load the model:
```bash
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
```

The model doesn't require Sliding Window Attention to function, so we can happily move on.

## Step 1: A simple harness 

Using `torch.jit.trace` is still the most mature method for converting models to CoreML. We usually encapsulate this in
a simple harness that allows you to modify the compute units used and the precision selected.

You can check out the initial harness [here](https://github.com/FL33TW00D/dots.ocr.ne/blob/master/convert.py). If we run
the following on the original code implementation:

```bash
uv run convert.py --precision FLOAT32 --compute_units CPU_AND_GPU
```
We should bump into the first (of many) issues.

## Step 2: Bug hunting

It is rare that a model will convert first time. Often, you will need to progressively make changes further and further
down the execution graph until you reach the final node.

Our first issue is the following error:

```bash
ERROR - converting 'outer' op (located at: 'vision_tower/rotary_pos_emb/192'):
In op "matmul", when x and y are both non-const, their dtype need to match, but got x as int32 and y as fp32
```
Luckily this error gives us quite a bit of information. We can look at the `VisionRotaryEmbedding` layer and see the following
code:

```python
def forward(self, seqlen: int) -> torch.Tensor:
    seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    freqs = torch.outer(seq, self.inv_freq)
    return freqs
```

Although `torch.arange` has a `dtype` argument, `coremltools` ignores this for `arange` and always outputs `int32`.
We can simply add a cast after the `arange` to fix this issue, [commit here](https://github.com/FL33TW00D/dots.ocr.ne/commit/fb968c54d35f7331d8b36b53d69e9767daba4ad3).

After fixing this, running the conversion again leads us to our next issue at `repeat_interleave`:

```bash
ERROR - converting 'repeat_interleave' op (located at: 'vision_tower/204'):
Cannot add const [None]
```

Whilst this error is less informative, we only have a single call to `repeat_interleave` in our vision encoder:

```python
cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0,
    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
)
```

[`cu_seqlens`](https://github.com/Dao-AILab/flash-attention/issues/850) is used for masking variable length sequences in `flash_attention_2`. It's derived from the `grid_thw`
tensor, which represents `time`, `height` and `width`. Since we are only processing a single image, we can simply remove
this call, [commit here](https://github.com/FL33TW00D/dots.ocr.ne/commit/417512e522b595e5fbddb19c6c66f69db0ecca78).

Onto the next! This time, we get a more cryptic error:

```bash
ERROR - converting '_internal_op_tensor_inplace_fill_' op (located at: 'vision_tower/0/attn/301_internal_tensor_assign_1'):
_internal_op_tensor_inplace_fill does not support dynamic index
```

This is again due to the masking logic to handle variable length sequences. Since we are only processing a single image (not
a video or batch of images), we don't really need attention masking at all! Therefore, we can just use a mask of all `True`. To prepare ourselves for the Neural Engine conversion, we also
switch from using a boolean mask to a float mask of all zeros, as the Neural Engine does not support `bool` tensors [commit here](https://github.com/FL33TW00D/dots.ocr.ne/commit/9a1cd2a4d9d4d0c0c96aef5265f6bd306f1d603e)

With all of this done, the model should now successfully convert to CoreML! However, when we _run_ the model, we get the
following error:

```bash
error: 'mps.reshape' op the result shape is not compatible with the input shape
```

This reshape could be in multiple places! Luckily, we can use a previous warning message to help us track down the issue:

```bash
TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  for t, h, w in grid_thw:
```

Most ML compilers [do **not** like dynamic control flow](https://x.com/tenderizzation/status/1930437803097371080). Luckily for us, as we are only processing a single image, we can
simply remove the loop and process the single `h, w` pair, [commit here](https://github.com/FL33TW00D/dots.ocr.ne/commit/c1049e0d53b20c30a612428a84a2b6203b3273f7).

And there we have it! If we run the conversion again, we should see that the model successfully converts and matches the
original PyTorch precision:

```bash
Max difference: 0.006000518798828125, Mean difference: 1.100682402466191e-05
```

## Step 3: Benchmarking

Now that we've got the model working, let's evaluate the size and performance. The good news is the model is working, the bad news is that it's over 5GB! This is completely untenable for on device deployment!
To benchmark the computation time, we can use the built in XCode tooling by calling:

```
open DotsOCR_FLOAT32.mlpackage
```
which will launch the XCode inspector for the model. After clicking `+ Performance Report` and launching a report on all compute devices, you should see something like the following:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dots-ocr-ne/perf_report_gpu.png" width="1515" alt="GPU Perf report" />

Over a second for a single forward pass of the vision encoder! We have lots of more work. 

In the second part of this series, we will work on the integration between `CoreML` and `MLX`, to run the full model on-device. In the third part, we will dive deep into the optimizations required to get this model running on the
Neural Engine, including quantization and dynamic shapes.

