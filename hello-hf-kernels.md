---
title: "Learn the Hugging Face Kernel Hub in 5 Minutes"
thumbnail: /blog/assets/hello-hf-kernels/kernel-hub-five-mins-short.png
authors:
- user: drbh
- user: danieldk
- user: Narsil
- user: pcuenq
- user: pagezyhf
- user: merve
- user: reach-vb
---

# 🏎️ Enhance Your Models in 5 Minutes with the Hugging Face Kernel Hub

**Boost your model performance with pre-optimized kernels, easily loaded from the Hub.**

Today, we'll explore an exciting development from Hugging Face: the **Kernel Hub**! As ML practitioners, we know that maximizing performance often involves diving deep into optimized code, custom CUDA kernels, or complex build systems. The Kernel Hub simplifies this process dramatically!

Below is a short example of how to use a kernel in your code.
```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation")

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

In the next sections we'll cover the following topics:

1.  **What is the Kernel Hub?** - Understanding the core concept.
2.  **How to use the Kernel Hub** - A quick code example.
3.  **Adding a Kernel to a Simple Model** - A practical integration using RMSNorm.
4.  **Reviewing Performance Impact** - Benchmarking the RMSNorm difference.
5.  **Real world use cases** - Examples of how the kernels library is being used in other projects.

We'll introduce these concepts quickly – the core idea can be grasped in about 5 minutes (though experimenting and benchmarking might take a bit longer!).

## 1. What is the Kernel Hub?


The [Kernel Hub](https://huggingface.co/kernels-community) (👈 Check it out!) allows Python libraries and applications to **load optimized compute kernels directly from the Hugging Face Hub**. Think of it like the Model Hub, but for low-level, high-performance code snippets (kernels) that accelerate specific operations, often on GPUs. 

Examples include advanced attention mechanisms (like [FlashAttention](https://huggingface.co/kernels-community/flash-attn) for dramatic speedups and memory savings). Custom [quantization kernels](https://huggingface.co/kernels-community/quantization) (enabling efficient computation with lower-precision data types like INT8 or INT4). Specialized kernels required for complex architectures like [Mixture of Experts (MoE) layers](https://huggingface.co/kernels-community/moe), which involve intricate routing and computation patterns. As well as [activation functions](https://huggingface.co/kernels-community/activation), and [normalization layers (like LayerNorm or RMSNorm)](https://huggingface.co/kernels-community/triton-layer-norm).

Instead of manually managing complex dependencies, wrestling with compilation flags, or building libraries like Triton or CUTLASS from source, you can use the `kernels` library to instantly fetch and run pre-compiled, optimized kernels.

For example, to enable **FlashAttention** you need just one line—no builds, no flags:

```python
from kernels import get_kernel

flash_attention = get_kernel("kernels-community/flash-attn")
```

`kernels` detects your exact Python, PyTorch, and CUDA versions, then downloads the matching pre‑compiled binary—typically in seconds (or a minute or two on a slow connection).

By contrast, compiling FlashAttention yourself requires:

* Cloning the repository and installing every dependency.
* Configuring build flags and environment variables.
* Reserving **\~96 GB of RAM** and plenty of CPU cores.
* Waiting **10 minutes to several hours**, depending on your hardware.
  (See the project’s own [installation guide](https://github.com/Dao-AILab/flash-attention#installation) for details.)

Kernel Hub erases all that friction: one function call, instant acceleration.

### Benefits of the Kernel Hub:

* **Instant Access to Optimized Kernels**: Load and run kernels optimized for various hardware starting with NVIDIA and AMD GPUs, without local compilation hassles.
* **Share and Reuse**: Discover, share, and reuse kernels across different projects and the community.
* **Easy Updates**: Stay up-to-date with the latest kernel improvements simply by pulling the latest version from the Hub.
* **Accelerate Development**: Focus on your model architecture and logic, not on the intricacies of kernel compilation and deployment.
* **Improve Performance**: Leverage kernels optimized by experts to potentially speed up training and inference.
* **Simplify Deployment**: Reduce the complexity of your deployment environment by fetching kernels on demand.
* **Develop and Share Your Own Kernels**: If you create optimized kernels, you can easily share them on the Hub for others to use. This encourages collaboration and knowledge sharing within the community.

> As many machine learning developers know, managing dependencies and building low-level code from source can be a time-consuming and error-prone process. The Kernel Hub aims to simplify this by providing a centralized repository of optimized compute kernels that can be easily loaded and run.

Spend more time building great models and less time fighting build systems!

## 2. How to Use the Kernel Hub (Basic Example)

Using the Kernel Hub is designed to be straightforward. The `kernels` library provides the main interface. Here's a quick example that loads an optimized GELU activation function kernel. (Later on, we'll see another example about how to integrate a kernel in our model).

File: [`activation_validation_example.py`](https://gist.github.com/drbh/aa4b8cfb79597e98be6cf0108644ce16)

```python
# /// script
# dependencies = [
#  "numpy",
#  "torch",
#  "kernels",
# ]
# ///

import torch
import torch.nn.functional as F
from kernels import get_kernel

DEVICE = "cuda"

# Make reproducible
torch.manual_seed(42)

# Download optimized activation kernels from the Hub
activation_kernels = get_kernel("kernels-community/activation")

# Create a random tensor on the GPU
x = torch.randn((4, 4), dtype=torch.float16, device=DEVICE)

# Prepare an output tensor
y = torch.empty_like(x)

# Run the fast GELU kernel
activation_kernels.gelu_fast(y, x)

# Get expected output using PyTorch's built-in GELU
expected = F.gelu(x)

# Compare the kernel output with PyTorch's result
torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)

print("✅ Kernel output matches PyTorch GELU!")

# Optional: print both tensors for inspection
print("\nInput tensor:")
print(x)
print("\nFast GELU kernel output:")
print(y)
print("\nPyTorch GELU output:")
print(expected)

# List available functions in the loaded kernel module
print("\nAvailable functions in 'kernels-community/activation':")
print(dir(activation_kernels))
```

**(Note:** If you have [`uv`](https://github.com/astral-sh/uv) installed, you can save this script as `script.py` and run `uv run script.py` to automatically handle dependencies.)

### What's happening here?

1.  **Import `get_kernel`**: This function is the entry point to the Kernel Hub via the `kernels` library.
2.  **`get_kernel("kernels-community/activation")`**: This line looks for the `activation` kernel repository under the `kernels-community` organization. It downloads, caches, and loads the appropriate pre-compiled kernel binary.
3.  **Prepare Tensors**: We create input (`x`) and output (`y`) tensors on the GPU.
4.  **`activation_kernels.gelu_fast(y, x)`**: We call the specific optimized function (`gelu_fast`) provided by the loaded kernel module.
5.  **Verification**: We check the output.

This simple example shows how easily you can fetch and execute highly optimized code. Now let's look at a more practical integration using RMS Normalization.

## 3. Add a Kernel to a Simple Model

Let's integrate an optimized **RMS Normalization** kernel into a basic model. We'll use the `LlamaRMSNorm` implementation provided in the `kernels-community/triton-layer-norm` repository (note: this repo contains various normalization kernels) and compare it against a baseline PyTorch implementation of RMSNorm.

First, define a simple RMSNorm module in PyTorch and a baseline model using it:


File: [`rmsnorm_baseline.py`](https://gist.github.com/drbh/96621d9eafec5dfa0ca9ca59f6fc1991)

```python
# /// script
# dependencies = [
#  "numpy",
#  "torch",
#  "kernels",
# ]
# ///
import torch
import torch.nn as nn

DEVICE = "cuda"

DTYPE = torch.float16  # Use float16 for better kernel performance potential


# Simple PyTorch implementation of RMSNorm for baseline comparison
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = variance_epsilon
        self.hidden_size = hidden_size

    def forward(self, x):
        # Assumes x is (batch_size, ..., hidden_size)
        input_dtype = x.dtype
        # Calculate variance in float32 for stability
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Apply weight and convert back to original dtype
        return (self.weight * x).to(input_dtype)


class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, eps=1e-5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.norm = RMSNorm(hidden_size, variance_epsilon=eps)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, output_size)

        # ensure all linear layers weights are 1 for testing
        with torch.no_grad():
            self.linear1.weight.fill_(1)
            self.linear1.bias.fill_(0)
            self.linear2.weight.fill_(1)
            self.linear2.bias.fill_(0)
            self.norm.weight.fill_(1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)  # Apply RMSNorm
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Example usage
input_size = 128
hidden_size = 256
output_size = 10
eps_val = 1e-5

baseline_model = (
    BaselineModel(input_size, hidden_size, output_size, eps=eps_val)
    .to(DEVICE)
    .to(DTYPE)
)
dummy_input = torch.randn(32, input_size, device=DEVICE, dtype=DTYPE)  # Batch of 32
output = baseline_model(dummy_input)
print("Baseline RMSNorm model output shape:", output.shape)
```

Now, let's create a version using the `LlamaRMSNorm` kernel loaded via `kernels`.

File: [`rmsnorm_kernel.py`](https://gist.github.com/drbh/141373363e83ea0345807d6525e1fb64)

```python
# /// script
# dependencies = [
#  "numpy",
#  "torch",
#  "kernels",
# ]
# ///
import torch
import torch.nn as nn
from kernels import get_kernel, use_kernel_forward_from_hub

# reuse the model from the previous snippet or copy the class
# definition here to run this script independently
from rmsnorm_baseline import BaselineModel

DEVICE = "cuda"
DTYPE = torch.float16  # Use float16 for better kernel performance potential


layer_norm_kernel_module = get_kernel("kernels-community/triton-layer-norm")

# Simply add the decorator to the LlamaRMSNorm class to automatically replace the forward function
# with the optimized kernel version
# 
# Note: not all kernels ship with layers already mapped, and would require calling the function directly
# However in this case, the LlamaRMSNorm class is already mapped to the kernel function. Otherwise we'd need to
# call the function directly like this:
# ```python
# layer_norm_kernel_module.rms_norm_fn(
#     hidden_states,
#     self.weight,
#     bias=None,
#     residual=None,
#     eps=self.variance_epsilon,
#     dropout_p=0.0,
#     prenorm=False,
#     residual_in_fp32=False,
# )
# ```
@use_kernel_forward_from_hub("LlamaRMSNorm")
class OriginalRMSNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = variance_epsilon
        self.hidden_size = hidden_size

    def forward(self, x):
        # Assumes x is (batch_size, ..., hidden_size)
        input_dtype = x.dtype
        # Calculate variance in float32 for stability
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Apply weight and convert back to original dtype
        return (self.weight * x).to(input_dtype)


class KernelModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        device="cuda",
        dtype=torch.float16,
        eps=1e-5,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # OriginalRMSNorm will be replaced with the optimized kernel layer
        # when the model is loaded
        self.norm = OriginalRMSNorm(hidden_size, variance_epsilon=eps)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, output_size)

        # ensure all linear layers weights are 1 for testing
        with torch.no_grad():
            self.linear1.weight.fill_(1)
            self.linear1.bias.fill_(0)
            self.linear2.weight.fill_(1)
            self.linear2.bias.fill_(0)
            self.norm.weight.fill_(1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# Example usage
input_size = 128
hidden_size = 256
output_size = 10
eps_val = 1e-5

kernel_model = (
    KernelModel(
        input_size, hidden_size, output_size, device=DEVICE, dtype=DTYPE, eps=eps_val
    )
    .to(DEVICE)
    .to(DTYPE)
)

baseline_model = (
    BaselineModel(input_size, hidden_size, output_size, eps=eps_val)
    .to(DEVICE)
    .to(DTYPE)
)

dummy_input = torch.randn(32, input_size, device=DEVICE, dtype=DTYPE)  # Batch of 32

output = baseline_model(dummy_input)
output_kernel = kernel_model(dummy_input)
print("Kernel RMSNorm model output shape:", output_kernel.shape)

# Verify outputs are close (RMSNorm implementations should be numerically close)
try:
    torch.testing.assert_close(output, output_kernel, rtol=1e-2, atol=1e-2)
    print("\nBaseline and Kernel RMSNorm model outputs match!")
except AssertionError as e:
    print("\nBaseline and Kernel RMSNorm model outputs differ slightly:")
    print(e)
except NameError:
    print("\nSkipping output comparison as kernel model output was not generated.")


```

**Important Notes on the `KernelModel`:**

* **Kernel Inheritance:** The `KernelRMSNorm` class inherits from `layer_norm_kernel_module.layers.LlamaRMSNorm`, which is the RMSNorm implementation in the kernel. This allows us to use the optimized kernel directly.
* **Accessing the Function:** The exact way to access the RMSNorm function (`layer_norm_kernel_module.layers.LlamaRMSNorm.forward`, `layer_norm_kernel_module.rms_norm_forward`, or something else) **depends entirely on how the kernel creator structured the repository on the Hub.** You may need to inspect the loaded `layer_norm_kernel_module` object (e.g., using `dir()`) or check the kernel's documentation on the Hub to find the correct function/method and its signature. I've used `rms_norm_forward` as a plausible placeholder and added error handling.
* **Parameters:** We now only define `rms_norm_weight` (no bias), consistent with RMSNorm.

## 4. Benchmarking the Performance Impact

How much faster is the optimized Triton RMSNorm kernel compared to the standard PyTorch version? Let’s benchmark the forward pass to find out.

File: [`rmsnorm_benchmark.py`](https://gist.github.com/drbh/c754a4ba52bcc46190ae4a45516fb190)

```python
# /// script
# dependencies = [
#  "numpy",
#  "torch",
#  "kernels",
# ]
# ///
import torch

# reuse the models from the previous snippets or copy the class
# definitions here to run this script independently
from rmsnorm_baseline import BaselineModel
from rmsnorm_kernel import KernelModel

DEVICE = "cuda"
DTYPE = torch.float16  # Use float16 for better kernel performance potential


# Use torch.cuda.Event for accurate GPU timing (ensure function is defined)
def benchmark_model(model, input_tensor, num_runs=100, warmup_runs=10):
    model.eval()  # Set model to evaluation mode
    dtype = input_tensor.dtype
    model = model.to(input_tensor.device).to(dtype)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_runs
    return avg_time_ms


input_size_bench = 4096
hidden_size_bench = 4096  # RMSNorm performance is sensitive to this dimension
output_size_bench = 10
eps_val_bench = 1e-5

# Create larger models and input for benchmark
# Ensure both models are fully converted to the target DEVICE and DTYPE
baseline_model_bench = (
    BaselineModel(
        input_size_bench, hidden_size_bench, output_size_bench, eps=eps_val_bench
    )
    .to(DEVICE)
    .to(DTYPE)
)
kernel_model_bench = (
    KernelModel(
        input_size_bench,
        hidden_size_bench,
        output_size_bench,
        device=DEVICE,
        dtype=DTYPE,
        eps=eps_val_bench,
    )
    .to(DEVICE)
    .to(DTYPE)
)

# call both with larger batch sizes to warm up the GPU
# and ensure the models are loaded
warmup_input = torch.randn(4096, input_size_bench, device=DEVICE, dtype=DTYPE)
_ = kernel_model_bench(warmup_input)
_ = baseline_model_bench(warmup_input)

batch_sizes = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]

print(
    f"{'Batch Size':<12} | {'Baseline Time (ms)':<18} | {'Kernel Time (ms)':<18} | {'Speedup'}"
)
print("-" * 74)

for batch_size in batch_sizes:
    # Call cuda synchronize to ensure all previous GPU operations are complete
    torch.cuda.synchronize()

    # Create random input tensor
    # Ensure the input tensor is on the correct device and dtype
    bench_input = torch.randn(batch_size, input_size_bench, device=DEVICE, dtype=DTYPE)

    # Run benchmarks only if kernel was loaded successfully
    baseline_time = benchmark_model(baseline_model_bench, bench_input)

    kernel_time = -1  # Sentinel value

    kernel_time = benchmark_model(kernel_model_bench, bench_input)

    baseline_time = round(baseline_time, 4)
    kernel_time = round(kernel_time, 4)
    speedup = round(baseline_time / kernel_time, 2) if kernel_time > 0 else "N/A"
    if kernel_time < baseline_time:
        speedup = f"{speedup:.2f}x"
    elif kernel_time == baseline_time:
        speedup = "1.00x (identical)"
    else:
        speedup = f"{kernel_time / baseline_time:.2f}x slower"
    print(f"{batch_size:<12} | {baseline_time:<18} | {kernel_time:<18} | {speedup}")

```

**Expected Outcome:**
As with LayerNorm, a well-tuned RMSNorm implementation using Triton can deliver substantial speedups over PyTorch’s default version—especially for memory-bound workloads on compatible hardware (e.g., NVIDIA Ampere or Hopper GPUs) and with low-precision types like `float16` or `bfloat16`.


**Keep in Mind:**
* Results may vary depending on your GPU, input size, and data type.
* Microbenchmarks can misrepresent real-world performance.
* Performance hinges on the quality of the kernel implementation.
* Optimized kernels might not benefit small batch sizes due to overhead.


Actual results will depend on your hardware and the specific kernel implementation. Here's an example of what you might see (on a L4 GPU):


| Batch Size | Baseline Time (ms) | Kernel Time (ms) | Speedup |
| ---------- | ------------------ | ---------------- | ------- |
| 256        | 0.2122             | 0.2911           | 0.72x   |
| 512        | 0.4748             | 0.3312           | 1.43x   |
| 1024       | 0.8946             | 0.6864           | 1.30x   |
| 2048       | 2.0289             | 1.3889           | 1.46x   |
| 4096       | 4.4318             | 2.2467           | 1.97x   |
| 8192       | 9.2438             | 4.8497           | 1.91x   |
| 16384      | 18.6992            | 9.8805           | 1.89x   |
| 32768      | 37.079             | 19.9461          | 1.86x   |
| 65536      | 73.588             | 39.593           | 1.86x   |


## 5. Real World Use Cases

The `kernels` library is still growing but is already being used in various real world projects, including:
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference/blob/d658b5def3fe6c32b09b4ffe36f770ba2aa959b4/server/text_generation_server/layers/marlin/fp8.py#L15): The TGI project uses the `kernels` library to load optimized kernels for text generation tasks, improving performance and efficiency.
- [Transformers](https://github.com/huggingface/transformers/blob/6f9da7649f2b23b22543424140ce2421fccff8af/src/transformers/integrations/hub_kernels.py#L32): The Transformers library has integrated the `kernels` library to use drop in optimized layers without requiring any changes to the model code. This allows users to easily switch between standard and optimized implementations.


## Get Started and Next Steps!

You've seen how easy it is to fetch and use optimized kernels with the Hugging Face Kernel Hub. Ready to try it yourself?

1.  **Install the library:**
    ```bash
    pip install kernels torch numpy
    ```
    Ensure you have a compatible PyTorch version and gpu driver installed.

2.  **Browse the Hub:** Explore available kernels on the Hugging Face Hub under the [`kernels` tag](https://huggingface.co/models?other=kernels) or within organizations like [`kernels-community`](https://huggingface.co/kernels-community). Look for kernels relevant to your operations (activations, attention, normalization like LayerNorm/RMSNorm, etc.).

3.  **Experiment:** Try replacing components in your own models. Use `get_kernel("user-or-org/kernel-name")`. **Crucially, inspect the loaded kernel object** (e.g., `print(dir(loaded_kernel))`) or check its Hub repository documentation to understand how to correctly call its functions/methods and what parameters (weights, biases, inputs, epsilon) it expects.

4.  **Benchmark:** Measure the performance impact on your specific hardware and workload. Don't forget to check for numerical correctness (`torch.testing.assert_close`).

5.  **(Advanced) Contribute:** If you develop optimized kernels, consider sharing them on the Hub!

## Conclusion

The Hugging Face Kernel Hub provides a powerful yet simple way to access and leverage optimized compute kernels. By replacing standard PyTorch components with optimized versions for operations like RMS Normalization, you can potentially unlock significant performance improvements without the traditional complexities of custom builds. Remember to check the specifics of each kernel on the Hub for correct usage. Give it a try and see how it can accelerate your workflows!
