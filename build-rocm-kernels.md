---
title: "Easily Build and Share ROCm Kernels with Hugging Face" 
thumbnail: /blog/assets/build-rocm-kernels/thumbnail.png
authors:
- user: badaoui
- user: daniehua
- user: ColorsWind
- user: ftyghome
---

![Easily Build and Share ROCm Kernels with Hugging Face](/blog/assets/build-rocm-kernels/thumbnail.png)

## Intoduction

Custom kernels are the backbone of high-performance deep learning, enabling GPU operations tailored precisely to your workload; whether that’s image processing, tensor transformations, or other compute-heavy tasks. But compiling these kernels for the right architectures, wiring all the build flags, and integrating them cleanly into PyTorch extensions can quickly become a mess of CMake/Nix, compiler errors, and ABI issues, which is not fun. Hugging Face’s [**kernel-builder**](https://github.com/huggingface/kernel-builder) and [**kernels**](https://github.com/huggingface/kernels) libraries make it easy to share these kernels with the [**kernels-community**](https://huggingface.co/kernels-community), with support for multiple GPU and accelerator backends, including CUDA, ROCm, Metal, and XPU. This ensures your kernels are fast, portable, and seamlessly integrated with PyTorch.

In this guide, we focus exclusively on ROCm-compatible kernels and show how to build, test, and share them using [kernel-builder](https://github.com/huggingface/kernel-builder/tree/main). You’ll learn how to create kernels that run efficiently on AMD GPUs, along with best practices for reproducibility, packaging, and deployment.

This ROCm-specific walkthrough is a streamlined version of the original kernel-builder guide. If you’re looking for the broader CUDA-focused version, you can find it here: [A Guide to Building and Scaling Production-Ready CUDA Kernels](https://huggingface.co/blog/kernel-builder).

## Build Steps

We will use the GEMM kernel from [RadeonFlow_Kernels](https://github.com/RadeonFlow/RadeonFlow_Kernels) as an example. If you want to go straight to the guide, [click here](#step-1-project-structure).

### About the kernel 

> [!NOTE]
> This section was written by the **RadeonFlow GEMM** kernel authors to introduce the kernel.  
> Authors: [ColorsWind](https://huggingface.co/ColorsWind), [Zesen Liu](https://huggingface.co/ftyghome), and [Andy](https://huggingface.co/jpy794)



The **RadeonFlow GEMM** kernel is a high-performance, FP8 block-wise matrix multiplication implementation optimized for the AMD Instinct MI300X GPU. GEMM (General Matrix Multiplication) is the core building block behind most deep learning workloads: given two matrices A and B, you compute their product C = A × B. Here it’s implemented in FP8, a low-precision floating-point format that trades a bit of accuracy for much higher throughput and lower memory bandwidth. This kernel was developed for the [AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025), it was awarded the 🏆 **Grand Prize** in **June 2025**, recognizing its excellence in performance and innovation on AMD hardware.

The kernel operates on quantized inputs using the `e4m3fnuz` floating-point format and applies per-block scaling to preserve accuracy during low-precision computation. The `e4m3fnuz` format is an FP8 variant with 4 exponent bits and 3 mantissa bits, designed to be efficient for neural network workloads. Because FP8 has a much smaller dynamic range than FP16/FP32, we apply per-block scaling factors (a_scale and b_scale) so that each block of values is rescaled into a numerically “comfortable” range before and after computation, which helps preserve accuracy despite the low precision. It takes the following arguments:

```
(a, b, a_scale, b_scale, c)
```
where `a` and `b` are the input matrices, `a_scale` and `b_scale` are the scaling factors for `a` and `b` respectively,
and `c` is the output matrix:
* `a` is K × M in e4m3fnuz
* `b` is K × N in e4m3fnuz
* `a_scale` is (K // 128) ×  M in fp32
* `b_scale` is (K // 128) × (N // 128) in fp32
* `c` is M × N in bf16

The kernel is precompiled for specific matrix shapes and assumes a transposed memory layout (as required by the competition). To support additional shapes or alternative memory layouts, you must modify the kernel launcher.

So now that we have a high-performance ROCm kernel, the natural question is: how do we integrate it into a real PyTorch workflow and share it with others? That’s exactly what we’ll cover next, using `kernel-builder` and `kernels` to structure, build, and publish the ROCm kernel.

> [!NOTE]  
> This is a fairly technical guide, but you can still follow it step by step without understanding every detail and everything will work fine. If you’re curious, you can always come back later to dig deeper into the concepts.

### Step 1: Project Structure

The Hugging Face Kernel Builder expects your files to be organized like this:

```
gemm/
├── build.toml
├── gemm
│   └── gemm_kernel.h
├── flake.nix
└── torch-ext
    ├── torch_binding.cpp
    ├── torch_binding.h
    └── gemm
        └── __init__.py
```
 

- **build.toml**: The project manifest; it’s the brain of the build process.
- **gemm/**: Your raw CUDA source code where the GPU magic happens.
- **flake.nix**: The key to a perfectly reproducible build environment.
- **torch-ext/gemm/**: The Python wrapper for the raw PyTorch operators

Sometimes your project might depend on other files, like tests or helper scripts, and you can add them without any issues.
In our case, our project will be structured like this:

```
gemm/
├── build.toml
├── gemm
│   ├── gemm_kernel.h
│   ├── gemm_kernel_legacy.h
│   ├── transpose_kernel.h
│   └── gemm_launcher.hip
├── include
│   ├── clangd_workaround.h
│   ├── gpu_libs.h
│   ├── gpu_types.h
│   └── timer.h
├── src/utils
│   ├── arithmetic.h
│   └── timer.hip
├── tests/checker
│   ├── checker.cpp
│   ├── metrics.h
│   └── checker.h
├── flake.nix
└── torch-ext
    ├── torch_binding.cpp
    ├── torch_binding.h
    └── gemm
        └── __init__.py
```

If you look at the original files of the gemm kernel in the RadeonFlow Kernels, they are HIP source files with `.cpp ` extensions. As a first step, you need to change these extensions to either .h or .hip depending on their content and usage:
- Use `.h` for header files containing kernel declarations, inline functions, or template code that will be included in other files
- Use `.hip` for implementation files containing HIP/GPU code that needs to be compiled separately (e.g., kernel launchers, device functions with complex implementations)

In our example, `gemm_kernel.h`, `gemm_kernel_legacy.h`, and `transpose_kernel.h` are header files, while `gemm_launcher.hip` is a HIP implementation file. This naming convention helps the kernel-builder correctly identify and compile each file type.

### Step 2: Configuration Files Setup 

#### The `build.toml` Manifest 

This file orchestrates the entire build. It tells the kernel-builder what to compile and how everything connects. 

```toml
[general]
name = "gemm"
universal = false

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h",
]

[kernel.gemm]
backend = "rocm"
rocm-archs = [
    "gfx942",
]

depends = ["torch"]

src = [
  "include/clangd_workaround.h",
  "include/gpu_libs.h",
  "include/gpu_types.h",
  "include/timer.h",
  "gemm/gemm_kernel.h",
  "gemm/gemm_kernel_legacy.h",
  "gemm/gemm_launcher.hip",
  "gemm/transpose_kernel.h",
  "src/utils/arithmetic.h",
  "src/utils/timer.hip",
  "tests/checker/metrics.h",
]

include = ["include"]
```

**general**

This section contains general project configuration settings.

- **name** (required): The name of your project. This should match your kernel name and will be used for the Python package.
- **universal** (optional): the kernel is a universal kernel when set to `true`. A universal kernel is a pure Python package (no compiled files). Universal kernels do not use the other sections described below. A good example of a universal kernel is a Triton kernel. Default: `false`

**torch**

This section describes the Torch extension configuration. It defines the Python bindings that will expose your kernel to PyTorch.

- **src** (required): A list of source files and headers for the PyTorch extension. In our case, this includes the C++ binding files that create the Python interface.

**kernel.gemm**

Specification of a kernel named "gemm". You can define multiple kernel sections in the same build.toml file if you have multiple kernels.

- **backend** (required): The compute backend for the kernel. We use "rocm" for AMD GPU support.
- **rocm-archs** (required for ROCm): A list of ROCm architectures that the kernel should be compiled for. "gfx942" targets the MI300 series GPUs.
- **depends** (required): A list of dependencies. We depend on "torch" to use PyTorch's tensor operations.
- **include** (optional): Include directories relative to the project root. This helps the compiler find header files.

#### The `flake.nix` Reproducibility File 

To ensure anyone can build your kernel on any machine, we use a flake.nix file. It locks the exact version of the kernel-builder and its dependencies. (You can just copy and paste this example and change the description)

```nix
{
  description = "Flake for GEMM kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:

    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
```

#### Writing the Kernel

Now for the GPU code. Inside `gemm/gemm_launcher.hip`, we define how the GEMM kernel is launched.
Depending on the configuration, we either call the new optimized `gemm/gemm_kernel` or fall back to the legacy implementation (`gemm/gemm_kernel_legacy`).

```C
// ... previous includes and definitions
extern "C" void run(
    void *a, void *b, void *as, void *bs, void *c,
    int m, int n, int k,
    PerfMetrics *metrics, hipStream_t job_stream0
) {
    const __FP8_TYPE *a_ptr = static_cast<const __FP8_TYPE *>(a);
    const __FP8_TYPE *b_ptr = static_cast<const __FP8_TYPE *>(b);
    __BF16_TYPE *c_ptr = static_cast<__BF16_TYPE *>(c);
    const float *as_ptr = static_cast<const float *>(as);
    const float *bs_ptr = static_cast<const float *>(bs);

    KernelTimerScoped timer(timers, 2LL * m * n * k,
        metrics ? &metrics->entries[0].time : nullptr,
        metrics ? &metrics->entries[0].gflops : nullptr, job_stream0);

    // Dispatch GEMM to the fastest available implementation
    switch (pack_shape(m, n, k)) {
        DISPATCH_GEMM(1024, 1536, 7168, 256, 128, 128, 4, 2, 512, 4, 16);
        DISPATCH_GEMM(6144, 7168, 2304, 256, 128, 128, 4, 2, 512, 1, 16);
        default: {
            printf("Error: Unsupported shape M=%d, K=%d, N=%d\n", m, k, n);
            abort();
        }
    }
}
// ...
```

#### Registering a Native PyTorch Operator 

This step is key. We’re not just making the function available in Python; we’re turning it into a native PyTorch operator. That means it becomes a first-class part of PyTorch itself, accessible through `torch.ops`.

The file `torch-ext/torch_binding.cpp` handles this registration. 

```C
#include <torch/all.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>

#include "registration.h"
#include "torch_binding.h"

// Forward declaration of the C function from gemm_launcher.hip
extern "C" {
    struct PerfMetrics;
    void run(void *a, void *b, void *as, void *bs, void *c, int m, int n, int k, PerfMetrics *metrics, hipStream_t job_stream0);
}

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs) {
    
    // Validate tensor properties
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on GPU device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on GPU device");
    TORCH_CHECK(as.device().is_cuda(), "Scale tensor as must be on GPU device");
    TORCH_CHECK(bs.device().is_cuda(), "Scale tensor bs must be on GPU device");
    TORCH_CHECK(out.device().is_cuda(), "Output tensor out must be on GPU device");
    
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(as.is_contiguous(), "Scale tensor as must be contiguous");
    TORCH_CHECK(bs.is_contiguous(), "Scale tensor bs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "Output tensor out must be contiguous");
    
    // Get matrix dimensions from tensor shapes
    // Assuming a is [M, K], b is [K, N], out is [M, N]
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    TORCH_CHECK(b.size(0) == K, "Matrix dimensions mismatch: a.size(1) != b.size(0)");
    TORCH_CHECK(out.size(0) == M, "Output tensor dimension mismatch: out.size(0) != M");
    TORCH_CHECK(out.size(1) == N, "Output tensor dimension mismatch: out.size(1) != N");
    
    // Use default HIP stream (stream 0)
    const hipStream_t stream = 0;
    
    // Call the C function
    run(a.data_ptr(), b.data_ptr(), as.data_ptr(), bs.data_ptr(), out.data_ptr(),
        M, N, K, nullptr, stream);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm(Tensor! out, Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> ()");
  ops.impl("gemm", torch::kCUDA, &gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```
The `torch_binding.h` file contains function declarations. For instance, the `gemm` kernel has the following declaration in `torch_binding.h`:

```h
#pragma once

#include <torch/torch.h>

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs);
```

### Setting up the `__init__.py` wrapper 

In `torch-ext/gemm/` we need an `__init__.py` file to make this directory a Python package and to expose our custom operator in a user-friendly way. 

```python
from typing import Optional
import torch
from ._ops import ops

def gemm(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs: torch.Tensor, 
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
         
    if out is None:
        # Create output tensor with appropriate shape and dtype
        M, K = a.shape
        K_b, N = b.shape
        assert K == K_b, f"Matrix dimension mismatch: A has {K} cols, B has {K_b} rows"
        
        # Output should be BF16 type on the same device as inputs
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    
    ops.gemm(out, a, b, as_, bs)
    return out
```

### Step 3: Building the Kernel

The kernel builder uses Nix for building kernels. You can build or run the kernels directly if you have Nix installed on your system. We recommend installing Nix in the following way:

- **Linux**: use the [official Nix installer](https://nixos.org/download/).
- **macOS**: use the [Determinate Nix installer](https://docs.determinate.systems/determinate-nix/). In addition, Xcode 16.x is currently required to build kernels.

#### Getting Started with Nix

First of all, run this:

```bash
nix flake update
```

This generates a `flake.lock` file that pins the kernel builder and all its transitive dependencies. Commit both `flake.nix` and `flake.lock` to your repository to ensure that kernel builds are reproducible.

Since the kernel builder depends on many packages (e.g., every supported PyTorch version), it is recommended to enable the Hugging Face cache to avoid expensive rebuilds:

```bash
# Install cachix and configure the cache
cachix use huggingface
```

Or run it once without installing cachix permanently:

```bash
# Use cachix without installing it
nix run nixpkgs#cachix -- use huggingface
```

#### Building Kernels with Nix

A kernel that has a `flake.nix` file can be built with the build-and-copy command:

```bash
cd Build_RadeonFlow_Kernels/gemm
nix build . -L
```

The compiled kernel will then be in the local `build/` directory.

#### Development Shell for Local Development

The kernel-builder provides shells for developing kernels. In such a shell, all required dependencies are available, as well as `build2cmake` for generating project files:

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ cmake -B build-ext
$ cmake --build build-ext
```

If you want to test the kernel as a Python package, you can do so. `nix develop` will automatically create a virtual environment in `.venv` and activate it:

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ pip install --no-build-isolation -e .
```

Development shells are available for every build configuration. For instance, you can get a Torch 2.7 development shell with ROCm 6.3 using:

```bash
$ rm -rf .venv  # Remove existing venv if any
$ nix develop .#devShells.torch27-cxx11-rocm63-x86_64-linux
```

### Step 4: Uploading the kernel to the Hub 

Now that we built our kernel, we can test it and upload it to the Hub. 

#### Building the Kernel for All PyTorch and ROCm Versions

One small thing we'll want to do before we share is clean up all of the development artifacts that were generated during the build process to avoid uploading unnecessary files. 

```bash
build2cmake clean build.toml 
```

To build the kernel for all supported versions of PyTorch and ROCm, the kernel-builder tool automates the process: 

```bash
# Outside of the dev shell, run the following command
# if you are inside of the sandbox you can leave with `exit`
nix build . -L
``` 

> **Note:**  
> This process may take a while, as it will build the kernel for all supported versions of PyTorch and ROCm.  
> The output will be in the `result` directory.

The last step is to move the results into the expected build directory (this is where the kernels library will look for them). 

```bash
mkdir -p build
rsync -av --delete --chmod=Du+w,Fu+w result/ build/
```

#### Pushing to the Hugging Face Hub 

Pushing the build artifacts to the Hub will make it straightforward for other developers to use your kernel. 

First, create a new repo: 

```bash
hf repo create gemm
```

> Make sure you are logged in to the Hugging Face Hub using huggingface-cli login.

Now, in your project directory, connect your project to the new repository and push your code:

```bash
# Initialize git and connect to the Hugging Face Hub
git init
git remote add origin https://huggingface.co/<your-username>/gemm

# Pull the changes (just the default .gitattributes file)
git pull origin main
git lfs install
git checkout -b main

# Update to use LFS for the binary files
git lfs track "*.so"

# Add and commit your changes (being careful to only include the necessary files
# since our build2cmake command generated a lot of dev-specific files)
git add \
  build/ gemm/ include/ src/utils tests/checker \
  torch-ext/torch_binding.cpp torch-ext/torch_binding.h torch-ext/gemm \
  flake.nix flake.lock build.toml

git commit -m "feat: Created a compliant gemm kernel"
git push -u origin main
```
Fantastic! Your kernel is now on the Hugging Face Hub, ready for others to use and fully compliant with the kernels library. 

### Step 5: Let's use it :) 

With the **kernels** library, you don't "install" the kernel in the traditional sense. You load it directly from its Hub repository, which automatically registers the new operator.

```python
import torch
from kernels import get_kernel

# Load the kernel from the Hub
gemm = get_kernel("kernels-community/gemm")

# Matrix dimensions (must be supported - see gemm_launcher.cpp)
M, N, K = 1024, 1536, 7168
QUANT_SIZE = 128

# Setup device
device = torch.device("cuda")

# Create inputs - kernel expects A:(K,M), B:(K,N)
A_fp32 = torch.randn(M, K, device=device)
B_fp32 = torch.randn(K, N, device=device)

# Convert to FP8
A_fp8 = A_fp32.to(torch.float8_e4m3fnuz)
B_fp8 = B_fp32.to(torch.float8_e4m3fnuz)

# Create scale factors (uniform scaling)
A_scale = torch.ones(K // QUANT_SIZE, M, device=device, dtype=torch.float32)
B_scale = torch.ones(K // QUANT_SIZE, N // QUANT_SIZE, device=device, dtype=torch.float32)

C = torch.zeros(M, N, device=device, dtype=torch.bfloat16)

# Use the kernel
result = gemm.gemm(A_fp8, B_fp8, A_scale, B_scale, C)
```

That's it! Your ROCm kernel is now ready to use from the Hugging Face Hub.

## Conclusion 

Building and sharing ROCm kernels with the Hugging Face is now easier than ever. With a clean, reproducible workflow powered by Nix and seamless integration into PyTorch, developers can focus on optimizing performance rather than setup. Once built, your custom kernel can be shared on the Hugging Face Hub; making it instantly accessible to the community and usable across projects with just a few lines of code. 🚀 

## Related Libraries & Hub

- [kernel-builder](https://github.com/huggingface/kernel-builder) – Build and compile custom kernels.
- [kernels](https://github.com/huggingface/kernels) – Library to manage and load kernels from the Hub.
- [Kernels Community Hub](https://huggingface.co/kernels-community) – Share and discover kernels from the community.


