---
title: "A Guide to Building and Scaling Production-Ready CUDA Kernels"
thumbnail: /blog/assets/kernel-builder-tutorial/hello-hf-kernel-builder.png
authors:
  - user: drbh
  - user: danieldk
date: 2025-07-17
---

# From Zero to GPU: A Guide to Building and Scaling Production-Ready CUDA Kernels

Custom CUDA kernels give your models a serious performance edge, but building them for the real world can feel daunting. How do you move beyond a simple function and create a robust, scalable system without getting bogged down by endless build times and dependency nightmares?

This guide has you covered. We'll start by building a complete, modern CUDA kernel from the ground up. Then, we’ll tackle the tough production challenges, drawing on real-world engineering strategies to show you how to build systems that are not just fast, but also efficient and maintainable.

## What You’ll Learn

When you're done, other developers will be able to use your kernels directly from the hub like

```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
optimized_kernel = get_kernel("your-username/optimized-kernel")

# A sample input tensor
some_input = torch.randn((10, 10), device="cuda")

# Run the kernel
out = optimized_kernel.my_kernel_function(some_input)

print(out)
```

Rather watch a video? Check out the [YouTube video](https://youtu.be/HS5Pp_NLVWg?si=WP1aJ98q52lJn6F-&t=0) that accompanies this guide.

Lets Get Started! 🚀

## Part 1: Anatomy of a Modern CUDA Kernel

Let's build a practical kernel that converts an image from RGB to grayscale. This example uses PyTorch's modern C++ API to register our function as a first-class, native operator.

### **Step 1: The Project Structure**

A clean, predictable structure is the foundation of a good project. The Hugging Face Kernel Builder expects your files to be organized like this:

```bash
img2gray/
├── build.toml
├── csrc
│   └── img2gray.cu
├── flake.nix
└── torch-ext
    ├── registration.h
    ├── torch_binding.cpp
    └── torch_binding.h
```

- **`build.toml`**: The project manifest; it’s the brain of the build process.
- **`csrc/`**: Your raw CUDA source code where the GPU magic happens.
- **`flake.nix`**: The key to a perfectly reproducible\* build environment.
- **`torch-ext/`**: The C++ code that builds the bridge to PyTorch.

### **Step 2: The `build.toml` Manifest**

This file orchestrates the entire build. It tells the `kernel-builder` what to compile and how everything connects.

```toml
# build.toml
[general]
name = "img2gray"

# Defines the C++ files that bind to PyTorch
[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h"
]

# Defines the CUDA kernel itself
[kernel.img2gray]
backend = "cuda"
depends = ["torch"] # This kernel depends on the torch bindings
src = [
    "csrc/img2gray.cu",
]
```

### **Step 3: The `flake.nix` for Reproducibility**

To ensure anyone can build your kernel on any machine, we use a `flake.nix` file. It locks the exact version of the `kernel-builder` and its dependencies, eliminating "it works on my machine" issues.

```nix
# flake.nix
{
  description = "Flake for img2gray kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
    };
}
```

### **Step 4: Writing the CUDA Kernel**

Now for the GPU code. Inside **`csrc/img2gray.cu`**, we'll define a kernel that uses a 2D grid of threads—a natural and efficient fit for processing images.

```cpp
// csrc/img2gray.cu
#include <cstdint>
#include <torch/torch.h>

// This kernel runs on the GPU, with each thread handling one pixel
__global__ void img2gray_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // 3 channels for RGB
        uint8_t r = input[idx];
        uint8_t g = input[idx + 1];
        uint8_t b = input[idx + 2];

        // Luminosity conversion
        uint8_t gray = static_cast<uint8_t>(0.21f * r + 0.72f * g + 0.07f * b);
        output[y * width + x] = gray;
    }
}

// This C++ function launches our CUDA kernel
void img2gray_cuda(torch::Tensor input, torch::Tensor output) {
    const int width = input.size(1);
    const int height = input.size(0);

    // Define a 2D block and grid size for image processing
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    img2gray_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        width,
        height
    );
}
```

### **Step 5: Registering a Native PyTorch Operator**

This is the most important step. We're not just binding to Python; we're registering our function as a native PyTorch operator. This makes it a first-class citizen in the PyTorch ecosystem, visible under the `torch.ops` namespace.

The file **`torch-ext/torch_binding.cpp`** handles this registration.

```cpp
// torch-ext/torch_binding.cpp
#include <torch/library.h>
#include "torch_binding.h" // Declares our img2gray_cuda function

// 1. Define the operator's public signature in a new namespace.
TORCH_LIBRARY(img2gray, ops) {
    ops.def("img2gray(Tensor input, Tensor output) -> ()");
}

// 2. Link that signature to our actual CUDA C++ function.
TORCH_LIBRARY_IMPL(img2gray, CUDA, ops) {
    ops.impl("img2gray", &img2gray_cuda);
}
```

In simple terms, `TORCH_LIBRARY` sketches an outline of the function, and `TORCH_LIBRARY_IMPL` fills in that outline with our high-performance CUDA code.

#### **Why This Matters**

This approach is crucial for two main reasons:

- **Compatibility with `torch.compile`**: By registering our kernel this way, `torch.compile` can "see" it. This allows PyTorch to fuse your custom operator into larger computation graphs, minimizing overhead and maximizing performance. It's the key to making your custom code work seamlessly with PyTorch's broader performance ecosystem.

- **Hardware-Specific Implementations**: This system allows you to provide different backends for the same operator. You could add another `TORCH_LIBRARY_IMPL(img2gray, CPU, ...)` block pointing to a C++ CPU function. PyTorch's dispatcher would then automatically call the correct implementation—CUDA or CPU—based on the input tensor's device, making your code powerful and portable.

### **Step 6: Loading and Testing Your Custom Op**

With the `kernels` library, you don't "install" the kernel in the traditional sense. You load it directly from its Hub repository, which automatically registers the new operator.

```python
# test_kernel.py
import torch
from PIL import Image
import numpy as np
from kernels import get_kernel

# This downloads, caches, and loads the kernel library
# and makes the custom op available in torch.ops
img2gray_lib = get_kernel("drbh/img2gray")

# Prepare Data
img = Image.open("my_rgb_image.png").convert("RGB")
img_tensor = torch.from_numpy(np.array(img)).cuda()
output_tensor = torch.empty(img_tensor.shape[0], img_tensor.shape[1], dtype=torch.uint8, device="cuda")

# Run Your Custom Operator
# Call the op directly from the torch.ops namespace!
torch.ops.img2gray.img2gray(img_tensor, output_tensor)

# Save the Result
Image.fromarray(output_tensor.cpu().numpy()).save("my_grayscale_image.png")
print("Image converted successfully using a native PyTorch operator! ✅")
```

---

## Part 2: From One Kernel to Many: Solving Production Challenges

Once you have a ready-to-use kernel, there are some things you can do to make it easier to deploy your kernel. We will discuss using versioning as a tool to make API changes without breaking downstream use of kernels. After that, we will wrap up showing how you can make Python wheels for your kernel.

### Kernel Versions

You might decide to update your kernel after a while. Maybe you have found new ways of improving performance or perhaps you would like to extend the kernel's functionality. Some changes will require you to change the API of our kernel. For instance, a newer version might add a new mandatory argument to one of the functions provided by your kernel. This can be inconvenient to downstream users, because their code would break until they add this new argument.

A downstream user of a kernel can avoid such breakage by pinning the kernel that they use to a particular revision. For instance, since each a Hub repository is also a Git repository, they could use a Git commit shorthash to pin the kernel to a revision:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", revision="4148918")
```

Using a Git shorthash will reduce the chance of breakage, however it is hard to interpret and does not allow graceful upgrades within a version range. We therefore recommend to use [semantic versioning](https://semver.org/) for kernels. Adding a version to a kernel is easy: you simply add a Git tag of the form `vx.y.z` where _x.y.z_ is the version. For instance, if the current version of the kernel is 0.1.2, you can tag it as `v0.1.2`. The user could then get that version with `get_kernel`:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", revision="v0.1.2")
```

**NOTE (daniel): while writing this, I realize it's nice to be able to support version specifiers in `get_kernel`. It makes the semantic versioning story stronger and makes the transition to lockfiles (see below) more natural. I will now implement this feature and update this section afterwards.**

You can [tag](https://huggingface.co/docs/huggingface_hub/en/guides/cli#tag-a-model) a version with `huggingface-cli`:

```bash
$ huggingface-cli tag drbh/img2gray v0.1.2
```

#### Locking Kernels

Tags make it easy to version your kernels. However, managing versions in code can be unwieldy in large projects. If you want to change the version of a kernel, you will have to find all occurrences of `get_kernel` for that kernel and update their versions.

`kernels` offers a nice way of managing kernels at the project-level. To do so, add the `kernels` package to the build-system requirements in `pyproject.toml`. After doing so, you can specify a project's kernel requirements in the `tools.kernels` section:

```toml
[build-system]
requires = ["kernels", "setuptools"]
build-backend = "setuptools.build_meta"

[tool.kernels.dependencies]
"drbh/img2gray" = ">=0.1.2,<0.2.0"
```

The version can be specified with the same type of version specifiers as Python dependencies. This is another place where the version tags (`va.b.c`) come handy -- `kernels` will use a repository's version tags to query what versions are available. After specifying a kernel in `pyproject.toml`, you can lock it to a specific version using the `kernels` command-line utility. This utility is part of the `kernels` Python package:

```bash
$ kernels lock .
```

This generates a `kernels.lock` file with the latest kernel versions that are compatible with the bounds that are specified in `pyproject.toml`. `kernels.lock` should be committed to the project's Git repository, so that every user of the project will get the locked kernel versions. When newer kernels versions are released, you can run `kernels lock` again to update the lock file.

You need one last bit to fully implement locked kernels in a project. The `get_locked_kernel` is the counterpart to `get_kernel` that uses locked kernels. So to use locked kernels, replace every occurrence of `get_kernel` with `get_locked_kernel`:

```python
from kernels import get_kernel
img2gray_lib = get_locked_kernel("drbh/img2gray")
```

That's it! Every call of `get_locked_kernel("drbh/img2gray")` in the project will now use the version specified in `kernels.lock`.

### Pre-downloading Locked Kernels

The `get_locked_kernel` function will download the kernel when it is not available in the local Hub cache. This is not ideal for applications where you do not want to binaries at runtime. For example, when you are building a Docker image for an application, you usually want the kernels to be stored in the image along with the application. This can be done in two simple steps.

First, use the `load_kernel` function in place of `get_locked_kernel`:

```python
from kernels import get_kernel
img2gray_lib = load_kernel("drbh/img2gray")
```

As the name suggests, this function will only load a kernel, it will _never_ try to download the kernel from the Hub. `load_kernel` will raise an exception if the kernel is locally available. So, how do you make the kernels locally available? The `kernels` utility has you covered! Running `kernels download .` will download the kernels that are specified in `kernels.lock`. So e.g. in a Docker container you could add a step:

```docker
RUN kernels download /path/to/your/project
```

and the kernels will get baked into your Docker image.

### Creating Legacy Python Wheels

We strongly recommend downloading kernels from the Hub using the `kernels` package. This has many benefits:

- `kernels` supports loading multiple kernel versions of the same kernel in a Python process.
- `kernels` will automatically download a version of a kernel that is compatible with the CUDA and Torch versions of your environment.
- You will get all the benefits of the Hub: analytics, issue tracking, pull requests, forks, etc.
- The Hub and `kernel-builder` provide provenance and reproducibility, a user can see a kernel's source history and rebuild it in the same build environment for verification.

That said, some projects may require deployment of kernels as wheels. The `kernels` utility provides a simple solution to this. You can convert any Hub kernel into a set of wheels with a single command:

```bash
$ kernels to-wheel drbh/img2grey 0.1.2
☸ img2grey-0.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch26cu124cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-0.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-0.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-0.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-0.1.2+torch26cu118cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch26cu124cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch26cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-0.1.2+torch27cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
```

This wheel will be have like any other wheel, the kernel can be imported using a simple `import img2grey`.

# Conclusion

tbd

