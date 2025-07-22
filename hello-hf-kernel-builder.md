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

Let's Get Started! 🚀

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
    ├── torch_binding.cpp
    ├── torch_binding.h
    └── img2gray
        └── __init__.py
```

- **`build.toml`**: The project manifest; it’s the brain of the build process.
- **`csrc/`**: Your raw CUDA source code where the GPU magic happens.
- **`flake.nix`**: The key to a perfectly reproducible\* build environment.
- **`torch-ext/img2gray/`**: The Python wrapper for the raw PyTorch operators.

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
depends = ["torch"] # This kernel depends on the Torch library for the `Tensor` class.
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
void img2gray_cuda(torch::Tensor const &input, torch::Tensor &output) {
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

// First Define the operator's public signature in a new namespace.
TORCH_LIBRARY(img2gray, ops) {
    ops.def("img2gray(Tensor input, Tensor! output) -> ()");
}

// Next Link that signature to our actual CUDA C++ function.
TORCH_LIBRARY_IMPL(img2gray, CUDA, ops) {
    ops.impl("img2gray", &img2gray_cuda);
}
```

In simple terms, `TORCH_LIBRARY` sketches an outline of the function, and `TORCH_LIBRARY_IMPL` fills in that outline with our high-performance CUDA code.

#### **Why This Matters**

This approach is crucial for two main reasons:

- **Compatibility with `torch.compile`**: By registering our kernel this way, `torch.compile` can "see" it. This allows PyTorch to fuse your custom operator into larger computation graphs, minimizing overhead and maximizing performance. It's the key to making your custom code work seamlessly with PyTorch's broader performance ecosystem.

- **Hardware-Specific Implementations**: This system allows you to provide different backends for the same operator. You could add another `TORCH_LIBRARY_IMPL(img2gray, CPU, ...)` block pointing to a C++ CPU function. PyTorch's dispatcher would then automatically call the correct implementation—CUDA or CPU—based on the input tensor's device, making your code powerful and portable.

  **Setting up the `__init__.py` wrapper**

In the `torch-ext/img2gray/` we need an `__init__.py` file to make this directory a Python package and to expose our custom operator in a user-friendly way.

> [!NOTE]
> You'll see the `from ._ops import ops` line in the `__init__.py` file. This allows us to import the `ops` object that was registered in the C++ code, making it available in Python.

```python
# torch-ext/img2gray/ops.py
import torch

from ._ops import ops

def img2gray(input: torch.Tensor) -> torch.Tensor:
    # we expect input to be in BCHW format
    batch, channels, height, width = input.shape

    assert channels == 3, "Input image must have 3 channels (RGB)"

    output = torch.empty((batch, 1, height, width), device=input.device, dtype=input.dtype)

    for b in range(batch):
        single_image = input[b].permute(1, 2, 0).contiguous()  # HWC
        single_output = output[b].reshape(height, width)  # HW
        ops.img2gray(single_image, single_output)

    return output
```

### Step 6: Building the Kernel

Now that our kernel and its bindings are ready, it's time to build them. The `kernel-builder` tool simplifies this process.

You can build your kernel with a single command, `nix build . -L`; however, as developers, we'll want a faster, more iterative workflow. For that, we'll use the `nix develop` command to enter a development shell with all the necessary dependencies pre-installed.

More specifically, we can choose the exact CUDA and PyTorch versions we want to use. For example, to build our kernel for PyTorch 2.7 with CUDA 12.6, we can use the following command:

#### Drop into a Nix Shell

```bash
# Drop into a Nix shell (an isolated sandbox with all dependencies)
nix develop .#devShells.torch27-cxx11-cu126-x86_64-linux
```

Note that the `devShell` name above can be deciphered as:

```nix
nix develop .#devShells.torch27-cxx11-cu126-x86_64-linux
                        │       │         │       │
                        │       │         │       └─── Architecture: x86_64 (Linux)
                        │       │         └────────── CUDA version: 12.6
                        │       └──────────────────── C++ ABI: cxx11
                        └──────────────────────────── Torch version: 2.7
```

At this point, we'll be inside a Nix shell with all dependencies installed. We can now build the kernel.

#### Set Up Build Artifacts

```bash
build2cmake generate-torch build.toml
```

This command creates a handful of files used to build the kernel: `CMakeLists.txt`, `pyproject.toml`, `setup.py`, and a `cmake` directory. The `CMakeLists.txt` file is the main entry point for CMake to build the kernel.

#### Create a Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Now you can install the kernel in editable mode.

#### Compile the Kernel and Install the Python Package

```bash
pip install --no-build-isolation -e .
```

🙌 Amazing! We now have a custom-built kernel that follows best practices for Torch bindings, with a fully reproducible build process.

#### Sanity Check

To ensure everything is working, we can run a simple test to check if the kernel is registered correctly.

```python
# scripts/sanity.py
import torch
import img2gray # <- We can import the package now!
from PIL import Image
import numpy as np

img = Image.open("color.png").convert("RGB")
img_tensor = torch.from_numpy(np.array(img))
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).contiguous().cuda()  # BCHW

gray_tensor = img2gray.img2gray(img_tensor).squeeze()
Image.fromarray(gray_tensor.cpu().numpy()).save("gray.png")
```

### Step 7: Sharing with the World

Now that we have a working kernel, it's time to share it with other developers and the world\!

#### Building the Kernel for All Python and Torch Versions

Earlier, we built the kernel for a specific version of PyTorch and CUDA. However, to make it available to a wider audience, we need to build it for all supported versions. The `kernel-builder` tool can help us with that.

This is also where the concept of a `compliant kernel` comes into play. A compliant kernel is one that can be built and run for all supported versions of PyTorch and CUDA. Generally, this requires custom configuration; however, in our case, the `kernel-builder` tool will automate the process.

```bash
# Outside of the dev shell, run the following command
nix build . -L
```

> [!WARNING]
> This process may take a while, as it will build the kernel for all supported versions of PyTorch and CUDA. The output will be in the `result` directory.

The last step is to move the results into the expected `build` directory (this is where the `kernels` library will look for them).

```bash
mkdir -p build
rsync -av --delete --chmod=Du+w,Fu+w result/ build/
```

#### 7.2. Pushing to the Hugging Face Hub

First, create a new repo:

```bash
huggingface-cli repo create img2gray
```

> [!NOTE]
> Make sure you are logged in to the Hugging Face Hub using `huggingface-cli login`.

Now, in your project directory, connect your project to the new repository and push your code:

```bash
# Initialize git and connect to the Hugging Face Hub
git init
git remote add origin https://huggingface.co/<your-username>/img2gray

# Pull the changes (just the default .gitattributes file)
git pull origin main
git lfs install
git checkout -b main

# Update to use LFS for the binary files
git lfs track "*.so"

# Add and commit your changes
git add build/ csrc/ torch-ext/ flake.nix flake.lock build.toml
git commit -m "Initial commit"
git push -u origin main
```

Fantastic\! Your kernel is now on the Hugging Face Hub, ready for others to use and fully compliant with the `kernels` library.

### **Step 8: Loading and Testing Your Custom Op**

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

# Run Your Custom Operator
# Call the op directly from the torch.ops namespace!
output_tensor = img2gray_lib.img2gray(img_tensor, output_tensor)

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

Using a Git shorthash will reduce the chance of breakage, however it is hard to interpret and does not allow graceful upgrades within a version range. We therefore recommend to use [semantic versioning](https://semver.org/) for kernels. Adding a version to a kernel is easy: you simply add a Git tag of the form `vx.y.z` where _x.y.z_ is the version. For instance, if the current version of the kernel is 1.1.2, you can tag it as `v1.1.2`. You can then get that version with `get_kernel`:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", revision="v1.1.2")
```

Versioning becomes even more powerful with version bounds. In semantic versioning, the version `1.y.z`, must not have backward-incompatible changes in the public API for each succeeding `x` and `y`. So, if the kernel's version was `1.1.2` at the time of writing your code, you can ask the version to be at least `1.2.1`, but less than `2.0.0`:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", version=">=1.1.2,<2")
```

This will ensure that the code will always fetch the latest kernel from the `1.y.z` series. The version bound can be a Python-style [version specifier](https://packaging.pypa.io/en/stable/specifiers.html).

You can [tag](https://huggingface.co/docs/huggingface_hub/en/guides/cli#tag-a-model) a version with `huggingface-cli`:

```bash
$ huggingface-cli tag drbh/img2gray v1.1.2
```

#### Locking Kernels

In large projects, you may want to coordinate the kernel versions globally rather than in each `get_kernel` call. Moreover, it is often useful to lock kernels, so that all your users have the same kernel versions, which aids handling bug reports.

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
$ kernels to-wheel drbh/img2grey 1.1.2
☸ img2grey-1.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu124cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu118cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu124cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
```

This wheel will be have like any other wheel, the kernel can be imported using a simple `import img2grey`.

# Conclusion

tbd
