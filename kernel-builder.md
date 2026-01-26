---
title: "From Zero to GPU: A Guide to Building and Scaling Production-Ready CUDA Kernels"
thumbnail: /blog/assets/kernel-builder/kernel-builder.png
authors:
  - user: drbh
  - user: danieldk
---

# From Zero to GPU: A Guide to Building and Scaling Production-Ready CUDA Kernels

Custom CUDA kernels give your models a serious performance edge, but building them for the real world can feel daunting. How do you move beyond a simple GPU function to create a robust, scalable system without getting bogged down by endless build times and dependency nightmares?

We created the [`kernel-builder` library](https://github.com/huggingface/kernels) for this purpose. You can develop a custom kernel locally, and then build it for multiple architectures and make it available for the world to use.

In this guide we'll show you how to build a complete, modern CUDA kernel from the ground up. Then, we’ll tackle the tough production and deployment challenges, drawing on real-world engineering strategies to show you how to build systems that are not just fast, but also efficient and maintainable.

## What You’ll Learn

When you're done, other developers will be able to use your kernels directly from the hub like this:

```python
import torch

from kernels import get_kernel

# Download custom kernel from the Hugging Face Hub
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

### Step 1: Project Structure

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

### Step 2: The `build.toml` Manifest

This file orchestrates the entire build. It tells the `kernel-builder` what to compile and how everything connects.

```toml
# build.toml
[general]
name = "img2gray"
universal = false

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

### Step 3: The `flake.nix` Reproducibility File

To ensure anyone can build your kernel on any machine, we use a `flake.nix` file. It locks the exact version of the `kernel-builder` and its dependencies, eliminating "it works on my machine" issues.

```nix
# flake.nix
{
  description = "Flake for img2gray kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
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

### Step 4: Writing the CUDA Kernel

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

        // Luminance conversion
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

### Step 5: Registering a Native PyTorch Operator

This is the most important step. We're not just binding to Python; we're registering our function as a native PyTorch operator. This makes it a first-class citizen in the PyTorch ecosystem, visible under the `torch.ops` namespace.

The file **`torch-ext/torch_binding.cpp`** handles this registration.

```cpp
// torch-ext/torch_binding.cpp
#include <torch/library.h>
#include "registration.h" // included in the build
#include "torch_binding.h" // Declares our img2gray_cuda function

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("img2gray(Tensor input, Tensor! output) -> ()");
  ops.impl("img2gray", torch::kCUDA, &img2gray_cuda);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

```

In simple terms, `TORCH_LIBRARY_EXPAND` allows us to define our operator in a way that can be easily extended or modified in the future.

#### Why This Matters

This approach is crucial for two main reasons:

- **Compatibility with `torch.compile`**: By registering our kernel this way, `torch.compile` can "see" it. This allows PyTorch to fuse your custom operator into larger computation graphs, minimizing overhead and maximizing performance. It's the key to making your custom code work seamlessly with PyTorch's broader performance ecosystem.

- **Hardware-Specific Implementations**: This system allows you to provide different backends for the same operator. You could add another `TORCH_LIBRARY_IMPL(img2gray, CPU, ...)` block pointing to a C++ CPU function. PyTorch's dispatcher would then automatically call the correct implementation (CUDA or CPU) based on the input tensor's device, making your code powerful and portable.

  **Setting up the `__init__.py` wrapper**

In `torch-ext/img2gray/` we need an `__init__.py` file to make this directory a Python package and to expose our custom operator in a user-friendly way.

> [!NOTE]
> The `_ops` module is auto-generated by kernel-builder from a [template](https://github.com/huggingface/kernels/blob/3f2398f0b2f6ad0dee2f62b2b65e66f35e5b8425/build2cmake/src/templates/_ops.py) to provide a standard namespace for your registered C++ functions.

```python
# torch-ext/img2gray/__init__.py
import torch

from ._ops import ops


def img2gray(input: torch.Tensor) -> torch.Tensor:
    # we expect input to be in CHW format
    height, width, channels = input.shape
    assert channels == 3, "Input image must have 3 channels (RGB)"

    output = torch.empty((height, width), device=input.device, dtype=input.dtype)
    ops.img2gray(input, output)

    return output

```

### Step 6: Building the Kernel

Now that our kernel and its bindings are ready, it's time to build them. The `kernel-builder` tool simplifies this process.

You can build your kernel with a single command, `nix build . -L`; however, as developers, we'll want a faster, more iterative workflow. For that, we'll use the `nix develop` command to enter a development shell with all the necessary dependencies pre-installed.

More specifically, we can choose the exact CUDA and PyTorch versions we want to use. For example, to build our kernel for PyTorch 2.7 with CUDA 12.6, we can use the following command:

#### Drop into a Nix Shell

```bash
# Drop into a Nix shell (an isolated sandbox with all dependencies)
nix develop .#devShells.torch28-cxx11-cu128-x86_64-linux
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

At this point, we'll be inside a Nix shell with all dependencies installed. We can now build the kernel for this particular architecture and test it. Later on, we'll deal with multiple architectures before distributing the final version of the kernel.

#### Set Up Build Artifacts

```bash
build2cmake generate-torch build.toml
# this creates: torch-ext/img2gray/_ops.py, pyproject.toml, torch-ext/registration.h, setup.py, cmake/hipify.py, cmake/utils.cmake, CMakeLists.txt
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

🙌 Amazing! We now have a custom built kernel that follows best practices for PyTorch bindings, with a fully reproducible build process.

#### Development Cycle

To ensure everything is working correctly, we can run a simple test to check that the kernel is registered and it works as expected. If it doesn't, you can iterate by editing the source files and repeating the build, reusing the nix environment you created.

```python
# scripts/sanity.py
import torch
import img2gray

from PIL import Image
import numpy as np

print(dir(img2gray))

img = Image.open("kernel-builder-logo-color.png").convert("RGB")
img = np.array(img)
img_tensor = torch.from_numpy(img).cuda()
print(img_tensor.shape)  # HWC

gray_tensor = img2gray.img2gray(img_tensor).squeeze()
print(gray_tensor.shape)  # HW

# save the output image
gray_img = Image.fromarray(gray_tensor.cpu().numpy().astype(np.uint8), mode="L")
gray_img.save("kernel-builder-logo-gray.png")

```

### Step 7: Sharing with the World

Now that we have a working kernel, it's time to share it with other developers and the world!

One small thing we'll want to do before we share, is clean up all of the development artifacts that were generated during the build process to avoid uploading unnecessary files.

```bash
build2cmake clean build.toml
```

#### Building the Kernel for All PyTorch and CUDA Versions

Earlier, we built the kernel for a specific version of PyTorch and CUDA. However, to make it available to a wider audience, we need to build it for all supported versions. The `kernel-builder` tool can help us with that.

This is also where the concept of a `compliant kernel` comes into play. A compliant kernel is one that can be built and run for all supported versions of PyTorch and CUDA. Generally, this requires custom configuration; however, in our case, the `kernel-builder` tool will automate the process.

```bash
# Outside of the dev shell, run the following command
# if you are inside of the sandbox you can leave with `exit`
nix build . -L
```

> [!WARNING]
> This process may take a while, as it will build the kernel for all supported versions of PyTorch and CUDA. The output will be in the `result` directory.

> [!NOTE]
> The kernel-builder team actively maintains the [supported build variants](https://huggingface.co/docs/kernels/en/builder/build-variants), keeping them current with the latest PyTorch and CUDA releases while also supporting trailing versions for broader compatibility.

The last step is to move the results into the expected `build` directory (this is where the `kernels` library will look for them).

```bash
mkdir -p build
rsync -av --delete --chmod=Du+w,Fu+w result/ build/
```

#### Pushing to the Hugging Face Hub

Pushing the build artifacts to the Hub will make it straightforward for other developers to use your kernel, as we saw in [our previous post](https://huggingface.co/blog/hello-hf-kernels). We can use the `kernels upload` command for this:

```bash
kernels upload <path_to_kernel> --repo_id hub-username/img2gray
```

<details>
<summary>You can also follow a standard git-based process for the upload.

First, create a new repo:

```bash
hf repo create img2gray
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

# Add and commit your changes. (being careful to only include the necessary files
# since our build2cmake command generated a lot of dev specific files)
git add \
  build/ csrc/ \
  torch-ext/torch_binding.cpp torch-ext/torch_binding.h torch-ext/img2gray \
  flake.nix flake.lock build.toml

git commit -m "feat: Created a compliant img2gray kernel"
git push -u origin main
```

</details>

Fantastic! Your kernel is now on the Hugging Face Hub, ready for others to use and fully compliant with the `kernels` library. Our kernel and all of its build variants are now available at [drbh/img2gray](https://huggingface.co/drbh/img2gray/tree/main/build).

### Step 8: Loading and Testing Your Custom Op

With the `kernels` library, you don't "install" the kernel in the traditional sense. You load it directly from its Hub repository, which automatically registers the new operator.

```python
# /// script
# requires-python = "==3.10"
# dependencies = [
#     "kernels",
#     "numpy",
#     "pillow",
#     "torch",
# ]
# ///
import torch
from PIL import Image
import numpy as np
from kernels import get_kernel

# This downloads, caches, and loads the kernel library
# and makes the custom op available in torch.ops
img2gray_lib = get_kernel("drbh/img2gray")

img = Image.open("kernel-builder-logo-color.png").convert("RGB")
img = np.array(img)
img_tensor = torch.from_numpy(img).cuda()
print(img_tensor.shape)  # HWC

gray_tensor = img2gray_lib.img2gray(img_tensor).squeeze()
print(gray_tensor.shape)  # HW

# save the output image
gray_img = Image.fromarray(gray_tensor.cpu().numpy().astype(np.uint8), mode="L")
gray_img.save("kernel-builder-logo-gray2.png")

```

---

## Part 2: From One Kernel to Many: Solving Production Challenges

Once you have a ready-to-use kernel, there are some things you can do to make it easier to deploy your kernel. We will discuss using versioning as a tool to make API changes without breaking downstream use of kernels. After that, we will wrap up showing how you can make Python wheels for your kernel.

### Kernel Versions

You might decide to update your kernel after a while. Maybe you have found new ways of improving performance or perhaps you would like to extend the kernel's functionality. Some changes will require you to change the API of your kernel. For instance, a newer version might add a new mandatory argument to one of the public functions. This can be inconvenient to downstream users, because their code would break until they add this new argument.

A downstream user of a kernel can avoid such breakage by pinning the kernel that they use to a particular revision. For instance, since each Hub repository is also a Git repository, they could use a git commit shorthash to pin the kernel to a revision:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", revision="4148918")
```

Using a Git shorthash will reduce the chance of breakage; however, it is hard to interpret and does not allow graceful upgrades within a version range. We therefore recommend using the familiar [semantic versioning](https://semver.org/) system for Hub kernels. Adding a version to a kernel is easy: you simply add a Git tag of the form `vx.y.z` where _x.y.z_ is the version. For instance, if the current version of the kernel is 1.1.2, you can tag it as `v1.1.2`. You can then get that version with `get_kernel`:

```python
from kernels import get_kernel
img2gray_lib = get_kernel("drbh/img2gray", revision="v1.1.2")
```

Versioning becomes even more powerful with version bounds. In semantic versioning, the version `1.y.z`, must not have backward-incompatible changes in the public API for each succeeding `x` and `y`. So, if the kernel's version was `1.1.2` at the time of writing your code, you can ask the version to be at least `1.1.2`, but less than `2.0.0`:

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

The `kernels` library offers a nice way of managing kernels at the project-level. To do so, add the `kernels` package to the build-system requirements of your project, in the `pyproject.toml` file. After doing so, you can specify your project's kernel requirements in the `tools.kernels` section:

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

This generates a `kernels.lock` file with the latest kernel versions that are compatible with the bounds that are specified in `pyproject.toml`. `kernels.lock` should be committed to your project's Git repository, so that every user of the project will get the locked kernel versions. When newer kernels versions are released, you can run `kernels lock` again to update the lock file.

You need one last bit to fully implement locked kernels in a project. The `get_locked_kernel` is the counterpart to `get_kernel` that uses locked kernels. So to use locked kernels, replace every occurrence of `get_kernel` with `get_locked_kernel`:

```python
from kernels import get_kernel
img2gray_lib = get_locked_kernel("drbh/img2gray")
```

That's it! Every call of `get_locked_kernel("drbh/img2gray")` in the project will now use the version specified in `kernels.lock`.

### Pre-downloading Locked Kernels

The `get_locked_kernel` function will download the kernel when it is not available in the local Hub cache. This is not ideal for applications where you do not want to download binaries at runtime. For example, when you are building a Docker image for an application, you usually want the kernels to be stored in the image along with the application. This can be done in two simple steps.

First, use the `load_kernel` function in place of `get_locked_kernel`:

```python
from kernels import get_kernel
img2gray_lib = load_kernel("drbh/img2gray")
```

As the name suggests, this function will only load a kernel, it will _never_ try to download the kernel from the Hub. `load_kernel` will raise an exception if the kernel is not locally available. So, how do you make the kernels locally available? The `kernels` utility has you covered! Running `kernels download .` will download the kernels that are specified in `kernels.lock`. So e.g. in a Docker container you could add a step:

```docker
RUN kernels download /path/to/your/project
```

and the kernels will get baked into your Docker image.

> [!NOTE]
> Kernels use the standard Hugging Face cache, so all [HF_HOME](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) caching rules apply.

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

Each of these wheels will behave like any other Python wheel: the kernel can be imported using a simple `import img2grey`.

# Conclusion

This guide has walked you through the entire lifecycle of a production-ready CUDA kernel. You’ve seen how to build a custom kernel from the ground up, register it as a native PyTorch operator, and share it with the community on the Hugging Face Hub. We also explored best practices for versioning, dependency management, and deployment, ensuring your work is both powerful and easy to maintain.

We believe that open and collaborative development is the key to innovation. Now that you have the tools and knowledge to build your own high-performance kernels, we're excited to see what you create! We warmly invite you to share your work, ask questions, and start discussions on the [Kernel Hub](https://huggingface.co/kernels-community) or in our [mono repo](https://github.com/huggingface/kernels) for `kernels` and `kernel-builder`. Whether you’re a seasoned developer or just starting out, the community is here to support you.

Let's get building! 🚀
