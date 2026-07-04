---
title: "🤗 Kernels: Major Updates"
thumbnail: /blog/assets/revamped-kernels/thumbnail.png
authors:
  - user: sayakpaul
  - user: danieldk
  - user: drbh
---


# 🤗 Kernels: Major Updates

In our [previous post (From Zero to GPU)](https://huggingface.co/blog/kernel-builder), we introduced the 🤗 Kernels project, which aims at standardizing how custom kernels are packaged, distributed, and consumed. We want the project to be frictionless and secure, while making it as Hub-friendly as possible.

Over the past few months, we have worked towards this goal. In the process, we also almost completely redesigned the project. This post will summarize the major updates we have shipped and what’s coming.

**Table of contents**

* [Kernels – a new repository type](#kernels--a-new-repository-type)
* [Improved security](#improved-security)
* [Revamped CLIs](#revamped-clis)
* [More coverage of frameworks and backends](#more-coverage-of-frameworks-and-backends)
* [Foundation for agentic kernel development](#foundation-for-agentic-kernel-development)
* [Misc](#misc)
* [Conclusion](#conclusion)

## Kernels – a new repository type

We have introduced a new repository type on the Hub called [“kernels”](https://huggingface.co/kernels). This enables us to cater to users with compute-related specificities. For example, a user can get a sense of which accelerators, operating systems, and backend versions are supported for a given kernel:

<figure align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/revamped-kernels/flash-attn3.png" alt="Flash Attention 3 Kernel Page" width="600"/>
    <figcaption>Kernel page: <a href="https://huggingface.co/kernels/kernels-community/flash-attn3">kernels-community/flash-attn3</a></figcaption>
</figure>

One can browse all available kernels on the Hub here: [https://huggingface.co/kernels](https://huggingface.co/kernels). 

Making these kernels first-class citizens of the Hub also benefits the AI ecosystem. Users can now see trends across kernels, models, and the applications that use them. The kernels become more discoverable to users.

## Improved security

Security has always been of utmost importance to the kernels project. This has led to an early focus on reproducibility – a kernel user should be able to recompile a kernel to verify that it was compiled from publicly available source code. This is made possible by using Nix, which makes a build as pure as possible through hermetic evaluation of the build recipe and a strongly isolated build sandbox. We further improve provenance by embedding the source Git SHA1 into the kernel itself.

In recent months, we have added additional layers of defense: trusted kernel publishers and code signing.

### Trusted kernel publishers

With the new repo type, we also introduced “trusted publishers”. Since kernels execute code on a machine with the same privileges as the Python process they are used in, an attacker could compromise machines by uploading a malicious kernel and coaxing you to use that kernel. To help you avoid such malicious kernels, the kernels package will now only load kernels by *trusted publishers* by default. A trusted publisher is an organization that is trusted by the community to act in good faith.

We still want to support loading kernels from organizations or users that are not trusted publishers, but you have to explicitly opt in using the `trust_remote_code` argument when loading a kernel from the Hub:

```py 
from kernels import get_kernel

kernel_module = get_kernel(  
   “Atlas-Inference/gdn”, version=1, trust_remote_code=True  
)  
```

By default, users cannot publish kernel repositories on the Hub. They have to request to be a kernel publisher. Users and organizations can request for access from their account settings. This gives us time to treat these requests on a case-by-case basis.

### Kernel signing

An additional layer of security that we are adding is code signing. Code signing protects against the scenario where an attacker uploads a malicious kernel to a kernel repo from a trusted publisher whose Hub credentials were compromised. In code signing, a kernel is signed with a private key known only to the kernel developer and validated with a public key that is generally available. In the Hub compromise scenario, an attacker cannot sign the malicious kernel since they do not own the private key needed for signing.

To further improve security, we use Sigstore’s cosign to sign using ephemeral private keys. Since these signing keys are only valid for a limited time, an attacker typically cannot use the private key, even when it is leaked. We also verify that the kernel was signed by a trusted GitHub workflow from a trusted GitHub repository.

Kernel signing is already supported by `kernel-builder` and we have provided the `kernels verify-signature` to verify a kernel. Kernels does not verify the signature upon loading a kernel yet, since we would like to test this new functionality more before fully rolling it out. Preliminary notes on setting up code signing for your own kernels can be found in the kernels 0.16.0 release notes: [https://github.com/huggingface/kernels/releases/tag/v0.16.0](https://github.com/huggingface/kernels/releases/tag/v0.16.0). 

## Revamped CLIs

Previously, a bunch of utilities were intertwined between `kernels` and `kernel-builder`. We have established a better separation of concern between the CLI of `kernels` and `kernel-builder`. The mental model here is that `kernels` is a library for loading and preparing kernels for use. Therefore, it should not include anything related to “building” kernels. 

As a result of this, both `kernels` and `kernel-builder` are now much leaner and more specific. Refer to the documentation to learn more about this:

* [`kernels` CLI](https://huggingface.co/docs/kernels/en/cli)  
* [`kernel-builder` CLI](https://huggingface.co/docs/kernels/en/builder-cli)

## More coverage of frameworks and backends {#more-coverage-of-frameworks-and-backends}

We have extended support for frameworks, the most visible changes are:

* We added support for the Torch Stable ABI to kernels and kernel-builder. The Torch Stable ABI allows kernel developers to target a particular Torch version or any version that is released after it for roughly two years. For instance, a kernel that targets the Torch 2.9 Stable ABI support Torch \>= 2.9.  
* Apache TVM FFI is the first framework to be supported besides Torch. TVM FFI is standardized ABI for kernels that interoperates with other frameworks such as PyTorch, Jax and CuPy. This allows kernel developers to make kernels that runs across frameworks.

## Foundation for agentic kernel development

`kernel-builder` and `kernels` complement the rise of agentic kernel development wherein an agent is leveraged to come up with an (optimized) kernel from scratch. Together, they support a workflow in which agents can scaffold, build, benchmark, and iteratively optimize kernels.

Agentic kernel development is still nascent, and the right development loops will continue to evolve. That makes simple, clear fundamentals especially important where the tools should be easy to compose into whichever agent workflows or frameworks people choose to use.

`kernel-builder` helps enforce a structure in how kernel source code should be scaffolded and used to perform reproducible builds. This gives agents a predictable project layout and repeatable workflow to operate within. Its CLI is also meant to be [agent-optimized](https://huggingface.co/blog/is-it-agentic-enough). For example, this can mean non-interactive commands and outputs that are straightforward for an agent to interpret programmatically. To this end, we also have [backend-specific skills](https://huggingface.co/docs/kernels/en/cli-skills) to help agents navigate the idiosyncrasies of different backends. These skills can capture backend-specific toolchains, compilation paths, and performance considerations.

Building a kernel successfully isn’t the only goal, we need to ensure that it delivers actual speedups over a baseline on the target hardware. A successful build is therefore only the first validation step. Usually, this target hardware can include many different accelerators, even different families of the same accelerator.

 This makes it important to evaluate results across hardware vendors and generations where relevant. Our tight [integration with HF Jobs](https://huggingface.co/docs/kernels/en/builder/github-actions) can make this benchmarking process easy. Agents can use this integration to run benchmark suites, collect performance results, and compare them against a defined baseline.
 
This way, agents can run tests across different hardware configurations to get reliable feedback on the performance of the generated kernels and identify what needs to be done. That feedback can then inform the next optimization iteration.

Below are some examples of agent-augmented kernels. These illustrate the kinds of kernels that can be developed and evaluated through this workflow: 

* [https://huggingface.co/kernels/drbh/yamoe](https://huggingface.co/kernels/drbh/yamoe)   
* [https://huggingface.co/kernels/sayakpaul/qk-norm-rope](https://huggingface.co/kernels/sayakpaul/qk-norm-rope) 

## Misc

### Environment setup

The environment setup for building kernels with `kernel-builder` can be daunting. To make it easier for users, we now have an [installation script](https://huggingface.co/docs/kernels/en/builder/writing-kernels#quick-install) for setting up an environment in one click. If you prefer working with ephemeral instances, our [Terraform setup guide](https://github.com/huggingface/kernels/tree/main/terraform) is worth following. 

### System card for kernels

After the kernels are built, we create a system card for each kernel to expose useful information, including how to use it and its exposed interfaces. When the kernel is pushed to the Hub, this system card becomes the front matter for the kernel:

<figure align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/revamped-kernels/kernel-card.png" alt="System card for a kernel" width="600"/>
    <figcaption>System card for <a href="https://huggingface.co/kernels/kernels-community/flash-attn3">kernels-community/flash-attn3</a></figcaption>
</figure>

### Is a kernel compatible on my system?

Is a question one would ask multiple times to plan things better. Use the [`has_kernel()`](https://huggingface.co/docs/kernels/main/en/api/kernels#kernels.has_kernel) method for this purpose:

```py 
from kernels import has_kernel

print(has_kernel("kernels-community/activation", version=1))  
```

It returns a `bool`. If you’re looking for more explanations around why a given kernel isn’t supported then use [`get_kernel_variants()`](https://huggingface.co/docs/kernels/main/en/api/kernels#kernels.get_kernel_variants):

```py 
 from kernels import get_kernel_variants, VariantAccepted

for decision in get_kernel_variants("kernels-community/activation", version=1):  
    name = decision.variant.variant_str  
    if isinstance(decision, VariantAccepted):  
        print(f"{name}: compatible")  
    else:  
        print(f"{name}: rejected ({decision.reason})")  
```

It should print (depends on the machine you’re on):

```bash  
torch212-cxx11-cu130-aarch64-linux: compatible  
torch210-cu128-x86_64-windows: rejected (CPU (x86_64) does not match system CPU (aarch64))  
torch211-cu128-x86_64-windows: rejected (CPU (x86_64) does not match system CPU (aarch64))  
torch212-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch211-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch210-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch29-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
…  
```

### Improved manylinux_2_28 support

Kernel-builder has targeted `manylinux_2_28` almost since the beginning. We used to target `manylinux` by using a modern gcc toolchain compiled with glibc 2.28. To avoid compatibility issues with older versions of `libstdc++`, we statically linked libstdc++.

However, this approach recently resulted in some issues. Some `libstdc++` functionality uses global initialization. This can lead to corrupted data when multiple `libstdc++` versions come into play, such as the `libstdc++` that is linked dynamically by PyTorch and the `libstdc++` that is linked statically by a kernel. Some recent kernels use functionality (e.g. C++ regexes) that trigger global initializations, leading to such corrupted data, causing segfaults and other issues.

To solve this issue, kernels now link `libstdc++` dynamically. To ensure compatibility with old `libstdc++` versions, we now compile kernels with the official `manylinux_2_28` toolchain.

## Conclusion

Our goal with the Kernels project is to serve both kernel developers and users of custom kernels. We’re always keen on receiving feedback from the community on how we can improve it. Don’t hesitate to contribute!

*Acknowledgements: Thanks to [Aritra](ariG23498) for reviewing the post.*