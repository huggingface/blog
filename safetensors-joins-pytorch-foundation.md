---
title: "Safetensors is Joining the PyTorch Foundation"
thumbnail: /blog/assets/safetensors-joins-pytorch-foundation/thumbnail.png
authors:
- user: mcpotato
- user: lysandre
---

# Safetensors is Joining the PyTorch Foundation

Today, we're announcing that Safetensors has joined the PyTorch Foundation as a foundation-hosted project under the Linux Foundation, alongside DeepSpeed, Helion, Ray, vLLM, and PyTorch itself.

## How we got here

Safetensors started as a Hugging Face project born out of a concrete need: a way to store and share model weights that couldn't execute arbitrary code. The pickle-based formats that dominated the ecosystem at the time meant that there was a very real risk you’d be running malicious code. While this was an acceptable risk when ML was still budding, it would become unacceptable as open model sharing became central to how the ML community works.

The format we built is intentionally simple: a JSON header with a hard limit of 100MB, describing tensor metadata, followed by raw tensor data. Zero-copy loading that maps tensors directly from disk. Lazy loading so you can read individual weights without deserializing an entire checkpoint.

What we didn't fully anticipate was how broadly it would be adopted. Today, Safetensors is the default format for model distribution across the Hugging Face Hub and others, used by tens of thousands of models across all modalities in ML. It has become the preferred way for the open source ML community to share models.

## Why the PyTorch Foundation

We want Safetensors to truly belong to the community. The project has always been open source, but code contributions are just one part of its evolution. By bringing more companies and contributors into the governance of the project, we make sure that progress reflects the breadth of the community building on top of it. Joining the PyTorch Foundation means Safetensors now has a vendor-neutral home. The trademark, the repository, and the governance of the project sit with the Linux Foundation rather than any single company. Hugging Face's two core maintainers, Luc and Daniel, remain on the Technical Steering Committee and continue to lead the project day-to-day, but Safetensors now formally belongs to the community that depends on it.

We believe safety is best guaranteed when every contributor can build on what already exists; a principle now embedded in the project's governance itself.

## What this means for users and contributors

For the vast majority of users, nothing changes. The format is the same, the APIs are the same, the Hub integration is the same: no breaking changes. Models stored in Safetensors format today will continue to work exactly as they do now.

For contributors, the path to becoming a maintainer is now formally documented and open to anyone in the community. The project's governance lives in GOVERNANCE.md and MAINTAINERS.md in the repository. For organizations building on top of Safetensors, neutral governance under the Linux Foundation provides a stable, long-term foundation, entirely community-driven.

## What comes next

Safetensors is a well-established project, adopted by the ecosystem at large, but we're still convinced we're at the very beginning of the project. 

**We're working with the PyTorch team so that Safetensors may be used within PyTorch core as a serialization system for torch models.**

The coming months will see significant growth, and we couldn't think of a better home for that next chapter than the PyTorch Foundation. The roadmap ahead includes device-aware loading and saving, so tensors can load directly onto CUDA, ROCm, and other accelerators without unnecessary CPU staging. 

We're also building first-class APIs for Tensor Parallel and Pipeline Parallel loading, so each rank or pipeline stage loads only the weights it needs. And as the ecosystem's quantization landscape continues to evolve, we'll be formalizing support for FP8, block-quantized formats like GPTQ and AWQ, and sub-byte integer types. 

These are problems the whole ecosystem has a stake in solving, and being inside the PyTorch Foundation means we can work on them in collaboration with the other hosted projects rather than in parallel.

## Get involved

Safetensors is open source and contributions are welcome at every level, from bug reports and documentation to new features and participation in governance.
- **GitHub:** [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors)
- **Documentation:** [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)
- **PyTorch Foundation:** [pytorch.org/foundation](https://pytorch.org/foundation)

If you're a developer, researcher, or organization that builds on Safetensors and want to be more involved in shaping its direction, open an issue, start a discussion, or reach out to the maintainers directly. The project has always belonged to the community that uses it. The governance now reflects that too.

