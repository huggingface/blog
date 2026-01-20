---
title: "Introducing Waypoint-1: Real-time interactive video diffusion from Overworld"
thumbnail: /blog/assets/overworld/overworld_image.png
authors:
- user: lapp0
  guest: true
  org: overworld
- user: LouisCastricato
  guest: true
  org: overworld
- user: ScottieFox
  guest: true
  org: overworld
- user: shahbuland
  guest: true
  org: overworld
- user: xAesthetics
  guest: true
  org: overworld
---

# Waypoint-1: Real-time interactive video diffusion from [Overworld](http://over.world)

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/overworld-waypoint.gif" alt="waypoint launch grid" width=70%>
</p>

## Waypoint-1 **Weights** on the Hub
 - [Waypoint-1-Small](https://huggingface.co/overworld/Waypoint-1-Small)
 - [Waypoint-1-Medium](https://huggingface.co/overworld/Waypoint-1-Medium)

## Try Out The Model
**Overworld Stream:** https://overworld.stream

## What is Waypoint-1?

Waypoint-1 is Overworld’s real-time-interactive video diffusion model, controllable and prompted via text, mouse, and keyboard. You can give the model some frames, run the model, and have it create a world you can step into and interact with.

The backbone of the model is a frame-causal rectified flow transformer trained on 10,000 hours of diverse video game footage paired with control inputs and text captions. Waypoint-1 is a latent model, meaning that it is trained on compressed frames.

The standard among existing world models has become taking pre-trained video models and fine-tuning them with brief and simplified control inputs. In contrast, Waypoint-1 is trained from the get-go with a focus on interactive experiences. With other models, controls are simple: you can move and rotate the camera once every few frames, with severe latency issues. With Waypoint-1 you are not limited at all as far as controls are concerned. You can move the camera freely with the mouse, and input any key on the keyboard, and all this with zero latency. Each frame is generated with your controls as context. Additionally, the model runs fast enough to provide a seamless experience even on consumer hardware.

## How was it trained?

Waypoint-1 was pre-trained via diffusion forcing, a technique with which the model learns to denoise future frames given past frames. A causal attention mask is applied such that a token in any given frame can only attend to tokens in its own frame, or past frames, but not future frames. Each frame is noised randomly, and as such the model learns to denoise each frame separately. During inference, you can then denoise new frames one at a time, allowing you to generate a procedural stream of new frames.

While diffusion forcing presents a strong baseline, randomly noising all frames is misaligned with a frame-by-frame autoregressive rollout. This inference mismatch results in error accumulation, and noisy long rollouts. To address this problem we post-train with self forcing, a technique that trains the model to produce realistic outputs under a regime which matches inference behavior. Self-forcing via DMD has the added benefit of one-pass CFG, and few-step denoising.


## The Inference Library: [WorldEngine](https://github.com/Wayfarer-Labs/world_engine)

[WorldEngine](https://github.com/Wayfarer-Labs/world_engine) is Overworld’s high‑performance inference library for interactive world model streaming. It provides the core tooling for building inference applications in pure Python, optimized for low latency, high throughput, extensibility, and developer simplicity. The runtime loop is designed for interactivity: it consumes context frame images, keyboard/mouse inputs, and text, and outputs image frames for real‑time streaming.

On Waypoint‑1‑Small (2.3B) running on a 5090, WorldEngine sustains ~30,000 token‑passes/sec (single denoising pass; 256 tokens per frame) and achieves 30 FPS at 4 steps or 60 FPS at 2 steps

Performance comes from four targeted optimizations:

- [AdaLN feature caching](https://arxiv.org/html/2412.18911v1): Avoids repeated AdaLN conditioning projections through caching and reusing so long as prompt conditioning and timesteps stay the same between fwd passes.
- [Static Rolling KV Cache + Flex Attention](https://arxiv.org/pdf/2412.05496)
- Matmul fusion: Standard inference optimization using fused QKV projections.
- [Torch Compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) using `torch.compile(fullgraph=True, mode="max-autotune", dynamic=False)`


```python
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("Overworld/Waypoint-1-Small", device="cuda")

# Specify a prompt
engine.set_prompt("A game where you herd goats in a beautiful valley")

# Optional: Force the next frame to be a specific image
img = pipeline.append_frame(uint8_img)  # (H, W, 3)

# Generate 3 video frames conditioned on controller inputs
for controller_input in [
        CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
        CtrlInput(mouse=[0.1, 0.2]),
        CtrlInput(button={95, 32, 105}),
]:
    img = engine.gen_frame(ctrl=controller_input)
```

## Build with World Engine

We’re running a `world_engine` hackathon on 1/20/2026 - You can RSVP [here](https://luma.com/klpa49os). Teams of 2-4 are welcome and the prize is a 5090 GPU on the spot. We’d love to see what you can come up with to extend the world_engine and it should be a great event to meet like-minded founders, engineers, hackers and investors. We hope you can join us at 10am PST on January 20th for 8 hours of friendly competition!

## Stay in Touch

- [Website](http://over.world)
- [Discord (Developers)](https://discord.gg/mc6t9jjrR8)
- [Discord (Models/Players)](https://discord.gg/MEmQa7Wux4)
- [X/Twitter](https://x.com/overworld_ai)
