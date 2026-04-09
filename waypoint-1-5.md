---
title: "Waypoint-1.5: Higher-Fidelity Interactive Worlds for Everyday GPUs"
thumbnail: /blog/assets/overworld/waypoint-v1-5.png
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

# Waypoint-1.5: Higher-Fidelity Interactive Worlds for Everyday GPUs

**Waypoint-1.5 Weights on the Hub**
- [Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
- [Waypoint-1.5-1B-360P](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P)

**Try it**
- [https://overworld.stream](https://overworld.stream)
- [Biome desktop client](https://github.com/Overworldai/Biome/)
- [Hugging Face]()

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/waypoint-v1-5-grid.png" alt="waypoint 1.5" width=70%>
</p>

## What is Waypoint-1.5?

Waypoint-1.5 is Overworld’s next real-time video world model, built to bring interactive generative worlds to the hardware people actually own.

The first release of Waypoint showed that real-time generative worlds were possible. It proved that interactive world models could be more than passive video demos, and that locally runnable systems could begin to close the gap between generating a world and actually stepping into one.

Waypoint-1.5 builds directly on that foundation. This release improves visual fidelity, expands the range of hardware that can run the model locally, and takes another step toward interactive world simulation without datacenter-scale compute.

On desktop hardware including RTX 3090 through 5090, Waypoint-1.5 can generate real-time environments at up to 720p and 60 FPS. This release also introduces a 360p tier designed to run smoothly across a much broader range of consumer hardware, including gaming laptops, and (soon) Apple Silicon Macs.

## What’s new in Waypoint-1.5?

The biggest change in Waypoint-1.5 is accessibility.

With Waypoint-1, we proved the core experience. With Waypoint-1.5, we wanted to make that experience available on more machines without giving up real-time interactivity. That meant building two model tiers: a 720p model for higher-performance hardware, and a 360p model optimized for broader deployment.

We also scaled training dramatically. Waypoint-1.5 was trained on nearly 100x more data than Waypoint-1, which significantly improves the model’s ability to generate more coherent environments and more consistent motion over time.

Under the hood, Waypoint-1.5 also incorporates more efficient video modeling techniques to reduce redundant computation across frames. That matters because real-time world models are not judged only by how a single frame looks. They are judged by whether the world responds instantly, stays coherent as you move through it, and remains usable on local hardware.

## Why this matters for world models

A lot of recent progress in generative video and world models has focused on visual fidelity. Those results matter, but fidelity alone is not what makes an interactive world feel real.

What people remember is responsiveness. They remember whether the environment reacts to them, whether motion stays coherent, whether the world holds together as they explore it, and whether the whole experience feels immediate instead of delayed.

That is the gap we care about most: the difference between watching a generated scene and actually being inside one.

If world models only run on large GPU clusters, they remain impressive demos. If they run locally on consumer hardware, they become something much more useful: a foundation for interactive entertainment, creative tooling, simulation, and AI-native environments people can actually explore.

Waypoint-1.5 is designed around that idea: not just better videos, but more responsive and explorable worlds that remain accessible on consumer hardware.

## How to experience Waypoint-1.5

There are two ways to play Waypoint-1.5.

The first is local execution through [Overworld Biome](https://github.com/Overworldai/Biome/). This release is designed to run across a wide range of hardware configurations, and the updated Biome runtime makes local setup much simpler. With the new installer flow, users can go from download to running the model locally in minutes.

The second is [Overworld Stream](https://www.overworld.stream/), which lets you try Waypoint-1.5 instantly in the browser with no local setup required.

Whether you want immediate access or full local control, Waypoint-1.5 is built to support both.

Additionally, we provide [World Engine](https://github.com/Wayfarer-Labs/world_engine), our flexible, easy to use, core inference library powering our official clients, along with nearly a dozen third party clients and libraries.

## The path forward

Waypoint started with a simple question: what would it take for generative worlds to become truly interactive?

Early generative systems showed that models could produce convincing images and videos. But building environments that people can explore, control, and interact with in real time is a different challenge entirely.

Waypoint-1.5 is another step in that direction, improving fidelity and expanding hardware accessibility while continuing to push real-time interactive generation onto local machines.

We think the future of world models will not be defined only by what they can render, but by whether people can actually inhabit and interact with them in real time.

Download Waypoint-1.5, run it locally with Biome, or jump in instantly on Overworld.stream.

And if you build something fun, strange, or unexpectedly immersive with it, we’d love to see it.

## Stay in touch
- [Overworld website](https://over.world/)
- [Discord](https://discord.gg/MEmQa7Wux4)
- [X / Twitter](https://x.com/overworld_ai)
