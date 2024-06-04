---
title: "Introducing NPC-Playground, a 3D playground to interact with LLM-powered NPCs" 
thumbnail: /blog/assets/181_npc-gigax-cubzh/thumbnail.png
authors:
- user: Trist4x
  guest: true
  org: Gigax
- user: SolitonMa
  guest: true
  org: cubzh
- user: aduermael
  guest: true
  org: cubzh
- user: gdevillele
  guest: true
  org: cubzh
- user: caillef
  guest: true
  org: cubzh
- user: ThomasSimonini
---

# Introducing NPC-Playground, a 3D playground to interact with LLM-powered NPCs

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/181_npc-gigax-cubzh/thumbnail.png" alt="Thumbnail"/>

*AI-powered NPCs* (Non-Playable Characters) are **one of the most important breakthroughs brought** about by the use of LLMs in games.

LLMs, or Large Language Models, make it possible to design _"intelligent"_ in-game characters that **can engage in realistic conversations with the player, perform complex actions and follow instructions, dramatically enhancing the player's experience**. AI-powered NPCs represent a huge advancement vs rule-based and heuristics systems.

Today, we are excited to **introduce a groundbreaking demo**: *NPC-Playground* created through the collaboration of the [Cubzh](https://cu.bzh/) and [Gigax](https://github.com/GigaxGames/gigax) teams, showcasing the potential of smart LLM-powered NPCs.

VIDEO OF THE DEMO

You can play with the demo 👉 [here](https://huggingface.co/spaces/cubzh/ai-npcs) 

In this 3D demo, you can **interact with the NPCs and teach them new skills with just a few lines of Lua scripting!**

## The Tech Stack

To create this, the teams used three main tools:

- [Cubzh](https://cu.bzh/): the cross-platform UGC (user content generated) Game Engine.

- [Gigax](https://github.com/GigaxGames/gigax): the engine for smart NPCs.

- [Hugging Face Spaces](https://huggingface.co/spaces): the easiest way to host your game online.


## What is Cubzh?

[Cubzh](https://cu.bzh/) is the new UGC gaming platform to create cross-platform games with Lua and Cubes.

It offers a **rich gaming environment for players to create their own game experiences to share with friends**.
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/181_npc-gigax-cubzh/gigax.gif" alt="Cubzh"/>

With Cubzh, you can:

- **Create your own worlds** and craft new game items and avatars.

- Use a **library of +25k free assets created by Cubzh's community**.

- **Code games using a simple but powerful Lua scripting API**.

You can download and play Cubzh for free on Desktop via [Steam](https://store.steampowered.com/app/1386770/Cubzh_Open_Alpha/), [Epic Game Store](https://store.epicgames.com/en-US/p/cubzh-3cc767), or on Mobile via [Apple's App Store](https://apps.apple.com/th/app/cubzh/id1478257849), [Google Play Store](https://play.google.com/store/apps/details?id=com.voxowl.pcubes.android&hl=en&gl=US&pli=1) or even play directly from your [browser](https://app.cu.bzh/).

In this demo, Cubzh serves as the **game engine, providing an immersive experience**.


## What is Gigax?

[Gigax](https://github.com/GigaxGames/gigax) is the platform game developers use to run **LLM-powered NPCs at scale**.

Gigax has fine-tuned large language models for NPC interactions, **using the "function calling" principle. **

It's easier to think about this in terms of input/output flow:

- In **input**, the model reads [a text description](https://github.com/GigaxGames/gigax/blob/main/gigax/prompt.py) of a 3D scene, alongside a description of the recent events and a list of the NPC's available actions (e.g., `<say>`, `<jump>`, `<attack>`, etc.).

- The model then **outputs** one of these actions using parameters that refer to 3D entities that exist in the scene, e.g. `say NPC1 "Hello, Captain!"`.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/181_npc-gigax-cubzh/gigax.png" alt="gigax" />

Gigax has **open-sourced their stack!** You can download their [fine-tuned models on the 🤗 Hub](https://huggingface.co/Gigax), and clone their [inference stack on Github](https://github.com/GigaxGames/gigax).


## The NPC-Playground Demo

This new demo **leverages the strengths of Cubzh and Gigax, along with Hugging Face Spaces for online hosting**.

In this demo, you can **interact, have conversations with the NPCs and teach them new skills with just a few lines of Lua scripting!**

You can play with the demo 👉 [here](https://huggingface.co/spaces/cubzh/ai-npcs) 


## Make your own demo 🔥

Playing with the demo is just the first step! If you're **interested in customizing it**, [check out our comprehensive ML for Games Course tutorial for step-by-step instructions and resources](https://huggingface.co/learn/ml-games-course/unit3/introduction).

We **can't wait to see the amazing demos you're going to make 🔥**.

In addition, [you can check the documentation to learn more](https://huggingface.co/spaces/cubzh/ai-npcs/blob/main/README.md) on how to tweak NPC behavior and teach NPCs new skills.

-- 
The collaboration between Cubzh and Gigax has demonstrated **how advanced AI can transform NPC interactions, making them more engaging and lifelike.**

If you want to dive more into Cubzh and Gigax don’t hesitate to join their communities:

- [Cubzh Discord Server](https://discord.com/invite/cubzh) 
- [Gigax Discord Server](https://discord.gg/rRBSueTKXg)

And to stay updated on the latest updates on Machine Learning for Games, don't forget to [join the 🤗 Discord](https://discord.com/invite/JfAtkvEtRb)
