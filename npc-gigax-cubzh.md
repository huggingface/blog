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

Today, we are excited to **introduce a groundbreaking demo**: *NPC-Playground* created through the collaboration of the [Cubzh](https://github.com/cubzh/cubzh) and [Gigax](https://github.com/GigaxGames/gigax) teams, showcasing the potential of smart LLM-powered NPCs.

VIDEO OF THE DEMO

You can play with the demo directly on your browser 👉 [here](https://huggingface.co/spaces/cubzh/ai-npcs) 

In this 3D demo, you can **interact with the NPCs and teach them new skills with just a few lines of Lua scripting!**

## The Tech Stack

To create this, the teams used three main tools:

- [Cubzh](https://github.com/cubzh/cubzh): the cross-platform UGC (User Generated Content) game engine.

- [Gigax](https://github.com/GigaxGames/gigax): the engine for smart NPCs.

- [Hugging Face Spaces](https://huggingface.co/spaces): the most convenient online environment to host and iterate on game concepts in an open-source fashion.


## What is Cubzh?

[Cubzh](https://github.com/cubzh/cubzh) s a cross-platform UGC game engine, that aims to provide an open-source alternative to Roblox.

It offers a **rich gaming environment where users can create their own game experiences and play with friends**.
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/181_npc-gigax-cubzh/gigax.gif" alt="Cubzh"/>

In Cubzh, you can:

- **Create your own worlds items and avatars**.

- Build fast, using community **made voxel items** (+25K so far in the library) and **open-source Lua modules**.

- **Code games using a simple yet powerful Lua scripting API**.

Cubzh is in public Alpha. You can download and play Cubzh for free on Desktop via [Steam](https://store.steampowered.com/app/1386770/Cubzh_Open_Alpha/), [Epic Game Store](https://store.epicgames.com/en-US/p/cubzh-3cc767), or on Mobile via [Apple's App Store](https://apps.apple.com/th/app/cubzh/id1478257849), [Google Play Store](https://play.google.com/store/apps/details?id=com.voxowl.pcubes.android&hl=en&gl=US&pli=1) or even play directly from your [browser](https://app.cu.bzh/).

In this demo, Cubzh serves as the **game engine** running directly within a Hugging Face Space, users can easily clone it to experiment with custom scripts and NPC personas.


## What is Gigax?

[Gigax](https://github.com/GigaxGames/gigax) is the platform game developers use to run **LLM-powered NPCs at scale**.

Gigax has fine-tuned large language models for NPC interactions, **using the "function calling" principle.**

It's easier to think about this in terms of input/output flow:

- In **input**, the model reads [a text description](https://github.com/GigaxGames/gigax/blob/main/gigax/prompt.py) of a 3D scene, alongside a description of the recent events and a list of the NPC's available actions (e.g., `<say>`, `<jump>`, `<attack>`, etc.).

- The model then **outputs** one of these actions using parameters that refer to 3D entities that exist in the scene, e.g. `say NPC1 "Hello, Captain!"`.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/181_npc-gigax-cubzh/gigax.png" alt="gigax" />

Gigax has **open-sourced their stack!** You can download their [fine-tuned models on the 🤗 Hub](https://huggingface.co/Gigax), and clone their [inference stack on Github](https://github.com/GigaxGames/gigax).


## The NPC-Playground Demo

Interact with LLM-powered NPCs in our 3D Playground, in your browser: [huggingface.co/spaces/cubzh/ai-npcs](https://huggingface.co/spaces/cubzh/ai-npcs).

Just clone the repository and modify `cubzh.lua` to teach NPCs new skills with a few lines of Lua scripting!


## Make your own demo 🔥

Playing with the demo is just the first step! If you're **interested in customizing it**, [check out our comprehensive ML for Games Course tutorial for step-by-step instructions and resources](https://huggingface.co/learn/ml-games-course/unit3/introduction).


In addition, [you can check the documentation to learn more](https://huggingface.co/spaces/cubzh/ai-npcs/blob/main/README.md) on how to tweak NPC behavior and teach NPCs new skills.

We **can't wait to see the amazing demos you're going to make 🔥**. Share your demo on LinkedIn and X, and tag us  @cubzh_ @gigax @huggingface **we'll repost it** 🤗. 


-- 
The collaboration between Cubzh and Gigax has demonstrated **how advanced AI can transform NPC interactions, making them more engaging and lifelike.**

If you want to dive more into Cubzh and Gigax don’t hesitate to join their communities:

- [Cubzh Discord Server](https://discord.com/invite/cubzh) 
- [Gigax Discord Server](https://discord.gg/rRBSueTKXg)

And to stay updated on the latest updates on Machine Learning for Games, don't forget to [join the 🤗 Discord](https://discord.com/invite/JfAtkvEtRb)
