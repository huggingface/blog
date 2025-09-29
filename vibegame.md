---
title: "VibeGame: Exploring Vibe Coding Games"
thumbnail: /blog/assets/vibegame/thumbnail.png
authors:
- user: dylanebert
---

# VibeGame: Exploring Vibe Coding Games

## The Problem

People are trying to vibe code games. And it kind of works, at first. However, as the project grows, things begin to fall apart. Why? And what can we do about it?

I'll talk about the problem, how I fixed it, and where to go from here.

## What Is "Vibe Coding"?

First, what is vibe coding? It's originally coined by Andrej Karpathy in a [viral tweet](https://x.com/karpathy/status/1886192184808149383) where it's defined as where you "fully give in to the vibes, embrace exponentials and forget the code even exists".

However, since then, it's used descriptively to mean a lot of different things, anywhere from just "using AI when coding" to "not thinking about the code at all". In this blog post, I'll define it as: using AI as a high-level programming language to build something. Like other programming languages, this benefits from understanding what's going on under the hood, but doesn't necessarily require it.

With this interpretation, you could make a game without understanding code, though knowing the fundamentals still helps.

## Context Management

Earlier I mentioned that "as the project grows, things begin to fall apart". This is because there is [evidence](https://huggingface.co/blog/helmet) that as context window fills up, model performance begins to degrade. This is especially true for game development, where the context can grow very large, very quickly.

To address this issue, there are many personal ad-hoc solutions, such as writing LLM-specific context directly in the project files, or more comprehensive solutions like [Claude Code Development Kit](https://github.com/peterkrueck/Claude-Code-Development-Kit) for large-scale context management.

I couldn't find a lightweight, accessible solution, which doesn't rely on significant domain knowledge. So I made one: 🧅 [Shallot](https://github.com/dylanebert/shallot), a simple, lightweight, unopinionated context management system for Claude Code. It relies on two basic commands:

1. `/peel [prompt]` to load context at the beginning of a conversation
2. `/nourish` to update context at the end of a conversation

Anecdotally, this works well. However, it works best when the project stays lean and well-organized, so all relevant context can easily fit in the model's context window. While Claude Code is used here, the same principles generalize to other models.

Beyond context management tools, platform choice is critical. The platform should ideally naturally keep projects lean through high-level abstractions, while also being something AI models understand well. So, what existing platforms are best suited for vibe coding?

## Initial Exploration

I initially tried 3 different approaches to vibe coding games: Roblox MCP, Unity MCP, and web. For each, I tried to build a simple incremental game inspired by [Grass Cutting Incremental](https://roblox-grass-cutting-incremental.fandom.com/wiki/Roblox_Grass_Cutting_Incremental_Wiki), using Claude Code for each.

Here's how it went:

### Attempt 1: Roblox MCP

The [official MCP server](https://github.com/Roblox/studio-rust-mcp-server) from Roblox. This allows AI to interact with Roblox Studio by sending commands to run code.

Pros:
- Excellent level of abstraction with built-in game mechanics
- AI could very easily understand the syntax and convert instructions to code

Cons:
- No files, only using code to read data, which severely limits context management
- Very limited runtime information for AI to work with
- Proprietary walled garden

Roblox provides an excellent layer of abstraction for keeping the codebase lean and manageable, which is perfect for vibe coding. However, the walled garden and lack of context makes it infeasible for vibe coding, unless it's in-house at Roblox.

### Attempt 2: Unity MCP

The [unofficial MCP server](https://github.com/CoplayDev/unity-mcp) for Unity. This allows AI to interact with the Unity Editor: reading the console, managing assets, and validating scripts.

Pros:
- Full file system access

Cons:
- There are many ways to do everything in Unity, changing frequently across versions, causing AI to get confused
- Requires significant domain knowledge to tell the AI *how* to do things, rather than *what* to do
- AI performance was inconsistent and unreliable
- Proprietary engine (though much more transparent than Roblox)

Unity is a powerful engine with a lot of capabilities. However, the complexity and variability of the engine makes it difficult for AI to consistently produce good results without significant user domain knowledge.

### Attempt 3: Web Stack

The open web platform, using [three.js](https://threejs.org/) for 3D rendering, [rapier](https://rapier.rs/) for physics, and [bitecs](https://github.com/NateTheGreatt/bitECS) for game logic.

Pros:
- Far superior AI proficiency compared to game engines, likely due to massive training data
- Full file system access
- Fully open source stack with complete control/transparency

Cons:
- Relatively low level libraries, requiring essentially building the engine before building the game
- Lack of ecosystem for high-quality 3D games; web tends toward 2D games and simple 3D experiences

This approach had the best AI performance by far, likely due to the vast amount of web development data available during training. However, the low-level nature of the libraries meant that I had to essentially build a game engine before I could build the game itself. This allows us to work at a much higher level of abstraction, like we did with Roblox.

Despite requiring essentially building an engine first, this was the only approach that produced something *actually kind of fun* without requiring significant user domain knowledge. So I decided to explore this direction further.

### Comparison Summary

| Platform | AI Performance | Abstraction Level | Context Management | Open Source |
|----------|---------------|-------------------|-------------------|-------------|
| Roblox   | ⭐⭐⭐        | ⭐⭐⭐⭐⭐          | ⭐                | ❌          |
| Unity    | ⭐           | ⭐⭐              | ⭐⭐⭐            | ❌          |
| Web      | ⭐⭐⭐⭐⭐     | ⭐                | ⭐⭐⭐⭐⭐        | ✅          |

## The Solution: VibeGame

After these experiments, I had a clear picture: the web stack had excellent AI performance but was too low-level, while Roblox had perfect abstraction but lacked openness and context management.

So, what about combining the best of both?

Introducing [VibeGame](https://github.com/dylanebert/vibegame), a high-level declarative game engine built on top of three.js, rapier, and bitecs, designed specifically for AI-assisted game development.

### Design Philosophy

There were three key decisions that went into the design of VibeGame:

1. **Abstraction:** A high-level abstraction with built-in features like physics, rendering, and common game mechanics, keeping the codebase lean and manageable. This takes inspiration from popular high-level sandbox games/game "engines" like Roblox, Fortnite UEFN, and Minecraft.
2. **Syntax:** A declarative XML-like syntax for defining game objects and their properties, making it easy for AI to understand and generate code. This is similar to HTML/CSS, which AI models are already proficient in.
3. **Architecture:** An Entity-Component-System (ECS) architecture for scalability and flexibility. ECS separates data (components) from behavior (systems), encouraging the project to stay modular and organized as it grows, conducive to vibe coding and context management.

A basic game looks like this:

```html
<world canvas="#game-canvas" sky="#87ceeb">
  <!-- Ground -->
  <static-part pos="0 -0.5 0" shape="box" size="20 1 20" color="#90ee90"></static-part>

  <!-- Ball -->
  <dynamic-part pos="-2 4 -3" shape="sphere" size="1" color="#ff4500"></dynamic-part>
</world>

<canvas id="game-canvas"></canvas>

<script type="module">
  import * as GAME from 'vibegame';
  GAME.run();
</script>
```

See it in action in this [JSFiddle](https://jsfiddle.net/keLsxh5t/) or the [Live Demo](https://huggingface.co/spaces/dylanebert/VibeGame).

This will create a simple scene with a ground plane and a falling ball. The player, camera, and lighting are created automatically. All of this is modular and can be replaced. Arbitrary custom components and systems can be added as needed.

This comes bundled with an [llms.txt](https://raw.githubusercontent.com/dylanebert/VibeGame/refs/heads/main/llms.txt) file containing documentation about the engine, designed specifically for AI, to be included in its system prompt or initial context.

## So Does It Actually Work?

Yes.

Well, kind of.

[Here's the game](https://huggingface.co/spaces/dylanebert/grass_cutting_game) I built to test building a simple incremental grass collection game using VibeGame and Claude Code. It worked very well, requiring minimal domain knowledge for implementing the core game mechanics.

![Grass Cutting Game](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vibegame/grass_cutting_game.gif)

However, there are still some major caveats:
1. It works well for building what the game engine supports, i.e. a simple platformer or game that only relies on basic physics and rendering.
2. However, it struggles with anything more complex that isn't yet implemented in the engine, like interaction, inventory, multiplayer, combat, etc.

So, with a definition of vibe coding that is the one-shot "make me a game" approach, it doesn't work. However, with the definition of treating vibe coding like a high-level programming language, it works very well, but requires users to understand the engine's capabilities and limitations.

## Try It Yourself

To try it immediately, I built a demo where you can develop a game directly in the browser using VibeGame with [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct): [Live Demo on Hugging Face](https://huggingface.co/spaces/dylanebert/VibeGame).

You can also test it locally with a frontier model like Claude Code:

```bash
npm create vibegame@latest my-game
cd my-game
npm run dev  # or bun dev
```

Then, paste all the contents of the included `llms.txt` to `CLAUDE.md`, providing full documentation about the engine for the AI to reference (or point your own context management system to it). This works with other models as well.

## What's Next?

The engine is currently very barebones and only supports very basic mechanics (unless writing it from scratch). However, initial results are promising.

Next steps would be:

1. Flesh out the engine with more built-in mechanics, getting closer to par with early versions of Roblox or UEFN. This includes:
  - Interaction
  - Inventory/items
  - Multiplayer
  - Skinned meshes/animations with curated database
  - Audio with curated database
2. Improve the AI guidance systems, providing beginners with a better experience. This includes:
  - Clear messaging about engine capabilities/limitations
  - Guided prompts for common tasks
  - Many more examples and templates
  - Educational resources

It's also worth exploring how vibe coding games could harness more proven engines. For example, building a high-level sandbox game editor on top of Unity or Unreal Engine (similar to how Unreal Editor for Fortnite is built on Unreal Engine) could provide a more controlled environment for AI to work with, while leveraging the power of established engines.

We're also likely to see more in-house solutions from major players. 

[Follow me](https://x.com/dylan_ebert_) to keep up with what's going on in the space!

**Links:**
- [VibeGame](https://github.com/dylanebert/vibegame)
- [Hugging Face Demo](https://huggingface.co/spaces/dylanebert/VibeGame)
- [Example Game (made with VibeGame)](https://huggingface.co/spaces/dylanebert/grass_cutting_game)
- [Shallot Context Manager](https://github.com/dylanebert/shallot)
