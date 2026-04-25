---
title: "SkyRelic Multi-Agent Drone Delivery Environment"
thumbnail: /blog/assets/drone_env/thumbnail.png
authors:
- user: manikandan-n-07
- user: kaviyarasu2666
- user: 
---

<div align="center">

<img src="https://img.shields.io/badge/Multi--Agent-Supported-success?style=for-the-badge" />
<img src="https://img.shields.io/badge/OpenEnv-Compatible-blueviolet?style=for-the-badge&logo=huggingface" />
<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch" />
<img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi" />
<img src="https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

# 🚁 SkyRelic Multi-Agent Drone Env
### Autonomous Multi-Agent Neural Navigation Framework — Complete Technical Deep Dive

**A high-fidelity project for training and evaluating autonomous drone fleets. Featuring a modular architecture, real-time telemetry, and synchronized training logs across the entire ecosystem.**

[**🌐 Live Demo**](https://manikandan-n-07-drone-env.hf.space) · [**📖 API Docs**](http://localhost:8000/docs) · [**📦 PyPI**](https://pypi.org/project/drone-env) · [**🐛 Issues**](https://github.com/manikandan-n-07/drone-env/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [New in Multi-Agent Drone](#new-in-multi-agent-drone)
- [Fleet Mechanics (Multi-Agent)](#fleet-mechanics-multi-agent)
- [Command Quick Reference](#-command-quick-reference)
- [System Architecture](#system-architecture)
- [CORE — The Physics Engine (Matrix Engine)](#core--the-physics-engine-matrix-engine)
- [DATA — The Memory Warehouse](#data--the-memory-warehouse)
- [GRADERS — The Olympic Judge](#graders--the-olympic-judge)
- [RESULTS — The Visual EKG](#results--the-visual-ekg)
- [RL — The Neural Brain](#rl--the-neural-brain)
- [SERVER — The Global Control Tower](#server--the-global-control-tower)
- [SRC — The Design Gallery](#src--the-design-gallery)
- [TESTS — The Quality Control Lab](#tests--the-quality-control-lab)
- [TMP — The Scratchpad](#tmp--the-scratchpad)
- [ROOT FILES — The Cockpit](#root-files--the-cockpit)
- [Environment Mechanics](#environment-mechanics)
- [Neural Intelligence Layer](#neural-intelligence-layer)
- [API Reference](#api-reference)
- [Quickstart](#quickstart)
- [Training](#training)
- [LLM-Powered Inference](#llm-powered-inference)
- [Reward Engineering](#reward-engineering)
- [Grading & Evaluation](#grading--evaluation)
- [The Life of a Parcel (End-to-End Flow)](#the-life-of-a-parcel-end-to-end-flow)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Docker Deployment](#docker-deployment)
- [Hugging Face Submission](#hugging-face-submission)
- [Phase 2 Validation Updates](#phase-2-validation-updates)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

**SkyRelic Drone Env** is a production-grade, multi-agent reinforcement learning simulation framework. It provides a realistic urban delivery scenario where a fleet of drones must navigate procedurally generated grids, avoid obstacles, manage battery resources, and coordinate multi-parcel delivery missions.

The framework supports three operational modes:

| Mode | Description | Entry Point |
|------|-------------|-------------|
| **Deep RL Training** | Train a `PathQNet` DQN agent from scratch | `train.py` |
| **LLM-Guided Inference** | Drive the agent via any OpenAI-compatible LLM (e.g., Qwen, GPT-4) | `inference.py` |
| **Interactive Server** | REST API + browser-based dashboard | `server/app.py` |

---

## New in Multi-Agent Drone

- **Multi-Agent Capability**: Support for simultaneous drone operations with unified fleet state management.
- **Synchronized Telemetry**: All training episodes from `train.py` are now automatically recorded to `data/memory.json`.
- **Modular Architecture**: Complete refactoring into `core`, `rl`, `server`, and `graders` modules for industrial-grade maintainability.
- **Optimized Training Engine**: Added task-specific episode defaults and an automated "all-task" continuous training mode.
- **Enhanced Physics**: Improved collision detection and battery depletion logic for multi-drone scenarios.

---

## Fleet Mechanics (Multi-Agent)

Version 0.3.0 introduces high-fidelity fleet management. Instead of a single agent, the environment now handles multiple drones simultaneously:

- **Assignment Logic**: A nearest-neighbor heuristic assigns drones to pending packages dynamically.
- **Collision Avoidance**: Integrated physics checks ensure drones don't intercept each other on the same grid cell.
- **Unified Actions**: The `DroneAction` schema supports a mapped action dictionary `{drone_id: action}` for simultaneous control.

---

## 🚀 Command Quick Reference

| Action | Command |
| :--- | :--- |
| **Setup Project** | `uv sync` |
| **Dashboard Server** | `uv run python server/app.py` |
| **Train Full Fleet** | `uv run python train.py --task all` |
| **Train Easy (GPU)** | `uv run python train.py --task easy_delivery --episodes 100 --gpu` |
| **Train Medium (GPU)** | `uv run python train.py --task medium_delivery --episodes 100 --gpu` |
| **Train Hard (GPU)** | `uv run python train.py --task hard_delivery --episodes 100 --gpu` |
| **Run AI Inference** | `uv run python inference.py` |
| **Run AI Separately Inference** | `uv run python inference.py --task easy_delivery --steps 50` |
| **Local Validation** | `uv run openenv validate` |
| **Docker Build** | `docker build -t drone_env .` |
| **View Folders** | `dir data or dir results` |
| **Push to GitHub** | `git add . ; git commit -m "update" ; git push origin main` |
| **Deploy to HF** | `git push hf main` |

```
Targeted Testing: Use --task to choose easy_delivery, medium_delivery, or hard_delivery.
Step Limit: Use --steps 50 to force the simulation to end after 50 steps.
```

---

## System Architecture

The codebase follows a clean separation-of-concerns architecture across four distinct layers:

```
.
├── graders/                     # Unified Graders Package (Root)
│   ├── easy.py                  # Easy task scoring logic
│   ├── medium.py                # Medium task scoring logic
│   └── hard.py                  # Hard task scoring logic
├── core/                        # Simulation Logic Layer
│   ├── drone.py                 # Movement physics & battery drain
│   ├── grid_generator.py        # Map generation logic
│   ├── obstacles.py             # Collision & terrain detection
│   ├── state_manager.py         # Episodic state management
│   └── tasks.py                 # Mission difficulty configurations
├── rl/                          # Intelligence Layer
│   ├── model.py                 # Neural network architecture (DQN)
│   ├── policy.py                # Action selection policies
│   └── trainer.py               # Path analytics & learning engine
├── server/                      # Interface Layer
│   ├── app.py                   # FastAPI server & Grader discovery
│   ├── grid_world_environment.py # Main simulation environment
│   ├── map_generator.py         # Procedural map generation
│   └── static/                  # Dashboard Assets
├── data/                        # Persistence Layer
│   ├── memory.json              # Historical episode logs (JSON)
│   └── train.log                # Neural training logs
├── tests/                       # Validation Layer
│   ├── test_api.py              # Endpoint integration tests
│   └── test_env.py              # Physics & Grading unit tests
├── models.py                    # Unified Pydantic data models
├── client.py                    # CLI client for testing
├── __init__.py                  # Package marker (Root as drone_env)
├── train.py                     # Neural training entry point
├── inference.py                 # LLM-guided inference entry point
├── openenv.yaml                 # Mission Manifest (Tasks & Graders)
├── pyproject.toml               # Python project & dependency config
├── Dockerfile                   # Deployment container manifest
└── validate-submission.sh       # Submission validation script
```

### Component Interaction Flow

```
LLM / RL Agent
      │
      │  HTTP POST /step  {direction: "UP"}
      ▼
┌─────────────────────────────────┐
│   FastAPI Server  (app.py)      │
│   ┌──────────────────────────┐  │
│   │  DroneDeliveryEnvironment│  │
│   │  ┌────────┐ ┌──────────┐ │  │
│   │  │  grid_ │ │ core/*   │ │  │
│   │  │ world  │ │ physics  │ │  │
│   │  └────────┘ └──────────┘ │  │
│   └──────────────────────────┘  │
└─────────────────────────────────┘
      │
      │  DroneObservation (JSON)
      ▼
   Agent processes next step
```

---

## CORE/ — The Physics Engine (Matrix Engine)

> **Think of `core/` as the "Matrix Engine" of your drone world.** It is the invisible layer that translates raw physical reality into data that an AI can understand. Everything else in the project — the RL brain, the server, the graders — depends entirely on `core/` being correct.

### 1. Data Distillation — The Observation Pipeline

The AI doesn't see a "city"; it sees a collection of numbers. The `core/` folder is responsible for this translation:

- **The Grid Mapper**: Takes high-level objects (Buildings, Drones, Packages) and flattens them into a `DroneObservation` object — a clean numerical vector the AI can process.
- **Distance Vectors**: Calculates the "Manhattan Distance" between the drone and its delivery target, giving the AI a precise spatial sense.
- **Mathematical Precision**: If this translation is even slightly messy — e.g., if the drone thinks a building is at `(5,5)` but the world thinks it's at `(5.1, 5.1)` — the AI's bridge to reality breaks. `core/` ensures this bridge is **Mathematically Perfect**.

### 2. Reward Shaping — The Motivation Engine

In Reinforcement Learning, the drone has no ego — it only wants to "maximize the score." The `core/` folder defines what "Good" feels like:

- **Positive Sparse Rewards**: For the Meta Hackathon, SkyRelic strictly uses **Positive Sparse Rewards** instead of negative penalties for crashes. This creates a healthier learning signal.
- **Potential-Based Shaping**: The `core/` engine gives a small "pat on the back" reward when the drone gets closer to the target, guiding it through the "darkness" of a large 18×18 grid.
- **Human Values Encoded**: By adjusting the battery penalty vs. the delivery bonus in `core/tasks.py`, you tell the AI whether it should be "Careful" or "Fast." This is where human judgment enters the machine.

### 3. State Management — The Multi-Agent Orchestrator

Managing one drone is easy. Managing a fleet is a complex math problem:

- **Collision Avoidance**: If two drones try to land on the same grid cell, the `core/state_manager` acts as the **Air Traffic Controller** — rejecting the second drone's movement to maintain world consistency.
- **Resource Management**: Tracks global resources like `Battery` and `Package Availability`. Ensures that two drones aren't chasing the same package at the same time.
- **Stability Provider**: Without a central `core` manager, the simulation becomes "Non-Deterministic" (chaotic), and the AI would never learn a reliable strategy.

### Core Module Files

| File | Responsibility |
|------|---------------|
| `core/drone.py` | Movement physics & battery drain simulation |
| `core/grid_generator.py` | Procedural map/city generation logic |
| `core/obstacles.py` | Collision & terrain detection |
| `core/state_manager.py` | Episodic multi-agent state management |
| `core/tasks.py` | Mission difficulty configurations & reward constants |

> 🏆 **THE GOLDEN RULE of `core/`:** Total Independence. The `core/` folder must be able to run even if you delete the Neural Network (`rl/`) and the Website (`server/`). It is a self-contained universe. Training uses it to learn. FastAPI uses it to serve. Grader uses it to score. **If the foundation is solid, the AI on top will be unstoppable! 🚁🚀**

> 💡 **Currently**: Your `hard_delivery` mission is actively being simulated by this core engine. It's doing millions of calculations to ensure every battery drop and movement is fair!

---

## DATA/ — The Memory Warehouse

> **If `core/` is the Physical Universe, then `data/` is the Memory Warehouse of your system.** Think of it as two types of intelligence: **Muscle Memory** (the Model) and **The Journal** (the Logs).

### 1. Muscle Memory — `model.pth`

This is the result of thousands of hours of virtual flight.

- **How it works**: Every time the AI makes a good move, those neural patterns are saved as weights. The `model.pth` file is a snapshot of the AI's intuition at a specific point in time.
- **The Deep Need**: Without this file, every time you stop the computer, the AI would "forget" everything and have to start learning from scratch — like a newborn baby.
- **Why we ignore it in Git**: These files are heavy and change with every training run. In professional AI development, we keep the **Recipe** (code) in Git, but the **Result** (weights) in `data/` locally.

### 2. The Journal — `memory.json`

This is the "Black Box" flight recorder for every single mission.

- **How it works**: Every time a drone moves, the `rl/trainer.py` writes an entry: *"Step 5: Drone was at (2,3), it moved UP, it got +0.01 reward."*
- **The Deep Need**: This allows for **Analysis**. You can't see "Intelligence" by looking at code; you see it by looking at **Trends**. This file drives your Dashboard's "Fleet Performance" charts.
- **Why we track it in Git**: These are small text files. They provide "Proof" to your collaborators (or the Meta hackathon judges) that your AI actually learned and wasn't just guessing.

### 3. Task-Specific Isolation — The Hierarchy

We organize data into `/easy`, `/medium`, and `/hard`. This is critical:

- **The Problem of Overwriting**: If we had only one `memory.json`, the data from a simple 10×10 grid would get mixed with a complex 18×18 grid — poisoning both models.
- **The Deep Strategy**: By isolating them, we create **Specialized Experts**:
  - `data/hard/model.pth` becomes an expert at long-distance battery conservation.
  - `data/easy/model.pth` becomes an expert at high-speed precision delivery.
- **Validation Discovery**: When `server/app.py` runs, it scans all these folders, allowing the dashboard to show a "Cross-Difficulty Report" instantly.

### 4. Brain Drain Prevention — `train.log`

- **How it works**: Records the training process itself — Loss values, Epsilon decay, Training time.
- **The Deep Need**: If the AI stops learning, you look at `train.log` to see if the "Brain was losing focus" (High Loss) or if it "Explored too much" (High Epsilon).

### Data Folder Layout

```
data/
├── easy/
│   ├── model.pth          # Trained weights for Easy missions
│   └── memory.json        # Episode logs for Easy missions
├── medium/
│   ├── model.pth          # Trained weights for Medium missions
│   └── memory.json        # Episode logs for Medium missions
├── hard/
│   ├── model.pth          # Trained weights for Hard missions
│   └── memory.json        # Episode logs for Hard missions
└── train.log              # Unified training engine logs
```

> 🧠 **DEEP SUMMARY**: In modern AI, **Data is the actual Value**. The code for a DQN is common, but the `data/` folder contains your **Unique Experience** — evidence of all the thousands of "Virtual Hours" your drones have spent in the air. It is the bridge between a "Simulation" and an "Agent with Intelligence." 🚁💾🧠

> 💡 **Currently Writing**: Your `hard_delivery` training is pumping data into `data/hard/memory.json` and `data/train.log` as we speak!

---

## GRADERS/ — The Olympic Judge

> **Think of `graders/` as the "Olympic Judge" of your project.** While `core/` controls the physics and `rl/` provides the movement, `graders/` is the final authority that decides your rank on the leaderboard.

### 1. Training Reward vs. Evaluation Score — The Big Difference

This is the most important "deep" concept in RL:

| Concept | Rewards (The Coach) | Graders (The Exam) |
|---------|--------------------|--------------------|
| **When** | During flight — every step | After mission is complete |
| **Signal** | Small `+0.01` or `+0.05` signals | Final efficiency & success score |
| **Purpose** | Guide the drone through learning | Pure, unbiased performance measurement |
| **Location** | `core/tasks.py` | `graders/easy.py`, `medium.py`, `hard.py` |

> The Deep Need: We separate them so the AI can't "game the system." The Grader is a pure, unbiased measurement of performance.

### 2. Multi-Tier Logic — `easy.py`, `medium.py`, `hard.py`

Each difficulty level has a different "Expectation":

| Difficulty | Grid Size | Deliveries Required | Battery Penalty | Score Threshold |
|-----------|-----------|--------------------|----|---|
| **Easy** | 10×10 | 1 package | 0.10 (10%) | High (forgiving) |
| **Medium** | 14×14 | 3 packages | 0.15 (15%) | Medium |
| **Hard** | 18×18 | 5 packages | 0.25 (25%) | Strict |

The `graders/` package uses a unified formula but with different "Weights" for each task — allowing us to compare "Apple to Apples" when looking at scores across different map sizes.

### 3. The Meta Benchmark Interface — `openenv.yaml`

Hugging Face and Meta's automated validation tools (OpenEnv) don't know anything about your code. They only know what's in your `openenv.yaml`:

- **Discovery**: The `openenv.yaml` points directly to `drone_env.graders:grade_easy`.
- **The Deep Need**: By putting scoring logic in a separate `graders/` folder, you make it "Portable." Any automated tool in the world can import your grader and test your agent without needing to change a single line of your simulator.

### 4. Safety Clamping — Hackathon Compliance

One of the most important features: the **0.01 to 0.99 Score Clamp**:

- **The Problem**: Many automated benchmarks fail if a score is exactly `0.0` or `1.0` due to floating-point math issues.
- **The Solution**: Your `graders/` folder ensures that even a perfect mission returns `0.99` and a total failure returns `0.01`.
- **The Deep Need**: This makes your project **Robust** — your submission will never be rejected for "Out of Range" errors, no matter how the judge's computer is configured.

### 5. Composite Scoring Formula

If you look deep into the grader logic, it uses a weighted sum:

```
Score = (0.8 × Delivery Completion) + (0.1 × Battery Left) + (0.1 × Time Efficiency)
```

- **The Deep Strategy**: This forces the AI to not just "finish," but to finish **optimally**. It rewards drones that take the shortest path and save power — exactly what a real-world delivery company wants.

> 🏆 **DEEP SUMMARY**: The Grader is your Project's Reputation. It translates complex physics and neural movements into a **Single Number** that the world can understand. It is the bridge between a "Cool Simulation" and a "Validated Engineering Benchmark." **It is the "Final Whistle" that makes the game official.** 🏁🚀📡

> 💡 **Ready for Review**: Your `hard_delivery` logs will soon be passed through these graders to calculate your final training performance!

---

## RESULTS/ — The Visual EKG

> **Think of the `results/` folder as the "Visual EKG" (Heart Monitor) of your AI.** While `data/` stores raw memories, `results/` turns that data into **Static Evidence** that humans can understand at a glance.

### 1. Evidence of Intelligence — Reward Curves

This is the most important file in the folder.

- **How it works**: Plots your "Cumulative Reward" over time.
- **Deep Interpretation**:
  - If the line is **trending upwards** → your AI is successfully learning and abstracting rules from the environment.
  - If the line is **flat** → the AI is "stuck" or the task is too hard.
- **The Deep Need**: You cannot verify "intelligence" by looking at 11,000 lines of JSON logs. The `results/reward_curve.png` is your visual proof that "The Agent is smarter at Episode 100 than it was at Episode 1."

### 2. Theoretical Stability — Loss Curves

The `loss_curve.png` looks at the "Math Error" of the Neural Network.

- **High Loss**: The AI is still confused and its mathematical predictions are far from reality.
- **Declining/Flat Loss**: The AI's "Internal Model" of the city has converged. It now knows exactly what will happen when it moves `UP` or `DOWN`.
- **The Deep Need**: This tells you **when to STOP training**. If the loss is flat, further training is a waste of electricity — the AI has reached its maximum potential for that task.

### 3. Isolated Benchmarking — Task Folders

Just like `data/`, results are split into `/easy`, `/medium`, and `/hard`:

- You might notice the `loss_curve.png` in "Easy" flattens at Episode 50, but in "Hard," it's still dropping at Episode 100.
- This tells you that the "Hard" task needs more time or a bigger brain. Without folder isolation, you wouldn't know which task is responsible for which learning patterns.

### 4. Scientific Portability — Git & Sharing

We explicitly configured Git to **allow** these `.png` files while blocking the heavy `.pth` weights.

> In the AI community we say: *"Show me the curves, not the weights."* Anyone can run your code, but showing them the **Result** images proves that your current code actually *achieves* the performance you claim.

```
results/
├── easy/
│   ├── reward_curve.png    # Reward over training episodes (Easy)
│   └── loss_curve.png      # Neural loss convergence (Easy)
├── medium/
│   ├── reward_curve.png    # Reward over training episodes (Medium)
│   └── loss_curve.png      # Neural loss convergence (Medium)
└── hard/
    ├── reward_curve.png    # Reward over training episodes (Hard)
    └── loss_curve.png      # Neural loss convergence (Hard)
```

> 📈 **DEEP SUMMARY**: The `results/` folder is your **Bridge to Human Trust**. Neural Networks are "Black Boxes" — we don't know what they are thinking. But by visualizing the Rewards and Loss, we turn that black box into a transparent engine. **It is the "Certificate of Completion" for every training run you perform.** 📈🚁📡

> 💡 **Generating Soon**: As your `hard_delivery` training finishes, it will automatically generate these curves for the "Hard" difficulty!

---

## RL/ — The Neural Brain

> **The `rl/` folder is the "Neural Core" of your project.** If `core/` is the body, `rl/` is the **Brain**. It is the bridge between **Sensing** (seeing the grid) and **Acting** (moving the drone).

### 1. `model.py` — The Brain's Structure

This file contains the **PathQNet**, your custom Neural Network architecture.

**The Deep Layers:**

1. **The Input Layer**: Receives a flat 1D vector representing the city grid, the drone's energy, and the target location.
2. **The Hidden Layers**: These are the "Synapses." They use complex math (Linear layers and ReLU activation) to find hidden patterns — e.g., *"If I am at (2,3) and the package is at (10,10), the best general direction is South-East."*
3. **The Output Layer**: Outputs 5 numbers (logits). Each number represents the "Estimated Value" (the Q-Value) of one action.

> **The Deep Importance**: This is the **Capacity** for intelligence. If this network is too small, the drone will be "forgetful." If it's too big, it will "overfit" and fail when the city map changes.

### 2. `trainer.py` — The Learning Process

This is where the actual Reinforcement Learning magic happens.

- **The Replay Buffer**: The drone's **Short-Term Memory**. It doesn't learn from just one step; it stores 10,000 recent experiences and "reviews" them in random batches. This prevents the AI from getting "stuck" in a single repetitive movement pattern.
- **The Optimizer (Adam)**: Compares the AI's guess to the actual reward received and tweaks the numbers in `model.py` to make the error smaller next time.
- **Epsilon-Greedy Strategy**: Balances **Exploration** (trying new things) with **Exploitation** (using known good moves). Epsilon starts high (lots of exploration) and decays over time.

### 3. `policy.py` — The Decisive Will

Translates raw neural numbers into real actions.

- **The Decision Logic**: Takes the 5 numbers from `model.py` and picks the largest one (`argmax`).
- **Safety Wrapper**: Ensures that if the model predicts something impossible (like moving into a wall), the system handles it gracefully.
- **The Deep Importance**: During **Inference** (when you aren't training), `policy.py` is the only thing the server needs to run. It is the "Final Output" of all your hard work.

### RL Module Files

| File | Responsibility |
|------|---------------|
| `rl/model.py` | PathQNet neural network architecture (DQN) |
| `rl/trainer.py` | Replay buffer, Adam optimizer, Epsilon-Greedy strategy |
| `rl/policy.py` | Action selection, safety wrapper, inference logic |

> 🧠 **DEEP SUMMARY**: The `rl/` folder is where you turn a **Robot** into an **Agent**. A robot just follows instructions (code). An **Agent** learns from its own experience. The most important thing: the `rl/` folder is **Generic** — it doesn't know it's a drone. It only knows it's a "thing" that gets "positive numbers" when certain "input numbers" line up. This abstraction is what makes AI powerful. **It is the engine of Autonomous Behavior.** 🧠🛸📡

> 💡 **Currently Optimizing**: Your GPU is currently crunching the code in `rl/trainer.py` to update the neurons in `rl/model.py` right now!

---

## SERVER/ — The Global Control Tower

> **Think of the `server/` folder as the "Global Control Tower" and the "Public Face" of your project.** While `rl/` is the brain and `core/` is the body, `server/` allows the outside world — dashboards, judges, and users — to interact with them.

### 1. The Communication Hub — `app.py`

Built on **FastAPI**, this is the "Language Translator" of your system.

- **The Bridge**: Translates internal Python objects (like `DroneInfo`) into **JSON data** that a web browser can understand.
- **The Routes**:

| Endpoint | Purpose |
|----------|---------|
| `GET /reset` | The "New Game" signal — creates a fresh map and episode |
| `POST /step` | The "Heartbeat" — each call moves the world forward by 1 step |
| `GET /analyse` | Reads `data/` folder and aggregates all training stats into a report |
| `GET /graders` | Returns a list of all registered evaluation functions |
| `GET /tasks` | Exposes live configuration data directly from `core/tasks.py` |
| `GET /analyse/{task_id}` | Provides deep RL analytics from `memory.json` |

- **The Deep Importance**: This makes your code **Universal**. Because it uses standard HTTP (REST), an agent written in Java, C++, or even a person using a mobile phone can control your drones.

### 2. The Integrated Frontend — `static/`

This is the **Dashboard** you see in your browser.

- **Real-Time Visualization**: Uses a high-speed polling loop to keep the browser in sync with the Python world. Paints the grid, the drones, the trees, and the parcels.
- **Telemetry UX**: Turns boring numbers like `battery: 0.823` into visual gauges and progress bars.
- **The Deep Importance**: Provides **Trust through Visibility**. When a judge sees the drone navigating around a complex obstacle on the dashboard, it is much more convincing than just seeing `Success: True` in a text file.

### 3. The Package Wrapper — `grid_world_environment.py`

- **Standardization**: Wraps your entire `core/` logic into a class named `DroneDeliveryEnvironment`.
- **Compatibility**: Ensures your project follows the **OpenEnv Specification**.
- **The Deep Importance**: This is what allows your project to be one-click deployable. When you push to Hugging Face, the Meta automated systems look for this specific file to understand how to "Turn On" your environment.

### 4. Deployment Orchestration — `Dockerfile`

- **Isolated Containment**: Defines exactly which version of Python, PyTorch, and FastAPI your server needs.
- **The Deep Importance**: Makes your project **Immortal**. Ensures that 5 years from now, your project will still run exactly the same way, regardless of how your computer's OS has changed.

> 🌐 **DEEP SUMMARY**: The `server/` folder is your **Project's Interface with Reality**. Without the server, your AI is "trapped" in your terminal. With the server, your AI becomes a **Service**. **It is the "Skin" that protects the project and allows it to breathe!** 🌐🚁📡

> 💡 **Space Ready**: This is the folder that drives your [Live Hugging Face Space](https://huggingface.co/spaces/manikandan-n-07/drone_env). Every time someone visits that URL, they are talking to the code in this `server/` folder!

---

## SRC/ — The Design Gallery

> **Think of `src/` as the "Design Gallery" and "Blueprints" of your project's brand and identity.** While other folders contain the **Logic** (Python) and **Data** (JSON/PTH), `src/` contains the **Visual Language**.

### 1. Static Assets & Branding — `img/`

- **The Icon (`icon.png`)**: The face of your project. It appears in browser tabs (favicon) and on your Hugging Face Space.
- **The Deep Need**: In the professional AI world, **presentation is 50% of the value**. A project with a custom icon and professional branding is taken much more seriously by judges and researchers than a generic script.

### 2. Documentation Visuals — `svg/`

You have a `project_workflow.svg` in this folder.

- **How it works**: A scalable vector graphic that explains your code's architecture.
- **The Deep Need**: Complex AI systems are hard to understand just by reading code. An architect doesn't just give you a pile of bricks; they give you a **Blueprint**. This SVG is the blueprint that helps a new developer (or a judge) understand how the "Neural brain" talks to the "Physics world" at a glance.

### 3. Separation of Concerns — UI vs. Assets

| Folder | Purpose |
|--------|---------|
| `server/static/` | Files the **Browser needs to run** (CSS, JS, HTML) |
| `src/` | Files the **Developer needs to document** (raw images, SVGs, branding assets) |

By keeping `src/` separate, you ensure "Documentation Assets" don't bloat the "Runtime Code." This makes your actual web server faster while keeping your project organized for git viewers.

### 4. Scalability & Resolution

Because we use **SVGs** in `src/`:

- **Infinite Clarity**: You can zoom in 1000% on that workflow diagram and it will never get blurry.
- **The Deep Logic**: Your project is "Future-Proof." Whether someone views it on a 4K monitor or a mobile phone, your architectural diagrams will look crisp and professional.

> 🎨 **DEEP SUMMARY**: The `src/` folder is where you translate **Code into Understanding**. It is the "Translation Layer" between the developers' terminal and the human eye. **It is the "Museum" of your project's history and design decisions.** 🎨📐🚀

---

## TESTS/ — The Quality Control Lab

> **Think of `tests/` as the "Quality Control Lab" and "Insurance Policy" of your project.** In professional software engineering, **code that isn't tested is broken**.

### 1. Unit Testing vs. Integration Testing

| Layer | File | Tests | The Deep Need |
|-------|------|-------|---------------|
| **Unit Tests** | `test_env.py` | Small, specific pieces in isolation. *"If the drone moves one step, is exactly 0.005 battery subtracted?"* | Ensures the "Math" of your world is perfect |
| **Integration Tests** | `test_api.py` | How different parts work together. *"When I call `/reset`, does the `core` engine actually create a new map?"* | Ensures the "Pipes" of your system aren't leaking |

### 2. Regression Protection — The Safety Net

Imagine you want to add a "Fast Mode" to your drones next week. You change the battery logic in `core/`. You might accidentally break the "Hard" level logic.

**The Solution**: Run `uv run pytest tests/`. In 5 seconds, it re-runs 50 different "scenarios." If any fail, it flags the error **before** you push to Hugging Face.

> **The Deep Philosophy**: Testing allows you to **move fast without breaking things**.

### 3. Automated Validation Readiness

Because your project is part of the **OpenEnv benchmark**, other systems will try to "import" and use your environment.

- **The Mocking Pattern**: We use `pytest` to "mock" (simulate) an agent playing in your environment.
- **The Deep Logic**: This proves to the Meta judges that your environment is **"Stable."** It shows that if an agent takes 1,000 steps, the server won't crash and the state remains valid.

### 4. Code Coverage — Measuring Health

Run tests with a "Coverage" flag to see a percentage of how much of your code is actually exercised by tests.

- A project with **90% test coverage** is extremely reliable — almost every line of code has been verified by an automated "Judge."

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=. --cov-report=html

# Specific test files
uv run pytest tests/test_env.py -v
uv run pytest tests/test_api.py -v
```

> 🧪 **DEEP SUMMARY**: The `tests/` folder is what separates **"Hobby Projects"** from **"Industrial AI."** Hobbyists test by hand (clicking around the dashboard). Engineers write tests that verify everything in milliseconds. **It is the "Guardrail" that ensures your project remains professional, stable, and ready for any benchmark!** ✅

> 💡 **Certified Stable**: Run `uv run pytest tests/` anytime to see all these "inspections" pass in real-time!

---

## TMP/ — The Scratchpad

> **Think of `tmp/` as the "Kitchen Counter" or "Scratchpad" of your system.** While `data/` is the **Pantry** (long-term storage), `tmp/` is where you place things **temporarily** while you are working.

### 1. Transient Data vs. Persistent Data

| Type | Folder | Impact if Deleted |
|------|--------|------------------|
| **Persistent** | `data/` | Your AI loses its "Mind" (models) and "History" (logs) — catastrophic |
| **Transient** | `tmp/` | Nothing bad happens — the system just regenerates whatever it needs |

### 2. Process Control — Lock Files

Sometimes your server or training script needs to make sure it is the **only one** running.

- **The "Lock" Pattern**: The server might create `tmp/server.lock`. If you try to start a second server, it checks that folder, sees the file, and says: *"Aha! Someone is already at work here. I will stop."*
- **The Deep Need**: This prevents "Race Conditions" where two processes try to write to your `memory.json` at the same time and corrupt your data.

### 3. Intermediate Artifacts

During training, the system might generate temporary data that it doesn't want to keep forever:

1. The `trainer.py` calculates a massive table of gradients.
2. It saves a temporary dump to `tmp/` to free up RAM.
3. Once "Averaging" is done, it deletes the file.

This keeps your RAM usage low and allows your project to run on smaller computers (like the free tier of a Hugging Face Space) by using the hard drive as "Overflow Memory."

### 4. Sandbox Cleanup — Docker & Safety

In a production Docker environment, the `tmp/` folder is usually wiped clean automatically on restart. By pointing all "garbage" or "temporary" outputs to `tmp/`, you ensure that your main repository stays professional and clean — like having a dedicated trash can in a workshop.

> 📝 **DEEP SUMMARY**: The `tmp/` folder is the **"Buffer" of your Project**. It is the space where the system can be "messy" so that the rest of the folders can stay "clean." It allows us to build **Robust** systems that can handle crashes, restarts, and heavy calculations without hurting the core data. **It is the "Draft Paper" for the system's internal math.**

> 💡 **Maintenance**: You don't need to manually manage this folder. Most scripts and your system will automatically clean it up as needed!

---

## ROOT FILES — The Cockpit

> **The Root Folder Files are the "Cockpit" and the "Manifest" of your entire project.** While the subfolders contain the specialized machinery, the root files are what you use to **pilot** the system and define its **identity**.

### 1. The Entry Points — `train.py` & `inference.py`

These are the **"Main Power Switches"** of your project.

- **`train.py`**: The portal to **Neuro-Evolution**. Initiates the `rl/` brain and puts it into the `core/` world to learn.
- **`inference.py`**: The portal to **AI Implementation**. Bypasses the RL brain and lets a Large Language Model (LLM) take control.
- **The Deep Need**: By keeping these in the root, you provide a clear "Starting Line" for anyone using your project. They don't have to dig through folders to find how to "Start."

### 2. The Blueprint — `openenv.yaml`

This is the **"Official Passport"** of your environment for the Hugging Face Space.

- **The Content**: Tells the Meta OpenEnv benchmark exactly which Python classes to load and which `graders/` to use for each task.
- **The Deep Need**: Without this file, your project is just a collection of code. **With** this file, your project becomes a **standardized benchmark** that can compete on global leaderboards.

### 3. The Instruction Manual — `README.md`

This is your **"Command Center."**

- **The Deep Need**: In professional software, the README is the **User Interface for Developers**. It ensures that your knowledge — and the complex logic we built — is transmitted to whoever uses your code next.

### 4. The Environment Recipe — `pyproject.toml` & `uv.lock`

| File | Purpose |
|------|---------|
| `pyproject.toml` | **Human-Readable** list of what the project needs (`torch`, `fastapi`, `matplotlib`, etc.) |
| `uv.lock` | **Machine-Readable** deterministic lock — records the exact sub-version of every library used |

- **The Deep Need**: Solves the "It works on my machine" problem. Ensures that when you push to Hugging Face, the Space installs the **exact same** software environment you have locally.

### 5. The Security Filters — `.gitignore`

- **The Content**: Ensures your 900MB model files don't go to Git, while your 2KB logs do.
- **The Deep Need**: Keeps your repository **Lean and Professional**. Prevents your history from becoming bloated with "garbage" files.

### 6. The Validation Helpers — `validate-submission.sh` & `models.py`

- **`validate-submission.sh`**: Your local **"Mock Judge."** Runs a dry-run of the Meta scoring system to ensure you won't fail when you push for real.
- **`models.py`**: The **"Universal Dictionary."** Defines the Pydantic schemas (`DroneAction`, `DroneObservation`) that every other folder depends on — the "Common Language" that makes `core`, `rl`, and `server` able to talk to each other.

### Root Files Summary

| File | Role |
|------|------|
| `train.py` | Main Fleet Training Entry Point |
| `inference.py` | LLM-Guided Navigation Runner |
| `models.py` | Unified Pydantic Mappings (Fleet) |
| `openenv.yaml` | Mission Deployment Manifest |
| `pyproject.toml` | Modern Packaging & Dependencies |
| `uv.lock` | Deterministic dependency lock |
| `.gitignore` | Security & repo hygiene filter |
| `validate-submission.sh` | Official Hackathon Validator |
| `Dockerfile` | Production container manifest |
| `client.py` | CLI client for testing |

> 🏢 **DEEP SUMMARY**: The root files are the **"Orchestrator" of your ecosystem**. They don't do the "Heavy Lifting" (the math), but they do the **"Governing"**. They define the standards, the boundaries, and the execution paths for everything else. **If the folders are the "Staff," the root files are the "CEO" and the "Office Policies."** 🏢🚁🚀

---

## Environment Mechanics

### Grid World

The environment generates a procedural city grid with:

- 🏢 **Buildings** — Impassable obstacles
- 🛣️ **Roads** — Valid flight corridors
- 🌳 **Trees** — Soft obstacles (navigable but penalized)
- 📦 **Packages** — Delivery targets placed at random valid locations
- 🚁 **Drones** — Your autonomous agents

### Battery Physics

Each step costs battery proportional to the action taken:

```python
# Battery drain per action type
MOVE_COST    = 0.005   # Normal flight step
WAIT_COST    = 0.001   # Hovering in place
CRASH_COST   = 0.050   # Collision penalty
DELIVERY_BONUS = 1.0   # Full reward on successful delivery
```

### Action Space

| Action ID | Direction | Description |
|-----------|-----------|-------------|
| 0 | UP | Move north one cell |
| 1 | DOWN | Move south one cell |
| 2 | LEFT | Move west one cell |
| 3 | RIGHT | Move east one cell |
| 4 | WAIT | Hover in place |

### Observation Space

The `DroneObservation` object returned after each step:

```json
{
  "grid": [[0, 1, 0, ...], ...],
  "drone_position": [x, y],
  "target_position": [tx, ty],
  "battery": 0.823,
  "manhattan_distance": 12,
  "steps_taken": 47,
  "delivered": false
}
```

---

## Neural Intelligence Layer

### PathQNet Architecture

```python
class PathQNet(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # 5 Q-values for 5 actions
        )

    def forward(self, x):
        return self.network(x)
```

### DQN Training Loop

The training loop in `rl/trainer.py` follows the standard DQN algorithm:

1. **Observe** the current state `s`
2. **Select** action using epsilon-greedy: `a = argmax(Q(s)) or random`
3. **Execute** action → receive reward `r` and next state `s'`
4. **Store** `(s, a, r, s', done)` in replay buffer
5. **Sample** random mini-batch from buffer
6. **Compute** target: `y = r + γ × max(Q(s'))`
7. **Update** network weights by minimizing `(y - Q(s,a))²`
8. **Decay** epsilon: `ε = max(ε_min, ε × ε_decay)`

---

## API Reference

### Core Endpoints

#### `POST /reset`

Initializes a new episode with a fresh procedurally generated map.

```json
// Request body (optional)
{ "task": "hard_delivery" }

// Response
{
  "observation": { ... },
  "task": "hard_delivery",
  "episode_id": "ep_042"
}
```

#### `POST /step`

Advances the simulation by one timestep.

```json
// Request
{ "direction": "UP" }

// Response
{
  "observation": { ... },
  "reward": 0.05,
  "done": false,
  "info": { "battery": 0.818, "steps": 48 }
}
```

#### `GET /analyse/{task_id}`

Returns aggregated analytics from `memory.json` for the specified task.

```json
{
  "task": "hard_delivery",
  "total_episodes": 1000,
  "average_reward": 0.72,
  "success_rate": 0.81,
  "avg_steps_to_delivery": 34.2,
  "action_distribution": { "UP": 0.28, "DOWN": 0.24, ... }
}
```

#### `GET /graders`

Returns all registered evaluation functions from `openenv.yaml`.

```json
{
  "graders": [
    { "task": "easy_delivery",   "function": "drone_env.graders:grade_easy" },
    { "task": "medium_delivery", "function": "drone_env.graders:grade_medium" },
    { "task": "hard_delivery",   "function": "drone_env.graders:grade_hard" }
  ]
}
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/manikandan-n-07/drone-env
cd drone-env

# 2. Install dependencies with uv
uv sync

# 3. Launch the dashboard server
uv run python server/app.py

# 4. Open your browser
# Navigate to http://localhost:8000
```

---

## Training

### Train a Single Task

```bash
# Easy mission (10x10 grid, 1 package)
uv run python train.py --task easy_delivery --episodes 200 --gpu

# Medium mission (14x14 grid, 3 packages)
uv run python train.py --task medium_delivery --episodes 300 --gpu

# Hard mission (18x18 grid, 5 packages)
uv run python train.py --task hard_delivery --episodes 500 --gpu
```

### Train All Tasks Sequentially

```bash
uv run python train.py --task all
```

### Training Output

After training completes:

```
data/
├── easy/model.pth         ← Saved neural weights
├── easy/memory.json       ← Episode logs
results/
├── easy/reward_curve.png  ← Learning progress chart
└── easy/loss_curve.png    ← Network convergence chart
```

---

## LLM-Powered Inference

Run the drone using an LLM as the decision-making brain instead of the trained DQN:

```bash
# Set your API key
export HF_TOKEN=your_token_here

# Run with default model (Qwen)
uv run python inference.py

# Run specific task with step limit
uv run python inference.py --task hard_delivery --steps 100
```

### Supported LLM Backends

| Backend | Environment Variable | Example Model |
|---------|---------------------|---------------|
| Hugging Face Router | `HF_TOKEN` | `Qwen/Qwen2.5-7B-Instruct` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Custom | `API_BASE_URL` | Any OpenAI-compatible endpoint |

---

## Reward Engineering

### Reward Signal Breakdown

| Event | Easy | Medium | Hard |
|-------|------|--------|------|
| Delivery Success | +1.0 | +1.0 | +1.0 |
| Step Closer to Target | +0.01 | +0.01 | +0.01 |
| Step Away from Target | −0.005 | −0.005 | −0.005 |
| Battery drain per step | −0.10 | −0.15 | −0.25 |
| Collision with obstacle | −0.10 | −0.15 | −0.25 |
| Waiting (WAIT action) | −0.001 | −0.001 | −0.001 |

### Why Positive Sparse Rewards?

Traditional RL often uses heavy negative rewards for failures. SkyRelic uses **Positive Sparse Rewards** because:

1. The AI doesn't get discouraged by constant negative signals.
2. The learning signal is cleaner and more stable.
3. It aligns better with the Meta OpenEnv benchmark philosophy.

---

## Grading & Evaluation

### Composite Score Formula

```
Score = (0.8 × Delivery Completion Rate)
      + (0.1 × Remaining Battery %)
      + (0.1 × Time Efficiency)
```

Where:
- **Delivery Completion Rate** = packages delivered / total packages required
- **Remaining Battery %** = battery at end of episode
- **Time Efficiency** = 1 − (steps_taken / max_steps)

All scores are clamped to the `[0.01, 0.99]` range for benchmark compliance.

### Running the Grader Manually

```python
from drone_env.graders import grade_easy, grade_medium, grade_hard

# After an episode completes
result = {
    "delivered": 1,
    "required": 1,
    "battery_remaining": 0.72,
    "steps_taken": 34,
    "max_steps": 100
}

score = grade_easy(result)
print(f"Score: {score:.3f}")  # e.g., 0.871
```

---

## The Life of a Parcel (End-to-End Flow)

1. 🚀 **THE CALL**: You (or an automated agent) send `POST /reset` to the FastAPI server.
2. 🏗️ **THE CREATION**: The **Core Logic** generates a random 10×10 city with roads 🛣️, buildings 🏢, and trees 🌳. It places a **Parcel** 📦 at a random location.
3. 👁️ **THE SIGHT**: The server sends the "State" (JSON) back to the **UI Dashboard**. You see the drone appear in the grid.
4. 🧠 **THE BRAIN**: When you click **Start**, the **Neural Engine** (RL) looks at the map, calculates the distance, and picks the best direction.
5. 🛸 **THE FLIGHT**: The drone moves! The **Physics Engine** drains its battery and checks for crashes against buildings.
6. 🏁 **THE VICTORY**: Once the drone reaches the 📦, the **Grader** calculates your efficiency and updates your score!

---

## Project Structure

```bash
drone_env/
├── core/                        # World Simulation Engine
│   ├── tasks.py                 # Mission configurations & constants
│   └── state_manager.py         # Fleet-wide multi-agent state
├── rl/                          # Reinforcement Learning Layer
│   ├── model.py                 # PathQNet Neural Architecture
│   ├── trainer.py               # Telemetry & Experience Replay
│   └── policy.py                # Autonomous navigation heuristics
├── server/                      # FastAPI Backend & Orchestration
│   ├── app.py                   # API endpoints & log aggregator
│   ├── grid_world_environment.py # Project-wide Environment Interface
│   └── static/                  # Browser-based Dashboard UI
├── graders/                     # Evaluation & Validation Logic
│   ├── easy.py
│   ├── medium.py
│   ├── hard.py
│   └── __init__.py              # Unified Grader Discovery
├── data/                        # Local Intelligence & Persistence
│   ├── easy/                    # Easy task: model.pth & memory.json
│   ├── medium/                  # Medium task: model.pth & memory.json
│   └── hard/                    # Hard task: model.pth & memory.json
│   └── train.log                # Unified training engine logs
├── results/                     # Neural Performance Evidence
│   ├── easy/                    # Reward and Loss Curves (Easy)
│   ├── medium/                  # Reward and Loss Curves (Medium)
│   └── hard/                    # Reward and Loss Curves (Hard)
├── src/                         # Branding & Documentation Assets
│   ├── img/icon.png             # Project favicon & branding
│   └── svg/project_workflow.svg # Architecture blueprint diagram
├── tmp/                         # Transient runtime scratchpad
├── tests/                       # Automated API & Physics Tests
├── train.py                     # Main Fleet Training Entry Point
├── inference.py                 # LLM-Guided Navigation Runner
├── models.py                    # Unified Pydantic Mappings (Fleet)
├── openenv.yaml                 # Mission Deployment Manifest
├── pyproject.toml               # Modern Packaging & Dependencies
├── uv.lock                      # Deterministic dependency lock
├── Dockerfile                   # Production container manifest
├── client.py                    # CLI client for testing
└── validate-submission.sh       # Official Hackathon Validator
```

---

## Configuration Reference

### `pyproject.toml` Dependencies

```toml
[project]
name = "drone-env"
version = "0.3.0"
requires-python = ">=3.10"
dependencies = [
    "openenv-core[core]>=0.2.1",
    "torch>=2.0.0",
    "openai>=1.0.0",
    "python-multipart>=0.0.9",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    "matplotlib>=3.7.0",
]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token for LLM inference |
| `OPENAI_API_KEY` | — | OpenAI API key (alternative to HF) |
| `API_BASE_URL` | HF Router URL | Override LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | LLM model identifier |
| `DRONE_TASK` | `easy_delivery` | Default task for inference runner |
| `LOCAL_IMAGE_NAME` | `drone-inference-v1` | Local Docker image tag |

---

## Docker Deployment

```bash
# Build the Docker image
docker build -t drone_env .

# Run locally
docker run -p 8000:8000 drone_env

# Run with GPU support
docker run --gpus all -p 8000:8000 drone_env
```

### `Dockerfile` Overview

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
EXPOSE 8000
CMD ["uv", "run", "python", "server/app.py"]
```

---

## Hugging Face Submission

```bash
# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/manikandan-n-07/drone_env

# Validate before pushing
uv run openenv validate

# Deploy to Hugging Face Space
git push hf main
```

---

## Phase 2 Validation Updates

The **SkyRelic** environment has been updated to fully comply with the **Meta PyTorch Hackathon Phase 2 Deep Validation** requirements.

### 🛡️ Validation Fixes

- **Strict Score Clamping**: All mission scores and rewards are now strictly clamped to the **(0.01, 0.99)** range in the `graders/` package and `server/grid_world_environment.py`. This prevents the "out of range" (exactly 0.0 or 1.0) failures reported by the automated validator.
- **Full Identity Sync (Grader Discovery)**: Task and grader identifiers have been synchronized across the manifest (`openenv.yaml`), backend API, and simulation core using full Python module paths (e.g., `drone_env.graders:grade_easy`).
- **Differentiated Reward Scalars**: Reward scalars for step, wait, and collision penalties have been updated to difficulty-specific tiers:
  - **Easy Mission**: 0.10 (10%)
  - **Medium Mission**: 0.15 (15%)
  - **Hard Mission**: 0.25 (25%)
- **Task Discovery**: Fully registered 3 tasks (`easy_delivery`, `medium_delivery`, `hard_delivery`) with corresponding graders in `openenv.yaml`. The server now exposes a `/graders` endpoint for official task discovery.

### 📊 Dashboard UI Improvements

- **Technical Specifications Legend**: A new side-by-side comparison table has been added to the dashboard, allowing manual reviewers to verify grid sizes and reward weights for all 3 mission levels at a glance.
- **Auto-Analysis Engine**: Upon mission completion, the dashboard now automatically triggers an asynchronous call to `/analyse`, providing immediate feedback on **Average Reward**, **Success Trends**, and **Action Distributions**.
- **Refined Analytics**: Removed redundant "(Success Trend)" text from the completion modal for a cleaner, professional report format.

### 📡 API & Backend

- **New Endpoints**:
  - `/graders`: Returns a list of all registered evaluation functions.
  - `/tasks`: Exposes live configuration data directly from `core/tasks.py`.
  - `/analyse/{task_id}`: Provides deep RL analytics from `memory.json`.

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=. --cov-report=html

# Specific test files
uv run pytest tests/test_env.py -v
uv run pytest tests/test_api.py -v
```

---

## Contributing

1. Fork the repository on Hugging Face Hub
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes with descriptive messages
4. Run the test suite and validator before submitting: `uv run pytest tests/ -v`
5. Run local validation: `uv run openenv validate`
6. Open a Pull Request against `main`

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

Build system uses [Meta's BSD-licensed](https://opensource.org/licenses/BSD-3-Clause) `setuptools` configuration template.

---

<div align="center">

**Built with 🚁 for the OpenEnv ecosystem**

*Advancing autonomous agent research through high-fidelity simulation*

</div>

---

## Author

<div align="center">

# Manikandan N

*Developer & Creator of Drone Delivery Environment*

[![GitHub](https://img.shields.io/badge/GitHub-@manikandan--n--07-181717?style=flat-square&logo=github)](https://github.com/manikandan-n-07)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Manikandan_N-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/manikandan-n-35a1bb294)
[![Email](https://img.shields.io/badge/Email-manilunar07@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:maniluna07@gmail.com)

</div>

---