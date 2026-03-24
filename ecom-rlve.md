---
title: "Ecom-RLVE: Adaptive Verifiable Environments for E-Commerce Conversational Agents"
thumbnail: /blog/assets/ecom-rlve/thumbnail.png
authors:
- user: thebajajra
  guest: true
  org: owlgebra-ai
- user: ai-queen
  guest: true
  org: owlgebra-ai
- user: pmonad
  guest: true
  org: owlgebra-ai
- user: burtenshaw
---

# Ecom-RLVE: Adaptive Verifiable Environments for E-Commerce Conversational Agents

[![Data](https://img.shields.io/badge/🤗%20Catalog%20Data-Amazebay2M-yellow)](https://huggingface.co/datasets/owlgebra-ai/Amazebay-catalog-2M)
[![Code](https://img.shields.io/badge/Github-Code-black)](https://github.com/owlgebra-ai/EcomRLVE-Gym)

> **TL;DR** — We extend the RLVE framework ([Zeng et al., 2025](https://arxiv.org/abs/2511.07317)) from single-turn reasoning puzzles to **multi-turn, tool-augmented e-commerce conversations**. EcomRLVE-GYM provides 8 verifiable environments — product discovery, substitution, cart building, returns, order tracking, policy QA, bundle planning, and multi-intent journeys — each with procedural problem generation, a 12-axis difficulty curriculum, and algorithmically verifiable rewards. We train a Qwen 3 1.7B model with DAPO over 300 steps and present early results demonstrating that environment scaling and adaptive difficulty transfer to agentic, real-world task completion.
The project is evolving, started with [Pytorch OpenEnv Hackathon](https://cerebralvalley.ai/e/openenv-hackathon-sf).

---

## What a training episode looks like

Before we explain the framework, here is what a single EcomRLVE episode looks like at `d = 4`. The environment generates a hidden goal, a simulated user opens the chat, and the agent must use tools to satisfy the request. Every action is verified algorithmically — no LLM judge required.

```
╭──────────────── Episode: E_CART  d=4 ──────────────────╮
│ Hidden goal:                                            │
│   items:                                                │
│     - Anker 65W Charger, variant=USB-C, qty=2           │
│     - Spigen MagSafe Case, variant=Matte Black, qty=1   │
│                                                         │
│ Turn 1  [user]                                          │
│   "I need two 65-watt Anker chargers (USB-C) and a      │
│    Spigen MagSafe case in matte black."                  │
│                                                         │
│ Turn 2  [agent]                                         │
│   tool_calls:                                           │
│     - catalog.search("Anker 65W charger")               │
│     - catalog.search("Spigen MagSafe case")             │
│   assistant_message: "Let me find those for you."       │
│                                                         │
│ Turn 3  [agent]                                         │
│   tool_calls:                                           │
│     - catalog.get_variants("prod_8832")                 │
│     - catalog.get_variants("prod_2291")                 │
│   assistant_message: "Checking available variants…"     │
│                                                         │
│ Turn 4  [agent]                                         │
│   tool_calls:                                           │
│     - cart.add("prod_8832", variant="USB-C", qty=2)     │
│     - cart.add("prod_2291", variant="Matte Black",      │
│                qty=1)                                    │
│   answer: { "done": true }                              │
│                                                         │
│ Verifier:  F1 = 1.0  ✓   r_task = +1.0                 │
│            T_eff = 3      r_eff  = +0.5                 │
│            hall_rate = 0  r_hall =  0.0                  │
│            ───────────────────────────                   │
│            r_total = 0.75·1.0 + 0.15·0.5 + 0.10·0.0    │
│                    = 0.825                               │
╰─────────────────────────────────────────────────────────╯
```

The reward is fully computed by code: F1 over `(product, variant, qty)` tuples, an efficiency bonus for finishing in fewer turns, and a hallucination check that every recommended product ID was actually retrieved. If the agent had picked the Lightning variant instead of USB-C, the simulated user would have corrected it mid-dialogue — and the F1 would have dropped.

---

## Why RL for shopping agents?

Large language models can hold fluent conversations, yet deploying them as shopping assistants reveals a persistent gap: **fluency ≠ task completion**. A customer who asks *"find me a USB-C charger under \$25 that ships in two days"* needs an agent that invokes the right catalog search, filters on three hard constraints, avoids hallucinating product IDs it never retrieved, and handles follow-ups when the top result goes out of stock.

Supervised fine-tuning can teach surface-level tool use from demonstrations, but it cannot scale to the combinatorial space of constraint configurations, partial-information dialogues, and multi-step transactional workflows that real e-commerce demands.

Reinforcement learning with verifiable rewards (RLVR) offers an alternative: the agent optimises for *outcomes* — did the products satisfy the constraints? Was the cart correct? Was the return initiated for the right order line? The challenge is constructing reward functions that are both **verifiable** (no LLM-as-a-judge subjectivity) and **adaptive** (difficulty that grows with the policy's capability).

### From RLVE-Gym to EcomRLVE-GYM

[RLVE](https://arxiv.org/abs/2511.07317) (Zeng et al., 2025) introduced adaptive verifiable environments and built RLVE-Gym — 400 environments for sorting, multiplication, Sudoku, and other algorithmic-reasoning tasks. But those are all **single-turn, text-in / text-out** puzzles. The paper's own future-work section calls for extending to agentic domains.

EcomRLVE-GYM fills that gap: we stay in the **verifiable** regime (e-commerce outcomes *can* be checked algorithmically) while extending to **multi-turn, tool-augmented, agentic** conversations — environments where the agent must *act* (call tools, modify world state) rather than merely *reason* (produce a text answer).

E-commerce is a natural fit because customer-service outcomes are structurally verifiable:

![verifiable_signals_dark](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/dA0i6ZB3JDG-rqQtLRCy0.png)

Every signal above can be evaluated by a program with access to the hidden ground-truth goal. No human annotation or LLM-as-a-judge is needed.

---

## What the agent outputs

Every turn, the model produces a single JSON object — a message to the user, optional tool calls, and an optional answer submission:

```json
{
  "assistant_message": "Let me find those chargers for you.",
  "tool_calls": [
    {"name": "catalog.search", "args": {"query": "Anker 65W charger USB-C"}}
  ],
  "answer": null
}
```

When the agent believes the task is complete, it sets `"answer": {"done": true, ...}` with environment-specific fields (recommended IDs, selected order, etc.). Invalid JSON triggers immediate termination with `r = -1`, creating a strong gradient toward well-formed outputs from step one.

---

## The eight environments

Each environment is a tuple `E = (I, P, R)`: an **input** template, a procedural **problem generator** parameterised by difficulty `d`, and an algorithmic **reward verifier**. Rewards are terminal-only and lie in `[-1, 1]`.

| ID | Environment | What the agent does | Key reward signal | Pass condition |
|----|-------------|---------------------|-------------------|----------------|
| `E_PD` | Product Discovery | Find products meeting constraints | nDCG + constraint satisfaction | `r_task ≥ 0.95` |
| `E_SUB` | Substitution | OOS item → find alternative | Similarity-weighted nDCG | `r_task ≥ 0.95` |
| `E_CART` | Cart Building | Add correct items / variants / qty | Variant-aware F1 | `F1 = 1.0` |
| `E_RETURN` | Return + Replacement | Identify order line, initiate return | Selection + initiation + replacement | All sub-rewards pass |
| `E_STATUS` | Order Tracking | "Where is my order?" | Order ID + status match | Both exact match |
| `E_POLICY` | Policy QA | Answer deterministic policy question | Exact / ratio match | `r_task ≥ 0.95` |
| `E_BUNDLE` | Bundle Planning | Shopping list for a project | Category F1 − budget penalty | `F1 = 1` and within budget |
| `E_JOURNEY` | Multi-Intent Journey | Chained sub-tasks in one conversation | Average of sub-task rewards | All `r_j ≥ 0.95` |

The agent interacts with **15 tools** across five domains:

| Domain | Tools |
|--------|-------|
| **Catalog** | `catalog.search`, `catalog.rerank`, `catalog.get_product`, `catalog.get_variants` |
| **Cart** | `cart.view`, `cart.add`, `cart.remove`, `cart.set_quantity` |
| **Orders** | `order.list`, `order.get_status`, `order.checkout` |
| **Returns** | `return.check_eligibility`, `return.initiate`, `return.exchange` |
| **Policy** | `policy.search` |

Three environments worth highlighting beyond E_CART (covered in depth below):

**E_SUB — Substitution.** The user's desired product is out of stock. The agent must find alternatives that are both *similar* to the original and satisfy compatibility constraints. The reward blends cosine similarity with constraint satisfaction, and the similarity weight increases with difficulty — at high `d`, the user insists on something very close to the original, not just any compatible product.

**E_BUNDLE — Bundle Planning.** Given a project goal (*"I'm setting up a home office"*), the agent recommends products covering all required categories within a budget. The reward is category F1 minus a budget penalty `max(0, (cost - B) / B)` — overspending by 100%+ is harshly penalised, creating a strong gradient against ignoring price.

**E_JOURNEY — Multi-Intent Journey.** The most complex environment: the user chains 2–5 sub-tasks in one conversation (e.g., find a charger, then return a defective cable, then check order status). Each sub-task is scored by its atomic verifier, and `IsCorrect = 1` only if *every* sub-task scores ≥ 0.95 — the agent must near-perfectly complete them all.

Every environment shares a **composite reward**:

```
r = clip(0.75 * r_task + 0.15 * r_eff + 0.10 * r_hall, -1, 1)
```

with hard-fail override (`r = -1`) for invalid JSON, illegal tool calls, or safety violations.

**Fair efficiency scoring with UserActs.** A naive turn-count penalty punishes the agent for every turn — including ones the *user* caused. To fix this, the user simulator tags each response with a structured dialogue act:

| UserAct | Meaning | Penalised? |
|---------|---------|-----------|
| `confirm` | User confirms the agent's action | No |
| `clarify` | User provides previously omitted info | No |
| `correct` | User points out an agent mistake | Yes |
| `elaborate` | User adds new requirements | Yes |
| `ragequit` | User abandons the conversation | Yes |

The effective turn count discounts non-penalty acts: `T_eff = max(1, T - T_user_clarify)`. An agent that solves the task in 3 turns — one of which answers a user confirmation — pays efficiency cost for only 2 effective turns. `r_eff = 1 - 2·(T_eff - 1) / (T_max - 1)`.

**Hallucination penalty** checks whether recommended product IDs were actually retrieved: \(\text{hall\_rate} = |\{p \in L : p \notin \text{Seen}\}| / \max(|L|, 1)\), \(r_{\text{hall}} = -\text{clip}(\text{hall\_rate}, 0, 1)\). The agent cannot invent product IDs.

---

## Adaptive difficulty curriculum

RLVE uses a single integer `d` to parameterise difficulty. For algorithmic tasks, `d` maps to one structural parameter (array length, digit count). E-commerce is harder along *many dimensions at once*: number of constraints, missing information, retrieval noise, order-history depth. Collapsing these into a single number conflates orthogonal sources of difficulty.

![Screenshot 2026-03-08 at 11.27.11](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/SALZRvBC6TP1HG1ZxqWsh.png)

Our solution: a **12-dimensional difficulty vector** `θ(d)` that maps integer `d` to 12 generator parameters. Here are four representative axes to give the flavour:

| Axis | What it controls | d = 0 | d = 6 | d = 12 |
|------|-----------------|-------|-------|--------|
| **Constraint count** `m(d)` | How many product requirements the user has | 2 | 5 | 8 |
| **Slot omission** `p_missing(d)` | Probability the user omits a constraint | 5% | 70% | ~80% |
| **Retrieval noise** `ε_rank(d)` | Fraction of search results replaced by distractors | 0% | 12% | 24% |
| **Out-of-stock rate** `p_oos(d)` | Items that become unavailable mid-episode | 0% | 30% | 50% |

The remaining eight axes control output size, turn budget, input noise, context switches, retrieval depth, order-history depth, policy complexity, and tool budget. The full table is in the [companion technical report](https://github.com/owlgebra-ai/EcomRLVE-Gym).

**How difficulty advances.** Following RLVE, each environment maintains an independent sliding window `[l_i, h_i]`. Episodes sample `d ~ Uniform[l_i, h_i]`. After 32 rollouts at the upper bound, if the agent passes ≥ 90% of them, the window advances by one. A `d_delta = 4` cap keeps the window width at most 5 levels, ensuring the agent always trains near its capability frontier.

---

## Deep dive: Cart Building (E_CART)

Cart building is a good showcase because it requires the full search → inspect → act loop, has a binary ground truth, and introduces a challenge absent from most recommendation benchmarks: **variant selection**.

### The problem

The generator samples 1–5 target products (scaling with `d`), each potentially requiring a specific variant (USB-C vs Lightning, Matte vs Glossy) and a quantity > 1. The agent must:

1. Search the catalog to find each product
2. Call `catalog.get_variants` to see available options
3. Add the correct `(product_id, variant_id, qty)` tuples to the cart

### Why variants matter

Real product catalogs have sparse variant data — many products have none, and those that do typically vary only by colour or size. To create a richer discrimination task, we **synthesize variants at episode initialization**:

- A per-category priority list picks the most natural attribute to vary (electronics → `connector_type`; clothing → `size`; kitchen → `material`).
- For each target product, we generate 3 variants: 1 target + 2 plausible distractors. An "Anker 65W USB-C Charger" produces `{USB-C, Lightning, HDMI}`.
- The verifier checks **composite keys** `(product_id, variant_id)` — correct product but wrong variant means the unit is unmatched.

### Difficulty scaling

| Axis | d = 0 | d = 3 | d = 6 | d = 9 |
|------|-------|-------|-------|-------|
| **Distinct items** | 1 | 2 | 3 | 4 |
| **Variant required** | 21% | 66% | 93% | 99% |
| **Multi-quantity** | 0% | 30% | 50% | 50% |

At `d = 0` the agent adds a single product with no variant complexity — learning the basic `catalog.search → cart.add` workflow. At `d = 6` it juggles 3 items, nearly all requiring a specific variant, with half needing qty > 1.

### Reward

$$F_1 = \frac{2 \cdot \text{prec} \cdot \text{rec}}{\text{prec} + \text{rec} + \epsilon}$$

Precision and recall are measured over composite-keyed unit quantities. `IsCorrect` requires `F1 = 1.0` — *exact* cart correctness including variant selection.

**Mid-dialogue feedback.** If the agent adds the wrong variant, the simulated user corrects it (*"that's the Lightning version, but I need USB-C"*), creating a learning signal for self-correction within the dialogue.

### Trajectories: easy vs. hard

Two real E_CART episodes from a Qwen 3 1.7B agent. Same environment, same agent — difficulty alone changes the game.

| | **d = 1** — 1 item, no variants | **d = 8** — 3 items, variants + typos |
|---|---|---|
| **User message** | *"Shopping list: SonicWave Keypad. Please add everything."* | *"pls add the following to my **aart**: Eco-**Frienly HrenLefa** Charger by GreenLeaf, High-Performance **AqaPure** Air Fryer , **Lighrweiht** PixelVault Tank Top (3)."* — typos from noise axis |
| **Turn 1** | `user.get_visit_history()` → 4 results. Spots the target immediately. | `user.get_visit_history()` → 9 results. Must disambiguate across chargers, fryers, tops. |
| **Turn 2** | `cart.add("syn_000162", qty=1)` → Cart: 1 item, $2.74. *"I found the SonicWave Keypad."* User: *"Great, that's the exact one."* | No tool calls — presents 3 product cards. User: *"Add the Charger with 100W output, Air Fryer with Charcoal filter, 3 Tank Tops in size XS."* |
| **Turn 3** | `answer: { "done": true }` — **Done in 3 turns.** | `catalog.get_variants` × 3 → charger: 6 variants (100W, 120W, 30W…), air fryer: 6 (Bamboo, Permanent, **Charcoal**…), tank top: 7 (**XS**, S, XL…). User: *"I need three Tank Tops in XS."* |
| **Turn 4** | | `cart.add(charger, "100W", qty=1)` ✓ `cart.add(fryer, "Bamboo", qty=1)` ✗ should be Charcoal. `cart.add(tank_top, "size_XL", qty=3)` ✗ should be XS. User: *"Air Fryer has Bamboo not Charcoal, Tank Top is XL not XS."* |
| **Turn 5** | | Re-fetches tank top variants, adds 2× XS — **doesn't fix the air fryer**. User: *"Air Fryer still shows Bamboo. I need three XS, not two."* |
| **Turn 6** | | Re-fetches tank top variants *again* — **still ignores the air fryer**. User: *"Please change the Air Fryer variant."* |
| **Turn 7** | | *"I cannot find a Charcoal filter variant."* — It exists; the agent saw it in turn 3 but never selected it. User: *"Skip the Air Fryer then."* |
| **Turn 8** | | `answer: { "done": true }` — Cart still wrong: Charger ✓, Air Fryer (Bamboo) ✗, Tank Top (3×XL + 2×XS) ✗ |
| **Reward** | `r_task` = +1.00, `r_eff` = +0.33, `r_hall` = 0.00, **r_total = +0.80** ✓ | `r_task` ≈ 0.00, `r_eff` = −0.43, `r_hall` = 0.00, **r_total = −0.06** ✗ |
| **Outcome** | Cart matches goal. 3 turns, 2 effective. | Wrong variants, wrong quantities, user gave up. 8 turns, 6 effective. |

At d=1 the agent solves the task in 3 clean turns. At d=8 it spirals — picking Bamboo instead of Charcoal, XL instead of XS, never fixing the air fryer despite two user corrections, then hallucinating that the variant doesn't exist. This is exactly the kind of multi-step error cascade that the difficulty curriculum surfaces, and that adaptive training should teach the agent to recover from.

---

## User simulation

A verifiable environment needs a user simulator that behaves consistently but diversely. We use **Qwen3.5 (9.7B)** as the dialogue backbone, with two key mechanisms:

**Constraint-aligned persona weights.** Each episode samples a 5-dimensional preference weight vector `w` from a Dirichlet distribution. Dimensions corresponding to active constraints (price, rating, shipping, brand, similarity) are *boosted* — so if the user says "under \$25", the verifier's hidden utility actually cares about price. This eliminates a subtle observability inconsistency where the agent could be penalised for listening to the user.

**LLM-verbalized constraints.** Instead of template-based slot filling (which only covered 4 attribute types), the LLM generates natural initial messages covering all 17+ constraint attributes. The LLM also controls *strategic omission* — deliberately withholding some constraints to force the agent to ask clarifying questions, with explicit tracking of what was mentioned vs. omitted so the verifier doesn't penalise the agent for information it never received.

---

## Environment scaling

Following RLVE's methodology, we define nested environment collections:

**C1 ⊂ C2 ⊂ C4 ⊂ C8**

| Collection | Environments | Skills trained |
|------------|-------------|----------------|
| **C1** | Product Discovery | Retrieval + recommendation |
| **C2** | + Substitution | Similarity reasoning under constraints |
| **C4** | + Cart, Returns | Transactional workflows (cart manipulation, return initiation) |
| **C8** | + Status, Policy, Bundle, Journey | Knowledge retrieval, planning, compositionality |

We hypothesise — consistent with RLVE's findings — that C8 agents outperform single-environment specialists, even on the specialist's own task.

---

## Early results

We trained Qwen 3 1.7B with DAPO on C1 (product discovery) for 300 steps as an initial viability study.

| | Config |
|---|--------|
| **Base model** | Qwen 3 1.7B |
| **Algorithm** | DAPO (G = 4 rollouts/prompt) |
| **LR** | 1e-5 |
| **Catalog** | 2M products, FAISS index with `thenlper/gte-small` (384-dim) |
| **User sim** | Qwen3.5 9.7B |

![accuracy_10_levels_dots_each_reach (1)](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/sGyMSKDOJ4tqiRSgV7AOR.png)

We saw progressive growth in difficulty reached, confirming that adaptive scheduling produces a steady learning signal rather than the saturation (static-low) or starvation (static-high) patterns predicted by the RLVE paper.

---

## Try it yourself

Run an episode directly in your browser — pick an environment, set the difficulty, and watch the agent work:

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"
></script>

<gradio-app theme_mode="dark" space="thebajajra/EcomRLVE-Gym"></gradio-app>

The environments, verifiers, and training configs are all open-source:

```bash
git clone https://github.com/owlgebra-ai/EcomRLVE-Gym
cd EcomRLVE-Gym
pip install -e .
```

The 2M-product catalog is on the Hub:

```python
from datasets import load_dataset

catalog = load_dataset("owlgebra-ai/Amazebay-catalog-2M", split="train")
print(f"{len(catalog)} products loaded")
```

---

## What we inherit from RLVE — and what we add

| From RLVE | New in EcomRLVE |
|-----------|-----------------|
| `E = (I, P, R)` abstraction | Multi-turn dialogue episodes with tool use |
| Adaptive `[l, h]` sliding window | 15 tools across 5 domains; world-state mutation |
| Nested collections C_k | 12-axis difficulty (single `d` → rich parameter vector) |
| DAPO as RL algorithm | Persona-driven evaluation with Dirichlet-sampled weights |
| | LLM user simulator (Qwen3.5 9.7B) |
| | Synthetic variant generation + variant-aware F1 |
| | UserAct-based fair efficiency scoring |

---

## Conclusion

EcomRLVE-GYM shows that the RLVE framework — adaptive verifiable environments with procedural generation and algorithmic rewards — extends naturally from single-turn reasoning puzzles to multi-turn, tool-augmented e-commerce conversations. The key enablers are:

1. **Structural verifiability** of e-commerce outcomes (constraint satisfaction, cart correctness, order identification)
2. **Multi-axis difficulty** that captures the independent sources of complexity in real shopping conversations
3. **Persona-driven user simulation** that creates diverse but verifiable evaluation signals

By releasing 8 environments with a 12-dimensional difficulty curriculum, we provide the community with a concrete, extensible testbed for training agentic LLMs with RL — filling the gap between RLVE-Gym's algorithmic puzzles and the open challenge of real-world, tool-augmented task completion.

---

## References

1. Zeng, Z., Ivison, H., Wang, Y., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments.* ICML 2025. [arXiv:2511.07317](https://arxiv.org/abs/2511.07317)

2. Yu, Q., Zhang, Z., Zhu, R., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

3. Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

4. DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning in LLMs through Reinforcement Learning.* Nature.

5. Meta AI. (2024). *Llama 3.1: A Foundation Model for General Intelligence.* [llama.meta.com](https://llama.meta.com)

6. Qwen Team. (2025). *Qwen3 Technical Report.* [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
