---
title: "Ecom-RLVE: Adaptive Verifiable Environments for E-Commerce Conversational Agents"
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
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
> **TL;DR** — We extend the RLVE framework ([Zeng et al., 2025](__https://arxiv.org/abs/2511.07317__)) from single-turn reasoning puzzles to **multi-turn, tool-augmented e-commerce conversations**. ShopRLVE-GYM provides 8 verifiable environments — product discovery, substitution, cart building, returns, order tracking, policy QA, bundle planning, and multi-intent journeys — each with procedural problem generation, a 12-axis difficulty curriculum, and algorithmically verifiable rewards. We train a Qwen 3 1.7B model with DAPO over 300 steps and present early results demonstrating that environment scaling and adaptive difficulty transfer to agentic, real-world task completion.
The project is evolving, started with [Pytorch OpenEnv Hackathon](https://cerebralvalley.ai/e/openenv-hackathon-sf).
---

## 1. Introduction

### 1.1 Why Reinforcement Learning for E-Commerce Agents?

Large language models can hold fluent conversations, yet deploying them as autonomous shopping assistants reveals a persistent gap: **fluency ≠ task completion**. A customer who asks *"find me a USB-C charger under \$25 that ships in two days"* needs an agent that (i) invokes the right catalog search, (ii) filters on three hard constraints, (iii) avoids hallucinating product IDs it never retrieved, and (iv) gracefully handles follow-ups when the top result goes out of stock. Supervised fine-tuning (SFT) can teach surface-level tool usage from demonstrations, but it cannot scale to the combinatorial space of constraint configurations, partial-information dialogues, and multi-step transactional workflows that real e-commerce demands.

Reinforcement learning with verifiable rewards (RLVR) offers a principled alternative: rather than imitating demonstrations, the agent optimises for *outcomes* — did the recommended products actually satisfy the constraints? Was the cart correct? Was the return initiated for the right order line? The challenge has been constructing reward functions that are both **verifiable** (no LLM-as-a-judge subjectivity) and **adaptive** (difficulty that grows with the policy's capability).

### 1.2 From RLVE-Gym to ShopRLVE-GYM

Zeng et al. (2025) introduced **RLVE** (Reinforcement Learning with Adaptive Verifiable Environments) and built RLVE-Gym, a suite of 400 environments spanning sorting, multiplication, Sudoku, Hamiltonian paths, integration, and other algorithmic-reasoning tasks.

However, RLVE-Gym's 400 environments are entirely single-turn, text-in/text-out reasoning puzzles. The paper's own future-work section explicitly calls for extending the framework to **non-verifiable and agentic domains**:

ShopRLVE-GYM takes a different but complementary direction: we stay within the **verifiable** regime (e-commerce outcomes *can* be checked algorithmically) while extending RLVE to **multi-turn, tool-augmented, agentic** settings. This fills a gap the original paper identified but did not address — environments where the agent must *act* (call tools, modify world state) rather than merely *reason* (produce a text answer).

### 1.3 Contributions

1. **Eight atomic verifiable environments** for e-commerce conversation, each with a procedural problem generator, an algorithmic verifier, and an integer difficulty d.

2. **A 12-dimensional difficulty vector** \\(\theta(d)\\) that maps a single integer d to environment-specific parameters controlling constraint count, information gaps, retrieval noise, dialogue length, and more — all grounded in concrete e-commerce phenomena.

3. **A compositional reward function** combining task reward \\(r_{\text{task}}\\), efficiency reward \\(r_{\text{eff}}\\), and hallucination penalty \\(r_{\text{hall}}\\) with hard-fail overrides for format/tool/safety violations.

4. **Persona-driven user simulation** using **Qwen3.5 (9.7B)** as the dialogue backbone, with **constraint-aligned Dirichlet-sampled** latent preference weights that boost dimensions corresponding to active constraints — ensuring the verifier's hidden utility is observability-consistent with the user's communicated preferences.

5. **LLM-verbalized constraint utterances** — instead of template-based slot filling (which only covered 4 attribute types), we use the LLM to generate natural initial messages covering all 17+ constraint attributes, with **strategic omission** controlled by the LLM rather than random coin flips.

6. **Price constraint reconciliation** — price is always included as formal constraint #0 with difficulty-controlled slack, eliminating the prior mismatch where the template said "under $X×1.2" but the verifier checked a separately sampled price bound.

7. **Synthetic variant generation for cart verification** — at episode initialization, we synthesize realistic product variants (e.g., USB-C vs Lightning connectors, 65W vs 100W chargers, matte vs glossy finishes) using a hybrid category-based + data-driven attribute selection strategy. The verifier checks `(product_id, variant_id)` composite keys, not just product IDs, forcing the agent to learn to call `catalog.get_variants` and select the correct option.

8. **Nested environment collections** \\(\mathcal{C}_1 \subset \mathcal{C}_2 \subset \mathcal{C}_4 \subset \mathcal{C}_8\\) to study environment scaling in the agentic regime.

9. **UserAct-based fair efficiency scoring** — the user simulator tags every response with a structured dialogue act (confirm, clarify, correct, elaborate, continue, ragequit, done, dissatisfied). Turns driven by user-initiated clarification or confirmation are discounted from the efficiency penalty, so the agent is not punished for politely answering questions it didn't cause.

10. **Early training results** with Qwen 3 1.7B + DAPO (300 steps), demonstrating the viability of the approach.

---

## 2. Problem Formulation

### 2.1 The Verifiable Environment Abstraction

Following RLVE, each atomic environment is a tuple:

$$E^{(i)} = \bigl(I^{(i)},\; \mathcal{P}^{(i)},\; R^{(i)}\bigr)$$

\\(\mathcal{P}_d^{(i)}\\) is the **problem generator** at difficulty \\(d\\), sampling problem parameters \\(p \sim \mathcal{P}_d^{(i)}\\).

- \\(I_p^{(i)}\\) is the **instantiated input** — the initial user message plus any structured goal metadata.

- \\(R_p^{(i)}(o) \in [-1, 1]\\) is the **verifier** that maps an episode's output trajectory \\(o\\) to a scalar reward.

The critical property is *algorithmic verifiability*: \\(R_p\\) is a deterministic program, not an LLM judge. For product discovery, this means checking constraint satisfaction ratios and nDCG against a ground-truth evaluation pool. For cart building, it means computing F1 over the required multiset of (product, variant, quantity) tuples.

### 2.2 Why E-Commerce Is Verifiable

A common objection is that real-world agentic tasks are inherently subjective. E-commerce is a fortunate exception because customer-service outcomes are structurally verifiable:


![verifiable_signals_dark](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/dA0i6ZB3JDG-rqQtLRCy0.png)

Every one of these can be evaluated by a program with access to the hidden ground-truth goal. No human annotation is needed. No LLM-as-a-judge is needed.

### 2.3 Multi-Turn Episode Structure

Unlike RLVE-Gym's single-turn format, each ShopRLVE episode is a **multi-turn dialogue** between the agent and a simulated user. At each turn \\(t\\):

1. The agent produces a structured JSON action containing an `assistant_message`, optional `tool_calls`, and an optional `answer` submission.

2. Tool calls are executed against the environment's world state (catalog, cart, order history, policy KB).

3. The user simulator generates the next user message based on the hidden goal and dialogue state.

4. The episode terminates when the agent submits `answer.done = true`, the turn limit \\(T_{\max}(d)\\) is reached, or the simulated user ragequits.

Rewards are **terminal-only**: \\(r_t = 0\\) for all non-terminal steps, and the full composite reward is computed at episode termination. This is consistent with GRPO/DAPO's episodic reward formulation.

---

## 3. Catalog, Products, and Retrieval

### 3.1 Product Representation

The environment operates over a catalog \\(\mathcal{P} = \{p_1, \ldots, p_N\}\\) where each product \\(p\\) has:


![product_fields_dark](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/q7RZlT5rJDZD142lHjUKp.png)


### 3.2 Embeddings and Retrieval

Product embeddings are computed as:

$$e_p = \text{normalize}\bigl(f_{\text{enc}}(\text{title}(p) \oplus \text{desc}(p))\bigr) \in \mathbb{R}^D$$

where \\(f_{\text{enc}}\\) is `thenlper/gte-small` (D=384) by default. We build a FAISS ANN index over the catalog for efficient nearest-neighbor retrieval at episode time.

The `catalog.search` tool exposed to the agent performs:

1. Embed the query: \\(e_q = \text{normalize}(f_{\text{enc}}(q))\\)

2. Retrieve top-\\(k\\) by inner product: \\(\text{score}_{\text{vec}}(q, p) = e_q^\top e_p\\)

3. Apply metadata filters from the allowlist

4. Inject **difficulty-controlled retrieval degradation** (see §4.2)

### 3.3 Constraint Satisfaction

A hard constraint is a predicate \\(C_j : \mathcal{P} \to \{0, 1\}\\). Given a set of constraints \\(\mathcal{C} = \{C_1, \ldots, C_m\}\\), the **constraint satisfaction ratio** of a product is:

$$s(p \mid \mathcal{C}) = \frac{1}{m} \sum_{j=1}^{m} C_j(p) \in [0, 1]$$

This is the foundation of our evaluation pool construction: we rank the catalog by \\(s(p \mid \mathcal{C})\\) and take the top \\(K_{\text{eval}} = 500\\) products to form \\(\mathcal{P}_{\text{eval}}\\). All ranking metrics (nDCG, IDCG) are computed over this pool, making verification fast and deterministic.

---

## 4. The 12-Axis Difficulty Curriculum

### 4.1 Motivation: Why Multi-Axis?

RLVE uses a single integer \\(d\\) to parameterise difficulty. For algorithmic tasks (sorting, multiplication), \\(d\\) maps naturally to a single structural parameter (array length, digit count). E-commerce conversations, however, are harder along *many independent dimensions* simultaneously: a query can be difficult because it has many constraints, because information is missing, because the retrieval is noisy, or because the order history is large. Collapsing these into one number would conflate orthogonal sources of difficulty.


![Screenshot 2026-03-08 at 11.27.11](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/SALZRvBC6TP1HG1ZxqWsh.png)

Our solution is a **difficulty vector** \\(\theta(d) \in \mathbb{R}^{12}\\) that maps integer \\(d\\) to 12 generator parameters via fixed functional forms. Each axis is grounded in a concrete e-commerce phenomenon:

### 4.2 Axis Definitions

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Axis</th>
      <th>Formula</th>
      <th>E-Commerce Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><code>m(d)</code> — constraint count</td>
      <td><code>2 + floor(d/2)</code></td>
      <td>More product requirements → harder to satisfy all simultaneously. A customer asking for "red, under $30, 4+ stars, ships in 2 days, brand X" is harder than "red, under $30".</td>
    </tr>
    <tr>
      <td>2</td>
      <td><code>k_rec(d)</code> — output size</td>
      <td><code>min(10, 3 + floor(d/3))</code></td>
      <td>Recommending 3 good products is easier than recommending 8 — the agent must explore the catalog more broadly and maintain diversity.</td>
    </tr>
    <tr>
      <td>3</td>
      <td><code>T_max(d)</code> — turn budget</td>
      <td><code>4 + floor(d/2)</code></td>
      <td>Shorter dialogues force efficient tool use and direct questioning; longer budgets allow (but also require) multi-step reasoning.</td>
    </tr>
    <tr>
      <td>4</td>
      <td><code>p_missing(d)</code> — slot omission</td>
      <td><code>0.8 * sigmoid((d - 3)/1.5)</code></td>
      <td>Real customers rarely state all requirements upfront. At higher difficulty, the user omits constraints, forcing the agent to ask clarifying questions — a skill absent in single-turn settings.</td>
    </tr>
    <tr>
      <td>5</td>
      <td><code>p_noise(d)</code> — input noise</td>
      <td><code>clip(0.02d, 0, 0.25)</code></td>
      <td>Simulates typos, slang, and ASR errors (e.g., "charger" → "charjer"). Tests the agent's robustness to noisy natural language.</td>
    </tr>
    <tr>
      <td>6</td>
      <td><code>p_switch(d)</code> — context switch</td>
      <td><code>0.6 * sigmoid((d - 5)/2)</code></td>
      <td>Mid-conversation topic changes (e.g., "actually, I also need a case for my phone") — common in real chat sessions and catastrophic for agents that lose track of state.</td>
    </tr>
    <tr>
      <td>7</td>
      <td><code>top_k(d)</code> — retrieval depth</td>
      <td><code>max(20, 200 - 10d)</code></td>
      <td>Fewer search results force the agent to craft precise queries. At <code>d = 0</code>, the agent sees 200 candidates; at <code>d = 18</code>, only 20.</td>
    </tr>
    <tr>
      <td>8</td>
      <td><code>epsilon_rank(d)</code> — retrieval noise</td>
      <td><code>min(0.4, 0.02d)</code></td>
      <td>With probability <code>epsilon_rank</code>, each result in the ranked list is replaced by a random distractor from the same category. This simulates a noisy retrieval system and forces the agent to verify results via <code>catalog.get_product</code>.</td>
    </tr>
    <tr>
      <td>9</td>
      <td><code>p_oos(d)</code> — out-of-stock rate</td>
      <td><code>min(0.5, 0.05d)</code></td>
      <td>Items become unavailable during the episode. The agent must check availability and pivot — a common real-world failure mode.</td>
    </tr>
    <tr>
      <td>10</td>
      <td><code>H_orders(d)</code> — history depth</td>
      <td><code>1 + floor(d/2)</code></td>
      <td>More orders in history make "which order?" disambiguation harder. At <code>d = 9</code>, the user has 6 past orders — the agent must resolve references like "the charger I bought last week."</td>
    </tr>
    <tr>
      <td>11</td>
      <td><code>B_branch(d)</code> — policy complexity</td>
      <td><code>1 + floor(d/3)</code></td>
      <td>Policy rules gain more conditional clauses. At <code>d = 0</code>, the return window is a single number; at <code>d = 9</code>, it depends on category × membership tier × purchase channel.</td>
    </tr>
    <tr>
      <td>12</td>
      <td><code>B_tool(d)</code> — tool budget</td>
      <td><code>1 + floor(d/2)</code></td>
      <td>More tool calls per step are allowed at higher difficulty, but the agent must learn which tools to invoke and in what order.</td>
    </tr>
  </tbody>
</table>


### 4.3 Sigmoid Scheduling for Smooth Transitions

<p>
  Several axes use a logistic sigmoid <code>sigmoid(x) = 1 / (1 + e^(-x))</code> to create smooth S-curve transitions.
  For instance, <code>p_missing(d) = 0.8 * sigmoid((d - 3) / 1.5)</code> means:
</p>

<ul>
  <li>At <code>d = 0</code>: <code>p_missing ≈ 0.05</code> (user rarely omits info)</li>
  <li>At <code>d = 3</code>: <code>p_missing = 0.40</code> (50% of max; transition zone)</li>
  <li>At <code>d = 6</code>: <code>p_missing ≈ 0.70</code> (user routinely omits constraints)</li>
  <li>At <code>d → ∞</code>: <code>p_missing → 0.80</code> (asymptotic cap)</li>
</ul>

<p>
  This is deliberate: abrupt jumps in difficulty would create unstable learning signals,
  while sigmoid scheduling introduces new challenges gradually within the adaptive window.
</p>
---

## 5. The Eight Environments

### 5.1 Environment Overview

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Environment</th>
      <th>Intent</th>
      <th>Key Reward</th>
      <th>IsCorrect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>E_PD</code></td>
      <td>Product Discovery</td>
      <td>Find products meeting constraints</td>
      <td>nDCG + constraint satisfaction</td>
      <td><code>r_task &gt;= 0.95</code></td>
    </tr>
    <tr>
      <td><code>E_SUB</code></td>
      <td>Substitution</td>
      <td>OOS item → find alternative</td>
      <td>Similarity-weighted nDCG</td>
      <td><code>r_task &gt;= 0.95</code></td>
    </tr>
    <tr>
      <td><code>E_CART</code></td>
      <td>Cart Building</td>
      <td>Add correct items/variants/qty</td>
      <td>Variant-aware F1 over (product, variant, qty)</td>
      <td><code>F1 = 1.0</code></td>
    </tr>
    <tr>
      <td><code>E_RETURN</code></td>
      <td>Return + Replacement</td>
      <td>Return order line, find replacement</td>
      <td>Selection + initiation + replacement</td>
      <td>All sub-rewards pass</td>
    </tr>
    <tr>
      <td><code>E_STATUS</code></td>
      <td>Order Tracking</td>
      <td>"Where is my order?"</td>
      <td>Order ID + status match</td>
      <td>Both exact match</td>
    </tr>
    <tr>
      <td><code>E_POLICY</code></td>
      <td>Policy QA</td>
      <td>Deterministic policy question</td>
      <td>Exact/ratio match</td>
      <td><code>r_task &gt;= 0.95</code></td>
    </tr>
    <tr>
      <td><code>E_BUNDLE</code></td>
      <td>Bundle Planning</td>
      <td>Shopping list for a project</td>
      <td>Category F1 − budget penalty</td>
      <td><code>F1 = 1</code> and within budget</td>
    </tr>
    <tr>
      <td><code>E_JOURNEY</code></td>
      <td>Multi-Intent Journey</td>
      <td>Chained sub-tasks</td>
      <td>Average of sub-task rewards</td>
      <td><code>All r_j &gt;= 0.95</code></td>
    </tr>
  </tbody>
</table>

### 5.2 Reward Composition

<p>Every environment shares a common reward composition layer. At episode termination:</p>

<p><strong>Hard-fail check:</strong> If <code>format_invalid</code> OR <code>tool_invalid</code> OR <code>safety_violation</code>:</p>

<pre><code>r = -1
</code></pre>

<p><strong>Otherwise:</strong></p>

<pre><code>r = clip(w_task * r_task + w_eff * r_eff + w_hall * r_hall, -1, 1)
</code></pre>

<p>
  with weights
  <code>w_task = 0.75</code>,
  <code>w_eff = 0.15</code>,
  <code>w_hall = 0.10</code>.
</p>

<p><strong>Efficiency reward with UserAct discounting.</strong> A naive turn-count penalty punishes the agent equally for every turn — including turns where the <em>user</em> asked a follow-up question or confirmed an action. For example, if the user says <em>"yes, that's the one"</em> (a confirmation), the agent shouldn't lose efficiency credit for responding. To address this, the user simulator tags every response with a structured <strong>UserAct</strong>:</p>

<table>
  <thead>
    <tr><th>UserAct</th><th>Meaning</th><th>Penalised?</th></tr>
  </thead>
  <tbody>
    <tr><td><code>confirm</code></td><td>User confirms the agent's action is correct</td><td>No</td></tr>
    <tr><td><code>clarify</code></td><td>User provides previously omitted info (agent asked)</td><td>No</td></tr>
    <tr><td><code>correct</code></td><td>User points out an agent mistake</td><td>Yes</td></tr>
    <tr><td><code>elaborate</code></td><td>User adds new requirements</td><td>Yes</td></tr>
    <tr><td><code>continue</code></td><td>Generic acknowledgement / continuation</td><td>Yes</td></tr>
    <tr><td><code>ragequit</code></td><td>User abandons the conversation</td><td>Yes</td></tr>
    <tr><td><code>done</code></td><td>User confirms task complete</td><td>—</td></tr>
    <tr><td><code>dissatisfied</code></td><td>User expresses low satisfaction</td><td>Yes</td></tr>
  </tbody>
</table>

<p>Only <code>confirm</code> and <code>clarify</code> are non-penalty acts — these represent turns the user initiated, not agent errors. The <strong>effective turn count</strong> is:</p>

<pre><code>T_eff = max(1, T - T_user_clarify)
r_eff = 1 - (2 * (T_eff - 1)) / (T_max - 1)
</code></pre>

<p>
  where <code>T_user_clarify</code> counts turns tagged as <code>confirm</code> or <code>clarify</code>.
  An agent that solves the task in 3 turns — one of which is answering the user's confirmation
  question — pays efficiency cost for only 2 effective turns, not 3.
  <code>T_eff = 1</code> yields <code>+1</code> (solved immediately) and
  <code>T_eff = T_max</code> yields <code>-1</code> (used the entire budget).
</p>

<p>
  <strong>Why not discount <code>continue</code>?</strong> The <code>continue</code> act is the dialogue
  manager's catch-all fallback — it fires when no other act applies, which often masks
  turns where the agent should have been more efficient. Discounting it would over-credit
  the agent and collapse the efficiency gradient.
</p>

<p><strong>Hallucination penalty</strong> checks whether recommended product IDs were actually retrieved during the episode:</p>

<pre><code>hall_rate = |{p in L : p not in Seen}| / max(|L|, 1)
r_hall = -clip(hall_rate, 0, 1)
</code></pre>

<p>
  where <code>Seen</code> is the set of product IDs returned by
  <code>catalog.search</code> or <code>catalog.get_product</code> during the episode.
  This is a crucial guardrail: the agent cannot invent product IDs.
</p>

### 5.3 Detailed Environment Definitions

#### E_PD — Product Discovery

<p>
  The core recommendation environment. The generator samples a target product <code>p*</code>
  and always includes <strong>price as constraint #0</strong> with a difficulty-controlled
  bound <code>price ≤ p*.price × (1 + δ)</code> where <code>δ ~ U(0, 0.5·exp(-d/5))</code>.
  The remaining <code>m(d) - 1</code> constraint attributes are sampled from the full
  17-attribute allowlist (excluding price), and predicates
  <code>{C1, ..., Cm}</code> are constructed such that <code>Cj(p*) = 1</code>.
  The user's initial message is <strong>LLM-verbalized</strong> via Qwen3.5,
  covering all constraint types naturally. Strategic omission is LLM-controlled,
  with explicit tracking of mentioned vs. omitted attributes.
</p>

<p><strong>Task reward:</strong></p>

<pre><code>r_task = clip(0.55 * r_rank + 0.35 * r_cons + 0.10 * r_oos, -1, 1)
</code></pre>

<p>where:</p>

<ul>
  <li><code>r_rank = 2 * nDCG(rel = u(p)) - 1</code> using persona utility as relevance</li>
  <li><code>r_cons = 2 * s_best^(alpha(d)) - 1</code> with shaping exponent <code>alpha(d) = 4 + floor(d/4)</code></li>
  <li><code>r_oos = -clip(oos_rate, 0, 1)</code></li>
</ul>

<p>
  The constraint shaping exponent <code>alpha(d)</code> is a key design choice:
  at low difficulty (<code>alpha = 4</code>), partial constraint satisfaction still yields modest reward;
  at high difficulty (<code>alpha = 7+</code>), the reward surface becomes sharply peaked near
  <code>s_best = 1</code>, requiring near-perfect constraint matching.
</p>

#### E_SUB — Substitution

The user's desired product \\(p_0\\) is out of stock. The agent must find alternatives that are both *similar* to \\(p_0\\) and satisfy compatibility constraints. The relevance function blends similarity and constraint satisfaction:

$$\text{rel}_i = \lambda_{\text{sim}}(d) \cdot \phi_{\text{sim}}(p_i; p_0) + (1 - \lambda_{\text{sim}}(d)) \cdot s(p_i \mid \mathcal{C})$$

where \\(\lambda_{\text{sim}}(d) = \text{clip}(0.4 + 0.05d,\; 0.4,\; 0.8)\\) increases the importance of similarity at higher difficulty. This models the real-world pattern: at low difficulty, a broadly compatible substitute is fine; at high difficulty, the user insists on something very close to the original.

#### E_CART — Cart Building

<p>
  A transactional environment where the agent must add the correct products
  with the correct <strong>variants</strong> and <strong>quantities</strong> to a shopping cart.
  Unlike most recommendation tasks, cart building has a binary ground truth:
  either the cart exactly matches the requirement or it doesn't.
</p>

<p><strong>CART-specific difficulty axes.</strong> In addition to the shared 12-axis
parameters, CART introduces three environment-specific difficulty controls:</p>

<table>
  <thead>
    <tr>
      <th>Axis</th>
      <th>Formula</th>
      <th>d=0</th>
      <th>d=3</th>
      <th>d=6</th>
      <th>d=9</th>
      <th>Effect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>n_items(d)</code></td>
      <td><code>min(5, 1 + floor(d/3))</code></td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Number of distinct products to add</td>
    </tr>
    <tr>
      <td><code>p_var(d)</code></td>
      <td><code>sigmoid((d-2)/1.5)</code></td>
      <td>0.21</td>
      <td>0.66</td>
      <td>0.93</td>
      <td>0.99</td>
      <td>Probability each item requires a specific variant</td>
    </tr>
    <tr>
      <td><code>p_qty(d)</code></td>
      <td><code>min(0.5, 0.1·d)</code></td>
      <td>0.0</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>Probability of multi-quantity (drawn from U{2,4})</td>
    </tr>
  </tbody>
</table>

<p>At <code>d = 0</code>, the agent adds a single product with no variant or quantity
complexity — essentially learning the basic <code>catalog.search → cart.add</code> workflow.
At <code>d = 6</code>, it juggles 3 items, nearly all requiring specific variants,
with half needing quantities > 1.</p>

<p><strong>Synthetic variant generation.</strong> A key challenge in verifiable
cart building is that real product catalogs have sparse and inconsistent
variant data — many products have no variants at all, and those that do
typically vary only by color or size. To create a richer and verifiable
variant discrimination task, we <strong>synthesize variants at episode
initialization</strong> using a hybrid category-based + data-driven approach:</p>

<ol>
  <li><strong>Category mapping:</strong> A per-category priority list determines
  which attribute is most natural to vary. Electronics products vary by
  <code>connector_type</code> or <code>wattage</code>; clothing by
  <code>color</code>, <code>size</code>, or <code>material</code>;
  beauty products by <code>item_form</code> or <code>skin_type</code>;
  kitchen items by <code>material</code> or <code>color</code>.</li>
  <li><strong>Data-driven fallback:</strong> If the product's category has no
  mapping, we scan the product's actual attributes against the
  17-attribute allowlist and pick the first match.</li>
  <li><strong>Variant synthesis:</strong> For each target product, we generate
  3 <code>Variant</code> objects: 1 target (preserving the product's real
  attribute value when available) + 2 distractors (plausible alternatives
  from a predefined value pool). For example, an "Anker 65W USB-C Charger"
  produces variants <code>{USB-C, Lightning, HDMI}</code>, where USB-C is
  the target.</li>
  <li><strong>Injection:</strong> The synthetic variants are injected into the
  episode's <code>CatalogState</code> so that <code>catalog.get_variants(product_id)</code>
  returns them during the episode. The agent sees multiple options and must
  select the correct one.</li>
</ol>

<p>This approach covers all 17+ attribute types in the allowlist — not just
color and size — and creates a genuinely challenging discrimination task
that teaches the agent to read variant details carefully.</p>

<p><strong>Variant-aware verification.</strong> The verifier uses
<strong>composite keys</strong> to check cart correctness:</p>

<ul>
  <li>If a variant is required for product <code>p</code>: the key is
  <code>(product_id, variant_id)</code> — the agent must match both.</li>
  <li>If no variant is required: the key is just <code>product_id</code> —
  any variant (or none) is acceptable.</li>
</ul>

<p>The reward is computed as F1 over these composite-keyed unit quantities:</p>

$$F_1 = \frac{2 \cdot \text{prec} \cdot \text{rec}}{\text{prec} + \text{rec} + \epsilon}$$

<p>where precision and recall are measured over matched units.
Correct product but wrong variant → the unit is unmatched → precision and
recall both drop. <code>IsCorrect = 1[F1 = 1]</code> — a binary pass/fail
that requires <em>exact</em> cart correctness including variant selection.</p>

<p><strong>Mid-dialogue cart feedback.</strong> The user simulator provides
ground-truth feedback when the agent modifies the cart: if the wrong
variant is added, the simulated user says something like <em>"that's the
Lightning version, but I need USB-C"</em>. This creates a learning signal
for self-correction within the dialogue, rather than only at episode
termination.</p>

#### E_RETURN — Return + Replacement

A compound task: identify the correct order line \\((o^*, \ell^*)\\), initiate a return, and optionally find a replacement. The reward decomposes into:

$$r_{\text{task}} = \text{clip}\bigl(0.45 \cdot r_{\text{sel}} + 0.45 \cdot r_{\text{ret}} + 0.10 \cdot r_{\text{rep}},\; -1,\; 1\bigr)$$

Difficulty increases by growing the order history and placing purchase dates near the return-window boundary (\\(p_{\text{edge}}(d) = 0.7 \cdot \sigma((d-4)/1.5)\\)), forcing the agent to reason about eligibility edge cases.

#### E_STATUS — Order Tracking

The agent must resolve which order the user is asking about (potentially using indirect references like "the laptop I ordered last Tuesday") and report the correct status. At higher difficulty, the order history grows (\\(H_{\text{orders}}(d)\\)) and indirect references become more common (\\(p_{\text{ref}}(d) = \sigma((d-2)/1.5)\\)).

#### E_POLICY — Policy QA

Policy questions have deterministic answers governed by a rule engine. For numeric answers, the reward uses a smooth ratio score with aggressive shaping:

$$\rho = \left(\frac{\min(|x|, |y|)}{\max(|x|, |y|)}\right)^{\!\beta}, \quad \beta = 4, \qquad r_{\text{task}} = 2\rho - 1$$

At higher difficulty, the policy rules gain more conditional branches (\\(B_{\text{branch}}(d) = 1 + \lfloor d/3 \rfloor\\)), requiring the agent to navigate a deeper decision tree via the `policy.search` tool.

#### E_BUNDLE — Bundle Planning

Given a project goal (e.g., "I'm setting up a home office"), the agent must recommend products covering all required categories within an optional budget constraint. The reward combines category F1 with a budget penalty:

$$r_{\text{task}} = \text{clip}\bigl(r_{\text{base}} - \text{pen},\; -1,\; 1\bigr)$$

where \\(r_{\text{base}} = 2 \cdot F_1 - 1\\) and \\(\text{pen} = \max(0,\; (\text{cost} - B)/B)\\). The penalty is *not* pre-clipped — budget overruns beyond 100% are harshly penalised, creating a strong gradient against overspending.

#### E_JOURNEY — Multi-Intent Journey

The most complex environment: the user chains \\(L_{\text{int}}(d) = 2 + \lfloor d/4 \rfloor\\) sub-tasks (capped at 5) in a single conversation, with context switches governed by \\(p_{\text{switch}}(d)\\). Each sub-task is scored by the corresponding atomic verifier, and the composite reward is:

$$r_{\text{task}} = \text{clip}\!\left(\frac{1}{L}\sum_{j=1}^{L} r_j,\; -1,\; 1\right)$$

\\(\text{IsCorrect} = \mathbf{1}[\forall j,\; r_j \geq 0.95]\\) — the agent must near-perfectly complete *every* sub-task.

---

## 6. Persona-Driven User Simulation

### 6.1 Why Personas Matter

A verifiable environment needs a user simulator that behaves consistently but diversely. We decompose user behaviour into two components:

1. **Constraint-aligned preference weights** \\(w \in \Delta^4\\) (a 5-simplex), sampled per-episode from a Dirichlet distribution whose concentration parameters are **boosted for dimensions corresponding to active constraints**. For example, if the sampled constraints include `price ≤ $50` and `brand = Nike`, the Dirichlet alpha for the Price and Brand dimensions is multiplied by a boost factor (\\(\beta = 3.0\\)), making it overwhelmingly likely that the persona genuinely cares about the dimensions it was asked to constrain.

2. **Dialogue policy** driven by **Qwen3.5 (9.7B)**, conditioned on the persona, hidden goal, and conversation history. Template-based responses are retained as a deterministic fallback when the LLM is unavailable.

### 6.2 Persona Weight Dimensions

The five persona dimensions capture the primary axes along which e-commerce preferences vary:

<table>
  <thead>
    <tr>
      <th>k</th>
      <th>Dimension</th>
      <th>&phi;<sub>k</sub>(p) &isin; [0,1]</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>Price</strong></td>
      <td>
        <code>1 - clip((price(p) - P_low) / (P_high - P_low), 0, 1)</code>
      </td>
      <td>
        Lower price → higher score. <code>P_low</code>, <code>P_high</code> are
        min/max prices in the evaluation pool.
      </td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>Rating</strong></td>
      <td>
        <code>clip((rating(p) - 1) / 4, 0, 1)</code>
      </td>
      <td>
        Higher rating → higher score. Linearly mapped from <code>[1, 5]</code> to
        <code>[0, 1]</code>.
      </td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>Shipping</strong></td>
      <td>
        <code>1 - clip(ship_days(p) / S_max, 0, 1)</code>
      </td>
      <td>
        Faster shipping → higher score. <code>S_max = 14</code> days.
      </td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>Brand</strong></td>
      <td>
        <code>1[brand(p) = brand_pref]</code>
      </td>
      <td>Binary brand loyalty.</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>Similarity</strong></td>
      <td>
        <code>(e_p^T e_p0 + 1) / 2</code>
      </td>
      <td>
        Cosine similarity to a reference product, mapped to <code>[0,1]</code>.
        Used in substitution scenarios.
      </td>
    </tr>
  </tbody>
</table>

The hidden utility is:

$$u(p) = \sum_{k=1}^{5} w_k \cdot \phi_k(p) \in [0, 1]$$

**Sampling:** \\(w \sim \text{Dirichlet}(\alpha')\\) where \\(\alpha'_k = \alpha_k \cdot \beta^{\mathbb{1}[\text{dim}_k \in \text{constrained}]}\\) with base \\(\alpha = [2.0,\; 2.0,\; 1.0,\; 1.0,\; 1.0]\\) and boost \\(\beta = 3.0\\).

The mapping from constraint attributes to persona dimensions is:

- `price` → Price (dim 0)
- `rating`, `rating_count` → Rating (dim 1)
- `ship_days` → Shipping (dim 2)
- `brand`, `store` → Brand/Store (dim 3)

**The observability-consistency problem.** In a verifiable environment, the verifier evaluates the agent using the persona's hidden utility \\(u(p) = \sum_k w_k \cdot \phi_k(p)\\). If persona weights are sampled independently of constraints, a mismatch arises: the user says *"I need a charger under \$25"* (a price constraint), but a vanilla Dirichlet might assign \\(w_{\text{price}} \approx 0.05\\) — meaning the verifier's ground-truth ranking barely rewards price-appropriate products. The agent is effectively penalized for listening to the user. This is an **observability inconsistency**: the verifier's hidden reward surface contradicts the information the user communicated.

**Why Dirichlet (stochastic) rather than deterministic assignment.** A naive fix — setting \\(w_{\text{price}} = 1.0\\) whenever price is constrained — eliminates diversity: every price-constrained episode produces an identical persona, collapsing the training distribution. The Dirichlet boost preserves stochasticity: with a boosted \\(\alpha'_{\text{price}} = 2.0 \times 3.0 = 6.0\\) versus unboosted dimensions at \\(\alpha = 1.0\\), the expected weight share for price is \\(\mathbb{E}[w_{\text{price}}] = 6.0 / (6.0 + \sum \alpha'_{\text{other}}) \approx 0.5\\) — the persona *probably* cares about price, but how much varies episode to episode. Unconstrained dimensions still receive nonzero weight, modeling the realistic scenario where a user explicitly asks for a price cap but also has latent preferences for rating, shipping speed, or brand that they haven't stated. This creates a richer and more realistic training distribution while maintaining the invariant that communicated constraints are likely to matter in the verifier's scoring.

### 6.3 LLM-Based Dialogue Simulation

The dialogue simulator uses **Qwen3.5 (9.7B)** as the user backbone, served via any OpenAI-compatible inference endpoint (e.g., vLLM, TGI, or a cloud API). The model integrates deeply with the constraint pipeline while keeping inference latency low.

**LLM-verbalized constraint utterances (Fix 1).** Instead of the original template-based slot filling — which could only express 4 attribute types (brand, color, rating, ship_days) — the LLM generates natural initial messages covering **all 17+ constraint attributes** in the allowlist (material, connector type, wattage, screen size, finish type, etc.). The prompt includes the product category and full constraint list, and asks for a realistic shopper message. Templates serve as a deterministic fallback.

**Strategic omission (Fix 5).** Rather than random coin-flip omission (\\(p_{\text{missing}}\\)), the LLM is prompted to deliberately withhold a subset of constraints, returning both the message text and explicit `mentioned`/`omitted` attribute sets. These sets are tracked in the episode metadata so the verifier knows which constraints the agent was actually told about — resolving the prior issue where the verifier penalized agents for constraints they never saw.

**LLM dialogue responses (Fix 6).** Mid-dialogue user responses (satisfaction, dissatisfaction, continuation, done) are generated by the same LLM endpoint, conditioned on the dialogue context. This replaces the 5-element canned response lists that created only ~30 unique user strings across all episodes. Template fallback is preserved when the LLM endpoint is unavailable.

**Deterministic mode.** For reproducibility and debugging, the simulator supports a `deterministic_mode` flag that bypasses all LLM calls and uses template index 0 with no noise.

**Structured dialogue acts (UserAct).** Every user simulator response is tagged with a `UserAct` enum — one of 8 structured dialogue acts (`confirm`, `clarify`, `correct`, `elaborate`, `continue`, `ragequit`, `done`, `dissatisfied`). The act is determined by the dialogue manager's control flow: a response to a pending clarification slot is tagged `clarify`; cart feedback about a wrong variant is tagged `correct`; confirming that the right item was found is tagged `confirm`. These acts are recorded in the episode state (`user_act_history`) and consumed by the reward composer to compute fair efficiency scoring (see §5.2). This creates an information channel between the user simulator and the reward system that allows the verifier to distinguish agent-caused turns from user-caused turns — a distinction that is critical for learning efficient dialogue policies without discouraging helpful behaviour.

---

## 7. Adaptive Difficulty (RLVE-Style)

### 7.1 Per-Environment Adaptive State

<p>For each environment <code>E^(i)</code>, we maintain:</p>

<ul>
  <li><code>l_i</code>: lower bound difficulty (initially 0)</li>
  <li><code>h_i</code>: upper bound difficulty (initially 0)</li>
  <li><code>a_i</code>: correct rollout count at <code>d = h_i</code></li>
  <li><code>b_i</code>: total rollout count at <code>d = h_i</code></li>
</ul>

### 7.2 Sampling and Update

<p>When generating a new episode from environment <code>i</code>:</p>

<pre><code>d ~ Uniform[l_i, h_i]
</code></pre>

<p>After each completed rollout at difficulty <code>d = h_i</code>:</p>

<pre><code>b_i &larr; b_i + 1
if IsCorrect:
    a_i &larr; a_i + 1
</code></pre>

<p>When <code>b_i &gt;= tau_num = 32</code>:</p>

<pre><code>if a_i / b_i &gt;= tau_acc = 0.9:
    h_i &larr; h_i + 1
    if (h_i - l_i) &gt; d_delta:
        l_i &larr; h_i - d_delta
</code></pre>

<p>Then reset: <code>a_i &larr; 0</code>, <code>b_i &larr; 0</code>.</p>

<p>
  With <code>d_delta = 4</code>, the sliding window covers at most 5 difficulty levels
  (for example, <code>[3, 7]</code>), ensuring the agent is always exposed to problems
  near its capability frontier.
</p>

### 7.3 Why Adaptive > Static for E-Commerce

Static difficulty distributions are particularly problematic in e-commerce because the 12 axes interact non-linearly. A model that masters 2-constraint product discovery (\\(d=0\\)) needs to gradually encounter missing slots (\\(d \geq 3\\)), noisy inputs (\\(d \geq 5\\)), and retrieval degradation (\\(d \geq 10\\)) — not all at once. Adaptive scheduling lets each environment independently find the right frontier, even when environments advance at different rates (e.g., `E_POLICY` may be "easy" and advance quickly, while `E_JOURNEY` stays at low \\(d\\) for many steps).

---

## 8. Tool API and Action Schema

### 8.1 Unified Action Schema

All environments share a single strict JSON action schema:

```json

{

"assistant_message": "string",

"tool_calls": [

 {"name": "tool.name", "args": {"k": "v"}}

 ],

"answer": {

"env": "PD|SUB|CART|RETURN|STATUS|POLICY|BUNDLE|JOURNEY",

"recommended_product_ids": ["p1", "p2"],

"selected_order_id": "o123",

"done": true

 }

}

```

Invalid JSON → immediate termination with \\(r = -1\\). This creates a strong gradient toward well-formed outputs from the very first training step.

### 8.2 Tool Inventory

We expose 15 tools across 5 domains:

<table>
  <thead>
    <tr>
      <th>Domain</th>
      <th>Tools</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Catalog</strong></td>
      <td>
        <code>catalog.search</code>, <code>catalog.rerank</code>,
        <code>catalog.get_product</code>, <code>catalog.get_variants</code>
      </td>
      <td>Product retrieval and inspection</td>
    </tr>
    <tr>
      <td><strong>Cart</strong></td>
      <td>
        <code>cart.view</code>, <code>cart.add</code>,
        <code>cart.remove</code>, <code>cart.set_quantity</code>
      </td>
      <td>Cart state management</td>
    </tr>
    <tr>
      <td><strong>Orders</strong></td>
      <td>
        <code>order.list</code>, <code>order.get_status</code>,
        <code>order.checkout</code>
      </td>
      <td>Order history and checkout</td>
    </tr>
    <tr>
      <td><strong>Returns</strong></td>
      <td>
        <code>return.check_eligibility</code>, <code>return.initiate</code>,
        <code>return.exchange</code>
      </td>
      <td>Return and exchange workflows</td>
    </tr>
    <tr>
      <td><strong>Policy</strong></td>
      <td><code>policy.search</code></td>
      <td>Policy knowledge base lookup</td>
    </tr>
  </tbody>
</table>

Tool calls exceeding the per-step budget \\(B_{\text{tool}}(d)\\) return error results, triggering `tool_valid = false` and a hard-fail reward of \\(-1\\). This teaches the agent to plan its tool usage within budget.

---

## 9. Environment Collections and Scaling

Following RLVE's environment-scaling methodology, we define nested collections:

$$\mathcal{C}_1 = \{E_{\text{PD}}\} \subset \mathcal{C}_2 = \{E_{\text{PD}}, E_{\text{SUB}}\} \subset \mathcal{C}_4 \subset \mathcal{C}_8$$

where \\(\mathcal{C}_4 = \{E_{\text{PD}}, E_{\text{SUB}}, E_{\text{CART}}, E_{\text{RETURN}}\}\\) and \\(\mathcal{C}_8\\) includes all 8 environments.

The motivation mirrors RLVE's key finding: training on more environments develops more robust capabilities. For e-commerce, this means:

- \\(\mathcal{C}_1\\) trains pure retrieval + recommendation

- \\(\mathcal{C}_2\\) adds substitution reasoning (similarity + constraints)

- \\(\mathcal{C}_4\\) adds transactional skills (cart manipulation, return workflows)

- \\(\mathcal{C}_8\\) adds knowledge retrieval (policy QA), planning (bundles), and compositionality (journeys)

We hypothesise — consistent with RLVE's findings — that the multi-environment agent on \\(\mathcal{C}_8\\) will outperform single-environment specialists, even on the specialist's own task.

---

## 10. Training Setup

### 10.1 Base Model and Algorithm

- **Base model:** Qwen 3 1.7B

- **RL algorithm:** DAPO (Dynamic Sampling Policy Optimization) — a variant of GRPO that uses oversampling with dynamic filtering, discarding prompts where all rollouts receive identical rewards

- **Group size:** \\(G = 4\\) rollouts per prompt

- **Learning rate:** \\(1 \times 10^{-5}\\)

- **Training steps:** 300

- **Collection:** \\(\mathcal{C}_8\\) (all 8 environments; interactive demo also supports per-environment selection)


![accuracy_10_levels_dots_each_reach (1)](https://cdn-uploads.huggingface.co/production/uploads/6893dd21467f7d2f5f358a95/sGyMSKDOJ4tqiRSgV7AOR.png)

### 10.2 User Simulator

- **Model:** Qwen3.5 (9.7B), served via any OpenAI-compatible inference endpoint

- **Role:** Generates naturalistic user utterances, LLM-verbalized constraint messages, and contextual dialogue responses — conditioned on persona weights and hidden goals

- **Constraint verbalization:** All 17+ attribute types are expressible via LLM prompting; template fallback for deterministic mode

- **Strategic omission:** LLM-controlled (not random \\(p_{\text{missing}}\\) coin flips), with explicit mentioned/omitted tracking

- **Determinism:** Dialogue state transitions are deterministic; the LLM adds language variation only. Full template fallback when the LLM endpoint is unavailable

### 10.3 Reward Configuration

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>w_task</code></td>
      <td>0.75</td>
    </tr>
    <tr>
      <td><code>w_eff</code></td>
      <td>0.15</td>
    </tr>
    <tr>
      <td><code>w_hall</code></td>
      <td>0.10</td>
    </tr>
    <tr>
      <td><code>tau_acc</code></td>
      <td>0.9</td>
    </tr>
    <tr>
      <td><code>tau_num</code></td>
      <td>32</td>
    </tr>
    <tr>
      <td><code>d_delta</code></td>
      <td>4</td>
    </tr>
    <tr>
      <td><code>K_eval</code></td>
      <td>500</td>
    </tr>
  </tbody>
</table>

### 10.4 Infrastructure

- FAISS index over a 2M-product subset of the catalog

- Embedding model: `thenlper/gte-small` (384-dim)

- All episodes seeded deterministically for reproducibility

---

## 11. Experiments and Ablations

### 11.1 Qwen 3 1.7B + DAPO (300 Steps) — Early Results

We trained a Qwen 3 1.7B model using DAPO on the \\(\mathcal{C}_1\\) collection (product discovery only) for 300 training steps as an initial viability study.

We saw progressive growth

### 11.2 Planned Ablations

#### Adaptive vs. Static Difficulty

Following RLVE's ablation methodology, we will compare:

- **Adaptive:** \\(d \sim \text{Uniform}[\ell_i, h_i]\\) with RLVE-style updates

- **Static-Low:** \\(d \sim \text{Uniform}[0, 2]\\)

- **Static-High:** \\(d \sim \text{Uniform}[0, 10]\\)

We expect the static-low baseline to saturate early (the agent masters easy problems and stops improving), while static-high will suffer from low effective prompt ratios (most problems are too hard, yielding no learning signal).

#### Environment Scaling (\\(\mathcal{C}_1\\) → \\(\mathcal{C}_8\\))

We will train identical models on \\(\mathcal{C}_1\\), \\(\mathcal{C}_2\\), \\(\mathcal{C}_4\\), and \\(\mathcal{C}_8\\) and evaluate on:

1. **In-distribution:** Success rate on each environment in the training collection

2. **Transfer:** Performance on held-out environment configurations (unseen constraint combinations, unseen product categories)

3. **Compositionality:** Performance on \\(E_{\text{JOURNEY}}\\) for models trained without it (\\(\mathcal{C}_1\\) through \\(\mathcal{C}_4\\))

#### Reward Component Ablation

To validate the contribution of each reward component, we will train with:

- Task-only: \\(w_{\text{task}} = 1.0\\), \\(w_{\text{eff}} = w_{\text{hall}} = 0\\)

- No hallucination penalty: \\(w_{\text{hall}} = 0\\)

- No efficiency reward: \\(w_{\text{eff}} = 0\\)

We hypothesise that removing the hallucination penalty will lead to agents that fabricate product IDs, while removing the efficiency reward will produce agents that use excessive turns.

---

## 12. Design Decisions and Trade-offs

### 12.1 Terminal-Only vs. Dense Rewards

We use terminal-only rewards (\\(r_t = 0\\) for non-terminal steps). Dense per-step shaping (e.g., \\(r_t = P_t - P_{t-1}\\) based on progress) is designed but not implemented. The rationale:

1. **Simplicity:** Terminal rewards avoid the reward-hacking risks inherent in dense shaping (e.g., the agent could learn to trigger partial-progress signals without completing the task).

2. **Compatibility:** GRPO/DAPO naturally handle episodic rewards.

3. **Empirical signal:** If terminal rewards prove insufficient, dense shaping is a ready lever.

### 12.2 Constraint-Based Evaluation Pool

The original design called for embedding-based high-recall retrieval to construct the evaluation pool \\(\mathcal{P}_{\text{eval}}\\). We instead use constraint satisfaction scoring:

$$\mathcal{P}_{\text{eval}} = \text{top-}K_{\text{eval}}\bigl(\mathcal{P},\; \text{scored by } s(\cdot \mid \mathcal{C})\bigr)$$

This is faster (no embedding computation during verification), deterministic, and eliminates coupling between the evaluation pool and the retrieval model — a subtle source of information leakage.

## 13. Relationship to RLVE and Future Work

### 13.1 What We Inherit from RLVE

<table>
  <thead>
    <tr>
      <th>RLVE Component</th>
      <th>ShopRLVE Adoption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>E = (I, P, R)</code> abstraction</td>
      <td>Directly adopted; 8 environments</td>
    </tr>
    <tr>
      <td>Adaptive <code>[l, h]</code> window</td>
      <td>Identical algorithm, per-environment</td>
    </tr>
    <tr>
      <td><code>tau_acc = 0.6</code>, <code>d_delta = 4</code></td>
      <td>Same hyperparameters</td>
    </tr>
    <tr>
      <td>Nested collections <code>C_k</code></td>
      <td><code>C_1 ⊂ C_2 ⊂ C_4 ⊂ C_8</code></td>
    </tr>
    <tr>
      <td>DAPO as RL algorithm</td>
      <td>Adopted</td>
    </tr>
  </tbody>
</table>

### 13.2 What We Add

<table>
  <thead>
    <tr>
      <th>Extension</th>
      <th>Detail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Multi-turn episodes</strong></td>
      <td>RLVE-Gym is single-turn; we run multi-turn dialogues with tool use</td>
    </tr>
    <tr>
      <td><strong>Tool-augmented actions</strong></td>
      <td>15 tools across 5 domains; actions are structured JSON</td>
    </tr>
    <tr>
      <td><strong>World-state mutation</strong></td>
      <td>Cart and order state change during episodes</td>
    </tr>
    <tr>
      <td><strong>12-axis difficulty</strong></td>
      <td>Single integer maps to a rich parameter vector</td>
    </tr>
    <tr>
      <td><strong>Persona-driven evaluation</strong></td>
      <td>Hidden utility function using constraint-aligned Dirichlet-sampled weights</td>
    </tr>
    <tr>
      <td><strong>LLM user simulator</strong></td>
      <td><code>Qwen3.5 (9.7B)</code> as dialogue partner</td>
    </tr>
    <tr>
      <td><strong>Composite rewards</strong></td>
      <td>Task + efficiency + hallucination with hard-fail overrides</td>
    </tr>
    <tr>
      <td><strong>Synthetic variant generation</strong></td>
      <td>Category-aware variant synthesis at episode init; variant-aware F1 verification</td>
    </tr>
    <tr>
      <td><strong>UserAct fair efficiency</strong></td>
      <td>Simulator tags each response with a dialogue act; <code>confirm</code>/<code>clarify</code> turns are discounted from the efficiency penalty</td>
    </tr>
  </tbody>
</table>


---

## 14. Conclusion

ShopRLVE-GYM demonstrates that the RLVE framework — adaptive verifiable environments with procedural generation and algorithmic rewards — extends naturally from single-turn reasoning puzzles to multi-turn, tool-augmented e-commerce conversations. The key enablers are:

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
