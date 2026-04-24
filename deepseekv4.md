---
title: "DeepSeek-V4: a million-token context that agents can actually use"
thumbnail: /blog/assets/deepseekv4/thumbnail.png
authors:
- user: burtenshaw
---

# DeepSeek-V4: a million-token context that agents can actually use

DeepSeek released V4 today. Two MoE checkpoints are on the Hub: DeepSeek-V4-Pro at 1.6T total parameters with 49B active, and DeepSeek-V4-Flash at 284B total with 13B active. Both have a 1M-token context window. The benchmark numbers are competitive, but not SOTA. It doesn't matter. The real innovation is how DeepSeek v4 is designed for efficient large context length support, and hence as one of the best candidates for agentic tasks.

Focusing on long running agentic workloads. Running a frontier open model as an agent today breaks in predictable ways. The model stops. You reprompt. The trace blows past the context budget, or the KV cache fills the GPU, or tool-call round trips degrade halfway through a long task. **V4 is built to fix these known failures**, and point the way for the community to follow.

This post covers three things: what the architecture does differently to make long-context inference cheap, the agent-specific post-training decisions that compound on top of it, and some takeaways from the paper that help reason about these changes.  

## The KV cache problem for agents

A 1M context window is just capacity, not performance. Whether you can use it depends on the cost of every forward pass at that depth. For an agent running a long tool-use trajectory (a SWE-bench task, a multi-step browse session, a terminal session with hundreds of commands), every tool result is appended to the context, and every subsequent token pays the full attention cost against everything that came before.

Two numbers matter: single-token inference FLOPs and KV cache size. Both grow with sequence length. At 1M tokens, DeepSeek-V4-Pro requires 27% of single-token inference FLOPs compared with DeepSeek-V3.2, so it runs faster on the same hardware. It also uses 10% of the KV cache memory. V4-Flash drops these numbers even further: 10% of the FLOPs and 7% of the KV cache.

If we compare the KV cache memory against a established architecture like grouped query attention with 8 heads, stored in the usual bfloat16 format, DeepSeek v4 requires roughly 2% the cache size. This makes it much easier to deploy for very large context handling.

![Figure 1 from the DeepSeek-V4 technical report, benchmarks on the left, inference FLOPs and KV cache scaling on the right](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig1_efficiency.png)
*Figure 1: benchmark comparison (left), per-token FLOPs and accumulated KV cache against sequence length (right).*

## Hybrid attention: CSA and HCA

The efficiency gain comes from splitting attention into two mechanisms and interleaving them across layers.

**Compressed Sparse Attention (CSA)** compresses KV entries by 4x along the sequence dimension using softmax-gated pooling with a learned positional bias. A lightning indexer (FP4, ReLU-scored multi-head dot product) picks the top-k compressed blocks per query. It inherits the sparse-selection idea from DeepSeek Sparse Attention in V3.2, but runs it over blocks that are already 4x shorter than the original sequence. The indexer's search space shrinks with it.

![Figure 3: Compressed Sparse Attention, showing compressor, lightning indexer over compressed blocks, and sliding-window branch](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig3_csa.png)
*Figure 3: CSA. The compressor collapses every 4 tokens into one compressed KV entry. The lightning indexer picks the top-k compressed blocks per query. A sliding-window branch handles the most recent uncompressed tokens.*

**Heavily Compressed Attention (HCA)** compresses KV entries by 128x and drops the sparse selection. Every query attends densely to every compressed block. The compressed sequence is short enough that dense attention is cheap.

![Figure 4: Heavily Compressed Attention, 128x compression with dense MQA over compressed blocks](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig4_hca.png)
*Figure 4: HCA. A heavier compressor (128x vs. 4x) followed by dense attention over the compressed stream, with the same sliding-window branch for recency.*

The layers alternate between CSA and HCA. Different layers carry different attention patterns, and forcing one mechanism across all of them wastes capacity. In V4-Pro's 61-layer stack, layers 0–1 are HCA, layers 2–60 alternate CSA and HCA, and the MTP block at the end runs sliding-window only.

Both paths use FP8 storage for most KV entries and BF16 only for the RoPE dimensions. The lightning indexer inside CSA runs in FP4. These storage choices compound with the compression ratios to produce the 2% KV cache figure.

![Figure 2: overall architecture, showing embedding, hybrid CSA/HCA attention, DeepSeekMoE, manifold-constrained hyper-connections](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig2_architecture.png)
*Figure 2: overall architecture. Attention layers alternate between CSA and HCA. Feed-forward layers use DeepSeekMoE. Residual connections are replaced with manifold-constrained hyper-connections (mHC).*

## What changes for agents

Efficient long-context attention is necessary for agent workflows but not sufficient. The paper describes three post-training and infrastructure choices that target agent use cases directly.

### Interleaved thinking across tool calls

V3.2 kept reasoning traces across tool-result rounds but discarded them whenever a new user message arrived. For an agent handling a single user turn, this was fine. For multi-turn agentic workflows, where the user sends a follow-up after the agent has already chained several tool calls, the model lost its accumulated reasoning and had to reconstruct state.

V4 preserves reasoning content across user message boundaries when the conversation contains tool calls. The model retains the complete reasoning history across all rounds, including across user turns. This allows a coherent, cumulative chain of thought over long-horizon agent tasks. For conversational use without tools, the old behavior is preserved: reasoning is flushed on each turn to keep context concise.

![Figure 7: thinking management, with tools (top) preserves reasoning across turns; without tools (bottom) discards reasoning at each new user message](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig7_thinking.png)
*Figure 7: thinking with tools (top) preserves reasoning across all turns. Thinking without tools (bottom) discards reasoning at each new user message.*

### Tool-call schema with dedicated tokens

V4 introduces a `|DSML|` special token and an XML-based tool-call format. The XML format reduces escaping failures compared to JSON-in-string tool calls, a common failure mode when models emit nested quoted content.

The schema separates string parameters (passed as-is with `string="true"`) from structured parameters (passed as JSON with `string="false"`). This removes a class of parsing errors around numbers and booleans that JSON tool-call formats routinely hit.

### DSec: a sandbox built for RL rollouts

The agent behavior was trained with RL against real tool environments. The paper describes the sandbox infrastructure built for that purpose. DeepSeek Elastic Compute (DSec) is a Rust platform that exposes four execution substrates behind one Python SDK: function calls, containers, microVMs (Firecracker), and full VMs (QEMU). A single cluster runs hundreds of thousands of concurrent sandboxes.

Three DSec features matter for agent training: fast image loading via layered 3FS storage (so RL rollouts do not wait on container startup), preemption-safe trajectory replay (so interrupted training steps resume without re-running tool calls), and a uniform API across substrates (so training harnesses target function calls or full VMs without rewriting). These infrastructure decisions underpin the agent benchmark scores.

## Agent benchmark results

The knowledge and reasoning numbers are competitive but not leading. The agent numbers are where V4-Pro-Max separates from the field.

![DeepSeek-V4-Pro-Max benchmark comparison across frontier models](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/assets/dsv4_performance.png)

Specific numbers from the agent section of Table 6:

- Terminal Bench 2.0: V4-Pro-Max scores 67.9, ahead of GLM-5.1 (63.5) and K2.6 (66.7), behind GPT-5.4-xHigh (75.1) and Gemini-3.1-Pro (68.5).
- SWE Verified: 80.6 resolved, within a point of Opus-4.6-Max (80.8) and Gemini-3.1-Pro (80.6).
- MCPAtlas Public: 73.6, second only to Opus-4.6-Max (73.8).
- Toolathlon: 51.8, ahead of K2.6 (50.0), GLM-5.1 (40.7), and Gemini-3.1-Pro (48.8).

In the paper's internal R&D coding benchmark, 30 curated tasks across PyTorch, CUDA, Rust, and C++, V4-Pro-Max hits 67% pass rate, versus 47% for Sonnet 4.5 and 70% for Opus 4.5. In a survey of 85 DeepSeek developers using V4-Pro as their daily driver, 52% said it was ready to replace their current primary coding model and 39% leaned toward yes.

The long-context retrieval numbers are in Figure 9. MRCR 8-needle accuracy stays above 0.82 through 256K tokens and holds at 0.59 at 1M.

![Figure 9: MRCR 8-needle retrieval performance across context lengths up to 1M tokens](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig9_mrcr.png)
*Figure 9: MRCR 8-needle retrieval. V4-Pro-Max stays above 0.82 through 256K and holds at 0.59 at 1M.*

## Using the models

Four checkpoints are on the Hub. The instruct models use FP4 for MoE expert weights and FP8 for everything else. The base models are FP8 throughout.

- [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (1.6T / 49B activated, instruct)
- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (284B / 13B activated, instruct)
- [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base) (1.6T / 49B activated, base)
- [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) (284B / 13B activated, base)

Both instruct models support three reasoning modes: Non-think (fast, no chain of thought), Think High (explicit reasoning in `<think>` blocks), and Think Max (maximum reasoning effort with a dedicated system prompt). Think Max requires a context window of at least 384K tokens. The recommended sampling parameters across all modes are `temperature=1.0, top_p=1.0`.


The V4-Pro numbers on SWE Verified, MCPAtlas, and the internal R&D benchmark put it at parity with frontier closed models on agent tasks. The open question is how the community's tool harnesses adapt to the `|DSML|` schema and whether the interleaved thinking gains transfer to out-of-domain agent frameworks.

Figures in this blog post are from the technical report at [DeepSeek\_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).