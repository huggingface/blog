---
title: "Three MoEs"
# thumbnail: /blog/assets/three-moes/thumbnail.png
authors:
- user: drbh
date: 2025-08-25
---

# Three MoEs

Three Ways to Compute Mixture of Experts (MoE) in PyTorch

Mixture of Experts (MoE) looks complex, but under the hood it’s just:

1. Route tokens to experts.
2. Apply MLPs (one per expert).
3. Recombine outputs with routing weights.

Below are **three ways** to compute MoE in PyTorch — from simple to complex.
Each one does the same math. The difference is *how* we schedule the compute.

---

## Step 1: Routing

Every token chooses its top-k experts with softmaxed scores.

```python
import torch, torch.nn.functional as F

def create_routing(logits, top_k):
    batch_size, seq_len, num_experts = logits.shape

    # pick top-k experts
    weights, indices = torch.topk(logits, top_k, dim=-1)
    weights = F.softmax(weights, dim=-1)

    # build dense [BS, E] routing weight matrix
    routing_weights = torch.zeros(batch_size * seq_len, num_experts, device=logits.device)
    flat_indices = indices.reshape(-1, top_k)
    flat_weights = weights.reshape(-1, top_k)
    batch_indices = torch.arange(batch_size * seq_len, device=logits.device).unsqueeze(1).expand(-1, top_k)
    routing_weights[batch_indices, flat_indices] = flat_weights

    return routing_weights, indices.reshape(-1, top_k)

```

Let’s test it:

```python
torch.manual_seed(0)
B, S, H, E, K = 1, 4, 8, 3, 2
logits = torch.randn(B, S, E)

routing_weights, router_indices = create_routing(logits, top_k=K)

print("Router indices:\n", router_indices.view(B, S, K))
print("Routing weights (avg per expert):\n", routing_weights.mean(0))
# Router indices:
#  tensor([[[0, 1],
#          [0, 1],
#          [1, 0],
#          [2, 0]]])
# Routing weights (avg per expert):
#  tensor([0.6131, 0.2264, 0.1606])

```

✅ This confirms that each token picks 2 experts and assigns them weights that sum to 1.

---

## Method 1: Dense / Repeat Experts

The "naïve" way: **send every token to every expert**, then weight results at the end.

```python
def repeat_experts(hidden_states, routing_weights,
                   gate_up_proj, gate_up_proj_bias,
                   down_proj, down_proj_bias):
    B, S, H = hidden_states.shape
    E = routing_weights.shape[1]
    I = gate_up_proj.shape[-1] // 2  # intermediate size

    # flatten and repeat tokens for each expert
    hs = hidden_states.reshape(-1, H).repeat(E, 1).view(E, -1, H)  # [E, BS, H]

    # expert feedforward
    gate_up = torch.bmm(hs, gate_up_proj) + gate_up_proj_bias[..., None, :]  # [E, BS, 2I]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    glu = gate * torch.sigmoid(gate * 1.72)
    act = (up + 1) * glu

    downed = torch.bmm(act, down_proj) + down_proj_bias[..., None, :]        # [E, BS, H]

    # apply routing weights
    flat_rw = routing_weights.view(-1, E).t().unsqueeze(-1)  # [E, BS, 1]
    downed = downed * flat_rw
    out = downed.sum(0).view(B, S, H)
    return out

```

Quick test:

```python
hs = torch.randn(B, S, H)
gate_up_proj = torch.randn(E, H, 2*H)
gate_up_proj_bias = torch.zeros(E, 2*H)
down_proj = torch.randn(E, H, H)
down_proj_bias = torch.zeros(E, H)

out1 = repeat_experts(hs, routing_weights, gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)
print("Dense output shape:", out1.shape, "| sum:", out1.sum().item())
# Dense output shape: torch.Size([1, 4, 8]) | sum: 78.0574951171875

```

✅ Works. Simple, but wastes compute (all tokens go to all experts).

---

## Method 2: Sparse Loop Experts

Now we only process tokens actually assigned to each expert.
This avoids redundant compute but uses a **Python loop**.

```python
def experts(hidden_states, router_indices, routing_weights,
            gate_up_proj, gate_up_proj_bias,
            down_proj, down_proj_bias):
    B, S, H = hidden_states.shape
    E = routing_weights.shape[1]

    hs = hidden_states.reshape(-1, H)
    out = torch.zeros_like(hs)
    flat_dense = routing_weights.view(-1, E)

    for e in range(E):
        # tokens routed to this expert
        mask = (router_indices == e).any(-1)
        token_idx = torch.nonzero(mask).squeeze(-1)
        if token_idx.numel() == 0: continue

        x = hs.index_select(0, token_idx)
        gate_up = x @ gate_up_proj[e] + gate_up_proj_bias[e]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        glu = gate * torch.sigmoid(gate * 1.72)
        act = (up + 1) * glu

        y = act @ down_proj[e] + down_proj_bias[e]
        scales = flat_dense.index_select(0, token_idx)[:, e]
        out.index_add_(0, token_idx, y * scales.unsqueeze(-1))

    return out.view(B, S, H)

```

Check against Method 1:

```python
out2 = experts(hs, router_indices, routing_weights,
               gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)

print("Loop output shape:", out2.shape, "| sum:", out2.sum().item())
print("Allclose to dense:", torch.allclose(out1, out2, atol=1e-5))
# Loop output shape: torch.Size([1, 4, 8]) | sum: 78.0574951171875
# Allclose to dense: True

```

✅ Same result, but only processes assigned tokens.  
❌ Loop slows down on GPU with thousands of tokens.

---

## Method 3: Binned Experts

The "binned" method:

* Group tokens per expert.
* Run each expert once with a contiguous batch.

This removes Python loops and plays nice with GPUs, however is more complex to implement and requires performant kernels to efficiently re-arrange data. 

Below are simple implementations of the gather and scatter operations with a focus on understanding, rather than performance.

```python
def sort_tokens_by_expert(router_indices, num_experts):
    flat_indices = router_indices.flatten()
    sorted_values, sorted_indices = torch.sort(flat_indices)
    tokens_per_expert = torch.bincount(sorted_values, minlength=num_experts)
    bins = torch.cumsum(tokens_per_expert, dim=0)
    return sorted_indices, sorted_values, bins, tokens_per_expert

```

Helper: gather tokens and scatter.

Gather goes from tokens to experts.  
Scatter goes from experts to tokens.  

```python
def binned_gather(x, indices, bins, expert_capacity, top_k):
    E, H = bins.shape[0], x.shape[1]
    out = torch.zeros((E, expert_capacity, H), device=x.device)
    for e in range(E):
        start = 0 if e == 0 else bins[e-1]
        end = bins[e]
        n = min(end - start, expert_capacity)
        for i in range(n):
            flat_pos = indices[start + i]
            tok = flat_pos // top_k
            out[e, i] = x[tok]
    return out

def binned_scatter(x, indices, weights, bins, expert_capacity, top_k):
    E, C, H = x.shape
    N = indices.shape[0] // top_k
    out = torch.zeros((N, top_k, H), dtype=x.dtype, device=x.device)
    for e in range(E):
        start = 0 if e == 0 else bins[e-1]
        end = bins[e]
        n = end - start
        if n == 0:
            continue
        take = min(n, expert_capacity)
        for i in range(take):
            flat_pos = indices[start + i]       # flattened (token, slot)
            tok = flat_pos // top_k
            slot = flat_pos % top_k
            scale = weights[flat_pos] if weights is not None else 1.0
            out[tok, slot] = x[e, i] * scale
    return out.sum(dim=1)   

```

Now the main method:

```python
def binned_experts(hidden_states, router_indices, routing_weights,
                   gate_up_proj, gate_up_proj_bias,
                   down_proj, down_proj_bias,
                   expert_capacity):
    B, S, H = hidden_states.shape
    E, K = routing_weights.shape[1], router_indices.shape[1]

    indices, _, bins, _ = sort_tokens_by_expert(router_indices, E)
    x = binned_gather(hidden_states.view(-1, H), indices, bins, expert_capacity, K)

    gate_up = torch.bmm(x, gate_up_proj) + gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    glu = gate * torch.sigmoid(gate * 1.72)
    x = (up + 1) * glu
    x = torch.bmm(x, down_proj) + down_proj_bias[..., None, :]

    # build routing weights aligned to (token, slot)
    flat_dense = routing_weights.view(-1, E)                 # [B*S, E]
    flat_router = router_indices.view(-1, K)                 # [B*S, K]
    selected = torch.gather(flat_dense, 1, flat_router).reshape(-1)  # [B*S*K]

    # scatter back
    y = binned_scatter(x, indices, selected, bins, expert_capacity, K)      # [B*S, H]

```

Check:

```python
out3 = binned_experts(hs, router_indices, routing_weights,
                      gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
                      expert_capacity=S)

print("Binned output shape:", out3.shape, "| sum:", out3.sum().item())
print("Allclose to dense:", torch.allclose(out1, out3, atol=1e-4))
# Binned output shape: torch.Size([4, 8]) | sum: 78.0574951171875
# Allclose to dense: True

```

✅ Matches the other two, but efficient batching can be scalable with the right implementation.

## Key Takeaways

We've seen three functionally identical ways to compute an MoE forward pass:

**Dense / Repeated Experts:** The most straightforward method. It's easy to understand and implement but is inefficient as it performs redundant computations for every token-expert pair. This makes it unsuitable where memory and compute resources are limited.

**Sparse Loop Experts:** This approach is more intelligent, processing only the tokens that are actually routed to each expert. It eliminates wasted computation but relies on a Python for loop, which is notoriously slow on parallel hardware like GPUs and creates a performance bottleneck.

**Binned Experts:** This is the most complex but also can be the most performant and scalable solution. By sorting and grouping tokens by their assigned expert, we can process them in large, contiguous batches. This "binned" or "token shuffling" approach is ideal for GPUs.

While our implementation focused on clarity, real-world libraries use highly optimized, low-level kernels to perform the gather and scatter operations with minimal overhead. This efficient data shuffling is the key to unlocking the power of MoE models at scale.

## Conclusion

Ultimately, choosing the right method depends on the goal. The three approaches shown; dense, looped, and binned—are simply different strategies for organizing the same computation, each with its own structural trade-offs.

A direct, repeated computation offers a clear and linear logic path, which can be useful for debugging. A looped structure provides fine-grained, sequential control over how each expert is processed. The binned method organizes tokens by their target expert before computation, a strategy that can be advantageous in certain contexts, such as during particular training regimes.

These are just a few of many possible implementations. The engineering behind Mixture of Experts is a flexible space, and the best way to schedule token-expert interactions is an open field for exploration and new ideas!