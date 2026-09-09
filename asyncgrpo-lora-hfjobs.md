---
title: "Async GRPO with LoRA across Hugging Face Jobs: a bucket, a proxy, and no NCCL"
thumbnail: /blog/assets/asyncgrpo-lora-hfjobs/thumbnail.png
authors:
  - user: aminediroHF
  - user: qgallouedec
---

# Async GRPO with LoRA across Hugging Face Jobs: a bucket, a proxy, and no NCCL

LoRA support recently landed in TRL's `AsyncGRPOTrainer` with [PR #7017](https://github.com/huggingface/trl/pull/7017). The asynchronous trainer can now train an adapter instead of the full model, and it syncs only the LoRA adapter to vLLM. This post is about a real-world project built on top of it, once training and inference no longer share a machine.

LoRA training is particularly suited for RL, as stated in the amazing Thinking Machines post [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). They show that LoRA can match full fine-tuning for policy-gradient RL, even with rank 1. This stems from the fact that the advantage function only gives `~O(1)` bits of information per episode, so there is not that much to learn from each step, from a total-bits-of-information point of view. A rank-1 adapter has enough capacity to absorb it.

There is also a nice systems consequence of LoRA training. A rank-1 adapter for a 1.5B model is a few megabytes, while the full model is around 3 GB. Instead of sending the full policy to the inference workers after every update, we can just send the adapter. vLLM can also keep several adapters loaded at once. Old rollouts finish with the policy they started with, while new rollouts use the latest one.

TRL's `AsyncGRPOTrainer` already separates training and generation. The trainer and vLLM can run on different machines and at their own speed. This is easy in a single-node or cluster setting where both processes share a filesystem or can form an NCCL group. What we want is to run the same setup with [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). Essentially, an HF Job is one container running on one VM. This means that one Job cannot spawn multiple nodes (at least for now) to hold a trainer and a fleet of vLLM servers (we are limited to 8xH200 at most per node). The `AsyncGRPOTrainer` is built for exactly that kind of scale, so the question became: how far can we get if we drop the requirement that the trainer and the inference servers share a node? Well, with a full-weight sync, the answer would be "not far". Every update would have to move gigabytes between machines, which is what NCCL is for in a dense cluster, but Jobs can't communicate _across_ nodes. There is no shared local disk and obviously no shared `localhost`. With LoRA, a sync is only a few megabytes. For the filesystem part, HF Jobs provide volumes backed by [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)! These buckets can then be mounted as a FUSE filesystem in every Job and are enough to work as a shared FS between nodes. No network path between the Jobs is needed at all.

The setup ended up being quite small:

- a **trainer Job** running `AsyncGRPOTrainer` with LoRA (and FSDP, more on that later),
- **two vLLM Jobs**, each serving the base model plus whatever adapter the trainer last published,
- a **Storage Bucket** mounted in all three at the same path, which is how the adapter gets from the trainer to the servers,
- a **proxy server**. We'll dive deeper into why we need one, but at a high level we need a proxy that routes each rollout to the replica most likely to hold its KV cache, and broadcasts every adapter update to all vLLM replicas.

## The architecture: leveraging Hugging Face Jobs and Storage Buckets 🪣

[TRL PR #7017](https://github.com/huggingface/trl/pull/7017) adds an adapter-only sync path to `AsyncGRPOTrainer`. The trainer does not send tensors to vLLM. Every few optimizer steps, it saves the adapter under `<output_dir>/.vllm_lora/trl-policy-v{N}`, publishes the directory with an atomic rename, then sends its path to vLLM's `/v1/load_lora_adapter` endpoint. vLLM loads the files from disk, so the rollout worker can then request `model="trl-policy-v{N}"`.

This is how runtime adapter loading already works in vLLM. The endpoint takes a path, not tensors, so the trainer and the server are expected to share a filesystem. On a Slurm cluster, that is the network filesystem. On Jobs, we get the same thing by mounting a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) as a volume at the same path in every Job, like we mentioned earlier. Under the hood, it uses [`hf-mount`](https://github.com/huggingface/hf-mount), which exposes the bucket as a POSIX filesystem inside the container:

```sh
# every Job gets the same bucket at the same absolute path
hf jobs run ... -v hf://buckets/aminediroHF/asyncgrpo-lora-buckets:/lora ...
```

Nothing in TRL or vLLM had to change for this. The trainer writes to `/lora/<run>/.vllm_lora/` and the servers read from the same path. The path sent in the POST request is already valid inside every container.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/architecture.png" alt="Async GRPO with LoRA across Hugging Face Jobs. The trainer Job runs AsyncGRPOTrainer and the proxy, two vLLM Jobs serve the base model plus the latest adapter, and a Storage Bucket is mounted at /lora in all three.">
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">The three Jobs and the bucket. TRL talks to the proxy over localhost, the proxy talks to the replicas over HTTPS, and the adapter directory travels through the bucket mount.</figcaption>
</figure>

Note that we also store the checkpoints and the final adapter in the bucket. The HF Jobs are ephemeral, but a preempted trainer can resume training, as the final adapter is always persisted to the bucket and is never lost when the Job stops.

## The three Jobs

### The vLLM replicas

Each replica uses one GPU and the stock `vllm/vllm-openai` image. We only need to enable [runtime LoRA](https://docs.vllm.ai/en/latest/features/lora/?h=lora#serving-lora-adapters) loading and reserve enough adapter slots.

The number of adapter slots follows from `max_staleness`. In `AsyncGRPOTrainer`, every weight sync bumps the policy version by one, and `max_staleness` is how many versions a rollout sample may lag behind the current policy before the trainer discards it. With `max_staleness=4`, a sample generated under `trl-policy-v3` is still used for training while the trainer is at `v7`. A rollout that started under `v3` must also be able to finish under `v3`. So at any moment, vLLM has to serve the current policy plus the four before it. That is why the trainer keeps `max_staleness + 1` adapter versions registered and unloads anything older. Each sync loads the new version before it unloads the oldest one, which needs one more slot during the swap. That gives `--max-loras 6`. With only five, vLLM would silently evict a policy that still has rollouts in flight at every sync.

```sh
for replica in 1 2; do
hf jobs run --detach --flavor h200 --timeout 8h --secrets HF_TOKEN \
    --expose 8000 \                                          # reachable at https://<job_id>--8000.hf.jobs
    -v "hf://buckets/${BUCKET}:/lora:ro" \                   # read-only: the server only reads adapters
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \                  # enables /v1/load_lora_adapter
    -e VLLM_SERVER_DEV_MODE=1 \                              # enables /pause, /resume, /server_info (TRL needs all three)
    -- vllm/vllm-openai:v0.27.1 \
    vllm serve Qwen/Qwen2.5-Math-1.5B --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 --logprobs-mode processed_logprobs --generation-config vllm \
        --enable-lora --max-lora-rank 1 --max-loras 6      # max_staleness=4 -> 4+2 adapter slots
done
```

There is another possible design where the trainer keeps only the latest adapter and always publishes it under the same name. We did not go that way, because vLLM keys its prefix cache by adapter name. With a single name, KV blocks computed under the previous weights would still match after the swap, so the prefill would not be redone and a rollout could get its prefix from one policy version and its decode from the next. The trainer would have no way to tell, and it would show up as `ratio` drifting away from 1. Versioned names make this impossible: a name always means one set of weights, and a cached prefix can never match a newer version.

### The dataset choice: the sanity set

We chose [`sail/Sanity-Test-R1D-1.5B`](https://huggingface.co/datasets/sail/Sanity-Test-R1D-1.5B), the dataset from [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/pdf/2510.26788) (Qi et al., 2025). The reproduction code is in [`sail-sg/Precision-RL`](https://github.com/sail-sg/Precision-RL).

The authors generated 40 answers for each MATH problem with DeepSeek-R1-Distill-Qwen-1.5B. They kept the problems with a success rate between 20 % and 80 %, which gives 1,460 questions. This dataset is really good for RL validation as the questions are neither already solved nor completely hopeless for that model, which means a model can get a good signal early on to train on and improve.

This is awesome as a robust end-to-end test: if one vLLM replica silently serves the base model under an adapter name, we want to see that in the curve within a few dozen steps. Also, this dataset is small enough to cycle through in less than two hours.

We also take the hyperparameters from the paper's LoRA scripts in [`oat/scripts/lora`](https://github.com/sail-sg/Precision-RL/tree/main/oat/scripts/lora): `Qwen/Qwen2.5-Math-1.5B`, LoRA rank 1 with alpha 2, a learning rate of 4e-5, 8 samples per prompt, 128 completions per step, a maximum of 3,000 generated tokens and a 4,096-token context.

### The trainer

The trainer uses the same image with TRL installed from the PR branch (now from main). The training script is a normal `AsyncGRPOTrainer` script. The only Job-specific values are the output directory and the server URL.

```python
config = AsyncGRPOConfig(
    output_dir="/lora/sanity-lora-r1",       # on the bucket: adapters, checkpoints and the final adapter all land here
    vllm_server_base_url="http://localhost:8000",   # the proxy, not a vLLM Job; TRL never sees the Jobs URLs
    max_staleness=4,
    weight_sync_steps=4,                     # publish an adapter every 4 optimizer steps
    save_strategy="steps", save_steps=50,    # checkpoints go to the same bucket -> resume after preemption
    ...
)
trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen2.5-Math-1.5B",
    args=config,
    peft_config=LoraConfig(r=1, lora_alpha=2, target_modules="all-linear"),  # plain LoRA vLLM can serve as-is
    ...
)
```

> [!NOTE]
> During initialization, TRL calls `/server_info`. If it finds a `lora_config`, it uses adapter-only sync. Configurations vLLM cannot serve directly, such as DoRA, `modules_to_save`, or a rank above `--max-lora-rank`, fall back to merged-weight sync with a warning. The log should contain `Adapter-only vLLM sync enabled`.

## The proxy

Now onto the fun stuff. We need a proxy between the trainer and the vLLM Jobs for two reasons:

1. Exposed Job ports require an `Authorization: Bearer <HF token>` header on every request. The proxy is where that header gets added, so TRL does not need to know about it.

2. TRL refuses adapter-only sync when vLLM runs with `--data-parallel-size > 1`. This is a vLLM limitation rather than a TRL one. A call to `/v1/load_lora_adapter` only reaches the replica that answers it, so the other replicas would keep serving the base model under the new policy name.

We therefore run a small proxy at `127.0.0.1:8000` on the trainer Job and point TRL to it as if it were a single vLLM server. Besides adding the header, the proxy does two things:
- It sends each completion request to one replica, chosen so that the eight rollouts of a prompt land where their prefix is already cached (details on this below).
- It broadcasts every _state-changing_ request, such as **adapter loads, pause and resume**, to all replicas, so that a policy name means the same thing everywhere.

### Routing rollouts by KV prefix

A quick reminder of why this matters. Generating a completion has two phases. The prefill processes the whole prompt at once and computes the attention keys and values for every prompt token. The decode phase then produces one token at a time, and each new token attends to the keys and values of all the tokens before it. Those keys and values are the KV cache. Because attention is causal, the KV of a token depends only on the tokens before it, not on what comes after. Two requests that share a prefix therefore share the KV of that prefix, and a replica that already has it in cache can skip that part of the prefill entirely. The catch is that the cache lives on one replica. A request only benefits if it lands on the replica that has already seen its prefix.

vLLM stores its prefix KV cache in blocks of 16 tokens. For every problem, the rollout worker sends 8 requests with the same prompt. If they all reach the same replica, the first request computes the prefill and the next seven reuse it. With round-robin routing, half of them would go to a replica that does not have the prefix cached.

The router's job is to track which replica has seen each block hash. The hashes are chained, so the hash of block 3 represents blocks 1, 2 and 3, not just block 3. This mirrors causal attention: the KV of block 3 is only valid if blocks 1 and 2 are the same too. We also seed the chain with the adapter name. The KV cache depends also on the adapter that generated it, so a prefix cached for policy v3 is useless for policy v4.

<figure class="image text-center">
  <video controls autoplay loop muted playsinline style="max-width: 100%; margin: auto;">
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/kv-prefix-router.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">The routing decision for two prompts and four requests on two replicas: 16-token blocks, chained hashes, the common prefix, one affinity hit and one spill.</figcaption>
</figure>

The video plays through the whole decision. The steps below go through the same example in prose, with a real 135-token prompt from the sanity set.

**1. Split the prompt into blocks.** The router receives token ids and cuts them into 16-token blocks, just like vLLM. It only hashes complete blocks, so the last 7 tokens are ignored here.

**2. Hash the prefix.** Each block is hashed with the previous hash, starting from the adapter seed. `h3` therefore identifies blocks 1, 2 and 3 in order. Two prompts with the same first `k` blocks get the same hashes up to `hk`. Once one block changes, every hash after it changes too.

For example, the same prompt under `trl-policy-v4` starts from another seed and cannot match entries from `v3`. This is what we want because the old KV blocks were computed with different weights.

**3. Compare two prompts.** Problem 1 has 103 tokens. Both prompts start with the same 23-token chat template. Their first block is identical, but block 2 already contains the problem text. The hashes differ from there.

**4. Store the owners.** For every hash, the router remembers which replicas served it and which hashes came after it. We cap the successor set at two because we only need to know whether a block has one continuation or several. After a few prompts, the template block `h1` is owned by both replicas and already has several successors, `h2` to `h8` are owned by A only and each has a single successor, and `h2'` to `h6'` are owned by B only.

In practice, every prompt in a run starts with the same tokens. Here it is the chat template and the system prompt, which are the first 23 tokens of all 1,460 problems. In an agent setting it would be the tool descriptions, and in a multi-turn environment it would be the shared conversation history. These blocks are in every replica's cache within seconds, so matching on them tells us nothing about where a particular prompt lives.

A block is *common* if every replica has served it, or if it has more than one successor. Common blocks are ignored during routing because they do not identify a particular prompt.

**5. Pick a replica.** The router counts how many leading blocks match on each replica and removes the common prefix. What is left is the number of blocks specific to this prompt. Then:

- If one replica has specific blocks and it is not swamped, meaning it is at most 8 requests ahead of the least-loaded replica, the request goes there. We call this an **affinity** hit.
- If a replica has specific blocks but it is more than 8 requests ahead, we give up on the cache and send the request to the least-loaded replica. We call this a **spill**.
- If no replica has specific blocks, this is a new prompt. It goes to the least-loaded replica, round-robin on ties. We call this **unmatched**.

Here is the rule applied to four requests. Start from a state where replicas A and B both have 3 requests in flight, and only the template block `h1` is known, on both replicas.

- **Request 1, problem 0, rollout 1.** Both replicas match one block, the template, and that block is common. So nothing specific matches anywhere. The request is unmatched, both replicas are equally loaded, and round-robin sends it to A. The router records `h2` to `h8` as owned by A. A now has 4 requests in flight.
- **Request 2, problem 0, rollout 2.** Same prompt. A matches all 8 blocks, B matches only the template. After removing the one common block, A has 7 specific blocks and B has none. A is only 1 request ahead of B, well within the limit of 8, so the request goes to A. This is an affinity hit: A already has the whole prompt in its KV cache.
- **Request 3, problem 1, rollout 1.** A new prompt. Both replicas match only the template block, so nothing specific matches. The request is unmatched and goes to the least-loaded replica, B, which has 3 in flight against A's 5. The router records `h2'` to `h6'` as owned by B.
- **Request 4, problem 0, rollout 9.** Suppose that by now A has 12 requests in flight while B is back to 3. A still has the 7 specific blocks, but it is now 9 requests ahead of B, more than the limit. The request spills to B. B prefills problem 0 once, and the router records `h2` to `h8` as owned by B as well. Every replica has now served those blocks, so problem 0 becomes common too, and its later rollouts are placed by load alone.

**6. Reuse the prefill.** Request 2 is the reason for doing all this. It reuses the prefill computed by Request 1: blocks 1 to 8 are already in A's KV cache, so A skips straight to decoding the completion. Had it gone to B, B would have prefilled all 135 tokens again while A's cache sat unused. Request 3 shows why the `common` rule is needed. Without it, the shared chat template would make every new prompt look like a cache hit. Request 4 keeps the load bounded. Saving one prefill is not worth letting a replica fall far behind.

```python
def choose(self, upstreams, model, prompt):
    hashes = self.block_hashes(model, prompt)         # chained blake2b over 16-token blocks, seeded with `model`
    matched = self.matched_prefix(hashes)             # per replica: leading blocks it has served
    common = self.common_prefix_len(hashes)           # leading blocks that identify no prompt (see below)
    specific = [max(0, m - common) for m in matched]  # what actually distinguishes replicas
    least = min(u.inflight for u in upstreams)
    best = max(range(self.n), key=lambda i: (specific[i], -upstreams[i].inflight))

    if specific[best] > 0 and upstreams[best].inflight - least <= self.cfg.imbalance:
        pick = best                                   # affinity: the replica that has this prompt, and is not swamped
    else:
        candidates = [i for i in range(self.n) if upstreams[i].inflight == least]
        pick = candidates[self.rr % len(candidates)]  # spill or new prompt: least-loaded, round robin on ties
        self.rr += 1
    ...record `pick` as an owner of every block, and each block's successor...
    return upstreams[pick]
```

The `common` prefix was the annoying part. Every request starts with the same system prompt and chat template. A simple longest-prefix match would give the first replica a match for almost every new prompt. We detect the shared prefix through fan-out instead: a block with several different successors is common, while a block that always leads to the same successor belongs to a particular prompt. Only the blocks after that common prefix count as affinity. [`LORA_PROXY.md`](https://github.com/AmineDiro/hfjobs-lora-buckets/blob/main/LORA_PROXY.md) has the same walkthrough with the real token ids.

### Broadcasting the adapter

The proxy sends adapter loads to every replica. We treat the operation as all-or-nothing. Each replica has its own bucket mount, so they do not necessarily see a new adapter at exactly the same time. A `No adapter found for <path>` error usually means that one mount has not caught up yet, and we retry only that replica. For any other error, we unload the adapter from the replicas that accepted it. A policy name must never exist on only half of the replicas.

```python
async def load_one(u):
    while True:
        status, _, out = await send(u, "POST", "/v1/load_lora_adapter", headers, body)
        if status == 200 or "No adapter found" not in out.decode() or time.monotonic() > deadline:
            return u, status, out
        await asyncio.sleep(cfg.lora_retry_s)          # this replica's mount has not seen the directory yet

results = await asyncio.gather(*(load_one(u) for u in ups))
if any(st != 200 for _, st, _ in results):
    await asyncio.gather(*(send(u, "POST", "/v1/unload_lora_adapter", headers, unload) for u, st, _ in results if st == 200))
    return web.Response(status=504 if timed_out else st, text="rolled back on the others")
```

We broadcast `/pause`, `/resume` and `/v1/unload_lora_adapter` in the same way. `/health` returns 200 only if every replica is healthy. `/server_info` and `/v1/models` only need one answer. From TRL's point of view, the proxy is a single `data_parallel_size=1` server, so it selects adapter-only sync.

We initially wondered whether a Python asyncio proxy would become a bottleneck. It does not. There are at most 128 non-streaming JSON requests in flight, and routing only computes a few hashes. One process handles this easily.

## Full run results

The numbers below come from the trainer's [logged metrics](https://huggingface.co/docs/trl/en/async_grpo_trainer#logged-metrics) on trackio. The run uses `Qwen/Qwen2.5-Math-1.5B`, LoRA `r=1` on `all-linear`, 128 completions per step and 8 rollouts per prompt. It runs for 500 steps and saves a checkpoint every 50 steps. The trainer uses an `h200x2` Job and each of the two vLLM replicas uses one `h200` Job. Running all three costs around $20 per hour.

### Weight sync

| per sync, trainer's clock, 126 syncs | before | now (p50) |
|---|---|---|
| whole sync | 30.8 s | **8.5 s** (min 6.6, max 9.2) |
| of which: pause both replicas | 0.3 s | 0.3 s |
| adapter all-gather and save to the bucket | 0.6 s | 1.1 s |
| both replicas accept the adapter | ~29 s | ~7 s |

All 252 adapter loads succeeded: 126 syncs times 2 replicas. Six succeeded on the second attempt and 246 on the third. The remaining 7 seconds come from the 2.5-second upload and the proxy's 2-second retry interval, not from the mount anymore. Setting `PROXY_LORA_RETRY_S=0.5` should bring the sync closer to 4 seconds.

### Routing

At the end of the run, after 64 728 rollouts, the proxy's counters read:

```
routed [31928, 32800]  affinity 54712  spilled 820  unmatched 9196
```

With 8 rollouts per prompt, at least one request out of eight must be cold. The theoretical minimum is therefore 12.5 %. The router gets 14.2 % unmatched requests, 84.5 % affinity hits and 1.3 % spills. The traffic difference between the replicas is below 3 %. There is not much left to gain here unless we start looking at each replica's load using deeper inference-side metrics.

### Where the time goes

The first configuration has a pretty obvious problem: the trainer is the bottleneck, not generation. Over the 500 steps:

| per optimizer step, p50 | |
|---|---|
| step | 22.9 s |
| forward + backward | 21.9 s |
| waiting for rollouts | 0.02 s |
| rollout queue occupancy | 476 of 512 |
| trainer MFU | 3.9 % |

The rollout queue stays full and the worker is mostly blocked by backpressure. The second replica is useless in this configuration. Later in the post, we go through five runs that move the bottleneck between training and generation and make the full run 3.9× faster.

### Reward

![Figure 0](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig0-r1-dp2-reward.png)

*Figure 0. trackio run `r1-dp2`. Panels: `reward` with its 20-step rolling mean and 50-step block means, and `ratio` on a 0.99 to 1.01 axis. Reward climbs from 0.15 to 0.44 over 500 steps; `ratio` stays between 0.9993 and 1.0004 throughout.*

500 steps took 3 h 27 min. Mean reward per 50-step block:

```
steps    1-50   51-100  101-150 151-200 201-250 251-300 301-350 351-400 401-450 451-500
reward   0.151  0.208   0.290   0.350   0.375   0.404   0.402   0.427   0.425   0.445
```

Mean reward goes from 0.145 over the first 20 steps to 0.438 over the last 20. More importantly for this test, `ratio` stays at 1.000 for every step. The policy served by vLLM always matches the one used by the trainer to score the rollout. This held across all 126 syncs. Mean staleness was 1.5 policy versions, against a maximum of 4. The [trackio dashboard](https://huggingface.co/spaces/aminediroHF/async-grpo-lora-buckets) has the full curves.

## Bonus: chasing the bottleneck across the wire

Async RL is a pipeline between training and generation. Making one side faster does nothing if the other side cannot keep up. Fortunately, in `AsyncGRPOTrainer` we've added enough timings and metrics to see this directly.

All the useful metrics are documented in the [Logged metrics](https://huggingface.co/docs/trl/en/async_grpo_trainer#logged-metrics) section. `perf/rollout_wait_s` tells us how long the trainer waits for samples. `rollout/backpressure_s` tells us how long generation waits for space in the rollout queue. They are diametrically opposed to each other and should not both be high. Together with the queue size, they tell us which side is slow.

We ran five experiments. Each one starts from a problem visible in the previous run's dashboard. Unless mentioned otherwise, the model, recipe and three-Job layout stay the same. The names below are the trackio run names.

### Reading the dashboard

We keep these four groups of metrics visible:

- **`perf/step_s`** and **`perf/fwd_bwd_s`**: how long an optimizer step takes, and how much of it is forward+backward. If the second is nearly the first, the trainer is compute-bound.
- **`perf/rollout_wait_s`**: how long the trainer sat waiting for samples before it could start a step. Near zero means generation is ahead of training.
- **`sample/rollout_queue_size`** against `queue_maxsize`: the buffer between the two sides. Full means generation is being throttled; empty means the trainer is starving.
- **`rollout/backpressure_s`** and **`rollout/score_block_s`**: how long the rollout worker sat blocked because that buffer was full. Both are the same stall seen from the generation side, propagated backwards through the scoring stage.

The diagnosis is simple. A full queue with zero rollout wait and high backpressure means the trainer is too slow. An empty queue with rising rollout wait and no backpressure means generation is too slow. Comparing `perf/mfu_wall_clock` with `perf/mfu_fwd_bwd` also shows how much time the trainer GPUs spend waiting instead of training.

### Run 1, `r1-dp2`: a trainer that cannot keep up

![Figure 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig1-r1-dp2-trainer-bound.png)

*Figure 1. trackio run `r1-dp2`. Panels: `perf/step_s`, `perf/fwd_bwd_s`, `sample/rollout_queue_size`, `rollout/backpressure_s`. Step time and forward+backward overlap almost completely; the queue sits pinned near 476 of 512 and backpressure never drops below 11 s per rollout group: trainer-bound.*

`perf/step_s` is 22.9 s and `perf/fwd_bwd_s` is 21.9 s. Forward and backward take 96 % of the step. The queue stays around 476 out of 512, the trainer waits only 0.02 s for rollouts, and the rollout worker spends 15 seconds per group blocked by backpressure. The two vLLM replicas generate faster than the trainer consumes. The reported 4.6k tokens/s is not their actual limit; they simply have nowhere to put more output.

The batch metrics explain the terrible 3.9 % MFU. `batch/microbatches_per_step` is 64 and `batch/samples_per_row` is 1.0. Each rank processes one sequence of around 1.2k tokens, 64 times per step. This comes from the reference recipe's `per_device_train_batch_size=1`. For a 1.5B model on an H200, this is completely latency-bound.

### Run 2, `r1-dp2-tb16k`: pack the microbatch

The trainer also supports token-budget batching. With `token_budget > 0`, it packs several samples into one padding-free row per rank. An optimizer step processes `gradient_accumulation_steps` rows. We set `token_budget=16384` and `gradient_accumulation_steps=6`.

![Figure 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig2-r1-dp2-vs-tb16k-packing.png)

*Figure 2. trackio runs `r1-dp2` and `r1-dp2-tb16k` overlaid over their first 154 steps. Panels: `batch/samples_per_row`, `batch/microbatches_per_step`, `perf/step_s`, `perf/fwd_bwd_s`, `perf/mfu_fwd_bwd`, `rollout/generated_tok_s`. Packing takes samples per row from 1 to 13, microbatches from 64 to 6, step time from 23 s to 5.9 s, and generation from 4.2k to 27.5k tok/s with no change on the vLLM side.*

`batch/samples_per_row` goes from 1.0 to 12.7 and the number of microbatches drops from 64 to 6. The rows are 95 % full. Forward and backward fall from 21.9 s to 5.6 s, while MFU rises from 3.9 % to 19 %. We now train on around 150 samples per step because the rows pack better than the mean-length estimate predicted. Setting `gradient_accumulation_steps=5` would bring this closer to 128.

Generation also jumps from 4.6k to 25k tokens/s, even though we changed nothing on the vLLM side. The queue is no longer constantly full, so the replicas can finally run. This is why we do not like optimizing pipeline stages in isolation. The slowest stage hides the real performance of everything before it.

### Run 3, `r1-dp2-tb16k-nockpt`: stop recomputing the forward

`perf/fwd_s` is 1.34 s while `perf/fwd_bwd_s` is 5.6 s. A normal backward costs roughly twice the forward, and with frozen base weights it should be closer to once. A ratio of 3.2 is suspicious.

The reason is `gradient_checkpointing=True`, which is the default in `AsyncGRPOConfig` but not in `TrainingArguments`. Every microbatch recomputes its forward during the backward. This also explains why a 16k-token row only uses 25 GB on a 141 GB H200.

![Figure 3](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig3-tb16k-vs-nockpt-crossover.png)

*Figure 3. trackio runs `r1-dp2-tb16k` and `r1-dp2-tb16k-nockpt` overlaid over their first 134 steps. Panels: `perf/fwd_s`, `perf/fwd_bwd_s`, `perf/weight_sync_s`, `sample/rollout_queue_size`, `perf/rollout_wait_s`, `perf/mfu_fwd_bwd`. Forward+backward drops by one forward; the queue falls from ~420 to ~60 and rollout wait rises from 0.02 s to 0.5 s: the bottleneck crosses to generation.*

With `gradient_checkpointing=False`, forward and backward drop to 4.6 s, almost exactly one forward less, and MFU reaches 23 %. The queue now falls to 71 and rollout wait rises from 0.04 s to 0.6 s. The trainer consumes samples faster than two replicas generate them. We moved the bottleneck to generation.

This exposes two more costs. A 7.6-second weight sync every four steps now takes 25 % of wall-clock time. It was only 8 % when each step took 23 seconds. Also, backward is still 2.5 times slower than forward. With frozen base weights, there are around 2 seconds per step that do not look like normal model math.

### Run 4, `r1-dp3-tb16k-nockpt`: three replicas, and a surprise

Since generation was now too slow, we added a third replica. We also reduced the adapter retry interval from 2 s to 0.5 s and disabled `fsdp_reshard_after_forward` to check whether FSDP2 re-gathers caused the extra 2 seconds in backward.

![Figure 4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig4-dp2-vs-dp3-inflight-cap.png)

*Figure 4. trackio runs `r1-dp2-tb16k-nockpt` and `r1-dp3-tb16k-nockpt` overlaid over their first 134 steps. Panels: `perf/weight_sync_s`, `rollout/generated_tok_s`, `rollout/inflight`, `perf/fwd_bwd_s`. Sync falls from 7.6 s to 5.8 s; generation and forward+backward do not move; `rollout/inflight` reads 128 in both runs, which is the cap the third replica ran into.*

Weight sync falls from 7.6 s to 5.8 s, so the shorter retry helps. Forward and backward stay at 4.6 s, which rules out resharding. Generation moves from 25k to only 26k tokens/s. The third replica does basically nothing.

The reason was sitting in `rollout/inflight`: 128 in every run. The proxy shows those requests split as 44 + 43 + 41 across the three replicas. `max_inflight_tasks` limits concurrency for the whole rollout worker, not per replica. A 1.5B model on an H200 processes 43 and 130 concurrent sequences at almost the same cost per token. Splitting 128 requests over three GPUs gives nearly the same throughput as splitting them over two.

So vLLM was not the limit. Our own client-side constant was. We had set it conservatively because we did not know how hundreds of long HTTPS requests would behave through the public Jobs proxy. At this point, 130,000 rollouts had crossed it without a single transport error.

### Run 5, `r1-dp3-inflight384`: lift the cap

`max_inflight_tasks=384` and `queue_maxsize=768`, nothing else.

![Figure 5](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig5-all-runs-scoreboard.png)

*Figure 5. All five trackio runs (`r1-dp2`, `r1-dp2-tb16k`, `r1-dp2-tb16k-nockpt`, `r1-dp3-tb16k-nockpt`, `r1-dp3-inflight384`) overlaid, x-axis in steps. Panels: `perf/step_s`, `reward`, `sample/rollout_queue_size`, `sample/staleness_mean`. Step time falls from 22.9 s to 4.8 s across the series while the reward curves stay on top of each other; the last run's queue refills to ~690 of 768 and its staleness settles at 2. Runs 2 to 4 were stopped early once the dashboard had answered the question.*

With 384 requests in flight, each replica gets 128. The queue quickly fills to around 690 out of 768 and stays there. Backpressure returns to 5 seconds and rollout wait falls to 0.03 seconds. Training is the bottleneck again. Forward and backward take 4.6 seconds, weight sync adds an amortized 1.5 seconds, and median step time is 4.8 seconds.

There is a cost. Mean staleness rises from 1.5 to 2.0 versions because samples wait longer in the larger queue. This is still below `max_staleness=4`, and `ratio` remains exactly 1.000.

### The scoreboard

| 500 steps | run 1 `r1-dp2` | run 5 `r1-dp3-inflight384` |
|---|---|---|
| wall clock | 3 h 27 min | **53 min** |
| `perf/step_s`, p50 | 22.9 s | 4.8 s |
| `perf/fwd_bwd_s`, p50 | 21.9 s | 4.6 s |
| `perf/mfu_fwd_bwd` | 3.9 % | 23.5 % |
| `batch/samples_per_step` | 128 | 168 |
| samples trained | 64 000 | 84 078 |
| `perf/weight_sync_s`, p50 | 8.5 s | 6.2 s |
| `sample/staleness_mean` | 1.5 | 2.0 |
| reward, first 20 → last 20 steps | 0.145 → 0.438 | 0.145 → 0.416 |

![Figure 6](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/asyncgrpo-lora-hfjobs/fig6-reward-vs-wallclock.png)

*Figure 6. trackio runs `r1-dp2` and `r1-dp3-inflight384`, reward against wall-clock minutes since the first optimizer step. Same recipe, same 500 steps, same final reward; run 5 gets there in 52 minutes instead of 3 h 26 min.*

The final run is 3.9× faster and trains on 31 % more samples, with basically the same reward curve. Packing, disabling checkpointing and raising the in-flight limit made the difference. The shorter retry interval helped a little. Disabling resharding and adding a third replica without raising concurrency did nothing. In each case, the dashboard made this clear within the first ten minutes.

### What is still on the table

The step now costs 4.6 s of compute plus 1.5 s of amortized sync. We still want to investigate two things:

- **Weight sync is 25 % of wall clock.** The 6.2 s split into 0.2 s to pause the engines, 0.9 s to save the adapter and 5 s for the replicas to load it. The bucket upload has a floor of around 2.5 s. Syncing every eight steps instead of four would halve this cost but increase staleness. TRL also pauses the engines during the full sync. Since adapter names are versioned and the previous policy stays loaded, this pause may not be necessary on the adapter-only path.
- **Backward is still 2.5× slower than forward.** With frozen base weights, we expected closer to 1×. FSDP2 resharding is not the cause. Our current suspects are the chunked LM-head loss recomputing its projection during backward and the memory-bound LoRA operations on every linear layer. A short `torch.profiler` trace should answer this.

Also, this trainer does not need three replicas. Two replicas with 192 requests in flight each should run this recipe just as fast. A third one only becomes useful with a larger policy model, longer completions or multi-turn environments.

## Things we learned

- Mount the bucket at the same absolute path in every Job. The trainer sends a path and vLLM resolves it locally. Nothing checks that both paths point to the same place.
- `close()` returns before the upload reaches the bucket. We verified this for files from 1 KB to 128 MiB. A completed local write does not mean another Job can already read it.
- Be careful when polling a path before it exists. A negative lookup may be cached, which caused the entire 30-second delay here.
- Empty directories on the mount do not persist. In one test, `os.makedirs` followed by a file write 24 seconds later failed with `ENOENT`. Write a file immediately after creating the directory.
- Exposed Job ports require a bearer token. The Jobs proxy can keep a generation request open for at least four minutes, so 3,000-token completions work fine.
- Check `rollout/inflight` before adding replicas. `max_inflight_tasks` limits the entire pipeline. More replicas only split the same requests if you do not raise it.
- Check `gradient_checkpointing`. `AsyncGRPOConfig` enables it by default, unlike `TrainingArguments`. For a 1.5B model on a 141 GB GPU, it only wastes compute.
- Reusing replicas across runs causes adapter-name collisions because TRL starts again at `trl-policy-v1`. Unload adapters from the previous run first. The launcher does this automatically.
- Remember to stop the server Jobs. They do not terminate by themselves. `./run_all.sh --wait` cancels them when training finishes.

## Try it

```sh
git clone https://github.com/AmineDiro/hfjobs-lora-buckets && cd hfjobs-lora-buckets
hf auth login
MAX_STEPS=20 RUN_TAG=smoke ./run_all.sh --wait        # ~15 min, three Jobs, cancels the servers when done
MAX_STEPS=500 ./run_all.sh --wait                     # run 1: the reference batch shape, ~3.5 h
TOKEN_BUDGET=16384 GRAD_ACCUM=6 GRADIENT_CHECKPOINTING=0 PROXY_LORA_RETRY_S=0.5 \
  MAX_INFLIGHT=384 QUEUE_MAXSIZE=768 MAX_STEPS=500 ./run_all.sh --wait   # run 5: same recipe, ~55 min
```

`BUCKET_LATENCY_RESULTS.md` contains all the bucket measurements. `LORA_PROXY.md` walks through the routing decision with a real prompt. `tests/test_lora_proxy.py` runs the proxy against two fake vLLM servers, so you can modify the routing without paying for GPUs.

## References

- John Schulman et al., [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), Thinking Machines Lab, September 2025. The case that rank-1 LoRA matches full fine-tuning for policy-gradient RL, and why.
- TRL, [`AsyncGRPOTrainer`](https://huggingface.co/docs/trl/en/async_grpo_trainer) and its [logged metrics](https://huggingface.co/docs/trl/en/async_grpo_trainer#logged-metrics).
- TRL [PR #7017](https://github.com/huggingface/trl/pull/7017): PEFT/LoRA support for `AsyncGRPOTrainer` with adapter-only vLLM sync.
- Hugging Face [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) and [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets); [`hf-mount`](https://github.com/huggingface/hf-mount).
- [`hf-mount-repro`](https://github.com/AmineDiro/hf-mount-repro): the two-script reproduction of the 30-second negative-cache stall.
- The [trackio dashboard](https://huggingface.co/spaces/aminediroHF/async-grpo-lora-buckets) for every run in this post.
- Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin, [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/pdf/2510.26788), arXiv:2510.26788, 2025. Source of the sanity dataset [`sail/Sanity-Test-R1D-1.5B`](https://huggingface.co/datasets/sail/Sanity-Test-R1D-1.5B) and of the LoRA recipe, [`sail-sg/Precision-RL`](https://github.com/sail-sg/Precision-RL), `oat/scripts/lora/bf16_grpo_tis_lora.sh`.

```bibtex
@article{qi2025precisionrl,
  title={Defeating the Training-Inference Mismatch via FP16},
  author={Qi, Penghui and Liu, Zichen and Zhou, Xiangxin and Pang, Tianyu and Du, Chao and Lee, Wee Sun and Lin, Min},
  journal={arXiv preprint arXiv:2510.26788},
  year={2025}
}
```
