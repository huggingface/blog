---
title: "Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL"
thumbnail: /blog/assets/delta-weight-sync/thumbnail.png
authors:
  - user: aminediroHF
  - user: qgallouedec
  - user: kashif
  - user: lewtun
  - user: edbeeching
  - user: albertvillanova
  - user: lvwerra
---

# Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL

> **TL;DR**, because you have models to train and we respect that:
>
> - Async RL has a dirty secret: every step, the trainer has to ship the whole model to the inference engine. For a 7B in bf16 that is 14 GB. For a frontier 1T model checkpoint that is on the order of a terabyte. Per step.
> - It turns out you do not have to. Between two consecutive RL optimizer steps, **roughly 99% of bf16 weights are bit-identical** (and never less than 98% in the worst case). The actual delta is tiny.
> - We landed [a TRL PR](https://github.com/huggingface/trl/pull/5417) that encodes just the changed elements as a **sparse safetensors file**, uploads it to a **Hugging Face Bucket**, and tells vLLM to fetch it. On Qwen3-0.6B, the per-step payload drops from 1.2 GB to **20 to 35 MB**.
> - The cherry on top: we ran a full disaggregated training where the **trainer was on one box**, **vLLM lived in a Hugging Face Space**, the **Wordle environment lived in another Space**, and weights flowed through a single Hub bucket. No shared cluster, no RDMA, no VPN.
>
> Async RL just got a lot cheaper. Read on.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-timeline.static.hf.space" width="100%" height="460" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">Two ways to ship the same weights. Red is wall-clock time during which no tokens are being generated.</figcaption>
</figure>

---

## 1. The One Terabyte Problem

If you read our previous post on [the landscape of async RL training](https://huggingface.co/blog/huggingface/async-rl-training-landscape), you already know the punchline. Every async RL library, regardless of how it spells "actor model" or which color its NCCL backend is painted, eventually trips over the same root: **weight synchronization**.

The inference engine speaks the policy of step N. The trainer just finished step N+1. The fresh weights have to get from one side to the other before the inference engine starts drifting hopelessly off-policy. This sits on the critical path whether you are running sync or async: a blocking transfer is _wasted idle compute_ of GPUs not generating tokens. With a sparse delta path you collapse that idle time into seconds, and the trainer does not even have to wait for the inference engine to be ready: it just publishes "weights ready" and uploads the weights to the shared bucket the moment its optimizer step finishes, while the inference engine fetches on its own time.

Fireworks put a very memorable number on this in their post [Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think): for a frontier 1T-parameter checkpoint at fp8 (their setting), a full snapshot is **1024 GiB**, and that is what conventional wisdom says you have to ship every time you update your rollout fleet. That is the kind of number that gets people to start drawing diagrams with mega-clusters, RDMA fabrics, and dedicated cross-region links. Their measured average delta between adjacent checkpoints lands at **20.3 GiB, or 1.98% of the full model**, and "more than 98% of weights in bf16 format remain bit-equivalent between consecutive checkpoints".

Cursor's [Composer 2 report](https://huggingface.co/papers/2603.24477) tells a parallel story. They run training and inference in different regions and stitch them together with a **shared S3 bucket** (their exact words), into which the trainer uploads compressed weight diffs _every training step_. Each cluster independently downloads and reconstructs from the shared delta chain, "requiring no direct connectivity to the training cluster". The two sides never speak to each other about parameters directly. The bucket is the wire.

Both papers agree on three things, and we want to repeat them slowly, because the rest of this post is essentially a faithful open source translation:

1. Most of the weights have not actually changed between two adjacent RL steps.
2. If you send only the parts that changed, your bandwidth bill collapses by roughly two orders of magnitude.
3. If you route those tiny diffs through a shared object store, you no longer need the trainer and the inference cluster to live in the same data center.

The only thing missing was a version of this story that you can `pip install`. So we wrote one.

## 2. Why bf16 RL Weights Are Almost Always Sparse

Before we wire anything up, it is worth understanding why this whole game is even winnable. The "98% of weights do not change" claim sounds suspiciously like one of those numbers that works in the demo and falls apart in the wild. It is not. It falls out of how bf16 arithmetic works at the learning rates RL uses.

A bf16 number has 7 mantissa bits. Between two consecutive powers of two, there are exactly \\(2^7 = 128\\) representable values, so the spacing between adjacent bf16 numbers around \\(|w|\\) is roughly \\(|w| \cdot 2^{-7}\\). An update gets absorbed by the bf16 cast whenever it sits below _half_ of that spacing, i.e., when \\(|\Delta w| < |w|/256\\). This is the "bf16 visibility threshold" PULSE plots in their Figure 3.

Now look at what Adam does. At an RL learning rate of, say, \\(3 \times 10^{-6}\\), the update to a single weight is:

\\(\Delta w = -\eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}\\)

The normalized step \\(\hat{m}/(\sqrt{\hat{v}}+\epsilon)\\) is roughly order one, so \\(|\Delta w| \approx \eta \approx 3 \times 10^{-6}\\). For most weights, \\(|w|\\) sits somewhere around \\(10^{-2}\\) to \\(10^{-1}\\) (PULSE reports a median of 0.019 for representative LLM weights). The threshold \\(|w|/256\\) at that magnitude is around \\(4 \times 10^{-5}\\) to \\(4 \times 10^{-4}\\), which is _bigger_ than the update.

In other words: the optimizer is whispering, and bf16 cannot hear it. The update gets absorbed by rounding, the byte representation of \\(w\\) does not change, and from the inference engine's perspective, this weight did not move. Multiply that by a few hundred million parameters, and you get the >99% sparsity number, for free, with zero approximation.

This is exactly the argument made formal in the PULSE paper ([Mihai & Belilovsky, 2026](https://huggingface.co/papers/2602.03839)). They define two thresholds. The **absorption bound** \\(10\eta\\) is the conservative worst case for an Adam update, and the **effective bound** \\(\eta\\) is the regime you actually live in. The **bf16 visibility threshold** is \\(|w|/256\\). Whenever the update sits below the visibility threshold, it gets absorbed, and the bf16 byte does not change. Their Figure 3 plots both bounds against a cloud of representative LLM weights, and the conclusion is unambiguous: at \\(\eta = 3 \times 10^{-6}\\), the absorption bound itself already sits below the visibility threshold for almost every weight in the model. They measure this empirically across Qwen2.5 (0.5B/1.5B/7B), Llama-3.2-3B, and Gemma-3-4B, and consistently find a mean per-step sparsity of **\~99%, with a standard deviation of 0.2 to 0.4% over 400 training steps**. The worst-case step stays above 98%. So <1% changed is not a lucky measurement; it is what the arithmetic guarantees.

We do not have to predict this analytically (and indeed, we tried predicting the change mask from Adam's \\(m\\) and \\(v\\) statistics, but recall was a sad 30%, more on that later). We just need to **observe which bytes flipped**. That is a tiny boolean tensor per parameter, computed right around the optimizer step.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-bf16-ulp.static.hf.space" width="100%" height="780" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">Drag the learning rate down to RL territory and watch the cast-back-to-bf16 marker snap to the original tick. The 256-element grid on the bottom left is the aggregate effect across a tiny model.</figcaption>
</figure>

## 3. HF Buckets and the Architecture

Here is where the second piece of the story comes in, and where this post stops being a translation of Fireworks/Cursor and starts being a Hugging Face thing.

### 3.1 What is a Bucket?

A **Bucket** is a repo type on the Hub designed for high-frequency object storage. No commit ceremony, no PR workflow, no LFS quirks. You add files, you list files, you download files. The Python interface is two functions:

```python
from huggingface_hub import batch_bucket_files, download_bucket_files

# Trainer side
batch_bucket_files("my-org/wordle-deltas", add=[(buffer, "deltas/step_000042.safetensors")])

# Inference side
download_bucket_files("my-org/wordle-deltas", files=[("deltas/step_000042.safetensors", local_path)])
```

That is it. Two function calls and your weights are in flight.

Under the hood, buckets are backed by **Xet**, the Hub's content-defined chunking storage layer. Xet looks at every file you upload, slices it into chunks based on its actual content (not fixed offsets), and deduplicates against everything already in the bucket. The practical upshot, which is delightful in this context, is that even if we were too lazy to write the sparse encoding and just uploaded full anchors every step, Xet would _still_ only transfer the changed chunks. Sparse encoding + Xet stack: we pay for what moved, and we pay for it once.

This is the open source equivalent of the "shared S3 bucket" both Fireworks and Cursor reach for, except the storage layer already knows about content hashing, your existing HF token already has permission, and it composes natively with the rest of the stack (Spaces, datasets, models).

### 3.2 The Three Boxes

The full architecture has exactly three boxes and one shared substrate:

- **Trainer.** Wherever you want. One GPU, eight GPUs, a laptop with a USB-attached H100, we will not judge. Owns the model weights, runs the optimizer, emits sparse deltas.
- **HF Bucket.** A single repo, two prefixes: `anchors/` for occasional full snapshots and `deltas/` for the sparse patches in between. This is the only thing both sides agree on.
- **vLLM rollout server.** Wherever you want, and crucially _not necessarily where the trainer is_. Pulls from the bucket, applies the delta, and serves rollouts.
- **Environment.** Hangs off the rollout server in the usual way (HTTP, function calls, whatever your env speaks).

The property to internalize, the one Cursor's paper sells hard and that holds verbatim here: **the trainer and the rollout server never talk to each other about weights**. They exchange a tiny POST containing `{"repo_id": ..., "filename": ...}`, and that is the entire control plane. The actual byte transfer happens between each side and the bucket, in parallel, with no shared network fabric.

Why that matters in practice:

- The rollout server can be in another region, another cloud, or behind NAT inside a Hugging Face Space. It does not care.
- N inference replicas can pull the same delta from the same bucket, and Xet deduplicates the bytes across all of them.
- The trainer never has to know how many inference replicas exist, or where, or whether one of them just crashed.

The trainer writes. Replicas read. The Hub does the plumbing.

## 4. The Protocol

Now we can open the hood. The protocol has four parts: a wire format, a bucket layout, a 30 line vLLM extension, and a trainer side change detector. It is honestly less code than it sounds.

### 4.1 Safetensors as the Wire Format

We picked [safetensors](https://github.com/huggingface/safetensors) for the on-disk and on-wire format. It is already the canonical checkpoint format on the Hub, every reasonable framework can read it, and the header carries arbitrary string metadata. That metadata field is where we hide the protocol.

There are two kinds of files in the bucket.

**Anchors** look like a normal checkpoint: one tensor per parameter, full bf16 weights, written every \\(N\\) syncs (we default to \\(N=10\\)).

```
anchors/step_000010.safetensors
  ├── model.layers.0.self_attn.q_proj.weight   (bf16, full)
  ├── model.layers.0.self_attn.k_proj.weight   (bf16, full)
  └── ...
metadata:
  sparse=False, model_version=10, sparsity=0.0
```

**Deltas** are the interesting bit. For each parameter that actually changed, we store two entries: a flat int32 tensor of element indices, and a bf16 tensor of values at those indices.

```
deltas/step_000011.safetensors
  ├── model.layers.0.self_attn.q_proj.weight.indices   (int32, [num_changed])
  ├── model.layers.0.self_attn.q_proj.weight.values    (bf16,  [num_changed])
  ├── model.layers.0.mlp.gate_proj.weight.indices
  ├── model.layers.0.mlp.gate_proj.weight.values
  └── ...
metadata:
  sparse=True, model_version=11, sparsity=0.9938, changed_params=[...]
```

A few nice consequences of this choice:

- A delta is a _file_. You can open it with `safe_open(...)` in Python and inspect every tensor in it. No proprietary framing, no length prefixes, no version handshake.
- The metadata is self-describing. The receiver reads `sparse=True/False` and branches. There is no separate manifest.
- It is zero-copy via mmap on the inference side, which matters when you are doing this every few seconds.

The cadence is straightforward: anchor every Nth step, delta in between. Both end up in the same bucket under `anchors/` and `deltas/` prefixes. Each new inference replica only needs to grab the most recent anchor and then replay the deltas since.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-anchor-delta.static.hf.space" width="100%" height="600" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">Ten training steps. Anchor (full snapshot) on step 1 and step 6, sparse delta on every other step. Files land in the bucket as you watch.</figcaption>
</figure>

### 4.2 The Trainer Side: a Boolean Mask From an Optimizer Hook

The trainer needs to know which bf16 elements actually flipped. We do this with a tiny `BF16ChangeDetector` that registers a pre-step and post-step hook on the optimizer:

```python
class BF16ChangeDetector:
    def __init__(self, model, optimizer):
        self._pre_step_bf16: dict[str, torch.Tensor] = {}
        self._validated_masks: dict[str, torch.Tensor] = {}
        optimizer.register_step_pre_hook(self._pre_step_hook)
        optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, opt, args, kwargs):
        for p in self._params:
            self._pre_step_bf16[name_of(p)] = p.detach().to(torch.bfloat16).cpu().clone()

    def _post_step_hook(self, opt, args, kwargs):
        for p in self._params:
            self._validated_masks[name_of(p)] = (
                p.detach().to(torch.bfloat16).cpu() != self._pre_step_bf16[name_of(p)]
            )
```

The actual code in the PR has a bit more plumbing (matching optimizer param objects to model params via `data_ptr()`, because Accelerate wraps them as different Python objects), but the idea fits on a napkin: snapshot, step, diff.

This is ground truth. We _tried_ the more elegant path of predicting the mask from Adam's \\(m\\) and \\(v\\) statistics, using the bf16 ULP threshold directly. It works in principle. In practice, recall was around 30%, which means we would have shipped a delta missing two thirds of the actual updates. Adam's normalization is messy enough that the analytical threshold is not tight. So we just compare bytes. It costs one bf16 CPU snapshot of the model on the trainer side, which we are willing to pay.

The four phases of the new `_sync_weight` flow are:

1. **Upload while inference keeps running.** The trainer encodes the masked elements into a safetensors buffer and pushes it to the bucket. vLLM is still happily serving the old policy during this whole step.
2. **Pause vLLM.** A short HTTP call, hundreds of milliseconds.
3. **Signal `/update_weights`.** Send the bucket coordinates. vLLM downloads, applies, returns.
4. **Resume.** vLLM is back on the air.

The log lines tell the story:

```
Delta: 1234567/200000000 elements changed (sparsity=99.38%)
[delta_engine] uploaded user/wordle-deltas/deltas/step_000042.safetensors (27.4 MB, ...)
Weight sync: done. Total 9.4s (inference paused 1.1s)
```

The line that matters is the parenthesis. Inference was paused for **1.1 seconds**. The remaining 9.4 seconds were spent uploading, which occurred while the rollout server was still generating tokens. With NCCL, we were paying the full sync time as pause time. Here we are paying for it as background time.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-diff.static.hf.space" width="100%" height="700" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">A single sync, end to end. Switch between delta-over-bucket and NCCL broadcast, and try the replica count toggle to see the fan-out story.</figcaption>
</figure>

### 4.3 The vLLM Side: a 30 Line Extension

vLLM has a clean abstraction for this called `WeightTransferEngine`. We implement a `DeltaWeightTransferEngine` whose `receive_weights` method is, in spirit:

```python
def receive_weights(self, update_info, load_weights):
    download_bucket_files(update_info.repo_id, files=[(update_info.filename, local_path)])
    with safe_open(local_path, framework="pt", device="cpu") as f:
        meta = PatchMetadata.from_metadata_dict(f.metadata())
        if not meta.sparse:
            # Anchor: feed every tensor and snapshot for future deltas
            for name in f.keys():
                tensor = f.get_tensor(name)
                self._bf16_snapshot[name] = tensor.clone()
                load_weights([(name, tensor)])
        else:
            # Delta: apply (indices, values) to snapshot, hand full tensor to vLLM
            for name in json.loads(meta.changed_params):
                indices = f.get_tensor(f"{name}.indices").long()
                values = f.get_tensor(f"{name}.values")
                snap = self._bf16_snapshot[name].flatten()
                snap[indices] = values
                self._bf16_snapshot[name] = snap.reshape(self._bf16_snapshot[name].shape)
                load_weights([(name, self._bf16_snapshot[name])])
```

We register it via vLLM's `--worker-extension-cls` flag, which means **no fork of vLLM is required**. You install TRL into the same image as vLLM, point the CLI at our class, and you are done.

Worth mentioning: vLLM itself has an in-flight effort to land sparse weight transfer natively, [vllm-project/vllm#40096](https://github.com/vllm-project/vllm/pull/40096). It adds `receive_sparse_weights()` and `trainer_send_sparse_weights()` directly on the `WeightTransferEngine` base class, with patches encoded as `(indices, values)` and applied in place via `index_copy_()`, removing the GPU/CPU validation roundtrip entirely. The PR reports a transfer of **0.16 MB in 0.40 ms** for a sparse patch on Qwen3-1.7B versus **942 MB in 192 ms** for a full dense send.

One honest caveat in our implementation on the inference side: we keep a CPU bf16 snapshot of the model so we can reconstruct full tensors from sparse `(indices, values)` patches, because `load_weights` in vLLM today expects full tensors. Once [#40096](https://github.com/vllm-project/vllm/pull/40096) (or its successor) lands and exposes an in-place sparse `load_weights` path, we can apply the indices directly on the GPU and drop the snapshot!

## 5. Standing It Up on Spaces, For Real

This is the part we are smug about. Everything we have described so far works on your laptop, but the point of routing weights through a Hub bucket is that the trainer and the rollout server do not have to live anywhere near each other. So we ran a fully disaggregated training with three machines, none of which share a network:

- A box with one GPU running the **trainer**.
- A **Hugging Face Space** (Docker SDK, L4 GPU) running **vLLM** with our extension class.
- A second **Hugging Face Space** (CPU) running the **Wordle environment** server with 256 concurrent session capacity.
- A **Hub bucket** in the middle.

Setting this up is genuinely a few `hf` CLI calls. The vLLM Space's `Dockerfile` is essentially the upstream vLLM image plus `pip install trl@...` plus the entrypoint:

```dockerfile
FROM vllm/vllm-openai:latest
RUN pip install "trl @ git+https://github.com/huggingface/trl.git@delta-weight-sync"
ENV VLLM_SERVER_DEV_MODE=1
EXPOSE 7860
ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-1.7B", \
    "--host", "0.0.0.0", "--port", "7860", \
    "--worker-extension-cls", "trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension", \
    "--weight-transfer-config", "{\"backend\":\"nccl\"}", \
    "--max-model-len", "32768", \
    "--gpu-memory-utilization", "0.8"]
```

Deploy it as a Space:

```bash
hf repos create $USER/vllm-wordle-inference \
    --type space --space-sdk docker --flavor l4x1 \
    --secrets HF_TOKEN=$HF_TOKEN
hf upload $USER/vllm-wordle-inference examples/scripts/openenv/vllm_space/ --type space
```

And kick off training from anywhere on the planet that can talk HTTPS:

```bash
python examples/scripts/openenv/async_wordle.py \
    --vllm-server-url https://$USER-vllm-wordle-inference.hf.space \
    --env-url https://openenv-wordle.hf.space \
    --delta-sync-repo-id $USER/wordle-deltas \
    --model Qwen/Qwen3-1.7B
```

The trainer never opens a port. The Space never sees the trainer's IP. The Wordle environment does not know either of them exists. They all talk to the Hub. Training converged on the immediate-EOS sanity check, then on real Wordle rollouts: reward went up, delta payloads stayed in the 20 to 35 MB band, and the inference-paused window per sync stayed around a second. The full run logs are linked in the companion PR.

## 6. So What Does This Actually Unlock?

A few things, and we think they are big.

**Async RL training without a cluster.** If you have one GPU and a Hugging Face account, you can now do real disaggregated training. Your trainer is on the GPU; your rollout fleet lives in Spaces; your environment lives in another Space; weights move through a bucket. This used to require either a colocated setup (with all the throughput compromises that brings) or a real cluster with shared networking. It does not anymore.

**Multi-replica inference, for free.** Stand up two vLLM Spaces, or ten. They all pull from the same bucket. Xet content-addresses storage so consecutive anchors share chunks at rest (which keeps your bucket from blowing up), and the Hub's edge cache makes repeated downloads of the same file cheap to serve. Want a globally distributed rollout fleet? That is now a small DevOps exercise, not a research project.

**A wire format you can debug with your existing tools.** A delta is a safetensors file. You can `safe_open` it from a notebook, list its keys, inspect the indices, compute the sparsity yourself. We have spent enough hours in tcpdump on opaque NCCL streams to appreciate this.

**A path to frontier scale.** The 20 to 35 MB number is for Qwen3-0.6B. The interesting question is what the curve looks like once you turn the dial up. Let us do the napkin math.

Take Llama-3.1-405B. In bf16 that is **810 GB** on disk. PULSE measures \~99% mean per-step sparsity at RL learning rates, so the actual delta sits around 1% of the parameters. Their deployment-measured encoding hits **108 MB on a 7B model**, which is the **\~130×** reduction PULSE reports. Scaled linearly to 405B, the delta lands at roughly **6 GB per step**.

What does that buy you in wall-clock? NCCL is fast inside a cluster, sure. Assume a generous 100 GB/s aggregate broadcast bandwidth (multi-node, RDMA, the works). A full sync is `810 GB / 100 GB/s ≈ 8 seconds` of inference pause, every step. With the delta path, the trainer streams 6 GB to a bucket _in the background_ while generation keeps running, and the rollout server's actual paused window is just the apply step, which on this scale lands at a couple of seconds. So even before we leave the cluster, delta cuts the visible pause by 4× and the bytes on the wire by \~130×.

Now leave the cluster. NCCL straight up does not work across clouds. Once you want a rollout fleet in `us-east`, another in `eu-west`, maybe one in a Hugging Face Space, the bucket-based path is the _only_ path. At 1 GB/s of usable internet bandwidth, a single full broadcast would take 13 minutes; the delta does it in 6 seconds.

For a 1 TB-class model in the Fireworks framing, their own measured numbers show **20.3 GiB deltas vs the 1024 GiB full snapshot **, a \~50× reduction. PULSE's tighter, sparse encoding would push that further (extrapolating \~15 GB per delta, closer to \~65×). Either way, you are in a regime where shipping weights through commodity object storage stops being a hack and starts being the only sensible architecture.

## 7. What's Still on Our Plate

We are not pretending this is finished. Here is the honest list.

- **Two CPU bf16 snapshots, one too many.** The trainer keeps one (for the change detector) and the rollout server keeps one (to reconstruct full tensors for vLLM's `load_weights`). The first one we are stuck with until someone finds a tight analytical mask, which is harder than it looks. The second one goes away when vLLM gains a sparse `load_weights` API. PR forthcoming.
- **Fixed anchor cadence.** We currently dump a full anchor every \\(N\\) steps. An adaptive policy ("anchor when cumulative drift exceeds X") would cut anchor cost on long runs.
- **Multi-node FSDP2 trainers.** The `BF16ChangeDetector` is built around per-process optimizer hooks. It should generalize cleanly to FSDP2, but we have not measured it at multi-node scale yet. There is a `TODO` in the PR with our name on it.
- **Hooking into the optimizer.** Our attempt at predicting the mask from \\((m, v)\\) alone gave low recall, which means the analytical bf16 threshold is doing something more subtle than the textbook formula suggests. We would love to hear from anyone who has cracked this.
- **Stacking with on-the-wire compression.** Sparse safetensors and per-chunk gzip are orthogonal. We have not tried combining them yet. Although we don't expect huge compression gains.

## 8. Try It

- The PR: [huggingface/trl#5417](https://github.com/huggingface/trl/pull/5417). Branch is `delta-weight-sync`.
- The full Wordle example: `examples/scripts/openenv/async_wordle.py`.
- The Spaces Dockerfiles: `examples/scripts/openenv/vllm_space/` and `examples/scripts/openenv/wordle_space/`.
- Background reading: our [async RL landscape post](https://huggingface.co/blog/huggingface/async-rl-training-landscape), the [Fireworks 1 TB post](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think), the [Cursor Composer 2 report](https://huggingface.co/papers/2603.24477).
