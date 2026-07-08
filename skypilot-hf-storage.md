---
title: "Run AI workloads on any cloud, store on Hugging Face: zero-egress storage with SkyPilot"
thumbnail: /blog/assets/skypilot-hf-storage/thumbnail.png
authors:
  - user: njha
    guest: true
  - user: michaelvll
    guest: true
    org: skypilot-org
  - user: hopechong
    guest: true
    org: skypilot-org
  - user: XciD
  - user: julien-c
---

# Run AI workloads on any cloud, store on Hugging Face: zero-egress storage with SkyPilot

For most teams, models and datasets live in a bucket in one region of one cloud. The GPUs you can get, whether for development, training, or serving, increasingly sit on a different cloud than your data. The moment those two come apart, you pay a cross-cloud transfer tax just to read your own data onto your own GPUs.

Together with Hugging Face, we've joined the two halves: your models and datasets stay on the Hub, and SkyPilot runs the compute (dev, training, or serving) on whatever cluster has the GPUs. Mount a Hugging Face Bucket or any Hub repo into a SkyPilot job with one `hf://` URL and the `HF_TOKEN` you already have, then launch it wherever capacity is. Hugging Face charges no egress, so reading your data onto those GPUs costs nothing, on any cloud.

Here's what's new:

- **Your Hub data in any job.** `store: hf` mounts a Hugging Face **Bucket** (read-write) or any **model / dataset / Space repo** (read-only) into a SkyPilot task with one `hf://` URL and your existing `HF_TOKEN`, via `MOUNT` or `COPY`.
- **Run it on any GPU, on any cloud.** [SkyPilot](https://docs.skypilot.co/) finds that job compute across 20+ clouds, Kubernetes, Slurm, and on-prem, so the same run uses whichever of your reserved or on-demand GPUs is available, on any vendor.
- **No egress to read your data.** Hugging Face Storage charges [no egress or CDN fees](https://huggingface.co/pricing), so wherever SkyPilot lands the job, it reads your models and datasets straight from the same bucket, with no per-cloud copies and no egress bill to pull them in.
- **Xet-backed dedup.** Buckets are built on [Xet](https://huggingface.co/docs/hub/xet/overview), so incremental checkpoints and model variants only store and transfer the chunks that changed.
- **Built together.** [Hugging Face](https://huggingface.co/) and [SkyPilot](https://docs.skypilot.co/) shipped this jointly, and the Hugging Face team upstreamed the `hf-mount` FUSE fixes that make it work in unprivileged containers.

## Hugging Face Storage is now a first-class SkyPilot backend

![SkyPilot mounts Hugging Face models, datasets, and checkpoints into jobs running on reserved GPU clusters across CoreWeave, Nebius, GCP, and 20+ more, with zero-egress reads.](/blog/assets/skypilot-hf-storage/architecture.png)

SkyPilot tasks already read and write cloud object stores (S3, GCS, Azure, R2, and many more) by mounting them at a local path. Hugging Face Storage now joins that list as `store: hf`, reached through the `hf://` scheme:

```yaml
file_mounts:
  # A Hugging Face Bucket, read-write, for checkpoints, logs, processed data.
  /checkpoints:
    source: hf://buckets/my-org/qwen-sft
    store: hf
    mode: MOUNT # or COPY
  # A model repo, mounted read-only.
  /base-model:
    source: hf://Qwen/Qwen3.5-4B
    store: hf
    mode: MOUNT
  # A dataset repo, pinned to a revision, read-only.
  /data:
    source: hf://datasets/my-org/my-dataset@main
    store: hf
    mode: MOUNT
```

That one `hf://` scheme covers the whole lifecycle: read the **model** and **dataset** from their repos, write **checkpoints** to a Bucket while you train, publish the finished model back to a repo, and pull it onto inference servers when you serve. Most teams already keep their models and datasets on the Hub, so there is no migration step and no new storage account to create.

`MOUNT` uses Hugging Face's [`hf-mount`](https://github.com/huggingface/hf-mount) FUSE backend, so a bucket or repo shows up as a local path next to SkyPilot's other FUSE mounts (`gcsfuse`, `blobfuse2`, `rclone`, `goofys`). The fetching happens at the filesystem layer: when your code issues a `read()`, the driver pulls just those bytes from the Xet backend, so only the data you actually touch crosses the network, and `hf-mount` keeps an on-disk cache so repeat reads stay local. That on-disk cache is the behavior SkyPilot gives its other backends under [`MOUNT_CACHED`](https://docs.skypilot.co/en/latest/reference/storage.html), where a plain `MOUNT` instead streams every read from the bucket with nothing kept locally. For the `hf` store, `MOUNT` and `MOUNT_CACHED` behave the same, so either mode keeps the cache.

Because reads are lazy, a process can start working through a large file before the whole file has downloaded, instead of blocking on a full copy first. That keeps the GPU busy almost immediately, training on data as it streams in rather than sitting idle (and billing) while a dataset or checkpoint copies down. It pays off most on the first epoch, when nothing is cached yet. `COPY` takes the other route and downloads through `huggingface_hub` up front, with no special requirements.

Authentication is the token you already have. Set `HF_TOKEN` in your environment and hand it to a run with [`--secret HF_TOKEN`](https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html); SkyPilot uses it for the mount on whatever cloud the job lands. One token works whether the job lands on AWS, GCP, Azure, Nebius, Lambda, or your own Kubernetes cluster, so there are no per-cloud bucket keys to juggle.

## No egress: storage stops deciding where you run

GPU capacity rarely comes from one place anymore. To get enough H100s and H200s, teams hold reserved and committed capacity across several vendors at once (a block on a hyperscaler, a cluster on a neocloud, maybe an on-prem rack) and run wherever they have allocation. SkyPilot is built for this: one job spec, scheduled across 20+ clouds, Kubernetes, and on-prem, landing on whichever reserved cluster is free.

Object storage has been the catch. Object stores are regional and per-cloud, so feeding a GPU or an inference server that sits in a different vendor's data center means either keeping a copy of your data in every vendor's bucket or paying to pull it across. Most clouds charge egress (around $0.09/GB out of AWS) the moment data leaves their network, and often between regions inside one cloud. Pulling a base model onto every inference node, or iterating a dataset for several epochs from a cluster on another cloud, adds a hefty bill on top of GPUs you have already reserved. Teams end up pinning each run to whichever vendor holds the data and leaving the rest of their capacity idle.

Hugging Face Storage takes that cost off the table where it bites: the read side. With [no egress or CDN fees](https://huggingface.co/pricing) and storage at $12-18/TB/month (versus AWS S3 at roughly $23/TB plus egress), the same bucket is reachable from every one of those clusters, and reading from it is free no matter where the GPUs run. Writing back still costs your compute cloud's usual egress, the same as it would to any off-cloud store, but for most AI work the reads dominate: a dataset streamed over many epochs, or model weights pulled onto every new training or inference node. So you stop pinning each run to whichever vendor holds a copy of the data.

## A quick benchmark

To collect some benchmark numbers, we ran a small fine-tune: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) on the [`HuggingFaceH4/Multilingual-Thinking`](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset with TRL's [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer), mounting the model read-only from its Hub repo and writing every checkpoint to a Hugging Face Bucket. The same SkyPilot YAML ran on AWS, GCP, and Lambda, changing only `--infra`. SkyPilot placed each job wherever GPUs were free, and all three read and wrote the same bucket.

```yaml
# qwen-sft.yaml. Launch anywhere: sky launch qwen-sft.yaml --infra aws|gcp|...
resources:
  accelerators: H100:1 # or whatever the cloud has

file_mounts:
  /base-model:
    source: hf://Qwen/Qwen3.5-4B # read-only, lazy-mounted from the Hub
    store: hf
    mode: MOUNT
  /checkpoints:
    source: hf://buckets/my-org/qwen-sft # read-write Bucket
    store: hf
    mode: MOUNT

run: |
  python train.py --model /base-model --output_dir /checkpoints
```

What we measured:

- **The model loaded free on every cloud.** Lazy reads pull only what `from_pretrained` touches, so it was ready to train in about 30 seconds (up to ~500 MB/s). Because Hugging Face charges no egress, that pull cost nothing; had the model lived in S3, every read to a GPU on another cloud would have been billed egress (~$0.09/GB on AWS).
- **Checkpoints streamed straight to the bucket** at up to ~170 MB/s (8.43 GB of weights each) and persisted past the GPU instance.

Per cloud, checkpoints wrote to the bucket at:

| Cloud              | GPU  | Checkpoint write |
| :----------------- | :--- | :--------------- |
| AWS (us-east-2)    | L40S | ~168 MB/s        |
| GCP (us-central1)  | L4   | ~123 MB/s        |
| Lambda (us-west-3) | H100 | ~112 MB/s        |

## Xet-backed storage: dedup for checkpoints and model variants

Hugging Face Buckets are built on [Xet](https://huggingface.co/docs/hub/xet/overview), which uses [content-defined chunking](https://huggingface.co/docs/hub/xet/deduplication) to split files into ~64 KB chunks and store each unique chunk once. Because the boundaries follow the content, an edit changes only the chunks it touches and the rest are recognized as already stored. This pays off in a few places:

- **Incremental and adapter checkpoints.** When you freeze layers, train adapters, or otherwise leave most weights untouched between saves, only the changed chunks upload instead of the whole checkpoint.
- **Model variants that share a base.** Fine-tunes and quantizations of one base model overlap heavily, so the shared chunks are stored once across all of them.
- **Datasets you append to.** Logs like conversation traces or inference outputs grow by appending rows to large Parquet files. The existing row groups stay byte-identical, so only the new rows transfer: in Hugging Face's [test](https://huggingface.co/blog/parquet-cdc), appending 10K rows to a 100K-row table moved about 10 MB instead of the full ~106 MB. (If you edit or delete rows in place, write with `use_content_defined_chunking=True` to keep changes local.)
- **Re-uploads skip what's already stored.** In our test, re-uploading an 8.43 GB blob already in the bucket took about 8 seconds, versus 24 seconds for the first upload, because only chunk hashes move. The same mechanism lets server-side `hf buckets cp` between repos and buckets copy by reference instead of re-uploading bytes.

How much you save depends on how much your artifacts overlap, but the deduplication is automatic: you write a checkpoint as usual, and only the new chunks leave the machine.

## Get started

```bash
pip install "skypilot[huggingface]"
hf auth login  # or: export HF_TOKEN=<your-token>
```

Add an `hf://` mount to any SkyPilot task and launch. `MOUNT` needs a base image with glibc 2.34+ and `/dev/fuse`.

## Built together: Hugging Face and SkyPilot

The initial `store: hf` support [started as a contribution](https://github.com/skypilot-org/skypilot/pull/9418) from Nikhil Jha. The Hugging Face team [carried it forward](https://github.com/skypilot-org/skypilot/pull/9698) and upstreamed the `hf-mount` FUSE fixes that let it mount in unprivileged containers, the default on many Kubernetes clusters. The SkyPilot team wired it into the storage backend. The whole path is open source: SkyPilot, Hugging Face's `hf-mount`, and the `huggingface_hub` client.

## Resources

- [SkyPilot storage docs](https://docs.skypilot.co/en/latest/reference/storage.html)
- [Hugging Face Storage Buckets guide](https://huggingface.co/docs/hub/storage-buckets)
- [`hf-mount`](https://github.com/huggingface/hf-mount)
- [Xet: content-defined chunking and deduplication](https://huggingface.co/docs/hub/xet/deduplication)
- [SkyPilot Slack community](https://slack.skypilot.co/)
