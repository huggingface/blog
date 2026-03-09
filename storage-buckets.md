# Introducing Storage Buckets on the Hugging Face Hub

**TL;DR:**  Storage Buckets are a new repo type on the Hub providing S3-like object storage, powered by the Xet storage backend. Unlike git-based versioned repositories, buckets are designed for use cases where you need simple, fast, mutable storage — training checkpoints, logs, intermediate artifacts, or any large collection of files that doesn’t need version control. Now available to everyone on the Hub.

TODO: Add hero visual showing Buckets as the working storage layer on the Hub, next to versioned model and dataset repos.

Model repos and dataset repos are great for final artifacts — things you publish, document, version, and share. But production ML creates a lot of files before anything is ready to be published: raw inputs, processed shards, rolling checkpoints, optimizer states, embeddings, logs, traces, caches, and all the operational data that keeps a pipeline moving. Those files change constantly, often come from many jobs at once, and most of the time don’t benefit from a Git history on every update.

**Storage Buckets** are a new repo type on the Hub designed for exactly this. They work much closer to object storage like S3: mutable, simple, and high-throughput. You can browse them on the Hub, script them from Python, and manage them with the `hf` CLI. And because they are backed by [Xet](https://huggingface.co/docs/hub/en/xet), they are especially efficient for ML artifacts that share a lot of content with one another.

## Why we built Buckets

Git starts to feel like the wrong abstraction pretty quickly when you're dealing with:

- a training cluster writing checkpoints and optimizer states throughout a run
- data pipelines processing raw datasets iteratively
- agents storing traces, memory, and shared knowledge graphs

The storage need in all these cases is the same: write fast, overwrite when needed, sync directories, remove stale files, and keep things moving.

A Bucket is a non-versioned storage container on the Hub. It lives under a user or organization namespace, has standard Hugging Face permissions, can be private or public, has a page you can open in your browser, and can be addressed programmatically with a handle like `hf://buckets/username/my-training-bucket`.

## Why Xet matters

Buckets are built on [Xet](https://huggingface.co/docs/hub/en/xet), Hugging Face’s chunk-based storage backend, and this matters more than it might seem.

Instead of treating files as monolithic blobs, Xet breaks content into chunks and deduplicates across them. Upload a processed dataset that’s mostly similar to the raw one? Many chunks already exist. Store successive checkpoints where large parts of the model are frozen? Same story. Buckets skip the bytes that are already there, which means less bandwidth, faster transfers, and more efficient storage.

This is a natural fit for ML workloads. Training pipelines constantly produce families of related artifacts — raw and processed data, successive checkpoints, agent traces and derived summaries — and Xet is designed to take advantage of that overlap.

For Enterprise customers, billing is based on deduplicated storage, so shared chunks directly reduce the billed footprint. Deduplication helps with both speed and cost.

TODO: Add diagram from hf.co/storage.

## Pre-warming: bringing data close to compute

Buckets live on the Hub, which means global storage by default. But not every workload can afford to pull data from wherever it happens to live — for distributed training and large-scale pipelines, storage location directly affects throughput.

Pre-warming lets you bring hot data closer to the cloud provider and region where your compute runs. Instead of data traveling across regions on every read, you declare where you need it and Buckets make sure it's already there when your jobs start. This is especially useful for training clusters that need fast access to large datasets or checkpoints, and for multi-region setups where different parts of a pipeline run in different clouds.

TODO: Add a map or architecture figure showing one Bucket pre-warmed into specific cloud regions.

## A bucket workflow from the CLI

Enough about what Buckets are and why they exist. Let's see how they actually work in practice.

Install the `hf` CLI and log in:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```

Create a bucket for your project:

```bash
hf buckets create my-training-bucket --private
```

Say your training job is writing checkpoints locally to `./checkpoints`. Sync that directory into the Bucket:

```bash
hf buckets sync ./checkpoints hf://buckets/username/my-training-bucket/checkpoints
```

For large transfers, you might want to see what will happen before anything moves. `--dry-run` prints the plan without executing anything:

```bash
hf buckets sync ./checkpoints hf://buckets/username/my-training-bucket/checkpoints --dry-run
```

You can also save the plan to a file for review and apply it later:

```bash
hf buckets sync ./checkpoints hf://buckets/username/my-training-bucket/checkpoints --plan sync-plan.jsonl
hf buckets sync --apply sync-plan.jsonl
```

Once done, inspect the Bucket from the CLI:

```bash
hf buckets list username/my-training-bucket -h
```

or browse it directly on the Hub at `https://huggingface.co/buckets/username/my-training-bucket`.

That is the whole loop. Create a bucket, sync your working data into it, check on it when you need to, and save the versioned repo for when something is worth publishing. For one-off operations, `hf buckets cp` copies individual files and `hf buckets remove` cleans up stale objects.

TODO: Add a terminal screenshot showing `hf buckets sync --dry-run`, followed by the Bucket page in the browser.

## Using Buckets from Python

Everything above also works from Python via the `huggingface_hub` library. The API follows the same pattern: create, sync, inspect.

```python
from huggingface_hub import create_bucket, list_bucket_tree, sync_bucket

create_bucket("my-training-bucket", private=True, exist_ok=True)

sync_bucket(
    "./checkpoints",
    "hf://buckets/username/my-training-bucket/checkpoints",
)

for item in list_bucket_tree(
    "username/my-training-bucket",
    prefix="checkpoints",
    recursive=True,
):
    print(item.path, item.size)
```

This makes it straightforward to integrate Buckets into training scripts, data pipelines, or any service that manages artifacts programmatically. The Python client also supports batch uploads, selective downloads, deletes, and bucket moves for when you need finer control.

## From Buckets to versioned repos

Buckets are intentionally not the final stop in the lifecycle of an artifact. They are the fast, mutable place where artifacts live while they are still in motion. Once something becomes a stable deliverable, it belongs in a model or dataset repo with documentation, version history, and a clean public or internal interface.

That boundary is already useful today, and it is going to become even more powerful. On the roadmap, we plan to make it possible to transfer data directly between Buckets and model or dataset repos, in both directions. We are not putting a timeline on that here, but the use cases are clear. A team could keep frequent checkpoints in a Bucket and promote the final weights into a model repo. A distributed cluster could write processed shards into a Bucket all day and commit them into a dataset repo once the dataset is complete. The working layer and the publishing layer would stay separate, while still fitting into one continuous Hub-native workflow.

TODO: Add a simple workflow diagram showing Bucket -> model repo and cluster -> Bucket -> dataset repo.

## Conclusion and resources

Storage Buckets bring a missing storage layer to the Hub. They give you a Hub-native place for the mutable, high-throughput side of ML: checkpoints, processed data, agent traces, logs, and everything else that is useful before it becomes final.

Because they are built on Xet, Buckets are not just easier to use than forcing everything through Git. They are also more efficient for the kinds of related artifacts ML systems produce all the time. That means faster transfers, better deduplication, and on Enterprise plans, billing that benefits from the deduplicated footprint.

If you already use the Hub, Buckets let you keep more of your workflow in one place. If you come from S3-style storage, they give you a familiar model with better alignment to ML artifacts and a clear path toward final publication on the Hub.

Read more and try it yourself:

- [Storage overview](https://huggingface.co/storage)
- [Buckets guide](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)
- [CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [CLI reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli)
- [Installation guide](https://huggingface.co/docs/huggingface_hub/en/installation)
- [Example Bucket on the Hub](https://huggingface.co/buckets/julien-c/my-training-bucket)
