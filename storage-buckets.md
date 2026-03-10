# Introducing Storage Buckets on the Hugging Face Hub

Model and dataset repos are great for publishing final artifacts. But production ML generates a constant stream of intermediate files (checkpoints, optimizer states, processed shards, logs, traces, etc.) that change often, arrive from many jobs at once, and rarely need version control.

**Storage Buckets** are a new Hub repo type for exactly this: mutable, S3-like object storage you can browse on the Hub, script from Python, or manage with the `hf` CLI. And because they are backed by [Xet](https://huggingface.co/docs/hub/en/xet), they are especially efficient for ML artifacts that share content across files.

## Why we built Buckets

Git starts to feel like the wrong abstraction pretty quickly when you're dealing with:

- a training cluster writing checkpoints and optimizer states throughout a run
- data pipelines processing raw datasets iteratively
- Agents storing traces, memory, and shared knowledge graphs

The storage need in all these cases is the same: write fast, overwrite when needed, sync directories, remove stale files, and keep things moving.

A Bucket is a non-versioned storage container on the Hub. It lives under a user or organization namespace, has standard Hugging Face permissions, can be private or public, has a page you can open in your browser, and can be addressed programmatically with a handle like `hf://buckets/username/my-training-bucket`.

## Why Xet matters

Buckets are built on [Xet](https://huggingface.co/docs/hub/en/xet), Hugging Face’s chunk-based storage backend, and this matters more than it might seem.

Instead of treating files as monolithic blobs, Xet breaks content into chunks and deduplicates across them. Upload a processed dataset that’s mostly similar to the raw one? Many chunks already exist. Store successive checkpoints where large parts of the model are frozen? Same story. Buckets skip the bytes that are already there, which means less bandwidth, faster transfers, and more efficient storage.

This is a natural fit for ML workloads. Training pipelines constantly produce families of related artifacts — raw and processed data, successive checkpoints, Agent traces and derived summaries — and Xet is designed to take advantage of that overlap.

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

Everything above also works from Python via [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) (available since [v1.5.0](https://github.com/huggingface/huggingface_hub/releases/tag/v1.5.0)). The API follows the same pattern: create, sync, inspect.

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

Bucket support is also available in JavaScript via [`@huggingface/hub`](https://www.npmjs.com/package/@huggingface/hub) (since v2.10.5), so you can integrate Buckets into Node.js services and web applications as well.

## Filesystem integration

Buckets also work through `HfFileSystem`, the [fsspec](https://filesystem-spec.readthedocs.io/)-compatible filesystem in `huggingface_hub`. This means you can list, read, write, and glob Bucket contents using standard filesystem operations — and any library that supports fsspec can access Buckets directly.

```python
from huggingface_hub import hffs

# List files in a bucket directory
hffs.ls("buckets/username/my-training-bucket/checkpoints", detail=False)

# Glob for specific files
hffs.glob("buckets/username/my-training-bucket/**/*.parquet")

# Read a file directly
with hffs.open("buckets/username/my-training-bucket/config.yaml", "r") as f:
    print(f.read())
```

Because fsspec is the standard Python interface for remote filesystems, libraries like pandas, Polars, and Dask can read from and write to Buckets using `hf://` paths with no extra setup:

```python
import pandas as pd

# Read a CSV directly from a Bucket
df = pd.read_csv("hf://buckets/username/my-training-bucket/results.csv")

# Write results back
df.to_csv("hf://buckets/username/my-training-bucket/summary.csv")
```

This makes it easy to plug Buckets into existing data workflows without changing how your code reads or writes files.

## From Buckets to versioned repos

Buckets are intentionally not the final stop in the lifecycle of an artifact. They are the fast, mutable place where artifacts live while they are still in motion. Once something becomes a stable deliverable, it belongs in a model or dataset repo with documentation, version history, and a clean public or internal interface.

That boundary is already useful today, and it is going to become even more powerful. On the roadmap, we plan to make it possible to transfer data directly between Buckets and model or dataset repos, in both directions. We are not putting a timeline on that here, but the use cases are clear. A team could keep frequent checkpoints in a Bucket and promote the final weights into a model repo. A distributed cluster could write processed shards into a Bucket all day and commit them into a dataset repo once the dataset is complete. The working layer and the publishing layer would stay separate, while still fitting into one continuous Hub-native workflow.

TODO: Add a simple workflow diagram showing Bucket -> model repo and cluster -> Bucket -> dataset repo.

## Trusted by launch partners

Before opening Buckets to everyone, we ran a private beta with a small group of launch partners.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/buckets/buckets-launch-partners.png"/>
</div>

A huge thank you to Jasper, Arcee, IBM, and PixAI for testing early versions, surfacing bugs, and sharing feedback that directly shaped this feature.

## Conclusion and resources

Storage Buckets bring a missing storage layer to the Hub. They give you a Hub-native place for the mutable, high-throughput side of ML: checkpoints, processed data, Agent traces, logs, and everything else that is useful before it becomes final.

Because they are built on Xet, Buckets are not just easier to use than forcing everything through Git. They are also more efficient for the kinds of related artifacts ML systems produce all the time. That means faster transfers, better deduplication, and on Enterprise plans, billing that benefits from the deduplicated footprint.

If you already use the Hub, Buckets let you keep more of your workflow in one place. If you come from S3-style storage, they give you a familiar model with better alignment to ML artifacts and a clear path toward final publication on the Hub.

Buckets are included in existing [Hub storage plans](https://huggingface.co/docs/hub/en/storage-limits#storage-plans). Free accounts come with storage to get started, and PRO and Enterprise plans offer higher limits. See the [storage page](https://huggingface.co/storage) for details.

Read more and try it yourself:

- [Buckets guide](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)
- [Installation guide](https://huggingface.co/docs/huggingface_hub/en/installation)
- CLI [guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and [reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli)
- [Example Bucket on the Hub](https://huggingface.co/buckets/julien-c/my-training-bucket)
- [Storage overview](https://huggingface.co/storage)
