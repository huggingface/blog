---
title: "Growing Beyond Git LFS with Xet" 
thumbnail: /blog/assets/growing-with-xet/thumbnail.png
authors:
  - user: jsulz
    org: xet-team
  - user: jgodlewski
    org: xet-team
  - user: sirahd
    org: xet-team
---

# Growing Beyond Git LFS with Xet

In January of this year, the Xet storage backend moved into production, shifting [~6% of Hub downloads through the infrastructure](https://huggingface.co/blog/xet-on-the-hub). This represented a significant milestone, but it was just the beginning. Over 6 months, 20 PB and 500,000 repositories joined the move to Xet as the Hub outgrows Git LFS and shifts to a storage system that scales with the workloads of AI builders. 

Today, more than 1 million people and organizations on the Hub are using Xet. As the [default on the Hub for new users and organizations](https://huggingface.co/changelog/xet-default-for-new-users), that number grows every day. With only a few dozen GitHub issues, forum threads, and Discord messages, this is perhaps the quietest migration of this magnitude. 

How? It helps that the [Xet Team](https://huggingface.co/xet-team) came prepared with years of experience building and supporting the content addressed store (CAS) and [Rust client](https://github.com/huggingface/xet-core) that provide the foundation. Without these pieces, Git LFS is still the future on the Hub, but the unsung heroes of this story are:

1. An integral piece of infrastructure known as the Git LFS Bridge
2. Background content migrations that run around the clock

Together, these components have allowed us to aggressively migrate PBs in the span of days without worrying about the impact to the Hub community. They're what is giving us the piece of mind to move even faster in the coming weeks and months.

## Bridges and backward compatibility

Early on, we made a few key design decisions: 
- There would be no "cut-over" migration from Git LFS to Xet
- A repository could be a mixture of Xet and LFS files
- Migrations don't require "locks"; a migration can run in the background without blocking uploads or downloads

These seemingly straightforward decisions had significant implications, but ones that were user and community driven. One such implication was that users would not have to alter their workflow or download a new client to interact with Xet-enabled repositories.

If you have a Xet-aware client (e.g., `hf-xet`, the Xet integration with `huggingface_hub`), uploads and downloads pass through every piece of the Xet stack. The client either [breaks up files into chunks using content defined chunking](https://huggingface.co/blog/from-files-to-chunks) when uploading or requests reconstruction information when downloading. On upload, [chunks are passed to CAS and stored in S3](https://huggingface.co/blog/rearchitecting-uploads-and-downloads). During downloads, [CAS provides the chunk ranges the client needs to request from S3](https://huggingface.co/blog/rearchitecting-uploads-and-downloads#a-custom-protocol-for-uploads-and-downloads) to reconstruct the file locally.

For older versions of `huggingface_hub` or [huggingface.js](https://github.com/huggingface/huggingface.js), which do not support chunk-based file transfers, you can still download and upload to Xet repos, but through a very different route. These clients download through our Git LFS Bridge. When invoked, the Bridge constructs and returns a single [presigned URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html), mimicking the LFS protocol. This URL is requested from CAS, which then does the work of reconstructing the file from S3 itself and returning it to the client.

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/growing-with-xet/bridge.png" alt=" ">
    <figcaption>Git LFS Bridge architecture.</figcaption>
</figure>

Meanwhile, when a non-Xet-aware client uploads a file, the file is first uploaded to LFS storage where we migrate the content in the background to Xet. This “background migration process,” only [briefly mentioned in our docs](https://huggingface.co/docs/hub/en/storage-backends#backward-compatibility-with-lfs), powers both the migrations to Xet and backward compatibility. 

Every time a file needs to be migrated from LFS to Xet, a webhook is triggered, pushing the event to a distributed queue where it is processed by an orchestrator. The orchestrator:

- Enables Xet on the repo if the event calls for it
- Fetches a listing of LFS revisions for every LFS file in the repo
- Batches the files into jobs based on size or number of files; either a 1000 files or 500MB, whichever comes first
- Places the jobs on another queue for individual workers

These jobs are then picked up by migration worker pods where each pod:

- Downloads the LFS files in the batch
- Uploads the LFS files to the Xet content addressed store using xet-core

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/growing-with-xet/flow.png" alt=" ">
    <figcaption>Migration flow triggered by a webhook event; starting at the orchestrator for brevity.</figcaption>
</figure>

Migration flow triggered by a webhook event; starting at the orchestrator for brevity. 

## Scaling migrations

In April, we tested the system limits by onboarding [bartowski](https://huggingface.co/bartowski). With nearly 500 TB across 2,000 repos the load uncovered a few weak links: 

- Temporary shard files for global dedupe were first written to `/tmp` and then moved into the shard cache. On our worker pods, however, `/tmp` and the [Xet cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#chunk-based-caching-xet) sat on different mount points, so the move failed and the shard files were never cleaned up. They piled up until `/tmp` filled, triggering a wave of **`No space left on device`** errors.
- After supporting the [launch of Llama 4](https://huggingface.co/blog/llama4-release), we’d scaled CAS for bursty downloads, but the migration workers flipped the script as hundreds of multi-gigabyte uploads overwhelmed CAS
- On paper, the migration workers were capable of significantly more throughput than what was reported; profiling the pods revealed network and EBS I/O bottlenecks

Fixing this three-headed monster meant touching every layer - patching xet-core, resizing CAS, and beefing up the worker node specs. Fortunately, [bartowski](https://huggingface.co/bartowski) was game to wait and every repo made its way to Xet. These same lessons powered the moves of some of the biggest storage users on the Hub: 

- RichardErkhov (1.7PB and 25,000 repos!)
- mradermacher (6.1PB and 42,000 repos 🤯)

All told, this seemingly simple system is behind the migration of well over a dozen PBs of models and datasets and is keeping 500,000 repos in sync with Xet storage all without missing a beat. 

Our CAS throughput has grown by an order of magnitude between the first and latest large-scale moves:

- **Bartowski migration:** CAS sustained ~35 Gb/s, with ~5 Gb/s coming from regular Hub traffic.
- **mradermacher and** RichardErkhov **migrations:** CAS peaked around ~300 Gb/s, while still serving ~40 Gb/s of everyday load.

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/growing-with-xet/grafana.png" alt=" ">
    <figcaption>CAS throughput from late March through to today. Each spike corresponds to a migration with CAS's baseline throughput steadily increasing to just shy of 100 Gb/s</figcaption>
</figure>

CAS throughput from late March through to today. Each spike corresponds to a migration with CAS's baseline throughput steadily increasing to just shy of 100 Gb/s.

## Zero friction, faster transfers

When we began replacing LFS, our goals were twofold:

1. Do no harm
2. Drive the most impact as fast as possible

Faster file transfers alleviate a big pain, and our benchmarks show that `hf-xet` is ~125% faster than `hf-transfer` on downloads and ~25% faster on uploads (with that number increasing along with the size of the workload). But our goal was to make this as seamless as possible.

Designing this migration and backwards compatibility allowed us to:

- Introduce and harden `hf-xet` before including it in `huggingface_hub` as a required dependency
- Support the community uploading to and downloading from Xet-enabled repos through whatever means they use today while our infrastructure handles the rest

Instead of waiting for all upload paths to become Xet-aware or forcing the community to adopt a specific workflow when interacting with a Xet repository, we could begin migrating the Hub to Xet today with minimal user impact. In short, let teams keep their workflows and organically transition to Xet with infrastructure to support the long-term goal of a unified storage system. 

## Xet for everyone

In January and February, we onboarded power users to provide feedback and pressure-test the infrastructure. Since then, we launched [a waitlist for early adopters](https://huggingface.co/join/xet) to preview Xet-enabled repositories. 8,000 organizations and users organically signed up and moved more than 200,000 repositories (15 PB) to Xet. Soon after, Xet became the default for new users on the Hub.

We support some of the largest creators (Meta, Google, Microsoft, OpenAI, Qwen) while the community keeps working uninterrupted.

What's next?

Starting this month, we're bringing Xet to everyone. Watch for an email providing access to Xet and once you have it, update to the latest `huggingface_hub` (`pip install -U huggingface_hub`) to unlock faster transfers right away. This will also mean: 

- All of your existing repositories will migrate from LFS to Xet
- All newly created repos will be Xet-enabled by default

If you upload or download from the Hub using your browser or use Git, that's fine. Chunk-based support for both is coming soon. In the meantime use whichever workflow you already have; no restrictions. 

Next up: open-sourcing the Xet protocol and the entire infrastructure stack. The future of storing and moving bytes that scale to AI workloads is on the Hub, and we're aiming to bring it to everyone.

If you have any questions, drop us a line in the comments 👇, [**open a discussion**](https://huggingface.co/spaces/xet-team/README/discussions/new) on the [**Xet team**](https://huggingface.co/xet-team) page.