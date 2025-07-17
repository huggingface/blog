---
title: "Migrating the Hub from Git LFS to Xet" 
thumbnail: /blog/assets/migrating-the-hub-to-xet/thumbnail.png
authors:
  - user: jsulz
    org: xet-team
  - user: jgodlewski
    org: xet-team
  - user: sirahd
    org: xet-team
---

# Migrating the Hub from Git LFS to Xet

In January of this year, Hugging Face's [Xet Team](https://huggingface.co/xet-team) deployed a new storage backend, and shortly after shifted [~6% of Hub downloads through the infrastructure](https://huggingface.co/blog/xet-on-the-hub). This represented a significant milestone, but it was just the beginning. In 6 months, 500,000 repositories holding 20 PB joined the move to Xet as the Hub outgrows Git LFS and transitions to a storage system that scales with the workloads of AI builders. 

Today, more than 1 million people on the Hub are using Xet. In May, it became the [default on the Hub for new users and organizations](https://huggingface.co/changelog/xet-default-for-new-users). With only a few dozen GitHub issues, forum threads, and Discord messages, this is perhaps the quietest migration of this magnitude. 

How? For one, the team came prepared with years of experience building and supporting the content addressed store (CAS) and [Rust client](https://github.com/huggingface/xet-core) that provide the system's foundation. Without these pieces, Git LFS may still be the future on the Hub. However, the unsung heroes of this migration are:

1. An integral piece of infrastructure known internally as the Git LFS Bridge
2. Background content migrations that run around the clock

Together, these components have allowed us to aggressively migrate PBs in the span of days without worrying about the impact to the Hub or the community. They're giving us the peace of mind to move even faster in the coming weeks and months ([skip to the end](#xet-for-everyone) 👇 to see what's coming).

## Bridges and backward compatibility

In the early days of planning the migration to Xet, we made a few key design decisions: 
- There would be no "hard cut-over" from Git LFS to Xet
- A Xet-enabled repository should be able to contain both Xet and LFS files
- Repository migrations from LFS to Xet don't require "locks"; that is, they can run in the background without disrupting downloads or uploads

Driven by our commitment to the community, these seemingly straightforward decisions had significant implications. Most importantly, we did not believe users and teams should have to immediately alter their workflow or download a new client to interact with Xet-enabled repositories.

If you have a Xet-aware client (e.g., `hf-xet`, the Xet integration with `huggingface_hub`), uploads and downloads pass through the entire Xet stack. The client either [breaks up files into chunks using content defined chunking](https://huggingface.co/blog/from-files-to-chunks) while uploading, or requests file reconstruction information when downloading. On upload, [chunks are passed to CAS and stored in S3](https://huggingface.co/blog/rearchitecting-uploads-and-downloads). During downloads, [CAS provides the chunk ranges the client needs to request from S3](https://huggingface.co/blog/rearchitecting-uploads-and-downloads#a-custom-protocol-for-uploads-and-downloads) to reconstruct the file locally.

For older versions of `huggingface_hub` or [huggingface.js](https://github.com/huggingface/huggingface.js), which do not support chunk-based file transfers, you can still download and upload to Xet repos, but these bytes take a different route.  When a Xet-backed file is requested from the Hub along the `resolve` endpoint, the Git LFS Bridge constructs and returns a single [presigned URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html), mimicking the LFS protocol. The Bridge then does the work of reconstructing the file from the content held in S3 and returns it to the requester.

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/migrating-the-hub-to-xet/bridge.png" alt="Git LFS Bridge flow">
    <figcaption>Greatly simplified view of the Git LFS Bridge - in reality this path includes a few more API calls and components like the CDN fronting the Bridge, DynamoDB for file metadata, and S3 itself.</figcaption>
</figure>

To see this in action, right click on the image above and open it in a new tab. The URL redirects from
`https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/migrating-the-hub-to-xet/bridge.png` to one that begins with `https://cas-bridge.xethub.hf.co/xet-bridge-us/...`. You can also use `curl -vL` on the same URL to see the redirects in your terminal. 

Meanwhile, when a non-Xet-aware client uploads a file, it is sent first to LFS storage then migrated to Xet. This “background migration process,” only [briefly mentioned in our docs](https://huggingface.co/docs/hub/en/storage-backends#backward-compatibility-with-lfs), powers both the migrations to Xet and upload backward compatibility. It is behind the migration of well over a dozen PBs of models and datasets and is keeping 500,000 repos in sync with Xet storage all without missing a beat.

Every time a file needs to be migrated from LFS to Xet, a webhook is triggered, pushing the event to a distributed queue where it is processed by an orchestrator. The orchestrator:

- Enables Xet on the repo if the event calls for it
- Fetches a listing of LFS revisions for every LFS file in the repo
- Batches the files into jobs based on size or number of files; either 1000 files or 500MB, whichever comes first
- Places the jobs on another queue for migration worker pods

These migration workers then pick up the jobs and each pod:

- Downloads the LFS files listed in the batch
- Uploads the LFS files to the Xet content addressed store using [xet-core](https://github.com/huggingface/xet-core)

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/migrating-the-hub-to-xet/flow.png" alt="Migration flow">
    <figcaption>Migration flow triggered by a webhook event; starting at the orchestrator for brevity.</figcaption>
</figure>

## Scaling migrations

In April, we tested this system's limits by reaching out to [bartowski](https://huggingface.co/bartowski) and asking if they wanted to test out Xet. With nearly 500 TB across 2,000 repos, bartowski's migration uncovered a few weak links: 

- Temporary [shard files for global dedupe](https://huggingface.co/blog/from-chunks-to-blocks#scaling-deduplication-with-aggregation) were first written to `/tmp` and then moved into the shard cache. On our worker pods, however, `/tmp` and the [Xet cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#chunk-based-caching-xet) sat on different mount points. The move failed and the shard files were never removed. Eventually the disk filled, triggering a wave of **`No space left on device`** errors.
- After supporting the [launch of Llama 4](https://huggingface.co/blog/llama4-release), we'd scaled CAS for bursty downloads, but the migration workers flipped the script as hundreds of multi-gigabyte uploads pushed CAS beyond its resources
- On paper, the migration workers were capable of significantly more throughput than what was reported; profiling the pods revealed network and [EBS](https://aws.amazon.com/ebs/) I/O bottlenecks

Fixing this three-headed monster meant touching every layer - patching xet-core, resizing CAS, and beefing up the worker node specs. Fortunately, [bartowski](https://huggingface.co/bartowski) was game to work with us while every repo made its way to Xet. These same lessons powered the moves of the biggest storage users on the Hub like [RichardErkhov](https://huggingface.co/RichardErkhov) (1.7PB and 25,000 repos) and [mradermacher](https://huggingface.co/mradermacher) (6.1PB and 42,000 repos 🤯).

CAS throughput, meanwhile, has grown by an order of magnitude between the first and latest large-scale migrations:

- **Bartowski migration:** CAS sustained ~35 Gb/s, with ~5 Gb/s coming from regular Hub traffic.
- **mradermacher and RichardErkhov migrations:** CAS peaked around ~300 Gb/s, while still serving ~40 Gb/s of everyday load.

<figure class="image text-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/migrating-the-hub-to-xet/grafana.png" alt="Cas throughput">
    <figcaption>CAS throughput; each spike corresponds to a significant migration with the baseline throughput steadily increasing to just shy of 100 Gb/s as of July 2025</figcaption>
</figure>

## Zero friction, faster transfers

When we began replacing LFS, we had two goals in mind:

1. Do no harm
2. Drive the most impact as fast as possible

Designing with our initial constraints and these goals allowed us to:

- Introduce and harden `hf-xet` before including it in `huggingface_hub` as a required dependency
- Support the community uploading to and downloading from Xet-enabled repos through whatever means they use today while our infrastructure handles the rest
- Learn invaluable lessons - from scale to how our client operated on distributed file systems - from incrementally migrating the Hub to Xet

Instead of waiting for all upload paths to become Xet-aware, forcing a hard cut-over, or pushing the community to adopt a specific workflow, we could begin migrating the Hub to Xet immediately with minimal user impact. In short, let teams keep their workflows and organically transition to Xet with infrastructure supporting the long-term goal of a unified storage system. 

## Xet for everyone

In January and February, we onboarded power users to provide feedback and pressure-test the infrastructure. To get community feedback, we launched [a waitlist](https://huggingface.co/join/xet) to preview Xet-enabled repositories. Soon after, Xet became the default for new users on the Hub.

We now support some of the largest creators on the Hub ([Meta Llama](https://huggingface.co/meta-llama), [Google](https://huggingface.co/google), [OpenAI](https://huggingface.co/openai), and [Qwen](https://huggingface.co/Qwen)) while the community keeps working uninterrupted.

What's next?

Starting this month, we're bringing Xet to everyone. Watch for an email providing access to Xet and once you have it, update to the latest `huggingface_hub` (`pip install -U huggingface_hub`) to unlock faster transfers right away. This will also mean: 

- All of your existing repositories will migrate from LFS to Xet
- All newly created repos will be Xet-enabled by default

If you upload or download from the Hub using your browser or use Git, that's fine. Chunk-based support for both is coming soon. In the meantime use whichever workflow you already have; no restrictions. 

Next up: open-sourcing the Xet protocol and the entire infrastructure stack. The future of storing and moving bytes that scale to AI workloads is on the Hub, and we're aiming to bring it to everyone.

If you have any questions, drop us a line in the comments 👇, [**open a discussion**](https://huggingface.co/spaces/xet-team/README/discussions/new) on the [**Xet team**](https://huggingface.co/xet-team) page.
