---
title: "Streaming `LeRobotDataset`: Train on large-scale data without pre-downloading"
thumbnail: /blog/assets/lerobot-dataset-v3/lerobot-dataset-v3.png
authors:
- user: fracapuano
- user: lhoestq
- user: rcadene
---

**TL;DR** We are introducing a streaming mode for `LeRobotDataset` that lets you iterate over massive robotics datasets without downloading them to disk first. It uses bounded buffers, random shard sampling, and on-the-fly video frame decoding to deliver high throughput with a small memory footprint. We also add native support for time-window queries via `delta_timestamps`, powered by an episode-aware, backtrackable iterator that can peek backward and forward efficiently.

## Table of Contents

- [What is streaming in LeRobot?](#what-is-streaming-in-lerobot)
- [Key features](#key-features)
- [StreamingLeRobotDataset](#streamingleRobotDataset)
- [Delta timestamps and the Backtrackable iterator](#delta-timestamps-and-the-backtrackable-iterator)
- [Assessing performance](#assessing-performance)
- [How to checkout and try](#how-to-checkout-and-try)
- [Wrapping up](#wrapping-up)

## What is streaming in LeRobot?

Training on real-world robotics datasets often means terabytes of multi-modal data. Fully downloading those datasets is slow and storage‑heavy. Streaming loads only what you need as you iterate, with optional random sampling across shards and efficient on-the-fly video decoding. The result is a familiar `IterableDataset` experience that scales to large datasets while remaining memory efficient.

## Key features

- **Memory‑efficient streaming**: Loads data on demand instead of all at once.
- **Buffer‑based sampling**: Maintains a bounded buffer of recent items for fast access.
- **Sharded loading**: Randomly samples across multiple shards for speed and variety.
- **Video frame decoding**: Streams only the frames you ask for, leveraging `torchcodec`.
- **Random sampling**: Ensures i.i.d.-like access patterns suitable for ML training.

## StreamingLeRobotDataset

The new class `StreamingLeRobotDataset` extends the standard dataset with streaming capabilities while keeping the public API simple and familiar.

```python
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

dataset = StreamingLeRobotDataset(
    repo_id="lerobot/aloha_mobile_cabinet",
    buffer_size=1000,
    max_num_shards=16,
)

for item in dataset:
    # Process streaming item
    pass
```

### Profiling helper

An opt‑in profiling script reports timing breakdowns and end‑to‑end throughput to help you size buffers and validate randomness during iteration.

```bash
python -m lerobot.common.datasets.profile_streaming_dataset \
  --repo-id lerobot/aloha_mobile_cabinet
```

## Delta timestamps and the Backtrackable iterator

Many algorithms need temporal context (e.g., multi‑frame observations or action chunks). Streaming now supports native time‑window queries using `delta_timestamps` and a lightweight backtrackable iterator.

### Enhanced Backtrackable iterator

- **Bidirectional buffering**: Separate buffers for history (`_back_buf`) and lookahead (`_ahead_buf`).
- **Configurable windows**: `history` and `lookahead` derived from `delta_timestamps` or set explicitly.
- **Memory efficient**: Caches only the frames strictly required by your query.
- **Episode‑aware**: Prevents crossing episode boundaries; returns padding and masks when needed.

```python
# Create iterator with 5‑frame history and 3‑frame lookahead
backtrackable = Backtrackable(dataset, history=5, lookahead=3)

# Access patterns
current = next(backtrackable)           # Move forward
previous = backtrackable.peek_back(1)   # Look back without moving
future = backtrackable.peek_ahead(2)    # Look ahead without moving
```

Core methods:

- `peek_back(n)`: Access n frames back without cursor movement
- `peek_ahead(n)`: Access n frames ahead, pre‑fetching if needed
- `can_peek_back/ahead()`: Check availability before access
- Episode boundary protection via custom exceptions

### Delta timestamps integration

- **Automatic validation**: Delta timestamp keys are checked against dataset features at init.
- **Temporal queries**: Use negative deltas for the past and positive for the future.
- **Padding handling**: Missing frames are zero‑padded; a `torch.Bool` mask is returned at `{key}.pad_masking`.
- **Video compatible**: Works seamlessly with multi‑timestep video decoding via `torchcodec`.

```python
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

delta_timestamps = {
    "observation.images.cam_right_wrist": [-0.1, -0.05, 0.0],  # 100ms, 50ms ago, current
    "action": [0.0, 0.02, 0.04],                               # current, +20ms, +40ms
}

dataset = StreamingLeRobotDataset(
    repo_id="lerobot/aloha_mobile_cabinet",
    delta_timestamps=delta_timestamps,
)

for item in dataset:
    # Each requested key includes a time dimension T
    print(item["action"].shape)         # e.g., (3, action_dim)
    print(item["action.pad_masking"])   # torch.tensor([...])
```

## Assessing performance

We used the built‑in profiling script to measure warmup, per‑sample latency, per‑run time, throughput, and access randomness on `lerobot/aloha_mobile_cabinet` with `buffer_size=1000` and up to 16 shards.

### Headline numbers

- **No delta timestamps**
  - Warmup: ~7.8 s
  - Per‑sample: 10.26 ± 154.29 ms (median 6.00 ms)
  - Per‑run: 28.59 ± 0.70 s
  - Throughput: **69.95 samples/s**

- **With delta timestamps (multi‑timestep retrieval)**
  - Warmup: ~15.06 s
  - Per‑sample: 21.91 ± 313.48 ms (median 13.69 ms)
  - Per‑run: 60.11 ± 0.91 s
  - Throughput: **33.27 samples/s**

The delta‑timestamps path roughly halves throughput, as expected, due to additional multi‑timestep video frame queries and padding/masking logic. Importantly, streaming still avoids pre‑downloading and keeps memory usage bounded.

### What dominates time

- Warmup time is largely due to initializing the `torchcodec` `VideoDecoder`.
- During iteration, hotspots are frame construction and video retrieval (the `make_frame` path), while buffer shuffling and shard switching remain lightweight.

### Randomness analysis

- Correlation between iteration index and dataset frame index: **0.0772** (values close to 0 indicate randomized access).
- The two‑level randomization (random shard selection + buffer shuffle) maintains this property both with and without delta timestamps.

### Practical tips

- Reuse dataloaders between epochs (avoid re‑initialization) to amortize warmup.
- Increase `buffer_size` until you see diminishing returns given your RAM/network.
- Tune `max_num_shards` to balance parallel fetch and remote overhead.
- For heavy multi‑timestep video access, prefer GPU‑side batch sizes that keep decoding pipelines saturated.

## How to checkout and try

1) Install dependencies (ensure versions compatible with the latest `torchcodec`):

```text
torch        2.7.0
torchaudio   2.7.0
torchcodec   0.4.0
torchvision  0.22.0
imageio      2.37.0  # for ffmpeg
```

2) Run the profiler to validate your setup:

```bash
python -m lerobot.common.datasets.profile_streaming_dataset \
  --repo-id lerobot/aloha_mobile_cabinet
```

3) Try delta timestamps in streaming:

```python
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

delta_timestamps = {
    "observation.images.cam_right_wrist": [-0.1, -0.05, 0.0],
    "action": [0.0, 0.02, 0.04],
}

dataset = StreamingLeRobotDataset(
    repo_id="lerobot/aloha_mobile_cabinet",
    delta_timestamps=delta_timestamps,
)

for item in dataset:
    # Stacked frames with masks for easier filtering
    pass
```

## Wrapping up

Streaming removes the download barrier for large robotics datasets while keeping training‑friendly properties like random sampling and low memory usage. With native `delta_timestamps` and an episode‑aware backtrackable iterator, it’s straightforward to retrieve temporal context for learning algorithms—all while decoding only the frames you actually use.

If you try streaming on your setup, share feedback and results in the GitHub repo. Happy training! 🚀


