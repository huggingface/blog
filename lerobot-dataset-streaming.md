---
title: "StreamingLeRobotDataset: Training on large-scale data without downloading"
thumbnail: /blog/assets/streaming-lerobot-dataset/thumbnail.png
authors:
- user: fracapuano
- user: lhoestq
- user: cadene
- user: aractingi
---

**TL;DR** We introduce streaming mode for `LeRobotDataset`, allowing users to iterate over massive robotics datasets without ever having to download them. `StreamingLeRobotDataset` is a new dataset class fully integrated with `lerobot` enabling fast, random sampling and on-the-fly video decoding to deliver high throughput with a small memory footprint. We also add native support for time-window queries via `delta_timestamps`, powered by a custom backtrackable iterator that steps both backward and forward efficiently. All datasets currently released in `LeRobotDataset:v3.0` can be used in streaming mode, by simply using `StreamingLeRobotDataset`.

## Table of Contents
- [Installing lerobot](#installing-lerobot)
- [Why Streaming Datasets](#why-streaming-datasets)
- [Using your dataset in streaming mode](#using-your-dataset-in-streaming-mode)
  - [Profiling helper](#profiling-helper)
- [Starting simple: streaming single frames](#starting-simple-streaming-single-frames)
- [Retrieving multiple frames: the "backtrackable" iterator](#retrieving-multiple-frames-the-backtrackable-iterator)
- [Conclusion](#conclusion)

## Installing `lerobot`

[`lerobot`](https://github.com/huggingface/lerobot) is the end-to-end robotics library developed at Hugging Face, supporting real-world robotics as well as state of the art robot learning algorithms.
The library allows to record datasets locally directly on real-world robots, and to store datasets on the Hugging Face Hub.
You can read more about the robots we currently support [here](https://huggingface.co/docs/lerobot/), and browse the thousands of datasets already contributed by the open-source community on the Hugging Face Hub [here 🤗](https://huggingface.co/datasets?modality=modality:timeseries&task_categories=task_categories:robotics&sort=trending).

We [recently introduced](https://huggingface.co/blog/lerobot-datasets-v3) a new dataset format enabling streaming mode. Both functionalities will ship with `lerobot-v0.4.0`, and you can access it right now building the library from source! You can find the installation instructions for lerobot [here](https://huggingface.co/docs/lerobot/en/installation).

## Why Streaming Datasets

Training robot learning algorithms using large-scale robotics datasets can mean having to process terabytes of multi-modal data.
For instance, a popular manipulation dataset like [DROID](https://huggingface.co/datasets/lerobot/droid_1.0.1/tree/main), containing 130K+ episodes amounting to a total of 26M+ frames results in 4TB of space: a disk and memory requirement which is simply unattainable for most institutions.

Moreover, fully downloading those datasets is slow and storage‑heavy, further hindering accessibility to the larger community.
In contrats, being able to stream chunks of a given dataset by processing it *online* provides a way to process large-scale robotics data regardless with very limited computational resources.
Streaming lets you load only what you need as you iterate through a large dataset, leveraging on-the-fly video decoding and the familiar `IterableDataset` interface used in Hugging Face datasets.

`StreamingLeRobotDataset` enables:
- **Disk & Memory‑efficient Access to Data**: Streams batches of data from a remote server and loads them onto memory rather than downloading and loading everything all at once.
- **Random sampling**: Learning from real-world trajectories collected by human demonstrator is a challenging task as it breaks the typical i.i.d. assumption made. Being able to randomly access frames mitigates this problem.
- **Time-windowing with `delta_timestamps`**: Most robot learning algorithms, based on either reinforcement learning (RL) or behavioral cloning (BC), tend to operate on a stack of observations and actions. To accommodate for the specifics of robot learning training, `StreamingLeRobotDataset` provides a native windowing operation, whereby we can use the *seconds* before and after any given observation using a `delta_timestamps` argument.

## Using your dataset in streaming mode

The new `StreamingLeRobotDataset` simply extends the standard `LeRobotDataset` with streaming capabilities, all while keeping the public API simple and familiar. You can try it with any dataset on the Hugging Face Hub simply by using:

```python
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

repo_id = "lerobot/droid_1.0.1" # 26M frames! Would require 4TB of disk space if downloaded locally (:
dataset = StreamingLeRobotDataset(repo_id)  # instead of LeRobotDataset(repo_id)

for frame in dataset:
    # Process the frame
    ...    
```

### Profiling helper

We assess the performance of our streaming datasets on two critical dimensions: (1) samples throughput (measured in fps) and (2) frame-index randomness.
Having high throughput in frames-per-second (fps) helps removing bottlenecks while processing the dataset, whereas high levels of randomness are 

We are very excited to share this first version of streaming, and cannot wait for the community to show us what they build on top of it.
You can always profile the performance of `StreamingLeRobotDataset` in terms of both fps and randomness by running:
```bash
python -m lerobot.scripts.profile_streaming--repo-id lerobot/svla_so101_pickplace  # change this with any other dataset
```
While we expect our randomness measurements to be robust across deployment scenarios, the samples throughput is likely going to vary depending on the connection speed.

## Starting simple: Streaming Single Frames
`StreamingLeRobotDataset` supports streaming mode for large dataset, so that frames (individual items within a dataset) can be fetched on-the-fly from a remote server instead of being loaded from a local disk.

[`LeRobotDataset:v3`](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3), the local version of the otherwise streaming-based dataset, stores information in:
- `data/*.parquet` files, containing tabular data representing robot controls and actions
- `videos/*.mp4` files, with video data for the capture of the dataset.
The dataset format also contains metadata files, which `StreamingLeRobotDataset` fully downloads to disk and loads in memory considering their typically negligible size (~100 MB for TBs of data).

Streaming frames is achieved by:
- Using the `IterableDataset` interface developed for the [`datasets` 🤗 library](https://huggingface.co/docs/datasets/en/stream) as a backbone for `LeRobotDataset`
- On-the-fly video decoding using the [`torchcodec`](https://docs.pytorch.org/torchcodec/stable/generated_examples/decoding/file_like.html) library


These two factors allow to step through an iterable, retrieving frames on the fly and exclusively locally via a series of `.next()` calls, without ever loading the dataset into memory.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/streaming-single-frame.png)

If were loading the dataset into memory, frames randomization could be achieved via indexing with shuffled indices.
However, because in streaming mode the dataset is only accessed iteratively via a series of `.next()` calls, one does not have random access to individual frames within the dataset, which would result in sequential-only access to the frames.
In other words, plotting the `index` of the retrieved frame alongside the iteration index would be a straight line, like:

<p>
<center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/iteration_index.png" width="300" />
</center>
</p>

Indeed, we can measure the correlation coefficient of the streamed `index` and the `iteration_index` to measure the randomness of the streaming procedure, where high levels of randomness correspond to a low (absolute) correlation coefficient and low levels of randomness result in high (either positive or negative) correlation.
In practice,
```python
from lerobot.dataset.streaming_dataset import StreamingLeRobotDataset

repo_id = "lerobot/svla_so101_pickplace"  # small, fits into memory and used for benchmarking
dataset = StreamingLeRobotDataset(repo_id)
dataset_iter = iter(dataset)

n_samples = 1_000  # the number of .next() calls
frame_indices = np.zeros(n_samples)
iter_indices = np.zeros(n_samples)

for i in range(n_samples):
  frame = next(dataset_iter)
  frame_indices[i] = frame["index"]
  iter_indices[i] = i

correlation = np.corrcoef(frame_indices, iter_indices)[0, 1]
print(correlation)
```

The image above, for instance, corresponds to a correlation coefficient of 1.0.

Low randomness when streaming frames is very problematic in those use cases where datasets are processed for training purposes.
In such context, items need to typically be shuffled so to mitigate the inherent inter-dependancy between successive frames recorded via demonstrations.
Similarily to the `datasets 🤗` library, we solve this issue maintaining a buffer of frames in memory, typically much smaller than the original datasets (1000s of frames versus 100Ms or 1Bs).
Storing this buffer in memory effectively allows for frames randomization via interleaving shuffling of the with `.next()` calls.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/random-from-iterable.png)

Because the `.next()` call for the dataset is now stacked on top of a process to fill in an intermediate buffer, an initialization overhead is introduced, to allow the buffer to be filled.
The smaller the size of the buffer, the lower the overhead introduced and randomization level. Conversely, larger buffer size correspond to higher levels of randomization at the expense of a bigger overhead consequential to having to fill a larger buffer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/streaming-buffers.png)

Typically, large datasets are stored in multiple files, which are accessed as multiple iterables to avoid having to load all of them in memory at ones. 
This can help introducing more randomness in the ordering of the frames stored by randomly sampling one of these iterables first to then feed the buffer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/random-from-many-iterables.png)

We benchmarked the throughput our streaming datasets against its non-streaming counterpart for a small-scale dataset that we can fully load into memory.
Streaming frames from memory instead than loading the entire dataset in memory has similar throughput (you can reproduce our results using the streaming profiler)!


## Retrieving multiple frames: the "backtrackable" iterator

Besides, single-frame straming, `StreamingLeRobotDataset` supports streaming mode for large dataset with the possibility to access multiple frames (individual items within the dataset) at the same time via the `delta_timestamps` argument.
When a dataset can be loaded in memory (`LeRobotDataset`) accessing multiple frames at once is fairly trivial: one can leverage random access to simply index the dataset and retrieve multiple frames at once.
However, when the dataset is not loaded into memory and instead iteratively processed via a sequence of `next()` calls, retrieving multiple frames is not necessarily as straightforward.
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/streaming-multiple-frames.png)

To solve this problem, we wrap the underlying dataset iterable with a custom iterable which we call [`Backtracktable`](https://github.com/huggingface/lerobot/blob/55e752f0c2e7fab0d989c5ff999fbe3b6d8872ab/src/lerobot/datasets/utils.py#L829), allowing for bidirectional access. 
Effectively, this iterable allows to efficiently retrieve frames both before and ahead of the current frame.

This custom iterable provides:
- *Bidirectional access*, having separate buffers for history (`_back_buf`) and lookahead (`_ahead_buf`) elements.
- *Episode‑aware* access, prevents crossing the episode boundaries enforcing consistency for the frames requested within an arbitrary episode.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/streaming-with-delta.png)

```python
from datasets import load_dataset
from lerobot.datasets.utils import Backtrackable
ds = load_dataset("c4", "en", streaming=True, split="train")
rev = Backtrackable(ds, history=3, lookahead=2)

x0 = next(rev)  # forward
x1 = next(rev)
x2 = next(rev)

# Look ahead
x3_peek = rev.peek_ahead(1)  # next item without moving internal cursor
x4_peek = rev.peek_ahead(2)  # two items ahead

# Look back
x1_again = rev.peek_back(1)  # previous item without moving internal cursor
x0_again = rev.peek_back(2)  # two items back

# Move backward
x1_back = rev.prev()  # back one step
next(rev)  # returns x2, continues forward from where we were
```

The backtracktable class has the following core methods:
- `peek_back(n)`: Access *n* frames back without stepping the underlying iterable, thereby maintaining a local cursor *fixed*
- `peek_ahead(n)`: Access *n* frames ahead, pre‑fetching if needed
- `can_peek_back()` and `can_peek_ahead()`: Check availability before access

When retrieving mulitple frames chaining `next()` calls within this custom iterable, one risks to cross episode's boundaries due to the lack of global information within each local `next()` call. Therefore, we find it is particularly important to add checks such as `can_peek_back()`/`can_peek_ahead()` to enforce the episode boundaries and avoid retrieving frames from different episodes.
When the requested frames are not available, the dataset-level next call returns all the available frames and padding frames for the unavailable positions, alongside a padding mask for downstream processing.

Similarily to `LeRobotDataset`, you can pass `delta_timestamps` to the class constructor.

```python
from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

delta_timestamps = {
    "action": [0.0, 0.02, 0.04],                               # current, +20ms, +40ms
}
repo_id = "lerobot/svla_so101_pickplace"  # small, fits into memory and used for benchmarking

dataset = StreamingLeRobotDataset(
    repo_id=repo_id,
    delta_timestamps=delta_timestamps,
)

for item in dataset:
    # Each requested key includes a time dimension T
    print(item["action"].shape)         # e.g., (3, action_dim)
    print(item["action.pad_masking"])   # torch.tensor([...])
```

The delta‑timestamps path roughly halves throughput, as expected, due to additional multi‑timestep video frame queries and padding/masking logic. Importantly, streaming still avoids pre‑downloading and keeps memory usage bounded.

Besides assessing throughput and randomness, you can also c-profiled the execution of our example on [how to train a dummy model in streaming mode](https://github.com/huggingface/lerobot/blob/main/examples/5_train_with_streaming.py) on `lerobot/droid`, a large scale manipulation dataset openly available on the Hugging Face Hub 🤗.

Profiling training, we find that the overall execution process is largely dominated by stepping through the `torch.utils.data.DataLoader`, which in turn we observed being mainly dominated by the buffer filling stage at initialization.
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/training-is-data-bound.png)

Indeed, while `next()` calls after the buffer has been filled exhibit similar performance to the one of a regular, memory-loaded dataset, initializing the buffer incurs in a significant overhead.
This is due to both the need to step through the dataset enough times to fill the buffer, and to initialize the connection to the `VideoDecoder` backend used to retrieve image frames on-the-fly.
As of now, this overhead can be partially mitigated by reducing the size of buffer only, which however has a negative impact on the level of randomness that can be achieved, and should therefore be tuned accordingly.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/buffer-impact-on-init-and-correlation.png)

You can reproduce our profiling findings with:
```bash
pip install snakeviz  # installs the profiler visualizer
python -m cProfile -o droid_training.prof examples/5_train_with_streaming.py
snakeviz droid_training.prof  # opens a localhost
```

## Conclusion

Streaming removes the download barrier for large robotics datasets while keeping training‑friendly properties like random sampling and low memory usage. With native multi-frame support and an episode‑aware backtrackable iterator, streaming mode provides a straightforward way to retrieve temporal context for learning algorithms, all while decoding exclusively the frames you actually use.

You can easily integrate the new streaming functionality in your setup with a one-line change, swapping your `LeRobotDataset` with a `StreamingLeRobotDataset`. 
We are very excited to share this feature with the community, and are eager to hear any feedback either on the [GitHub repo](https://github.com/huggingface/lerobot/issues) or in our [Discord server](https://discord.gg/ttk5CV6tUw).

Happy training 🤗


