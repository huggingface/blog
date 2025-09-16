---
title: "`LeRobotDataset:v3.0`: Bringing large-scale datasets to `lerobot`"
thumbnail: /blog/assets/lerobot-dataset-v3/thumbnail.png
authors:
- user: fracapuano
- user: aractingi
- user: lhoestq
- user: CarolinePascal
- user: pepijn223
- user: jadechoghari
- user: rcadene
- user: aliberts
- user: AdilZtn
- user: nepyope
- user: imstevenpmwork
---

**TL;DR** Today we release `LeRobotDataset:v3`! In our previous `LeRobotDataset:v2` release, we stored one episode per file, hitting file-system limitations when scaling datasets to millions of episodes. `LeRobotDataset:v3` packs multiple episodes in a single file, using relational metadata to retrieve information at the individual episode level from multi-episode files. The new format also natively supports accessing datasets in streaming mode, allowing to process large datasets on the fly.We provide a one-liner util to convert all datasets in the LeRobotDataset format to the new format, and are very excited to share this milestone with the community ahead of our next stable release!

## Table of Contents

- [Install `lerobot`, and record a dataset](#install-lerobot-and-record-a-dataset)
- [The (New) Format Design](#the-new-format-design)
- [Acknowledgements](#acknowledgements)
- [Convert your dataset to v3.0](#convert-your-dataset-to-v30)
- [Code Example: Using `LeRobotDataset` with `torch.utils.data.DataLoader`](#code-example-using-lerobotdataset-with-torchutilsdatadataloader)
- [Wrapping up](#wrapping-up)

# LeRobotDataset, v3.0

`LeRobotDataset` is a standardized dataset format designed to address the specific needs of robot learning, and it provides unified and convenient access to robotics data across modalities, including sensorimotor readings, multiple camera feeds and teleoperation status.
Our dataset format also stores general information regarding the way the data is being collected (*metadata*), including a textual description of the task being performed, the kind of robot used and measurement details like the frames per second at which both image and robot state streams are sampled.
Metadata are useful to index and search across robotics datasets on the Hugging Face Hub!

Within `lerobot`, the robotics library we are developing at Hugging Face, `LeRobotDataset` provides a unified interface for working with multi-modal, time-series data, and it seamlessly integrates both with the Hugging Face and Pytorch ecosystems.
The dataset format is designed to be easily extensible and customizable, and already supports openly available datasets from a wide range of embodiments—including manipulator platforms such as the SO-100 arms and ALOHA-2 setup, real-world humanoid data, simulation datasets, and even self-driving car data!
You can explore the current datasets contributed by the community using the [dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)! 🔗

Besides scale, this new release of `LeRobotDataset` also enables support for a *streaming* functionality, allowing to process batches of data from large datasets on the fly, without having to download prohibitively large collections of data onto disk.
You can access and use any dataset in `v3.0` in streaming mode by using the dedicated `StreamingLeRobotDataset` interface!
Streaming datasets is a key milestones towards more accessible robot learning, and we are very excited about sharing this with the community 🤗

<div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
  <figure style="margin:0; text-align:center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/asset1datasetv3.png" alt="LeRobotDataset v3 diagram" width="200" />
    <figcaption style="font-size:0.9em; color:#666;">From episode-based to file-based datasets</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/streaming-lerobot.png" alt="StreamingLeRobotDataset" width="500" />
    <figcaption style="font-size:0.9em; color:#666;">We directly enable dataset streaming from the Hugging Face Hub for on-the-fly processing.</figcaption>
  </figure>
</div>

## Install `lerobot`, and record a dataset
[`lerobot`](https://github.com/huggingface/lerobot) is the end-to-end robotics library developed at Hugging Face, supporting real-world robotics as well as state of the art robot learning algorithms.
The library allows to record datasets locally directly on real-world robots, and to store datasets on the Hugging Face Hub.
You can read more about the robots we currently support [here](https://huggingface.co/docs/lerobot/)!

`LeRobotDataset:v3` is going to be a part of the `lerobot` library starting from `lerobot-v0.4.0`, and we are very excited about sharing it early with the community. You can install the latest `lerobot-v0.3.x` supporting this new dataset format directly from PyPI using:
```bash
pip install "https://github.com/huggingface/lerobot/archive/33cad37054c2b594ceba57463e8f11ee374fa93c.zip"  # this will affect your local lerobot installation!
```
Follow the community's progress towards a stable release of the library [here](https://github.com/huggingface/lerobot/issues/1654) 🤗

Once you have installed a version of lerobot which supports the new dataset format, you can record a dataset with our signature robot arm, the SO-101, by using teleoperation alongside the following instructions:
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube"
```
Head out the [official documentation](https://huggingface.co/docs/lerobot/en/il_robots?record=Command#record-a-dataset) to see how to record a dataset for your use case.

# The (New) Format Design

A core design choice behind `LeRobotDataset` is separating the underlying data storage from the user-facing API.
This allows for efficient serialization and storage while presenting the data in an intuitive, ready-to-use format. Datasets are organized into three main components:
1.  **Tabular Data**: Low-dimensional, high-frequency data such as joint states, and actions are stored in efficient [Apache Parquet](https://parquet.apache.org/) files, and typically offloaded to the more mature `datasets` library, providing fast, memory-mapped access or streaming-based access.
2.  **Visual Data**: To handle large volumes of camera data, frames are concatenated and encoded into MP4 files. Frames from the same episode are always grouped together into the same video, and multiple videos are grouped together by camera. To reduce stress on the file system, groups of videos for the same camera view are also broken into multiple subdirectories.
3.  **Metadata**: A collection of JSON files which describes the dataset's structure in terms of its metadata, serving as the relational counterpart to both the tabular and visual dimensions of data. Metadata includes the different feature schemas, frame rates, normalization statistics, and episode boundaries.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/dataset-v3-pillars.png)

To support datasets with potentially millions of episodes (resulting in hundreds of millions/billions of individual frames), we merge data from different episodes into the same high-level structure.
Concretely, this means that any given tabular collection and video will not contain information about one episode only, but a concatenation of the information available in multiple episodes.
This keeps the pressure on the file system manageable, both locally and on remote storage providers like Hugging Face.
We can then leverage metadata to gather episode-specific information, e.g. the timestamp a given episode starts or ends in a certain video.

Concretely, datasets are organized as repositories containing:

* **`meta/info.json`**: This is the central metadata file. It contains the complete dataset schema, defining all features (e.g., `observation.state`, `action`), their shapes, and data types. It also stores crucial information like the dataset's frames-per-second (`fps`), codebase version, and the path templates used to locate data and video files.
* **`meta/stats.json`**: This file stores aggregated statistics (mean, std, min, max) for each feature across the entire dataset. These are used for data normalization and are accessible via `dataset.meta.stats`.
* **`meta/tasks.jsonl`**: Contains the mapping from natural language task descriptions to integer task indices, which are used for task-conditioned policy training.
* **`meta/episodes/`**: This directory contains metadata about each individual episode, such as its length, corresponding task, and pointers to where its data is stored. For scalability, this information is stored in chunked Parquet files rather than a single large JSON file.
* **`data/`**: Contains the core frame-by-frame tabular data in Parquet files. To improve performance and handle large datasets, data from **multiple episodes are concatenated into larger files**. These files are organized into chunked subdirectories to keep file sizes manageable. Therefore, a single file typically contains data for more than one episode.
* **`videos/`**: Contains the MP4 video files for all visual observation streams. Similar to the `data/` directory, video footage from **multiple episodes is concatenated into single MP4 files**. This strategy significantly reduces the number of files in the dataset, which is more efficient for modern file systems. The path structure (`/videos/<camera_key>/<chunk>/file_...mp4`) allows the data loader to locate the correct video file and then seek to the precise timestamp for a given frame.


## Migrate your `v2.1` dataset to `v3.0`
`LeRobotDataset:v3.0` will be released with `lerobot-v0.4.0`, together with the possibility to easily convert any dataset currently hosted on the Hugging Face Hub to the new `v3.0` using:
```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30--repo-id=<HFUSER/DATASET_ID>
```
We are very excited about sharing this new format early with the community! While we develop `lerobot-v0.4.0`, you can still convert your dataset to the newly updated version by using the latest `lerobot-v0.3.x` supporting this new dataset format directly from PyPI using:
```bash
pip install "https://github.com/huggingface/lerobot/archive/33cad37054c2b594ceba57463e8f11ee374fa93c.zip"  # this will affect your lerobot installation!
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=<HFUSER/DATASET_ID>
```
Note that this is a pre-release, and generally unstable version. You can follow the status of the development of our next stable release [here](https://github.com/huggingface/lerobot/issues/1654)!

The convertion script `convert_dataset_v21_to_v30.py` aggregates the multiple episodes `episode-0000.mp4, episode-0001.mp4, episode-0002.mp4, ...`/`episode-0000.parquet, episode-0001.parquet, episode-0002.parquet, episode-0003.parquet, ...` into single files `file-0000.mp4`/`file-0000.parquet`, and updates the metadata accordingly, to be able to retrieve episode-specific information from higher-level files.

### Code Example: Using `LeRobotDataset` with `torch.utils.data.DataLoader`

Every dataset on the Hugging Face Hub containing the three main pillars presented above (Tabular and Visual Data, as well as relational Metadata), and can be accessed with a single line.

Most robot learning algorithms, based on reinforcement learning (RL) or behavioral cloning (BC), tend to operate on a stack of observations and actions.
For instance, RL algorithms typically use a history of previous observations `o_{t-H_o:t}`, and
BC algorithms are instead typically trained to regress chunks of multiple actions.
To accommodate for the specifics of robot learning training, `LeRobotDataset` provides a native windowing operation, whereby we can use the *seconds* before and after any given observation using a `delta_timestamps` argument.

Conveniently, by using `LeRobotDataset` with a PyTorch `DataLoader` one can automatically collate the individual sample dictionaries from the dataset into a single dictionary of batched tensors.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "yaak-ai/L2D-v3"

# Load from the Hugging Face Hub (will be cached locally)
dataset = LeRobotDataset(repo_id)

# Get the 100th frame in the dataset by index
sample = dataset[100]
print(sample)
# The sample is a dictionary of tensors
# {
#     'observation.state': tensor([...]),
#     'action': tensor([...]),
#     'observation.images. front_left': tensor([C, H, W]),
#     'timestamp': tensor(1.234),
#     ...
# }
delta_timestamps = {
    "observation.images.front_left": [-0.2, -0.1, 0.0]  # 0.2 and 0.1 seconds *before* any observation
}
dataset = LeRobotDataset(
    repo_id
    delta_timestamps=delta_timestamps
)

# Accessing an index now returns a stack of frames for the specified key
sample = dataset[100]

# The image tensor will now have a time dimension
# 'observation.images.wrist_camera' has shape [T, C, H, W], where T=3
print(sample['observation.images.front_left'].shape)

batch_size=16
# Wrap the dataset in a DataLoader to process it in batches for training purposes
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size
)

# 3. Iterate over the DataLoader in a training loop
num_epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    for batch in data_loader:
        # 'batch' is a dictionary where each value is a batch of tensors.
        # For example, batch['action'] will have a shape of [32, action_dim].

        # If using delta_timestamps, a batched image tensor might have a
        # shape of [32, T, C, H, W].

        # Move data to the appropriate device (e.g., GPU)
        observations = batch['observation.state.vehicle'].to(device)
        actions = batch['action.continuous'].to(device)
        images = batch['observation.images.front_left'].to(device)

        # Next do amazing_model.forward(batch)
        ...
```

## Streaming compatibility
You can also use any dataset in `v3.0` format in streaming mode, without the need to downloading it locally, by using the `StreamingLeRobotDataset` class.

```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# Streams directly from the Hugging Face Hub, without downloading the dataset into disk or loading into memory
repo_id = "yaak-ai/L2D-v3"
dataset = StreamingLeRobotDataset(repo_id)
```
## Wrapping up

`LeRobotDataset v3.0` is a stepping stone towards scaling up robotics datasets supported in LeRobot. By providing a format to store and access large collections of robot data we are making progress towards democratizing robotics, allowing the community to train on possibly millions of episodes without even downloading the data itself!

You can try the new dataset format by installing the latest `lerobot-v0.3.x`, and share any feedback [on GitHub](https://github.com/huggingface/lerobot/issues) or on our [Discord server](https://discord.gg/ttk5CV6tUw)! 🤗

## Acknowledgements
We thank the fantastic [yaak.ai](https://www.yaak.ai/) team for their precious support and feedback while developing LeRobotDataset:v3. 
Go ahead and [follow their organization](https://huggingface.co/yaak-ai) on the Hugging Face Hub!
We are always looking to collaborate with the community and share early features. [Reach out](mailto:francesco.capuano@huggingface.co) if you would like to collaborate 😊

