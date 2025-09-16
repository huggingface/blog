---
title: "`LeRobotDataset-v3.0`: Bringing large-scale datasets to lerobot"
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

**TL;DR** Today we release `LeRobotDataset:v3`! With `LeRobotDataset:v2` we stored one episode per file, and therefore hit file system limitations when scaling datasets to the million-episode scale. `LeRobotDataset:v3` packs multiple episodes in a single file, and uses relational metadata to retrieve information at the individual episode level from multi-episode files, allowing to scale robotics datasets in the `LeRobotDataset` format.

## Table of Contents

## Table of Contents

- [Install `lerobot`, and record a dataset](#install-lerobot-and-record-a-dataset)
- [The (New) Format Design](#the-new-format-design)
- [Acknowledgements](#acknowledgements)
- [Convert your dataset to v3.0](#convert-your-dataset-to-v30)
- [Code Example: Using `LeRobotDataset` with `torch.utils.data.DataLoader`](#code-example-using-lerobotdataset-with-torchutilsdatadataloader)
- [Wrapping up](#wrapping-up)

# LeRobotDataset, v3.0

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/asset1datasetv3.png" alt="LeRobotDataset v3 diagram" width="300" />
</p>

`LeRobotDataset` is a standardized dataset format designed to address the specific needs of robot learning, and it provides unified and convenient access to robotics data across modalities, including sensorimotor readings, multiple camera feeds and teleoperation status.
`LeRobotDataset` also stores general information regarding the data collected, like the task being performed by the teleoperator, the kind of robot used and measurement details like the frames per second at which both image and robot state streams are recorded.

`LeRobotDataset` provides a unified interface for working with multi-modal, time-series data, seamlessly integrating with both the PyTorch and Hugging Face ecosystems. 
The dataset format is designed to be easily extensible and customizable, and already supports openly available datasets from a wide range of embodiments—including manipulator platforms such as the SO-100 and ALOHA-2, real-world humanoid data, simulation datasets, and even self-driving car data!

The format is optimized for efficient training while remaining flexible enough to handle the diverse data types common in robotics, all while promoting reproducibility and ease of use.
You can explore the current datasets contributed by the community using using the [dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)! 🔗

### Install `lerobot`, and record a dataset
With `lerobot`, you can record and store on the Hugging Face Hub datasets collected on a variety of real-world robots.
Read more about the robots we currently support [here](https://huggingface.co/docs/lerobot/).

`LeRobotDataset-v3` is going to be a part of the `lerobot` library starting from `lerobot-v0.4.0`, and we are very excited about sharing it early with the community. You can install the latest `lerobot-v0.3.x` supporting this new dataset format directly from PyPI using:
```
pip install "https://github.com/huggingface/lerobot/archive/847e74f62827253507dd7aca18da028596811f31.zip"
```
Follow the community's progress towards a stable release of the library [here](https://github.com/huggingface/lerobot/issues/1654) 🤗
Once you have installed a version of lerobot which supports the new dataset format, you can record a dataset for your SO-101 arm by using teleoperation alongside the following instructions:
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

Check out the [official documentation](https://huggingface.co/docs/lerobot/en/il_robots?record=Command#record-a-dataset) to see how to record a dataset for your use case!


## The (New) Format Design

A core design choice behind `LeRobotDataset` is separating the underlying data storage from the user-facing API.
This allows for efficient serialization and storage while presenting the data in an intuitive, ready-to-use format.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobotdataset-v3/dataset-v3-pillars.png)

A dataset is organized into three main components:

1.  **Tabular Data**: Low-dimensional, high-frequency data such as joint states, and actions are stored in efficient [Apache Parquet](https://parquet.apache.org/) files, and typically offloaded to the more mature `datasets` library, providing fast, memory-mapped access.
2.  **Visual Data**: To handle large volumes of camera data, frames are concatenated and encoded into MP4 files. Frames from the same episode are always grouped together into the same video, and multiple videos are grouped together by camera. To reduce stress on the file system, groups of videos for the same camera view are also broken into multiple subdirectories, after a given threshold number.
3.  **Metadata**: A collection of JSON files which describes the dataset's structure in terms of its metadata, serving as the relational counterpart to both the tabular and visual dimensions of data. Metadata includes the different feature schemas, frame rates, normalization statistics, and episode boundaries.

For scalability, and to support datasets with potentially millions of trajectories resulting in hundreds of millions or billions of individual camera frames, we merge data from different episodes into the same high-level structure.
Concretely, this means that any given tabular collection and video will not contain information about one episode only, but a concatenation of the information available in multiple episodes.
This keeps the pressure on the file system---both locally and on remote storage providers like Hugging Face---manageable, at the expense of leveraging more heavily the metadata, e.g., used to reconstruct information relative to at which position a given episode starts or ends.

*   **`meta/info.json`**: This is the central metadata file. It contains the complete dataset schema, defining all features (e.g., `observation.state`, `action`), their shapes, and data types. It also stores crucial information like the dataset's frames-per-second (`fps`), codebase version, and the path templates used to locate data and video files.
*   **`meta/stats.json`**: This file stores aggregated statistics (mean, std, min, max) for each feature across the entire dataset. These are used for data normalization and are accessible via `dataset.meta.stats`.
*   **`meta/tasks.jsonl`**: Contains the mapping from natural language task descriptions to integer task indices, which are used for task-conditioned policy training.
*   **`meta/episodes/`**: This directory contains metadata about each individual episode, such as its length, corresponding task, and pointers to where its data is stored. For scalability, this information is stored in chunked Parquet files rather than a single large JSON file.
*   **`data/`**: Contains the core frame-by-frame tabular data in Parquet files. To improve performance and handle large datasets, data from **multiple episodes are concatenated into larger files**. These files are organized into chunked subdirectories to keep file sizes manageable. Therefore, a single file typically contains data for more than one episode.
*   **`videos/`**: Contains the MP4 video files for all visual observation streams. Similar to the `data/` directory, video footage from **multiple episodes is concatenated into single MP4 files**. This strategy significantly reduces the number of files in the dataset, which is more efficient for modern file systems. The path structure (`/videos/<camera_key>/<chunk>/file_...mp4`) allows the data loader to locate the correct video file and then seek to the precise timestamp for a given frame.


## Acknowledgements
We thank the fantastic [yaak.ai](https://www.yaak.ai/) team for their precious support and feedback while developing LeRobotDataset-v3. 
Go ahead and [follow their organization](https://huggingface.co/yaak-ai) on the Hugging Face Hub! 🤗
We are always looking to collaborate with the community and share early features. [Reach out](mailto:francesco.capuano@huggingface.co) if you would like to collaborate 😊

## Convert your dataset to v3.0
`LeRobotDataset-v3.0` will be released with `lerobot-v0.4.0`, together with the possibility to easily convert any dataset you currently host on the Hugging Face Hub to the new `v3.0` using:
```bash
python lerobot.datasets.v30.convert_dataset_v21_to_v30.py --repo-id=<HFUSER/DATASET_ID>
```
We are very excited about sharing this new format early with the community! While you wait for `lerobot-v0.4.0`, you can still convert your dataset to the newly updated version by using the latest `lerobot-v0.3.x` supporting this new dataset format directly from PyPI using:
```bash
pip install "https://github.com/huggingface/lerobot/archive/847e74f62827253507dd7aca18da028596811f31.zip"  # this will affect your lerobot installation!
python lerobot.datasets.v30.convert_dataset_v21_to_v30.py --repo-id=<HFUSER/DATASET_ID>
```
Note that this is a pre-release, generally unstable version. You can follow the status of the `lerobot-v0.4.0` release [here](https://github.com/huggingface/lerobot/issues/1654)!
Once you have installed a version of lerobot supporting the new dataset format, you can convert any dataset to the new version by running:

The migration script aggregates the multiple episodes `episode-0000.mp4, episode-0001.mp4, episode-0002.mp4, ...`/`episode-0000.parquet, episode-0001.parquet, episode-0002.parquet, episode-0003.parquet, ...` into single files `file-0000.mp4`/`file-0000.parquet`, and updates the metadata accordingly to be able to retrieve episode-specific information from higher-level files.

## Code Example: Using `LeRobotDataset` with `torch.utils.data.DataLoader`

Every dataset on the Hugging Face Hub containing the three main pillars presented above (Tabular and Visual Data, as well as relational Metadata) can be accessed with a single line.
Most reinforcement learning (RL) and behavioral cloning (BC) algorithms tend to operate on a stack of observations and actions.
For instance, RL algorithms typically use a history of previous observations `o_{t-H_o:t}` to mitigate partial observability.
BC algorithms are instead typically trained to regress chunks of multiple actions rather than single controls.
To accommodate for the specifics of robot learning training, `LeRobotDataset` provides a native windowing operation, whereby we can use the *seconds* before and after any given observation using `delta_timestamps`. 
Unavailable frames are padded, and the padding mask is provided to remove these from training.
Notably, this all happens within the `LeRobotDataset` and is entirely transparent to higher-level wrappers such as `torch.utils.data.DataLoader`.

Conveniently, by using `LeRobotDataset` with a PyTorch `DataLoader` one can automatically collate the individual sample dictionaries from the dataset into a single dictionary of batched tensors.

```python
from lerobot.datasets import LeRobotDataset

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
#     'observation.images.wrist_camera': tensor([C, H, W]),
#     'timestamp': tensor(1.234),
#     ...
# }
delta_timestamps = {
    "observation.images.wrist_camera": [-0.2, -0.1, 0.0]  # 0.2 and 0.1 seconds *before* any observation
}
dataset = LeRobotDataset(
    repo_id
    delta_timestamps=delta_timestamps
)

# Accessing an index now returns a stack of frames for the specified key
sample = dataset[100]

# The image tensor will now have a time dimension
# 'observation.images.wrist_camera' has shape [T, C, H, W], where T=3
print(sample['observation.images.wrist_camera'].shape)

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
        observations = batch['observation.state'].to(device)
        actions = batch['action'].to(device)
        images = batch['observation.images.wrist_camera'].to(device)

        # Next do amazing_model.forward(batch)
        ...
```
## Wrapping up

`LeRobotDataset v3.0` is a stepping stone towards scaling up robotics datasets supported in LeRobot. 
We achieve this by:
- packing many episodes per file with relational metadata for precise indexing
- enabling native frame and delta-frame retrieval for temporal context
- allowing aggregation of multiple sources into a single logical dataset

Try it now by installing the latest `lerobot`, record or convert a dataset, and share any feedback on GitHub `https://github.com/huggingface/lerobot/issues` 🎯