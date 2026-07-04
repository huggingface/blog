---
title: "LeRobot v0.6.0: Imagine, Evaluate, Improve"
thumbnail: /blog/assets/lerobot-release-v060/thumbnail.png
authors:
  - user: imstevenpmwork
  - user: pepijn223
  - user: CarolinePascal
  - user: lilkm
  - user: Maximellerbach
  - user: nepyope
  - user: nikodembartnik
  - user: Nico-robot
  - user: thomwolf
---

# LeRobot v0.6.0: Imagine, Evaluate, Improve

This new release is about closing the robot learning loop: policies that imagine the future before acting, reward models that tell you when your robot succeeds, a deployment CLI that turns failures into training data, and six new simulation benchmarks to measure it all. It also brings depth sensing, VLM-powered dataset annotation, custom video encoding, cloud training on HF Jobs, and a much leaner install.

## TL;DR

LeRobot v0.6.0 introduces world model policies (VLA-JEPA, FastWAM, LingBot-VA) that learn to imagine the future, a wave of new VLAs (GR00T N1.7, MolmoAct2, EO-1, EVO1, Multitask DiT), and a new reward models API (Robometer, TOPReward). It ships six new simulation benchmarks unified under `lerobot-eval`, the `lerobot-rollout` CLI with DAgger-style human-in-the-loop corrections, FSDP training, and cloud training on HF Jobs. Datasets get depth support, an automatic language annotation pipeline, custom video encoding, and up to 2x faster data loading, all on top of a leaner installation.

![LeRobot 0.6.0](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/lerobot%20v0.6.0.png)


## Table of contents

- [LeRobot v0.6.0: Imagine, Evaluate, Improve](#lerobot-v060-imagine-evaluate-improve)
  - [TL;DR](#tldr)
  - [Table of contents](#table-of-contents)
  - [World models: policies that imagine](#world-models-policies-that-imagine)
    - [VLA-JEPA](#vla-jepa)
    - [FastWAM](#fastwam)
    - [LingBot-VA](#lingbot-va)
  - [VLAs: the model zoo keeps growing](#vlas-the-model-zoo-keeps-growing)
    - [GR00T N1.7](#gr00t-n17)
    - [MolmoAct2](#molmoact2)
    - [EO-1](#eo-1)
    - [Multitask DiT](#multitask-dit)
    - [EVO1](#evo1)
  - [Reward models: knowing when your robot succeeds](#reward-models-knowing-when-your-robot-succeeds)
    - [Robometer](#robometer)
    - [TOPReward](#topreward)
  - [Datasets: faster loading, richer data](#datasets-faster-loading-richer-data)
    - [Your codec, your rules](#your-codec-your-rules)
    - [Depth support, end to end](#depth-support-end-to-end)
    - [Language annotations at scale](#language-annotations-at-scale)
    - [Up to 2x faster data loading](#up-to-2x-faster-data-loading)
  - [Benchmarks: one CLI to evaluate them all](#benchmarks-one-cli-to-evaluate-them-all)
  - [Training \& inference](#training--inference)
    - [lerobot-rollout: deployment gets its own CLI](#lerobot-rollout-deployment-gets-its-own-cli)
    - [FSDP: train models bigger than your GPU](#fsdp-train-models-bigger-than-your-gpu)
    - [Cloud training with HF Jobs](#cloud-training-with-hf-jobs)
  - [Codebase: leaner and cleaner](#codebase-leaner-and-cleaner)
    - [A note on breaking changes](#a-note-on-breaking-changes)
  - [Community \& ecosystem](#community--ecosystem)
  - [Final thoughts](#final-thoughts)

## World models: policies that imagine

![LingBot-VA imagined rollout vs real rollout](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/lingbot_va_viz_1.gif)

The robotics world is asking a big question: do world models actually help robot policies? v0.6.0 brings three policies to LeRobot to help answer that question. Each one learns to imagine the future as part of its training, and each takes a different path to keep that imagination affordable.

### VLA-JEPA

![VLA JEPA Controlling a Robot with LeRobot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/vla-jepa.gif)

VLA-JEPA teaches a compact VLA (built on Qwen3-VL-2B) to predict the future in latent space while it learns to act: during training, a JEPA world model has to anticipate upcoming frames from the model's own actions. The trick is that the world model then disappears at inference, so you get world-model supervision at zero extra inference cost. Three ready-to-use checkpoints are on the Hub, including a DROID-pretrained base for fine-tuning:

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.repo_id=${HF_USER}/my_finetuned_policy
```

Check out the [VLA-JEPA documentation](https://huggingface.co/docs/lerobot/vla_jepa) and the [paper](https://arxiv.org/abs/2602.10098) to learn more.

### FastWAM

FastWAM asks the question in its paper title: do world action models need test-time future imagination? It pairs a ~5B video-generation expert with a compact action expert in a single network, so the model literally learns to dream its own rollouts. At inference it skips the dreaming entirely and directly denoises action chunks. Fine-tune it from [lerobot/fastwam_base](https://huggingface.co/lerobot/fastwam_base), and read more in the [documentation](https://huggingface.co/docs/lerobot/fastwam).

### LingBot-VA

LingBot-VA goes one step further: an autoregressive video-action model that predicts future video and actions together, chunk by chunk, and feeds real observations back in to keep its imagination grounded. You can even save what the robot imagined (`--policy.save_predicted_video=true`) and compare it with what actually happened, and inference runs on a single 24–32 GB GPU. Check out the [documentation](https://huggingface.co/docs/lerobot/lingbot_va) and the [paper](https://arxiv.org/pdf/2601.21998) for the technical details.

## VLAs: the model zoo keeps growing

### GR00T N1.7

![GROOT N1.7 in LeRobot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/groot.gif)

We upgraded our NVIDIA GR00T integration to GR00T N1.7, the newest open generation of NVIDIA's cross-embodiment foundation model. N1.7 swaps the previous VLM for Cosmos-Reason2-2B (built on Qwen3-VL) feeding a flow-matching action head, and our integration is parity-tested against NVIDIA's original Isaac-GR00T implementation: same inputs, same outputs. Flash-attention is now optional, so `pip install 'lerobot[groot]'` just works, and you can load [NVIDIA's published checkpoints](https://huggingface.co/nvidia/GR00T-N1.7-3B) directly.

> [!NOTE]
> GR00T N1.7 replaces N1.5 in LeRobot. If you need N1.5, pin `lerobot==0.5.1`.

### MolmoAct2

![MolmoAct2 Zero-Shot in LeRobot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/molmoact2_4.gif)

MolmoAct2, the Allen Institute for AI's vision-language-action model, is now ported into LeRobot with the full lifecycle covered: fine-tuning (full or LoRA), evaluation, and real-robot deployment. Ready-made checkpoints with calibration correction baked in mean you can run it zero-shot (no need to record a dataset!) on an SO-100/101:

```bash
lerobot-rollout \
  --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{cam0: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, cam1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --task="pick up the red cube" --duration=30
```

Inference fits in ~12 GB at bf16, and LoRA fine-tuning fits on a single 24 GB GPU. See the [MolmoAct2 documentation](https://huggingface.co/docs/lerobot/molmoact2) for the full deployment guide.

### EO-1

EO-1, a VLA pretrained upstream on interleaved vision-text-action data, joins LeRobot: a Qwen2.5-VL-3B backbone with a flow-matching action head, contributed by one of the paper's own authors. Train it with the standard `lerobot-train` workflow using `--policy.type=eo1`. Details in the [documentation](https://huggingface.co/docs/lerobot/eo1) and the [paper](https://arxiv.org/abs/2508.21112).

### Multitask DiT

The Multitask Diffusion Transformer policy brings the TRI Large Behavior Models recipe to LeRobot: a ~450M-parameter diffusion transformer conditioned on CLIP vision and language embeddings, so one model learns many tasks selected via natural language. It supports both diffusion and flow-matching objectives, and it is small enough to train yourself. See the [documentation](https://huggingface.co/docs/lerobot/multi_task_dit).

### EVO1

VLAs don't have to be huge. EVO1 packs its policy into 0.77B parameters, an InternVL3-1B backbone with a flow-matching action head, small enough to fine-tune and run in real time on modest GPUs. It ships with two-stage fine-tuning and Real-Time Chunking support out of the box. See the [EVO1 documentation](https://huggingface.co/docs/lerobot/evo1) and the [paper](https://arxiv.org/abs/2511.04555).

## Reward models: knowing when your robot succeeds

Success detection and progress estimation are the missing halves of the robot learning loop, and v0.6.0 gives them a home. LeRobot now has a unified reward models API (`lerobot.rewards`), mirroring the policies API, with four reward models behind one interface: the HIL-SERL reward classifier, SARM, and two new additions.

### Robometer

![LeRobot Robometer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/rm_robometer.gif)

Robometer is a pretrained, general-purpose reward model: point [lerobot/Robometer-4B](https://huggingface.co/lerobot/Robometer-4B) at any LeRobot dataset and it scores task progress and success from raw video plus a language instruction, with no task-specific training required. It is built on Qwen3-VL-4B and trained via trajectory comparisons over a dataset of more than one million robot trajectories ([RSS 2026 paper](https://arxiv.org/abs/2603.02115)).

### TOPReward

![LeRobot TOPReward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/rm_topreward.gif)

TOPReward goes fully zero-shot: no reward weights at all. It wraps an off-the-shelf VLM (Qwen3-VL) and reads the log-probability of the token "True" given the trajectory video and the task instruction. Any capable VLM becomes a reward function.

Both ship with labeling scripts that write per-frame progress curves into your dataset, ready for reward-aware behavior cloning (RA-BC), dataset quality inspection, and progress-overlay videos. Check the [Robometer](https://huggingface.co/docs/lerobot/robometer) and [TOPReward](https://huggingface.co/docs/lerobot/topreward) docs.

## Datasets: faster loading, richer data

### Your codec, your rules

Recording is no longer stuck with one hard-coded codec. The new `--dataset.rgb_encoder.*` options expose the full encoding surface (codec, quality, pixel format, GOP, presets), and `vcodec=auto` probes for hardware encoders like NVENC, VideoToolbox, VAAPI, and QSV before falling back to the default software AV1 encoder. For existing datasets, one command re-encodes everything:

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/my_dataset \
    --operation.type reencode_videos \
    --operation.rgb_encoder.vcodec h264 \
    --operation.rgb_encoder.crf 23
```

Full details in the [video encoding documentation](https://huggingface.co/docs/lerobot/video_encoding_parameters).

### Depth support, end to end

![LeRobot Depth Camera](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/depth.gif)

Plug in an Intel RealSense, set `use_depth: true`, and LeRobot records depth maps end to end: captured in millimeters, compressed as compact 12-bit depth video streams alongside your RGB cameras, and decoded back to physical units at training time. Depth renders live during recording and in `lerobot-dataset-viz`, and it works across SO-100/101, Koch, OpenArm, reBot, Unitree G1 and more.

### Language annotations at scale

Your dataset stops being one task string per episode. LeRobot datasets now natively store rich language annotations (timestamped subtasks, plans, memory, corrections, speech, and per-camera VQA pairs), and the new `lerobot-annotate` CLI fills them in automatically using a VLM that watches your episodes:

```bash
lerobot-annotate \
    --repo_id=${HF_USER}/my_dataset \
    --new_repo_id=${HF_USER}/my_dataset_annotated \
    --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct \
    --push_to_hub=true
```

A YAML recipe layer then renders these annotations into chat-style training messages at sample time: exactly the data tomorrow's long-horizon, talking robot policies will train on. Scale it up with HF Jobs, and read the [annotation pipeline docs](https://huggingface.co/docs/lerobot/annotation_pipeline) to learn more.

### Up to 2x faster data loading

Training on video datasets is now up to ~2x faster out of the box: multi-camera frames decode in parallel, dataloader workers ship compact uint8 frames (4x less memory between processes), and persistent workers keep decoder caches alive across epochs. Loading a subset of a large dataset (`episodes=[...]`) went from minutes to milliseconds (275 s down to 0.06 s in our benchmark). Sampling is also deterministic and resumable now, so interrupted trainings restart sample-exact.

## Benchmarks: one CLI to evaluate them all

<!-- TODO: gif idea: grid of rollouts across the six new benchmarks, hosted at documentation-images/lerobot-blog/release-v0.6.0/ -->

v0.5.0 planted the flag on LeRobot as an evaluation hub for VLAs; v0.6.0 makes it the real deal with six new simulation benchmarks, all runnable through the same `lerobot-eval` CLI, each with a docs page, a Docker image, and a SmolVLA baseline checkpoint smoke-tested in CI:

- [LIBERO-plus](https://huggingface.co/docs/lerobot/libero_plus) stress-tests VLAs with roughly 10,000 perturbed variants of LIBERO across seven axes, from lighting and camera viewpoints to rewritten instructions. It tells you when a policy breaks.
- [RoboTwin 2.0](https://huggingface.co/docs/lerobot/robotwin) covers 50 bimanual manipulation tasks on SAPIEN with heavy domain randomization, and comes with [more than 100k ready-to-train trajectories](https://huggingface.co/datasets/lerobot/robotwin_unified) on the Hub.
- [RoboCasa365](https://huggingface.co/docs/lerobot/robocasa) spans 365 kitchen tasks in 2,500 procedurally generated kitchens on a mobile manipulator, the largest task surface in our lineup.
- [RoboCerebra](https://huggingface.co/docs/lerobot/robocerebra) evaluates long-horizon behavior with episodes that chain 3 to 6 sub-goals under language-grounded intermediate instructions, plus a 6,660-episode dataset.
- [RoboMME](https://huggingface.co/docs/lerobot/robomme) is a memory exam: can your policy count repetitions, track hidden objects, and imitate demonstrated procedures? 16 tasks across 4 memory suites.
- [VLABench](https://huggingface.co/docs/lerobot/vlabench) tests knowledge and reasoning in manipulation, from physics questions to composite tasks like brewing coffee end to end.

```bash
lerobot-eval \
  --policy.path=lerobot/smolvla_robotwin \
  --env.type=robotwin \
  --env.task=beat_block_hammer \
  --eval.n_episodes=100 --eval.batch_size=1
```

Simulator backends are Linux affairs with their own install steps; each docs page has the exact recipe, and every benchmark ships a ready-made Docker image if you'd rather skip the setup.

Together with LIBERO, Meta-World, and NVIDIA IsaacLab-Arena, that makes nine benchmark families under one roof, and a new [Adding a New Benchmark guide](https://huggingface.co/docs/lerobot/adding_benchmarks) documents exactly how to plug in yours. Evaluation also got faster: parallel eval now defaults to async vectorized environments, benchmarked at up to 2x faster.

## Training & inference

### lerobot-rollout: deployment gets its own CLI

Deploying a policy used to be a hack on top of `lerobot-record`. The new `lerobot-rollout` CLI makes deployment its own workflow, with pluggable strategies and inference backends (including Real-Time Chunking for slow compatible VLAs). The `base` strategy just runs the policy. `sentry` records continuously, rotating episodes and uploading to the Hub as it goes. `highlight` keeps a ring buffer and saves the last N seconds when you hit a key, so an interesting moment is never lost. `episodic` mirrors the classic episode/reset recording workflow. And `dagger` turns deployment into data collection.

With the DAgger strategy, you watch your policy run, hit a key (or a USB foot pedal) the moment it goes wrong, take over with your leader arm to record the correction, and hand control back. Actuated leaders are driven to the follower's pose before you take over, so the handover is jerk-free. Every correction frame is tagged with an `intervention` flag, and the resulting dataset is ready for the next fine-tune:

```bash
lerobot-rollout \
    --strategy.type=dagger \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=${HF_USER}/dagger_corrections \
    --dataset.single_task="Grasp the block"
```

Deploy, collect corrections, fine-tune, repeat: the robot learning flywheel is now a CLI flag. Read the [deployment docs](https://huggingface.co/docs/lerobot/inference).

<!-- TODO: gif idea: DAgger takeover moment (policy fails, human grabs leader arm, correction recorded), hosted at documentation-images/lerobot-blog/release-v0.6.0/ -->

### FSDP: train models bigger than your GPU

Robot foundation models are outgrowing single GPUs. LeRobot training now supports FSDP (fully sharded data parallel) through Accelerate: parameters, gradients, and optimizer state are sharded across GPUs, and checkpoints are gathered back into a plain single-file `model.safetensors` that loads like any other policy. You can even resume an FSDP run on a different number of GPUs. See the [multi-GPU training docs](https://huggingface.co/docs/lerobot/multi_gpu_training).

### Cloud training with HF Jobs

No GPU? No problem. The exact same `lerobot-train` command now runs in the cloud by adding one flag:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/my_policy \
  --job.target=a10g-small
```

LeRobot pushes your local dataset to a private Hub repo if needed, submits the job, streams logs to your terminal (Ctrl-C detaches, the job keeps running), and pushes the trained policy to the Hub at the end. Pick anything from a T4 to 8x H200 with `--job.target` (compute is billed pay-as-you-go).

## Codebase: leaner and cleaner

- `pip install lerobot` is now genuinely lightweight, with roughly 40% fewer base dependencies. Feature-scoped extras (`[training]`, `[core_scripts]`, `[evaluation]`, ...) cover the rest, and missing-dependency errors tell you exactly which extra to add.
- Supported PyTorch moves to 2.7–2.11, with CUDA 12.8 wheels pinned out of the box for Linux `uv` installs. `--policy.dtype=bfloat16` now drives real mixed-precision training through Accelerate.
- A committed `uv.lock` is the authoritative dependency spec for CI, Docker, and development, and the docs include `uv` install routes for every step, down to picking your CUDA wheel with a single flag.
- `--display_mode=foxglove` streams teleoperation, recording, and rollouts to [Foxglove](https://foxglove.dev), the visualization tool much of the robotics world already uses. It works with remote setups, and `lerobot-dataset-viz` gets scrubbable dataset playback.
- Pip-installable `lerobot_env_*` packages now self-register their environments. The plugin system covers all five component types: robots, cameras, teleoperators, policies, and envs.
- Keyboard controls during recording now work on Wayland, over SSH, on headless rigs, and on macOS without Accessibility permissions.

### A note on breaking changes

v0.6.0 cleans house, and a few changes need your attention when upgrading:

- `pip install lerobot` no longer includes dataset or training dependencies; add the extra you need (e.g. `lerobot[training]`).
- GR00T N1.5 is replaced by N1.7 (pin `lerobot==0.5.1` if you need N1.5).
- The minimum PyTorch version is now 2.7.
- `eval_freq` was renamed to `env_eval_freq` in the train config.
- The RL stack was rebuilt: the `sac` policy type is now `gaussian_actor` under the new modular RL API.
- Legacy per-frame `subtask_index` annotations are superseded by the new language columns.
- `--dataset.vcodec` was renamed to `--dataset.rgb_encoder.vcodec`, such that RGB and depth cameras video codecs may be set separately.

Check the [release notes](https://github.com/huggingface/lerobot/releases) for the full list and migration pointers. <!-- TODO: link migration guide when published -->

## Community & ecosystem

- LeLab puts the whole LeRobot workflow (calibrate, teleoperate, record, train locally or on HF Jobs, deploy) in a browser UI, no CLI required. It currently supports the SO-ARM101. [Try it out!](https://github.com/huggingface/leLab)
- Isaac Teleop lets you teleoperate an SO-101 with a VR controller through [NVIDIA's Isaac Teleop stack](https://github.com/NVIDIA/IsaacTeleop) over CloudXR/OpenXR, the result of a collaboration with the NVIDIA team. See the [documentation](https://huggingface.co/docs/lerobot/isaac_teleop).
- The new [compute hardware guide](https://huggingface.co/docs/lerobot/hardware_guide) answers the two questions every newcomer asks: which GPU do I need, and how long will training take? It gives measured VRAM envelopes per policy family and reference training times from an RTX 4090 to 4x H100.
- The rewritten [Adding a Policy guide](https://huggingface.co/docs/lerobot/bring_your_own_policies) shows how to ship your own policy, in-tree or as a plugin package with no PR needed.

## Final thoughts

Beyond these headline features, v0.6.0 includes hundreds of bug fixes, documentation improvements, and quality-of-life upgrades across the codebase, from smarter defaults to more reliable CI.

We want to extend a huge thank you to everyone in the community. This release includes work from academia, industry and hobbyists teams who chose LeRobot as the home for their models and benchmarks. Every PR and bug report pushes open-source robotics forward.

Stay tuned for more to come 🤗 Get started [here](https://github.com/huggingface/lerobot)!
– The LeRobot team ❤️
