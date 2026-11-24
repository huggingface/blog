---
title: "Taking Trainium3 to the Track: Generating Videos with LTX-2.5"
thumbnail: /blog/assets/video-trainium3/thumbnail.jpg
authors:
- user: dacorvo
---

# Taking Trainium3 to the Track: Generating Videos with LTX-2.5

Thanks to our partnership with AWS, I was lucky enough to get access to a Trainium3 machine for a month. I spent most of the month testing Hugging Face models on it, and getting more of them to work (up to GLM-5, all 754B parameters of it, on a single instance).

With five days left, I allowed myself a little fun. When you have a Ferrari, every now and then you want to take it out on the track. So I asked myself: what if I tried to generate videos?

I picked [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) from Lightricks, which generates video and sound together. Here is the result.

<video controls playsinline preload="metadata" width="100%" aria-label="The seal film, made with LTX-2.5 on Trainium3" src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/entertainment/seal.mp4"></video>

## What I wanted to check

I had two goals. The first was to see how easy it is to generate videos with [diffusers](https://github.com/huggingface/diffusers) on Trainium3. Not the simple clips from the examples: I wanted to make short films from a script, closer to a real use case, like drafting a movie. The second was performance. I wanted to get close to what you get with CUDA, or with LTX's own API.

For CUDA, I found no real figure to compare against. The only one is LTX's claim of 6.8 s for a 10-second 720p clip on four B200 GPUs, with no detail on how it was measured. So I set CUDA aside and compared with LTX's online API instead, which LTX says makes a 10-second 1080p clip in 23.7 s. In the end, we match it, and even beat it by about a second: 22 to 23 s for a clip of the same size, once the models are loaded, on half of the machine. Not bad, considering how little time I had to optimize the pipelines.

And it was relatively simple. I only steered a coding agent, with fairly high-level directions. My films are an amateur's, and I'm quite happy with them. The tools behind them, though, are close to production grade.

## How it works

The input is a film: a JSON list of shots, each with a name and a prompt, much like a storyboard. LTX-2.5 does better with long prompts, so each prompt is a paragraph on the style, the subject, the motion, the camera, the light and the sound. A shot can start from the last frames of an earlier shot. From its last frame, a shot comes back to a subject after a cut, with the same look (prompts alone do not keep it). From its last 25 frames (1 s), a shot continues its motion. A shot can also end on the first frames of a later shot, so that it leads into it. Finally, a shot can have its own seed, to keep the take I picked among several, and it can be trimmed or have a caption.

Making a film from shots works like animation, which has two ways to go. Straight ahead, each shot continues the one before. That is how I made the fox film, a walk in the snow with nothing to build up to. Pose to pose, you make the key shot first, until you get one you like. Then the shots before it end on its first frames, and the shots after it start from its last frames. When a film builds up to a payoff, the payoff is the key shot. I found this order by trial and error, and made the seal film that way. Here are its four shots, with their prompts shortened:

```json
[
  {"name": "dream", "prompt": "Style: cinematic, realistic, shallow depth of field. A plump seal ... is fast asleep on warm golden sand under palm trees ...",
   "seed": 3},
  {"name": "nap", "prompt": "Style: cinematic, realistic, wide-angle lens. ... a plump grey seal is fast asleep on a small ice floe ...",
   "seed": 1, "to": {"shot": "iceberg", "frames": 9}},
  {"name": "iceberg", "prompt": "Style: cinematic, realistic, wide-angle lens. ... the whole iceberg slowly rolls over ... and washes over the seal. ..."},
  {"name": "shake", "prompt": "Style: cinematic, realistic, wide-angle lens. ... It slowly shakes its head and body ... and flops down on its belly on the floe, ...",
   "from": {"shot": "iceberg", "frames": 25}}
]
```

The payoff is `iceberg`: the iceberg rolls over, and its wave drenches the seal. `nap`, the sleeping seal, ends on its first 9 frames, so that the seal, the floe and the iceberg stay the same into the payoff. `shake` starts from its last 25 frames: the dripping seal shakes off the water and flops down on the floe. The `dream` is made on its own, and a cut joins it to the rest.

On the diffusers side, three pipelines share the models of [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers). `LTX2Pipeline` makes the shots from their prompt alone. `LTX2ConditionPipeline` makes the shots that start or end on frames of another shot. Both follow the two-stage distilled recipe of the model card: 8 steps at 1024x512, then `LTX2LatentUpsamplePipeline` upscales the latents x2, then 3 steps at 2048x1024.

The diffusers pipelines do all of the generation. The rest, reading the film file and editing the film, is about 160 lines of Python. The code is in [dacorvo/video-generation](https://github.com/dacorvo/video-generation). The films I kept are in the [dacorvo/video-generation](https://huggingface.co/buckets/dacorvo/video-generation/tree/entertainment) bucket, with their JSON files: the seal, the fox, and the goat, which trims two of its shots and draws a caption over its payoff.

## Optimizing for Trainium3

The vanilla pipeline took 12 min 22 s for a 10-second clip on the 4 NeuronCores of one chip, and the NeuronCores computed for about 45 s of it. Vanilla here means `LTX2Pipeline` as the model card runs it, with only what it takes to run on the NeuronCores: the transformer compiled with `torch.compile(backend="neuron")`, and split over 4 cores by tensor parallelism, as it does not fit on one. A 10-second clip at 1024x512 now takes 131 s on the same 4 cores. The 2048x1024 clip, out of reach of the vanilla pipeline, takes 22 to 23 s on 32 cores once the models are loaded, from the prompt to the mp4. Getting there took three kinds of work: making the model run on the NeuronCores, splitting it over more of them, and making the host keep up with them.

### What worked out of the box

- The code of the diffusers pipelines, which runs on the host and calls the transformer compiled for the NeuronCores.
- Regional compilation: diffusers' `compile_repeated_blocks` compiles one graph for the 48 blocks of the transformer.
- Loading a model split by tensor parallelism, which diffusers does for any model that has a plan (`_tp_plan`).
- `torch.distributed` on the NeuronCores, within a chip and across chips: device meshes, DTensors, all-to-all and all-gather.

### What needed a workaround

| Problem | Workaround | Gain |
|---|---|---|
| The LTX-2.5 transformer has no tensor-parallel plan in diffusers | A plan, and an attention processor that runs on the heads of one rank | The transformer runs on 4 cores |
| torch-neuronx computes a bf16 gelu entirely in bf16, constants included ([#1049](https://github.com/aws-neuron/torch-neuronx/issues/1049)) | Compute the gelus in fp32 | Transformer error against fp32: 0.297 → 0.158 (CPU bf16: 0.160) |
| Each rank (one process per NeuronCore) ran the whole pipeline, and they all encoded the prompt on the CPUs of one socket | Run the pipeline on rank 0 only, and send the other ranks the arguments of each call | Prompt encoding: about 205 s → 7.5 s |
| Each new process traces and lowers the whole transformer again, even with every graph in the cache | Compile one block with `compile_repeated_blocks` | First call of the transformer: 388 s → 18 s |
| torch-neuronx's allocator leaves most of its memory segments unused ([#1052](https://github.com/aws-neuron/torch-neuronx/issues/1052)) | Pack the parameters into buffers of at most 1 GiB | Memory per core: 22.96 GiB → 10.2 GiB |
| The attention runs as plain ops unless its length is a multiple of 128 tokens | Generate at 1024x512 (512 tokens per frame), with `NEURON_LOGICAL_NC_CONFIG=2` | Transformer step: 4.3 s → 3.0 s |
| The video decode is slow on CPU, and more CPU cores do not help | Decode on the NeuronCores, split by frames, and pass the frames to rank 0 through shared memory | Decode at 2048x1024: 357 s → 1.6 s |
| At 2048x1024, the transformer block split over the 4 cores of a chip fails to load | Split the video tokens over 8 chips (sequence parallelism) | Stage 2 runs on 32 cores, at 3.0 s a step |
| A strided slice copied to the device is filled by a slow copy on the device ([#1050](https://github.com/aws-neuron/torch-neuronx/issues/1050)) | `.contiguous()` before the copy | Step on 16 cores: 5.6 s → 1.0 s |
| Each of the 32 ranks runs as many torch threads as its NUMA node has CPUs, and rank 0, which runs the text encoder, the upsampler and the vocoder, shares them with the others | Give rank 0 CPUs of its own, and the other ranks their share of threads | Text encoder: 7.4 s → 3.3 s. Stage 1 step: 0.9 s → 0.6 s |
| The pipeline pads the prompt to 1024 tokens, and the text encoder, on CPU, encodes the padding too | Run the text encoder on the tokens of the prompt only | Prompt encoding: 3.4 s → 1.4 s |

Three issues were never explained: the transformer compiled as one graph returned NaN at one size, the first shot that ends on keyframes was slow in each new process (even with every graph in the cache), and the latent upsampler fails to load next to the transformer, so it stays on the CPU.

## Lessons learned

- **Most of the time went around the compute, not into it.** The NeuronCores computed for about 45 s of the vanilla run's 12 min 22 s. Most of the fixes removed host work, tracing and copies. Even at the end, the transformer runs for about 13 s of the 22 s it takes to make a clip. The last gains came from the host: rank 0 got CPUs of its own, the frames reach it through shared memory, and it makes the sound while the cores decode the frames.
- **On Trainium, plan for the size of the graphs.** The shapes decide whether the attention takes the fast kernel, and graphs that are too large fail to load. Splitting the tokens over more chips made them small enough.
- **Check against the CPU first.** Each split of the models was checked on CPU before it ran on the device, and the comparison with CPU found the gelu issue.
- **Keep the traced code stable.** In the build I used, torch-neuronx's cache key included the file and line of each traced op, so an edit could trigger a recompile of up to 30 minutes per graph. I had already reported it ([aws-neuron/torch-neuronx#1027](https://github.com/aws-neuron/torch-neuronx/issues/1027)), and the issue is now closed.
- **A film needs more than prompts.** Prompts alone do not keep a character, or a mountain, from one shot to the next, so shots start or end on frames of other shots. Each shot also stays in one place: when a shot in an early version of the fox film took the fox from the forest to the lake, the model made a cross-dissolve to another fox.
- **Make several takes of each shot.** The model often gets a detail wrong: the same prompt gave the goat funny horns in 3 takes out of 4. I made 4 takes of each new shot and picked one.

The directions I gave the agent were mostly principles: make the best use of the time on the instance, and favor sequence parallelism over wider tensor parallelism.

## Professional use cases

On the performance side, once its models are loaded, one process makes a 10-second 2048x1024 clip with sound in 22 to 23 s. That rivals LTX's paid API, on half of the instance. With its 32 attention heads, the transformer splits over 32 NeuronCores at most, so the other half could make a second clip at the same time (I did not measure it). A service would keep its processes up, as a process takes about 3 minutes to load the models, and 80 s for its first clip.

On the feature side, diffusers has what it takes to draft a movie. The shots come from prompts, and keep their subject from one shot to the next by starting or ending on frames of other shots. Takes differ by their seed, and the film file keeps the one I picked. Run again with the same code, a film file makes the same film, byte for byte. A shot can also start from an image, such as a storyboard frame or a character sheet, which the condition pipeline takes the same way as frames (I did not try it in a film).

What keeps it at drafts:

- The model still gets details wrong, so each new shot needs several takes.
- Each shot makes its own sound. The film evens out the background level, but the sound does not carry over from one shot to the next.
- A shot lasts at most 20 s, the model's limit.

LTX calls LTX-2.5 a world model, with robotics as an emerging use. Robots, cars and drones learn from videos of what their cameras see, and a world model can make more of them, including events too rare or too risky to film. So I also tried it on real footage, with the same code: a shot can start from the last second of a video filmed by a camera. One frame was not enough, as the model cannot tell from it how fast things move. Here is what the front camera of a car filmed in a snowstorm ([Boreas](https://www.boreas.utias.utoronto.ca/)), next to the video generated from its last second, 28 s in three shots:

<video controls playsinline preload="metadata" width="100%" aria-label="Real footage from Boreas next to the video generated from its last second" src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/robotics/boreas-snow/happened-side-by-side.mp4"></video>

From the same second, the model also makes what could have happened, such as slush thrown on the lens by a passing truck. The [robotics](https://huggingface.co/buckets/dacorvo/video-generation/tree/robotics) folder of the bucket has this example, and a second one from the camera of a humanoid robot ([GR1-100](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1)), where a person helps the robot carry an orange. Here the prompts drive the videos, not the commands of a robot or a car, which a world model for training follows. For that, a team fine-tunes the [pre-trained checkpoint of LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Pre-Trained) on the frames and actions of its own robots.

The Ferrari made it around the track, on half of its cylinders.


