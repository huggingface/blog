---
title: "The model that didn't exist, so you made it yourself"
thumbnail: /blog/assets/building-with-ml-intern/thumbnail.png
authors:
- user: ysharma
- user: abidlabs
---

# The model that didn't exist, so you made it yourself 

<p align="center">
<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/ml_intern_intro_noaudio_cropped.mp4" width="400" ></video>
</p>

Last week, I wanted a small version of the [prompt rewriter](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) that ships with Qwen-Image 2.1. The official one is a 9B model that needs about 20 GB of memory and thinks for thousands of tokens before writing a single paragraph. On the Hub, I found only compressed copies of that same 9B model. So I described what I wanted to [ML Intern](https://huggingface.co/chat/?mode=ml-intern), and the next day I had a [0.8B version](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B) that runs on a CPU. It returns valid output 99.7% of the time and uses about a quarter of the teacher's tokens. The compute for the whole project, including having the 9B model label 8,797 example requests, came to USD 16.

Over the course of the next few days, I made five more models the same way. Each one started as a message in HuggingChat with ML-intern switched on, and each one ended as a public model on the Hub with its evaluation in the model card. ML-intern plans the work, asks me for a budget before it spends anything, runs a small test before the real job, then trains, evaluates and publishes on Hugging Face hardware.

## How I write the first message

The first message is where I spend my effort. My first prompt, for the *citrus* model shared below, was about 450 words. By my 6th project it was closer to 2,000, because each project taught me something I wanted in the next one. All seven prompts are on GitHub at [yvrjsharma/ml-intern-prompts](https://github.com/yvrjsharma/ml-intern-prompts), exactly as I wrote them.

A prompt starts with the idea in one line and why I want it. Then it names the exact pieces: the dataset, the base model, the training script. Anything I have already checked goes under a heading that literally says "Verified facts, do not re-derive", so the agent spends its budget on the work instead of rediscovering what I know. For the camera-angle LoRA that section listed which trainer had just added transparent-image support, and which open GitHub issues made the fallback trainer risky.

**Two lines in the prompt are critical**. The **first** asks for a baseline before any training. For example, the _citrus prompt_ says: "Also report the base model's zero-shot score on the same metric before training so we can see the gain." Without it you get a trained model and no idea whether it is better than what you started with. The **second** is a smoke test with a check attached. For the image LoRAs I asked for 50 training steps, then a check that the saved weights had actually changed, before paying for the full run.

At the end of the prompt, I lay out the expected deliverables and limit the cost. I define what belongs in the model card and include a instruction like: "Cap total spend at USD 12 and ask me before exceeding it." Because ML-intern begins every task with zero dollar budget and needs permission before executing paid jobs, this spending limit stays strictly enforced. When you leave out a budget, the agent suggests a couple of paths depending on project size and asks which one you prefer.

You don't necessarily need all of that on your first attempt. For example, I didn't have the *verified-facts* section in my _citrus_ brief and ML Intern still produced a model that more than [tripled the accuracy](https://huggingface.co/ML-Intern-lab/citrus-disease-vlm#results) of the [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) model. Let me walk you through what I have achieved with Ml-Intern in a matter of a couple of weeks.

## A model that knows your field

A general vision model can describe a yellowing citrus leaf. However, telling you whether it is a mite problem or a magnesium deficiency, and the bio and non-bio remedies to treat the plant is very hard. Using Claude, I put together a training dataset merged from three sources hosted by the [Project-AgML](https://huggingface.co/Project-AgML) organization on the Hub. The resulting [citrus-disease-vlm-instruct](https://huggingface.co/datasets/ML-Intern-lab/citrus-disease-vlm-instruct) is a dataset containing 3,017 annotated images across 21 distinct pests, illnesses, nutritional gaps, and treatment approaches. ML-intern handled the fine-tuning of Qwen3.5-2B using these examples, making sure to benchmark the foundation model beforehand.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/citrus-doctor.mp4"></video>

On the 335 test photos, the base model named the right problem 14.9% of the time. After two epochs on one A10G, the fine-tuned model got 52.8%. Compute cost, about USD 1.90.

Check out: [Model](https://huggingface.co/ML-Intern-lab/citrus-disease-vlm) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/citrus-disease-vlm-instruct) · [Citrus Doctor App](https://huggingface.co/spaces/ML-Intern-lab/citrus-doctor)

## A model that draws your character

Image models know plenty of characters. Huggy, drawn in the flat style of the Hugging Face brand assets, was not one of them. I asked ML-Intern for a LoRA on [FLUX.2 klein base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B), trained on 84 captioned drawings from [Chunte/huggy_for_training](https://huggingface.co/datasets/Chunte/huggy_for_training) dataset.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/huggy-model-samples.mp4"></video>

The agent saved a checkpoint every 100 steps and drew the same set of prompts with each one, which made choosing easy. Step 200 was the first where Huggy was fully *on-model*. From step 500 on, Huggy's style started bleeding into prompts that had nothing to do with Huggy! The trained LoRA also works on the distilled klein model at 4 steps. Compute cost, about USD 7.60.

Check out: [Model](https://huggingface.co/ML-Intern-lab/huggy-flux2-klein-lora) · [Dataset](https://huggingface.co/datasets/Chunte/huggy_for_training) · [Huggy Generator App](https://huggingface.co/spaces/ML-Intern-lab/huggy-generator)

## A model that does a new trick

1. **Camera-angle LoRAs** are among the most-liked community add-ons for earlier Qwen-Image models. You can give the model a picture of an object and ask to see it 45 degrees from the left. When I checked a few days after the Qwen-Image 2.1 model release, nobody had made one, so I tasked ML-intern to build it. 

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/viewpoint-orbit-lora.gif" alt="Qwen camera angle LoRA" width="400" >
</p>

ML-intern rendered 1,030 scanned household objects from [Google Scanned Objects](https://huggingface.co/datasets/suvadityamuk/google-scanned-objects) at 24 angles each, 24,722 transparent images, on a CPU job that cost a few cents. It later finalised 461 objects for training and 40 held out for testing, and 1,844 before-and-after training pairs spread evenly over 23 camera instructions.

Training ran 2,000 steps in about 90 minutes on one A100 (~USD 3.75). The whole project took about half a day and 48 jobs, counting the ones that failed on missing packages or wrong paths and had to be resubmitted by ML-Intern. Total compute cost, about USD 16.

Check out: [Model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/gso-orbit-rgba) · [Viewpoint Orbit App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA)

2. **Doodle-in LoRA** is another cool idea. Upload a photo with a magenta scribble on it and add a short prompt naming an object. The LoRA replaces the scribble with that object while keeping the original lighting and composition consistent.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/qwen-image-2.1-doodle-in-lora-sample2.mp4"></video>

No dataset existed for this, so my prompt described how to make one. Start from a real photo in Open Images, remove one object with the LaMa inpainting model, and draw a scribble where the object used to be. The untouched photo is the target. ML-intern wrote and tested the pair-building scripts in a CPU sandbox, then ran them as GPU jobs, recording the author and license of every source photo along the way. It built 6,042 training pairs and a 160-pair test set, where 40 of the test pairs come from 23 object classes kept out of training entirely.

Before training, it measured the base model on its own and with the Viggle turbo LoRA, and checked that running edits in batches produced identical images, which made the evaluation cheaper. Training ran 2,000 steps in 1 hour 38 minutes on one A100 (~USD 4), and a comparison of the saved checkpoints on 48 test pairs picked step 500.

Paired with the [Viggle turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) at 6 steps the LoRA performed really well. [67.5%](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA#main-comparison--160-test-pairs) of objects detected where they were drawn, at 4.7 seconds per edit. Objects from the 23 unseen classes landed as reliably as the rest (65.0% versus 64.2%). The project took a little over a day and 59 jobs. Total compute cost, about USD 24.


Check out: [Model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/doodle-in-pairs) · [Doodle-in App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA)

## A model that fits your device

1. The pocket rewriter from the top of this post is the first one. ML-intern started by generating 8,797 short image requests with a small instruct model through Inference Providers, following a mix set in my prompt: photos, posters, logos, infographics and more, about a third of them asking for exact text in quotes, and many in languages other than English. The 9B teacher then rewrote all of them on one A100 in 2 hours 37 minutes (~USD 6.50). After filtering for quality, 1,840 examples were selected for training dataset.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/qwen-image-2.1-pocket-studio.png" alt="Qwen Image 2.1 Pocket Studio">

Training the 0.8B and 2B students took 12 and 18 minutes on an A10G (USD 0.75 for both). The 0.8B also ships as an 812 MB GGUF file for running on a CPU. The project took about 11 hours and 24 jobs. Total compute cost, about USD 16.

Check out: [Pocket rewriter 0.8B Student](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B) · [2B Student](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-2B) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/Qwen-Image-2.1-rewriter-distill) · [Pocket Studio App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-pocket-studio) · [Compare the teacher-student in Rewriter Arena](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-rewriter-arena)

2. Agate-Preview-002-4step is the second. [Logolabs' Agate Preview 002](https://huggingface.co/Logolabs/agate-preview-002) is a 260M-parameter text-to-image model, small enough for a browser, but it needs 50 steps with guidance, which is 100 network passes per image. I asked ML-intern to distill it down to just 4 passes!

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/agate-preview-002-4step.mp4"></video>


It took two runs. The first cached 155,000 training images as latents, baked the guidance into the model, then cut the step count in stages from 16 to 8 to 4, all on A100s. The 4-step student beat the teacher run at the same 4 steps on GenEval and FID, after that ML-intern exported it to ONNX for the browser. This run took about 13 hours and USD 22.

I did a second training run by asking ML-Intern to improve the 4-step student a bit more. It then made 24,000 more image pairs with the teacher at 16 steps and fine-tuned the 4-step student against them for about an hour. GenEval went from 0.509 to 0.536, against the teacher's 0.563 at 50 steps, with just 4 steps (one-fourth the compute). ML-intern re-exported the browser version. The second run took about 8 hours. Total compute cost across both runs, about USD 37.


Check out: [Agate 4-step model](https://huggingface.co/ML-Intern-lab/agate-preview-002-4step) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/agate-preview-002-4step-latents) · [Run Agate in your browser](https://huggingface.co/spaces/ML-Intern-lab/agate-preview-002-4step-webgpu) · [Agate 4-step LIVE](https://huggingface.co/spaces/ML-Intern-lab/agate-preview-002-4step-live)

## What it cost

| Model | Base model | Compute |
|---|---|---|
| Citrus Doctor | Qwen3.5-2B | USD 1.90 |
| Huggy LoRA | FLUX.2 klein base 4B | USD 7.60 |
| Pocket rewriter (0.8B and 2B) | Qwen3.5-0.8B and 2B | USD 16.05 |
| Viewpoint Orbit LoRA | Qwen-Image 2.1 | USD 16 |
| Doodle-in LoRA | Qwen-Image 2.1 | USD 24.30 |
| Agate 4-step (two runs) | Agate Preview 002 | USD 37 |
| **Total** | | **about USD 103** |

These are the GPU and CPU job charges reported for each session.

## Make yours

ML-intern is in [HuggingChat](https://huggingface.co/chat/). Switch on ML-intern mode and paste a prompt. If you want a starting point, [my example prompts](https://github.com/yvrjsharma/ml-intern-prompts) are free to copy. Start from a model you wish existed and a dataset you have, or one you can describe. Give it a small budget, ask for a baseline and a smoke test, and read what comes back before you raise the cap.

If you make something with it, share it on X and tag [@Gradio](https://x.com/Gradio) and [@HuggingFace](https://x.com/huggingface). We would love to see the models you make that nobody else would think to build for you.