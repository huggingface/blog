---
title: "SmolVLM2: Bringing Video Understanding to Every Device" 
thumbnail: /blog/assets/smolvlm2/banner.png
authors:
- user: orrzohar
  guest: true
  org: Stanford
- user: mfarre
- user: andito
- user: merve
- user: pcuenq
- user: cyrilzakka
- user: xenova
---

# SmolVLM2: Bringing Video Understanding to Every Device

## TL;DR: SmolVLM can now watch 📺 with even better visual understanding

SmolVLM2 represents a fundamental shift in how we think about video understanding - moving from massive models that require substantial computing resources to efficient models that can run anywhere. Our goal is simple: make video understanding accessible across all devices and use cases, from phones to servers.

We are releasing models in three sizes (2.2B, 500M and 256M), MLX ready (Python _and_ Swift APIs) from day zero.
We've made all models and demos available [in this collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7). 

Want to try SmolVLM2 right away? Check out our [interactive chat interface](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2) where you can test visual and video understanding capabilities of SmolVLM2 2.2B through a simple, intuitive interface.
<br>

## Table of Contents

- [SmolVLM2: Bringing Video Understanding to Every Device](#smolvlm2-bringing-video-understanding-to-every-device)
  - [TL;DR: SmolVLM can now watch 📺 with even better visual understanding](#tldr-smolvlm-can-now-watch--with-even-better-visual-understanding)
  - [Table of Contents](#table-of-contents)
  - [Technical Details](#technical-details)
    - [SmolVLM2 2.2B: Our New Star Player for Vision and Video](#smolvlm2-22b-our-new-star-player-for-vision-and-video)
    - [Going Even Smaller: Meet the 500M and 256M Video Models](#going-even-smaller-meet-the-500m-and-256m-video-models)
    - [Suite of SmolVLM2 Demo applications](#suite-of-smolvlm2-demo-applications)
      - [iPhone Video Understanding](#iphone-video-understanding)
      - [VLC media player integration](#vlc-media-player-integration)
      - [Video Highlight Generator](#video-highlight-generator)
  - [Using SmolVLM2 with Transformers and MLX](#using-smolvlm2-with-transformers-and-mlx)
    - [Transformers](#transformers)
      - [Video Inference](#video-inference)
      - [Multiple Image Inference](#multiple-image-inference)
    - [Inference with MLX](#inference-with-mlx)
      - [Swift MLX](#swift-mlx)
    - [Fine-tuning SmolVLM2](#fine-tuning-smolvlm2)
  - [Read More](#read-more)




## Technical Details

We are introducing three new models with 256M, 500M and 2.2B parameters. The 2.2B model is the go-to choice for vision and video tasks, while the 500M and 256M models represent **the smallest video language models ever released**.

While they're small in size, they outperform any existing models per memory consumption. Looking at Video-MME (the go-to scientific benchmark in video), SmolVLM2 joins frontier model families on the 2B range and we lead the pack in the even smaller space.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-videomme2.png" width="50%" alt="SmolVLM2 Performance">


*Video-MME stands out as a comprehensive benchmark due to its extensive coverage across diverse video types, varying durations (11 seconds to 1 hour), multiple data modalities (including subtitles and audio), and high-quality expert annotations spanning 900 videos totaling 254 hours. Learn more [here](https://video-mme.github.io/home_page.html).*



### SmolVLM2 2.2B: Our New Star Player for Vision and Video


Compared with the previous SmolVLM family, our new 2.2B model got better at solving math problems with images, reading text in photos, understanding complex diagrams, and tackling scientific visual questions. This shows in the model performance across different benchmarks:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-score-gains.png" width="50%" alt="SmolVLM2 Vision Score Gains">



When it comes to video tasks, 2.2B is a good bang for the buck. Across the various scientific benchmarks we evaluated it on, we want to highlight its performance on Video-MME where it outperforms all existing 2B models.

We were able to achieve a good balance on video/image performance thanks to the data mixture learnings published in [Apollo: An Exploration of Video Understanding in Large Multimodal Models](https://apollo-lmms.github.io/)

It’s so memory efficient, that you can run it even in a free Google Colab.

<details>
<summary>Python Code</summary>

```python
# Make sure we are running the latest version of Transformers
!pip install git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "path_to_video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```
</details>


### Going Even Smaller: Meet the 500M and 256M Video Models

Nobody dared to release such small video models until today.

Our new [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) model has video capabilities very close to SmolVLM 2.2B, but at a fraction of the size: we're getting the same video understanding capabilities with less than a quarter of the parameters 🤯.

And then there's our little experiment, the SmolVLM2-256M-Video-Instruct. Think of it as our "what if" project - what if we could push the boundaries of small models even further? Taking inspiration from what [IBM achieved](https://ds4sd.github.io/docling/examples/pictures_description/) with our base [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) a few weeks ago, we wanted to see how far we could go with video understanding. While it's more of an experimental release, we're hoping it'll inspire some creative applications and specialized fine-tuning projects.


### Suite of SmolVLM2 Demo applications

To demonstrate our vision in small video models, we've built three practical applications that showcase the versatility of these models.

#### iPhone Video Understanding
<table style="border-collapse: collapse;">
<tr>
<td width="300" style="border: none;">
<center>
<iframe width="300" height="533" src="https://www.youtube.com/embed/G1yQlHTk_Ig" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
We've created an iPhone app running SmolVLM2 completely locally. Using our 500M model, users can analyze and understand video content directly on their device - no cloud required. Interested in building iPhone video processing apps with AI models running locally? We're releasing it very soon - <a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-iPhone-waitlist" target="_blank">fill this form to test and build with us!</a>
</td>
</tr>
</table>

#### VLC media player integration
<table style="border-collapse: collapse;">
<tr>
<td width="500" style="border: none;">
<center>
<iframe width="500" height="281" src="https://www.youtube.com/embed/NGHCFEW7DCg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
Working in collaboration with VLC media player, we're integrating SmolVLM2 to provide intelligent video segment descriptions and navigation. This integration allows users to search through video content semantically, jumping directly to relevant sections based on natural language descriptions. While this is work in progress, you can experiment with the current playlist builder prototype <a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-XSPFGenerator" target="_blank">in this space.</a>
</td>
</tr>
</table>

#### Video Highlight Generator
<table style="border-collapse: collapse;">
<tr>
<td width="500" style="border: none;">
<center>
<iframe width="500" height="281" src="https://www.youtube.com/embed/ZT2oS8EqnKI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
Available as a Hugging Face Space, this application takes long-form videos (1+ hours) and automatically extracts the most significant moments. We've tested it extensively with soccer matches and other lengthy events, making it a powerful tool for content summarization. <a href="https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2-HighlightGenerator" target="_blank">Try it yourself in our demo space.</a>
</td>
</tr>
</table>





## Using SmolVLM2 with Transformers and MLX

We make SmolVLM2 available to use with transformers and MLX from day zero. In this section, you can find different inference alternatives and tutorials for video and multiple images.

### Transformers

The easiest way to run inference with the SmolVLM2 models is through the conversational API – applying the chat template takes care of preparing all inputs automatically.

You can load the model as follows.

```python

# Make sure we are running the latest version of Transformers
!pip install git+https://github.com/huggingface/transformers.git

from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to(DEVICE)
```

#### Video Inference

You can pass videos through a chat template by passing in `{"type": "video", "path": {video_path}`. See below for a complete example. 

```python
import torch

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "path_to_video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```



#### Multiple Image Inference

In addition to video, SmolVLM2 supports multi-image conversations. You can use the same API through the chat template.

```python
import torch


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the differences between these two images?"},
            {"type": "image", "path": "image_1.png"},
            {"type": "image", "path": "image_2.png"} 
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
```


### Inference with MLX 

To run SmolVLM2 with MLX on Apple Silicon devices using Python, you can use the excellent [mlx-vlm library](https://github.com/Blaizzy/mlx-vlm).
First, you need to install `mlx-vlm` from [this branch](https://github.com/Blaizzy/mlx-vlm/pull/208) using the following command:
 
 ```bash
pip install git+https://github.com/pcuenca/mlx-vlm.git@smolvlm
```

Then you can run inference on a single image using the following one-liner, which uses [the unquantized 500M version of SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx):

```bash
python -m mlx_vlm.generate \
  --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
  --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg \
  --prompt "Can you describe this image?"
```

We also created a simple script for video understanding. You can use it as follows:

```bash
python -m mlx_vlm.smolvlm_video_generate \
  --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
  --system "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action." \
  --prompt "What is happening in this video?" \
  --video /Users/pedro/Downloads/IMG_2855.mov \
  --prompt "Can you describe this image?"
```

Note that the system prompt is important to bend the model to the desired behaviour. You can use it to, for example, describe all scenes and transitions, or to provide a one-sentence summary of what's going on.

#### Swift MLX

The Swift language is also supported through the [mlx-swift-examples repo](https://github.com/ml-explore/mlx-swift-examples), which is what we used to build our iPhone app.

Until [our in-progress PR](https://github.com/ml-explore/mlx-swift-examples/pull/206) is finalized and merged, you have to compile the project [from this fork](https://github.com/cyrilzakka/mlx-swift-examples), and then you can use the `llm-tool` CLI on your Mac as follows.

For image inference:

```bash
./mlx-run --debug llm-tool \
    --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
    --prompt "Can you describe this image?" \
    --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg \
    --temperature 0.7 --top-p 0.9 --max-tokens 100
```

Video analysis is also supported, as well as providing a system prompt. We found system prompts to be particularly helpful for video understanding, to drive the model to the desired level of detail we are interested in. This is a video inference example:

```bash
./mlx-run --debug llm-tool \
    --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
    --system "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action." \
    --prompt "What is happening in this video?" \
    --video /Users/pedro/Downloads/IMG_2855.mov \
    --temperature 0.7 --top-p 0.9 --max-tokens 100
```

If you integrate SmolVLM2 in your apps using MLX and Swift, we'd love to know about it! Please, feel free to drop us a note in the comments section below!

### Fine-tuning SmolVLM2

You can fine-tune SmolVLM2 on videos using transformers 🤗
We have fine-tuned the 500M variant in Colab on video-caption pairs in [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) dataset for demonstration purposes. Since the 500M variant is small, it's better to apply full fine-tuning instead of QLoRA or LoRA, meanwhile you can try to apply QLoRA on cB variant. You can find the fine-tuning notebook [here](https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb).


## Read More

We would like to thank Raushan Turganbay, Arthur Zucker and Pablo Montalvo Leroux for their contribution of the model to transformers.

We are looking forward to seeing all the things you'll build with SmolVLM2!
If you'd like to learn more about the SmolVLM family of models, feel free to read the following:

[SmolVLM2 - Collection with Models and Demos](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)
