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


To demonstrate our vision in small video models, we've built three practical applications that showcase the versatility of these models.

### iPhone Video Understanding
<table style="border-collapse: collapse;">
<tr>
<td width="600" style="border: none;">
<center>
<iframe width="300" height="533" src="https://www.youtube.com/embed/G1yQlHTk_Ig" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>
</td>
<td valign="top" style="border: none;">
We've created an iPhone app running SmolVLM2 completely locally. Using our 500M model, users can analyze and understand video content directly on their device - no cloud required. Interested in building iPhone video processing apps with AI models running locally? We're releasing it very soon - <a href="https://huggingface.co/datasets/HuggingFaceTB/smolvlm2-iphone-waitlist" target="_blank">fill this form to test and build with us!</a>
</td>
</tr>
</table>

### Video Highlight Generator
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

### VLC media player integration
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

### SmolVLM2 Collection

We've made all models and demos available [in this collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7). 

Want to try SmolVLM2 right away? Check out our [interactive chat interface](huggingface.co/spaces/HuggingFaceTB/SmolVLM2) where you can test visual and video understanding capabilities of SmolVLM2 2.2B through a simple, intuitive interface.
<br>

## Table of Contents

- [SmolVLM2: Bringing Video Understanding to Every Device](#smolvlm2-bringing-video-understanding-to-every-device)
  - [TL;DR: SmolVLM can now watch 📺 with even better visual understanding](#tldr-smolvlm-can-now-watch--with-even-better-visual-understanding)
    - [iPhone Video Understanding](#iphone-video-understanding)
    - [Video Highlight Generator](#video-highlight-generator)
    - [VLC media player integration](#vlc-media-player-integration)
    - [SmolVLM2 Collection](#smolvlm2-collection)
  - [Table of Contents](#table-of-contents)
  - [Technical Details](#technical-details)
    - [SmolVLM2 2.2B: Our New Star Player for Vision and Video](#smolvlm2-22b-our-new-star-player-for-vision-and-video)
    - [Going Even Smaller: Meet the 500M and 256M Video Models](#going-even-smaller-meet-the-500m-and-256m-video-models)
  - [Using SmolVLM2 and Fine-tuning it with Transformers and MLX](#using-smolvlm2-and-fine-tuning-it-with-transformers-and-mlx)
    - [Transformers](#transformers)
      - [Video Inference](#video-inference)
      - [Multiple Image Inference](#multiple-image-inference)
      - [Interleaving Image, Text and Video](#interleaving-image-text-and-video)
    - [Inference with MLX](#inference-with-mlx)
    - [Fine-tuning SmolVLM2](#fine-tuning-smolvlm2)
  - [Read More](#read-more)




## Technical Details

We are introducing three new models with 256M, 500M and 2.2B parameters. The 2.2B model is the go-to choice for vision and video tasks, while the 500M and 256M models represent **the smallest video language models ever released**.

While they're small in size, they outperform any existing models per memory consumption. Looking at Video-MME (the go-to scientific benchmark in video), SmolVLM2 joins frontier model families on the 2B range and we lead the pack in the even smaller space.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-videomme2.png" width="50%" alt="SmolVLM2 Performance">


*Video-MME stands out as a comprehensive benchmark due to its extensive coverage across diverse video types, varying durations (11 seconds to 1 hour), multiple data modalities (including subtitles and audio), and high-quality expert annotations spanning 900 videos totaling 254 hours. Learn more [here](https://video-mme.github.io/home_page.html).*



### SmolVLM2 2.2B: Our New Star Player for Vision and Video


The new 2.2B model got better at solving math problems with images, reading text in photos, understanding complex diagrams, and tackling scientific visual questions, we see this reflected in the model performance across different benchmarks:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm2-score-gains.png" width="50%" alt="SmolVLM2 Vision Score Gains">



When it comes to video tasks, 2.2B is a good bang for the buck. Across the different scientific benchmarks where we evaluated it we want to highlight its performance on Video-MME where it outperforms all existing 2B models. 

We were able to achieve a good balance on video/vision performance thanks to the data mixture learnings published in [Apollo: An Exploration of Video Understanding in Large Multimodal Models](https://apollo-lmms.github.io/)

It’s so memory efficient, that you can run it even in a free Google Colab.

<details>
<summary>Python Code</summary>

```python
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

Our new [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) model has very close video capabilities than the original SmolVLM 2.2B, but at a fraction of the size: we're getting the same video understanding capabilities with less than a quarter of the parameters.

And then there's our little experiment, the SmolVLM2-256M-Video-Instruct. Think of it as our "what if" project - what if we could push the boundaries of small models even further? Taking inspiration from what [IBM achieved](https://ds4sd.github.io/docling/examples/pictures_description/) with our base [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) a few weeks ago, we wanted to see how far we could go with video understanding. While it's more of an experimental release, we're hoping it'll inspire some creative applications and specialized fine-tuning projects.




## Using SmolVLM2 and Fine-tuning it with Transformers and MLX

We make SmolVLM2 available to use with transformers and MLX from day zero. In this section, you can find different inference alternatives and tutorials for video and multiple images.

### Transformers

There are two ways to infer SmolVLM2 models, one is through a chat template and the other one gives you more control by passing in media through the processor.

You can load the model as follows.

```python

from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to(DEVICE)
```

#### Video Inference

You can pass videos through a chat template by passing in {“type”: “video”, “path”:{video_path}”. See below complete example. 

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

You can infer multiple images through a chat template as well. 

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

#### Interleaving Image, Text and Video

You can interleave image, video and text together by passing in `<image>` and `<video>` tokens inside text, cutting text through and inserting image lines in between.

```python
import torch


messages = [
    {
        "role": "user",
        "content": [
{"type": "text", "text": "What is the similarity between this image <image>"},

            {"type": "image", "path": "image_1.png"},
{"type": "text", "text": "and this image <image>"},
{"type": "image", "path": "image_2.png"},
            
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

You can run SmolVLM2 on MLX with following one-liner: 
```bash
python3 -m mlx_vlm.generate –model HuggingFaceTB/SmolVLM-256M-Instruct –max-tokens 400 –temp 0.0 -image {YOUR_IMAGE_PATH}
```

### Fine-tuning SmolVLM2




## Read More

We are looking forward to see all the things you'll build with SmolVLM2!
If you'd like to learn more about SmolVLM family of models, feel free to read the following:

[SmolVLM2 - Collection with Models and Demos](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)
[SmolVLM - small yet mighty Vision Language Model](https://huggingface.co/blog/smolvlm)
