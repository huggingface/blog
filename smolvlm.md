---
title: SmolVLM - small yet mighty Vision Language Model
thumbnail: /blog/assets/smolvlm/banner.png
authors:
- user: andito
- user: merve
- user: miquel
- user: eliebak
- user: pcuenq
---

## TLDR

This blog post introduces SmolVLM a 2B SOTA VLM given its memory footprint. SmolVLM is small, fast, memory-efficient, and fully open-source. All model checkpoints, VLM datasets, training recipes and tools are released under the Apache 2.0 license.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolvlm_ecosystem.png" width="800" height="auto" alt="Image description">


## What is SmolVLM?

This year has seen a boom in multimodal AI with many large vision language models released. The trends were to initially scale up compute, later scale up the data diversity by generating synthetic data with large models, and, recently, scale down to make these models more efficient. Small open models allow local deployment to browser or edge devices, cut inference costs, and enable user customization. Some notable examples of these models include PaliGemma 3B, moondream2, and Qwen2VL.

In this blog post, we introduce SmolVLM, a new family of 2B small vision language models that can be used commercially and deployed to smaller local setups, with completely open training pipelines. 

We release three models: SmolVLM-Base, which can be used for downstream fine-tuning, SmolVLM-Synthetic, the fine-tuned variant on synthetic data, and SmolVLM Instruct, the fine-tuned instruction variant, which can be used out of the box for interactive end-user applications. 

This release comes with open-source models integrated into transformers, [a demo built on SmolVLM Instruct](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM), and a supervised fine-tuning script. We have used the datasets previously used for Idefics3: [the Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix), which are also fully open-source. [TODO: give links to all before release]

# Table of Contents
- [TLDR](#TLDR)
- [What is SmolVLM?](#what-is-smolvlm)
  * [Model capabilities](#model-capabilities)
  * [Architecture](#Architecture)
- [Performance](#performance)
  * [Benchmarks](#benchmarks)
  * [Memory](#memory)
  * [Throughput](#throughput)
  * [Video](#video)
  * [VLMEvalKit integration](#VLMEvalKit)
- [Use SmolVLM with Transformers](#use-smolvlm-with-transformers)
- [Training Details](#training-details)
  * [Dataset](#dataset)
  * [Context extension](#context-extension)
  * [Checkpoint selection](#checkpoint-selection)
- [Fine-Tuning](#fine-tuning)
- [Wrapping up](#wrapping-up) 


## Model capabilities

<table><thead><tr><th>Input Image</th><th>Input Prompt</th><th>Answer</th></tr></thead><tbody><tr> 
  <tr>
    <td style="vertical-align: middle;"><img src="https://huggingface.co/datasets/huggingface/release-assets/resolve/main/wat_arun.jpg" width="300"/></td>
    <td style="vertical-align: middle;">
    <strong>I'm planning a visit to this temple, give me travel tips.</strong>
    </td>
    <td style="vertical-align: middle;">
      The Grand Palace is the most famous landmark in Bangkok and it's worth visiting for its stunning architecture and history. Here are some of our favorite things about the palace: 

**The Main Building:** 
Located at the center of the complex, the main building features intricate tile work with traditional Thai motifs that cover every inch of the structure. This includes patterns such as dragons, mythical creatures, flowers, and geometric shapes. The roofline has an elaborate design featuring multiple layers of tiles adorned with gold accents. 

**Towers & Spires:** 
There are several towers within the Grand Palace including the Phra Nakhon Si Ayutthaya National Museum Tower which houses exhibits on Thai culture and heritage. Another notable tower is the Wat Yai Chai Mongkol Tower where you can find beautiful murals depicting scenes from Buddhist mythology.

**Architecture Style:** 
The architectural style combines elements of Hindu and Khmer styles seen commonly throughout Southeast Asia. You'll notice influences like stupas (dome-shaped structures), lotus flower carvings, and ornate details typical of these regions' religious buildings.

**Visitor Experience:** 
While exploring the Grand Palace, visitors should be aware of certain rules due to security measures taken by authorities. These include no photography inside certain areas or using flashbulbs outside. Additionally, there may be queues during peak tourist seasons so plan accordingly if you're traveling here.

### Travel Tips For Visiting The Grand Palace:
* **Best Time To Visit**: The best time to visit would likely be early morning before sunrise when temperatures drop significantly making walking around easier.
* **Transportation**: Public transportation options exist but they might not provide direct access to the Grand Palace itself; consider taking a taxi instead.
* **Food Options**: There aren't many dining facilities available near the Grand Palace yet local street food vendors do offer delicious meals nearby.
* **Clothing Guidelines**: While not mandatory, wearing traditional Thai clothing (sarongs) could add cultural immersion experience. However, shorts/trousers are allowed provided they don't go below knee level.
* **Safety Measures**: Visitors must follow signs indicating restricted zones marked "No Photography" among others. Always carry identification documents just in case.

By following these guidelines, your trip will undoubtedly enhance both your understanding of Thailand’s rich historical tapestry and personal enjoyment!
    </td>
  </tr>
  <tr>
    <td style="vertical-align: middle;"><img src="https://huggingface.co/datasets/huggingface/release-assets/resolve/main/weather_events.png" width="300"/></td>
    <td style="vertical-align: middle;">
    <strong>Where do the severe droughts happen according to this diagram?</strong>
    </td>
    <td style="vertical-align: middle;">
      The severe droughts are happening in eastern and southern Africa.
    </td>
  </tr>
  <tr>
    <td style="vertical-align: middle;"><img src="https://huggingface.co/datasets/huggingface/release-assets/resolve/main/invoice.png" width="300"/></td>
    <td style="vertical-align: middle;">
    <strong>What is the due date and the invoice date?</strong>
    </td>
    <td style="vertical-align: middle;">
 The due date is 26/02/2019, and the invoice date is 11/02/2019.
    </td>
  </tr>
</table>

## Architecture

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/self_attention_architecture_smolvlm.png" width="800" height="auto" alt="Image description">

For SmolVLM, we closely followed the architecture from Idefics3, to the point that we use the same implementation in transformers. There are, however a few key differences:
 - We replaced Llama 3.1 8B with SmolLM2 1.7B as the language backbone.
 - We more aggressively compress the patched information by reducing the information 9x using the pixel shuffle strategy, compared to 4x with idefics3.
 - We use patches of 384*384, instead of 364x364, because 384 is divisible by 3, which is necessary for our pixel shuffle strategy to work.
 - For this, we change the vision backbone to use shape-optimized SigLIP with patches of 384x384 pixels and inner patches of 14x14.


```
All the models in this release are built on the shape-optimized SigLIP as image encoder and SmolLM2 for text decoder part, trained on datasets with Apache 2.0 license. We release the SmolVLM checkpoints under the Apache 2.0 license.
```

## Performance

### Benchmarks

We present benchmarks for the tasks we mention in training details.
| Model             | MMMU (val) | MathVista (testmini) | MMStar (val) | DocVQA (test) | TextVQA (val) | Min GPU RAM required (GB) |
|-------------------|------------|----------------------|--------------|---------------|---------------|---------------------------|
| SmolVLM           | 38.8       | 44.6                | 42.1         | 81.6          | 72.7          | 5.02                      |
| Qwen-VL 2B        | 41.1       | 47.8                | 47.5         | 90.1          | 79.7          | 13.70                     |
| InternVL2 2B      | 34.3       | 46.3                | 49.8         | 86.9          | 73.4          | 10.52                     |
| PaliGemma 3B 448px| 34.9       | 28.7                | 48.3         | 32.2          | 56.0          | 6.72                      |
| moondream2        | 32.4       | 24.3                | 40.3         | 70.5          | 65.2          | 3.87                      |
| MiniCPM-V-2       | 38.2       | 39.8                | 39.1         | 71.9          | 74.1          | 7.88                      |
| MM1.5 1B          | 35.8       | 37.2                | 0.0          | 81.0          | 72.5          | NaN                       |


### Memory

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ram_smolvlm.png" width="900" height="auto" alt="Image description">

SmolVLM provides the best memory usage among the existing suite of vision language models in transformers. This allows it to be efficiently run on-device, such as a laptop! Below, you can find the GPU memory usage in GBs for each model, running inference with one or two input images, and using the same images and text prompts in all tests. SmolVLM's efficiency in image encoding is built into the model. SmolVLM encodes each 384x384 image patch to 81 tokens. This results in SmolVLM encoding our test prompt and a single image in 1.2k tokens, whereas Qwen2-VL uses 16k tokens. This also explains why the memory consumption increases so much for 2 images with Qwen and InternVL. In contrast, the increase is much more moderate for SmolVLM and PaliGemma, which use a similar approach.

### Throughput

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/throughput_smolvlm.png" width="1100" height="auto" alt="Image description">

SmolVLM's tiny memory footprint also implies that it requires far fewer computations to prefill the model and generate. Compared to Qwen2-VL, the prefill throughput is 3.3 to 4.5 times faster, and the generation throughput is 7.5 to 16 times faster.  

### Video

Given SmolVLM's long context and the possibility of tweaking the internal frame resizing of the model, we explored its suitability as an accessible option for basic video analysis tasks, particularly when computational resources are limited.

In our evaluation of SmolVLM's video understanding capabilities, we implemented a straightforward video processing pipeline [code], extracting up to 50 evenly sampled frames from each video while avoiding internal frame resizing.
This simple approach yielded surprisingly competitive results on the CinePile benchmark, with a score of 27.14%, a performance that positions the model between InterVL2 (2B) and Video LlaVa (7B).


The quantitative results align with our qualitative testing, looking at an example from the FineVideo dataset:

<div><iframe width="100%" height="200" src="https://www.youtube.com/embed/afV0a64FRgM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

```
Question: what is shown in the close-up? 
Assistant: In the close-up, there are bottles of Heinz tomato ketchup,
  honey, soy sauce, and garlic on a kitchen counter.

Question: describe what is happening in the video. 
Assistant: The video shows a man cooking in a kitchen.
  He is wearing a black shirt and a black apron.
  He is standing in front of a counter and is holding a bottle of Heinz tomato ketchup,
  a bottle of honey, soy sauce, and a bowl of minced garlic
```

While in the second question, we see some temporal understanding limitations (the cook points to one ingredient after the other rather than pointing/holding all of them at the same time) SmolVLM demonstrated great scene understanding and object recognition capabilities.


### VLMEvalKit integration

We integrated SmolVLM with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to facilitate easy evaluation across additional benchmarks. 

By running the following command, you can evaluate SmolVLM or your fine-tuned SmolVLM model.

```bash
python run.py --data <benchmarks> --model SmolVLM --work-dir <output_directory>
```

For example, to evaluate on MMMU dev validation set and MathVista mini and store the results in a folder called smol.

```bash
python run.py --data MMMU_DEV_VAL MathVista_MINI --model SmolVLM --work-dir smol
```

## Use SmolVLM with Transformers


You can easily load SmolVLM using the `Auto` classes in transformers. Under the hood, the model and processor are mapped to the same implementations used for Idefics3. 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smol_vlm_code_1.png" width="1100" height="auto" alt="Image description">

Image and text can be interleaved arbitrarily, and you can pass in multiple images. Here’s how you can use the chat template and pass in the formatted input to the processor.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smol_vlm_code_2.png" width="1100" height="auto" alt="Image description">

Start generating with preprocessed input and decode the generated output.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smol_vlm_code_3.png" width="1100" height="auto" alt="Image description">


## Training Details

### Dataset

We trained SmolVLM using the same data that we used for Idefics3. Mainly, we used The Cauldron and Docmatix. The full list of datasets we used can be consulted here (link to the datasets) 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mixture_the_cauldron.png" width="1100" height="auto" alt="Image description">

### Context extension

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/training_loss_smolvlm.png" width="1100" height="auto" alt="Image description">

SmolLM2’s pre-training context window is insufficient for VLMs. Images are encoded into many tokens, and we wanted to support multiple images. To address this, we extended it to 16k tokens by increasing the RoPE base value from 10k to 273k, following the guidelines in [“Scaling Laws of RoPE-based Extrapolation”](https://arxiv.org/abs/2310.05209). We fine-tuned the model on a mixture of long- and short-context datasets.
For long-context datasets, we used the “books” subset of Dolma (primarily Project Gutenberg) and code documents with 8k+ tokens from The Stack, each contributing 20% to the final mixture. For short-context datasets, we streamlined the original SmolLM2 pre-training mix to include 20% FineWeb-Edu, 20% DCLM, and 20% from our math dataset (to be released soon). The math dataset was upsampled to mitigate a performance drop observed on GSM8k during the context extension process.
All experiments were implemented using the [EasyContext repository](https://github.com/jzhang38/EasyContext).

### Checkpoint Selection

For our training run, we saved checkpoints every 25 optimization steps, allowing us to evaluate and potentially recover the model's state at different points in training. This practice is crucial for identifying the optimal model version, as training longer doesn't always guarantee better performance.
We evaluated the performance across multiple vision-language benchmarks, each weighted according to their importance. The core benchmarks included the following:

- General multimodal understanding (MMMU and MMStar) which are the most comprehensive benchmark. 
- Document and text-based visual question answering (DocVQA and TextVQA)
- Mathematical Reasoning (MathVista)
- Diagram understanding (AI2D)
- General multimodal understanding (MMMU and MMStar). 

To select the optimal checkpoint, we created a single metric by combining these benchmarks with different manually assigned weights to reflect their relative importance in assessing the model's capabilities. We used this single metric to select the best checkpoint. Generally, the models tended to do great on most benchmarks with more training, but their relative performance on DocVQA would decrease considerably. 

## Fine-tuning

You can fine-tune SmolVLM using transformers and apply alignment techniques using TRL 🚀

We provide a notebook to fine-tune it on the VQAv2 dataset, optionally using  LoRA, QLoRA or full fine-tuning. In the notebook, you can find some tricks to save up even more memory and have a larger batch size to fit SmolVLM inside consumer GPUs, like L4, for training. With batch sizes of 4, 8-bit loading with QLoRA and gradient checkpointing we can fine-tune in L4, and it consumes around ~16 GBs of VRAM. This makes it possible to fine-tune your SmolVLM using Colab! You can play around with the parameters to get a nice point in training duration-memory trade-off. 

SmolVLM also comes with TRL integration so you can apply Direct Preference Optimization (DPO) easily through the CLI. Get started by running pip install trl accelerate and then run the following command to fine-tune on [RLAIF-V] (https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset.

``` bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml examples/scripts/dpo_vlm.py  \\
  --dataset_name HuggingFaceH4/rlaif-v_formatted --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \\
  --per_device_train_batch_size 8 --gradient_accumulation_steps 32 --dataset_num_proc 32 \\
  --output_dir dpo_smolvlm_rlaif-v --bf16 --torch_dtype bfloat16 --use_peft --lora_target_modules=all-linear 
```


## Wrapping Up

We introduced SmolVLM, a fully open, small, and mighty VLM for the community! We also provide tools for the community to use and customize it. We are looking forward to seeing what you will create with SmolVLM.

Below are some resources if you would like to read more about all things related to SmolVLM.

Start playing with SmolVLM using [this demo](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM).
Learn how to fine-tune SmolVLM on VQAv2 using [this notebook](https://github.com/merveenoyan/smol-vision/blob/main/Idefics_FT.ipynb ) 
Learn more about [vision language models](https://huggingface.co/blog/vlms)