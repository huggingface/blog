---
title: "Welcome Llama 4 Maverick & Scout on Hugging Face"
thumbnail: /blog/assets/llama_4.png
authors:
- user: burtenshaw
- user: reach-vb
- user: pcuenq
- user: clem
- user: rajatarya
  guest: true
  org: xet-team
- user: jsulz
  guest: true
  org: xet-team
- user: lysandre
---

# Welcome Llama 4 Maverick & Scout on Hugging Face

We are incredibly excited to welcome the next generation of large language models from Meta to the Hugging Face Hub: [Llama 4 Maverick (\~400B)](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Original) and [Llama 4 Scout (\~109B)\!](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Original) 🤗 Both are Mixture of Experts (MoE) models with 17B active parameters.

Released today, these powerful, natively multimodal models represent a significant leap forward. We've worked closely with Meta to ensure seamless integration into the Hugging Face ecosystem, including both transformers and TGI from day one. 

This is just the start of our journey with Llama 4\. Over the coming days we’ll continue to collaborate with the community to build amazing models, datasets, and applications with Maverick and Scout\! 🔥

## What is Llama 4?

Llama 4, developed by Meta, introduces a new auto-regressive Mixture-of-Experts (MoE) architecture.This generation includes two models:

- The highly capable **Llama 4 Maverick** with 17B active parameters out of \~400B total, with 128 experts.  
- The efficient **Llama 4 Scout** also  has 17B active parameters out of \~109B total, using just 16 experts. 

Both models leverage early fusion for native multimodality, enabling them to process text and image inputs. Maverick and Scout are both trained on up to 40 trillion tokens on data encompassing 200 languages (with specific fine-tuning support for 12 languages including Arabic, Spanish, German, and Hindi). 

For deployment, Llama 4 Scout is designed for accessibility, fitting on a single server-grade GPU via on-the-fly 4-bit or 8-bit quantization, while Maverick is available in BF16 and FP8 formats. These models are released under the custom Llama 4 Community License Agreement, available on the model repositories.

## Features and Integrations on Hugging Face

To help the community leverage these state-of-the-art models immediately, we're thrilled to announce the following integrations:

* **Model Checkpoints on the Hub:** Both Llama 4 Maverick and Llama 4 Scout model weights are available directly on the Hugging Face Hub under the `meta-llama` organization. This includes both base and instruction tuned variants. This allows for easy access, exploration, and download. You need to accept the license terms on the model card before accessing the weights.  
* **Hugging Face `transformers` integration**: Get building now\! Llama 4 models are fully integrated with `transformers` (version `v4.51.0`). This allows for easy loading, inference, and fine-tuning using familiar APIs, including support for their native multimodal capabilities, and downstream libraries like TRL.  
* **Automatic support for tensor-parallel** and automatic device mapping in transformers.
* **Text Generation Inference (TGI) Support:** For optimized and scalable deployment, both models are supported by TGI. This allows for high-throughput text generation, making it easier to integrate Llama 4 into production applications.  
* **Quantization Support:** Code for on-the-fly int4 quantization is provided for Scout, minimizing performance degradation while enabling deployment on smaller hardware footprints. Maverick includes FP8 quantized weights for efficient deployment on compatible hardware.  
* **Xet Storage:** To improve uploads, downloads, and support faster iteration on community finetuned models we’ve launched all Llama 4 models using the [Xet storage backend](https://huggingface.co/blog/xet-on-the-hub). This storage system was designed for faster uploads & downloads and with Llama 4 it achieves \~25% deduplication. All derivative (finetune, quantizations, etc.) models should have higher deduplication (\~40%) saving the community even more time & bandwidth.

## Context Length and Architecture Choices

The Llama 4 models were pre-trained with a context length of 256K. The Instruct models were fine-tuned to support much larger context lengths: 1M in the large 128 experts version (Maverick), and 10M (!) for the 16 experts version (Scout).

| Model           | Instruct | Context Length |
|-----------------|:--------:|:--------------:|
| Scout (16E)     |     ✅    |       10M      |
| Maverick (128E) |     ✅    |       1M       |
| Scout (16E)     |          |      256K      |
| Maverick (128E) |          |      256K      |

These large context lengths come with a few very interesting architecture choices. Until an official technical report is published, this is what we know so far.

* **No RoPE (NoPE) layers**

NoPE (cute name, +1 charisma points), which was explored as far back as 2022, just forgoes the traditional positional encoding schemes, such as RoPE, that are most times applied in transformers models. In the case of Llama 4, NoPE layers are used every 4 layers. These layers are crucial for long context, as they use the full causal mask over the context.

For RoPE layers (three out of 4), _chunked attention_ is used.

Meta refers to the _interleaved_ use of NoPE layers, together with temperature scaling (as explained below), as the `iRoPE` architecture.

_If you want to learn more about positional encodings, we recommend [Chris' recent post](https://huggingface.co/blog/designing-positional-encoding)._

* **Chunked attention** (in RoPE layers)

As a way to reduce memory requirements, Llama 4 uses chunked attention in the layers that work with traditional RoPE positional encodings (three out of 4 decoder layers). The best way to visualize how chunked attention works is through this ASCII representation that was extracted from the [transformers source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py#L848-L857):

```
'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚ 
'▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚ 
'▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚ 
'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚ 
'▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚ 
'?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■ 
```

This diagram shows the attention mask that would be used if the chunked attention length was 3. In the case of Llama 4, chunked attention length is `8192`. This means that RoPE layers can only keep track of context in 8K blocks, while NoPE layers have access to the full context.

* **Attention Temperature Tuning**

Attention blocks applied to long contexts have a problem: the attention probability scores _fade_ closer to zero as the sequence length increases. This is a known consequence of applying the _softmax_ function to very long sequences. To address this problem, Llama 4 uses a scaled softmax, which the model refers to as temperature tuning. This is applied in the NoPE layers, but not in the RoPE ones as these attend to shorter sub-sequences.

This method is a way to improve generalization for arbitrary context lengths, and probably one of the key factors to achieve 10M context length in Llama 4 Scout.

* **QK Normalization**

Llama Scout (the 16 experts version) uses an additional L2 normalization of the Query and Key states in RoPE layers, after RoPE embeddings have been applied. Remember that Llama Scout Instruct is the model that supports the largest 10M context length.

* **MoE interleaving**

Llama Scout is a full MoE consisting of 16 experts. Llama Maverick uses 128 experts, but MoE and dense layers alternate. Therefore, experts are applied in half of the layers.


## How to Use with Transformers

Getting started with Llama 4 using `transformers` is straightforward. Make sure you have `transformers v4.51.0` or later installed (`pip install -U transformers huggingface_hub[hf_xet]`). Here's a quick example using the instruction-tuned Maverick model responding about two images, using tensor parallel for maximum speed. You need to run this script on an instance with 8 GPUs, using a command like:  
`torchrun –nproc-per-instance=8 script.py`

```py
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
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

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])
```

Make sure to check the model cards on the repos ([Llama 4 Maverick (\~400B)](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Original) and [Llama 4 Scout (\~109B)](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Original)) for detailed usage instructions, including multimodal examples, specific prompt formats (like system prompts), quantization details, and advanced configuration options\!

## Evaluation Scores

Evaluation results confirm the strength of these models, showing state-of-the-art performance that significantly outperforms predecessors like Llama 3.1 405B. For instance, on reasoning and knowledge tasks, the instruction-tuned Maverick achieves 80.5% on MMLU Pro and 69.8% on GPQA Diamond, while Scout scores 74.3% and 57.2% respectively.

<!-- expander -->
<details>

<summary>Click to expand Evaluation Results</summary>
### Pre-trained models

| Pre-trained models |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Category | Benchmark | \# Shots | Metric | Llama 3.1 70B | Llama 3.1 405B | **Llama 4 Scout** | **Llama 4 Maverick** |
| Reasoning & Knowledge | MMLU | 5 | macro\_avg/acc\_char	 | 79.3 | 85.2 | 79.6 | 85.5 |
|  | MMLU-Pro | 5 | macro\_avg/em | 53.8 | 61.6 | 58.2 | 62.9 |
|  | MATH | 4 | em\_maj1@1 | 41.6 | 53.5 | 50.3 | 61.2 |
| Code | MBPP | 3 | pass@1 | 66.4 | 74.4 | 67.8 | 77.6 |
| Multilingual | TydiQA | 1 | average/f1 | 29.9 | 34.3 | 31.5 | 31.7 |
| Image | ChartQA | 0 | relaxed\_accuracy | No multimodal support |  | 83.4 | 85.3 |
|  | DocVQA | 0 | anls |  |  | 89.4 | 91.6 |

### Instruction tuned models	

| Instruction tuned models |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | ----- | :---: | :---: |
| Category | Benchmark | \# Shots | Metric | Llama 3.3 70B | Llama 3.1 405B | **Llama 4 Scout** | **Llama 4 Maverick** |
| Image Reasoning | MMMU | 0 | accuracy | No multimodal support |  | 69.4 | 73.4 |
|  | MMMU Pro^ | 0 | accuracy |  |  | 52.2 | 59.6 |
|  | MathVista | 0 | accuracy |  |  | 70.7 | 73.7 |
| Image Understanding | ChartQA | 0 | relaxed\_accuracy |  |  | 88.8 | 90.0 |
|  | DocVQA (test) | 0 | anls |  |  | 94.4 | 94.4 |
| Coding | LiveCodeBench (10/01/2024-02/01/2025) | 0 | pass@1 | 33.3 | 27.7 | 32.8 | 43.4 |
| Reasoning & Knowledge | MMLU Pro | 0 | macro\_avg/em | 68.9 | 73.4 | 74.3 | 80.5 |
|  | GPQA Diamond | 0 | accuracy | 50.5 | 49.0 | 57.2 | 69.8 |
| Multilingual | MGSM | 0 | average/em | 91.1 | 91.6 | 90.6 | 92.3 |
| Long context | MTOB (half book) eng-\>kgv/kgv-\>eng | \- | chrF | Context window is 128K |  | 42.2/36.6 | 54.0/46.4 |
|  | MTOB (full book) eng-\>kgv/kgv-\>eng | \- | chrF |  |  | 39.7/36.3 | 50.8/46.7 |

</details>

## Acknowledgments

Releasing a giant like Llama 4 takes a colossal effort across teams, geographies and a lot of VMs. In no particular order we’d like to thank Arthur, Lysandre, Cyril, Pablo, Marc, Mohammed from the Transformers team. We are grateful to the full vLLM team for richful discussions, insights, shared testing and debugging during this intense integration with many challenges. With larger optimisation needs, we’d like to thank Mohit for single-handedly adding support to Llama 4 in TGI. These chonky models require some serious engineering at the storage level. This took a lot of effort from Ajit, Rajat, Jared, Di, Yucheng and the rest of the [Xet team](http://hf.co/xet-team) too.

There’s a lot of people involved in this effort, thanks a lot to the rest of the Hugging Face, vLLM and Meta Llama teams for the brilliant synergy\!

## References

* To learn more about Xet Storage: [blog post](https://huggingface.co/blog/xet-on-the-hub), and [Hub docs](https://huggingface.co/docs/hub/storage-backends).  
* Check out Meta’s release [blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
