---
title: "Falcon2: An 11B parameter pretrained language model and VLM, trained on over 5000B tokens and 11 languages" 
thumbnail: /blog/assets/179_falcon2-11b/thumbnail.jpg
authors:
- user: Quent-01
  guest: true
  org: tiiuae
- user: nilabhra
  guest: true
  org: tiiuae
- user: rcojocaru
  guest: true
  org: tiiuae
- user: Mughaira
  guest: true
  org: tiiuae
- user: gcamp
  guest: true
  org: tiiuae
- user: yasserTII
  guest: true
  org: tiiuae
- user: SanathNarayan
  guest: true
  org: tiiuae
- user: griffintaur
  guest: true
  org: tiiuae
- user: clefourrier
- user: SailorTwift
---

## Table of Contents

- [The Falcon2 Models](#the-falcon-models)
- [11B LLM Training Details](#training)
- [11B LLM Evaluation](#evaluation)
- [11B LLM Using the Model](#using)
- [11B VLM Training](#vlm-training)
- [11B VLM Evaluation](#vlm-evaluation)
- [11B VLM Using the Model](#vlm-using)
- [Licensing information](#license)

# [The Falcon2 Models](#the-falcon-models)

[TII](www.tii.ae) is launching a new generation of models, [Falcon2](https://falconllm.tii.ae/), focused on providing the open-source community with a series of smaller models with enhanced performance and multi-modal support to enable cheaper inference and encourage the development of more downstream applications with improved usability.

The first generation of Falcon models, featuring [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) and [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B), made a significant contribution to the open-source community, promoting the release of advanced LLMs with permissive licenses. For more information on the previous generation of Falcon models, see the [RefinedWeb, Penedo et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fa3ed726cc5073b9c31e3e49a807789c-Abstract-Datasets_and_Benchmarks.html) and [The Falcon Series of Open Language Models, Almazrouei et al., 2023](https://arxiv.org/abs/2311.16867) papers.

The second generation of models is focused on increased usability and integrability, building a multi-modal ecosystem. We start this journey by releasing not only the base [11B LLM](https://huggingface.co/tiiuae/falcon-11B) but also the [11B VLM model](https://huggingface.co/tiiuae/Falcon-11B-vlm) that offers image understanding capabilities. The VLM will allow users to engage in chats about visual content using text.

As with our previous work, the models offer support mainly in English but have good capabilities in ten other languages, including Spanish, French, and German.

# [Falcon2-11B LLM](#training)

### Training Data
Falcon2-11B was trained on over 5,000 GT (billion tokens) of RefinedWeb, a high-quality filtered and deduplicated web dataset, enhanced with curated corpora. It followed a four-stage training strategy. The first three stages were focused on increasing the context length, from 2048 to 4096 and finally to 8192 tokens. The last stage aimed to further enhance performance using only high-quality data.

Overall, the data sources included RefinedWeb-English, Refined Web-Europe (*cs*, *de*, *es*, *fr*, *it*, *nl*, *pl*, *pt*, *ro*, *sv*), high-quality technical data, code data, and conversational data extracted from public sources.

The training stages were as follows:

| Stage   | Context Length   | GT   | 
|---------|------------------|------|
| Stage 1 | 2048             | 4500 |
| Stage 2 | 4096             | 250  | 
| Stage 3 | 8192             | 250  | 
| Stage 4 | 8192             | 500  | 

The data was tokenized with [11B](https://huggingface.co/tiiuae/falcon-11B), the same tokenizer as for the previous Falcon models.

### Model Architecture
The following table summaries some of the crucial details about the model architecture:
| Design choice                | value|
|------------------------------|-----|
| Number of Transformer Blocks | 60  |
| Number of Query Heads        | 32  |
| Number of Key/Value Heads    | 8   |
| Head Dimensions              | 128 |
| Parallel Attention           | yes |
| MLP Upscale Factor           | 4   |

### Training Procedure
Falcon2-11B was trained on 1024 A100 40GB GPUs for the majority of the training, using a 3D parallelism strategy (TP=8, PP=1, DP=128) combined with ZeRO and Flash-Attention 2.

### Training Hyperparameters
| Hyperparameter | Value     |
|----------------|-----------|
| Precision      | bfloat16  | 
| Optimizer	     | AdamW     |
| Max LR         | 3.7e-4    |
| Min LR         | 1.89e-5   |
| LR schedule    | Cos decay (stage 1) |
| Context length | 8192 (stages 3 and 4) | 
| Weight decay   | 1e-1      |
| Z-loss         | 1e-4      |
| Batch size     | Variable  |

## [Evaluation](#evaluation)

### English performance

Performance on Open LLM Leaderboard tasks:
| Checkpoint  | GT    | HellaSwag-10 | Winogrande-5 | ArcChallenge-25 | TruthfulQA-0    | MMLU-5 | GSMK8k-5 | Average   |
|-------------|-------|--------------|--------------|-----------------|----------|--------|----------|-----------|
| Falcon2-11B | 5500  | 82.91       | 78.30       | 59.73          | 52.56   | 58.37 | 53.83   |  64.28   |
| Falcon-40B  | 1000  | 85.28       | 81.29       | 61.86          | 41.65   | 56.89 | 21.46   |  58.07   |
| Falcon-7B   | 1500  | 78.13       | 72.38       | 47.87          | 34.26   | 27.79 | 4.62    |  44.17   |
| Gemma-7B    | 6000  | 82.47       | 78.45       | 61.09          | 44.91   | 66.03 | 52.77   |  64.29   |
| Llama3-8B   | 15000 | 82.09       | 77.35       | 59.47          | 43.90   | 66.69 | 44.79   |  62.38   |
| Mistral-7B  | N/A   | 83.31       | 78.37       | 59.98          | 42.15   | 64.16 | 37.83   |  60.97   |

The Hugging Face Leaderboard team provided an official evaluation of our model on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) tasks. The model performs better than models such as Llama3-8B (trained on three times more data) and Mistral-7B, and on par with Gemma-7b.

Zero shot performance:
| Checkpoint  | GT   | HellaSwag | ArcEasy | Winogrande  | ArcChallenge |
|-------------|------|-----------|----------|------------|--------------|
| Falcon2-11B | 5500 | 82.07    | 77.78   | 78.30     | 50.17       |
| Falcon-40B  | 1000 | 82.82    | 81.86   | 76.4      | 54.69       |
| Falcon-7B   | 1500 | 76.31    | 74.74   | 67.17     | 43.43       | 

The evaluation results show that the Falcon2-11B shows similar performance to Falcon-40B, at a four times smaller model size!

### Multilingual capabilities
Using the [Multilingual LLM leaderboard](https://huggingface.co/spaces/uonlp/open_multilingual_llm_leaderboard), we compare the Falcon2-11B model to the Llama-7B and Bloom-7B. For reference, we also include Falcon-40B (that supports the same languages), Falcon-7B (that supports French) and Mistral-7B.

| Model       | Language ID | ArcChallenge-25 | Hellaswag | MMLU 25 | TQA | Average |
|-------------|-------------|----------|----------|----------|-----------|----------|
| Falcon2-11B | *de*        | 43.7     | 67.96    | 38.3     | 47.53     | **49.37** |
|             | *es*        | 46.2     | 73.63    | 37.9     | 46.43     | **51.06** |
|             | *fr*        | 45.8     | 72.41    | 39.53    | 47.30     | **51.27** |
|             | *it*        | 45.6     | 70.83    | 38.05    | 47.14     | **50.42** |
|             | *nl*        | 41.7     | 69.05    | 38.29    | 48.81     | **49.47** |
|             | *ro*        | 42.4     | 66.24    | 38.01    | 45.53     | **48.04** |
| Falcon-40B  | *de*        | 45.1     | 68.3     | 36.2     | 39.8      | 47.4  |
|             | *es*        | 48.5     | 73.9     | 37.2     | 39.0      | 49.6  |
|             | *fr*        | 47.6     | 72.9     | 37.3     | 38.5      | 49.1  |
|             | *it*        | 46.3     | 70.2     | 36.4     | 40.7      | 48.4  |
|             | *nl*        | 42.9     | 68.4     | 36.5     | 40.9      | 47.1  |
|             | *ro*        | 43.2     | 66.0     | 35.7     | 39.8      | 46.2  |
| Falcon-7B   | *fr*        | 37.3     | 64.1     | 28.4     | 34.0      | 40.9  |
| Mistral-7B  | *de*        | 41.2     | 58.7     | 40.5     | 44.9      | 46.3  |
|             | *es*        | 44.2     | 65.3     | 42.4     | 43.1      | 48.7  |
|             | *fr*        | 44.9     | 64.4     | 41.9     | 43.0      | 48.6  |
|             | *it*        | 43.2     | 60.9     | 39.7     | 43.1      | 46.7  |
|             | *nl*        | 40.0     | 57.9     | 41.4     | 43.3      | 45.7  |
|             | *ro*        | 40.7     | 53.6     | 39.3     | 43.6      | 44.3  |
| Llama-7B    | *de*        | 35.1     | 49.9     | 29.9     | 38.3      | 38.3  |
|             | *es*        | 36.8     | 56.4     | 30.3     | 37.0      | 40.1  |
|             | *fr*        | 37.3     | 55.7     | 30.5     | 39.9      | 40.9  |
|             | *it*        | 35.8     | 52.0     | 29.9     | 39.6      | 39.3  |
|             | *nl*        | 33.6     | 48.7     | 29.8     | 40.0      | 38.0  |
|             | *ro*        | 32.4     | 44.9     | 29.7     | 37.0      | 36.0  |
| Bloom-7B    | *de*        | 26.3     |  32.4     | 28.1    | 43.7      | 32.6  |
|             | *es*        | 38.1     | 56.7     | 28.9     | 40.4      | 41.0  |
|             | *fr*        | 36.7     | 56.6     | 29.9     | 40.9      | 41.0  |
|             | *it*        | 29.0     | 40.8     | 27.6     | 43.7      | 35.3  |
|             | *nl*        | 23.1     | 31.7     | 27.5     | 42.7      | 31.3  |
|             | *ro*        | 26.9     | 31.8     | 27.4     | 46.1      | 33.1  |

In the spirit of the original Falcon models, the Falcon2-11B was trained not only on English data but also on ten other languages. Our multilingual evaluation results show that the model presents good capabilities in the six languages (*de*, *es*, *fr*, *it*, *nl*, *ro*) featured on the Multilingual LLM Leaderboard and actually shows higher performance that the Falcon-40B and several other multilingual models on all the cited languages.

We will soon release more extensive evaluation results for multilingual capabilities in the [Falcon2-11B model card](https://huggingface.co/tiiuae/falcon-11B)!

### Code genration capabilities

We check the model's performance on code generation against the [BigCode Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) on the HumanEval benchmark for the Python language, obtaining pass@1 of 29.59%.

## [Using Falcon2-11B](#using)
```python
from transformers import AutoTokenizer
import transformers
import torch

model = "tiiuae/falcon-11B"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```
And then, you'd run text generation using code like the following:
```python
sequences = pipeline(
   "Can you explain the concept of Quantum Computing?",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

# [Falcon2-11B VLM](#vlm-training)
The [Falcon2-11B VLM](https://huggingface.co/tiiuae/Falcon-11B-vlm) is a vision-language model (VLM) for additionally handling image inputs and answering the queries corresponding to the images. To achieve this, we integrate the pretrained CLIP ViT-L/14 vision encoder with our Falcon2-11B chat-finetuned model and train with image-text data. 
For enhancing the VLM's perception of fine-grained details w.r.t small objects in images, we employ a dynamic encoding mechanism at high-resolution for image inputs, similar to [LLaVA-Next](https://llava-vl.github.io/blog/2024-01-30-llava-next/).

### Training 
The training is done in two stages: pretraining and finetuning. In both stages, the visual encoder weights are kept frozen. In the pretraining stage, the LLM is kept frozen, and only the multimodal projector is trained on 558K image-caption pairs. 
This enables the multimodal projector to learn a mapping from visual to text embedding space. During finetuning, both the projector and LLM weights are trained on a corpus of 1.2M image-text instruction data from public datasets, which also includes multi-round conversations.

### [Evaluation](#vlm-evaluation)

| Model | MME | GQA | SQA | POPE | VQAv2 | TextVQA | MM-Bench | SEED-IMG | Average |
|----|----|----|----|----|----|----|----|----|----|
| Falcon2-11B VLM | **1589/343** | 64.5 | **74.9** | **88.4** | 82.1 | 66.7 | **72.0** | **72.3** |**74.4** |
| LLaVA-1.6 (Vicuna-7B) | 1519/332 | 64.2 |  70.1 | 86.5 | 81.8 | 64.9 | 67.4 | 70.2 | 72.1 |
| LLaVA-1.6 (Vicuna-13B) | 1575/326 | **65.4** | 73.6 | 86.2 | **82.8** | **67.1** | 70.0 | 71.9 |73.8 |
| LLaVA-1.6 (Mistral-7B) | 1498/321 |64.8 | 72.8 | 86.7 | 82.2 | 65.7 | 68.7 | 72.2 |73.3 |


## [Using Falcon2-11B-FalconVLM](#vlm-using)
```python
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import requests
import torch

processor = LlavaNextProcessor.from_pretrained("tiiuae/falcon-11B-vlm")
model = LlavaNextForConditionalGeneration.from_pretrained("tiiuae/falcon-11B-vlm", torch_dtype=torch.bfloat16)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
cats_image = Image.open(requests.get(url, stream=True).raw)
prompt = 'User: <image>\nWrite a long paragraph about this picture.'

inputs = processor(prompt, images=cats_image, return_tensors="pt", padding=True).to('cuda:0')

model.to('cuda:0')
output = model.generate(**inputs, max_new_tokens=256)


prompt_length = inputs['input_ids'].shape[1]
generated_captions = processor.decode(output[0], skip_special_tokens=True).strip()

print(generated_captions)
```

# [License information](#license)

The Falcon2 models are made available under the [TII Falcon License 2.0](https://falconllm-staging.tii.ae/falcon-2-terms-and-conditions.html), the permissive Apache 2.0-based software license which includes an [acceptable use policy](https://falconllm-staging.tii.ae/falcon-2-acceptable-use-policy.html) that promotes the responsible use of AI. This license was crafted within the spirit of TII's commitment to the open source community.
