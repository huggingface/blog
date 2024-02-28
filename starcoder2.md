---
title: "StarCoder2 and The Stack v2" 
thumbnail: /blog/assets/177_starcoder2/sc2-banner.png
authors:
- user: lvwerra
- user: loubnabnl
- user: anton-l
- user: nouamanetazi
---

# StarCoder2 and The Stack v2

<div class="flex items-center justify-center">
<img src="https://huggingface.co/datasets/bigcode/admin/resolve/main/sc2-banner.png" alt="StarCoder2">
</div>

BigCode is releasing StarCoder2, the next generation of transparently trained open code LLMs. All StarCoder2 variants were trained on [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2/), a new large and high-quality code dataset. We release all models, datasets, and the processing as well as the training code. Check out the [paper](https://drive.google.com/file/d/17iGn3c-sYNiLyRSY-A85QOzgzGnGiVI3/view?usp=sharing) for details.

## What is StarCoder2?

StarCoder2 is a family of open LLMs for code and comes in 3 different sizes with 3B, 7B and 15B parameters. The flagship StarCoder2-15B model is trained on over 4 trillion tokens and 600+ programming languages from The Stack v2. All models use Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and were trained using the Fill-in-the-Middle objective. 

StarCoder2 offers three model sizes: a 3 billion-parameter model trained by ServiceNow, a 7 billion-parameter model trained by Hugging Face, and a 15 billion-parameter model trained by NVIDIA using NVIDIA NeMo on NVIDIA accelerated infrastructure:

- [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b) was trained on 17 programming languages from The Stack v2 on 3+ trillion tokens.
- [StarCoder2-7B](https://huggingface.co/bigcode/starcoder2-7b) was trained on 17 programming languages from The Stack v2 on 3.5+ trillion tokens.
- [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b) was trained on 600+ programming languages from The Stack v2 on 4+ trillion tokens.

StarCoder2-15B is the best in its size class and matches 33B+ models on many evaluations. StarCoder2-3B matches the performance of StarCoder1-15B:

<div class="flex items-center justify-center">
<img src="https://huggingface.co/datasets/bigcode/admin/resolve/main/sc2-evals.png" alt="StarCoder2 Evaluation">
</div>

## What is The Stack v2?

<div class="flex items-center justify-center">
<img src="https://huggingface.co/datasets/bigcode/admin/resolve/main/stackv2-banner.png" alt="The Stack v2">
</div>

The Stack v2 is the largest open code dataset suitable for LLM pretraining. The Stack v2 is larger than The Stack v1, follows an improved language and license detection procedure, and better filtering heuristics. In addition, the training dataset is grouped by repositories, allowing to train models with repository context. 

||[The Stack v1](https://huggingface.co/datasets/bigcode/the-stack/)|[The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2/)|
|-|-|-|
| full | 6.4TB | 67.5TB |
| deduplicated | 2.9TB | 32.1TB | 
| training dataset | ~200B tokens | ~900B tokens |

This dataset is derived from the Software Heritage archive, the largest public archive of software source code and accompanying development history. Software Heritage, launched by Inria in partnership with UNESCO, is an open, non-profit initiative to collect, preserve, and share the source code of all publicly available software. We are grateful to Software Heritage for providing access to this invaluable resource. For more details, visit the [Software Heritage website](https://www.softwareheritage.org).

The Stack v2 can be accessed through the [Hugging Face Hub](https://huggingface.co/datasets/bigcode/the-stack-v2/).

## About BigCode

BigCode is an open scientific collaboration led jointly by Hugging Face and ServiceNow that works on the responsible development of large language models for code.

## Links

### Models
- [Paper](https://drive.google.com/file/d/17iGn3c-sYNiLyRSY-A85QOzgzGnGiVI3/view?usp=sharing): A technical report about StarCoder2 and The Stack v2.
- [GitHub](https://github.com/bigcode-project/starcoder2/): All you need to know about using or fine-tuning StarCoder2.
- [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b): Small StarCoder2 model.
- [StarCoder2-7B](https://huggingface.co/bigcode/starcoder2-7b): Medium StarCoder2 model.
- [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b): Large StarCoder2 model.

### Data & Governance
- [StarCoder2 License Agreement](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement): The model is licensed under the BigCode OpenRAIL-M v1 license agreement.
- [StarCoder2 Search](https://huggingface.co/spaces/bigcode/search-v2): Full-text search for code in the pretraining dataset.
- [StarCoder2 Membership Test](https://stack.dataportraits.org): Blazing fast check of code that was present in the pretraining dataset.

### Others
- [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode): Code with StarCoder!
- [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

You can find all the resources and links at [huggingface.co/bigcode](https://huggingface.co/bigcode)!
