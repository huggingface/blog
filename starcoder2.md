---
title: "StarCoder2 and The Stack v2" 
thumbnail: **todo**
authors:
- user: lvwerra
- user: loubnabnl
- user: anton-l
---

# StarCoder2 and The Stack v2

**TODO** add SC2 banner 


With StarCoder2 BigCode is releasing the next generation transparently trained code LLMs. All models were trained on The Stack v2, a significantly larger and higher quality code dataset. We release all models, datasets, and the processing as well as training code .

## StarCoder2

StarCoder2-15B model is a 15B parameter model trained on 600+ programming languages from The Stack v2, with opt-out requests excluded. The model uses Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and was trained using the Fill-in-the-Middle objective on 4+ trillion tokens. 

StarCoder2 offers three model sizes: a 3 billion-parameter model trained by ServiceNow, a 7 billion-parameter model trained by Hugging Face, and a 15 billion-parameter model built by NVIDIA with NVIDIA NeMo and trained on NVIDIA accelerated infrastructure. 

StarCoder2-15B is the best in it's size class and matching 33B+ models in many instances:

**TODO** add eval plot

## The Stack v2

**TODO** the stack banner

The Stack v2 is the largest open code dataset suitable for LLM pretraining. The Stack v2 is larger, follows an improved language and license detection procedure, and better filtering heuristics. In addition, the training dataset is grouped by repositories, allowing to train models with repository context. 

||The Stack v1|The Stack v2|
|-|-|-|
| full | 6.4TB | 67.5TB |
| deduplicated | 2.9TB | 32.1TB | 
| training dataset | ~200B tokens | ~900B tokens |

 This dataset is derived from the Software Heritage archive, the largest public archive of software source code and accompanying development history. Software Heritage is an open, non profit initiative to collect, preserve, and share the source code of all publicly available software, launched by Inria, in partnership with UNESCO. We acknowledge Software Heritage for providing access to this invaluable resource. For more details, visit the [Software Heritage website](https://www.softwareheritage.org).

## About BigCode

BigCode is an open scientific collaboration led jointly by Hugging Face and ServiceNow that works on the responsible development of large language models for code.

## Links

### Models
- [Paper](): A technical report about StarCoder2 and The Stack v2.
- [GitHub](https://github.com/bigcode-project/starcoder2/): All you need to know about using or fine-tuning StarCoder2.
- [StarCoder2](https://huggingface.co/bigcode/starcoder2-15b): Model on the Hugging Face Hub.

### Tools & Demos
- [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode): Code with StarCoder!
- [StarCoder Playground](https://huggingface.co/spaces/bigcode/bigcode-playground): Write with StarCoder!
- [StarCoder Editor](https://huggingface.co/spaces/bigcode/bigcode-editor): Edit with StarCoder!

### Data & Governance
- [StarCoder License Agreement](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement): The model is licensed under the BigCode OpenRAIL-M v1 license agreement.
- [StarCoder Search](https://huggingface.co/spaces/bigcode/search-v2): Full-text search code in the pretraining dataset.
- [StarCoder Membership Test](https://stack.dataportraits.org): Blazing fast test if code was present in pretraining dataset.

You can find all the resources and links at [huggingface.co/bigcode](https://huggingface.co/bigcode)!
