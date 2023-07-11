---
title: "Open-Source Text Generation & LLM Ecosystem at Hugging Face"
thumbnail: /blog/assets/os_llms/thumbnail.png
authors:
- user: merve
---

<h1>Open-Source Text Generation & LLM Ecosystem at Hugging Face</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->


Text generation and conversational technologies have been around for ages. Earlier challenges in working with these technologies were controlling both the coherence and diversity of the text through inference parameters and discriminative biases. More coherent outputs were less creative and closer to the original training data and sounded less human. Recent developments overcame these challenges, and user-friendly UIs enabled everyone to try these models out. Services like ChatGPT have recently put the spotlight on powerful models like GPT-4 and caused an explosion of open-source alternatives like LLaMA to go mainstream. We think these technologies will be around for a long time and become more and more integrated into everyday products. In this post, we will go through a brief background on how they work, the types of text generation models that exist, and the Hugging Face tools you can use to incorporate open-source LLMs into your products.

## Brief Background on Text Generation

Text generation models are essentially trained with the objective of completing an incomplete text or generating text in the form of a response to a given instruction or question. Models that complete incomplete text are called Causal Language Models, and famous examples are GPT-3 by OpenAI and LLaMa by Meta AI. 

![Causal LM Output](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/os_llms/text_generation.png)

Causal language models are optimized using a process called reinforcement learning from human feedback (RLHF). This optimization is mainly made over how natural and coherent the text sounds rather than the validity of the answer. Explaining how RLHF works is outside the scope of this blog post, but you can find more information about this process [here](https://huggingface.co/blog/rlhf).

One concept you need to know before we move on is fine-tuning. This is the process of taking a very large model and transferring the knowledge contained in this base model to the use case: a downstream task. These tasks can come in the form of instructions. As the model size grows, the models can generalize better to the instructions that do not exist in the pre-training data.

For example, the base model GPT-3 is a base causal language model, and the models in the backend of the ChatGPT (which is the UI for GPT-series models) are fine-tuned on prompts that can consist of conversations or instructions through RLHF. It’s an important distinction to make between these models. 

On Hugging Face Hub, you can find both causal language models and causal language models fine-tuned on instruction (which we’ll give links to later in this blog post). LLaMA is one of the first open-source LLMs to outperform closed-source ones. A research group led by Together has created a reproduction of LLaMA's dataset, called Red Pajama, and trained LLMs and instruction fine-tuned models on it. You can read more about it [here](https://www.together.xyz/blog/redpajama) and find [the model checkpoints on Hugging Face Hub](https://huggingface.co/models?sort=trending&search=togethercomputer%2Fredpajama). By the time this blog post is written, three of the largest causal language models with open-source licenses are [MPT-30B by MosaicML](https://huggingface.co/mosaicml/mpt-30b), [XGen by Salesforce](https://huggingface.co/Salesforce/xgen-7b-8k-base) and [Falcon by TII UAE](https://huggingface.co/tiiuae/falcon-40b), available completely open on Hugging Face Hub.

The second type of text generation model is commonly referred to as the text-to-text generation model. These models are trained on text pairs, which can be questions and answers or instructions and responses. The most popular ones are T5 and BART (which, as of now, aren’t state-of-the-art). Google has recently released the FLAN-T5 series of models. FLAN is a recent technique developed for instruction fine-tuning, and FLAN-T5 is essentially T5 fine-tuned using FLAN. As of now, the FLAN-T5 series of models are state-of-the-art and open-source, available on [Hugging Face Hub](https://huggingface.co/models?search=google/flan). Below you can see an illustration of how these models work.

![FLAN-T5 Illustration](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/os_llms/flan_t5.png)

Having more variation of open-source text generation models enables companies to keep privacy with their data, ability to adapt models to their domains quicker, and cut costs for inference instead of relying on closed paid APIs. All open-source causal language models on Hugging Face Hub can be found [here](https://huggingface.co/models?pipeline_tag=text-generation) and text-to-text generation models can be found [here](https://huggingface.co/models?pipeline_tag=text2text-generation&sort=trending).

Snippets to use these models are given in either the model repository or the documentation page of that model type in Hugging Face.

## Licensing

Many text generation models are either closed-source or the license limits commercial use. Fortunately, open-source alternatives are starting to appear and being embraced by the community as building blocks for further development, fine-tuning, or integration with other projects. Some notable fully open-source models include [MPT-30B](https://huggingface.co/mosaicml/mpt-30b) and [Falcon](https://huggingface.co/tiiuae/falcon-40b). They both are causal language models distributed with the permissive Apache 2.0 license that allows commercial use.

The Hugging Face Hub also hosts various models fine-tuned for instruction or chat use. They come in various styles and sizes depending on your needs:

[MPT-30B-Chat](https://huggingface.co/mosaicml/mpt-30b-chat), by Mosaic ML, uses the CC-BY-NC-SA license, which does not allow commercial use. However, [MPT-30B-Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct) uses CC-BY-SA 3.0, which can be used commercially.

[Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) and  [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) both use the Apache 2.0 license, so commercial use is also permitted. Another popular model is OpenAssistant, built on Meta's LLaMa model using a custom instruction-tuning dataset. Since the original LLaMa model can only be used for research, the OpenAssistant checkpoints built on LLaMa don’t have full open-source licenses. However, there are OpenAssistant models built on open-source models like [Falcon](https://huggingface.co/models?search=openassistant/falcon) or [pythia](https://huggingface.co/models?search=openassistant/pythia) that use permissive licenses.

Finally, the instruction-tuned [XGen model](https://huggingface.co/Salesforce/xgen-7b-8k-inst) only allows research use.

If you're looking to fine-tune a model on an existing instruction dataset, you need to know how a dataset was compiled. Some of the existing instruction datasets are either crowd-sourced or use outputs of existing models (e.g., the models behind ChatGPT). ALPACA dataset created by Stanford is created through the outputs of models behind ChatGPT, which OpenAI prohibits using for training models. Moreover, there are various crowd-sourced instruction datasets with open-source licenses, like [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (created by thousands of people voluntarily!) or [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k). If you'd like to create a dataset yourself, you can check out [the dataset card of Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k#sources) on how to create an instruction dataset. Models fine-tuned on these datasets can be distributed. 

### How can you serve these models?

Response time and latency for concurrent users are a big challenge for serving these large models. To tackle this problem, Hugging Face has released [text-generation-inference](https://github.com/huggingface/text-generation-inference) (TGI), an open-source serving solution for large language models built on Rust, Python, and gRPc. TGI is integrated into inference solutions of Hugging Face, [Inference Endpoints](https://huggingface.co/inference-endpoints), and [Inference API](https://huggingface.co/inference-api), so you can directly create an endpoint with optimized inference with few clicks, or simply send a request to Hugging Face's Inference API to benefit from it, instead of integrating TGI to your own platform. 

![Screenshot from HuggingChat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/os_llms/huggingchat_ui.png)

TGI currently powers [HuggingChat](https://huggingface.co/chat/), Hugging Face's open-source chat UI for LLMs. This service currently uses OpenAssistant as the backend model. You can chat as much as you want with HuggingChat and enable the search feature for validated responses. You can also give feedback to each response for model authors to train better models. The UI of HuggingChat is also [open-sourced](https://github.com/huggingface/chat-ui), and we are working on more features for HuggingChat to allow more functions, like generating images inside the chat.

![HuggingChat Search](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/os_llms/huggingchat_web.png)

### How to find the best model?

Hugging Face hosts an LLM leaderboard [here](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). This leaderboard is created by evaluating community-submitted models on text generation benchmarks on Hugging Face’s clusters. If you can’t find the language or domain you’re looking for, you can filter them [here](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

![Open LLM Leaderboard](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/os_llms/LLM_leaderboard.png)

### Models created with love by Hugging Face with BigScience and BigCode

Hugging Face has co-led two science initiatives, BigScience and BigCode. As a result of them, two large models were created, [BLOOM](https://huggingface.co/bigscience/bloom) 🌸 and [StarCoder](https://huggingface.co/bigcode/starcoder) 🌟. StarCoder is a causal language model trained on code from GitHub (with 80+ programming languages 🤯). It’s not fine-tuned on instructions, and thus, it serves more as a coding assistant to complete a given code, e.g., translate Python to C++, explain concepts (what’s recursion), or act as a terminal. You can try all of the StarCoder checkpoints [in this application](https://huggingface.co/spaces/bigcode/bigcode-playground). It also comes with a [VSCode extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode).

BLOOM is a causal language model trained on 46 languages and 13 programming languages. It is the first open-source model to have more parameters than GPT-3. You can find available checkpoints in [BLOOM documentation](https://huggingface.co/docs/transformers/model_doc/bloom).

### Parameter Efficient Fine Tuning (PEFT)

If you’d like to fine-tune one of the existing large models on your own instruction dataset, it is nearly impossible to do so on consumer hardware and later deploy them (since the instruction models are the same size as the original checkpoints that are used for fine-tuning). [PEFT](https://huggingface.co/docs/peft/index) is a library that allows you to do parameter-efficient fine-tuning techniques. This means that rather than training the whole model, you can train a very small number of additional parameters, enabling much faster training with very little performance degradation. With PEFT, you can do low-rank adaptation (LoRA), prefix tuning, prompt tuning, and p-tuning.

You can check out further resources for more information on text generation.

**Further Resources**
- AWS has released TGI-based LLM deployment deep learning containers called LLM Inference Containers. Read about them [here](https://aws.amazon.com/tr/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/).
- [Text Generation task page](https://huggingface.co/tasks/text-generation) to find out more about the task itself.
- PEFT announcement [blog post](https://huggingface.co/blog/peft).
- Read about how Inference Endpoints utilizes TGI [here](https://huggingface.co/blog/inference-endpoints-llm).
