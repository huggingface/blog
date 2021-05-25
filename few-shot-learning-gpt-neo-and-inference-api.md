---
title: 'Few-shot learning in practice: GPT-Neo and the 🤗 Accelerated Inference API'
# thumbnail: /blog/assets/22_few_shot_learning_gpt_neo_and_inference_api/thumbnail.png
---

<h1>
    Few-shot learning in practice: GPT-Neo and the 🤗 Accelerated Inference API
</h1>

<div class="blog-metadata">
    <small>Published May 30, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/sagemaker-distributed-training-seq2seq.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/philschmid">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1613142338662-5ff5d596f244529b3ec0fb89.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>philschmid</code>
            <span class="fullname">Philipp Schmid</span>
        </div>
    </a>
</div>

<script defer src="https://gpt-neo-accelerated-inference-api.s3-eu-west-1.amazonaws.com/fewShotInference.js"></script>
<few-shot-inference-widget ></few-shot-inference-widget>


## What is few-shot learning?

Few-Shot learning refers to the practice of feeding a learning model (language model) with a very small amount of data, contrary to the normal practice of fine-tuning using a large amount of data.

This technique has been mostly used in computer vision, but with some of the latest Language Models, like [EleutherAI GPT-Neo](https://www.eleuther.ai/projects/gpt-neo/) and [OpenAI GPT-3](https://openai.com/blog/gpt-3-apps/), we can now use it in Natural Language Processing (NLP). 

In NLP few-shot learning is used together with a large Language Model. These Language Models learned to perform a large number of tasks implicitly in order to reduce perplexity during their pre-training on large text datasets. This enables the Language model to generalize/understand unseen (but related) tasks with just a few examples.

Few-Shot NLP examples consist of three main components: 

- **Task Description:** A short description of what the model should do, e.g. translate English to France
- **Examples:** A few shot examples demonstrating/showing the model what it should do, e.g. sea otter => loutre de mer
- **Prompt:** The beginning of example where the model should generate the missing text for completion, e.g. cheese =>

![few-shot-prompt](assets/22_few_shot_learning_gpt_neo_and_inference_api/few-shot-prompt.png)  
<small>Image from <a href="https://arxiv.org/abs/2005.14165" target="_blank">Language Models are Few-Shot Learners</a></small>

Creating these few-shot examples can be tough since you need to elaborate the “task” you want to perform to the model. A common problem is that models, especially smaller ones seem to be very sensitive to the way the example is written.

OpenAI contributed in their [GPT-3 Paper](https://arxiv.org/abs/2005.14165) an observation that the few-shot prompting ability grows as the number of language model parameters grows.

![few-shot-performance](assets/22_few_shot_learning_gpt_neo_and_inference_api/few-shot-performance.png)  
<small>Image from <a href="https://arxiv.org/abs/2005.14165" target="_blank">Language Models are Few-Shot Learners</a></small>

An approach to optimize Few-Shot Learning in production is to learn a common representation for a task and then train task-specific classifiers on top of this representation.

---

## What is GPT-Neo?

GPT⁠-⁠Neo is a family of transformer-based language models from [EleutherAI](https://www.eleuther.ai/projects/gpt-neo/) based on the GPT architecture. [EleutherAI](https://www.eleuther.ai)'s primary goal is to train a model that is equivalent in size to GPT⁠-⁠3 and make it available to the public under an open license.

All of the currently available GPT-Neo checkpoints are trained with the Pile dataset, a large text corpus that is extensively documented in ([Gao et al., 2021](https://arxiv.org/abs/2101.00027)). As such, it is expected to function better on the text that matches the distribution of its training text; we recommend keeping this in mind when designing systems that rely on its output and in considering how the system might impact different groups of users. For further discussion on these questions, we refer you to e.g. ([Bender et al., 2021](https://dl.acm.org/doi/10.1145/3442188.3445922))

---

## 🤗 Accelerated Inference API

The Accelerated Inference API is our hosted service to run inference on any of the 10,000+ models publicly available on the 🤗 Model Hub, or your own private models, via simple API calls. The API includes acceleration on CPU and GPU with [up to 100x speedup](https://huggingface.co/blog/accelerated-inference) compared to out of the box deployment of Transformers.

To integrate Few-Shot Learning predictions with `GPT-Neo` in your own apps, you can use the 🤗 Accelerated Inference API with the code snippet below. You can find your API Token [here](https://huggingface.co/settings/token), if you don't have an account you can get started [here](https://huggingface.co/pricing).

```python
import json
import requests

API_TOKEN = ""

def query(payload='',parameters=None,options={'use_cache': False}):
    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
		headers = {"Authorization": f"Bearer {API_TOKEN}"}
    body = {"inputs":payload,'parameters':parameters,'options':options}
    response = requests.request("POST", API_URL, headers=headers, data= json.dumps(body))
    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
        return "Error:"+" ".join(response.json()['error'])
    else:
      return response.json()[0]['generated_text']

parameters = {
    'max_new_tokens':25,  # number of generated tokens
    'temperature': 0.5,   # controlling the randomness of generations
    'end_sequence': "###" # stopping sequence for generation
}

prompt="...."             # few-shot prompt

data = query(prompt,parameters,options)
```

---

## Unfair advantage

Our partnerships and open-source collaborations with hardware and cloud vendors like Intel, NVIDIA, Qualcomm, Amazon, and Microsoft enable us to tune our API infrastructure with the latest hardware optimization techniques.

As Machine Learning Engineers at Hugging Face we certainly have an unfair advantage sitting in the same (virtual) offices as the 🤗 Transformers and 🤗 Tokenizers maintainers.

### When You/your company should use the Accelerated Inference API:

- When your idea/product wants to benefit from rapid development through the easy integration of all ~11 000 models of the [Hugging Face Hub](https://huggingface.co/models).
- When you want to get up to [100x performance optimizations](https://huggingface.co/blog/accelerated-inference) out of the box.
- When you don't have in-house expertise on how to host billion parameter large NLP models
- When you want transparency which model is used for inference compared to, e.g. [Amazon Comprehend](https://aws.amazon.com/de/comprehend/) or [Google Cloud NLP](https://cloud.google.com/natural-language)

---

If you want to feel the speed on our infrastructure, start a [free trial](https://huggingface.co/pricing) and we’ll get in touch. If you want to benefit from our experience optimizing inference on your own infrastructure participate in our [🤗 Expert Acceleration Program](https://huggingface.co/support).
