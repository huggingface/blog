---
title: "Jupyter Agents: training LLMs to reason with notebooks"
thumbnail: /blog/assets/jupyter-agent-2/thumbnail.png
authors:
- user: baptistecolle
- user: hannayukhymenko
- user: lvwerra
---

# Jupyter Agents: training LLMs to reason with notebooks

The past year has been all about giving LLMs more tools and autonomy to solve more complex and open ended tasks. The goal of the **Jupyter Agent** is to give the model the ultimate tool: code execution. 

A natural way to display mutli-step code execution together reasoning is within a Jupyter Notebook with code and markdown cells. So we built Jupyter Agent to act as an agent that can execute code directly inside a Jupyter notebook and use this environment to solve data analysis and data science tasks. Think of it like *Cursor*, but living natively inside your data science workflow.  
We built a [demo](https://huggingface.co/spaces/lvwerra/jupyter-agent-2) of this vision with **Qwen-3 Coder**, currently one of the strongest coding models. This is a follow-up to our earlier work on [jupyter-agent (v1)](https://huggingface.co/spaces/lvwerra/jupyter-agent).


While large models are starting to show useful behavior, the key question is how we can continue improving them. To this end, we focus on strengthening smaller models to perform well on agentic data science tasks as they currently struggle to compete with the large models.

The goal of this project is to build a pipeline to first generate high-quality training data, then fine-tune an existing small model, and finally evaluate whether the model's performance improves on relevant benchmarks.

Let’s begin with the last step: selecting a strong benchmark for evaluating models on data science tasks.

## 🏁 Primer: the DABStep Benchmark

In order to understand if we are making progress towards better data science agents we need a benchmark to measure such capabilities. Last year, in partnership with Adyen, we introduced the **DABStep benchmark**: a way to evaluate data science agents on realistic tasks. The setup is simple: provide the LLM with datasets and ask it to answer non-trivial data questions.  

Example tasks:

| Question | Answer |
|----------|---------|
| Which card scheme had the highest average fraud rate in 2023? | SwiftCharge |
| For the year 2023, focusing on the merchant *Crossfit Hanna*, if we incentivize users to switch to a different Authorization Characteristics Indicator, which option would be the most cost-effective? | E:346.49 |

This benchmark remains challenging for today’s LLMs — e.g. the best out-of-the-box model is Claude 4 Sonnet which reaches not even 20% accuracy on the hard tasks.  
You can explore the live leaderboard [here] (https://huggingface.co/spaces/adyen/DABstep).

## 🎯 First Baseline

Now that we identified a good benchmark we can try to climb it! We set out to build a dataset for fine-tuning such that  even **a small data agent model** could perform well on DABStep.  

Our first choice was [**Qwen3-4B-Thinking-2507**](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507): extremely small (fast to iterate with, easy to run), yet strong enough to act in agentic scenarios.  

Baseline results:  
- *Easy tasks:* **44.4%**  
- *Hard tasks:* **2.1%**  

Not great — but a promising starting point, since it left a lot of room for improvement. Let's see how we can improve it!

## 🔧 Primer on Scaffolding

A core aspect of agents that sets it apart from a pure chat model is the scaffolding built around the model to steer its behaviour. The evaluation script in DABStep for example uses [smolagents](https://github.com/huggingface/smolagents) to execute code. Smolagents comes with predefined behaviors, prompting structures, and expected formats.  

We also studied the [**Qwen-Agent**](https://github.com/QwenLM/Qwen-Agent) codebase, where the authors tailoring scaffolding to the model. This makes sense: Claude Code, for example, works shockingly well with Claude Sonnet because their scaffolding is aligned.  

So, we restructured our scaffolding:  
- Stripped it down to ~200 lines of code.  
- No external dependencies.  
- Inspired by the spirit of [**tiny-agents**](https://huggingface.co/blog/tiny-agents).  

👉 Check it out here: [utils.py](https://huggingface.co/spaces/lvwerra/jupyter-agent-2/blob/main/utils.py).

**Results:** accuracy jumped from **44.4% → 59.7% (easy split)**. 🚀  

**Our loop:**  
- While loop with two tools: *code execution* to run the code and *final_answer* to return the final answer.  
- We differ from Qwen-Agent by explicitly adding a **final_answer** tool — which in our testing has improved performance.  
- Compared to smolagents, we simplified the scaffolding by removing a lot of prompts and tools. Smolagents also hardcode a lot of assumptions into the model by using the ReACT framework.

## 🏃‍♂️ Training Pipeline

With simplified scaffolding in place, we focused on fine-tuning Qwen3-4B for **data science agentic tasks**.  

## ⚙️ Dataset Pipeline

The recipe to improve a model on a certain task or behaviour is to train it on data that reflects the tasks as closely as possible. A natural starting point is to look at real Jupyter Notebooks and find notebooks that align closely with the task that we plan to tackle, namely data analysis. 

Kaggle notebooks offer a wealth of high quality data analysis notebooks and are made available by Kaggle:

**Datasets:**  
- **Kaggle Notebooks dataset**: ~2TB of notebooks.
- **Kaggle Datasets**: 5TB of kaggle datasets that we manually downloaded and linked to the notebooks.
- Rich metadata for each notebook (authors, datasets used, etc.).  

Now that we have good results with a base model it's time to build a dataset that will help us improve it even further. We designed a multi-stage pipeline using [Datatrove](https://github.com/huggingface/datatrove) to clean and prepare Kaggle notebooks at scale.  

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/jupyter-agent-dataset-pipeline.png" alt="Jupyter Agent Dataset Pipeline"/>

Here’s how each step worked:

### 1. Large-scale deduplication
We started with ~2TB of Kaggle notebooks and reduced it to ~250GB reusing our work from the BigCode project. As part of the StarCoder2 training data processing the notebooks (without output cells) were already deduplicated.
Most Kaggle notebooks are small variations or near-identical copies, so this step was essential.  
*Key insight:* ~90% of raw notebooks are duplicates, which would have skewed training if left unfiltered.

### 2. Downloading linked datasets
Most Kaggle notebooks reference external datasets via Kaggle metadata. To make sure the code inside notebooks could actually run, we built a pipeline that automatically fetched these linked datasets. This step was crucial, since many notebooks would otherwise be incomplete or non-executable.  

Using the **kagglehub** package, we downloaded thousands of datasets — about **5TB** in total. To keep things manageable and relevant:  
- We filtered out datasets containing model checkpoints, large multimodal corpora, or LLM-related files.  
- We also excluded very large datasets (10GB+) that couldn’t fit into the virtual [E2B sandboxes](https://e2b.dev/) we used for execution.  

By the end, we had a rich collection of executable notebooks paired with their datasets, providing the foundation for training agents in realistic, runnable environments.

### 3. Edu scoring
We scored notebooks based on educational quality using [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B). We saw that using the whole notebook was not optimal, as many contained trivial or broken code. Our educational scoring approach is detailed in edu_scoring.py.

**TL;DR:** We assigned each notebook a score from 1–5 based on clarity, completeness, and educational value, and kept only those above a chosen threshold. This filtering removed about 70% of the notebooks.

This is similar to the insight from the BeyondWeb paper, which showed that using high-quality data is better for synthetic data generation — a step we relied on for QA (Question-Answer) generation.
This helped the model learn from “high quality” notebooks instead of noisy ones.

### 4. Filtering irrelevant notebooks
We excluded notebooks about training LLMs or unrelated to data analysis.
We also removed notebooks that didn’t actually use datasets through an automated LLM-based filtering process using Qwen3-32B. The implementation of filtering can be found in [`extract_packages_and_files.py`](https://github.com/huggingface/jupyter-agent/blob/main/data/pipelines/extract_packages_and_files.py).

**TL;DR:** We prompted Qwen3-32B to identify and remove notebooks that either (1) had nothing to do with data analysis, or (2) didn’t actually use datasets. This step removed about 20% of the notebooks.

This ensured we trained only on relevant data science tasks.

### 5. QA generation
Using the cleaned notebooks, we generated question–answer pairs using [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B). The questions and answer are grounded in the real notebook traces so the QA pairs are based on real code execution results.
**Prompt design:** we asked the LLM to produce natural questions that could realistically be asked of the dataset, then validated whether the notebook provided a correct answer.  

*Challenge:* We had to try many prompts to get higher-difficulty questions because LLMs tended to generate trivial ones like "what is the size of the dataset".  
*Insight:* We broke this into two steps because LLMs tended to hallucinate answers:  
1. Generate the question and answer.  
2. Ask another LLM (with access to the notebook) to check whether the answer was correct. 

The complete prompting strategy and implementation is available in [`qa_generation.py`](https://github.com/huggingface/jupyter-agent/blob/main/data/pipelines/qa_generation.py).

### 6. Trace generation
Finally we want to generate clean code executions traces since even the original notebooks after processing are often open ended and verbose with lots of irrelevant parts. However, we want our Jupyter Agent to get to the result efficiently. To generate cleaner notebook traces for training we generated traces synthetically based on the original notebooks.  
We have prompted [Qwen-3-Coder-480B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) model to generate a jupyter notebook code to answer the question from the previously generated synthetic QA pair. 
Traces captured step-by-step code execution, including intermediate outputs, which are crucial for agent training.  

We used [E2B](https://e2b.dev/) for our agent to solve the synthetic QA pairs, which required fetching Kaggle datasets so the code could actually run via E2B.  

*Challenge 1:* Many datasets were unavailable.  
*Trick:* Since LLMs are strong at code and have a decent world model, we prompted them to **act as a code interpreter** when the dataset was missing.  

Beginning of the prompt:
```
You are a stateful Python code interpreter that executes code in a persistent environment. Your role is to execute Python code while maintaining state across multiple code cells, similar to a Jupyter notebook environment.
[REST OF THE PROMPT]
```

*Challenge 2:* [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) model does not support thinking mode - how can we extract code commentary? By default it often outputs just a brief comment followed by several steps of code execution. However, we'd like some reasoning or comments between every cell. 
*Trick:* When switching from [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) to [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) we noticed that often output message content was empty. This turns out to be a previously known quirk of Qwen3-Coder models in which when using tool calling the model would not return an empty assistant response. We enforce some text commentary through tooling by passing 'comment' as a required field in the code execution tool call. This way when non-reasoning model is used for code cell generation it will by default output some description of its actions from 1st POV, emulating the thinking traces structure.

**Note:** the generated final answer in the notebook may vary from the answer specified in the QA pair. This is caused by the fact that the agent model could use data preprocessing methods and steps different from the original Kaggle notebook and the synthetic question would not usually specify them. This discrepancy is normal and lays foundation for a new exciting research direction of how language models tend to treat data analysis and whether they do it differently from humans. For full transparency we keep both LLM-generated final answer and original answer from the real Kaggle notebook as a signal of model's performance. We encourage the community to try different dataset mixes to see how they can push performance even further.

### 7. Final curation
We truncated overly long outputs and filtered out trivial traces to prevent content length issues and keep only high-quality traces.  
We kept non-trivial, multi-turn traces aligned with DABStep-style tasks.  
The resulting [Jupyter Agent Dataset](https://huggingface.co/datasets/data-agents/jupyter-agent-dataset) became the foundation for SFT on Qwen3-4B models with 51k synthetic notebooks and almost 0.2B tokens.

With this dataset in hand, the natural next step is to see whether it actually helps our model become a stronger data science agent. Let’s move on to the training pipeline and evaluate the impact!

## 🏃‍♂️ Training Pipeline

With the curated dataset ready, we turned to the key question: **does this data actually help the model get better at solving data analysis tasks?**
To find out, we set up a simple fine-tuning pipeline and ran experiments to measure the impact of training on our synthetic notebooks.

Some training steps turned out to be particularly interesting and gave us useful insights:

- For trace generation, we used LLMs to generate QA pairs, which gave us a **verifiable environment**.  
- Finally, we fine-tuned **Qwen3-4B** with [TRL](https://huggingface.co/docs/trl).  
  - Used `assistant_loss_only=True` → small performance boost.
  - Added netfune noise for full-parameter multi-epoch training → avoids overfitting.  

**Challenges:**  
- Prompting models for tool calling is tricky: not all prompts deliver the same performance ([Qwen docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)).  
- We had to manually test each one to find what worked best.  
- There’s no standardization in response formats for tool calling, making it difficult to switch between models.  
- Native Qwen's generation prompt is not adapted to `assistant_loss_only=True` training mode in TRL which requires to have generation tokens by default. Thus, we adapt the original chat templates by wrapping the assistant response part in the generation tags.
- Training thinking models on short reasoning texts may disrupt model capabilities → full-parameter training works better comparing to PEFT in this case. 

Our complete training implementation, including hyperparameter configurations and template adaptations, is available in our [finetuning directory](https://github.com/huggingface/jupyter-agent/tree/main/finetuning) in our repo.

## 📊 Results

First, we generated our final dataset using [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) which contains high quality code and short reasoning-like traces. Afterwards, we started our training and we have experimented with various configurations like PEFT/adapters vs. full-parameter tuning, learning rate, number of epochs, adding noise and others. We found out, that full-parameter fine-tuning allows the model to learn and replicate the [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) behavior response quality better with shorter supporting commentary fitting more to the data analysis task without unnecessary long reasoning. 

We have done a small ablation study on the impact of no. training epochs:

| Model | No. of epochs | DABstep (Easy) |
|----------|---------|---------|
| Qwen-3-4B-Instruct-2507 (Base) | 0 | 38.67% |
| Qwen-3-4B-Instruct-2507 (Our Scaffolding) | 0 | 52.78% |
| Qwen-3-4B-Instruct-2507 | 2 | 63.89% |
| Qwen-3-4B-Instruct-2507 | 3 | 73.61% |
| Qwen-3-4B-Instruct-2507 | 5 | **75%** |
| Qwen-3-4B-Instruct-2507 | 7 | 70.83% |

We observe that it is beneficial to have a bit more epochs than usual for SFT with lower learning rate and higher neftune noise (7). Finally, we compare our trained models with implemented scaffolding to define the pure impact of our training dataset. In summary, we can see up to 36%/22% boost on DABStep easy score compared with base/scaffolded model:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/training_dabstep_easy.png" alt="DABstep Easy Score"/>

We can also see, that the hard score can increase too even though our dataset is focused on easier questions:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/training_dabstep_hard.png" alt="DABstep Hard Score"/>

From figures above one can notice a noticeable impact of both new scaffolding and tuning on our synthetic notebooks. This makes Qwen-4B (with our pipeline + scaffolding) a state-of-the-art small-model agent on DABStep.

In practice, the model can now solve a wide range of realistic Kaggle-style data analysis tasks with consistent execution.  
It’s not yet strong enough for the hardest queries, but we’ve shown that even small models can become powerful agents when paired with the right data and scaffolding.

## Try Jupyter Agent Yourself

These results demonstrate that even small models can become powerful data science agents with the right training approach. Ready to try it yourself? We've made everything openly available so you can experiment with our fine-tuned models and dataset.

We openly release best-performing checkpoints of tuned Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507 together with the training dataset, which you can try out and experiment with:

* [Jupyter Agent Dataset](https://huggingface.co/datasets/data-agents/jupyter-agent-dataset)
* [Jupyter-Agent-Qwen3-4B-Instruct](https://huggingface.co/data-agents/jupyter-agent-qwen3-4b-instruct)
* [Jupyter-Agent-Qwen3-4B-Thinking](https://huggingface.co/data-agents/jupyter-agent-qwen3-4b-thinking)

You can load Jupyter Agent Dataset in just a couple of lines using the following code:

```python
from datasets import load_dataset
# To load the train split of a specific subset, such as non-thinking, you can do
ds = load_dataset("data-agents/jupyter-agent-dataset", split="non-thinking")
# apply chat template
tokenizer.apply_chat_template(ds[0]["text"])
```

You can also use sourced Kaggle datasets directly with E2B code execution using the following code:

```python
import kagglehub
import e2b_code_interpreter as e2b
from datasets import load_dataset

# load the Jupyter Agent Dataset
ds = load_dataset("data-agents/jupyter-agent-dataset", split="thinking")
# get the kaggle dataset name
dataset_name = ds[0]["kaggle_dataset_name"]
# load the dataset locally from Kaggle Hub
path = kagglehub.dataset_download(dataset_name)
print(path) # this is the folder path where the dataset is downloaded
# initialize sandbox
sandbox_init = e2b.Sandbox(timeout=240)
# write used file to E2B sandbox
file_name = ds[0]["files_used"][0]
file_name = file_name.split('/')[-1] if '/' in file_name else file_name
with open(f"{path}/{file_name}", "rb") as file:
    sandbox_init.files.write(f"/home/user/input/{file_name}", file)
# execute code with E2B
execution = sandbox_init.run_code("<some code>")
```

You use tuned Jupyter Agent Qwen-based models following the Qwen documentation code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "data-agents/jupyter-agent-qwen3-4b-instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

For Thinking model you can decode both thinking response and content using the next code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "data-agents/jupyter-agent-qwen3-4b-thinking"

# ...use same processing code from above...
try:
    # index finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

We hope that our findings will help and inspire others to continue progress in developing new and more powerful notebook coding agents. 

## 🔮 Next Steps

- *Harder tasks:* Generate more challenging, multi-step questions that better reflect real-world analysis.  
- *Scaling up:* Train on larger volumes of curated traces to push beyond the current 3.4% performance on the hard split.  
- *Distillation:* Investigate knowledge distillation, which has shown strong results for improving small models.  
- *Reinforcement Learning (RL):* Build an RL environment, which has been shown to achieve state-of-the-art performance on agentic tasks. Since our QA setup already provides a verifiable environment, we could leverage it directly for RL training.

Maybe this will lead to… **Jupyter-Agent 3.** 😉  

We're excited to see what the community builds next. Dive into our [jupyter-agent dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset) on the 🤗 Hub and explore the codebase at [https://github.com/huggingface/jupyter-agent](https://github.com/huggingface/jupyter-agent) to start your own experiments on agents for jupyter notebooks.