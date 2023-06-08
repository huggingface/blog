---
title: "Can foundation models label data like humans?"
thumbnail: /blog/assets/llm-leaderboard/leaderboard-thumbnail.png
authors:
- user: nazneen
- user: natolambert
- user: sheonhan
- user: edbeeching
- user: lewtun

---
# Can foundation models label data like humans?

<!-- {blog_metadata} -->
<!-- {authors} -->

In 2023, it’s a common story that a new large language model (LLM) is released to the tune of “our model is preferred to ChatGPT N% of the time,” and what is omitted from that sentence is that the model is preferred in some type of GPT4 prompting scheme. What these points are trying to show is a proxy for a different measurement: scores provided by human labelers. The process of training models with reinforcement learning from human feedback (RLHF) has proliferated interfaces for and data of comparing two model completions to each other. This data is used in the RLHF process to train a reward model that predicts a preferred text, but the idea of rating and ranking model outputs has grown to be a more general tool in evaluation.

In terms of iteration speed, using a language model to evaluate model outputs is highly efficient, but there’s a sizeable missing piece: **investigating if the downstream tool-shortcut is calibrated with the original form of measurement.** In this blog post, we’ll zoom in on where you can and cannot trust the data labels you get from the LLM of your choice by expanding the HuggingFace H4 evaluation suite.

Since the advent of ChatGPT, we have seen unprecedented growth in the development of Large Language Models (LLMs), and particularly chatty models that are fine-tuned to follow instructions given in the form of prompts. However, how these models compare is unclear due to the lack of benchmarks designed to test their performance rigorously. Evaluating instruction and chatty models is intrinsically difficult because a large part of user preference is centered around qualitative style while in the past NLP evaluation was far more defined. 

Leaderboards have begun to emerge, such as the [LMSYS](https://leaderboard.lmsys.org/), [nomic / GPT4All](https://gpt4all.io/index.html), to compare some aspects of these models, but there needs to be a complete source comparing model capabilities. Some use existing NLP benchmarks that can show question and answering capabilities and some are crowdsourced rankings from open-ended chatting. In order to present a more general picture of evaluations the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) has been expanded, including automated academic benchmarks, professional human labels, and GPT4 evals. 

## Preference modeling and evaluation

Any point in a training process where humans are needed to curate the data is inherently expensive. To date, there are only a few human labeled preference datasets available ************for training************ these models, such as [Anthropic’s HHH data](https://huggingface.co/datasets/Anthropic/hh-rlhf) or OpenAI’s [Learning to Summarize](https://huggingface.co/datasets/openai/summarize_from_feedback) / [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) datasets. The same preference labels can be generated on **************************************************************************************************************************model outputs to create a relative Elo ranking between models**************************************************************************************************************************. When the source of text given to labelers is generated from a model of interest, the data becomes doubly interesting. 

We are doing that for training our H4 models, and we started seeing interesting things so we wanted to do a more controlled study of existing open-source models and how that preference collection process would translate and compare to the currently popular GPT4/ChatGPT evaluations of preferences.

To do this, we curated a secret set of instruction prompts and completions from a popular set of open-source models: Koala13b, Vicuna13b, OAsst 12b, and Dolly12b.  [************************************************TODO ADD LINKS TO MODELS]************************************************

![Untitled](Can%20foundation%20models%20label%20data%20like%20humans%20c1330e52ed1d4598a4fe5fdd56745b47/Untitled%201.png)

Items to add:

- Rankings vs. ratings
- Automated evals and prompting (especially when withheld)
- Generations and dialogue formats (example of a complete version is [FastChat’s `conversation.py`](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py))

Automated: Uses Eleuther harness

We collected a set of high-quality, human-written prompts for diverse task categories, including generation, brainstorming, question answering, summarization, commonsense, and coding-related. The dataset has 327 prompts across these categories, and 25 are coding-related.

Here are the stats on the prompt and demonstration length. 

|  | prompt | completion |
| --- | --- | --- |
| count | 327 | 327 |
| length (mean ± std. dev.) in tokens | 24 ± 38 | 69 ± 79 |
| min. length | 3 | 1 |
| 25% length | 10 | 18 |
| 50% length | 15 | 42 |
| 75% length | 23 | 83 |
| max  | 381 | 546 |

**Human:**  We partnered with Scale AI to collect high-quality human annotations for a handful of open-source instruction-tuned models on our blind test set. We requested annotators to rate responses for helpfulness and truthfulness in a pairwise setting. We generated nC2 combinations for each prompt, where n is the number of models we evaluate. Here is an example snapshot of the instructions and the interface Scale provided for our evaluations.

<p float="left">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-leaderboard/interface.png" width="100" />
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-leaderboard/scale-human-eval-0601.png" width="100" /> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/llm-leaderboard/tie_counts.png" width="100" />
</p>


GPT4:

A prompt adapted from the [FastChat evaluation prompts](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/prompt.jsonl), encouraging shorter length for faster and cheaper generations (as the explanations are disregarded most of the time):

```
### Question
{question}

### The Start of Assistant 1's Answer
{answer_1}
### The End of Assistant 1's Answer

### The Start of Assistant 2's Answer
{answer_2}
### The End of Assistant 2's Answer

### System
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please compare the helpfulness, relevance, accuracy, level of details of their responses.
The rating should be from the set of 1, 2, 3, 4, 5, 6, 7, or 8, where higher numbers indicated that Assistant 2 was better than Assistant 1.
Please first output a single line containing only one value indicating the preference between Assistant 1 and 2.
In the subsequent line, please provide a brief explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
```

## Examples

Human Eval

Discussion points:

- Riley Goodside point on per-token information from a LLM, so outputing the score first could be limiting
- Positional bias across multiple models? Potential causes?
- Rating vs. ranking with models (chatGPT does ratings well, but rankings not well).

| Category | Correlation: GPT4 to Human Labels |
| --- | --- |
| Brainstorm | 0.60 |
| Generation | 0.55 |
| Commonsense | 0.46 |
| Question answering | 0.44 |
| Summarization | 0.40 |
| Natural language to code | 0.33 |

### Citation

```
@article{rajani2023llm_labels,
  author = {Rajani, Nazneen, and Lambert, Nathan and Han, Sheon and Wang, Jean and Nitski, Osvald and Beeching, Edward and Tunstall, Lewis},
  title = {Can foundation models label data like humans?},
  journal = {Hugging Face Blog},
  year = {2023},
  note = {https://huggingface.co/blog/llm-v-human-data},
}
```
