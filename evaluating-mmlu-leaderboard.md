---
title: "What is going on with the Open LLM Leaderboard"
thumbnail: /blog/assets/142_safetensors_official/thumbnail.png
authors:
- user: clefourier
---

<!-- {blog_metadata} -->
<!-- {authors} -->

Recently an interesting controversy arose on Twitter following the release of a new Large Language Model (LLM), the **Falcon model 🦅**, and its addition to the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), a public leaderboard comparing open access AI models.

The discussion centered around MMLU, one of the four displayed evaluations, a benchmark for measuring Massive Multitask Language Understanding (more about it below).

The community was surprised to realize that MMLU evaluation numbers of the current top model on the leaderboard, the **LLaMA model 🦙**, were significantly lower than the numbers in the published LLaMa paper.

Community members questioned the numbers so we decided to dive in a rabbit hole to understand what was going on with this and how to fix it.

In our quest, we were joined by members from both the LLaMA and the Falcon teams so this blog post is actually written by 6 hands between HuggingFace, Falcon and LLaMa team members. Isn’t that super cool?

Along this journey with us you’ll learn a lot about the ways you can evaluate a model on a single evaluation and what to believe or not in the numbers you see online or in paper.

Ready? Then buckle up, we’re starting our trip.

# What's the Open LLM Leaderboard?

First, note that the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is actually just a wrapper running the open-source benchmarking library [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) created by the EleutherAI collective of AI hackers famous for creating The Pile or training the GPT-J, GPT-Neo-X 20B and Pythia models. A serious team in the AI space. The [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) wrapper runs evaluations using the Eleuther AI harness on the spare cycles of Hugging Face’s compute cluster and stores the results in a dataset while displaying the resulting numbers and rankings on an open leaderboard.

For the LLaMA model, the MMLU numbers obtained when running the [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) significantly differ from the MMLU numbers in the published LLaMa paper.

Why is that the case? Well it turns out that the LLaMA team adapted another code implementation available online: the original evaluation code proposed by the UC Berkeley team which developed the MMLU benchmark, that we will call here the [original code implementation](https://github.com/hendrycks/test).

When diving further, we found yet another interesting code implementation of MMLU out there: the one provided in Stanford’s very comprehensive evaluation benchmark [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/) (HELM).

Both EleutherAI’s LM Harness and Stanford’s HELM benchmarks are interesting because they gather many evaluations in a single codebase (including MMLU) and thus give a much wider view of a model’s performances. This is the reason the Open LLM Leaderboard is wrapping such “holistic” benchmarks instead of using separate code base for each individual evaluation.

To settle the case, we decided to run these all these possible implementations of the same MMLU evaluations and Open LLM Leaderboard on a set of models to compare them and we were surprised by the results:

![Leaderboard rankings](./assets/evaluating-mmlu-leaderboard/leaderboard-ranking.png)

These different implementations of the same benchmark give widely different numbers and, more worryingly, even change the ranking order of the models on the leaderboard.

Let’s try to understand where this discrepancy comes from. But first let’s briefly see how we automatically evaluate behaviors in modern LLM.

# How we automatically evaluate a model in today’s LLM world

MMLU is a multiple choice question test, so a rather simple benchmark (versus open-ended questions) but as we’ll see, this still leaves a lot of room for implementation details and differences. The benchmark consists of questions with 4 possible answers covering 57 general knowledge domains grouped in coarse grained categories: “Humanities”, “Social Sciences”, “STEM” and a catch-all “Other”

For each question, only one of the provided answers is the correct one. Here is an example that you can [explore here](https://huggingface.co/datasets/cais/mmlu/viewer/college_medicine/dev?row=0)

```
Question: Glucose is transported into the muscle cell:


Choices:
- A via protein transporters called GLUT4.
- B only in the presence of insulin.
- C via hexokinase.
- D via monocarbylic acid transporters.


Correct answer: A
```

Note: you can very easily explore more of this dataset [in the dataset viewer here](https://huggingface.co/datasets/cais/mmlu/viewer)

Large language models are quite simple models in the AI model zoo. They take a string of text as input (called a “prompt”), which is cut into tokens (words, sub-words or characters, depending on how the model is built). From this input, they generate a distribution of probability for the next token, over all the tokens they know (so called the “vocabulary” of the model): you can therefore get how `probable’ any token is as a continuation of the input prompt.

From these probabilities we can choose a token, for instance the most probable (or we can introduce some slight noise with a sampling to avoid having “too mechanical” answers). Adding the selected token to the prompt and feeding it back to the model allows to create whole sentences as continuation of the input prompt. This is how ChatGPT or Hugging Chat generate answers for instance.

![Probabilities one token](./assets/evaluating-mmlu-leaderboard/llm-01.png)

In summary, we have two main ways to get information out of a model to evaluate it:
1. get the **probabilities** that some specific tokens groups are continuations of the prompt – and **compare these probabilities together** for our predefined possible choices;
2. get a **text generation** from the model (by repeatedly selecting tokens as we’ve seen) – and **compare these text generations** to the texts of various predefined possible choices.

Both these techniques have pros and cons. Text generations (option 2) are pretty much always possible to get even for commercial API models, which is not always the case for probabilities (option 1) - however, models sometimes generate answers that, though technically correct, are counted as false because they are not exactly equal to the references (for instance the model use the word “always” instead of the expected “yes” which will be counted as wrong answer). Using probabilities (option 1), on the other hand, is sometimes a more favorable evaluation as it constrains the model outputs to be in a given set (we only compare the possible answers and ignore the rest). Evaluation numbers can be higher as we’ll see. Also, sometimes the probabilities are very, very close to one another and this may then artificially create a signal out of noise.

Armed with this knowledge, let dive into our three implementations of MMLU in more detail: what input is sent to the model, what is expected as outputs and how these outputs are compared.

# Fifty shades of running the same evaluation

Looking at the prompts
Let’s compare an example of prompt each benchmark sends to the models:

Original implementation
HELM
AI Harness (as of Jan 2023)
The following are multiple choice questions (with answers) about  us foreign policy.
How did the 2008 financial crisis affect America's international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:
The following are multiple choice questions (with answers) about us foreign policy.

Question: How did the 2008 financial crisis affect America's international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:


Question: How did the 2008 financial crisis affect America's international reputation?
Choices:
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:

<div>
<table><p>
  <tbody>
 <tr style="text-align: left;">
  <td>Original implementation</td>
  <td>HELM</td>
  <td>AI Harness (as of Jan 2023)</td>
 </tr>
  <tr style=" vertical-align: top;">
    <td>The following are multiple choice questions (with answers) about  us foreign policy.
How did the 2008 financial crisis affect America's international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:
</td>
    <td>The following are multiple choice questions (with answers) about us foreign policy.

Question: How did the 2008 financial crisis affect America's international reputation?
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:
</td>
    <td>Question: How did the 2008 financial crisis affect America's international reputation?
Choices:
A. It damaged support for the US model of political economy and capitalism
B. It created anger at the United States for exaggerating the crisis
C. It increased support for American global leadership under President Obama
D. It reduced global use of the US dollar
Answer:
</td>
  </tr>
  </tbody>
</table><p>
</div>

The differences between them can seem minute, did you spot them all? Here they are:
- First sentence, instruction, and topic: Few differences. HELM add an extra space and the Eleuther LM Harness does not include the topic line
- Question: HELM and the LM Harness add a “Question:” prefix
- Choices: Eleuther LM Harness prepends them with the keyword “Choice”

# Now how do we evaluate the model from these prompts?

Let’s start with how the original MMLU implementation extracts the predictions of the model. In the original implementation we compare the probabilities predicted by the model, on the four answers only:

![Probabilities four answers only](./assets/evaluating-mmlu-leaderboard/llm-02.png)

This can be beneficial for the model in some case, for instance, as you can see here:


|                                           | MMLU (HELM) | MMLU (Harness) | MMLU (Original) |
|:------------------------------------------|------------:|---------------:|----------------:|
| huggingface/llama-65b                     |       **0.637** |          0.488 |           **0.636** |
| tiiuae/falcon-40b                         |       0.571 |          **0.527** |           0.558 |
| huggingface/llama-30b                     |       0.583 |          0.457 |           0.584 |
| EleutherAI/gpt-neox-20b                   |       0.256 |          0.333 |           0.262 |
| huggingface/llama-13b                     |       0.471 |          0.377 |           0.47  |
| huggingface/llama-7b                      |       0.339 |          0.342 |           0.351 |
| tiiuae/falcon-7b                          |       0.278 |          0.35  |           0.254 |
| togethercomputer/RedPajama-INCITE-7B-Base |       0.275 |          0.34  |           0.269 |

