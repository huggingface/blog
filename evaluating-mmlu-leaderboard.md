---
title: "What is going on with the Open LLM Leaderboard"
thumbnail: /blog/assets/142_safetensors_official/thumbnail.png
authors:
- user: clefourier
---

<!-- {blog_metadata} -->
<!-- {authors} -->

Recently an interesting discussion arose on Twitter following the release of **Falcon 🦅** and its addition to the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), a public leaderboard comparing open access large language models.

The discussion centered around [MMLU](https://arxiv.org/abs/2009.03300), one of the four displayed evaluations, a benchmark for measuring Massive Multitask Language Understanding.

The community was surprised that MMLU evaluation numbers of the current top model on the leaderboard, the **LLaMA model 🦙**, were significantly lower than the numbers in the published LLaMa paper.

Community members questioned the numbers, so we decided to dive in a rabbit hole to understand what was going on and how to fix it.

In our quest, we were joined by members from both the LLaMA and the Falcon teams so this blog post is actually written by 6 hands between HuggingFace, Falcon and LLaMa team members. Isn’t that super cool?

Along this journey with us you’ll learn a lot about the ways you can evaluate a model on a single evaluation and whether or not to believe the numbers you see online (or in papers) .

Ready? Then buckle up, we’re starting our trip 🚀.

# What's the Open LLM Leaderboard?

First, note that the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is actually just a wrapper running the open-source benchmarking library [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) created by the EleutherAI collective of AI hackers famous for creating The Pile and training GPT-J, GPT-Neo-X 20B, and Pythia serious team in the AI space! The [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) wrapper runs evaluations using the Eleuther AI harness on the spare cycles of Hugging Face’s compute cluster, and stores the results in a dataset while displaying the resulting numbers and rankings in the Spaces powering the leaderboard.

For the LLaMA models, the MMLU numbers obtained with the [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) significantly differ from the MMLU numbers reported in the LLaMa paper.

Why is that the case? Well it turns out that the LLaMA team adapted another code implementation available online: the original evaluation code proposed by the UC Berkeley team which developed the MMLU benchmark, that we will call here the [original code implementation](https://github.com/hendrycks/test).

When diving further, we found yet another interesting implementation of MMLU: the one provided in Stanford’s very comprehensive evaluation benchmark [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/) (HELM).

Both EleutherAI’s LM Harness and Stanford’s HELM benchmarks are interesting because they gather many evaluations in a single codebase (including MMLU), and thus give a much wider view of a model’s performances. This is the reason the Open LLM Leaderboard is wrapping such “holistic” benchmarks instead of using individual code bases for each evaluation.

To settle the case, we decided to run these three possible implementations of the same MMLU evaluation on a set of models to compare the results. We were quite surprised by the results:

![Leaderboard rankings](./assets/evaluating-mmlu-leaderboard/leaderboard-ranking.png)

These different implementations of the same benchmark give widely different numbers and, even change the ranking order of the models on the leaderboard!

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

Both these techniques have pros and cons. Text generations (option 2) are pretty much always possible to get even for commercial API models, which is not always the case for probabilities (option 1) - however, models sometimes generate answers that, though technically correct, are counted as false because they are not exactly equal to the references (for instance the model use the word “always” instead of the expected “yes” which will be counted as wrong answer). Using probabilities (option 1), on the other hand, is sometimes a more favorable evaluation as it constrains the model outputs to be in a given set (we only compare the possible answers and ignore the rest). As we’ll see, this can boost evaluation numbers. Also, sometimes the probabilities are very, very close to one another and this may then artificially create a signal out of noise.

Armed with this knowledge, let's dive into our three implementations of MMLU, to find out what input is sent to models, what is expected as outputs, and how these outputs are compared.

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

![Probabilities four answers only wrong proba](./assets/evaluating-mmlu-leaderboard/llm-03.png)

In this case, the model got a +1 score for ranking the correct answer highest among the 4 options. But if we take a look at the full vocabulary it would have rather generated a word outside of our four options: the word “Zygote” (this is more of an example than a real use case 🙂)

How can we make sure that the model does as few as possible of these types of errors?

We can use a “**few shots**” approach in which we provide the model with one or several examples in the prompt, with their expected answers as well. Here is how it looks:

![Probabilities four answers only wrong proba few shot](./assets/evaluating-mmlu-leaderboard/llm-04.png)

Here, the model has one example of the expected behavior and is thus less likely to predict answers outside of the expected range of answers.

Since this improves performance, MMLU is typically evaluated in 5 shots (prepending 5 examples to each prompt) in all our evaluations: the original implementation, EleutherAI LM Harness and HELM. (Note: Across benchmarks, though the same 5 examples are used, their order of introduction to the model can vary, which is also a possible source of difference, that we will not investigate here. You also obviously have to pay attention to avoid leaking some answers in the few shot examples you use…)

Let’s now turn to the HELM implementation. While the few-shot prompt is generally similar, the way the model is evaluated is quite different from the original implementation we’ve just seen: we use the next token output probabilities from the model to select a text generation and we compare it to the text of the expected answer as displayed here:

![helm-generation](./assets/evaluating-mmlu-leaderboard/llm-05.png)

In this case, if our Zygote token was instead the highest probability one (as we’ve seen above), the model answer (“Zygote”) would be wrong and the model would not score any point for this question:

![helm-wrong-generation](./assets/evaluating-mmlu-leaderboard/llm-06.png)

Now we finally turn to the - EleutherAI Harness implementation as of January 2023 (updated in June 2023) which was used to compute the first numbers for the leaderboard. As we will see, we’ve got here yet another way to compute a score for the model on the very same evaluation dataset.

In the case of the EleutherAI LM Harness, we are using the probabilities again but this time the probabilities of the full answer sequence, with the letter followed by the text of the answer, for instance “C. The second pharyngeal arch”. To compute the probability for a full answer we get the probability for each token (like we saw above) and gather them. For numerical stability we gather them by summing the logarithm of the probabilities and we can decide (or not) to compute a normalization in which we divide the sum by the number of tokens to avoid advantaging too much longer answers (more on this later). Here is how it looks like:

![harness-generations](./assets/evaluating-mmlu-leaderboard/llm-07.png)

Here is a table summary of the answers provided and generated by the model to summarize what we’ve seen up to now:

<div>
<table><p>
  <tbody>
 <tr style="text-align: left;">
  <td>Original implementation</td>
  <td>HELM</td>
  <td>AI Harness (as of Jan 2023)</td>
 </tr>
  <tr style=" vertical-align: top;">
    <td> We compare the probabilities of the following letter answers:
</td>
    <td>The model is expected to generate as text the following letter answer:
</td>
    <td>We compare the probabilities of the following full answers:
</td>
  </tr>
  <tr style=" vertical-align: top;">
    <td>  A
 B
 C
 D
</td>
    <td>A
</td>
    <td> A. It damaged support for the US model of political economy and capitalism
 B. It created anger at the United States for exaggerating the crisis
 C. It increased support for American global leadership under President Obama
 D. It reduced global use of the US dollar
</td>
  </tr>
  </tbody>
</table><p>
</div>

We’ve seen all the benchmarks! Now let’s compare the model scores on these three possible  ways to evaluate the models:


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

So have we found the ultimate method yet here?

By taking a look at this table, you might be thinking “Hmm for all multiple choice questions, the log likelihood seems to give the highest overall scores so maybe it's the best way to uncover model's skills”

Well, if you’re still reading and are ready to be even more puzzled, let’s take a very brief final look at another evaluation on the leaderboard: the [AllenAI Reasoning Challenge, so called ARC](https://allenai.org/data/arc).

# The ARC challenge

The [AI2 Reasoning Challenge](https://allenai.org/data/arc) (short ARC) is another one of the four evaluations selected on the Open LLM Leaderboard. Like MMLU,it’s also a multiple choice benchmark.

While MMLU attempts to capture knowledge across many categories, ARC is focused on reasoning questions from science exams as you can see on this example from the dataset (that you can find at https://huggingface.co/datasets/ai2_arc/viewer/ARC-Challenge/validation?row=6)

```
Question: How are the particles in a block of iron affected when the block is melted?

Choices:
A. The particles gain mass.
B. The particles contain less energy.
C. The particles move more rapidly.
D. The particles increase in volume.

Gold answer: C
```

Note: you can very easily explore more of this dataset in the dataset viewer at https://huggingface.co/datasets/ai2_arc/viewer/ARC-Challenge/validation

At first sight this dataset looks quite similar to our MMLU dataset, right? So we would expect our MMLU learnings to transfer to ARC. Let’s explore all the ways we’ve seen these multiple choice evaluation being implemented and test them on this dataset:

