---
title: "What's going on with the Open LLM Leaderboard?"
thumbnail: /blog/assets/evaluating-mmlu-leaderboard/thumbnail.png
authors:
- user: clefourrier
- user: SaylorTwift
- user: slippylolo
- user: thomwolf
---

# What's going on with the Open LLM Leaderboard?


Recently an interesting discussion arose on Twitter following the release of [**Falcon 🦅**](https://huggingface.co/tiiuae/falcon-40b) and its addition to the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), a public leaderboard comparing open access large language models.

The discussion centered around one of the four evaluations displayed on the leaderboard: a benchmark for measuring [Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (shortname: MMLU).

The community was surprised that MMLU evaluation numbers of the current top model on the leaderboard, the [**LLaMA model 🦙**](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), were significantly lower than the numbers in the [published LLaMa paper](https://arxiv.org/abs/2302.13971).

So we decided to dive in a rabbit hole to understand what was going on and how to fix it 🕳🐇

In our quest, we discussed with both the great [@javier-m](https://huggingface.co/javier-m) who collaborated on the evaluations of LLaMA and the amazing [@slippylolo](https://huggingface.co/slippylolo) from the Falcon team. This being said, all the errors in the below should be attributed to us rather than them of course!

Along this journey with us you’ll learn a lot about the ways you can evaluate a model on a single evaluation and whether or not to believe the numbers you see online and in papers.

Ready? Then buckle up, we’re taking off 🚀.

## What's the Open LLM Leaderboard?

First, note that the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is actually just a wrapper running the open-source benchmarking library [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) created by the [EleutherAI non-profit AI research lab](https://www.eleuther.ai/) famous for creating [The Pile](https://pile.eleuther.ai/) and training [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b), [GPT-Neo-X 20B](https://huggingface.co/EleutherAI/gpt-neox-20b), and [Pythia](https://github.com/EleutherAI/pythia). A team with serious credentials in the AI space!

This wrapper runs evaluations using the Eleuther AI harness on the spare cycles of Hugging Face’s compute cluster, and stores the results in a dataset on the hub that are then displayed on the [leaderboard online space](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

For the LLaMA models, the MMLU numbers obtained with the [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) significantly differ from the MMLU numbers reported in the LLaMa paper.

Why is that the case?

## 1001 flavors of MMLU

Well it turns out that the LLaMA team adapted another code implementation available online: the evaluation code proposed by the original UC Berkeley team which developed the MMLU benchmark available at https://github.com/hendrycks/test and that we will call here the **"Original implementation"**.

When diving further, we found yet another interesting implementation for evaluating on the very same MMLU dataset: the evalution code provided in Stanford’s [CRFM](https://crfm.stanford.edu/) very comprehensive evaluation benchmark [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/) that we will call here the **HELM implementation**.

Both the EleutherAI Harness and Stanford HELM benchmarks are interesting because they gather many evaluations in a single codebase (including MMLU), and thus give a wide view of a model’s performance. This is the reason the Open LLM Leaderboard is wrapping such “holistic” benchmarks instead of using individual code bases for each evaluation.

To settle the case, we decided to run these three possible implementations of the same MMLU evaluation on a set of models to rank them according to these results:
- the Harness implementation ([commit e47e01b](https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4))
- the HELM implementation ([commit cab5d89](https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec))
- the Original implementation (with Hugging Face integration by the amazing [@olmer](https://huggingface.co/olmer) at https://github.com/hendrycks/test/pull/13)

(Note that the Harness implementation has been recently updated - more in this at the end of our post)

The results are surprising:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-01-ter-01.png)

You can find the full evaluation numbers at the end of the post.

These different implementations of the same benchmark give widely different numbers and even change the ranking order of the models on the leaderboard!

Let’s try to understand where this discrepancy comes from 🕵️But first, let’s briefly understand how we can automatically evaluate behaviors in modern LLMs.

## How we automatically evaluate a model in today’s LLM world

MMLU is a multiple choice question test, so a rather simple benchmark (versus open-ended questions) but as we’ll see, this still leaves a lot of room for implementation details and differences. The benchmark consists of questions with four possible answers covering 57 general knowledge domains grouped in coarse grained categories: “Humanities”, “Social Sciences”, “STEM”, etc

For each question, only one of the provided answers is the correct one. Here is an example:

```
Question: Glucose is transported into the muscle cell:


Choices:
A. via protein transporters called GLUT4.
B. only in the presence of insulin.
C. via hexokinase.
D. via monocarbylic acid transporters.


Correct answer: A
```

Note: you can very easily explore more of this dataset [in the dataset viewer](https://huggingface.co/datasets/cais/mmlu/viewer/college_medicine/dev?row=0) on the hub.

Large language models are simple models in the AI model zoo. They take a *string of text* as input (called a “prompt”), which is cut into tokens (words, sub-words or characters, depending on how the model is built) and fed in the model. From this input, they generate a distribution of probability for the next token, over all the tokens they know (so called the “vocabulary” of the model): you can therefore get how `probable’ any token is as a continuation of the input prompt.

We can use these probabilities to choose a token, for instance the most probable (or we can introduce some slight noise with a sampling to avoid having “too mechanical” answers). Adding our selected token to the prompt and feeding it back to the model allows to generate another token and so on until whole sentences are created as continuations of the input prompt:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-01.png)

This is how ChatGPT or Hugging Chat generate answers.

In summary, we have two main ways to get information out of a model to evaluate it:
1. get the **probabilities** that some specific tokens groups are continuations of the prompt – and **compare these probabilities together** for our predefined possible choices;
2. get a **text generation** from the model (by repeatedly selecting tokens as we’ve seen) – and **compare these text generations** to the texts of various predefined possible choices.

Armed with this knowledge, let's dive into our three implementations of MMLU, to find out what input is sent to models, what is expected as outputs, and how these outputs are compared.

## MMLU comes in all shapes and sizes: Looking at the prompts

Let’s compare an example of prompt each benchmark sends to the models by each implementation for the same MMLU dataset example:

<div>
<table><p>
  <tbody>
 <tr style="text-align: left;">
  <td>Original implementation <a href="https://github.com/hendrycks/test/pull/13">Ollmer PR</a></td>
  <td>HELM <a href="https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec">commit cab5d89</a> </td>
  <td>AI Harness <a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4">commit e47e01b</a></td>
 </tr>
  <tr style=" vertical-align: top;">
    <td>The following are multiple choice questions (with answers) about  us foreign policy. <br>
How did the 2008 financial crisis affect America's international reputation? <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
    <td>The following are multiple choice questions (with answers) about us foreign policy. <br>
 <br>
Question: How did the 2008 financial crisis affect America's international reputation? <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
    <td>Question: How did the 2008 financial crisis affect America's international reputation? <br>
Choices: <br>
A. It damaged support for the US model of political economy and capitalism <br>
B. It created anger at the United States for exaggerating the crisis <br>
C. It increased support for American global leadership under President Obama <br>
D. It reduced global use of the US dollar <br>
Answer:
</td>
  </tr>
  </tbody>
</table><p>
</div>

The differences between them can seem small, did you spot them all? Here they are:
- First sentence, instruction, and topic: Few differences. HELM adds an extra space, and the Eleuther LM Harness does not include the topic line
- Question: HELM and the LM Harness add a “Question:” prefix
- Choices: Eleuther LM Harness prepends them with the keyword “Choices”

## Now how do we evaluate the model from these prompts?

Let’s start with how the [original MMLU implementation](https://github.com/hendrycks/test/pull/13) extracts the predictions of the model. In the original implementation we compare the probabilities predicted by the model, on the four answers only:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-02.png)

This can be beneficial for the model in some case, for instance, as you can see here:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-03.png)

In this case, the model got a +1 score for ranking the correct answer highest among the 4 options. But if we take a look at the full vocabulary it would have rather generated a word outside of our four options: the word “Zygote” (this is more of an example than a real use case 🙂)

How can we make sure that the model does as few as possible of these types of errors?

We can use a “**few shots**” approach in which we provide the model with one or several examples in the prompt, with their expected answers as well. Here is how it looks:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-04.png)

Here, the model has one example of the expected behavior and is thus less likely to predict answers outside of the expected range of answers.

Since this improves performance, MMLU is typically evaluated in 5 shots (prepending 5 examples to each prompt) in all our evaluations: the original implementation, EleutherAI LM Harness and HELM. (Note: Across benchmarks, though the same 5 examples are used, their order of introduction to the model can vary, which is also a possible source of difference, that we will not investigate here. You also obviously have to pay attention to avoid leaking some answers in the few shot examples you use…)

**HELM:** Let’s now turn to the [HELM implementation](https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec). While the few-shot prompt is generally similar, the way the model is evaluated is quite different from the original implementation we’ve just seen: we use the next token output probabilities from the model to select a text generation and we compare it to the text of the expected answer as displayed here:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-05.png)

In this case, if our "Zygote" token was instead the highest probability one (as we’ve seen above), the model answer ("Zygote") would be wrong and the model would not score any points for this question:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-06.png)

**Harness:** Now we finally turn to the - [EleutherAI Harness implementation as of January 2023](https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4) which was used to compute the first numbers for the leaderboard. As we will see, we’ve got here yet another way to compute a score for the model on the very same evaluation dataset (note that this implementation has been recently updated - more on this at the end).

In this case, we are using the probabilities again but this time the probabilities of the full answer sequence, with the letter followed by the text of the answer, for instance “C. The second pharyngeal arch”. To compute the probability for a full answer we get the probability for each token (like we saw above) and gather them. For numerical stability we gather them by summing the logarithm of the probabilities and we can decide (or not) to compute a normalization in which we divide the sum by the number of tokens to avoid giving too much advantage to longer answers (more on this later). Here is how it looks like:

![png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/evaluating-mmlu-leaderboard/LLM-07.png)

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
    <td>  A <br>
 B <br>
 C <br>
 D
</td>
    <td>A
</td>
    <td> A. It damaged support for the US model of political economy and capitalism <br>
 B. It created anger at the United States for exaggerating the crisis <br>
 C. It increased support for American global leadership under President Obama <br>
 D. It reduced global use of the US dollar
</td>
  </tr>
  </tbody>
</table><p>
</div>

We’ve covered them all!

Now let’s compare the model scores on these three possible ways to evaluate the models:


|                                           | MMLU (HELM) | MMLU (Harness) | MMLU (Original) |
|:------------------------------------------|------------:|---------------:|----------------:|
| llama-65b                     |       **0.637** |          0.488 |           **0.636** |
| tiiuae/falcon-40b                         |       0.571 |          **0.527** |           0.558 |
| llama-30b                     |       0.583 |          0.457 |           0.584 |
| EleutherAI/gpt-neox-20b                   |       0.256 |          0.333 |           0.262 |
| llama-13b                     |       0.471 |          0.377 |           0.47  |
| llama-7b                      |       0.339 |          0.342 |           0.351 |
| tiiuae/falcon-7b                          |       0.278 |          0.35  |           0.254 |
| togethercomputer/RedPajama-INCITE-7B-Base |       0.275 |          0.34  |           0.269 |

We can see that for the same dataset, both absolute scores and model rankings (see the first figure) are very sensitive to the evaluation method we decide to use.

Let's say you've trained yourself a perfect reproduction of the LLaMA 65B model and evaluated it with the harness (score 0.488, see above). You're now comparing it to the published number (evaluated on the original MMLU implementation so with a score 0.637). With such a 30% difference in score you're probably thinking: "Oh gosh, I have completly messed up my training 😱". But nothing could be further from the truth, these are just numbers which are not at all comparable even if they're both labelled as "MMLU score" (and evaluated on the very same MMLU dataset).

Now, is there a "best way" to evaluate a model among all the ones we've seen? It's a tricky question. Different models may fare differently when evaluated one way or another as we see above when the rankings change. To keep this as fair as possible, one may be tempted to select an implementation where the average score for all tested models is the highest so that we "unlock" as many capabilities as possible from the models. In our case, that would mean using the log-likelihood option of the original implementation. But as we saw above, using the log-likelihood is also giving some indications to the model in some way by restricting the scope of possible answers, and thus is helping the less powerful models maybe too much. Also log-likelihood is easy to access for open-source models but is not always exposed for closed source API models.

And you, reader, what do you think? This blog post is already long so it's time to open the discussion and invite your comments. Please come discuss this topic in the following discussion thread of the Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/82

## Conclusion

A key takeaway lesson from our journey is that evaluations are strongly tied to their implementations–down to minute details such as prompts and tokenization. The mere indication of "MMLU results" gives you little to no information about how you can compare these numbers to others you evaluated on another library.

This is why open, standardized, and reproducible benchmarks such as the [EleutherAI Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness/) or [Stanford HELM](https://github.com/stanford-crfm/helm/) are invaluable to the community. Without them, comparing results across models and papers would be impossible, stifling research on improving LLMs.
  
**Post scriptum**: In the case of the Open LLM Leaderboard we’ve decided to stick to using community maintained evaluation libraries. Thankfully during the writing of this blog post, the amazing community around the EleutherAI Harness, and in particular [ollmer](https://github.com/EleutherAI/lm-evaluation-harness/issues/475)
have done an amazing work updating the evaluation of MMLU in the harness to make it similar to the original implementation and match these numbers.

We are currently updating the full leaderboard with the updated version of the [EleutherAI Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness/), so expect to see scores coming from the Eleuther Harness v2 coming up in the next few weeks! (Running all the models again will take some time, stay tuned :hugs:)

## Acknowledgements:
We are very grateful to Xavier Martinet, Aurélien Rodriguez and Sharan Narang from the LLaMA team for helpful suggestions in this blog post as well as having answered all our questions. 

## Reproducibility hashes:
Here are the commit hashes of the various code implementations used in this blog post.

- EleutherAI LM harness implementation commit e47e01b: https://github.com/EleutherAI/lm-evaluation-harness/tree/e47e01beea79cfe87421e2dac49e64d499c240b4
- HELM implementation commit cab5d89: https://github.com/stanford-crfm/helm/tree/cab5d89fadbff86190f29ddfa497301958eaf2ec
- Original MMLU implementation (with Hugging Face integration by the amazing [@olmer](https://huggingface.co/olmer)): https://github.com/hendrycks/test/pull/13

