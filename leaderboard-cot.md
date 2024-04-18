---
title: "Introducing the Open CoT Leaderboard"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_decodingtrust.png
authors:
- user: ggbetz
  guest: true
  org: logikon
- user: scacean
  guest: true
  org: logikon
- user: yakazimir
  guest: true
  org: allenai
- user: clefourrier
---

# Introducing the Open Chain of Thought Leaderboard
[Chain-of-thought prompting](https://huggingface.co/docs/transformers/main/en/tasks/prompting#chain-of-thought) is emerging as a powerful and effective design pattern for LLM-based apps and agents. The basic idea of chain-of-thought prompting is to let a model generate a step-by-step solution (“reasoning trace”) before answering a question or taking a decision. With the Open CoT Leaderboard we’re tracking LLMs’ ability to generate effective chain-of-thought traces for challenging reasoning tasks. 

Unlike most performance based leaderboards, we’re not scoring the absolute accuracy a model achieves on a given task, but the difference between the accuracy with and without chain-of-thought prompting:

```
accuracy gain Δ = accuracy with CoT – accuracy w/o CoT.
```

This allows us to truly inspect the impact that chain-of-thought has on model accuracy.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script>
<gradio-app theme_mode="light" space="logikon/open_cot_leaderboard"></gradio-app>

## What’s the motivation behind such a leaderboard for chain-of-thought?

Chain-of-thought prompting is a universally applicable prompting strategy that may improve explainability and accuracy of LLM-based apps and agents (see, e.g., [this collection](https://github.com/logikon-ai/awesome-deliberative-prompting#readme) for recent research and implementations)). With frameworks like Langchain or LMQL, it’s straightforward to insert sophisticated reasoning chains in your apps. But even if you’ve never heard about chain-of-thought before, you may have noticed, while using a ChatBot, that it tends to proceeds step by step before answering your query. So, a systematic, up-to-date comparison of LLMs’ ability to generate effective chain-of-thought traces may inform the decisions of builders and users when choosing a model. 

Over time, static accuracy-based benchmarks risk to become less informative: Does a model score well because of its superior skill, because it has seen the correct answers during training, or because it has been developed in a competitive context that is governed by this very benchmark? These widely acknowledged issues are addressed by recent eval approaches such as ChatBot arenas, the use of LLMs as judges, or dynamic benchmarks with programmatically generated tasks. The Open CoT Leaderboard contributes to these efforts. It seems more robust to training data contamination because knowing the answer to a question doesn’t ensure that one can reason effectively about it. Plus, the leaderboard’s metric is inversely related to the default metric of other accuracy-based leaderboards (higher accuracy w/o CoT means, all else being equal, lower marginal accuracy gain due to CoT).

It is now common to distinguish fast and implicit (system-1) from slow and explicit (system-2) reasoning. These two modes are equally important dimensions of human cognition. To understand—for example in debates about AGI—what LLMs are capable of, it seems pivotal to assess LLMs comprehensively along all dimensions of human cognition. In accuracy-based benchmarks without CoT, LLMs arguably solve tasks fast and implicitly. To assess the system-2 reasoning skill of LLMs, we need chain-of-thought based benchmarks. The Open CoT Leaderboard may therefore also inform broader debates about the significance of LLMs and AGI.  

## Which tasks are used?

The Open CoT Leaderboard evaluates LLMs’ ability to generate effective chain-of-thought reasoning traces for the following tasks:
- LogiQA (new translation of [original version](https://github.com/lgw863/LogiQA-dataset), and [version 2.0](https://github.com/csitfun/LogiQA2.0) with new examples)
- LSAT [dataset](https://arxiv.org/abs/2108.00648) (including subsets on [analytical reasoning](https://huggingface.co/datasets/hails/agieval-lsat-ar), [logical reasoning](https://huggingface.co/datasets/hails/agieval-lsat-lr), and [reading comprehension](https://huggingface.co/datasets/hails/agieval-lsat-rc))

Except for the original version of LogiQA, all these tasks are part of the [AGIEval benchmark](https://arxiv.org/abs/2304.06364), and have been re-published as [logikon-bench](https://huggingface.co/datasets/logikon/logikon-bench).

We’ve chosen these tasks because they
1. are generic, i.e. can be solved through reasoning and just require commonsense knowledge;
2. are still relatively difficult even for the most powerful LLMs (leaving enough room for improvement through chain-of-thought);
3. have been introduced as AI benchmarks before (in AGIEval) and are widely used (e.g., in the Nous benchmark suite).

All tasks are rendered as multiple-choice problems, with the answer options being enumerated in the prompt.

We use the following prompt template for assessing baseline and CoT accuracies – the reasoning traces (italics) are only added in the case “with CoT”:

```
Answer the following question about the given passage. Base your answer on the reasoning below.

Passage: <passage>
    
Question: <question>

A. <choice1>
B. <choice2>
…

Reasoning: <reasoning>
        
Answer:
```

## How are chain-of-thought traces generated? 
[#cot-generation]

The literature on chain-of-thought prompting has exploded in the last year, and the wealth of prompting strategies for CoT (involving, e.g., decoding, prompt wording, prompt chains, decomposition, aggregation and revision of reasoning traces) has become [sheerly mindblowing](https://github.com/logikon-ai/awesome-deliberative-prompting/#readme).

To cope with this variety, the evaluation pipeline behind the Open CoT Leaderboard implements CoT generation through modular and easily extensible prompt chains. So far, we have implemented two fairly simple prompting strategies: 
- Classic: Problem presentation followed by “Let’s think step by step.” instruction.
- Reflect: Problem presentation followed by instruction to reflect upon the problem in general terms before solving it step by step.

Let’s have a look at an example. The following problem is from the LSAT logical reasoning benchmark.

```
Editorial: The structure of the present school calendar was established to satisfy the requirements of early-twentieth-century agricultural life. In those days, farmers needed their children to have long breaks during which they could remain at home and help with the harvest. The contemporary school year is thus made up of periods of study interspersed with long breaks. But agricultural life no longer occupies most of our citizens, so we can now make changes that serve the interests of children. Therefore, long breaks should be removed from the school calendar.

Which one of the following is an assumption on which the editorial's argument depends?
A) A change in the present school calendar that shortened breaks would serve the interests of agricultural life.
B) Children of farmers need to continue observing a school calendar made up of periods of study interspersed with long breaks.
C) Long breaks in the school calendar should be replaced with breaks that are no longer than workers' average vacations.
D) A school calendar made up of periods of study without long breaks would serve the interests of children more than a school calendar with long breaks.
E) During long breaks children have a tendency to forget what they have learned.
```

Mixtral-8x7B-Instruct-v0.1 is doing fairly well in generating the following chain-of-thought traces:

 <table>
  <tr>
    <th>With prompting strategy Classic</th>
    <th>With prompting strategy Reflect</th>
  </tr>
  <tr>
    <td>The editorial argues that the current school calendar, with long breaks, was designed to accommodate agricultural life but should now be changed because agricultural life no longer occupies most of our citizens. It concludes that long breaks should be removed from the school calendar.

The assumption that the argument depends on is the belief that removing long breaks from the school calendar would serve the interests of children more than the current calendar. This is not explicitly stated in the editorial, but it is the underlying assumption that justifies the proposed change.

Therefore, the correct answer is:
D) A school calendar made up of periods of study without long breaks would serve the interests of children more than a school calendar with long breaks.</td>
    <td>The decision problem in abstract terms is to identify the assumption on which the editorial's argument depends. 

Common mistakes for this kind of problem include:
1. Misinterpreting the argument's premises or conclusion.
2. Overlooking the specific requirement of identifying an assumption.
3. Confusing an assumption with a conclusion or a premise.

A plan for how to solve this problem could be:
1. Identify the premises and conclusion of the argument.
2. List possible assumptions that could bridge the gap between the premises and the conclusion.
3. Evaluate each assumption to see if it is necessary for the argument.

Solving the problem step by step, following the plan:

1. The premises of the argument are:
   - The structure of the present school calendar was established to satisfy the requirements of early-twentieth-century agricultural life.
   - Agricultural life no longer occupies most of our citizens.

2. The conclusion of the argument is:
   - Long breaks should be removed from the school calendar.

3. Possible assumptions could be:
   - A school calendar made up of periods of study without long breaks would serve the interests of children more than a school calendar with long breaks.
   - Children of farmers no longer need to remain at home and help with the harvest during long breaks.
   - A change in the present school calendar that shortened breaks would not serve the interests of agricultural life.

4. Evaluating each assumption:
   - The argument does not depend on the assumption that children of farmers no longer need to remain at home and help with the harvest during long breaks, because the argument is about the interests of all children, not just those of farmers.
   - The argument does not depend on the assumption that a change in the present school calendar that shortened breaks would not serve the interests of agricultural life, because the argument is about the interests of children, not agricultural life.
   - The argument does depend on the assumption that a school calendar made up of periods of study without long breaks would serve the interests of children more than a school calendar with long breaks. This is the assumption that justifies the conclusion.

Therefore, the assumption on which the editorial's argument depends is D) A school calendar made up of periods of study without long breaks would serve the interests of children more than a school calendar with long breaks.</td>
  </tr>
</table> 

Each of our two prompting strategies—Classic and Reflect—is combined and run with multiple decoding parameters:
- Greedy.
- Beam search (n=2).
- Sampling (T=.3).

This gives us, in combination, six “CoT generation regimes.” When we evaluate a model, we let it generate, for every example in the test datasets, one chain-of-thought trace for each regime. We obtain, accordingly, six different numerical scores for accuracy with chain-of-thought (namely one for each regime). In the Open CoT Leaderboard, we report (for every model/task) the best marginal accuracy gain achieved under any regime.

## What are the main take-aways so far?

We’re gradually extending the Open CoT Leaderboard by evaluating more and more models, but current results (model count=30) already suggest some interesting insights.

- Mighty dwarfs: We have been very pleased to see that relatively small (7B) open LLMs are capable of effective, i.e. accuracy-improving chain-of-thought reasoning. 🎉
- Instruction- and chat-finetuning helps: Finetuned models score much better than their corresponding base models. More specifically, finetuning may improve both the baseline accuracy without CoT and the marginal accuracy gains achieved through CoT.
- Vindication of *textbooks-is-all-you-need*: The Phi-2 model clearly outperforms the Mixtral MoE base model on the Open CoT Leaderboard.
- Variable and ambiguous effects of CoT: Digging a bit deeper, we see that there is no single preferred or superior CoT generation [regime](#cot-generation). What works best for one model and one task might not work for another model, or another task. And sometimes CoT reduces accuracy rather than increasing it. We take this as a reminder that finding an implementation of CoT that is universally effective, reliable and robust remains a challenging problem.

## What are the next steps? – And how to contribute.

We’re planning to move ahead in different directions. And contributions to all these efforts are more than welcome. 

First, we’d love to evaluate your models! You can 📬 submit any open LLMs for evaluation on the Open CoT Leaderboard space, using the Submission tab!

Then, we’d love some help on the following coding and data analysis tasks.
- **Carry out in-depth analysis of full evaluation results.**
    For example, a qualitative analysis of the generated CoT traces to check whether they actually point to the correct answer choice. We’ve created a [notebook](https://github.com/logikon-ai/cot-eval/blob/main/notebooks/CoT_Leaderboard_Results_Exploration.ipynb) that shows how to access and explore the eval results and reasoning traces which back up the Open Cot Leaderboard. You can build on that and share your own analyses in the corresponding repo (or somewhere else, of course). Feel free to open an issue with suggestions or questions. In case you plan to use the data for research projects and want feedback, just drop a note.
- **Create Open CoT Dashboard.**
    The Open CoT Leaderboard contends with ranking models according to marginal accuracy gains. It doesn’t display the baseline accuracies, the variance, the scores for different CoT generation [regimes](#cot-generation), properties of the generated reasoning traces (e.g., length), etc. We think it would be super informative to complement the leaderboard with a dashboard (e.g., as an extra tab or a separate  HF space) that presents all this info and can be interactively explored by users. In case you’re interested in building such an Open CoT Dashboard (with or without us), just reach out.
- **More CoT chains.**
    We’re pondering implementing further CoT generation regimes. Promising candidates are, for example, self-consistency, tree-of-thought, self-check, or debating. Want to help us with that? Get in touch! (🤫: Why not choose such a project for your master’s or bachelor’s thesis?)
- **More tasks and test datasets.**
    The Open CoT Leaderboard is arguably built on a rather narrow set of benchmarks. Once we have free compute resources, we’d like to include further challenging reasoning tasks. We’d be happy to learn which tasks you’d like to see included in the Open CoT Leaderboard.

Here’s where we can exchange our ideas and collaborate:
- For non-technical suggestions and feedback, join the [discussion](https://huggingface.co/spaces/logikon/open_cot_leaderboard/discussions) at the leaderboard’s HF space.
- For technical feedback and questions, open an issue at our GitHub [repo](https://github.com/logikon-ai/cot-eval).

Looking forward to hearing from you!
