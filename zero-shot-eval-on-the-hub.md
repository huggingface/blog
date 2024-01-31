---
title: "Very Large Language Models and How to Evaluate Them" 
thumbnail: /blog/assets/106_zero_shot_eval_on_the_hub/thumbnail.png
authors:
- user: mathemakitten
- user: Tristan
- user: abhishek
- user: lewtun
- user: douwekiela
---

# Very Large Language Models and How to Evaluate Them 


Large language models can now be evaluated on zero-shot classification tasks with [Evaluation on the Hub](https://huggingface.co/spaces/autoevaluate/model-evaluator)! 

Zero-shot evaluation is a popular way for researchers to measure the performance of large language models, as they have been [shown](https://arxiv.org/abs/2005.14165) to learn capabilities during training without explicitly being shown labeled examples. The [Inverse Scaling Prize](https://github.com/inverse-scaling/prize) is an example of a recent community effort to conduct large-scale zero-shot evaluation across model sizes and families to discover tasks on which larger models may perform worse than their smaller counterparts.

![dataset](assets/106_zero_shot_eval_on_the_hub/zeroshot.jpg)

## Enabling zero-shot evaluation of language models on the Hub

[Evaluation on the Hub](https://huggingface.co/blog/eval-on-the-hub) helps you evaluate any model on the Hub without writing code, and is powered by [AutoTrain](https://huggingface.co/autotrain). Now, any causal language model on the Hub can be evaluated in a zero-shot fashion. Zero-shot evaluation measures the likelihood of a trained model producing a given set of tokens and does not require any labelled training data, which allows researchers to skip expensive labelling efforts. 

We’ve upgraded the AutoTrain infrastructure for this project so that large models can be evaluated for free 🤯! It’s expensive and time-consuming for users to figure out how to write custom code to evaluate big models on GPUs. For example, a language model with 66 billion parameters may take 35 minutes just to load and compile, making evaluation of large models accessible only to those with expensive infrastructure and extensive technical experience. With these changes, evaluating a model with 66-billion parameters on a zero-shot classification task with 2000 sentence-length examples takes 3.5 hours and can be done by anyone in the community. Evaluation on the Hub currently supports evaluating models up to 66 billion parameters, and support for larger models is to come.

The zero-shot text classification task takes in a dataset containing a set of prompts and possible completions. Under the hood, the completions are concatenated with the prompt and the log-probabilities for each token are summed, then normalized and compared with the correct completion to report accuracy of the task.

In this blog post, we’ll use the zero-shot text classification task to evaluate various [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) models on [WinoBias](https://uclanlp.github.io/corefBias/overview), a coreference task measuring gender bias related to occupations. WinoBias measures whether a model is more likely to pick a stereotypical pronoun to fill in a sentence mentioning an occupation, and observe that the results suggest an [inverse scaling](https://github.com/inverse-scaling/prize) trend with respect to model size.

## Case study: Zero-shot evaluation on the WinoBias task

The [WinoBias](https://github.com/uclanlp/corefBias) dataset has been formatted as a zero-shot task where classification options are the completions. Each completion differs by the pronoun, and the target corresponds to the anti-stereotypical completion for the occupation (e.g. "developer" is stereotypically a male-dominated occupation, so "she" would be the anti-stereotypical pronoun). See [here](https://huggingface.co/datasets/mathemakitten/winobias_antistereotype_test) for an example:

![dataset](assets/106_zero_shot_eval_on_the_hub/dataset.png)

Next, we can select this newly-uploaded dataset in the Evaluation on the Hub interface using the `text_zero_shot_classification` task, select the models we’d like to evaluate, and submit our evaluation jobs! When the job has been completed, you’ll be notified by email that the autoevaluator bot has opened a new pull request with the results on the model’s Hub repository.

![Evaluation on the Hub](assets/106_zero_shot_eval_on_the_hub/eval_hub.png)

Plotting the results from the WinoBias task, we find that smaller models are more likely to select the anti-stereotypical pronoun for a sentence, while larger models are more likely to learn stereotypical associations between gender and occupation in text. This corroborates results from other benchmarks (e.g. [BIG-Bench](https://arxiv.org/abs/2206.04615)) which show that larger, more capable models are more likely to be biased with regard to gender, race, ethnicity, and nationality, and [prior work](https://www.deepmind.com/publications/scaling-language-models-methods-analysis-insights-from-training-gopher) which shows that larger models are more likely to generate toxic text. 

![Winobias](./assets/106_zero_shot_eval_on_the_hub/winobias.png)

## Enabling better research tools for everyone

Open science has made great strides with community-driven development of tools like the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) by EleutherAI and the [BIG-bench](https://github.com/google/BIG-bench) project, which make it straightforward for researchers to understand the behaviour of state-of-the-art models. 

Evaluation on the Hub is a low-code tool which makes it simple to compare the zero-shot performance of a set of models along an axis such as FLOPS or model size, and to compare the performance of a set of models trained on a specific corpora against a different set of models. The zero-shot text classification task is extremely flexible—any dataset that can be permuted into a Winograd schema where examples to be compared only differ by a few words can be used with this task and evaluated on many models at once. Our goal is to make it simple to upload a new dataset for evaluation and enable researchers to easily benchmark many models on it.

An example research question which can be addressed with tools like this is the inverse scaling problem: while larger models are generally more capable at the majority of language tasks, there are tasks where larger models perform worse. The [Inverse Scaling Prize](https://github.com/inverse-scaling/prize) is a competition which challenges researchers to construct tasks where larger models perform worse than their smaller counterparts. We encourage you to try zero-shot evaluation on models of all sizes with your own tasks! If you find an interesting trend along model sizes, consider submitting your findings to round 2 of the [Inverse Scaling Prize](https://github.com/inverse-scaling/prize).

## Send us feedback!

At Hugging Face, we’re excited to continue democratizing access to state-of-the-art machine learning models, and that includes developing tools to make it easy for everyone to evaluate and probe their behavior. We’ve previously [written](https://huggingface.co/blog/eval-on-the-hub) about how important it is to standardize model evaluation methods to be consistent and reproducible, and to make tools for evaluation accessible to everyone. Future plans for Evaluation on the Hub include supporting zero-shot evaluation for language tasks which might not lend themselves to the format of concatenating completions to prompts, and adding support for even larger models.

One of the most useful things you can contribute as part of the community is to send us feedback! We’d love to hear from you on top priorities for model evaluation. Let us know your feedback and feature requests by posting on the Evaluation on the Hub [Community](https://huggingface.co/spaces/autoevaluate/model-evaluator/discussions) tab, or the [forums](https://discuss.huggingface.co/)!



