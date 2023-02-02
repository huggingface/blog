---
title: Hyperparameter Search with Transformers and Ray Tune
thumbnail: /blog/assets/06_ray_tune/ray-hf.jpg
authors:
- user: ray-project
  guest: true
---

<h1>
    Hyperparameter Search with Transformers and Ray Tune
</h1>

{blog_metadata}

{authors}

##### A guest blog post by Richard Liaw from the Anyscale team

With cutting edge research implementations, thousands of trained models easily accessible, the Hugging Face [transformers](https://github.com/huggingface/transformers) library has become critical to the success and growth of natural language processing today.

For any machine learning model to achieve good performance, users often need to implement some form of parameter tuning. Yet, nearly everyone ([1](https://medium.com/@prakashakshay90/fine-tuning-bert-model-using-pytorch-f34148d58a37), [2](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#advantages-of-fine-tuning)) either ends up disregarding hyperparameter tuning or opting to do a simplistic grid search with a small search space.

However, simple experiments are able to show the benefit of using an advanced tuning technique. Below is [a recent experiment run on a BERT](https://medium.com/distributed-computing-with-ray/hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b) model from [Hugging Face transformers](https://github.com/huggingface/transformers) on the [RTE dataset](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool). Genetic optimization techniques like [PBT](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#population-based-training-tune-schedulers-populationbasedtraining) can provide large performance improvements compared to standard hyperparameter optimization techniques.



<table>
  <tr>
   <td><strong>Algorithm</strong>
   </td>
   <td><strong>Best Val Acc.</strong>
   </td>
   <td><strong>Best Test Acc.</strong>
   </td>
   <td><strong>Total GPU min</strong>
   </td>
   <td><strong>Total $ cost</strong>
   </td>
  </tr>
  <tr>
   <td>Grid Search
   </td>
   <td>74%
   </td>
   <td>65.4%
   </td>
   <td>45 min
   </td>
   <td>$2.30
   </td>
  </tr>
  <tr>
   <td>Bayesian Optimization +Early Stop
   </td>
   <td>77%
   </td>
   <td>66.9%
   </td>
   <td>104 min
   </td>
   <td>$5.30
   </td>
  </tr>
  <tr>
   <td>Population-based Training
   </td>
   <td>78%
   </td>
   <td>70.5%
   </td>
   <td>48 min
   </td>
   <td>$2.45
   </td>
  </tr>
</table>


If you’re leveraging [Transformers](https://github.com/huggingface/transformers), you’ll want to have a way to easily access powerful hyperparameter tuning solutions without giving up the customizability of the Transformers framework.


![alt_text](/blog/assets/06_ray_tune/ray-hf.jpg "image_tooltip")


In the Transformers 3.1 release, [Hugging Face Transformers](https://github.com/huggingface/transformers) and [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) teamed up to provide a simple yet powerful integration. [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a popular Python library for hyperparameter tuning that provides many state-of-the-art algorithms out of the box, along with integrations with the best-of-class tooling, such as [Weights and Biases](https://wandb.ai/) and tensorboard.

To demonstrate this new [Hugging Face](https://github.com/huggingface/transformers) + [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) integration, we leverage the [Hugging Face Datasets library](https://github.com/huggingface/datasets) to fine tune BERT on [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

To run this example, please first run:

**`pip install "ray[tune]" transformers datasets scipy sklearn torch`**

Simply plug in one of Ray’s standard tuning algorithms by just adding a few lines of code.


```python
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = load_dataset('glue', 'mrpc')
metric = load_metric('glue', 'mrpc')

def encode(examples):
    outputs = tokenizer(
        examples['sentence1'], examples['sentence2'], truncation=True)
    return outputs

encoded_dataset = dataset.map(encode, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(
    "test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    n_trials=10 # number of trials
)
```

By default, each trial will utilize 1 CPU, and optionally 1 GPU if available.
You can leverage multiple [GPUs for a parallel hyperparameter search](https://docs.ray.io/en/latest/tune/user-guide.html#resources-parallelism-gpus-distributed)
by passing in a `resources_per_trial` argument.

You can also easily swap different parameter tuning algorithms such as [HyperBand](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#asha-tune-schedulers-ashascheduler), [Bayesian Optimization](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html), [Population-Based Training](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#population-based-training-tune-schedulers-populationbasedtraining):

To run this example, first run: **`pip install hyperopt`**

```python
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    # Choose among many libraries:
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg=HyperOptSearch(metric="objective", mode="max"),
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=ASHAScheduler(metric="objective", mode="max"))
```


It also works with [Weights and Biases](https://wandb.ai/) out of the box!

![alt_text](/blog/assets/06_ray_tune/ray-wandb.png "image_tooltip")



### Try it out today:

*   `pip install -U ray`
*   `pip install -U transformers datasets`
*   Check out the [Hugging Face documentation](https://huggingface.co/transformers/) and [Discussion thread](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10)
*   [End-to-end example of using Hugging Face hyperparameter search for text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb)

If you liked this blog post, be sure to check out:

*   [Transformers + GLUE + Ray Tune example](https://docs.ray.io/en/latest/tune/examples/index.html#hugging-face-huggingface-transformers)
*   Our [Weights and Biases report](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Huggingface-Transformers--VmlldzoyMTc2ODI) on Hyperparameter Optimization for Transformers
*   The [simplest way to serve your NLP model](https://medium.com/distributed-computing-with-ray/the-simplest-way-to-serve-your-nlp-model-in-production-with-pure-python-d42b6a97ad55) from scratch
