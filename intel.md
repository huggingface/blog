---
title: "Intel and Hugging Face Partner to Democratize Machine Learning Hardware Acceleration"
thumbnail: /blog/assets/80_intel/01.png
authors:
- user: juliensimon
---



<h1>Intel and Hugging Face Partner to Democratize Machine Learning Hardware Acceleration</h1>



{blog_metadata}
{authors}

![image](assets/80_intel/01.png)

The mission of Hugging Face is to democratize good machine learning and maximize its positive impact across industries and society. Not only do we strive to advance Transformer models, but we also work hard on simplifying their adoption.

Today, we're excited to announce that Intel has officially joined our [Hardware Partner Program](https://huggingface.co/hardware).  Thanks to the [Optimum](https://github.com/huggingface/optimum-intel) open-source library, Intel and Hugging Face will collaborate to build state-of-the-art hardware acceleration to train, fine-tune and predict with Transformers.

Transformer models are increasingly large and complex, which can cause production challenges for latency-sensitive applications like search or chatbots. Unfortunately, latency optimization has long been a hard problem for Machine Learning (ML) practitioners. Even with deep knowledge of the underlying framework and hardware platform, it takes a lot of trial and error to figure out which knobs and features to leverage.

Intel provides a complete foundation for accelerated AI with the Intel Xeon Scalable CPU platform and a wide range of hardware-optimized AI software tools, frameworks, and libraries. Thus, it made perfect sense for Hugging Face and Intel to join forces and collaborate on building powerful model optimization tools that let users achieve the best performance, scale, and productivity on Intel platforms.

“*We’re excited to work with Hugging Face to bring the latest innovations of Intel Xeon hardware and Intel AI software to the Transformers community, through open source integration and integrated developer experiences.*”, says Wei Li, Intel Vice President & General Manager, AI and Analytics.

In recent months, Intel and Hugging Face collaborated on scaling Transformer workloads. We published detailed tuning guides and benchmarks on inference ([part 1](https://huggingface.co/blog/bert-cpu-scaling-part-1), [part 2](https://huggingface.co/blog/bert-cpu-scaling-part-2)) and achieved [single-digit millisecond latency](https://huggingface.co/blog/infinity-cpu-performance) for DistilBERT on the latest Intel Xeon Ice Lake CPUs. On the training side, we added support for [Habana Gaudi](https://huggingface.co/blog/getting-started-habana) accelerators, which deliver up to 40% better price-performance than GPUs.

The next logical step was to expand on this work and share it with the ML community. Enter the [Optimum Intel](https://github.com/huggingface/optimum-intel) open source library! Let’s take a deeper look at it.

## Get Peak Transformers Performance with Optimum Intel
[Optimum](https://github.com/huggingface/optimum) is an open-source library created by Hugging Face to simplify Transformer acceleration across a growing range of training and inference devices. Thanks to built-in optimization techniques, you can start accelerating your workloads in minutes, using ready-made scripts, or applying minimal changes to your existing code. Beginners can use Optimum out of the box with excellent results. Experts can keep tweaking for maximum performance. 

[Optimum Intel](https://github.com/huggingface/optimum-intel) is part of Optimum and builds on top of the [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) (INC). INC is an [open-source library](https://github.com/intel/neural-compressor) that delivers unified interfaces across multiple deep learning frameworks for popular network compression technologies, such as quantization, pruning, and knowledge distillation. This tool supports automatic accuracy-driven tuning strategies to help users quickly build the best quantized model.

With Optimum Intel, you can apply state-of-the-art optimization techniques to your Transformers with minimal effort. Let’s look at a complete example.

## Case study: Quantizing DistilBERT with Optimum Intel

In this example, we will run post-training quantization on a DistilBERT model fine-tuned for classification. Quantization is a process that shrinks memory and compute requirements by reducing the bit width of model parameters. For example, you can often replace 32-bit floating-point parameters with 8-bit integers at the expense of a small drop in prediction accuracy.

We have already fine-tuned the original model to classify product reviews for shoes according to their star rating (from 1 to 5 stars). You can view this [model](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews) and its [quantized](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews-quantized?) version on the Hugging Face hub. You can also test the original model in this [Space](https://huggingface.co/spaces/juliensimon/amazon-shoe-reviews-spaces). 

Let’s get started! All code is available in this [notebook](https://gitlab.com/juliensimon/huggingface-demos/-/blob/main/amazon-shoes/03_optimize_inc_quantize.ipynb). 

As usual, the first step is to install all required libraries. It’s worth mentioning that we have to work with a CPU-only version of PyTorch for the quantization process to work correctly.

```
pip -q uninstall torch -y 
pip -q install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip -q install transformers datasets optimum[neural-compressor] evaluate --upgrade
```

Then, we prepare an evaluation dataset to assess model performance during quantization. Starting from the dataset we used to fine-tune the original model, we only keep a few thousand reviews and their labels and save them to local storage.

Next, we load the original model, its tokenizer, and the evaluation dataset from the Hugging Face hub.

```
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "juliensimon/distilbert-amazon-shoe-reviews"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eval_dataset = load_dataset("prashantgrao/amazon-shoe-reviews", split="test").select(range(300))
```

Next, we define an evaluation function that computes model metrics on the evaluation dataset. This allows the Optimum Intel library to compare these metrics before and after quantization. For this purpose, the Hugging Face [evaluate](https://github.com/huggingface/evaluate/) library is very convenient!

```
import evaluate

def eval_func(model):
    task_evaluator = evaluate.evaluator("text-classification")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_dataset,
        metric=evaluate.load("accuracy"),
        label_column="labels",
        label_mapping=model.config.label2id,
    )
    return results["accuracy"]
```

We then set up the quantization job using a [configuration]. You can find details on this configuration on the Neural Compressor [documentation](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md). Here, we go for post-training dynamic quantization with an acceptable accuracy drop of 5%. If accuracy drops more than the allowed 5%, different part of the model will then be quantized until it an acceptable drop in accuracy or if the maximum number of trials, here set to 10, is reached.

```
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion

tuning_criterion = TuningCriterion(max_trials=10)
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(
    approach="dynamic",
    accuracy_criterion=accuracy_criterion,
    tuning_criterion=tuning_criterion,
)
```

We can now launch the quantization job and save the resulting model and its configuration file to local storage.

```
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer

# The directory where the quantized model will be saved
save_dir = "./model_inc"
quantizer = INCQuantizer.from_pretrained(model=model, eval_fn=eval_func)
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
```

The log tells us that Optimum Intel has quantized 38 ```Linear``` and 2 ```Embedding``` operators.

```
[INFO] |******Mixed Precision Statistics*****|
[INFO] +----------------+----------+---------+
[INFO] |    Op Type     |  Total   |   INT8  |
[INFO] +----------------+----------+---------+
[INFO] |   Embedding    |    2     |    2    |
[INFO] |     Linear     |    38    |    38   |
[INFO] +----------------+----------+---------+
```

Comparing the first layer of the original model (```model.distilbert.transformer.layer[0]```) and its quantized version (```inc_model.distilbert.transformer.layer[0]```), we see that ```Linear``` has indeed been replaced by ```DynamicQuantizedLinear```, its quantized equivalent.

```
# Original model

TransformerBlock(
  (attention): MultiHeadSelfAttention(
    (dropout): Dropout(p=0.1, inplace=False)
    (q_lin): Linear(in_features=768, out_features=768, bias=True)
    (k_lin): Linear(in_features=768, out_features=768, bias=True)
    (v_lin): Linear(in_features=768, out_features=768, bias=True)
    (out_lin): Linear(in_features=768, out_features=768, bias=True)
  )
  (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (ffn): FFN(
    (dropout): Dropout(p=0.1, inplace=False)
    (lin1): Linear(in_features=768, out_features=3072, bias=True)
    (lin2): Linear(in_features=3072, out_features=768, bias=True)
  )
  (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
)
```

```
# Quantized model

TransformerBlock(
  (attention): MultiHeadSelfAttention(
    (dropout): Dropout(p=0.1, inplace=False)
    (q_lin): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_channel_affine)
    (k_lin): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_channel_affine)
    (v_lin): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_channel_affine)
    (out_lin): DynamicQuantizedLinear(in_features=768, out_features=768, dtype=torch.qint8, qscheme=torch.per_channel_affine)
  )
  (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (ffn): FFN(
    (dropout): Dropout(p=0.1, inplace=False)
    (lin1): DynamicQuantizedLinear(in_features=768, out_features=3072, dtype=torch.qint8, qscheme=torch.per_channel_affine)
    (lin2): DynamicQuantizedLinear(in_features=3072, out_features=768, dtype=torch.qint8, qscheme=torch.per_channel_affine)
  )
  (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
)
```

Very well, but how does this impact accuracy and prediction time?

Before and after each quantization step, Optimum Intel runs the evaluation function on the current model. The accuracy of the quantized model is now a bit lower  (``` 0.546```) than the original model (```0.574```). We also see that the evaluation step of the quantized model was 1.34x faster than the original model. Not bad for a few lines of code!

```
[INFO] |**********************Tune Result Statistics**********************|
[INFO] +--------------------+----------+---------------+------------------+
[INFO] |     Info Type      | Baseline | Tune 1 result | Best tune result |
[INFO] +--------------------+----------+---------------+------------------+
[INFO] |      Accuracy      | 0.5740   |    0.5460     |     0.5460       |
[INFO] | Duration (seconds) | 13.1534  |    9.7695     |     9.7695       |
[INFO] +--------------------+----------+---------------+------------------+
```

You can find the resulting [model](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews-quantized) hosted on the Hugging Face hub. To load a quantized model hosted locally or on the 🤗 hub, you can do as follows :


```
from optimum.intel.neural_compressor import INCModelForSequenceClassification

inc_model = INCModelForSequenceClassification.from_pretrained(save_dir)
```

## We’re only getting started

In this example, we showed you how to easily quantize models post-training with Optimum Intel, and that’s just the beginning. The library supports other types of quantization as well as pruning, a technique that zeroes or removes model parameters that have little or no impact on the predicted outcome.

We are excited to partner with Intel to bring Hugging Face users peak efficiency on the latest Intel Xeon CPUs and Intel AI libraries. Please [give Optimum Intel a star](https://github.com/huggingface/optimum-intel) to get updates, and stay tuned for many upcoming features!

*Many thanks to [Ella Charlaix](https://github.com/echarlaix) for her help on this post.*















