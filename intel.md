---
title: "Intel and Hugging Face Partner to Democratize Machine Learning Hardware Acceleration"
thumbnail: /blog/assets/XXX
---

<h1>Intel and Hugging Face Partner to Democratize Machine Learning Hardware Acceleration</h1>

<div class="blog-metadata">
    <small>Published May 16, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/intel.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="https://twitter.com/julsimon">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1633343465505-noauth.jpeg?w=128&h=128&f=face" title="Julien Simon">
        <div class="bfc">
            <code>juliensimon</code>
            <span class=fullname">Julien Simon</span>
        </div>
    </a>
</div>


Today, Hugging Face is excited to announce that Intel has officially joined the [Hardware Partner Program](https://huggingface.co/hardware).  Thanks to the [Optimum](https://github.com/huggingface/optimum-intel) open-source library, the two companies will keep simplifying state-of-the-art hardware acceleration for the Machine Learning (ML) community,

The mission of Hugging Face is to democratize good machine learning and maximize its positive impact across industries and society. Not only do we strive to advance the state of the art for Transformers models, but we also work hard on simplifying their adoption.

Transformer models are increasingly large and complex, which can cause production challenges for latency-sensitive applications like search or chatbots. Optimizing latency has long been a long and difficult endeavor for machine learning practitioners. Even with deep knowledge of the underlying machine learning framework and hardware platform, figuring out which knobs and features to leverage takes a lot of trial and error.

With the Intel Xeon Scalable CPU platform and a wide range of hardware-optimized AI software tools, frameworks, and libraries, Intel provides a complete foundation for accelerated AI. Thus, it made perfect sense for Hugging Face and Intel to join forces and collaborate on building powerful and simple model optimization tools.

“*We’re excited to work with Hugging Face to bring the latest innovations of Intel Xeon hardware and Intel AI software to the Transformers community through open source integration and integrated developer experiences.*”, says Wei Li, Intel Vice President & General Manager, AI and Analytics.

In the past months, Intel and Hugging Face started to collaborate on scaling Transformer inference on Intel Xeon CPUs. We published detailed tuning guides and benchmarks ([part 1](https://huggingface.co/blog/bert-cpu-scaling-part-1), [part 2](https://huggingface.co/blog/bert-cpu-scaling-part-2)) and we achieved [single-digit millisecond latency](https://huggingface.co/blog/infinity-cpu-performance) for DistilBERT on the latest Intel Xeon Ice Lake CPUs.

The next logical step was to expand on this work and share it with the ML community. Enter the [Optimum Intel](https://github.com/huggingface/optimum-intel) open source library! Let’s take a deeper look at it.

## Get Peak Transformers Performance with Optimum Intel
[Optimum](https://github.com/huggingface/optimum) is an open-source library created by Hugging Face to simplify Transformer acceleration across a growing range of training and inference devices. Thanks to built-in optimization techniques, you can start optimizing your workloads in minutes, using ready-made scripts, or applying minimal changes to your existing code. Beginners can use Optimum out of the box with great results, and experts can keep tweaking for maximum performance. 

[Optimum Intel](https://github.com/huggingface/optimum-intel) is part of Optimum and builds on top of the [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) (INC). INC is an [open-source library](https://github.com/intel/neural-compressor) that delivers unified interfaces across multiple deep learning frameworks for popular network compression technologies, such as quantization, pruning, and knowledge distillation. This tool supports automatic accuracy-driven tuning strategies to help users quickly find out the best quantized model. It also implements different weight pruning algorithms to generate pruned models with predefined sparsity goals and supports knowledge distillation to distill the knowledge from the teacher model to the student model.

With Optimum Intel, you can apply state-of-the-art optimization techniques to your Transformers with minimal effort. Let’s look at a complete example.

## Case study: Quantizing DistilBERT for X% Speedup with Optimum Intel

In this example, we’re going to run post-training quantization on a classification model based on the DistilBERT architecture. Quantization is a process that reduces memory and compute requirements by reducing the bit width of model parameters. For example, you can often replace 32-bit floating-point parameters with 8-bit integers at the expense of a small drop in prediction accuracy.

We have already fine-tuned the original model to classify product reviews for shoes according to their star rating (from 1 to 5 stars). You can view this [model](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews) and its [quantized](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews-quantized?) version on the Hugging Face hub. You can also test the original model in this [Space](https://huggingface.co/spaces/juliensimon/amazon-shoe-reviews-spaces). 

Let’s get started! All code is available in this [notebook](https://gitlab.com/juliensimon/huggingface-demos/-/blob/main/amazon-shoes/03_optimize_inc_quantize.ipynb). 

As usual, the first step is to install all required libraries. It’s worth mentioning that we have to work with a CPU-only version of PyTorch for the quantization process to work correctly.

```
pip -q uninstall torch -y 
pip -q install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip -q install transformers datasets optimum[intel] evaluate --upgrade
```

Then, we prepare an evaluation dataset to assess the model performance during quantization. Starting from the dataset we used to fine-tune the original model, we only keep a few thousand reviews and their labels and save them to local storage.

Next, we load the original model from the Hugging Face hub, its tokenizer, and the evaluation dataset.

```
model_name = "juliensimon/distilbert-amazon-shoe-reviews" 
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name) 
eval_dataset = load_from_disk("./data/amazon_shoe_reviews_test")
```

Next, we define an evaluation function that measures the accuracy of the model on the evaluation dataset. This allows the Optimum Intel library to compare metrics before and after quantization. For this purpose, the Hugging Face [evaluate](https://github.com/huggingface/evaluate/) library is very convenient!

```
from evaluate import evaluator

def eval_func(model): 
    eval = evaluator("text-classification") 
    results = eval.compute(
        model_or_pipeline=model, tokenizer=tokenizer, 
        data=eval_dataset, metric=evaluate.load("accuracy"), 
    label_column="labels", label_mapping=model.config.label2id
    ) 
    return results["accuracy"]
```

We then set up the quantization job using a [configuration file](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews/blob/main/quantize.yml) that we download from the Hugging Face hub. You can find details on this file in the INC [documentation](https://intel.github.io/neural-compressor/docs/tuning_strategies.html). Here, we go for post-training dynamic quantization with an acceptable accuracy drop of 3%.

```
quantization:
    approach: post_training_dynamic_quant
tuning:
    accuracy_criterion:
        relative: 0.03               
```

Next, we create the corresponding quantization objects with the Optimum Intel API.

```
from optimum.intel.neural_compressor import IncOptimizer, IncQuantizer
from optimum.intel.neural_compressor.configuration import IncQuantizationConfig

quantization_config = IncQuantizationConfig.from_pretrained(
    config_name_or_path="juliensimon/distilbert-amazon-shoe-reviews",
    config_file_name='quantize.yml'
)
inc_quantizer = IncQuantizer(model, quantization_config, eval_func=eval_func)
quantizer = inc_quantizer.fit()
```

We can now launch the quantization job.

```
inc_optimizer = IncOptimizer(model, quantizer=quantizer)
inc_model = inc_optimizer.fit()
```

The log tells us that 39 ```Linear``` operators have been quantized. 

```
[INFO] |*****Mixed Precision Statistics*****|
[INFO] +--------------+-----------+---------+
[INFO] |   Op Type    |   Total   |   INT8  |
[INFO] +--------------+-----------+---------+
[INFO] |    Linear    |     38    |    38   |
[INFO] +--------------+-----------+---------+
[INFO] Pass quantize model elapsed time: 900.74 ms
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

Before and after each quantization step, Optimum Intel runs the evaluation function on the current model. Interestingly, the accuracy of the quantized model is now a bit higher (```0.6906```) than the original model (```0.6893```). Likewise, we see that the quantized model predicts the evaluation set 13% faster than the original model. Not bad for a few lines of code!

```
[INFO] |***********************Tune Result Statistics**********************|
[INFO] +--------------------+-----------+---------------+------------------+
[INFO] |     Info Type      |  Baseline | Tune 1 result | Best tune result |
[INFO] +--------------------+-----------+---------------+------------------+
[INFO] |      Accuracy      |  0.6893   |    0.6906     |     0.6906       |
[INFO] | Duration (seconds) | 106.3149  |    92.4634    |     92.4634      |
[INFO] +--------------------+-----------+---------------+------------------+
```

As the accuracy didn’t drop, Optimum Intel stopped the quantization job after the first step. Had the accuracy drop been higher than the 3% defined in the configuration file, Optimum Intel would have tried to quantize different parts of the models until the drop was within budget, or the maximum number of trials was reached.

Finally, we save the model and its configuration file to local storage.

```
import yaml

model_dir = "./model_inc"
inc_model.model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
with open("{}/best_configure.yaml".format(model_dir), "w") as f:
    yaml.dump(inc_model.tune_cfg, f, default_flow_style=False)
```

Once we’ve created a new model [repository](https://huggingface.co/juliensimon/distilbert-amazon-shoe-reviews-quantized) on the Hugging Face hub and pushed the model to it, we can load the model again in the usual way and work with it.

```
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForSequenceClassification

inc_model = IncQuantizedModelForSequenceClassification.from_pretrained(
    "juliensimon/distilbert-amazon-shoe-reviews-quantized")
```

## We’re only getting started

In this example, we showed you how to easily quantize models post-training with Optimum Intel, and that’s just the beginning. The library supports other types of quantization, as well as pruning, a technique that zeroes or removes model parameters that have little or no impact on the predicted outcome.

We are excited to partner with Intel to make the peak CPU efficiency from the latest Xeon hardware and Intel AI libraries accessible to Hugging Face users. Stay tuned - [give Optimum Intel a star](https://github.com/huggingface/optimum-intel) to get updates - for many new features to come!

*Many thanks to Ella Charlaix for her help on this post.*















