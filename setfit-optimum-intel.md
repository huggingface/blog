---
title: "Blazing Fast SetFit Inference with 🤗 Optimum Intel on Xeon"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: dkorat
  guest: true
- user: tomaarsen
- user: orenpereg
  guest: true
- user: moshew
  guest: true
- user: echarlaix
- user: aprabh2
  guest: true
---

# Blazing Fast SetFit Inference with 🤗 Optimum Intel on Xeon

SetFit is a promising solution for a common modeling problem: how to deal with lack of labeled data for training. Developed with Hugging Face’s research partners at [Intel Labs](https://www.intel.com/content/www/us/en/research/overview.html) and the [UKP Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp), SetFit is an efficient framework for few-shot fine-tuning of [Sentence Transformers](https://sbert.net/) models. 

SetFit achieves high accuracy with little labeled data - for example, SetFit [outperforms](https://arxiv.org/pdf/2311.06102.pdf) GPT-3.5 in 3-shot prompting and with 5 shot it also outperforms 3-shot GPT-4 on the Banking 77 financial intent dataset.

Compared to LLM based methods, SetFit has two unique advantages:

<p>🗣 <strong>No prompts or verbalisers</strong>: few-shot in-context learning with LLMs requires handcrafted prompts which make the results brittle, sensitive to phrasing and dependent on user expertise. SetFit dispenses with prompts altogether by generating rich embeddings directly from a small number of labeled text examples.</p>

<p>🏎 <strong>Fast to train</strong>: SetFit doesn't rely on LLMs such as GPT-3.5 or Llama2 to achieve high accuracy. As a result, it is typically an order of magnitude (or more) faster to train and run inference with.</p>

For more details on SetFit, check out our [paper](https://arxiv.org/abs/2209.11055), [blog](https://huggingface.co/blog/setfit), [code](https://github.com/huggingface/setfit), and [data](https://huggingface.co/SetFit).

Setfit has been widely adopted by the AI developer community, with \~100k downloads per month and [\~1500](https://huggingface.co/models?library=setfit) SetFit models on the Hub, and growing with an average of ~4 models per day!

## Faster!

In this blog post, we'll explain how you can accelerate inference with SetFit by **7.8x** on Intel CPUs, by optimizing your SetFit model with 🤗 [Optimum Intel](https://github.com/huggingface/optimum-intel). We’ll show how you can achieve huge throughput gains by performing a simple post-training quantization step on your model. This can enable production-grade deployment of SetFit solutions using Intel Xeon CPUs. 

[Optimum Intel](https://github.com/huggingface/optimum-intel) is an open-source library that accelerates end-to-end pipelines built with Hugging Face libraries on Intel Hardware. Optimum Intel includes several techniques to accelerate models such as low-bit quantization, model weight pruning, distillation, and an accelerated runtime.

The runtime and optimizations included in [Optimum Intel](https://github.com/huggingface/optimum-intel) take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512), Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs to accelerate models. Specifically, it has built-in [BFloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) (bf16) and int8 GEMM accelerators in every core to accelerate deep learning training and inference workloads. AMX accelerated inference is introduced in PyTorch 2.0 and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) (IPEX) in addition to other optimizations for various common operators.

Optimizing pre-trained models can be done easily with Optimum Intel; many simple examples can be found [here](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc).
Our blog is accompanied by a [notebook](https://github.com/huggingface/setfit/blob/main/notebooks/setfit-optimum-intel.ipynb) for a step-by-step walkthrough.

## Step 1: Quantize the SetFit Model using 🤗 Optimum Intel

In order to optimize our SetFit model, we will apply quantization to the model body, using [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) (INC), part of Optimum Intel.

**Quantization** is a very popular deep learning model optimization technique for improving inference speeds. It minimizes the number of bits required to represent the weights and/or activations in a neural network. This is done by converting a set of high-precision  numbers into a lower-bit data representations, such as INT8. Moreover, quantization can enable faster computations in lower precision.

Specifically, we'll apply post-training static quantization (PTQ). PTQ can reduce the memory footprint and latency for inference, while still preserving the accuracy of the model, with only a small unlabeled calibration set and without any training.
Before you begin, make sure you have all the necessary libraries installed and that your version of Optimum Intel is at least `1.14.0` since the functionality was introduced in that version:

```bash
pip install --upgrade-strategy eager optimum[ipex]
```

### Prepare a Calibration Dataset

The calibration dataset should be able to represent the distribution of unseen data. In general, preparing 100 samples is enough for calibration. We'll use the `rotten_tomatoes` dataset in our case, since it’s composed of movie reviews, similar to our target dataset, `sst2`.

First, we’ll load 100 random samples from this dataset. Then, to prepare the dataset for quantization, we'll need to tokenize each example. We won’t need the “text” and “label” columns, so let’s remove them.

```python
calibration_set = load_dataset("rotten_tomatoes", split="train").shuffle(seed=42).select(range(100)) 

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)
 
tokenizer = setfit_model.model_body.tokenizer
calibration_set = calibration_set.map(tokenize, remove_columns=["text", "label"])
```

### Run Quantization

Before we run quantization, we need to define the desired quantization process - in our case - **Static Post Training Quantization**, and use `optimum.intel` to run the quantization on our calibration dataset:

```python
from optimum.intel import INCQuantizer
from neural_compressor.config import PostTrainingQuantConfig

setfit_body = setfit_model.model_body[0].auto_model
quantizer = INCQuantizer.from_pretrained(setfit_body)
optimum_model_path = "/tmp/bge-small-en-v1.5_setfit-sst2-english_opt"
quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", domain="nlp")

quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_set,
    save_directory=optimum_model_path,
    batch_size=1,
)
tokenizer.save_pretrained(optimum_model_path)
```

That’s it! We now have a local copy of our quantized SetFit model. Let’s test it out.

## Step 2: Benchmark Inference

In our [notebook](https://github.com/huggingface/setfit/blob/main/notebooks/setfit-optimum-intel.ipynb), we’ve set up a `PerformanceBenchmark` class to compute model latency and throughput, as well as an accuracy measure. Let’s use it to benchmark our Optimum Intel model with two other commonly used methods:

 - Using PyTorch and 🤗 Transformers library with fp32.
 - Using [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) (IPEX) runtime with bf16 and tracing the model using TorchScript.

Load our test dataset, `sst2`, and run the benchmark using PyTorch and 🤗 Transformers library:

```python
from datasets import load_dataset
from setfit import SetFitModel
test_dataset = load_dataset("SetFit/sst2")["validation"]

model_path = "dkorat/bge-small-en-v1.5_setfit-sst2-english"
setfit_model = SetFitModel.from_pretrained(model_path)
pb = PerformanceBenchmark(
    model=setfit_model,
    dataset=test_dataset,
    optim_type="bge-small (transformers)",
)
perf_metrics = pb.run_benchmark()
```
For the second benchmark, we'll use [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) (IPEX) with bf16 precision and TorchScript tracing. 
To use IPEX we simply import the IPEX library and apply `ipex.optimize()` to the target model, which, in our case, is the SetFit (transformer) model body:

```python
dtype = torch.bfloat16
body = ipex.optimize(setfit_model.model_body, dtype=dtype)
```

For TorchScript tracing, we generate a random sequence based on the model's maximum input length, with tokens sampled from the tokenizer's vocabulary:

```python
tokenizer = setfit_model.model_body.tokenizer
d = generate_random_sequences(batch_size=1, length=tokenizer.model_max_length, vocab_size=tokenizer.vocab_size)

body = torch.jit.trace(body, (d,), check_trace=False, strict=False)
setfit_model.model_body = torch.jit.freeze(body)
```
Now let's run the benchmark using our quantized Optimum model. We’ll first need to define a wrapper around our SetFit model which plugs in our quantized model body at inference (instead of the original model body). Then, we can run the benchmark using this wrapper.

```python
from optimum.intel import IPEXModel

class OptimumSetFitModel:
    def __init__(self, setfit_model, model_body):
        model_body.tokenizer = setfit_model.model_body.tokenizer
        self.model_body = model_body
        self.model_head = setfit_model.model_head


optimum_model = IPEXModel.from_pretrained(optimum_model_path)
optimum_setfit_model = OptimumSetFitModel(setfit_model, model_body=optimum_model)

pb = PerformanceBenchmark(
    model=optimum_setfit_model,
    dataset=test_dataset,
    optim_type=f"bge-small (optimum-int8)",
    model_path=optimum_model_path,
    autocast_dtype=torch.bfloat16,
)
perf_metrics.update(pb.run_benchmark())
```

## Results
<p align="center">
    <img src="assets/178_setfit_optimum_intel/latency.png" width=500>
</p>
<p align="center">
    <em>Accuracy vs latency at batch size=1</em>
</p>


|                      | bge-small (transformers) | bge-small (ipex-bfloat16) | bge-small (optimum-int8) |
|----------------------|---------------------|---------------------------|---------------------------|
| Model Size           | 127.32 MB           | 63.74 MB                  | 44.65 MB                  |
| Accuracy on test set | 88.4%               | 88.4%                     | 88.1%                     |
| Latency (bs=1) | 15.69 +/- 0.57 ms | 5.67 +/- 0.66 ms | 4.55 +/- 0.25 ms |

When inspecting the performance at batch size 1, there’s a **3.45x reduction in latency** with our optimized model. Note that this is achieved with virtually no drop in accuracy! 
It's also worth mentioning that the model size has shrunk by **2.85x**. 

<p align="center">
    <img src="assets/178_setfit_optimum_intel/throughput.png" width=500>
</p>

We move on to our main focus, which is the reported throughputs with different batch sizes.
Here, the optimization has garnered even greater speedups. When comparing the highest achievable throughput (at any batch size), the optimized model is **7.8x faster than the original transformers fp32 model!**

## Summary

In this blog post, we have showed how to use quantization capabilities present in 🤗 Optimum Intel to optimize SetFit models. After running a quick and easy post-training quantization procedure, we've observed that accuracy level was preserved, while inference throughput increased by **7.8x**. This optimization method can be readily applied to any existing SetFit deployment running on Intel Xeon.


## References

* Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg, 2022. "Efficient Few-Shot Learning Without Prompts". https://arxiv.org/abs/2209.11055
