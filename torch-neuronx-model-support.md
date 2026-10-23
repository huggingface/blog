---
title: "Does Your Model Run on AWS Trainium? Look It Up"
thumbnail: /blog/assets/torch-neuronx-model-support/thumbnail.jpg
authors:
- user: dacorvo
---

# Does Your Model Run on AWS Trainium? Look It Up

In a [previous post](https://huggingface.co/blog/hf-torchneuron), we introduced TorchNeuron, the native PyTorch backend for AWS Trainium (the `torch-neuronx` package). With it, Transformers runs its own modeling code on Trainium, as it does on any other device, so any Transformers model on the Hub may run. Which ones actually do?

The [torch-neuronx model support](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support) Space answers that question for the most popular models on the Hub. It tells you whether a model runs unmodified on Trainium, and gives you the script to run it yourself.

[![torch-neuronx model support](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support/resolve/main/thumbnail.jpg)](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support)

## Look up a model

The Space opens on the **Selected** view: our own selection of models picked among the top models on the Hub by downloads and by trending score. Each row gives the result, the number of Trainium cores the model is sharded over, the wall time of the run (compilation included) and the peak device memory.

To look up a given model, switch to the **Trending** or **Downloads** view and type part of its name. These views follow the Hub's own rankings, one task at a time, whether a model was tested or not. When a model was left out on purpose, the Space tells you why (most of the time it was not prioritized, sometimes it is truly incompatible). You can also pick the versions of the Neuron compiler (`neuronx-cc`) and of Transformers to look at.

> [!NOTE]
> At the time of writing, only Transformers models are tested, and our selection is limited to `text-generation` and `image-text-to-text` models. We plan to expand to diffusers, sentence transformers, and other transformers tasks.

## Run it yourself

Select a model to see how to run it. The Space lists the Neuron stack the model was tested on, the packages to install on top of it, and the `run.py` script that ran. The script follows the recipe of the [TorchNeuron announcement](https://huggingface.co/blog/hf-torchneuron): it loads the model with `from_pretrained`, sharded when it needs to be, then generates with a static cache and `torch.compile(backend="neuron")`.

On a Trainium2 (`trn2`) instance with that stack, for example in the [AWS Neuron Deep Learning Container](https://github.com/aws-neuron/deep-learning-containers#torch-neuronx), install the packages, then run:

```bash
python3 run.py
```

No need for `torchrun`: the script launches its own `torch.distributed` ranks when the model is sharded.

A first run starts by compiling the model's graphs. To skip that step, copy the graphs the same compiler built during the test, from the public [NEFF cache](https://huggingface.co/buckets/aws-neuron/torch-neuronx-neff-cache) bucket. The Space gives you the command, with the right compiler version:

```bash
hf buckets sync hf://buckets/aws-neuron/torch-neuronx-neff-cache/neuronxcc-<version> /tmp/neff_cache
```

## What "runs" means

Each test loads the model on a `trn2` instance, sharded over the smallest number of Trainium cores that can hold its weights. Most models are sharded with tensor parallelism. Mixture-of-experts models without a tensor-parallel plan use expert parallelism only. The test then generates 32 tokens and checks them. On a single core, the tokens are compared with those of a CPU run of the same checkpoint. When the model is sharded, a small judge model checks that the text is fluent English.

At the time of writing, only models that fit on 4 Trainium2 devices are tested, which means 16 cores at most. A `PASS` says nothing about throughput, or about longer generations.

## Where things stand

At the time of writing, with `neuronx-cc` 2.28 and Transformers 5.20.2, 91 of the 100 selected models pass. The main gap is the mixture-of-experts models that run with expert parallelism, which we are actively working on.

Each sweep tests the Neuron SDK and the Transformers version of the day, and the results of previous sweeps stay available, so you can also compare how a model fares from one version to the next.

## The results are a dataset

Everything the Space shows comes from the public [`aws-neuron/torch-neuronx-model-support`](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support) dataset, with one Parquet file per pair of `neuronx-cc` and library versions.

Each row holds a model's result, the error when it failed, the versions of the whole stack and the `run.py` that ran. You can query it directly:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/aws-neuron/torch-neuronx-model-support/data")
df[df.model_id == "Qwen/Qwen3-0.6B"][["neuronx_cc", "library", "library_version", "neuron_cores", "result"]]
```
The [dataset card](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support) gives the exact meaning of each result.

Next time you wonder whether a model runs on Trainium, [look it up](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support).
