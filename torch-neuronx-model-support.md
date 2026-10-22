---
title: "Does Your Model Run on AWS Trainium? Look It Up"
thumbnail: /blog/assets/torch-neuronx-model-support/thumbnail.png
authors:
- user: dacorvo
---

# Does Your Model Run on AWS Trainium? Look It Up

With Optimum Neuron, a model ran on AWS Trainium when its architecture had a Neuron implementation, so checking
support meant checking a list of architectures. With [TorchNeuron](https://huggingface.co/blog/hf-torchneuron), the
native PyTorch backend for Trainium (the `torch-neuronx` package), there is no such list. Transformers runs its own
modeling code on Trainium, as it does on any other device, so any Transformers model on the Hub may run. Which ones
actually do?

The [torch-neuronx model support](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support) Space answers
that question for the most popular language and vision-language models on the Hub. It tells you whether a model runs
unmodified on Trainium, and gives you the script to run it yourself.

<iframe src="https://aws-neuron-torch-neuronx-model-support.static.hf.space" frameborder="0" width="100%" height="800"></iframe>

## Look up a model

The Space opens on the **Selected** view: the models of the latest test sweep, picked among the top `text-generation`
and `image-text-to-text` models on the Hub by downloads and by trending score. Each row gives the result, the
tensor-parallel degree, the wall time of the run (compilation included) and the peak device memory. Only Transformers
models are tested so far.

The **Trending** and **Downloads** views follow the Hub's own rankings, whether a model was tested or not. When a model
was left out on purpose, the Space tells you why: its weights do not fit on 4 Trainium2 devices, it is a quantized
checkpoint, it needs `trust_remote_code`, and so on. In these two views, you can also pick the versions of the Neuron compiler
(`neuronx-cc`) and of Transformers to look at.

## Run it yourself

Select a model to see how to run it. The Space lists the Neuron stack the model was tested on, the packages to install
on top of it, and the `run.py` script that ran. The script follows the recipe of the
[TorchNeuron announcement](https://huggingface.co/blog/hf-torchneuron): it loads the model with `from_pretrained`,
sharded over several NeuronCores when it needs to be, then generates with a static cache and
`torch.compile(backend="neuron")`.

On a Trainium2 (`trn2`) instance with that stack, for example in the
[AWS Neuron Deep Learning Container](https://github.com/aws-neuron/deep-learning-containers), install the packages,
then run:

```bash
python3 run.py
```

No need for `torchrun`: the script launches its own `torch.distributed` ranks when the model is sharded.

A first run starts by compiling the model's graphs. To skip that step, copy the graphs the same compiler built during
the test, from the public [NEFF cache](https://huggingface.co/buckets/aws-neuron/torch-neuronx-neff-cache) bucket. The
Space gives you the command, with the right compiler version:

```bash
hf buckets sync hf://buckets/aws-neuron/torch-neuronx-neff-cache/neuronxcc-<version> /tmp/neff_cache
```

## What "runs" means

Each test loads the model on a `trn2` instance with tensor parallelism, at the smallest degree that fits its bf16
weights. Mixture-of-experts models without a tensor-parallel plan use expert parallelism instead. The test then
generates 32 tokens and checks them. At TP=1, the tokens are compared with those of a CPU run of the same checkpoint.
At higher degrees, a small judge model checks that the text is fluent English.

For now, only models that fit on 4 Trainium2 devices are tested, which means a degree of 16 at most. A `PASS` says
nothing about throughput, or about longer generations. The
[dataset card](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support) gives the exact meaning of each
result.

## Where things stand

At the time of writing, with `neuronx-cc` 2.28 and Transformers 5.20.2, 91 of the 100 selected models pass. The models
that run with tensor parallelism almost all do (90 of 94), mixture-of-experts models included. The main gap is the
mixture-of-experts models without a tensor-parallel plan, which run with expert parallelism: 5 of the 6 fail at
compilation.

Each sweep tests the Neuron SDK and the Transformers version of the day, and the results of previous sweeps stay
available, so you can also compare how a model fares from one version to the next.

## The results are a dataset

Everything the Space shows comes from the public
[`aws-neuron/torch-neuronx-model-support`](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support)
dataset, with one Parquet file per pair of `neuronx-cc` and library versions. Each row holds a model's result, the
error when it failed, the versions of the whole stack and the `run.py` that ran. You can query it directly:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/aws-neuron/torch-neuronx-model-support/data")
df[df.model_id == "Qwen/Qwen3-0.6B"][["neuronx_cc", "library", "library_version", "tp", "result"]]
```

Next time you wonder whether a model runs on Trainium,
[look it up](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support).
