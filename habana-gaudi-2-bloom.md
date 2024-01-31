---
title: "Fast Inference on Large Language Models: BLOOMZ on Habana Gaudi2 Accelerator"
thumbnail: /blog/assets/habana-gaudi-2-bloom/thumbnail.png
authors:
- user: regisss
---

# Fast Inference on Large Language Models: BLOOMZ on Habana Gaudi2 Accelerator


This article will show you how to easily deploy large language models with hundreds of billions of parameters like BLOOM on [Habana® Gaudi®2](https://habana.ai/training/gaudi2/) using 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index), which is the bridge between Gaudi2 and the 🤗 Transformers library. As demonstrated in the benchmark presented in this post, this will enable you to **run inference faster than with any GPU currently available on the market**.

As models get bigger and bigger, deploying them into production to run inference has become increasingly challenging. Both hardware and software have seen a lot of innovations to address these challenges, so let's dive in to see how to efficiently overcome them!


## BLOOMZ

[BLOOM](https://arxiv.org/abs/2211.05100) is a 176-billion-parameter autoregressive model that was trained to complete sequences of text. It can handle 46 different languages and 13 programming languages. Designed and trained as part of the [BigScience](https://bigscience.huggingface.co/) initiative, BLOOM is an open-science project that involved a large number of researchers and engineers all over the world. More recently, another model with the exact same architecture was released: [BLOOMZ](https://arxiv.org/abs/2211.01786), which is a fine-tuned version of BLOOM on several tasks leading to better generalization and zero-shot[^1] capabilities.

Such large models raise new challenges in terms of memory and speed for both [training](https://huggingface.co/blog/bloom-megatron-deepspeed) and [inference](https://huggingface.co/blog/bloom-inference-optimization). Even in 16-bit precision, one instance requires 352 GB to fit! You will probably struggle to find any device with so much memory at the moment, but state-of-the-art hardware like Habana Gaudi2 does make it possible to perform inference on BLOOM and BLOOMZ models with low latencies.


## Habana Gaudi2

[Gaudi2](https://habana.ai/training/gaudi2/) is the second-generation AI hardware accelerator designed by Habana Labs. A single server contains 8 accelerator devices (called Habana Processing Units, or HPUs) with 96GB of memory each, which provides room to make very large models fit in. However, hosting the model is not very interesting if the computation is slow. Fortunately, Gaudi2 shines on that aspect: it differs from GPUs in that its architecture enables the accelerator to perform General Matrix Multiplication (GeMM) and other operations in parallel, which speeds up deep learning workflows. These features make Gaudi2 a great candidate for LLM training and inference.

Habana's SDK, SynapseAI™, supports PyTorch and DeepSpeed for accelerating LLM training and inference. The [SynapseAI graph compiler](https://docs.habana.ai/en/latest/Gaudi_Overview/SynapseAI_Software_Suite.html#graph-compiler-and-runtime) will optimize the execution of the operations accumulated in the graph (e.g. operator fusion, data layout management, parallelization, pipelining and memory management, and graph-level optimizations).

Moreover, support for [HPU graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) and [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) have just recently been introduced in SynapseAI, and these are well-suited for latency-sensitive applications as shown in our benchmark below.

All these features are integrated into the 🤗 [Optimum Habana](https://github.com/huggingface/optimum-habana) library so that deploying your model on Gaudi is very simple. Check out the quick-start page [here](https://huggingface.co/docs/optimum/habana/quickstart).

If you would like to get access to Gaudi2, go to the [Intel Developer Cloud](https://www.intel.com/content/www/us/en/secure/developer/devcloud/cloud-launchpad.html) and follow [this guide](https://huggingface.co/blog/habana-gaudi-2-benchmark#how-to-get-access-to-gaudi2).


## Benchmarks

In this section, we are going to provide an early benchmark of BLOOMZ on Gaudi2, first-generation Gaudi and Nvidia A100 80GB. Although these devices have quite a lot of memory, the model is so large that a single device is not enough to contain a single instance of BLOOMZ. To solve this issue, we are going to use [DeepSpeed](https://www.deepspeed.ai/), which is a deep learning optimization library that enables many memory and speed improvements to accelerate the model and make it fit the device. In particular, we rely here on [DeepSpeed-inference](https://arxiv.org/abs/2207.00032): it introduces several features such as [model (or pipeline) parallelism](https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism) to make the most of the available devices. For Gaudi2, we use [Habana's DeepSpeed fork](https://github.com/HabanaAI/deepspeed) that adds support for HPUs.


### Latency

We measured latencies (batch of one sample) for two different sizes of BLOOMZ, both with multi-billion parameters:
- [176 billion](https://huggingface.co/bigscience/bloomz) parameters
- [7 billion](https://huggingface.co/bigscience/bloomz-7b1) parameters

Runs were performed with DeepSpeed-inference in 16-bit precision with 8 devices and using a [key-value cache](https://huggingface.co/docs/transformers/v4.27.1/en/model_doc/bloom#transformers.BloomForCausalLM.forward.use_cache). Note that while [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) are not currently compatible with model parallelism in DeepSpeed (DeepSpeed v0.8.2, see [here](https://github.com/microsoft/DeepSpeed/blob/v0.8.2/deepspeed/inference/engine.py#L158)), HPU graphs are supported in Habana's DeepSpeed fork. All benchmarks are doing [greedy generation](https://huggingface.co/blog/how-to-generate#greedy-search) of 100 token outputs. The input prompt is:

> "DeepSpeed is a machine learning framework"

which consists of 7 tokens with BLOOM's tokenizer.

The results for inference latency are displayed in the table below (the unit is *seconds*).

| Model       | Number of devices | Gaudi2 latency (seconds) | A100-80GB latency (seconds) | First-gen Gaudi latency (seconds) |
|:-----------:|:-----------------:|:-------------------------:|:-----------------:|:----------------------------------:|
| BLOOMZ | 8 | 3.103 | 4.402 | / |
| BLOOMZ-7B | 8 | 0.734 | 2.417 | 3.321 |
| BLOOMZ-7B | 1 | 0.772 | 2.119 | 2.387 |

*Update: the numbers above were updated with the releases of Optimum Habana 1.6 and SynapseAI 1.10, leading to a* x*1.42 speedup on BLOOMZ with Gaudi2 compared to A100.*

The Habana team recently introduced support for DeepSpeed-inference in SynapseAI 1.8, and thereby quickly enabled inference for 100+ billion parameter models. **For the 176-billion-parameter checkpoint, Gaudi2 is 1.42x faster than A100 80GB**. Smaller checkpoints present interesting results too. **Gaudi2 is 2.89x faster than A100 for BLOOMZ-7B!** It is also interesting to note that it manages to benefit from model parallelism whereas A100 is faster on a single device.

We also ran these models on first-gen Gaudi. While it is slower than Gaudi2, it is interesting from a price perspective as a DL1 instance on AWS costs approximately 13\$ per hour. Latency for BLOOMZ-7B on first-gen Gaudi is 2.387 seconds. Thus, **first-gen Gaudi offers for the 7-billion checkpoint a better price-performance ratio than A100** which costs more than 30\$ per hour!

We expect the Habana team will optimize the performance of these models in the upcoming SynapseAI releases. For example, in our last benchmark, we saw that [Gaudi2 performs Stable Diffusion inference 2.2x faster than A100](https://huggingface.co/blog/habana-gaudi-2-benchmark#generating-images-from-text-with-stable-diffusion) and this has since been improved further to 2.37x with the latest optimizations provided by Habana. We will update these numbers as new versions of SynapseAI are released and integrated within Optimum Habana.


### Running inference on a complete dataset

The script we wrote enables using your model to complete sentences over a whole dataset. This is useful to try BLOOMZ inference on Gaudi2 on your own data.

Here is an example with the [*tldr_news*](https://huggingface.co/datasets/JulesBelveze/tldr_news/viewer/all/test) dataset. It contains both the headline and content of several articles (you can visualize it on the Hugging Face Hub). We kept only the *content* column and truncated each sample to the first 16 tokens so that the model generates the rest of the sequence with 50 new tokens. The first five samples look like:

```
Batch n°1
Input: ['Facebook has released a report that shows what content was most widely viewed by Americans between']
Output: ['Facebook has released a report that shows what content was most widely viewed by Americans between January and June of this year. The report, which is based on data from the company’s mobile advertising platform, shows that the most popular content on Facebook was news, followed by sports, entertainment, and politics. The report also shows that the most']
--------------------------------------------------------------------------------------------------
Batch n°2
Input: ['A quantum effect called superabsorption allows a collection of molecules to absorb light more']
Output: ['A quantum effect called superabsorption allows a collection of molecules to absorb light more strongly than the sum of the individual absorptions of the molecules. This effect is due to the coherent interaction of the molecules with the electromagnetic field. The superabsorption effect has been observed in a number of systems, including liquid crystals, liquid crystals in']
--------------------------------------------------------------------------------------------------
Batch n°3
Input: ['A SpaceX Starship rocket prototype has exploded during a pressure test. It was']
Output: ['A SpaceX Starship rocket prototype has exploded during a pressure test. It was the first time a Starship prototype had been tested in the air. The explosion occurred at the SpaceX facility in Boca Chica, Texas. The Starship prototype was being tested for its ability to withstand the pressure of flight. The explosion occurred at']
--------------------------------------------------------------------------------------------------
Batch n°4
Input: ['Scalene is a high-performance CPU and memory profiler for Python.']
Output: ['Scalene is a high-performance CPU and memory profiler for Python. It is designed to be a lightweight, portable, and easy-to-use profiler. Scalene is a Python package that can be installed on any platform that supports Python. Scalene is a lightweight, portable, and easy-to-use profiler']
--------------------------------------------------------------------------------------------------
Batch n°5
Input: ['With the rise of cheap small "Cube Satellites", startups are now']
Output: ['With the rise of cheap small "Cube Satellites", startups are now able to launch their own satellites for a fraction of the cost of a traditional launch. This has led to a proliferation of small satellites, which are now being used for a wide range of applications. The most common use of small satellites is for communications,']
```

In the next section, we explain how to use the script we wrote to perform this benchmark or to apply it on any dataset you like from the Hugging Face Hub!


### How to reproduce these results?

The script used for benchmarking BLOOMZ on Gaudi2 and first-gen Gaudi is available [here](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation). Before running it, please make sure that the latest versions of SynapseAI and the Gaudi drivers are installed following [the instructions given by Habana](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

Then, run the following:
```bash
git clone https://github.com/huggingface/optimum-habana.git
cd optimum-habana && pip install . && cd examples/text-generation
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.9.0
```

Finally, you can launch the script as follows:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py --model_name_or_path bigscience/bloomz --use_hpu_graphs --use_kv_cache --max_new_tokens 100
```

For multi-node inference, you can follow [this guide](https://huggingface.co/docs/optimum/habana/usage_guides/multi_node_training) from the documentation of Optimum Habana.

You can also load any dataset from the Hugging Face Hub to get prompts that will be used for generation using the argument `--dataset_name my_dataset_name`.

This benchmark was performed with Transformers v4.28.1, SynapseAI v1.9.0 and Optimum Habana v1.5.0.

For GPUs, [here](https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-ds-inference.py) is the script that led to the results that were previously presented in [this blog post](https://huggingface.co/blog/bloom-inference-pytorch-scripts) (and [here](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts#deepspeed-inference) are the instructions to use it). To use CUDA graphs, static shapes are necessary and this is not supported in 🤗 Transformers. You can use [this repo](https://github.com/HabanaAI/Model-References/tree/1.8.0/PyTorch/nlp/bloom) written by the Habana team to enable them.


## Conclusion

We see in this article that **Habana Gaudi2 performs BLOOMZ inference faster than Nvidia A100 80GB**. And there is no need to write a complicated script as 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/index) provides easy-to-use tools to run inference with multi-billion-parameter models on HPUs. Future releases of Habana's SynapseAI SDK are expected to speed up performance, so we will update this benchmark regularly as LLM inference optimizations on SynapseAI continue to advance. We are also looking forward to the performance benefits that will come with FP8 inference on Gaudi2.

We also presented the results achieved with first-generation Gaudi. For smaller models, it can perform on par with or even better than A100 for almost a third of its price. It is a good alternative option to using GPUs for running inference with such a big model like BLOOMZ.

If you are interested in accelerating your Machine Learning training and inference workflows using the latest AI hardware accelerators and software libraries, check out our [Expert Acceleration Program](https://huggingface.co/support). To learn more about Habana solutions, [read about our partnership and contact them here](https://huggingface.co/hardware/habana). To learn more about Hugging Face efforts to make AI hardware accelerators easy to use, check out our [Hardware Partner Program](https://huggingface.co/hardware).


### Related Topics

- [Faster Training and Inference: Habana Gaudi-2 vs Nvidia A100 80GB](https://huggingface.co/blog/habana-gaudi-2-benchmark)
- [Leverage DeepSpeed to Train Faster and Cheaper Large Scale Transformer Models with Hugging Face and Habana Labs Gaudi](https://developer.habana.ai/events/leverage-deepspeed-to-train-faster-and-cheaper-large-scale-transformer-models-with-hugging-face-and-habana-labs-gaudi/)

---

Thanks for reading! If you have any questions, feel free to contact me, either through [Github](https://github.com/huggingface/optimum-habana) or on the [forum](https://discuss.huggingface.co/c/optimum/59). You can also connect with me on [LinkedIn](https://www.linkedin.com/in/regispierrard/).

[^1]: “Zero-shot” refers to the ability of a model to complete a task on new or unseen input data, i.e. without having been provided any training examples of this kind of data. We provide the model with a prompt and a sequence of text that describes what we want our model to do, in natural language. Zero-shot classification excludes any examples of the desired task being completed. This differs from single or few-shot classification, as these tasks include a single or a few examples of the selected task.
