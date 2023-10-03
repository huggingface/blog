---
title: "Accelerating Hugging Face Transformers with AWS Inferentia2" 
thumbnail: /blog/assets/140_accelerate_transformers_with_inferentia2/thumbnail.png
authors:
- user: philschmid
- user: juliensimon
---

# Accelerating Hugging Face Transformers with AWS Inferentia2


<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

In the last five years, Transformer models [[1](https://arxiv.org/abs/1706.03762)] have become the _de facto_ standard for many machine learning (ML) tasks, such as natural language processing (NLP), computer vision (CV),  speech, and more. Today, many data scientists and ML engineers rely on popular transformer architectures like BERT [[2](https://arxiv.org/abs/1810.04805)], RoBERTa [[3](https://arxiv.org/abs/1907.11692)], the Vision Transformer [[4](https://arxiv.org/abs/2010.11929)], or any of the 130,000+ pre-trained models available on the [Hugging Face](https://huggingface.co) hub to solve complex business problems with state-of-the-art accuracy.

However, for all their greatness, Transformers can be challenging to deploy in production. On top of the infrastructure plumbing typically associated with model deployment, which we largely solved with our [Inference Endpoints](https://huggingface.co/inference-endpoints) service, Transformers are large models which routinely exceed the multi-gigabyte mark. Large language models (LLMs) like [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B), [Flan-T5](https://huggingface.co/google/flan-t5-xxl), or [Opt-30B](https://huggingface.co/facebook/opt-30b) are in the tens of gigabytes, not to mention behemoths like [BLOOM](https://huggingface.co/bigscience/bloom), our very own LLM, which clocks in at 350 gigabytes. 

Fitting these models on a single accelerator can be quite difficult, let alone getting the high throughput and low inference latency that applications require, like conversational applications and search. So far, ML experts have designed complex manual techniques to slice large models, distribute them on a cluster of accelerators, and optimize their latency. Unfortunately, this work is extremely difficult, time-consuming, and completely out of reach for many ML practitioners. 

At Hugging Face, we're democratizing ML and always looking to partner with companies who also believe that every developer and organization should benefit from state-of-the-art models. For this purpose, we're excited to partner with Amazon Web Services to optimize Hugging Face Transformers for AWS [Inferentia 2](https://aws.amazon.com/machine-learning/inferentia/)! It’s a new purpose-built inference accelerator that delivers unprecedented levels of throughput, latency, performance per watt, and scalability.

## Introducing AWS Inferentia2

AWS Inferentia2 is the next generation to Inferentia1 launched in 2019. Powered by Inferentia1, Amazon EC2 Inf1 instances delivered 25% higher throughput and 70% lower cost than comparable G5 instances based on NVIDIA A10G GPU, and with Inferentia2, AWS is pushing the envelope again. 

The new Inferentia2 chip delivers a 4x throughput increase and a 10x latency reduction compared to Inferentia. Likewise, the new [Amazon EC2 Inf2](https://aws.amazon.com/de/ec2/instance-types/inf2/) instances have up to 2.6x better throughput, 8.1x lower latency, and 50% better performance per watt than comparable G5 instances. Inferentia 2 gives you the best of both worlds: cost-per-inference optimization thanks to high throughput and response time for your application thanks to low inference latency.

Inf2 instances are available in multiple sizes, which are equipped with between 1 to 12 Inferentia 2 chips. When several chips are present, they are interconnected by a blazing-fast direct Inferentia2 to Inferentia2 connectivity for distributed inference on large models. For example, the largest instance size, inf2.48xlarge, has 12 chips and enough memory to load a 175-billion parameter model like GPT-3 or BLOOM.

Thankfully none of this comes at the expense of development complexity. With [optimum neuron](https://github.com/huggingface/optimum-neuron), you don't need to slice or modify your model. Because of the native integration in [AWS Neuron SDK](https://github.com/aws-neuron/aws-neuron-sdk), all it takes is a single line of code to compile your model for Inferentia 2. You can experiment in minutes! Test the performance your model could reach on Inferentia 2 and see for yourself.

Speaking of, let’s show you how several Hugging Face models run on Inferentia 2. Benchmarking time!

## Benchmarking Hugging Face Models on AWS Inferentia 2

We evaluated some of the most popular NLP models from the [Hugging Face Hub](https://huggingface.co/models) including BERT, RoBERTa, DistilBERT, and vision models like Vision Transformers.  

The first benchmark compares the performance of Inferentia, Inferentia 2, and GPUs. We ran all experiments on AWS with the following instance types: 
* Inferentia1 - [inf1.2xlarge](https://aws.amazon.com/ec2/instance-types/inf1/?nc1=h_ls) powered by a single Inferentia chip.
* Inferentia2 - [inf2.xlarge](https://aws.amazon.com/ec2/instance-types/inf2/?nc1=h_ls) powered by a single Inferentia2 chip.
* GPU - [g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/) powered by a single NVIDIA A10G GPU.

_Note: that we did not optimize the model for the GPU environment, the models were evaluated in fp32._

When it comes to benchmarking Transformer models, there are two metrics that are most adopted:
* **Latency**: the time it takes for the model to perform a single prediction (pre-process, prediction, post-process).
* **Throughput**: the number of executions performed in a fixed amount of time for one benchmark configuration

We looked at latency across different setups and models to understand the benefits and tradeoffs of the new Inferentia2 instance. If you want to run the benchmark yourself, we created a [Github repository](https://github.com/philschmid/aws-neuron-samples/tree/main/benchmark) with all the information and scripts to do so. 

### Results

The benchmark confirms that the performance improvements claimed by AWS can be reproduced and validated by real use-cases and examples. On average, AWS Inferentia2 delivers 4.5x better latency than NVIDIA A10G GPUs and 4x better latency than Inferentia1 instances. 

We ran 144 experiments on 6 different model architectures:
* Accelerators: Inf1, Inf2, NVIDIA A10G
* Models: [BERT-base](https://huggingface.co/bert-base-uncased), [BERT-Large](https://huggingface.co/bert-large-uncased), [RoBERTa-base](https://huggingface.co/roberta-base), [DistilBERT](https://huggingface.co/distilbert-base-uncased), [ALBERT-base](https://huggingface.co/albert-base-v2), [ViT-base](https://huggingface.co/google/vit-base-patch16-224)
* Sequence length: 8, 16, 32, 64, 128, 256, 512
* Batch size: 1

In each experiment, we collected numbers for p95 latency. You can find the full details of the benchmark in this spreadsheet: [HuggingFace: Benchmark Inferentia2](https://docs.google.com/spreadsheets/d/1AULEHBu5Gw6ABN8Ls6aSB2CeZyTIP_y5K7gC7M3MXqs/edit?usp=sharing).

Let’s highlight a few insights of the benchmark.

### BERT-base

Here is the latency comparison for running [BERT-base](https://huggingface.co/bert-base-uncased) on each of the infrastructure setups, with a logarithmic scale for latency. It is remarkable to see how Inferentia2 outperforms all other setups by ~6x for sequence lengths up to 256. 

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="BERT-base p95 latency" src="assets/140_accelerate_transformers_with_inferentia2/bert.png"></medium-zoom>
  <figcaption>Figure 1. BERT-base p95 latency</figcaption>
</figure>
<br>



### Vision Transformer

Here is the latency comparison for running [ViT-base](https://huggingface.co/google/vit-base-patch16-224) on the different infrastructure setups. Inferentia2 delivers 2x better latency than the NVIDIA A10G, with the potential to greatly help companies move from traditional architectures, like CNNs, to Transformers for - real-time applications.

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="ViT p95 latency" src="assets/140_accelerate_transformers_with_inferentia2/vit.png"></medium-zoom>
  <figcaption>Figure 1. ViT p95 latency</figcaption>
</figure>
<br>


## Conclusion

Transformer models have emerged as the go-to solution for many machine learning tasks. However, deploying them in production has been challenging due to their large size and latency requirements. Thanks to AWS Inferentia2 and the collaboration between Hugging Face and AWS, developers and organizations can now leverage the benefits of state-of-the-art models without the prior need for extensive machine learning expertise. You can start testing for as low as 0.76$/h.

The initial benchmarking results are promising, and show that Inferentia2 delivers superior latency performance when compared to both Inferentia and NVIDIA A10G GPUs. This latest breakthrough promises high-quality machine learning models can be made available to a much broader audience delivering AI accessibility to everyone. 