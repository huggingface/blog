---
title: "Case Study: Millisecond Latency using Hugging Face Infinity and modern CPUs"
thumbnail: /blog/assets/46_infinity_cpu_performance/thumbnail.png
---

<div style="background-color: #90EE90; padding: 10px; order: 2px solid green; margin-left: auto; margin-right: auto; width: 70em; text-align: left;">
  Infinity is no longer offered by Hugging Face as a commercial inference solution. To deploy and accelerate your models, we recommend the following new solutions:

  * [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) to easily deploy models on dedicated infrastructure managed by Hugging Face

  * Our open source acceleration libraries [🤗 Optimum Intel](https://huggingface.co/blog/openvino) and [🤗 Optimum ONNX Runtime](https://huggingface.co/docs/optimum/main/en/onnxruntime/overview) to accelerate your models on CPU.

  * Hugging Face [Expert Acceleration Program](https://huggingface.co/support), a commercial service for Hugging Face experts to work directly with your team to accelerate your Machine Learning roadmap and models. 
</div>


<h1>Case Study: Millisecond Latency using Hugging Face Infinity and modern CPUs</h1>

<div class="blog-metadata">
    <small>Published Jan 13, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/infinity-cpu-performance.md">
        Update on GitHub
    </a>
</div>



<div class="author-card">
    <a href="/philschmid">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1624629516652-5ff5d596f244529b3ec0fb89.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>philschmid</code>
            <span class="fullname">Philipp Schmid</span>
        </div>
    </a>
    <a href="/jeffboudier">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1605114051380-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>jeffboudier</code>
            <span class="fullname">Jeff Boudier</span>
        </div>
    </a>
    <a href="/mfuntowicz">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1583858935715-5e67c47c100906368940747e.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>mfuntowicz</code>
            <span class="fullname">Morgan Funtowicz</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

## Introduction 

Transfer learning has changed Machine Learning by reaching new levels of accuracy from Natural Language Processing (NLP) to Audio and Computer Vision tasks. At Hugging Face, we work hard to make these new complex models and large checkpoints as easily accessible and usable as possible. But while researchers and data scientists have converted to the new world of Transformers, few companies have been able to deploy these large, complex models in production at scale.

The main bottleneck is the latency of predictions which can make large deployments expensive to run and real-time use cases impractical. Solving this is a difficult engineering challenge for any Machine Learning Engineering team and requires the use of advanced techniques to optimize models all the way down to the hardware.

With [Hugging Face Infinity](https://huggingface.co/infinity), we offer a containerized solution that makes it easy to deploy low-latency, high-throughput, hardware-accelerated inference pipelines for the most popular Transformer models. Companies can get both the accuracy of Transformers and the efficiency necessary for large volume deployments, all in a simple to use package. In this blog post, we want to share detailed performance results for Infinity running on the latest generation of Intel Xeon CPU, to achieve optimal cost, efficiency, and latency for your Transformer deployments.


## What is Hugging Face Infinity

Hugging Face Infinity is a containerized solution for customers to deploy end-to-end optimized inference pipelines for State-of-the-Art Transformer models, on any infrastructure.

Hugging Face Infinity consists of 2 main services:
* The Infinity Container is a hardware-optimized inference solution delivered as a Docker container.
* Infinity Multiverse is a Model Optimization Service through which a Hugging Face Transformer model is optimized for the Target Hardware. Infinity Multiverse is compatible with Infinity Container.

The Infinity Container is built specifically to run on a Target Hardware architecture and exposes an HTTP /predict endpoint to run inference.

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Product overview" src="assets/46_infinity_cpu_performance/overview.png"></medium-zoom>
  <figcaption>Figure 1. Infinity Overview</figcaption>
</figure>
<br>

An Infinity Container is designed to serve 1 Model and 1 Task. A Task corresponds to machine learning tasks as defined in the [Transformers Pipelines documentation](https://huggingface.co/docs/transformers/master/en/main_classes/pipelines). As of the writing of this blog post, supported tasks include feature extraction/document embedding, ranking, sequence classification, and token classification.

You can find more information about Hugging Face Infinity at [hf.co/infinity](https://huggingface.co/infinity), and if you are interested in testing it for yourself, you can sign up for a free trial at [hf.co/infinity-trial](https://huggingface.co/infinity-trial).

---

## Benchmark 

Inference performance benchmarks often only measure the execution of the model. In this blog post, and when discussing the performance of Infinity, we always measure the end-to-end pipeline including pre-processing, prediction, post-processing. Please keep this in mind when comparing these results with other latency measurements. 

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Pipeline" src="assets/46_infinity_cpu_performance/pipeline.png"></medium-zoom>
  <figcaption>Figure 2. Infinity End-to-End Pipeline</figcaption>
</figure>
<br>

### Environment

As a benchmark environment, we are going to use the [Amazon EC2 C6i instances](https://aws.amazon.com/ec2/instance-types/c6i), which are compute-optimized instances powered by the 3rd generation of Intel Xeon Scalable processors. These new Intel-based instances are using the ice-lake Process Technology and support Intel AVX-512, Intel Turbo Boost, and Intel Deep Learning Boost.

In addition to superior performance for machine learning workloads, the Intel Ice Lake C6i instances offer great cost-performance and are our recommendation to deploy Infinity on Amazon Web Services.  To learn more, visit the [EC2 C6i instance](https://aws.amazon.com/ec2/instance-types/c6i) page. 


### Methodologies

When it comes to benchmarking BERT-like models, two metrics are most adopted:
* **Latency**: Time it takes for a single prediction of the model (pre-process, prediction, post-process)
* **Throughput**: Number of executions performed in a fixed amount of time for one benchmark configuration, respecting Physical CPU cores, Sequence Length, and Batch Size

These two metrics will be used to benchmark Hugging Face Infinity across different setups to understand the benefits and tradeoffs in this blog post.

---

## Results

To run the benchmark, we created an infinity container for the [EC2 C6i instance](https://aws.amazon.com/ec2/instance-types/c6i) (Ice-lake) and optimized a [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) model for sequence classification using Infinity Multiverse. 

This ice-lake optimized Infinity Container can achieve up to 34% better latency & throughput compared to existing cascade-lake-based instances, and up to 800% better latency & throughput compared to vanilla transformers running on ice-lake.

The Benchmark we created consists of 192 different experiments and configurations. We ran experiments for: 
* Physical CPU cores: 1, 2, 4, 8
* Sequence length: 8, 16, 32, 64, 128, 256, 384, 512
* Batch_size: 1, 2, 4, 8, 16, 32

In each experiment, we collect numbers for:
* Throughput (requests per second)
* Latency (min, max, avg, p90, p95, p99)

You can find the full data of the benchmark in this google spreadsheet: [🤗 Infinity: CPU Ice-Lake Benchmark](https://docs.google.com/spreadsheets/d/1GWFb7L967vZtAS1yHhyTOZK1y-ZhdWUFqovv7-73Plg/edit?usp=sharing).

In this blog post, we will highlight a few results of the benchmark including the best latency and throughput configurations.

In addition to this, we deployed the [DistilBERT](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) model we used for the benchmark as an API endpoint on 2 physical cores. You can test it and get a feeling for the performance of Infinity. Below you will find a `curl` command on how to send a request to the hosted endpoint. The API returns a `x-compute-time` HTTP Header, which contains the duration of the end-to-end pipeline.

```bash
curl --request POST `-i` \
  --url https://infinity.huggingface.co/cpu/distilbert-base-uncased-emotion \
  --header 'Content-Type: application/json' \
  --data '{"inputs":"I like you. I love you"}'
```

### Throughput

Below you can find the throughput comparison for running infinity on 2 physical cores with batch size 1, compared with vanilla transformers.

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Throughput" src="assets/46_infinity_cpu_performance/throughput.png"></medium-zoom>
  <figcaption>Figure 3. Throughput: Infinity vs Transformers</figcaption>
</figure>
<br>


| Sequence Length | Infinity    | Transformers | improvement |
|-----------------|-------------|--------------|-------------|
| 8               | 248 req/sec | 49 req/sec   | +506%       |
| 16              | 212 req/sec | 50 req/sec   | +424%       |
| 32              | 150 req/sec | 40 req/sec   | +375%       |
| 64              | 97 req/sec  | 28 req/sec   | +346%       |
| 128             | 55 req/sec  | 18 req/sec   | +305%       |
| 256             | 27 req/sec  | 9 req/sec    | +300%       |
| 384             | 17 req/sec  | 5 req/sec    | +340%       |
| 512             | 12 req/sec  | 4 req/sec    | +300%       |


### Latency 

Below, you can find the latency results for an experiment running Hugging Face Infinity on 2 Physical Cores with Batch Size 1. It is remarkable to see how robust and constant Infinity is, with minimal deviation for p95, p99, or p100 (max latency). This result is confirmed for other experiments as well in the benchmark.  

<br>
<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Latency" src="assets/46_infinity_cpu_performance/latency.png"></medium-zoom>
  <figcaption>Figure 4. Latency (Batch=1, Physical Cores=2)</figcaption>
</figure>
<br>

---

## Conclusion

In this post, we showed how Hugging Face Infinity performs on the new Intel Ice Lake Xeon CPU. We created a detailed benchmark with over 190 different configurations sharing the results you can expect when using Hugging Face Infinity on CPU, what would be the best configuration to optimize your Infinity Container for latency, and what would be the best configuration to maximize throughput.

Hugging Face Infinity can deliver up to 800% higher throughput compared to vanilla transformers, and down to 1-4ms latency for sequence lengths up to 64 tokens.

The flexibility to optimize transformer models for throughput, latency, or both enables businesses to either reduce the amount of infrastructure cost for the same workload or to enable real-time use cases that were not possible before. 

If you are interested in trying out Hugging Face Infinity sign up for your trial at [hf.co/infinity-trial](https://hf.co/infinity-trial) 


## Resources

* [Hugging Face Infinity](https://huggingface.co/infinity)
* [Hugging Face Infinity Trial](https://huggingface.co/infinity-trial)
* [Amazon EC2 C6i instances](https://aws.amazon.com/ec2/instance-types/c6i) 
* [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
* [DistilBERT paper](https://arxiv.org/abs/1910.01108)
* [DistilBERT model](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
* [🤗 Infinity: CPU Ice-Lake Benchmark](https://docs.google.com/spreadsheets/d/1GWFb7L967vZtAS1yHhyTOZK1y-ZhdWUFqovv7-73Plg/edit?usp=sharing)
