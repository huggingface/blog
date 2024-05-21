---
title: "Benchmarking Text Generation Inference" 
thumbnail: /blog/assets/tgi-benchmarking/tgi-benchmarking-thumbnail.png
authors:
- user: derek-thomas
---
# Introduction

In this blog we will be exploring [Text Generation Inference’s](https://github.com/huggingface/text-generation-inference) (TGI) little brother, the [TGI Benchmarking tool](https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md). It will help us understand how to profile TGI beyond simple throughput to better understand the tradeoffs to make decisions on how to tune your deployment for your needs. If you have ever felt like LLM deployments cost too much or if you want to tune your deployment to improve performance this blog is for you!

I’ll show you how to do this in a convenient [Hugging Face Space](https://huggingface.co/spaces). You can take the results and use it on an [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated) or other copy of the same hardware.


# Motivation

To get a better understanding of the need to profile, let's discuss some background information first.

Large Language Models (LLMs) are fundamentally inefficient. Based on [the way decoders work](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt), generation requires a new forward pass for each decoded token. As LLMs increase in size, and [adoption rates surge](https://a16z.com/generative-ai-enterprise-2024/) across enterprises, the AI industry has done a great job of creating new optimizations and performance enhancing techniques.

There have been dozens of improvements in many aspects of serving LLMs. We have seen [Flash Attention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention), [Paged Attention](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention), [streaming responses](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming), [improvements in batching](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher#maxwaitingtokens), [speculation](https://huggingface.co/docs/text-generation-inference/en/conceptual/speculation), [quantization](https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization) of many kinds, [improvements in web servers](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#architecture), adoptions of [faster languages](https://github.com/search?q=repo%3Ahuggingface%2Ftext-generation-inference++language%3ARust&type=code) (sorry python 🐍), and many more. There are also use-case improvements like [structured generation](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) and [watermarking](https://huggingface.co/blog/watermarking) that now have a place in the LLM inference world. The problem is that fast and efficient implementations require more and more niche skills to implement [[1]](#1). 

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a high-performance LLM inference server from Hugging Face designed to embrace and develop the latest techniques in improving the deployment and consumption of LLMs. Due to Hugging Face’s open-source partnerships, most (if not all) major Open Source LLMs are available in TGI on release day.

Oftentimes users will have very different needs depending on their use-case requirements. Consider prompt and generation in a **RAG use-case**: 
* Instructions/formatting
    * usually short, <200 tokens
* The user query
    * usually short, <200 tokens
* Multiple documents 
    * medium-sized, 500-1000 tokens per document, 
    * N documents where N<10
* An answer in the output 
    * medium-sized ~500-1000 tokens
 
In RAG it's important to have the right document to get a quality response, you increase this chance by increasing N which includes more documents. This means that RAG will often try to max out an LLM’s context window to increase task performance. In contrast, think about basic chat. Typical **chat scenarios** have significantly fewer tokens than RAG:
* Multiple turns
    * 2xTx50-200 tokens, for T turns
    * The 2x is for both User and Assistant 

Given that we have such different scenarios, we need to make sure that we configure our LLM server accordingly depending on which one is more relevant. Hugging Face has a [benchmarking tool](https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md) that can help us explore what configurations make the most sense and I'll explain how you can do this on a [Hugging Face Space](https://huggingface.co/docs/hub/en/spaces-overview). 


# Pre-requisites

Let’s make sure we have a common understanding of a few key concepts before we dive into the tool.

## Latency vs Throughput
<video style="width: auto; height: auto;" controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/LatencyThroughputVisualization.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

|                                                 |
|-------------------------------------------------|
| *Figure 1: Latency vs Throughput Visualization* |

* Token Latency – The amount of time it takes 1 token to be processed and sent to a user
* Request Latency – The amount of time it takes to respond to a request
* Time to First Token
* Throughput – how many tokens the server can return in a set amount of time 4 tokens per second in this case

Latency is a tricky measurement because it doesn’t tell you the whole picture. You might have a long generation or a short one which won't tell you much regarding your actual server performance.

It’s important to understand that Throughput and Latency are orthogonal measurements, and depending on how we configure our server, we can optimize for one or the other. Our benchmarking tool will help us understand the trade-off via a data visualization.


## Pre-filling and Decoding
|![Prefilling vs Decoding](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/prefilling_vs_decoding.png)|
|:--:|
|*Figure 2: Prefilling vs Decoding inspired by [[2]](#2)*|

Here is a simplified view of how an LLM generates text. The model (typically) generates a single token for each forward pass. For the **pre-filling stage** in orange, the full prompt (What is.. of the US?)  is sent to the model and one token (Washington) is generated.  In the **decoding stage** in blue, the generated token is appended to the previous input and then this (... the capital of the US? Washington) is sent through the model for another forward pass. Until the model generates the end-of-sequence-token (\<EOS\>), this process will continue: send input through the model, generate a token, append the token to input.

<br>
<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 10px;">
Thinking Question: Why does pre-filling only take 1 pass when we are submitting multiple unseen tokens as input?
<details>
<summary>Click to reveal the answer</summary>
We don’t need to generate what comes after “What is the”. We know its “capital” from the user.  </div>
</details>
</div>

I only included a short example for illustration purposes, but consider that pre-filling only needs 1 forward pass through the model, but decoding can take hundreds or more. Even in our short example we can see more blue arrows than orange. We can see now why it takes so much time to get output from an LLM! Decoding is usually where we spend more time thinking through due to the many passes.

# Benchmarking Tool

## Motivation

We have all seen comparisons of tools, new algorithms, or models that show throughput. While this is an important part of the LLM inference story, it's missing some key information. At a minimum (you can of course go more in-depth) we need to know what the throughput AND what the latency is to make good decisions. One of the primary benefits of the TGI benchmarking tool is that it has this capability. 

Another important line of thought is considering what experience you want the user to have. Do you care more about serving to many users, or do you want each user once engaged with your system to have a fast response? Do you want to have a better Time To First Token (TTFT) or do you want blazing fast tokens to appear once they get their first token even if the first one is delayed?

Here are some ideas on how that can play out. Remember there is no free lunch. But with enough GPUs and a proper configuration, you can have almost any meal you want.

<table>
  <tr>
   <td><strong>I care about…</strong>
   </td>
   <td><strong>I should focus on…</strong>
   </td>
  </tr>
  <tr>
   <td>Handling more users
   </td>
   <td>Maximizing Throughput
   </td>
  </tr>
  <tr>
   <td>People not navigating away from my page/app
   </td>
   <td>Minimizing TTFT
   </td>
  </tr>
  <tr>
   <td>User Experience for a moderate amount of users
   </td>
   <td>Minimizing Latency
   </td>
  </tr>
  <tr>
   <td>Well rounded experience
   </td>
   <td>Capping latency and maximizing throughput
   </td>
  </tr>
</table>

## Setup

The benchmarking tool is installed with TGI, but you need access to the server to run it. With that in mind I’ve provided this space [derek-thomas/tgi-benchmark-space](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space) to combine a TGI docker image (pinned to latest) and a jupyter lab working space. It's designed to be duplicated, so dont be alarmed if it's sleeping. It will allow us to deploy a model of our choosing and easily run the benchmarking tool via a CLI. I’ve added some notebooks that will allow you to easily follow along. Feel free to dive into the [Dockerfile](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space/blob/main/Dockerfile) to get a feel for how it’s built, especially if you want to tweak it. 

## Getting Started

Please note that it's much better to run the benchmarking tool in a jupyter lab terminal rather than a notebook due to its interactive nature, but I'll put the commands in a notebook so I can annotate and it's easy to follow along.

1. Click: <a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space?duplicate=true"><img style="margin-top:0;margin-bottom:0" src="https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg" alt="Duplicate Space"></a>
    * Set your default password in the `JUPYTER_TOKEN` [space secret](https://huggingface.co/docs/hub/spaces-sdks-docker#secrets) (it should prompt you upon duplication)
    * Choose your HW, note that it should mirror the HW you want to deploy on
2. Go to your space and login with your password
3. Launch `01_1_TGI-launcher.ipynb`
    * This will launch TGI with default settings using the jupyter notebook
4. Launch `01_2_TGI-benchmark.ipynb` 
    * This will launch the TGI benchmarking tool with some demo settings

## Main Components
|![Benchmarking Tool Numbered](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/TGI-benchmark-tool-numbered.png)|
|:--:|
|*Figure 3: Benchmarking Tool Components*|

* **Component 1**: Batch Selector and other information. 
    * Use your arrows to select different batches
* **Component 2** and **Component 4**: Pre-fill stats and histogram
    * The calculated stats/histogram are based on how many `--runs`
* **Component 3** and **Component 5**: Pre-fill Throughput vs Latency Scatter Plot
    * X-axis is latency (small is good)
    * Y-axis is throughput (large is good)
    * The legend shows us our batch-size
    * An “*ideal*” point would be in the top left corner (low latency and high throughput)

## Understanding the Benchmarking tool
|![Benchmarking Tool Charts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/TGI-benchmark-screenshot.png)|
|:--:|
|*Figure 4: Benchmarking Tool Charts*|

If you used the same HW and settings I did, you should have a really similar chart to Figure 4. The benchmarking tool is showing us the throughput and latency for different batch sizes (amounts of user requests, slightly different than the language when we are launching TGI) for the current settings and HW given when we launched TGI. This is important to understand as we should update the settings in how we launch TGI based on our findings with the benchmarking tool.

The chart in **Component 3** tends to be more interesting as we get longer pre-fills like in RAG. It does impact TTFT (shown on the X-axis) which is a big part of the user experience. Remember we get to push our input tokens through in one forward pass even if we do have to build the KV cache from scratch. So it does tend to be faster in many cases per token than decoding.

The chart in **Component 5** is when we are decoding. Let's take a look at the shape the data points make. We can see that for batch sizes of 1-32 the shape is mostly vertical at ~5.3s. This is really good. This means that for no degradation in latency we can improve throughput significantly! What happens at 64 and 128? We can see that while our throughput is increasing, we are starting to tradeoff latency.

For these same values let's check out what is happening on the chart in **Component 3**. For batch size 32 we can see that we are still about 1 second for our TTFT. But we do start to see linear growth from 32 -> 64 -> 128, 2x the batch size has 2x the latency. Further there is no throughput gain! This means that we don't really get much benefit from the tradeoff. 

<br>
<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 10px;">
Thinking Questions:
<ul>
  <li>What types of shapes do you expect these curves to take if we add more points?</li>
  <li>How would you expect these curves to change if you have more tokens (pre-fill or decoding)?</li>
</ul>
</div>

If your batch size is in a vertical area, this is great, you can get more throughput and handle more users for free. If your batch size is in a horizontal area, this means you are compute bound and increasing users just delays everyone with no benefit of throughput. You should improve your TGI configuration or scale your hardware.

Now that we learned a bit about TGI’s behavior in various scenarios we can try different settings for TGI and benchmark again. It's good to go through this cycle a few times before deciding on a good configuration. If there is enough interest maybe we can have a part 2 which dives into the optimization for a use-case like chat or RAG.

## Winding Down

It's important to keep track of actual user behavior. When we estimate user behavior we have to start somewhere and make educated guesses. These number choices will make a big impact on how we are able to profile. Luckily TGI can tell us this information in the logs, so be sure to check that out as well.

Once you are done with your exploration, be sure to stop running everything so you won't incur further charges.
* Kill the running cell in the `TGI-launcher.ipynb` jupyter notebook
* Hit `q` in the terminal to stop the profiling tool. 
* Hit pause in the settings of the space

# Conclusion

LLMs are bulky and expensive, but there are a number of ways to reduce that cost. LLM inference servers like TGI have done most of the work for us as long as we leverage their capabilities properly. The first step is to understand what is going on and what trade-offs you can make. We’ve seen how to do that with the TGI Benchmarking tool. We can take these results and use them on any equivalent HW in AWS, GCP, or Inference Endpoints. 

Thanks to Nicolas Patry and Olivier Dehaene for creating [TGI](https://github.com/huggingface/text-generation-inference) and its [benchmarking tool](https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md). Also special thanks to Nicholas Patry, Moritz Laurer, Nicholas Broad, Diego Maniloff, and Erik Rignér for their very helpful proofreading. 

# References
<a id="1">[1]</a> : Sara Hooker, [The Hardware Lottery](https://arxiv.org/abs/1911.05248), 2020

<a id="2">[2]</a> : Pierre Lienhart, [LLM Inference Series: 2. The two-phase process behind LLMs’ responses](https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5), 2023


