---
title: "Benchmarking Text Generation Inference" 
thumbnail: /blog/assets/tgi-benchmarking/tgi-benchmarking-thumbnail.png
authors:
- user: derek-thomas
---

# Introduction

In this blog we will be exploring [Text Generation Inference’s](https://github.com/huggingface/text-generation-inference) (TGI) little brother, the [TGI Benchmark tool](https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md). It will help us understand how to profile TGI beyond simple throughput to better understand the tradeoffs to make decisions on how to tune your deployment for your needs. If you have ever felt like LLM deployments cost too much or if you want to tune your deployment to improve performance that you are leaving on the table this blog is for you!

I’ll show you how to do this in a convenient [Hugging Face Space](https://huggingface.co/spaces). You can take the results and use it on an [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated) or other copy of the same hardware.


# Motivation

To get a better understanding of the need to profile, let's discuss some background information first.

Large Language Models (LLMs) are fundamentally inefficient. Based on [the way decoders work](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt), generation requires a new forward pass for each decoded token. As LLMs increase in size, and [adoption rates surge](https://a16z.com/generative-ai-enterprise-2024/) across enterprises, the AI industry has done a great job of creating new optimizations and performance enhancing techniques.

There have been dozens of improvements in many aspects of serving LLMs. We have seen [Flash Attention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention), [Paged Attention](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention), [streaming responses](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming), [improvements in batching](https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher#maxwaitingtokens), [speculation](https://huggingface.co/docs/text-generation-inference/en/conceptual/speculation), [quantization](https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization) of many kinds, [improvements in web servers](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#architecture), adoptions of [faster languages](https://github.com/search?q=repo%3Ahuggingface%2Ftext-generation-inference++language%3ARust&type=code) (sorry python 🐍), and many more. There are also use-case improvements like [structured generation](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) and [watermarking](https://huggingface.co/blog/watermarking) that now have a place in the LLM inference world. The problem is that fast and efficient implementations require more and more niche skills to implement [1]. 

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a high-performance LLM inference server from Hugging Face designed to embrace and develop the latest techniques in improving the deployment and consumption of LLMs. Due to Hugging Face’s open-source partnerships, most (if not all) major Open Source LLMs are available in TGI on release day.

Oftentimes users will have very different needs depending on their use-case requirements. Consider a RAG use-case: The input to an LLM in RAG will consist of instructions/formatting (usually short, &lt;200 tokens), the user query (usually short, &lt;200 tokens), multiple Documents (medium-sized, 500-1000 tokens per document, N documents where N&lt;10) and receive an answer in the output (medium-sized ~500-1000 tokens). In RAG it's important to have the right document to get a quality response, you increase this chance by increasing N which includes more documents. This means that RAG will often try to max out an LLM’s context window to increase task performance. In contrast, think about basic chat which will have multiple turns (Tx50-200 tokens, for T turns). Typical chat scenarios have significantly fewer tokens than RAG,  And given that we have such different scenarios, we need to make sure that we configure our LLM server accordingly depending on which one is more relevant. Hugging Face has a [benchmarking tool](https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md) that can help us explore what configurations make the most sense and I'll explain how you can do this on a [Hugging Face Space](https://huggingface.co/docs/hub/en/spaces-overview). 


# Pre-requisites

Let’s make sure we have a common understanding of a few key concepts before we dive into the tool.


## Throughput vs Latency



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


[Figure 1](https://www.totalphase.com/blog/2022/09/how-does-latency-throughput-affect-speed-system/)

Lets imagine this pipe in Figure 1 has tokens flowing through it. In this analogy, the left side of the pipe is the server and the right side is the user. There are 2 dimensions we want to consider: 



* Throughput - how many tokens flow through a section in a set amount of time (e.g. 100 tokens/sec)
* Token Latency - The amount of time it takes 1 token to flow from one end to the other. 
* Request Latency - The time it takes for 

It’s important to understand that these are orthogonal measurements, and depending on how we configure our server, we can optimize for one or the other. Our benchmarking tool will help us understand the trade-off via a data visualization.


## Pre-filling and Decoding

![Prefilling vs Decoding](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/prefilling_vs_decoding.png)

Here is a simplified view of how an LLM generates text. The model (typically) generates a single token for each forward pass. For the **pre-filling stage** in orange, the full prompt (What is.. of the US?)  is sent to the model and one token (Washington) is generated.  In the **decoding stage** in blue, the generated token is appended to the previous input and then this (... the capital of the US? Washington) is sent through the model for another forward pass. Until the model generates the end-of-sequence-token (&lt;EOS>), this process will continue: send input through the model, generate a token, append the token to input.

Note:

Why does pre-filling only take 1 pass when we are submitting many tokens as input?

We don’t need to generate what comes after “What is the”. We know its “capital” from the user. 

I only included a short example for illustration purposes, but consider that pre-filling only needs 1 forward pass through the model, but decoding can take hundreds or more. Even in our short example we can see more blue arrows than orange. We can see now why it takes so much time to get output from an LLM! Decoding is usually where we spend more time thinking through due to the many passes.


# Benchmark Tool


## Motivation

We have all seen comparisons of tools, new algorithms, or models that show throughput. While this is an important part of the LLM inference story, it's missing some key information. At a minimum (you can of course go more in-depth) we need to know what the throughput AND what the latency is to make good decisions. One of the primary benefits of the TGI benchmark tool is that it has this capability. 

Another important line of thought is considering what experience you want the user to have. Do you care more about serving to many users, or do you want each user once engaged with your system to have a fast response? Do you want to have a better Time To First Token (TTFT) or do you want blazing fast tokens to appear once they get their first token even if the first one is delayed?

Here are some ideas on how that can play out. Remember there is no free lunch. But with enough GPUs, you can have almost any meal you want.


<table>
  <tr>
   <td>I care about…
   </td>
   <td>I should focus on…
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



# Setup

The benchmarking tool is installed with TGI, but you need access to the server to run it. With that in mind I’ve provided this space [derek-thomas/tgi-benchmark-space](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space) to combine a TGI docker image (pinned to latest) and a jupyter lab working space. This will allow us to deploy a model of our choosing and easily run the benchmarking tool via a CLI. I’ve added some notebooks that will allow you to easily follow along. Feel free to dive into the [Dockerfile](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space/blob/main/Dockerfile) to get a feel for how it’s built, especially if you want to tweak it. 

Getting Started

Please note that it's much better to run the benchmarking tool in a jupyter lab terminal rather than a notebook due to its interactive nature, but I'll put the commands in a notebook so I can annotate and it's easy to follow along.



1. <a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space?duplicate=true"><img style="margin-top:0;margin-bottom:0" src="https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg" alt="Duplicate Space"></a>:
    * Set your default password in the `JUPYTER_TOKEN` [space secret](https://huggingface.co/docs/hub/spaces-sdks-docker#secrets) (it should prompt you upon duplication)
    * Choose your HW, note that it should mirror the HW you want to deploy on
2. Go to your space and login with your password
3. Launch `01_1_TGI-launcher.ipynb`
    * This will launch TGI with default settings using the jupyter notebook
4. Launch `01_2_TGI-benchmark.ipynb` 
    * This will launch the TGI benchmark tool with some demo settings


## Main Components

![Benchmarking Tool Numbered](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tgi-benchmarking/TGI-benchmark-tool-numbered.png)




* **1**: Batch Selector and other information. 
    * Use your arrows to select different batches
* **2** and **4**: Pre-fill stats and histogram
    * The calculated stats/histogram are based on how many `--runs`
* **3** and **5: **Pre-fill Throughput vs Latency Scatter Plot
    * X-axis is latency (small is good)
    * Y-axis is throughput (large is good)
    * The legend shows us our batch-size
    * An “ideal” point would be in the top left corner (low latency and high throughput)


# Understanding the Benchmark tool

If you used the same HW and settings I did, you should have a really similar chart to Figure 1. The benchmark tool is showing us the throughput and latency for different batch sizes (amounts of user requests, slightly different than the usage when we are launching TGI) for the current settings and HW given when we launched TGI. This is important to understand as we should update the settings in how we launch TGI based on our findings with the benchmark tool.

The chart in **3** tends to be more interesting as we get longer pre-fills like in RAG. It does impact TTFT (shown on the X-axis) which is a big part of the user experience. Remember we get to push our input tokens through in one forward pass even if we do have to build the KV cache from scratch. So it does tend to be faster in many cases per token than decoding.

The chart in **5** is when we are decoding. Let's take a look at the shape the data points make. We can see that for batch sizes of 1-32 the shape is mostly vertical at ~5.3s. This is really good. This means that for no degradation in latency we can improve throughput significantly! What happens at 64 and 128? We can see that while our throughput is increasing, we are starting to tradeoff latency.

For these same values let's check out what is happening on chart **3**. For batch size 32 we can see that we are still about 1 second for our TTFT. But we do start to see linear growth from 32 -> 64 -> 128, 2x the batch size has 2x the latency. Further there is no throughput gain! This means that we don't really get much benefit from the tradeoff. 

Note callout:

What types of shapes do you expect these curves to take if we add more points?

How would you expect these curves to change if you have more tokens (pre-fill or decoding)?


# Winding Down

It's important to keep track of actual user behavior. When we estimate user behavior we have to start somewhere and make educated guesses. These number choices will make a big impact on how we are able to profile. Luckily TGI can tell us this information in the logs, so be sure to check that out as well.

Once you are done with your exploration, be sure to stop running everything so you wont incur further charges.



* Kill the running cell in the `TGI-launcher.ipynb` jupyter notebook
* Hit `q` in the terminal to stop the profiling tool. 
* Hit pause in the settings of the space


# Conclusion

LLMs are bulky and expensive, but there are a number of ways to reduce that cost. LLM inference servers like TGI have done most of the work for us as long as we leverage their capabilities properly. The first step is to understand what is going on and what trade-offs you can make. We’ve seen how to do that with the TGI Benchmarking tool. We can take these results and use them on any equivalent HW in AWS, GCP, or Inference Endpoints. 


# Sources



1. [Hardware Lottery](https://hardwarelottery.github.io)
2. Figure 1 Source: [https://www.totalphase.com/blog/2022/09/how-does-latency-throughput-affect-speed-system/](https://www.totalphase.com/blog/2022/09/how-does-latency-throughput-affect-speed-system/)
3. Figure 2 inspiration: [https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5](https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5)
4. 