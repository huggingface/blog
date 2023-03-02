---
title: "How we sped up transformer inference 100x for 🤗 API customers"
thumbnail: /blog/assets/09_accelerated_inference/thumbnail.png
---

<h1>How we sped up transformer inference 100x for 🤗 API customers</h1>

<!-- {blog_metadata} -->

🤗 Transformers has become the default library for data scientists all around the world to explore state of the art NLP models and build new NLP features. With over 5,000 pre-trained and fine-tuned models available, in over 250 languages, it is a rich playground, easily accessible whichever framework you are working in.

While experimenting with models in 🤗 Transformers is easy, deploying these large models into production with maximum performance, and managing them into an architecture that scales with usage is a **hard engineering challenge** for any Machine Learning Engineer. 

This 100x performance gain and built-in scalability is why subscribers of our hosted [Accelerated Inference API](https://huggingface.co/pricing) chose to build their NLP features on top of it. To get to the **last 10x of performance** boost, the optimizations need to be low-level, specific to the model, and to the target hardware.

This post shares some of our approaches squeezing every drop of compute juice for our customers. 🍋


## Getting to the first 10x speedup

The first leg of the optimization journey is the most accessible, all about using the best combination of techniques offered by the [Hugging Face libraries](https://github.com/huggingface/), independent of the target hardware. 

We use the most efficient methods built into Hugging Face model [pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) to reduce the amount of computation during each forward pass. These methods are specific to the architecture of the model and the target task, for instance for a text-generation task on a GPT architecture, we reduce the dimensionality of the attention matrices computation by focusing on the new attention of the last token in each pass:

-| Naive version                                                                                             | Optimized version                                                                                       |
-|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
-|![](/blog/assets/09_accelerated_inference/unoptimized_graph.png)|![](/blog/assets/09_accelerated_inference/optimized_graph.png)|

Tokenization is often a bottleneck for efficiency during inference. We use the most efficient methods from the [🤗 Tokenizers](https://github.com/huggingface/tokenizers/) library, leveraging the Rust implementation of the model tokenizer in combination with smart caching to get up to 10x speedup for the overall latency.

Leveraging the latest features of the Hugging Face libraries, we achieve a reliable 10x speed up compared to an out-of-box deployment for a given model/hardware pair. As new releases of Transformers and Tokenizers typically ship every month, our API customers do not need to constantly adapt to new optimization opportunities, their models just keep running faster.

## Compilation FTW: the hard to get 10x
Now this is where it gets really tricky. In order to get the best possible performance we will need to modify the model and compile it targeting the specific hardware for inference. The choice of hardware itself will depend on both the model (size in memory) and the demand profile (request batching). Even when serving predictions from the same model, some API customers may benefit more from Accelerated CPU inference, and others from Accelerated GPU inference, each with different optimization techniques and libraries applied.

Once the compute platform has been selected for the use case, we can go to work. Here are some CPU-specific techniques that can be applied with a static graph:
- Optimizing the graph (Removing unused flow)
- Fusing layers (with specific CPU instructions)
- Quantizing the operations

Using out-of-box functions from open source libraries (e.g. 🤗 Transformers with [ONNX Runtime](https://github.com/microsoft/onnxruntime)) won’t produce the best results, or could result in a significant loss of accuracy, particularly during quantization. There is no silver bullet, and the best path is different for each model architecture. But diving deep into the Transformers code and ONNX Runtime documentation, the stars can be aligned to achieve another 10x speedup.

## Unfair advantage

The Transformer architecture was a decisive inflection point for Machine Learning performance, starting with NLP, and over the last 3 years the rate of improvement in Natural Language Understanding and Generation has been steep and accelerating. Another metric which accelerated accordingly, is the average size of the models, from the 110M parameters of BERT to the now 175Bn of GPT-3.

This trend has introduced daunting challenges for Machine Learning Engineers when deploying the latest models into production. While 100x speedup is a high bar to reach, that’s what it takes to serve predictions with acceptable latency in real-time consumer applications.

To reach that bar, as Machine Learning Engineers at Hugging Face we certainly have an unfair advantage sitting in the same (virtual) offices as the 🤗 Transformers and 🤗 Tokenizers maintainers 😬.  We are also extremely lucky for the rich partnerships we have developed through open source collaborations with hardware and cloud vendors like Intel, NVIDIA, Qualcomm, Amazon and Microsoft that enable us to tune our models x infrastructure with the latest hardware optimizations techniques.

If you want to feel the speed on our infrastructure, start a [free trial](https://huggingface.co/pricing) and we’ll get in touch.
If you want to benefit from our experience optimizing inference on your own infrastructure participate in our [🤗 Expert Acceleration Program](https://huggingface.co/support). 
