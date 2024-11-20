---
title: "Faster Text Generation with Self-Speculative Decoding"
thumbnail: /blog/assets/layerskip/thumbnail.png
authors:
- user: ariG23498
- user: melhoushi
  guest: true
  org: facebook
- user: pcuenq
- user: reach-vb
---

# Faster Text Generation with Self-Speculative Decoding

Self-speculative decoding, proposed in
[LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
is a novel approach to text generation. It combines the strengths of speculative decoding with early
exiting from a large language model (LLM). This method allows for efficient generation
by using the *same model's* early layers for drafting tokens, and later layers for verification.

This technique not only speeds up text generation, but it also achieves significant
memory savings and reduces computational latency. In order to obtain an end-to-end speedup, the
output of the earlier layers need to be close enough to the last layer. This is achieved by a
training recipe which, as described in the paper, can be applied during pretraining,
and also while fine-tuning on a specific domain. Self-speculative decoding is
especially efficient for real-world applications, enabling deployment on smaller GPUs and lowering
the overall hardware footprint needed for **large-scale inference**.

In this blog post, we explore the concept of self-speculative decoding, its implementation,
and practical applications using the 🤗 transformers library. You’ll learn about the
technical underpinnings, including **early exit layers**, **unembedding**, and **training modifications**.
To ground these concepts in practice, we offer code examples, benchmark comparisons with traditional
speculative decoding, and insights into performance trade-offs.

Dive straight into the following Hugging Face artifacts to know more about the method
and try it out yourself:

1. [Hugging Face Paper Discussion Forum](https://huggingface.co/papers/2404.16710)
2. [LayerSkip Model Collections](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a)
3. [Colab Notebook showcasing the in-depth working of self-speculative decoding](https://colab.research.google.com/drive/1V21LaHaZk_zjhvMLvsWgVSFm6-cn9XAl?usp=sharing)

## Speculative Decoding and Self-Speculative Decoding

![LayerSkip Demo GIF](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LayerSkip-Demo.gif)
*Illustration of LayerSkip inference on [`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
(Llama2 7B continually pretrained with the LayerSkip recipe).*

[Traditional speculative decoding](https://huggingface.co/blog/assisted-generation) uses **two** models:
a smaller one (draft model) to generate a sequence of draft tokens, and a larger one
(verification model) to verify the draft’s accuracy. The smaller model performs a significant
portion of the generation, while the larger model refines the results. This increases text
generation speed since the larger model verifies full sequences at once, instead of generating
one draft at a time.

In self-speculative decoding, the authors build on this concept but use the early layers of a
large model to generate draft tokens that are then verified by the model's deeper layers.
This "self" aspect of speculative decoding, which requires specific training,
allows the model to perform both drafting and verification. This, in turn, improves speed and
reduces computational costs compared to the traditional speculative decoding.

## Usage with `transformers`
 
In order to enable early-exit self-speculative decoding in the
[🤗 transformers library](https://github.com/huggingface/transformers), we
just need to add the `assistant_early_exit` argument to the `generate()` function.

Here is a simple code snippet showcasing the functionality.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

early_exit_layer = 4
prompt = "Alice and Bob"
checkpoint = "facebook/layerskip-llama2-7B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")
outputs = model.generate(**inputs, assistant_early_exit=early_exit_layer)
```

> **Note:** While the `assistant_early_exit` argument can potentially enable early-exit
self-speculative decoding for any decoder-only transformer, the logits from the intermediate layers
cannot be **unembedded** (process of decoding through LM Head, described later in the blog post)
unless the model is specifically trained for that. You will also **only obtain speedups** for
a checkpoint that was trained in such a way to increase the accuracy of earlier layers.
The [LayerSkip paper](https://arxiv.org/abs/2404.16710) proposes a training recipe to achieve that
(namely, applying early exit loss, and progressively increasing layer dropout rates). A collection
of Llama2, Llama3, and Code Llama checkpoints that have been continually pretrained with the
LayerSkip training recipe are provided [here](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a).

### Benchmarking

We ran an extensive list of benchmarks to measure the speedup of LayerSkip’s self-speculative
decoding with respect to autoregressive decoding on various models. We also compare self-speculative
decoding (based on early exiting) with standrad speculative decoding techniques. To reproduce the
results, you may find the code [here](https://github.com/gante/huggingface-demos/pull/1)
and the command to run each experiment in this
[spreadsheet](https://docs.google.com/spreadsheets/d/1YASFEJl5WPmiXbtW-5-PA5nVqtXlZM9YifaAuv49hhI/edit?usp=sharing).
All the experiments were ran on a single 80GB A100 GPU, except for Llama2 70B experiments that
ran on a node of 8 A100 GPUs.


#### Llama2 7B

| Model Variant | Layers | Assistant Model | Assistant Layers | Task | Total Layers | FLOPs/Input (G) | Time/Input (s) | FLOPs/Output (G) | Time/Output (s) | Efficiency |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| meta-llama/Llama-2-7b-hf         | 7     | TinyLlama/TinyLlama_v1.1 | 1    | summarization     | 8     | 2771.54    | 21.65     | 3368.48     | 26.32     | 1.22 |
| meta-llama/Llama-2-7b-hf         | 7     | apple/OpenELM-270M       | 0.27 | summarization     | 7.27  | 2607.82    | 20.37     | 4221.14     | 32.98     | 1.62 |
| meta-llama/Llama-2-7b-hf         | 7     | apple/OpenELM-450M       | 0.45 | summarization     | 7.45  | 3324.68    | 25.97     | 4178.66     | 32.65     | 1.26 |
| **facebook/layerskip-llama2-7B** | **7** | **Early Exit @ Layer 4** |      | **summarization** | **7** | **2548.4** | **19.91** | **3306.73** | **25.83** | **1.297** |

#### Llama2 13B

| Model Variant | Layers | Assistant Model | Assistant Layers | Task | Total Layers | FLOPs/Input (G) | Time/Input (s) | FLOPs/Output (G) | Time/Output (s) | Efficiency |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| meta-llama/Llama-2-13b-hf         | 13     | meta-llama/Llama-2-7b-hf | 7    | summarization     | 20     | 3557.07     | 27.79     | 4088.48     | 31.94     | 1.15 |
| meta-llama/Llama-2-13b-hf         | 13     | TinyLlama/TinyLlama_v1.1 | 1    | summarization     | 14     | 2901.92     | 22.67     | 4190.42     | 32.74     | 1.44 |
| meta-llama/Llama-2-13b-hf         | 13     | apple/OpenELM-270M       | 0.27 | summarization     | 13.27  | 2883.33     | 22.53     | 4521.12     | 35.32     | 1.57 |
| meta-llama/Llama-2-13b-hf         | 13     | apple/OpenELM-450M       | 0.45 | summarization     | 13.45  | 3267.69     | 25.53     | 4321.75     | 33.76     | 1.32 |
| **facebook/layerskip-llama2-13B** | **13** | **Early Exit @ Layer 4** |      | **summarization** | **13** | **4238.45** | **33.11** | **4217.78** | **32.95** | **0.995** |
| **facebook/layerskip-llama2-13B** | **13** | **Early Exit @ Layer 8** |      | **summarization** | **13** | **2459.61** | **19.22** | **4294.98** | **33.55** | **1.746** |

#### Llama2 70B

| Model Variant | Layers | Assistant Model | Assistant Layers | Task | Total Layers | FLOPs/Input (G) | Time/Input (s) | FLOPs/Output (G) | Time/Output (s) | Efficiency |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| meta-llama/Llama-2-70b-hf         | 70     | meta-llama/Llama-2-13b-hf | 13 | summarization     | 83     | 5036.54     | 46.3      | 12289.01    | 112.97    | 2.44 |
| meta-llama/Llama-2-70b-hf         | 70     | meta-llama/Llama-2-7b-hf  | 7  | summarization     | 77     | 4357.55     | 40.06     | 12324.19    | 113.3     | 2.83 |
| meta-llama/Llama-2-70b-hf         | 70     | TinyLlama/TinyLlama_v1.1  | 1  | summarization     | 71     | 4356.21     | 40.05     | 12363.22    | 113.66    | 2.84 |
| **facebook/layerskip-llama2-70B** | **70** | **Early Exit @ Layer 10** |    | **summarization** | **70** | **6012.04** | **54.96** | **1283.34** | **113.2** | **2.06** |

#### Llama3 8B

| Model Variant | Layers | Assistant Model | Assistant Layers | Task | Total Layers | FLOPs/Input (G) | Time/Input (s) | FLOPs/Output (G) | Time/Output (s) | Efficiency |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| meta-llama/Meta-Llama-3-8B       | 8     | meta-llama/Llama-3.2-1B  | 1 | summarization     | 9     | 1872.46     | 19.04     | 2859.35     | 29.08     | 1.53 |
| meta-llama/Meta-Llama-3-8B       | 8     | meta-llama/Llama-3.2-3B  | 3 | summarization     | 11    | 2814.82     | 28.63     | 2825.36     | 28.73     | 1.00 |
| **facebook/layerskip-llama3-8B** | **8** | **Early Exit @ Layer 4** |   | **summarization** | **8** | **1949.02** | **15.75** | **3571.81** | **28.87** | **1.83** |

#### Llama3.2 1B

| Model Variant | Layers | Assistant Model | Assistant Layers | Task | Total Layers | FLOPs/Input (G) | Time/Input (s) | FLOPs/Output (G) | Time/Output (s) | Efficiency |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **facebook/layerskip-llama3.2-1B** | **1** | **Early Exit @ Layer 4** | | **summarization** | **1** | **1195.28** | **9.96** | **2147.7** | **17.9** | **1.80** |

Some observations we can make from the results:

* As seen in the **Total Number of Parameters** column, self-speculative decoding consumes less memory
  because it does not require a separate draft model and weights for the draft stage layers are re-used.  
* For all model sizes and generations except Llama2 70B, the early-exit self-speculative decoding
  is faster than the regular two-model speculative decoding.  
    There could be different reasons for the relatively limited speedups of self-speculative decoding
    on Llama2 70B compared to other models, e.g., the LayerSkip checkpoint of Llama2 70B was continually
    pretrained with fewer tokens (328 M tokens for Llama2 70B compared to 52B tokens for Llama2 7B).
    But this is an area of improvement to investigate for future research. Nevertheless,
    self-speculative decoding for 70B is significantly faster than autoregressive decoding.

## Early Exit and Unembedding

One key technique in self-speculative decoding is early exit, where the generation process can halt
at a pre specified layer. To accomplish this, we **unembed** the logits from these layers by projecting
them onto the language model (LM) head to predict the next token. This allows the model to skip
subsequent layers and improve inference time.

Unembedding can be performed at any transformer layer, turning early-exit into an efficient
token-prediction mechanism. A natural question arises: how can the LM head be adapted to unembed
logits from earlier layers when it was initially trained to work with the final layer only? This
is where the training modifications come into play.

## Training Modifications: *Layer Dropout* and *Early Exit Loss*

In the training phase, we introduce **layer dropout**,
which allows the model to skip certain layers during training. The dropout rate increases
progressively in deeper layers, making the model less reliant on its later layers, as well as
enhancing the model's generalization and speeding up training.

In addition to layer dropout, **early exit loss** is applied to ensure the LM head learns to
unembed different layers. The total loss function for training the model with early exits is
given by a summation of normalized loss from each exit (intermediate layers). This technique enables
efficient training by distributing the learning task across all layers.

## Self-Drafting and Self-Verification

Once training is complete, we can apply self-speculative decoding during inference.
[The process](https://huggingface.co/docs/transformers/v4.46.3/en/llm_optims#speculative-decoding)
begins with **self-drafting**, where tokens are generated by exiting early from some intermediate
layer. The number of speculative tokens defines how many draft tokens are produced during
this stage, and the layer we exit at defines how large and accurate is the draft stage.
Both parameters can be specified at inference  based on a
[trade-off between speed and accuracy of the draft stage](https://huggingface.co/blog/assisted-generation).

The next stage is **self-verification**, where the full model is used to verify the draft tokens.
The verification model reuses the portion of cache from the draft model. If the draft tokens align
with the verified tokens, they are added to the final output, resulting in a better usage of the
memory bandwidth in our system, because it’s much more expensive to generate a sequence of tokens
with the full model than verifying a draft, as long as several of the tokens match.

In the self-verification stage, only the remaining layers are computed for verification, because
the results from the early layers are cached during the drafting phase.

## Optimizations: Shared Weights, Shared KV Cache, and Shared Compute

Self-speculative decoding benefits significantly from cache reuse, particularly the **KV cache**,
which stores key-value pairs computed during the drafting stage. This cache allows the model to skip
redundant calculations, as both the draft and verification stages use the same early layers.
Additionally, the **exit query cache** stores the query vector from the exit layer, allowing
verification to continue seamlessly from the draft stage.

Compared to traditional two-model speculative decoding, early-exit self-speculative decoding can
benefit from the following savings:

* **Shared Weights**: Reuses the weights from the first \\( E \\) layers for both drafting and verification.   
* **Shared KV Cache**: Reuses key-value pairs from the first \\( E \\) layers for both drafting and verification.  
* **Shared Compute**: Reuses the compute of the first \\( E \\) layers by using a
  **Exit Query Cache** that saves only the query vector of the exit layer \\(E-1\\) so that the
  verification process won’t need to compute layers \\( 0 \\) to \\( E-1 \\).

The combination of KV and exit query caches, known as the **KVQ cache**, reduces memory
overhead and improves inference latency.

So far, the 🤗 transformers library has implemented the first optimization (Shared Weights) in this
[pull request](https://github.com/huggingface/transformers/pull/34240). As the number of models
that use this method increases, we'll consider the additional optimizations.
Feel free to open a PR if you're interested!

## How Early Can We Exit?

The early exit layer of the draft stage is a hyperparameter that we can tune or modify during inference:

* The earlier we exit, the faster the generation of draft tokens are but the less accurate they will be.  
* The later we exit, the more accurate the draft tokens generated are but the slower their generation will be.

We wrote [a script](https://gist.github.com/mostafaelhoushi/1dd2781b896504bf0569a3ae4b8f9ecf)
to sweep across different early exit layers and measure the tokens per second on A100 GPUs.
In the Tables below we plot the tokens per second versus the early exit layer for different
Llama models for both LayerSkip and baseline checkpoints (you can view the full logs
[here](https://drive.google.com/drive/folders/145CUq-P_6tbPkmArL7qsjxUihjDgLnzX?usp=sharing)).

<hfoptions id="ablation">
<hfoption id="Llama3 8B">

| Normal | LayerSkip |
| :--: | :--: |
| ![llama 3 8b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/Llama-3-8B.png) | ![layer skip llama 3 8b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-Llama3-8B.png) |

</hfoption>
<hfoption id="Llama3.2 1B">

| Normal | LayerSkip |
| :--: | :--: |
| ![llama 3.2 1b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/Llama-3.2-1B.png) | ![layer skip llama 3.2 1b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-Llama3.2-1B.png) |

</hfoption>
<hfoption id="Llama2 7B">

| Normal | LayerSkip |
| :--: | :--: |
| ![llama 2 7b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/Llama-2-7B.png) | ![layer skip llama 2 7b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-Llama2-7B.png) |

</hfoption>
<hfoption id="Llama2 13B">

| Normal | LayerSkip |
| :--: | :--: |
| ![llama 2 13b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/Llama-2-13B.png) | ![layer skip llama 2 13b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-Llama2-13B.png) |

</hfoption>
<hfoption id="Llama2 70B">

| Normal | LayerSkip |
| :--: | :--: |
| ![llama 2 70b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/Llama-2-70B.png) | ![layer skip llama 2 70b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-Llama2-70B.png) |

</hfoption>
<hfoption id="Code Llama3 7B">

| Normal | LayerSkip |
| :--: | :--: |
| ![code llama 3 7b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/CodeLlama-7B.png) | ![code layer skip llama 3 7b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-CodeLlama-7B.png) |

</hfoption>
<hfoption id="Code Llama3 34B">

| Normal | LayerSkip |
| :--: | :--: |
| ![code llama 3 34b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/CodeLlama-34B.png) | ![code layer skip llama 3 34b](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/layerskip-assets/LS-CodeLlama-34B.png) |

</hfoption>
</hfoptions>

We can observe the following:

* For the baseline checkpoints that have not been pretrained or continually pretrained with the
  LayerSkip training recipe, early exit self-speculative decoding is slower than autoregressive decoding.
  This is because during training of most LLMs, earlier layers are not motivated to learn to predict
  the output, and hence generating tokens using earlier layers will have a very low acceptance rate.   
* On the other hand, for the Llama checkpoints that were continually pre-trained with the LayerSkip
  training, early exit self-speculative decoding has higher speedup than autoregressive decoding for
  at least a subset of the layers.  
  * For most models, except Llama3.2 1B, we notice a regular pattern when we traverse across
    layers: speedup starts low for the first few layers, increases gradually to a sweet spot, and
    then decreases again.   
  * The early exit layer sweet spot is when we have the optimal tradeoff between high accuracy of
    predictions and low overhead of generating tokens. This sweet spot depends on each model,
    and may also depend on the prompt or domain of the prompt.

## Conclusion

LayerSkip leverages the synergy between early exit, layer dropout, and cache reuse to create a fast
and efficient text generation pipeline. By training the model to unembed outputs from different
layers and optimizing the verification process with caches, this approach strikes a balance between
speed and accuracy. As a result, it significantly improves inference times in large language models
while maintaining high-quality outputs. It also reduces memory compared to traditional speculative
decoding techniques due to a single model used as both the draft and verification model.

Self-speculation is an exciting field where the same LLM can create draft tokens and fix itself. Other
self-speculation approaches include:

* [Draft & Verify](https://aclanthology.org/2024.acl-long.607/): where the draft stage involves
  skipping pre-determined attention and feed forward layers.  
* [MagicDec](https://arxiv.org/abs/2408.11049): where the draft stage uses a subset of the KV cache,
  which is useful for long context inputs.  
* [Jacobi Decoding](https://arxiv.org/abs/2305.10427) and [Lookahead Decoding](https://arxiv.org/abs/2402.02057):
  Where the draft stage are a series of “guess tokens” that could be either random or obtained from a n-gram lookup table.