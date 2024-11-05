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

Self-speculative decoding proposed in
[LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
is a novel approach to text generation, combining the strengths of speculative decoding and early
exit techniques within a large language model (LLM). This method allows for efficient generation
by using the *same model's* early layers for drafting tokens and later layers for verification.

By leveraging this technique, we not only speed up text generation but also achieve significant
memory savings and reduce computational latency. In order to obtain an end-to-end speedup, the
output of the earlier layers need to be close enough to the last layer. This is achieved by a
training recipe as described in the paper that could be applied as continual pretraining,
pretraining from scratch, or finetuning on a specific domain. This makes self-speculative decoding
especially efficient for real-world applications, enabling deployment on smaller GPUs and lowering
the overall hardware footprint needed for **large-scale inference**.

Dive straight into the Hugging Face artifacts to know more:

1. [Hugging Face Paper Discussion Forum](https://huggingface.co/papers/2404.16710)
2. [LayerSkip Model Collections](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a)
3. LayerSkip Space

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
large model to generate draft tokens. The early exit logits are used to predict tokens, which are
then verified by the model’s deeper layers. This "self" aspect of speculative decoding, which
requires specific training as explained later, allows the model to perform both drafting and verification.
This, in turn, improves speed and reduces computational costs compared to the traditional speculative decoding.

## Usage with `transformers`
 
In order to enable the early-exit self-speculative decoding in
[🤗 transformers library](https://github.com/huggingface/transformers), we
just need to add `early_exit` argument to a model’s `generate()` function.

Here is a simple code snippet showcasing the functionality.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

early_exit_layer = 4
prompt = "Alice and Bob"
checkpoint = "facebook/layerskip-llama2-7B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")
outputs = model.generate(**inputs, early_exit=early_exit_layer)
```

> **Note:** While the `early_exit` argument can potentially enable early-exit self-speculative decoding for any
decoder-only transformer, you will **only obtain speedups** for a checkpoint that was trained in such a
way to increase the accuracy of earlier layers. The [LayerSkip paper](https://arxiv.org/abs/2404.16710)
proposes a training recipe to achieve that (namely, applying early exit loss, and progressively
increasing layer dropout rates). A collection of Llama2, Llama3, and Code Llama checkpoints that
have been continually pretrained with the LayerSkip training recipe are provided
[here](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a).

### Benchmarking

We ran an extensive list of benchmarks to measure the speedup of LayerSkip’s self-speculative
decoding with respect to autoregressive decoding on various models. We also compare self-speculative
decoding, i.e., using early exit generation as the draft stage, with speculative decoding, i.e.,
using a separate assistant model as the draft stage. To reproduce the results, you may find the
code [here](https://github.com/gante/huggingface-demos/pull/1) and the command to run each experiment
in this
[spreadsheet](https://docs.google.com/spreadsheets/d/1YASFEJl5WPmiXbtW-5-PA5nVqtXlZM9YifaAuv49hhI/edit?usp=sharing).
All the experiments were ran on a single 80GB A100 GPU, except for Llama2 70B experiments that
ran on a node of 8 A100 GPUs.

<hfoptions id="benchmarking">
<hfoption id="Llama3.2 1B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Llama3.2 1B | `facebook/layerskip-llama3.2-1B` | Early Exit @ Layer 4 | **1B** | **1.80x** | **1.35x** | **1.15x** |

</hfoption>
<hfoption id="Llama3 8B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Llama3 8B | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Llama-3.2-1B` | 9B | 1.53x | 1.11x | 1.11x |
|  | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Llama-3.2-3B` | 11B | 1.00x | 0.80x | 0.83x |
|  | `facebook/layerskip-llama3-8B` | Early Exit @ Layer 4 | **8B** | **1.83x** | **1.36x** | **1.51x** |

</hfoption>
<hfoption id="Llama2 7B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Llama2 7B | `meta-llama/Llama-2-7b-hf` | `TinyLlama/TinyLlama_v1.1` | 8B | 1.22x | 1.18x | 0.85x |
|  | `facebook/layerskip-llama2-7B` | Early Exit @ Layer 4 | **7B** | **1.30x** | **1.47x** | **1.41x** |

</hfoption>
<hfoption id="Llama2 13B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Llama2 13B | `meta-llama/Llama-2-13b-hf` | `meta-llama/Llama-2-7b-hf` | 20B | 1.15x | 0.93x | 0.92x |
|  | `meta-llama/Llama-2-13b-hf` | `TinyLlama/TinyLlama_v1.1` | 14B | 1.44x | 1.04x | 1.30x |
|  | `facebook/layerskip-llama2-13B` | Early Exit @ Layer 8 | **13B** | **1.75x** | **1.35x** | **1.49x** |

</hfoption>
<hfoption id="Llama2 70B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Llama2 70B | `meta-llama/Llama-2-70b-hf` | `meta-llama/Llama-2-13b-hf` | 83B | 2.44x | 1.75x | 2.02x |
|  | `meta-llama/Llama-2-70b-hf` | `meta-llama/Llama-2-7b-hf` | 77B | 2.83x | 1.94x | **2.22x** |
|  | `meta-llama/Llama-2-70b-hf` | `TinyLlama/TinyLlama_v1.1` | 71B | **2.84x** | **2.16x** | 2.07x |
|  | `facebook/layerskip-llama2-70B` | Early Exit @ Layer 10 | **70B** | 2.06x | 1.83x | 1.52x |

</hfoption>
<hfoption id="Colde Llama 7B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Code Llama 7B | `codellama/CodeLlama-7b-hf` | `TinyLlama/TinyLlama_v1.1_math_code` | 8B | n/a | n/a | 0.30x |
|  | `facebook/layerskip-codellama-7B` | Early Exit @ Layer 4 | **7B** | n/a | n/a | **1.39x** |

</hfoption>
<hfoption id="Colde Llama 34B">

| Model | Target Checkpoint | Assistant | Total Number of Parameters | Speedup | | |
| :---- | :---- | :---- | :---- | ----- | :---- | :---- |
|  |  |  |  | **summarization** | **open-ended generation** | **code generation** |
| Code Llama 34B | `codellama/CodeLlama-34b-hf` | `codellama/CodeLlama-7b-hf` | 41B | n/a | n/a | 0.11x |
|  | `codellama/CodeLlama-34b-hf` | `TinyLlama/TinyLlama_v1.1_math_code` | 35B | n/a | n/a | 1.33x |
|  | `facebook/layerskip-codellama-34B` | Early Exit @ Layer 8 | **34B** | n/a | n/a | **1.54x** |

</hfoption>
</hfoptions>

Some observations we can make from the results:

* As seen in the **Total Number of Parameters** column, self-speculative decoding consumes less memory
  as it does not require a separate draft model and re-uses the weights of a subset of its layers for
  the draft stage.  
* For all model sizes and generations except Llama2 70B, the early-exit self-speculative decoding
  is faster than the regular two-model speculative decoding.  
* There could be different reasons for the relatively limited speedups of self-speculative decoding
  on Llama2 70B compared to other models, e.g., the LayerSkip checkpoint of Llama2 70B was continually
  pretrained with fewer tokens (328 M tokens for Llama2 70B compared to 52B tokens for Llama2 7B).
  But this is an area of improvement to investigate for future research. Nevertheless,
  self-speculative decoding for 70B is significantly faster than autoregressive decoding.

## Early Exit and Unembedding

One key technique in self-speculative decoding is early exit, where the generation process can halt
at a pre specified layer. To accomplish this, we **unembed** the logits from these layers and project
them onto the language model (LM) head to predict the next token. This allows the model to skip
subsequent layers and improve inference time.

Unembedding can be performed at any transformer layer, turning early-exit into an efficient
token-prediction mechanism. A natural question arises: how can the LM head be adapted to unembed
logits from earlier layers when it was initially trained to work with the final layer only? This
is where the training modifications come into play.

## Training Modifications: *Layer Dropout* and *Early Exit Loss*

In the training phase, the authors introduce **layer dropout**, also known as stochastic depth,
which allows the model to skip certain layers during training. The dropout rate increases
progressively across layers, enhancing the model's generalization and speeding up training.

In addition to layer dropout, **early exit loss** is applied to ensure the LM head learns to
unembed different layers. The total loss function for training the model with early exits is
given by a summation of normalized loss from each exit.

$$ J(X, Y, t) = \sum_{l=0}^{L-1} \tilde{e}(t, l) J_{\text{CE}}(g(x_{l+1}), Y) $$

Where \\( g() \\) is the unembedding operation, \\( x\_{l+1} \\) is the output of transformer layer
\\( l \\), \\( Y \\) is the index of the next token ground truth,\\( tilde{e}(t, l) \\) is a
normalized per-layer loss scale at layer \\( l \\), which we can keep constant throughout training
or create a schedule to make it dependent on iteration \\( t \\), ensuring that the early exit
layers are properly supervised during training. This technique enables efficient training by
distributing the learning task across all layers.

## Self-Drafting and Self-Verification

Once training is complete, we can apply self-speculative decoding during inference. The process
begins with **self-drafting**, where tokens are generated by exiting early from some intermediate
layer. The number of speculative tokens \\( d \\) defines how many draft tokens are produced during
this stage, and the layer we exit at \\( E \\) defines how large and accurate is the draft stage.
Both parameters can be specified at inference  based on a
[trade-off between speed and accuracy](https://huggingface.co/blog/assisted-generation).

The next stage is **self-verification**, where the full model is used to verify the draft tokens.
The verification model reuses the portion of cache from the draft model. If the draft tokens align
with the verified tokens, they are added to the final output, resulting in a better usage of the
memory bandwidth in our system, because it’s much more expensive to generate a sequence of tokens
with the full model than verifying a draft, as long as several of the tokens match

In the self-verification stage, the remaining layers \\( L - E \\) are computed for verification,
with \\( E \\) being the exit layer depth used during drafting.

## Optimizations: Shared Weights, Shared KV Cache, and Exit Query Cache

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

So far, the transformers library has implemented the first optimization (Shared Weights) in this
[pull request](https://github.com/huggingface/transformers/pull/34240).

## Ablation Studies

The early exit layer of the draft stage is a hyperparameter that we can tune or modify during inference:

* The earlier we exit, the faster the generation of candidate tokens are but the less accurate they will be.  
* The later we exit, the more accurate the tokens generated are but the slower their generation will be.

Hence, we wrote up a script [here](https://gist.github.com/mostafaelhoushi/1dd2781b896504bf0569a3ae4b8f9ecf)
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

LayerSkip leverages the synergy between early exits, layer dropout, and cache reuse to create a fast
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