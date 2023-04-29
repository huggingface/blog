---
title: "Assisted Generation: a new direction toward low-latency text generation"
thumbnail: /blog/assets/assisted-generation/thumbnail.png  # TODO
authors:
- user: joaogante
---

# Assisted Generation: a new direction toward low-latency text generation

<!-- {blog_metadata} -->
<!-- {authors} -->

Billions of dollars are being poured into large language models and related technologies. We have been finding that larger models show emergent behavior and can solve a wider array of tasks. However, as humans with ever-decreasing attention spawns, we also dislike their slow response times. Latency is critical for a good user experience, and smaller models are often used despite their lower quality (e.g. in [code completion](https://ai.googleblog.com/2022/07/ml-enhanced-code-completion-improves.html)).

Why is text generation so slow? What’s preventing you from deploying low-latency large language models without going bankrupt? In this blog post, we will revisit the bottlenecks for autoregressive text generation and introduce a new decoding method to tackle the latency problem. You’ll see that using our new method, assisted generation, you can reduce latency up to 10x in commodity hardware!

## Understanding text generation latency

The core of modern text generation is straightforward to understand. Let’s look at the central component, the ML model. Its input contains a text component, which includes the text generated so far, and, depending on the model, it may also have a component of another modality (for instance, Whisper also has an audio input). The model takes this input and does a forward pass: the input is fed to the model and passed sequentially along its layers until the unnormalized log probabilities for the next token are predicted (also known as logits). The term token refers to entire words, sub-words, or even individual characters, depending on the model. The [illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) is a great reference if you’d like to dive deeper into this part of text generation.

<!-- [GIF 1 -- FWD PASS] -->

A model forward pass gets you the logits for the next token, which you can manipulate to your will (e.g. set the probability of undesirable words or sequences to 0). The next step in text generation is to select the next token from these logits. Common strategies include picking the most likely token, known as greedy decoding, or sampling from this distribution, called multinomial sampling. Chaining model forward passes with next token selection iteratively you obtain text generation. This explanation is the tip of the iceberg, please refer to [our blog post on text generation](https://huggingface.co/blog/how-to-generate) for a deeper exploration.

<!-- [GIF 2 -- TEXT GENERATION] -->

From the description above, the bottleneck for text generation becomes clear: running a model forward pass for large models is slow, and you may need to do hundreds of them in a sequence. But let’s dive deeper: why are forward passes slow? Forward passes are typically dominated by matrix multiplications and, after a quick visit to [wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Communication-avoiding_and_distributed_algorithms), you call tell that memory bandwidth is the bottleneck in this operation. In other words, the bottleneck in the forward pass comes from loading the model layer weights into the computation cores of your device, not from performing the computations themselves. At the moment, you have three main avenues you can explore to get the most out of text generation, all tackling the performance of the model forward pass.

First, you have the hardware-specific model optimizations. For instance, your device may be compatible with [Flash Attention](https://github.com/HazyResearch/flash-attention), which speeds up the attention layer through a reorder of the operations, or [INT8 quantization](https://huggingface.co/blog/hf-bitsandbytes-integration), which reduces the size of the model weights.

Second, when you know you’ll get concurrent text generation requests, you can batch the inputs and massively increase the throughput with a small latency penalty. The model layer weights loaded into the device are now used on several input rows in parallel, which means that you’ll get more tokens out for approximately the same memory bandwidth burden. The catch with batching is that you need additional device memory (or to offload the memory somewhere) – at the end of this spectrum, you can see projects like [FlexGen](https://github.com/FMInference/FlexGen) which optimize throughput at the expense of latency.

<!-- [CODE EXAMPLE WITH TIME AND BATCHING] -->

Finally, if you have multiple devices available to you, you can distribute the workload using Tensor Parallelism and obtain lower latency. With [Tensor Parallelism](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism), you split the memory bandwidth burden across multiple devices, but you now have to consider inter-device communication bottlenecks in addition to the monetary cost of running multiple devices. The benefits depend largely on the model size: models that easily fit on a single consumer device see very limited benefits. Taking the results from this [DeepSpeed blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/), you see that you can spread a 17B parameter model across 4 GPUs to reduce the latency by 1.5x (Figure 7).

These three types of improvements can be used in tandem, resulting in [high throughput solutions](https://huggingface.co/blog/bloom-inference-pytorch-scripts). However, after applying hardware-specific optimizations, there are limited options to reduce latency – and the existing options are expensive. Let’s fix that!

## Language decoder forward pass, revisited

You’ve read above that each model forward pass yields the logits for the next token, but that’s actually an incomplete truth. During text generation that is typically the case – except on the first iteration of text generation, the model receives as input the latest generated token and cached internal computations for all other previous inputs. Caching is used to get slightly faster forward passes, but it’s not mandatory. When caching is disabled, the input contains the entire sequence of tokens generated so far and the output contains the logits corresponding to the next token for all positions in the sequence! The logits at position N correspond to the distribution for the next token if the input was only the first N tokens. In the particular case of greedy decoding, if you apply the argmax operator to these logits you will obtain the input sequence back (discarding the first token and adding the next token).


<!-- [CODE WITH EXAMPLE] -->


This means that you can use a model forward pass for a different purpose: instead of feeding some tokens to predict the next one, you can pass an entire sequence to the model and double-check whether the model would generate that same sequence (or part of it) with greedy decoding.


<!-- [GIF 3 -- FWD CONFIRMATION] -->


Let’s consider for a second that you have access to a magical latency-free oracle model that generates the same sequence as your model for a given input. For argument’s sake, it can’t be used directly, it’s an assistant to your generation procedure. Using the property described above, you could use the assistant model to get candidate output tokens and then use your model to confirm that they are indeed correct, in a single forward pass. In this utopian scenario, the latency of text generation with greedy decoding would be reduced by several orders of magnitude.

Walking a step towards reality, the assistant model lost its oracle properties. Now it’s a latency-free model that gets some of the candidate tokens incorrect, according to your model. Due to the autoregressive nature of the task, as soon as the assistant gets a token wrong, all subsequent candidates must be invalidated. However, that does not prevent you from querying the assistant again, after correcting the wrong token with your model, and repeating this process iteratively. Even if the assistant fails one in ten tokens, text generation with greedy decoding would have an order of magnitude less latency than before.

Obviously, there are no latency-free assistant models. Nevertheless, it is relatively easy to find a model that approximates some other model’s text generation outputs – smaller versions of the same architecture with similar training regimes often fit this property. Moreover, when the difference in model sizes becomes significant, the cost of using the smaller model as an assistant becomes an afterthought after factoring in the benefits of skipping forward passes! You now understand the basis for assisted generation.

## Greedy decoding with assisted generation

Assisted generation is a balancing act. You want the assistant to quickly generate a candidate sequence while being as accurate as possible. If the assistant is inaccurate, the method degenerates into overcomplicated greedy decoding. On the other hand, optimizing the quality of the candidate sequences may lead to slow assistants, resulting in a net slowdown. In assisted generation, we’ve included an additional requirement and a heuristic to ensure the time spent with the assistant stays in check.

First, the requirement – the assistant must have the exact same tokenizer as your model. If this requirement was not in place, expensive token decoding and re-encoding steps would have to be added. Furthermore, these additional steps would have to happen on the CPU, which in turn may need slow inter-device data transfers. Fast usage of the assistant is critical for the benefits of assisted generation to show up.

Finally, the heuristic. By this point, you have probably noticed the similarities between the movie Inception and assisted generation – you are, after all, running text generation inside text generation. There will be one assistant model forward pass per candidate token, and we know that forward passes are expensive. While you can’t know in advance the number of tokens that the assistant model will get right, you can keep track of this information and use it to limit the number of candidate tokens requested. We’ve added a heuristic that adapts the number of requested candidate tokens, based on the acceptance of past candidate sequences – some sections of the output are easier to anticipate than others.

Wrapping all up, here’s our original implementation of the assisted generation loop (code):
1. Use greedy decoding to generate a certain number of candidate tokens with the assistant model, producing `candidates`. The number of produced candidate tokens is initialized to `5` the first time assisted generation is called.
2. Using our model, do a forward pass with `candidates`, obtaining `logits`.
3. Use the token selection method (`argmax` for greedy search or `multinomial` for sampling) to get the `next_tokens` from `logits`.
4. Compare `next_tokens` to `candidates` and get the number of matching tokens. Remember that this comparison has to be done with left-to-right causality: after the first mismatch, all candidates are invalidated.
5. Use the number of matches to slice things up and discard variables related to unconfirmed candidate tokens. In essence, keep the matching tokens plus the first divergent token (which our model generates from a valid candidate subsequence).
6. Adjust the number of candidate tokens to be produced in the next iteration — our original heuristic increases it by `2` if ALL tokens match and decreases it by `1` otherwise.

<!-- [GIF 4 -- ASSISTED GENERATION] -->

We’ve designed the API such that this process is hassle-free for you. All you need to do is to pass the assistant model under the new `assistant_model` keyword argument and reap the latency gains! At the time of the release of this blog post, assisted generation is limited to a batch size of 1.


<!-- [CODE EXAMPLE WITH ASSISTED GENERATION] -->


Is the additional complexity worth it? Let’s have a look at the latency numbers for the greedy decoding case (results for sampling are in the next section), considering a batch size of 1. These results were pulled directly out of `transformers` without any additional optimizations, so you should be able to reproduce them locally.


<!-- [GRADIO WITH GREEDY DECODING PERFORMANCE NUMBERS] -->


Glancing at the collected numbers, we see that assisted generation can deliver significant latency reductions in some settings, but it is not a silver bullet – you should benchmark it before applying it to your use case. Nevertheless, we can conclude that assisted generation:
1. Requires access to an assistant model that is at least an order of magnitude smaller than your model (the bigger the difference, the better);
2. Gets up to 3x speedups in the presence of INT8 and up to 2x otherwise, when the model fits in the GPU memory;
3. If you’re playing with models that do not fit in your GPU, you can see up to 10x speedups;
4. Benefits from text-grounded tasks, like automatic speech recognition or summarization.

## Sample with assisted generation

Greedy decoding is suited for text-grounded tasks (automatic speech recognition, translation, summarization, …) or factual knowledge-seeking. Open-ended tasks requiring large levels of creativity, such as most uses of a language model as a chatbot, should rely on sampling instead. Assisted generation is naturally designed for greedy decoding, but that doesn’t mean that you can’t use assisted generation with multinomial sampling!

Drawing samples from a probability distribution for the next token will cause our greedy assistant to fail more often, reducing its latency benefits. However, we can control how sharp the probability distribution for the next tokens is, using the temperature coefficient that’s present in most sampling-based applications. At one extreme, with temperatures close to 0, sampling will approximate greedy decoding, favoring the most likely token. At the other extreme, with the temperature set to values much larger than 1, sampling will be chaotic, drawing from a uniform distribution. Low temperatures are, therefore, more favorable to your assistant model, retaining most of the latency benefits from assisted generation, as we can see below.


<!-- [TEMPERATURE RESULTS] -->


Why do you see it for yourself, so get a feeling of assisted generation?


<!-- [DEMO] -->


## Future directions

Assisted generation shows that modern text generation strategies are ripe for optimization. Understanding that it is currently a memory-bound problem, not a compute-bound problem, allows us to apply simple heuristics to get the most out of the available memory bandwidth, despite requiring additional computations. We believe that further refinement of the use of assistant models will get us even bigger latency reductions. Naturally, releasing high-quality small models to be used as assistants will be critical to realizing and amplifying the benefits.

Initially released under our transformers library, to be used with the `.generate()` function, we expect to offer it throughout the Hugging Face universe. Its implementation is also completely open-source so, if you’re working on text generation and not using our tools, feel free to use it as a reference.

Finally, assisted generation also surfaces a crucial question in text generation. It has been evolving with the constraint where all new tokens result from a fixed amount of compute, for a given model. This blog post reinforces the idea that it shouldn’t be the case: large subsections of the generated output can also be equally generated by models that are a fraction of the size. For that, we’ll need new model architectures and decoding methods – we’re excited to see what the future holds!
