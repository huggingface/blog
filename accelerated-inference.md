---
title: "Speeding up inference by 10-100x for 4000 models"
---

# **Speeding up inference by 10-100x for 5,000 models**

At HuggingFace we're running a [hosted inference API](https://huggingface.co/pricing) that runs
all the models from the hub for a small price. One of the perks of using the API besides running
your own is that we provide with accelerated API that speeds up models by up to 100x (mostly likely 10x). This blogpost describes how we are achieving it.


<p align="center">
  <img src="/blog/assets/accelerated-inference/thumbnail.png" />
</p>


This blogpost will assume some knowledge of the internals of transformers in general
and will link to relevant parts for everything we don't explain here.

## Running a pipeline (Open-source and already within transformers)

The first speedup you should get is already implemented for you for free directly
within [transformers](https://github.com/huggingface/transformers). And it's enabled
by using [pipelines](https://huggingface.co/transformers/main_classes/pipelines.html?highlight=pipelines) instead of coming up with your own code to run the inference.
It works only for Seq2Seq models and generation models, but that encompasses the
most popular models that we see used for inference. [gpt2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Bart](https://arxiv.org/abs/1910.13461),
[Pegasus](https://arxiv.org/abs/1912.08777), [t5](https://arxiv.org/abs/1910.10683), [gpt3](https://arxiv.org/abs/2005.14165) etc... It's also currently implemented only for [pytorch](https://pytorch.org/)
implementations, but we're working on enabling it on [TF](https://www.tensorflow.org/) too.

Within the architecture of transformers, there are Attention layers which are
the most important part of it's success. The core of it, is that it's a NxN
matrix computations between all the tokens of the input. That means for each
pass of the model we're computing a `N²` matrix for each layer of attention.
That's quite intensive. When we're running a `generation` task, that means we're
running **at each step** the full computation **on almost the same tokens** (minus the new one).

The first trick is that we don't need to compute everything again, but only the **new**
attention to the **new** token. so we're going to compute only a `N` matrix. We're still
running `N²` matrix multiplications in the fully connected parts.

Here is a small visualization:

| Naive version                                                                                             | Optimized version                                                                                       |
|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
|![](/blog/assets/accelerated-inference/unoptimized_graph.png)|![](/blog/assets/accelerated-inference/optimized_graph.png)|

That's achieved within transformers by using `past_key_values` input, which is
given to you when you call a model. 

Overall this performance boost achieves ~10x improvement.


## Fast tokenizers (Open-source and already within transformers)

The second part of the speedup is done within the [tokenizers](https://github.com/huggingface/tokenizers/) library.
If you are running `transformers>=4.0` you're most likely using that speedup already !
Tokenization is the process of changing text (a sequence of bytes) into tokens (a sequence
of ints) before feeding it to you preferred model. The tokenization process is usually quite slow
when done within `python` for instance. That's why we created the `tokenizers` library
that is implemented in Rust for performance.

Tokenizing is also an operation that is likely to be cached in most inference situation
so you can also get speedups if you do that correctly. Using `pipelines` from above
will get you all the caching you're going to need.

Overall the Fast tokenizers get a ~10x speedup for the tokenization part, so depending of
your usage it can be a significant part of the speedup too !


## Compiling your model (ONNX, onnxruntime, not fully open source yet)

For the last part of the speedup (another ~10x) we're going to need to dive 
a little bit more into gory details.
We're going to focus on CPU speedups as most inference are run on CPU (it's much cheaper than GPU) but
the same principles can be also applied to GPU.

In order to accelerate inference, there are 2 strategies we can employ with a static graph

 - Optimizing the graph (Removing unused flow)
 - Fusing layers (with specific CPU instructions)
 - Quantizing the operations

If you are using `ONNX` format for your model you can get all those 3 for free
by using [onnxruntime](https://github.com/microsoft/onnxruntime). The only problem
is that most likely you won't run the adjusted version of the graph that are used 
with the pipelines (that are highly dynamic) so you can't get the whole 100x if
you convert your graph blindly. So the first part is actually exporting the correct
graph. This requires sometimes quite a bit of changes with the `transformers` 
implementations of the model. It makes the code not quite as readable which is
why it's not open sourced (at least for now). The current goal of transformers
is to make it as easy as possible to understand with emphasis on the researchers
/ fine tuners users in mind. Making the code overly complex/bloated for the inference
part is too costly to be implemented within transformers. We **are** thinking 
about open-sourcing the solution in a separate repository, but we're not committing just yet.
Please let us know in the comments if you want it !

In order to get the best of this method, you need to be aware that the type of
CPU is important (AVX512 for instance will help tremendously), the fusing of layers
might not work out of the box with the transformers implementations of graphs, so you
need either to edit your ONNX implementations or override some `transformers` methods
in order to get all the inference optimized.
Running advanced `onnxruntime` options is also necessary
like [Gpt2Helper](https://github.com/microsoft/onnxruntime/blob/094384781ea0caa3931061609ca90a84b6a0b64c/onnxruntime/python/tools/transformers/gpt2_helper.py#L100) for instance.
Read the `onnxruntime` documentation thoroughly to get the maximum out of it.
Also be careful when using quantization, there **is** some precision loss, and
can lead to very bad results if not done correctly, so be careful when using it.

Be also careful when trying this method that the actual output of `onnxruntime` 
is platform dependent when you are agressive, meaning that you need to compute
the graph on each machine you are using, you cannot simply move the files around.
 

# Conclusion

Overall, we showed how to get ~100x speedup in inference time compared to naïve
implementations, by using *cached* graphs, *native* tokenizers and *compiling*
your model with a runtime + quantization optimizations. And that the real trick
is to actually combine all those methods at the same time in your product.

To test all these optimizations right now, you can check out the results in our [Hosted API inference](https://huggingface.co/pricing). Be mindful that we require paid subscription for the accelerated part, but we have a 7-day free trial.
so that should be enough to check what is achievable before doing the optimizations on your end.
We also scale for you automatically, saving you the trouble of figuring out how to best
deploy those large RAM intensive models, switching to GPU with a simple call parameter and more...
