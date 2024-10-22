---
title: "Releasing Outlines-core 0.1.0: structured generation in Rust and Python" 
thumbnail: /blog/assets/outlines-core/thumbnail.gif
authors:
- user: bwillard
  guest: true
  org: dottxt
- user: drbh
- user: erikkaum
- user: kc611
  guest: true
  org: dottxt
- user: remi
  guest: true
  org: dottxt
- user: umut-sahin
  guest: true
  org: dottxt
- user: willkurt
  guest: true
  org: dottxt
---

dottxt and Hugging Face are excited to announce that we have been collaborating on [outlines-core](https://github.com/dottxt-ai/outlines-core), a Rust port of [outlines](https://github.com/dottxt-ai/outlines)’s core algorithms for structured generation. On top of getting reliable output from LLMs with outlines, this Rust port offers several further benefits to users of outlines:

- Speed: Users can expect to see an 2x improvement in index compilation.
- Separation of Concerns: It's now easier to incorporate structured generation into other libraries. `outlines-core` is very lightweight.
- Portability: Having core algorithms in Rust allows binding for languages other than Python.

These improvements should not only improve the performance for existing `outlines` users, but also dramatically increase the ways users can incorporate structured generation into their LLM workflows. `outlines-core` is now public, integrated in `outlines`, and the version `0.1.0` of the Python bindings are out. You can find the repo [here](https://github.com/dottxt-ai/outlines-core).

## A quick primer on structured generation 🧑‍🎓 

### How it works

Structured generation means that your LLM is guaranteed to follow a desired format. This could be JSON, a Pydantic Model, a regular expression or a context-free grammar. The key is that  structured generation forbids the 'wrong' tokens from being generated.

Let’s take an extremely simple example. The LLM should generate a boolean, “true” or “false”. And nothing more. For the sake of illustration, let’s say that LLMs generate characters instead of tokens. So the first character is `"`, we can just skip the forward pass. For the second, we don’t need to sample from all possible characters. The LLM should just choose between `t` or `f`. 

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/outlines-core/graph.png"><br>
</p>

After that, regardless of the path we take, there is only one valid next character. If the LLM chose `t` as the first character, then it has to follow with `r`, `u` and `e`. And similarly if it chose `f` it follows with `a`, `l`, `s`, `e`. And the last `"` regardless of the path. There is of course more under the hood, for more in-depth we recommend this [dottxt blog](https://blog.dottxt.co/coalescence.html) and the [associated paper on arxiv](https://arxiv.org/abs/2307.09702).

### Why it’s important

It might not immediately be obvious how amazing structured generation can be. The first use-case many think of is “nice, now my LLM can return valid JSON, so I can treat it as an API and serialize/deserialize JSON reliably”. But that’s just scratching the surface. When you think about it, structure is everywhere, even in places where you least expect it like the [GSM8K benchmark](https://blog.dottxt.co/performance-gsm8k.html).

These are just a [few examples](https://dottxt-ai.github.io/outlines/cookbook/) of what structured generation enables:
- Generating synthetic data
- Extracting information from documents and images.
- Function calling/building agents
- Chain of Thought
- Or just making sure you LLM outputs a [valid tic-tac-toe board](https://x.com/dottxtai/status/1840826952577421646) or generating virtual worlds.

And, perhaps more surprising, it reduces the sensitivity of evaluations to the [specific prompt being used](https://huggingface.co/blog/evaluation-structured-outputs) and the [number of shots](https://blog.dottxt.co/prompt-efficiency.html). Apart from the amazing tricks that structure gives you, it’s also more performant. The dottxt blog has many good articles with performance benchmarks.

## Why rewrite in Rust? 🦀

### Speed

Probably the first thing that comes to your mind when you hear “rewrite in Rust” is performance. And yes, that’s the case for `outlines-core` as well. Several key parts are yet to be moved over to Rust, and despite that, we already see an [average 2x improvement](https://github.com/dottxt-ai/benchmarks) in compilation.

Before the Rust port, Outlines used Numba to accelerate the building of the index. While Numba is fast (the runtime performance is comparable to Rust), the JIT-compilation of the Numba functions added a source of latency during the first run, which was a source of frustration for many users. Using Rust means we can compile the index building functions ahead of time, adding no latency during the first run. While this was not important in a production context, it can make a huge difference during the experimentation phase!

### Safety and Reliability

One of the main reasons for rewriting Outlines in Rust is the emphasis on safety and reliability that Rust brings to the table. Rust's strong static typing and ownership model eliminates entire classes of bugs, such as null pointer dereferences and data races in concurrent code. This leads to more robust and secure software.

In the context of Outlines, safety is crucial. Structured generation often involves complex data structures and manipulations, especially when dealing with high-performance inference engines. By leveraging Rust's safety guarantees, we reduce the risk of runtime errors and undefined behaviors that can arise from memory mismanagement.

Additionally, Rust's compile-time checks encourage developers to write cleaner and more maintainable code. This improves the current codebase and makes future development more efficient. New contributors can onboard more quickly, and the code is easier to audit and verify for correctness.

### Separation of concerns

Outlines was designed to do more than providing the core algorithms for structured generation. Among other things, it includes integrations to other libraries like `transformers` which mean the library packs many dependencies. Separating the core algorithms from the Outlines library means that other libraries wishing to include structured generation can do so by importing a very lightweight library. So we can imagine in the near future libraries such as `transformers` and `llama-cpp-python` integrating structured generation directly. And the dottxt team can focus on the core algorithms.

### Portability

Most of LLM training is written in Python, but inference is slightly different. It happens on many different devices, on specialized servers and is written in a range of programming languages. This is why portability also matters for structured generation. By having the core functionality of `outlines` written in rust, we can now create bindings to other languages.

For example, this port makes the integration into the [text-generation-inference](https://github.com/huggingface/text-generation-inference) much smoother. TGI’s server logic is written in Rust, and we want to avoid having to call Python code as much as we possibly can. It also means libraries like `mistral.rs` or models implemented using [candle](https://github.com/huggingface/candle) can benefit from Outlines’s performance and capabilities. 

In the future we could explore bindings to JS/TS, allowing outlines to be used in transformers-js. Or potentially Swift bindings, making outlines natively usable on Apple devices. But for now the focus is going to be on the Python bindings, and continuing to make `outlines-core`’s feature set complete by expanding support for the JSON Schema specification.

## Contribute

Do you like working with structured generation, parsers, making LLMs output only valid JSON? Star the [library](https://github.com/dottxt-ai/outlines-core), tweet about it, join in and contribute! Share your work on Twitter, and with [dottxt’s](https://discord.com/invite/R9DSu34mGd) and Hugging Face's community.
