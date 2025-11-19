---
title: "Continuous batching from first principles" 
thumbnail: /blog/assets/continuous_batching/thumbnail.png
authors:
- user: ror
- user: ArthurZ
- user: mcpotato
---

# Continuous batching

![Title card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/banner.png)

## Introduction

If you've ever used Qwen, Claude, or any other AI chatbot, you've probably noticed something: it takes a while for the first word of the response to appear, and then words appear one-by-one on your screen with (hopefully) a regular and fast-paced frequency. That's because at the heart of it, all LLMs are just fancy next token predictors. They have to ingest your initial prompt to spit out one token, and then ingest that one as well and repeat the process until they feel like they have generated enough.

This generation process is computationally expensive: it requires processing billions of parameters for each token generated. To make these models practical for real-world applications, particularly when serving many users simultaneously, researchers and engineers have developed a range of efficient inference techniques.  
One of the most impactful optimizations is **continuous batching**, which is the focus of this article. To understand how continuous batching works and why it's so effective, we'll need to build up from the fundamentals of how LLMs process tokens.

## Attention

In order to understand continuous batching, it’s useful to have a clear understanding of the attention mechanism. In a Llama-like LLM, the neural network processes text by breaking it down into pieces that we call tokens, which are represented by vectors. For simplicity's sake, let's pretend one token corresponds to one word. For each token, the network computes a prediction of what the next token should be. It does this by applying a sequence of operations to each token.

Most operations in the network are **token-wise**: each token is processed independently, and the output for a given token depends only on that token's content, not on any other tokens in the sequence. Operations like this include layer normalization or matrix multiplication. However, to create connections between words in a sentence, we need some operations where tokens can influence each other. 

This is where attention comes in. **Attention layers are the only place where different tokens interact with each other**—something that even seasoned ML practitioners sometimes overlook. Understanding how a network connects tokens together means understanding attention.

Let's see how this works in practice. To explain the basics of attention, we'll focus on the case where we have one input prompt. This is the same as saying batch size is 1.

Consider the initial prompt "I am sure this project". It is tokenized as 7 tokens: `[<bos>, I, am, sure, this, pro, ject]`. The `<bos>` token is a special token that is added at the start of the prompt (BoS stands for Beginning of Sequence).  
As a broad-stroke picture, attention can be represented this way:

![attention.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/attention.png)


The incoming tokens, which we will call $x$, represent a tensor of shape $\left[1, n , d \right]$ because batch size is $1$, $n$ is the number of tokens and $d$ is the hidden dimension. 

Those tokens $x$ are then projected by three matrices: the query projection $W_q$,  the key projection $W_k$ and the value projection $W_v$. This produces three tensors $Q$, $K$ and $V$, all of shape $\left[1, n , A \right]$ where $A$ is the dimension of an attention head. We call them respectively the **query, key and value states.** 

Next, tensors $Q$ and $K$ are multiplied together to measure similarity between tokens, producing a tensor of shape $\left[ 1, n , n \right]$. This is why we say that attention has quadratic complexity in sequence length. Computing $QK^T$ requires $\mathcal{O} \left( n^2 d \right)$ operations, so the cost is a square of $n$ the sequence length.

We then apply a boolean **attention mask** to $QK^T$ to control which tokens can interact. In the figure above, the attention mask is a **causal mask**, meaning each token only interacts with tokens that came before it. This follows the intuition that a cause must come before its consequence—hence the name causal mask. The attention mask is crucial because it dictates all token interactions in the network. **Set all attention mask values to False and no token will ever interact with another in the whole network.** We'll examine attention masks more closely in a few paragraphs.

Finally, after applying the attention mask, we take a row-wise softmax and multiply the result by the value projection $V$ to get the output of one attention head, of shape $\left[ 1, n , A \right]$.

We are going to use a lot of attention visualization in this post, so to simplify things, we are going to condense the figure above just a bit.

**Why this matters:** In continuous batching, $Q$, $K$, and $V$ can have different numbers of tokens because we're processing different stages together (prefill and decode). Let's say $Q$ has shape $\left[1, n_Q , A \right]$, $K$ has shape $\left[ 1, n_K , A \right]$, and $V$ has shape $\left[ 1, n_V , A \right]$.

The attention scores $QK^T$ then have shape $\left[ 1, n_Q , n_K \right]$, and the attention mask has the same shape since it's applied point-wise to the scores.

After applying the attention mask and row-wise softmax, we multiply by $V$. Since we're multiplying a matrix of shape $\left[ 1, n_Q , n_K \right]$ by one of shape $\left[ 1, n_V , A \right]$, the inner dimensions must match: $n_K = n_V$. This means $V$ and $K$ always have the same length, so we can simplify our visualizations by only showing $K$.

Don't worry if this seems abstract—the figures will make it concrete.


Furthermore, since we know that the attention mask is applied to $QK^T$, we know they have the same shape. Instead of representing the attention scores, we will represent the attention mask in its place.
Finally, since $Q$, $K$ and $V$ are direct projections of $x$, no need to represent $x$. This gives the simplified figure where we only represent $Q$, $K$ and the attention mask:

![simple_attention.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/simple_attention.png)

This representation also underlines how we can **read an attention mask.** 

We read the mask row-by-row, which is the same as reading token-by-token: each row corresponds to one token's attention computation. A **green square** at position (row i, column j) means `True`—token j can influence token i. A **white square** means `False`—no interaction allowed.

For example, look at the third row for token "*am*". The "*I*" column is green, so "*I*" influences the computation of "*am*". The "*pro*" column is white, so "*pro*" doesn't influence "*am*" . This is causal masking at work: future tokens can't affect past ones.

The last layer of the model outputs a token prediction for each input token. In our context, generating the continuation of a single prompt, we only care about the next token prediction from the last token. The last token is "*ject*" in the figure above, and the associated predicton is "*will*".

To continue generation, we begin a new forward pass, which would naively look like this:

![naive_generate.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/naive_generate.png)

To compute the attention scores of the new token, we still need the key and value projections of the previous tokens. So we need to repeat the matrix multiplication of the old tokens (in grey in the figure above) with $W_k$ and $W_v$ to retrieve a result that was already computed once before. In other terms, we are wasting compute. Let's see how we can avoid that.

## KV-cache

Right off the bat, we notice that the last token does not impact the attention calculation of the other tokens:

![cant_see_me.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/cant_see_me.png)

This follows the idea of the causal mask: since "*will*" comes after all previous tokens, it does not change their attention calculation.
For text, causal attention is by far the most common. We will focus on that case from now on. But non-causal attention also exists, especially when dealing with images.
Considering we only need the next-token prediction for the "*will*" token, we can simplify the attention mechanism by only computing the output for this token.

Moreover, we already computed the $K$ and $V$ states for the tokens "*\<bos\>*", … ,  "*ject*" during the previous forward pass: if they have been stored, we do not need to recompute them again. This is the **KV cache**: the list of key and value states created during generation. It essentially allows one to reduce the compute cost of generating token $n+1$ from $\mathcal{O} \left( n^2 \right)$ to $\mathcal{O} \left( n \right)$ by avoiding recomputation of key and value projections, while paying a memory cost of  $\mathcal{O} \left( n \right)$.

![kv_cache.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/kv_cache.png)

In the figure above, only the tokens in white are computed: instead of computing the keys and values for 8 tokens, we compute them for 1. You can see that through KV caching, a lot of compute is saved.

Let's be a bit more specific around the cache size, because it's a good opportunity to tour the shapes present in our model. For a model with $\mathcal L$ attention layers and $H$ attention heads with head dimension $A$, the total cache size needed to store one token will be $2 *\mathcal L * AH$ with a factor of $2$ to account for both $K$ and $V$.  
For instance, Llama-2-7B with $\mathcal{L}=32$ layers, $H=32$ heads, and $A=128$ requires $2 \times 32 \times 128 = 8,192$ values per token per layer. With `float16` precision, this takes $2AH \times 2$ bytes $= 16$ KB in memory.

KV caching is useful when we have one input token, which is a stage we call **decoding**. But it can also be useful in the **prefill** stage, when we process the initial prompt and have many input tokens. Especially when there are large initial prompts that don't fit in GPU memory all at once.

## Chunked prefill

Up till now, we have looked at an example of prefill where we have $n=7$ tokens, but in practice initial prompts can be much longer. For instance, when using Cursor, you can add your repository as context before the prompt: this increases the prompt size by a lot. In such cases, the memory needed to store the activations for $n$ tokens can be larger than the available memory on the GPU. Thus we cannot perform prefill in a single forward pass: we have to split the prefill in chunks. This is called **chunked prefill**, and it's going to be one of the components needed to enable efficient inference.

Let's pretend that the available memory is very constrained, and that we can only pass $m=4$ tokens per forward pass. If we have an initial prompt with $n = 7$ tokens, we need to split it in $\lceil n /m \rceil = 2$ chunks (rounding up 7/4 = 1.75 to 2). We illustrate the example below using the same $n$ and $m$ notations:

![chunked prefill.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/chunked_prefill.png)

We can do that thanks to the KV cache. We store the KV states during the first prefill split, and during the second prefill split, we concatenate them left of the new KV states. We also adapt the attention mask accordingly. Visually, it looks like we split the non-chunked prefill in the middle.

The key insight: cached KV states let us process the prompt incrementally without losing information.

Although we showed here an example where we split the prefill into 2 chunks, chunked prefill can be used to split the prefill in any way we want, adapting flexibly to memory constraints.  
You should by now be equipped with the tools to understand Continuous Batching.

## Continuous batching

So far, we have only treated the case of batch size one, i.e. we only generate tokens for one prompt at a time. But in the context of evaluation or model serving, we want to generate tokens for a large number of prompts. To increase the **throughput**, which is the number of generated tokens divided by the time it took to generate them, the best course of action is to generate tokens for a batch of several prompts.

To batch prompts together, the naive way is to add an axis to both input tensors, which will be the batch axis. This way we can pass two prompts and two attention masks, one for each. However, this comes with a constraint on the shape of the inputs: we need all prompts to have the same length (since tensors must be rectangular). To achieve this, we usually add padding on the left so the new token prediction always comes from the rightmost token. We also modify the attention mask of each prompt accordingly. This is shown below:

![padding.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/padding.png)

where the padding tokens `<pad>` are coloured in orange. Then we can perform the forward pass as we used to, with the added dimension of the batch size. This is called **batched generation**—efficient for same-length prompts, but wasteful when lengths vary. It is illustrated below:

![batched_generation.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/batched_generation.png)

where `<eos>` means "End Of Sequence", this is a special token to indicate the model has reached the end of generation.

The drawback of batched generation is that if one prompt finishes generation before the other by generating an `<eos>` token, any further generated token is useless. And this goes on until the longest request of the batch finishes. Of course, we can remove the prompts that have reached an `<eos>` token from the batch and save some compute and memory, but saving resources is not the goal here: throughput is. 

Instead of just removing the finished prompt from the batch, we can replace it with a prompt that's waiting for generation. We will call this **dynamic scheduling**, but it is also called dynamic batching. Dynamic scheduling is great to maintain throughput while ensuring any token generated by a forward pass is relevant. But because of the way we batched prompts together, it has a major drawback: we need a lot of padding when swapping prompts.

![dynamic_batching.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/dynamic_batching.png)

The problem becomes even worse when batch size increases and initial prompts are long. The padding cost grows quadratically with both batch size and prompt length. If we have a batch of $B$ prompts that are in decoding phase and one finishes, dynamically introducing a prompt of $n$ initial tokens in the batch requires $(n-1)(B-1)$ padding tokens. For instance, with $B=8$ and $n=100$, we'd need $99 \times 7 = 693$ padding tokens!

Furthermore, practical optimizations like CUDA graphs or `torch.compile` require static tensor shapes. This forces us to pad all prompts to a fixed maximum length, dramatically increasing the padding waste. 

At this point, our main problem is padding, which is a consequence of the axis we added to batch sentences together. Thus, the ideal would be to get rid of this axis entirely—a radical rethinking of batching. If we do so, the only way to batch prompts together is to concatenate them:

![concatenate.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/concatenate.png)

 
But we don't want tokens from prompt 0 to interact with the tokens of prompt 1! Luckily for us, we have a way to control how tokens interact with one another: the attention mask. How we do this is displayed below:

![ragged_batching.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/ragged_batching.png)

Although we use different tints of green to illustrate the different parts of the attention mask, this is still a boolean mask with only greens for `True` and white for `False`. 
This way of batching prompts together is called **ragged batching** (because sequence lengths are 'ragged' or uneven), and it offers the benefit of added throughput without introducing the need for padding tokens.

In the figure above, we batch two full prompts together, although there is no limit to how many prompts can be batched together using ragged batching. The only limit is $m$, the number of tokens we can fit in a batch, with $m$ depending on the available memory on the GPU. 

For instance, here is one batching strategy used in practice to maximize throughput:
- We try to always reach $m$ tokens per batch
- We first add all the prompts in decoding phase to the batch, each accounting for 1 token
- We fill the remaining space with prefill phase prompts, relying on the flexibility of chunked prefill to exactly fill the batch as desired

Furthermore, we can also use dynamic scheduling to remove finished prompts from the batch whenever they finish. This combination of ragged batching and dynamic scheduling is called **continuous batching**—and it's the technique that powers modern LLM serving systems.

![continuous_batching.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/continuous_batching.png)

## Conclusion

Continuous batching combines three key techniques to maximize throughput in LLM serving:
1. **KV caching** to avoid recomputing past token representations
2. **Chunked prefill** to handle variable-length prompts within memory constraints  
3. **Ragged batching with dynamic scheduling** to eliminate padding waste and keep the GPU fully utilized

By removing the batch dimension and using attention masks to control token interactions, continuous batching allows mixing prefill and decode phases in the same batch, dramatically improving efficiency for serving multiple requests. This is why services like ChatGPT can handle thousands of concurrent users efficiently. 

In the next article in this series, we'll explore efficient KV cache management. If you'd like to see a deep dive on other continuous batching topics, please let us know in the comments!

*Acknowledgement: thanks to Arthur Zucker for producing the initial concept for the figures used in this article, and providing helpful reviews. And equal thanks to Luc Georges for the very thorough and detailed reviews throughout.*
