---
title: "A failed experiment: Infini-Attention, and why we should keep trying?" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: your_hf_user
- user: your_coauthor
---


The context length of language models is one of the central attributes besides the model’s performance. Especially since the emergence of in-context learning it has become more and more important to add relevant information to the model’s input. Thus the context length rapidly increased from paragraph (512 tokens of BERT/GPT-1) to pages (1024/2048 of GPT-2 and GPT-3 respecively) to books (128k of Claude) all the way to collections of books (1-10M tokens of Gemini). However extending standard attention to such length remains challenging.

Even with ring attention, the number of GPUs required to train an Llama 3 8B on a 1-million-token context length with a batch size of 1 is 512 GPUs [link]. As scaling law has shown [link], there is a strong correlation between model size and its downstream performance, that means the bigger the model, the better (of course, both models should be both well-trained). So we not only want a 1m context length, but we want a 1m context length on the biggest model (e.g., Llama 3 8B 400B). And there are only a few companies in existence that have the resources to do so.

Motivated by this, we explore an alternative approach to standard attention: infini-attention. The paper was released by researchers from Google in April 2024 [link]. Instead of computing attention scores between every word, Infini attention divides the sequence into segments, compresses earlier segments into a fixed buffer, and allows the next segment to retrieve memory from the earlier segments while limiting attention scores to words within the current segment.  A key advantage is that it uses the same query within a segment to access information from both its own segment and the compressed memory. This enables us to cheaply extend the context length for a pretrained model. 

Since it updates the compressed memory on the go, and success is not guaranteed, but if it all works out we can have infinite context length, as it only keeps a single buffer for all the memory of earlier segments. However, by definition, compression has limits on the amount of information it can effectively compress. The question now is: how good is the compressed memory if it works?

However, conceptually understanding a new method can be relatively easy compared to actually making it work, and this is rarely shared publicly (we spent 90% of our time debugging the convergence issue). Motivated by this, we want to share how we reproduced the Infini-attention paper, what motivated us throughout the debugging process, and how hard it is to make these things work. 

Following the recent release of Llama 3 8B, which has a context length limit of 8k tokens, we sought to extend this length to 1m tokens. In this blog post, we will start by explaining how Infini Attention works. We’ll then outline our reproduction principles and describe our initial small-scale experiment. We will discuss the challenges we faced, how we addressed them, and conclude with a summary of our findings and other ideas we explored. If you’re interested in testing our trained checkpoint, you can find it in the following repo [link] (note that, as the technique doesn’t work well enough, so we did not invest much effort in cleaning up the code)

# Section 1: Reproduction Principles

We found the following rules being helpful when implementing a new method and use it as guiding principles for a lot of our work:

+ Start with the smallest model size that provides good signals.
+ Always train a solid baseline to measure progress.
+ To determine if a factor improves performance, train two identical models except for the difference in the factor being tested

With these principles in mind, let's dive into how Infini Attention actually works. Understanding the mechanics will be crucial as we move forward with our experiments.


# Section 2: How does infini attention works


- Step 1: Split the input sequence into smaller, fixed-size chunks called segments.
- Step 2: Calculate the standard causal dot-product attention within each segment.
- Step 3: Pull out relevant information from the compressive memory using the current segment’s query vector.
- Step 4: Combine the local context (from the current segment) with the long-term context (retrieved from the compressive memory) to generate the final output. So that we can have both short-term and long-term contexts are considered in the attention output.
- Step 5: Update the compressive memory by adding the key-value states from the current segment. So that we accumulate the context over time.
- Step 6: As we move from one segment to the next, we discard the previous segment's attention states and pass along the updated compressed memory to the next segment.

Now that we've got a handle on the theory, it's time to roll up our sleeves and get our hands dirty with some actual experiments. We started small to get quick feedback and iterate rapidly.

# Section 3: Start with experiments on a small scale

For rapid experiment feedback, we initially trained Infini attention on a 200m llama using nanotron [link] on Fineweb dataset [link]. We used a batch size of 2 million tokens, a context length of 256, gradient clipping of 1, and weight decay of 0.1, the first 5,000 iterations were a linear warmup, while the remaining steps were cosine decay, with a learning rate of 3e-5. We found that it somewhat works, and if you look at these sample generations, you can see that Infini attention generates content related to the earlier segment.
