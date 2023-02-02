---
title: "Hugging Face Reads, Feb. 2021 - Long-range Transformers"
thumbnail: /blog/assets/14_long_range_transformers/EfficientTransformerTaxonomy.png
authors:
- user: VictorSanh
---

<figure>
  <img src="/blog/assets/14_long_range_transformers/EfficientTransformerTaxonomy.png" alt="Efficient Transformers taxonomy"/>
  <figcaption>Efficient Transformers taxonomy from Efficient Transformers: a Survey by Tay et al.</figcaption>
</figure>

# Hugging Face Reads, Feb. 2021 - Long-range Transformers

{blog_metadata}

{authors}

Co-written by Teven Le Scao, Patrick Von Platen, Suraj Patil, Yacine Jernite and Victor Sanh.

> Each month, we will choose a topic to focus on, reading a set of four papers recently published on the subject. We will then write a short blog post summarizing their findings and the common trends between them, and questions we had for follow-up work after reading them. The first topic for January 2021 was [Sparsity and Pruning](https://discuss.huggingface.co/t/hugging-face-reads-01-2021-sparsity-and-pruning/3144), in February 2021 we addressed Long-Range Attention in Transformers.

## Introduction

After the rise of large transformer models in 2018 and 2019, two trends have quickly emerged to bring their compute requirements down. First, conditional computation, quantization, distillation, and pruning have unlocked inference of large models in compute-constrained environments; we’ve already touched upon this in part in our [last reading group post](https://discuss.huggingface.co/t/hugging-face-reads-01-2021-sparsity-and-pruning/3144). The research community then moved to reduce the cost of pre-training.

In particular, one issue has been at the center of the efforts: the quadratic cost in memory and time of transformer models with regard to the sequence length. In order to allow efficient training of very large models, 2020 saw an onslaught of papers to address that bottleneck and scale transformers beyond the usual 512- or 1024- sequence lengths that were the default in NLP at the start of the year.

This topic has been a key part of our research discussions from the start, and our own Patrick Von Platen has already dedicated [a 4-part series to Reformer](https://huggingface.co/blog/reformer). In this reading group, rather than trying to cover every approach (there are so many!), we’ll focus on four main ideas:

* Custom attention patterns (with [Longformer](https://arxiv.org/abs/2004.05150))
* Recurrence (with [Compressive Transformer](https://arxiv.org/abs/1911.05507))
* Low-rank approximations (with [Linformer](https://arxiv.org/abs/2006.04768))
* Kernel approximations (with [Performer](https://arxiv.org/abs/2009.14794))

For exhaustive views of the subject, check out [Efficient Transfomers: A Survey](https://arxiv.org/abs/2009.06732) and [Long Range Arena](https://arxiv.org/abs/2011.04006).

## Summaries

### [Longformer - The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

Iz Beltagy, Matthew E. Peters, Arman Cohan

Longformer addresses the memory bottleneck of transformers by replacing conventional self-attention with a combination of windowed/local/sparse (cf. [Sparse Transformers (2019)](https://arxiv.org/abs/1904.10509)) attention and global attention that scales linearly with the sequence length. As opposed to previous long-range transformer models (e.g. [Transformer-XL (2019)](https://arxiv.org/abs/1901.02860), [Reformer (2020)](https://arxiv.org/abs/2001.04451), [Adaptive Attention Span (2019)](https://arxiv.org/abs/1905.07799)), Longformer’s self-attention layer is designed as a drop-in replacement for the standard self-attention, thus making it possible to leverage pre-trained checkpoints for further pre-training and/or fine-tuning on long sequence tasks.

The standard self-attention matrix (Figure a) scales quadratically with the input length:

<figure>
  <img src="/blog/assets/14_long_range_transformers/Longformer.png" alt="Longformer attention"/>
  <figcaption>Figure taken from Longformer</figcaption>
</figure>

Longformer uses different attention patterns for autoregressive language modeling, encoder pre-training & fine-tuning, and sequence-to-sequence tasks.
* For autoregressive language modeling, the strongest results are obtained by replacing causal self-attention (a la GPT2) with dilated windowed self-attention (Figure c). With \\(n\\) being the sequence length and \\(w\\) being the window length, this attention pattern reduces the memory consumption from \\(n^2\\) to \\(wn\\), which under the assumption that \\(w << n\\), scales linearly with the sequence length.
* For encoder pre-training, Longformer replaces the bi-directional self-attention (a la BERT) with a combination of local windowed and global bi-directional self-attention (Figure d). This reduces the memory consumption from \\(n^2\\) to \\(w n + g n\\) with \\(g\\) being the number of tokens that are attended to globally, which again scales linearly with the sequence length.
* For sequence-to-sequence models, only the encoder layers (a la BART) are replaced with a combination of local and global bi-directional self-attention (Figure d) because for most seq2seq tasks, only the encoder processes very large inputs (e.g. summarization). The memory consumption is thus reduced from \\(n_s^2+ n_s n_t +n_t^2\\) to \\(w n_s +gn_s +n_s n_t +n_t^2\\) with \\(n_s\\) and \\(n_t\\) being the source (encoder input) and target (decoder input) lengths respectively. For Longformer Encoder-Decoder to be efficient, it is assumed that \\(n_s\\) is much bigger than \\(n_t\\).

#### Main findings

* The authors proposed the dilated windowed self-attention (Figure c) and showed that it yields better results on language modeling compared to just windowed/sparse self-attention (Figure b). The window sizes are increased through the layers. This pattern further outperforms previous architectures (such as Transformer-XL, or adaptive span attention) on downstream benchmarks.
* Global attention allows the information to flow through the whole sequence and applying the global attention to task-motivated tokens (such as the tokens of the question in QA, CLS token for sentence classification) leads to stronger performance on downstream tasks. Using this global pattern, Longformer can be successfully applied to document-level NLP tasks in the transfer learning setting.
* Standard pre-trained models can be adapted to long-range inputs by simply replacing the standard self-attention with the long-range self-attention proposed in this paper and then fine-tuning on the downstream task. This avoids costly pre-training specific to long-range inputs.

#### Follow-up questions

* The increasing size (throughout the layers) of the dilated windowed self-attention echoes findings in computer vision on increasing the receptive field of stacked CNN. How do these two findings relate? What are the transposable learnings?
* Longformer’s Encoder-Decoder architecture works well for tasks that do not require a long target length (e.g. summarization). However, how would it work for long-range seq2seq tasks which require a long target length (e.g. document translation, speech recognition, etc.) especially considering the cross-attention layer of encoder-decoder’s models?
* In practice, the sliding window self-attention relies on many indexing operations to ensure a symmetric query-key weights matrix. Those operations are very slow on TPUs which highlights the question of the applicability of such patterns on other hardware.

### [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap

[Transformer-XL (2019)](https://arxiv.org/abs/1901.02860) showed that caching previously computed layer activations in a memory can boost performance on language modeling tasks (such as *enwik8*). Instead of just attending the current \\(n\\) input tokens, the model can also attend to the past \\(n_m\\) tokens, with \\(n_m\\) being the memory size of the model. Transformer-XL has a memory complexity of \\(O(n^2+ n n_m)\\), which shows that memory cost can increase significantly for very large \\(n_m\\). Hence, Transformer-XL has to eventually discard past activations from the memory when the number of cached activations gets larger than \\(n_m\\). Compressive Transformer addresses this problem by adding an additional compressed memory to efficiently cache past activations that would have otherwise eventually been discarded. This way the model can learn better long-range sequence dependencies having access to significantly more past activations.


<figure>
  <img src="/blog/assets/14_long_range_transformers/CompressiveTransformer.png" alt="Compressive Tranformer recurrence"/>
  <figcaption>Figure taken from Compressive Transfomer</figcaption>
</figure>

A compression factor \\(c\\) (equal to 3 in the illustration) is chosen to decide the rate at which past activations are compressed. The authors experiment with different compression functions \\(f_c\\) such as max/mean pooling (parameter-free) and 1D convolution (trainable layer). The compression function is trained with backpropagation through time or local auxiliary compression losses. In addition to the current input of length \\(n\\), the model attends to \\(n_m\\) cached activations in the regular memory and \\(n_{cm}\\) compressed memory activations allowing a long temporal dependency of  \\(l × (n_m + c n_{cm})\\), with \\(l\\) being the number of attention layers. This increases Transformer-XL’s range by additional \\(l × c × n_{cm}\\) tokens and the memory cost amounts to \\(O(n^2+ n n_m+ n n_{cm})\\). Experiments are conducted on Reinforcement learning, audio generation, and natural language processing. The authors also introduce a new long-range language modeling benchmark called [PG19](https://huggingface.co/datasets/pg19).

#### Main findings

* Compressive Transformer significantly outperforms the state-of-the-art perplexity on language modeling, namely on the enwik8 and WikiText-103 datasets. In particular, compressed memory plays a crucial role in modeling rare words occurring on long sequences.
* The authors show that the model learns to preserve salient information by increasingly attending the compressed memory instead of the regular memory, which goes against the trend of older memories being accessed less frequently.
* All compression functions (average pooling, max pooling, 1D convolution) yield similar results confirming that memory compression is an effective way to store past information.

#### Follow-up questions

* Compressive Transformer requires a special optimization schedule in which the effective batch size is progressively increased to avoid significant performance degradation for lower learning rates. This effect is not well understood and calls into more analysis.
* The Compressive Transformer has many more hyperparameters compared to a simple model like BERT or GPT2: the compression rate, the compression function and loss, the regular and compressed memory sizes, etc.  It is not clear whether those parameters generalize well across different tasks (other than language modeling) or similar to the learning rate, make the training also very brittle.
* It would be interesting to probe the regular memory and compressed memory to analyze what kind of information is memorized through the long sequences. Shedding light on the most salient pieces of information can inform methods such as [Funnel Transformer](https://arxiv.org/abs/2006.03236) which reduces the redundancy in maintaining a full-length token-level sequence.

### [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma

The goal is to reduce the complexity of the self-attention with respect to the sequence length \\(n\\)) from quadratic to linear. This paper makes the observation that the attention matrices are low rank (i.e. they don’t contain \\(n × n\\) worth of information) and explores the possibility of using high-dimensional data compression techniques to build more memory efficient transformers.

The theoretical foundations of the proposed approach are based on the Johnson-Lindenstrauss lemma. Let’s consider \\(m\\)) points in a high-dimensional space. We want to project them to a low-dimensional space while preserving the structure of the dataset (i.e. the mutual distances between points) with a margin of error \\(\varepsilon\\). The Johnson-Lindenstrauss lemma states we can choose a small dimension \\(k \sim 8 \log(m) / \varepsilon^2\\) and find a suitable projection into Rk in polynomial time by simply trying random orthogonal projections.

Linformer projects the sequence length into a smaller dimension by learning a low-rank decomposition of the attention context matrix. The matrix multiplication of the self-attention can be then cleverly re-written such that no matrix of size \\(n × n\\) needs to be ever computed and stored.

Standard transformer:

$$\text{Attention}(Q, K, V) = \text{softmax}(Q * K) * V$$

                  (n * h)	            (n * n)   (n * h)

Linformer:

$$\text{LinAttention}(Q, K, V) = \text{softmax}(Q * K * W^K) * W^V * V$$

                  (n * h)	            (n * d)   (d * n)   (n * h)

#### Main findings

* The self-attention matrix is low-rank which implies that most of its information can be recovered by its first few highest eigenvalues and can be approximated by a low-rank matrix.
* Lot of works focus on reducing the dimensionality of the hidden states. This paper shows that reducing the sequence length with learned projections can be a strong alternative while shrinking the memory complexity of the self-attention from quadratic to linear.
* Increasing the sequence length doesn’t affect the inference speed (time-clock) of Linformer, when transformers have a linear increase. Moreover, the convergence speed (number of updates) is not impacted by Linformer's self-attention.

<figure>
  <img src="/blog/assets/14_long_range_transformers/Linformer.png" alt="Linformer performance"/>
  <figcaption>Figure taken from Linformer</figcaption>
</figure>

#### Follow-up questions

* Even though the projections matrices are shared between layers, the approach presented here comes in contrast with the Johnson-Lindenstrauss that states that random orthogonal projections are sufficient (in polynomial time). Would random projections have worked here? This is reminiscent of Reformer which uses random projections in locally sensitive hashing to reduce the memory complexity of the self-attention.

### [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller

The goal is (again!) to reduce the complexity of the self-attention with respect to the sequence length \\(n\\)) from quadratic to linear. In contrast to other papers, the authors note that the sparsity and low-rankness priors of the self-attention may not hold in other modalities (speech, protein sequence modeling). Thus the paper explores methods to reduce the memory burden of the self-attention without any priors on the attention matrix.

The authors observe that if we could perform the matrix multiplication \\(K × V\\) through the softmax ( \\(\text{softmax}(Q × K) × V\\) ), we wouldn’t have to compute the \\(Q x K\\) matrix of size \\(n x n\\) which is the memory bottleneck. They use random feature maps (aka random projections) to approximate the softmax by:

$$\text{softmax}(Q * K) \sim Q’ * K’ = \phi(Q) * \phi(K)$$

, where \\(phi\\) is a non-linear suitable function. And then:

$$\text{Attention}(Q, K, V) \sim \phi(Q) * (\phi(K) * V)$$

Taking inspiration from machine learning papers from the early 2000s, the authors introduce **FAVOR+** (**F**ast **A**ttention **V**ia **O**rthogonal **R**andom positive (**+**) **F**eatures) a procedure to find unbiased or nearly-unbiased estimations of the self-attention matrix, with uniform convergence and low estimation variance.

#### Main findings

* The FAVOR+ procedure can be used to approximate self-attention matrices with high accuracy, without any priors on the form of the attention matrix, making it applicable as a drop-in replacement of standard self-attention and leading to strong performances in multiple applications and modalities.
* The very thorough mathematical investigation of how-to and not-to approximate softmax highlights the relevance of principled methods developed in the early 2000s even in the deep learning era.
* FAVOR+ can also be applied to efficiently model other kernelizable attention mechanisms beyond softmax.

#### Follow-up questions

* Even if the approximation of the attention mechanism is tight, small errors propagate through the transformer layers. This raises the question of the convergence and stability of fine-tuning a pre-trained network with FAVOR+ as an approximation of self-attention.
* The FAVOR+ algorithm is the combination of multiple components. It is not clear which of these components have the most empirical impact on the performance, especially in view of the variety of modalities considered in this work.

## Reading group discussion

The developments in pre-trained transformer-based language models for natural language understanding and generation are impressive. Making these systems efficient for production purposes has become a very active research area. This emphasizes that we still have much to learn and build both on the methodological and practical sides to enable efficient and general deep learning based systems, in particular for applications that require modeling long-range inputs.

The four papers above offer different ways to deal with the quadratic memory complexity of the self-attention mechanism, usually by reducing it to linear complexity. Linformer and Longformer both rely on the observation that the self-attention matrix does not contain \\(n × n\\) worth of information (the attention matrix is low-rank and sparse). Performer gives a principled method to approximate the softmax-attention kernel (and any kernelizable attention mechanisms beyond softmax). Compressive Transformer offers an orthogonal approach to model long range dependencies based on recurrence.

These different inductive biases have implications in terms of computational speed and generalization beyond the training setup. In particular, Linformer and Longformer lead to different trade-offs: Longformer explicitly designs the sparse attention patterns of the self-attention (fixed patterns) while Linformer learns the low-rank matrix factorization of the self-attention matrix. In our experiments, Longformer is less efficient than Linformer, and is currently highly dependent on implementation details. On the other hand, Linformer’s decomposition only works for fixed context length (fixed at training) and cannot generalize to longer sequences without specific adaptation. Moreover, it cannot cache previous activations which can be extremely useful in the generative setup. Interestingly, Performer is conceptually different: it learns to approximate the softmax attention kernel without relying on any sparsity or low-rank assumption. The question of how these inductive biases compare to each other for varying quantities of training data remains.

All these works highlight the importance of long-range inputs modeling in natural language. In the industry, it is common to encounter use-cases such as document translation, document classification or document summarization which require modeling very long sequences in an efficient and robust way. Recently, zero-shot examples priming (a la GPT3) has also emerged as a promising alternative to standard fine-tuning, and increasing the number of priming examples (and thus the context size) steadily increases the performance and robustness. Finally, it is common in other modalities such as speech or protein modeling to encounter long sequences beyond the standard 512 time steps.

Modeling long inputs is not antithetical to modeling short inputs but instead should be thought from the perspective of a continuum from shorter to longer sequences. [Shortformer](https://arxiv.org/abs/2012.15832), Longformer and BERT provide evidence that training the model on short sequences and gradually increasing sequence lengths lead to an accelerated training and stronger downstream performance. This observation is coherent with the intuition that the long-range dependencies acquired when little data is available can rely on spurious correlations instead of robust language understanding. This echoes some experiments Teven Le Scao has run on language modeling: LSTMs are stronger learners in the low data regime compared to transformers and give better perplexities on small-scale language modeling benchmarks such as Penn Treebank.

From a practical point of view, the question of positional embeddings is also a crucial methodological aspect with computational efficiency trade-offs. Relative positional embeddings (introduced in Transformer-XL and used in Compressive Transformers) are appealing because they can easily be extended to yet-unseen sequence lengths, but at the same time, relative positional embeddings are computationally expensive. On the other side, absolute positional embeddings (used in Longformer and Linformer) are less flexible for sequences longer than the ones seen during training, but are computationally more efficient. Interestingly, [Shortformer](https://arxiv.org/abs/2012.15832) introduces a simple alternative by adding the positional information to the queries and keys of the self-attention mechanism instead of adding it to the token embeddings. The method is called position-infused attention and is shown to be very efficient while producing strong results.

## @Hugging Face 🤗: Long-range modeling

The Longformer implementation and the associated open-source checkpoints are available through the Transformers library and the [model hub](https://huggingface.co/models?search=longformer). Performer and Big Bird, which is a long-range model based on sparse attention, are currently in the works as part of our [call for models](https://twitter.com/huggingface/status/1359903233976762368), an effort involving the community in order to promote open-source contributions. We would be pumped to hear from you if you’ve wondered how to contribute to `transformers` but did not know where to start!

For further reading, we recommend checking Patrick Platen’s blog on [Reformer](https://arxiv.org/abs/2001.04451), Teven Le Scao’s post on [Johnson-Lindenstrauss approximation](https://tevenlescao.github.io/blog/fastpages/jupyter/2020/06/18/JL-Lemma-+-Linformer.html), [Efficient Transfomers: A Survey](https://arxiv.org/abs/2009.06732), and [Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006).

Next month, we'll cover self-training methods and applications. See you in March!
