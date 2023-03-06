---
title: "Introduction to Switch Transformers"
thumbnail: /blog/assets/119_switch_transformers/thumbnail.png
---

<h1>
  	Introduction to Switch Transformers"
</h1>

<div class="blog-metadata">
    <small>Published December 9, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/switch_transformers.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/arthurzucker">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648631395099-62441cb7456803e95009a08f.png?w=200&h=200&f=face" title="Avatar">
        <div class="bfc">
            <code>arthurzucker</code>
            <span class="fullname">Arthur Zucker</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/ybelkada">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648631057413-noauth.png?w=200&h=200&f=face" title="Avatar">
        <div class="bfc">
            <code>ybelkada</code>
            <span class="fullname">Younes Belkada</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4?usp=sharing_">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# MoEs make their way to the Hugging Face ecosystem

If you are following the recent advances in NLP, you have probably heard about the sparse architecture called Mixture of Experts (MoE) models. Research on MoEs has been around for a few years, but the advance of huge clusters allowed the democratization of the architecture.

In terms of software adoption, we have seen several MoE implementations, for example, in MetaAI’s [`fairseq`](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm) library and Microsoft’s [`DeepSpeed`](https://arxiv.org/abs/2201.05596) library.


With the publication of several research papers and the adoption of MoEs in increasingly more tasks, this architecture has gained popularity in recent months.

Google open-sourced the largest MoE-based models last year in the paper [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). The paper released a trillion parameter model, Switch-c-2048 (3.1 Terabytes!).
The model is now publicly available on [HuggingFace Hub](https://huggingface.co/models?search=switch), making it the biggest model on the platform.


In an effort to democratize its usage and centralize the research efforts around common implementations, we are providing a base implementation of `SwitchTransformer` with the top-1 routing mechanism. We hope that the community will be able to build upon it and easily share their results.


In this post, we aim to provide an overview of the technical specifications of the Switch Transformer architecture. We will also show you how to train and evaluate your first Switch Transformer model using  🤗 Transformers.

Let's get started! 

## Mixture of Experts (MoE) models in a few words

MoEs have been widely democratized by Shazeer et al. in the paper [Outrageously Large Neural Networks: the Sparsely-Gated Mixture of Experts Layer](https://openreview.net/pdf?id=B1ckMDqlg). The primary motivation behind the creation of these layers is the need to increase the number of parameters (thereby model capacity) to improve predictive performance. In the paper, the authors obtain more than 1000x improvements in model capacity with a small loss in computational efficiency. 

The core idea is to replace a single model block (such as the [transformer](https://arxiv.org/abs/1706.03762) block) that can take $N$ inputs by the so-called "expert" blocks, and select which expert block gets to process what inputs. This way, you are still only using $N$ weights, but the number of weights that you can be using is `N x num_experts` instead of `N`.

| ![MoE figure](/assets/119_switch_transformers/moe.png) | 
|:--:|
| <b> The first MoE layer as defined by Shazeer et al. from the paper ["Outrageously Large Neural Networks: the Sparsely-Gated Mixture of Experts Layer"](https://openreview.net/pdf?id=B1ckMDqlg) | Image source: https://openreview.net/pdf?id=B1ckMDqlg </b>|

Expanding more, an MoE layer consists of a set of $n$ "expert networks" and a "gating network" (Switch Transformers refer this layer as the "router"). The gating network selects a sparse combination of experts to process each input. Although training an MoE-based model will result in a larger model (more parameters), [experiments in Switch Transformers](https://arxiv.org/abs/2101.03961) showed that it is possible to drastically improve model performance for a given computational budget.

According to the authors, the [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) was trained with up to 7x speedup in pre-training with the identical computational resources needed to train a T5-base and T5-large model.

## Switch Transformers in a nutshell

Switch Transformers follow the [T5 architecture](https://arxiv.org/abs/1910.10683). Our implementation closely follows that of [T5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py), with the following exception. The [`DenseActDense` layer](https://github.com/huggingface/transformers/blob/8fb4d0e4b46282d96386c229b9fb18bf7c80c25a/src/transformers/models/t5/modeling_t5.py#L281) is replaced with a [`SparseMLP` layer](https://github.com/huggingface/transformers/blob/8fb4d0e4b46282d96386c229b9fb18bf7c80c25a/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L300).

In Switch Transformers, the [`SwitchTransformersDenseActDense` layer](https://github.com/huggingface/transformers/blob/8fb4d0e4b46282d96386c229b9fb18bf7c80c25a/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L265) in the attention mechanism is replaced with a [`SwitchTransformersSparseMLP` layer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L300). The `SparseMLP` layer is composed of a router, and a list of `experts`, where each `expert` is a [`DenseActDense` module](https://github.com/huggingface/transformers/blob/8fb4d0e4b46282d96386c229b9fb18bf7c80c25a/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L265). Instead of passing each token to the single `DenseActDense`, the tokens go through the router which decides which `DenseActDense` (or expert) from the list of experts, should process the token. The routing mechanism is called `top-1` in `SwitchTransformers` as a token is only routed to a single expert, with the highest probability. 

One hyperparameter here is the `expert_capacity`, which defines the maximum number of tokens that can be processed by a single expert. Once the capacity is reached on an expert, any new tokens that are routed will be ignored and will reach the next hidden states via only the residual connection. The following figure will present the various cases regarding this idea of expert capacity.

| ![MoE figure](/assets/119_switch_transformers/routing.png) | 
|:--:|
| <b>Image source: https://arxiv.org/abs/2101.03961 </b>|

As seen in the figure above, Expert 1 will ignore the blue token as it has already reached its maximum capacity (even though the token was routed there).  This will result in using the hidden state in the next stage as it is.

This phenomenon leads to an important question. **How can we ensure the diversity of the routing mechanism?** That is to make sure that the distribution of routed experts is uniform. Some recent studies have shown that learning such a routing mechanism encourages token clustering around expert centroids. A poor expert routing strategy can cause certain experts to be under-trained, leading to an individual expert being under or over-specialized.

Therefore, it is important to revisit the routing algorithm to tackle this issue. Recently, Google developed the so-called "Expert Choice (EC)" algorithm. The top-k tokens are assigned to the experts with a pre-determined buffer capacity rather than having tokens chosen for the top-k experts. This approach significantly improves training efficiency and downstream performance. Besides, it even guarantees load-balancing and allows for a variable number of experts for each token. In an 8B/64E (8 billion activated parameters, 64 experts) model, EC routing accelerates training convergence more than two times compared to the top-1 and top-2 gating mechanisms. Read more about the method in the [original blog post](https://ai.googleblog.com/2022/11/mixture-of-experts-with-expert-choice.html).


## How good are these models?

Switch Transformer checkpoints that have been publicly released were pre-trained on the [C4 dataset](https://huggingface.co/datasets/c4), with [masked language modeling](https://huggingface.co/tasks/fill-mask). Therefore they are not ready to use in a sequence-to-sequence formulation such as [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5), for instance. One needs to fine-tune the models to a task of preference before using it. However, the authors published some results of the fine-tuned checkpoints and the results seem to be better than classic T5 models. The following results were all taken from the original [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) paper.

| Model | GLUE | SQuAD | SuperGLUE | Winogrande (XL) |
| :---: | :---: | :---: | :---: | :---: |
| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 |
| Switch-Base | 86.7 | 87.2 | 79.5 | 73.3 |
| T5-Large | 87.8 | 88.1 | 82.7 | 79.1 |
| Switch-Large | 88.5 | 88.6 | 84.7 | 83.0 |

| Model | XSum | ANLI (R3) | ARC Easy | ARC Chal. |
| :---: | :---: | :---: | :---: | :---: |
| T5-Base | 18.7 | 51.8 | 56.7 | 35.5 |
| Switch-Base | 20.3 | 54.0 | 61.3 | 32.8 |
| T5-Large | 20.9 | 56.6 | 68.8 | 35.5 |
| Switch-Large | 22.3 | 58.6 | 66.0 | 35.5 |

| Model | CB Web QA | CB Natural QA | CB Trivia QA |
| :---: | :---: | :---: | :---: |
| T5-Base | 26.6 | 25.8 | 24.5 |
| Switch-Base | 27.4 | 26.8 | 30.7 |
| T5-Large | 27.7 | 27.6 | 29.5 |
| Switch-Large | 31.3 | 29.5 | 36.9 |


## How to use it in Hugging Face ecosystem

With the introduction of Switch Transformers in the Hugging Face ecosystem, users and practitioners can train and deploy their MoE models with just a few lines of code! 

TIP: In order to use big models on low ressource machines, make sure to install `accelerate`, it will take care of balancing the load on GPU and CPU, which is very convenient as `MoE` models are very heavy! 

```python
# pip install accelerate
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-128")
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-128", device_map="auto")

input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
>>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>
```

If you are interested in training your first MoE-based model, please check out this [Colab Notebook](https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4#scrollTo=xsgNPx4FrtH_). It shows you how to fine-tune a Switch Transformer model for summarization and how you can easily share the fine-tuned model on the Hugging Face Hub.

## Conclusion


### Further reading

- Review of MoE models in Deep Learning: https://arxiv.org/pdf/2209.01667.pdf
- Switch Transformers paper: https://arxiv.org/pdf/2101.03961.pdf
- Microsoft blogpost: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/scaling-speech-language-and-vision-models-with-mixture-of/ba-p/3295750
- More MoE-related papers and code: https://github.com/XueFuzhao/awesome-mixture-of-experts
