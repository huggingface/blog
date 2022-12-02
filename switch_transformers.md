---
title: "Switch Transformers"
thumbnail: /blog/assets/119_switch_transformers/thumbnail.png
---

<h1>
  	Switch Transformers
</h1>

<div class="blog-metadata">
    <small>Published April 25, 2022.</small>
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
    <a href="/younesbelkada">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1648631057413-noauth.png?w=200&h=200&f=face" title="Avatar">
        <div class="bfc">
            <code>younesbelkada</code>
            <span class="fullname">Younes Belkada</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4#scrollTo=xsgNPx4FrtH_">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


Mixture of Experts (MoE) -based models make their way to the Hugging Face ecosystem

If you are following the recent advances in NLP, you have probably heard about the sparse architecture called Mixture of Experts (MoE) models. Research on MoEs has been around for a few years, but the advance of huge clusters allowed the democratization of the architecture. In


In terms of software adoption, we have seen several MoE implementations, for example in MetaAI’s `fairseq` library, and Microsoft’s `DeepSpeed` library, which supports MoE training.

With the publication of several research papers and the adoption of Mixture of Experts in increasingly more tasks, this now efficient architecture has gained popularity in the recent months.

Google open-sourced the largest MoE models last year, in the paper “Switch Transformers: Scaling to Trillion Parameters models with simple and efficient sparsity”. Together with the paper, a trillion parameter model, Switch-c-2048 (3.1 Terabytes !) was released.
The model is now publicly available on Hugging Face Hub, making it the biggest model available.


In an effort to democratize its usage, as well as centralized research efforts, we are providing a base implementation of the `SwitchTransformer`, with the top-1 routing mechanism.


Let’s dive into the technical specifications of this architecture and how to train and evaluate your first MoE model using Huggin Face `transformers`! Let’s get started ! :hugs: :party:

Switch Transformers in a nutshell
In early concepts, the experts defined an entire neural network and the MoE was similar to ensemble methods.

Switch Transformers follows the T5 architecture, and the implementation is heavily based on our T5 code, with the exception of the `DenseActDense`, which can be replaced with `SparseMLP`.

In Switch Transformers, the `SwitchTransformersDenseActDense` layer in the attention mechanism is replaced with a `SwitchTransformersSparseMLP` layer. The `SparseMLP` layer is composed of a router, and a list of `experts`, where each expert is as `DenseActDense` module. Instead of passing each token to the single `DensActDense`, the tokens go through the router which decides which `DenseActDense` (or expert) from the list of experts, should process the token. The routing mechanism is called `top-1` in `SwitchTransformers` as a token is only routed to a single expert, with the highest probability. One hyperparameter is the `expert_capacity` which defines the maximum number of tokens that can be processed by a single expert. Once the capacity is reached on an expert, any new tokens that are routed will be ignored and will reach the next hidden states via only the residual connection. The following figures will present the various cases.

![MoE figure](/assets/119_switch_transformers/thumbnail.png)

As it can be seen on the figure above, the blue token is going to be ignored by the expert 1, even though it has been routed there since the expert has already reached its maximum expert capacity. This will result in using the hidden state on the next stage as it is.

How to ensure the diversity of the routing mechanism, i.e, make sure that the distribution of routed experts is uniform. Some recent studies have shown that learning such a routing mechanism encourages token clustering around expert centroids. Also a poor expert routing strategy can cause certain experts to be under-trained, leading to an expert being under or over-specialized.

Therefore, to tackle this issue it is important to revisit the routing algorithm. Recently, Google developed the so-called Expert Choice (EC) algorithm. The top-k tokens are assigned the experts with a predetermined buffer capacity rather than having tokens choose the top-k experts. This approach achieves significant improvements in training efficiency and downstream performance while guaranteeing even load balancing and allowing a variable number of experts for each token. In an 8B/64E (8 billion activated parameters, 64 experts) model, EC routing accelerates training convergence by more than two times when compared to the top-1 and top-2 gating mechanisms. Read more about the method in the [original blogpost](https://ai.googleblog.com/2022/11/mixture-of-experts-with-expert-choice.html)

Microsoft introduced X-MoE, which consists of routers that use a learnable temperature, by estimating the routing scores on a low-dimensional hypersphere.
How good are these models?
The models that have been publicly released are pre-trained on C4 dataset, on masked language modeling task. Therefore they are not ready to use as Flan-T5 for instance. One needs to fine-tune the models to any task before using it. However, authors published some results of the fine-tuned checkpoints and the results seem to be better than classic T5 models.

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


How to use it in Hugging Face ecosystem
With the introduction of Switch Transformers in the Hugging Face ecosystem, we let users and practitioners train and deploy their first MoE models! Let’s see how to do so with a few lines of code.
Together with the Trillion parameter model, google has released a set of smaller models so that anyone can experiment with this architecture, check them out here: https://huggingface.co/models?sort=downloads&search=google%2Fswitch- 

```python
# pip install accelerate
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/switch-large-128")
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-large-128", device_map="auto")

input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
>>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>
```

If you are interested in training your first MoE, please checkout the [fine tuning google colab script](https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4#scrollTo=xsgNPx4FrtH_) and share your first fine-tuned MoE on the Hub!

## Further reading
- Review of MoE models in Deep Learning: https://arxiv.org/pdf/2209.01667.pdf
- Switch Transformers paper: https://arxiv.org/pdf/2101.03961.pdf
- Microsoft blogpost: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/scaling-speech-language-and-vision-models-with-mixture-of/ba-p/3295750
- More MoE-related papers and code: https://github.com/XueFuzhao/awesome-mixture-of-experts
