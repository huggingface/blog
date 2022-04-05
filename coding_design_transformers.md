# <del>Don't</del> Repeat Yourself${}^{\textbf{*}}$.

"Don't repeat yourself", or DRY, is a well-known principle of software development. The principle originates from "The pragmatic programmer", one of the most read books on code design.
The principle's simple message makes obvious sense: Don't rewrite a logic that already exists somewhere else. This ensures the code remains in sync, making it easier to maintain and more robust. Any change to this logical pattern will uniformly affect all of its dependencies.

At first glance, the design of Hugging Face's Transformers library couldn't be more contrary to the DRY principle. Code for the attention mechanism is more or less copied over 50 times into different model files. Sometimes code of the whole BERT model is copied into other model files. We often force new model contributions that are identical to existing models - besides a small logical tweak - to copy all of the existing code. Why do we do this? Are we just too lazy or overwhelmed to centralize all logical pieces into one place?

No, we are not lazy - it's a very conscious decision to not apply the DRY design principle to the Transformers library. Instead, we decided to adopt a different design principle which we like to call the "single model file" policy. The single model file policy states that all code necessary for the forward pass of a model is in one and only one file - called the model file. If a reader wants to understand how BERT works for inference, she should only have to look into BERT's `modeling_bert.py` file. 
We usually reject any attempt to abstract identical sub-components of different models into a new centralized place. We don't want to have a `attention_layer.py` that includes all possible attention mechanisms. Again why do we do this?

In short the reasons are:
- Transformers is built for and by the open-source community.
- Our product are models and our customers are users reading or tweaking model code.
- The field of machine learning evolves extremely fast. 
- Machine Learning models are static.

## Built by and for the open-source community
Transformers is built to actively incentivize external contributions. A contribution is often either a bug fix or a new model contribution. If a bug is found in one of the model files, we want to make it as easy as possible for the finder to fix it. There is little that is more demotivating than fixing a bug only to see that it caused 100 failures of other models. 

Because model code is independent from all other models, it's fairly easy for someone that only understands the one model she is working with to fix it. Similarly, it's easier to add new modeling code and review the corresponding PR if only a single new model file is added. The contributor does not have to figure out how to add new functionality to a centralized attention mechanism without breaking existing models. The reviewer can easily verify that none of the existing models are broken.

## Modeling code is our product
We assume that a significant amount of users of the Transformers library not only read the documentation, but also look into the actual modeling code and potentially modify it. This hypothesis is backed by the Transformers library being forked over 10,000 times and the Transformers paper being cited over a thousand times.
Therefore it is of utmost importance that someone reading Transformers modeling code for the first time can easily understand and potentially adapt it. Providing all the necessary logical components in order in a single modeling file help a lot to achieve improved readability and adaptability. Additionally, we care a great deal about sensible variable/method naming and prefer expressive/readable code over character-efficient code. 

## Machine Learning is evolving at a neck breaking speed
Research in the field of machine learning, and especially neural networks, evolves extremely fast. A model that was state-of-the-art a year ago might be outdated today. We don't know which attention mechanism, which position embedding, or which architecture will be the best in a year and therefore we cannot really define standard logical patterns that apply for all models. 

As an example, two years ago, one might have defined BERT's self attention layer as the standard attention layer used by all Transformers models. Logically, a "standard" attention function could have been moved into a central `attention.py` file. But then came attention layers that add relative positional embeddings in each attention layer (T5), multiple different forms of chunked attention (Reformer, Longformer, BigBird), and separate attention mechanism for position and word embeddings (DeBERTa), etc... Every time we would have to have asked ourselves whether the "standard" attention function should be adapted or whether it would have been better to add a new attention function to `attention.py`. But then how do we name it? `attention_with_positional_embd`, `reformer_attention`, `deberta_attention`? 

It's dangerous to give logical components of machine learning models general names because the perception of what this component stands for might change or become outdated very quickly. E.g. does chunked attention correspond to GPTNeo's, Reformer's or BigBird's chunked attention? Is the attention layer a self-attention layer, a cross-attentional layer or does it include both? However, if we name attention layers by their model's name, then we should directly put the attention function in the corresponding modeling file.

## Machine Learning models are static
The Transformers library is a unified and polished collection of machine learning models that have been created by different research teams. Every machine learning model is usually accompanied by a paper and its official GitHub repository. Once a machine learning model is published, it is rarely adapted or changed afterwards.

Instead research teams tend to publish a new model that is built upon previous models, but rarely make significant changes to already published code. This is an important realization when deciding on the design principles of the Transformers library.
It means that once a model architecture has been added to Transformers, the fundamental components of the model don't change anymore. Bugs are often found and fixed, methods and variables might be renamed, the output or input format of the model might be slightly changed, but the core components of the model don't change anymore. As a consequence, the need to apply global changes to all models in Transformers is significantly reduced making it less important that every logical pattern only exist once since it's very rarely changed.

A second realization is that models do **not** depend on each other in a bidirectional way. More recent published models might depend on existing models, but it's quite obvious that an existing model cannot logically depend on its successor. E.g. T5 is partly built upon BERT and therefore T5's modeling code might logically depend on BERT's modeling code, but BERT cannot logically depend in any way on T5. Thus, it would not be logically sound to refactor BERT's attention function to also work with T5's attention function - someone reading through BERT's attention layer should not have to know anything about T5. Again, this advocates against centralizing components such as the attention layer into modules that can be accessed by all models.

On the other hand, the modeling code of successor models can very well logically depend on its predecessor model. E.g. DeBERTa-v2 modeling code does logically depend 
to some extent on DeBERTa's modeling code. Maintainability is significantly improved by ensuring the modeling code of DeBERTa-v2 stays in sync with DeBERTa's. Fixing a bug in 
DeBERTa should ideally also fix the same bug in DeBERTa-v2. How can we maintain the single model file policy while ensuring that successor models stay in sync with their predecessor model? 

Now, we explain why we put the asterisk ${}^{\textbf{*}}$ behind *"Do repeat yourself"*. We indeed don't blindly copy paste all existing modeling code even if it looks this way. One of Transformers' core maintainer, Sylvain Gugger, found a great mechanism that respects both the single file policy and keeps maintainability cost in bounds. This mechanism, loosely called *"the copying mechanism"* allows us to mark logical components, such as an attention layer function, with a `# Copied from <predecessor_model>.<function>` statement which enforces the marked code to be identical to the `<function>` of the `<predecessor_model>`. E.g. this line of over [DeBERTa-v2's class](TODO: line here) enforces the whole class to be identical to [DeBERTa's class]( ) except for the prefix `DeBERTav2`.
This way, the copying mechanism keeps modeling code very easy to understand while significantly reducing maintenance. If some code is changed in a function of a predecessor model that is referred to by a function of its successor model, there are tools in place that automatically correct the successor model's function.

What do you think? We are more than interested in hearing your opinion! Click this link to answer directly on our forum.
