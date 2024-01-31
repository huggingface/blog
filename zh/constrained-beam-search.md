---
title: 在 🤗 Transformers 中使用约束波束搜索引导文本生成
thumbnail: /blog/assets/53_constrained_beam_search/thumbnail.png
authors:
- user: cwkeam
  guest: true
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 在 🤗 Transformers 中使用约束波束搜索引导文本生成


<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/53_constrained_beam_search.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt=" 在 Colab 中打开 "/>
</a>

## **引言**

本文假设读者已经熟悉文本生成领域波束搜索相关的背景知识，具体可参见博文 [如何生成文本: 通过 Transformers 用不同的解码方法生成文本](https://huggingface.co/blog/zh/how-to-generate)。

与普通的波束搜索不同，**约束** 波束搜索允许我们控制所生成的文本。这很有用，因为有时我们确切地知道输出中需要包含什么。例如，在机器翻译任务中，我们可能通过查字典已经知道哪些词必须包含在最终的译文中; 而在某些特定的场合中，虽然某几个词对于语言模型而言差不多，但对最终用户而言可能却相差很大。这两种情况都可以通过允许用户告诉模型最终输出中必须包含哪些词来解决。

### **这事儿为什么这么难**

然而，这个事情操作起来并不容易，它要求我们在生成过程中的 _某个时刻_ 在输出文本的 _某个位置_ 强制生成某些特定子序列。

假设我们要生成一个句子 `S`，它必须按照先 $t_1$ 再  $t_2$ 的顺序包含短语 $p_1={ t_1, t_2 }$。以下定义了我们希望生成的句子 $S$:

$$ S_{期望} = { s_1, s_2, …, s_k, t_1, t_2, s_{k+1}, …, s_n } $$

问题是波束搜索是逐词输出文本的。我们可以大致将波束搜索视为函数 $B(\mathbf{s}_{0:i}) = s_{i+1}$，它根据当前生成的序列 $\mathbf{s}_{0:i}$ 预测下一时刻 $i+1$ 的输出。但是这个函数在任意时刻 $i < k$ 怎么知道，未来的某个时刻 $k$ 必须生成某个指定词？或者当它在时刻 $i=k$ 时，它如何确定当前那个指定词的最佳位置，而不是未来的某一时刻 $i>k$？

![为何约束搜索很难](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/why_constraints_are_hard.png)

如果你同时有多个不同的约束怎么办？如果你想同时指定使用短语 $p_1={t_1, t_2}$ _和_ 短语 $p_2={ t_3, t_4, t_5, t_6}$ 怎么办？如果你希望模型在两个短语之间 **任选一个** 怎么办？如果你想同时指定使用短语 $p_1$ 以及短语列表 ${p_{21}, p_{22}, p_{23}}$ 中的任一短语怎么办？

上述需求在实际场景中是很合理的需求，下文介绍的新的约束波束搜索功能可以满足所有这些需求！

我们会先简要介绍一下新的 _**约束波束搜索**_ 可以做些什么，然后再深入介绍其原理。

## **例 1: 指定包含某词**

假设我们要将 `"How old are you?"` 翻译成德语。它对应两种德语表达，其中 `"Wie alt bist du?"` 是非正式场合的表达，而 `"Wie alt sind Sie?"` 是正式场合的表达。

不同的场合，我们可能倾向于不同的表达，但我们如何告诉模型呢？

### **使用传统波束搜索**

我们先看下如何使用 _**传统波束搜索**_ 来完成翻译。

```
!pip install -q git+https://github.com/huggingface/transformers.git
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 *'-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


    Output:
    ----------------------------------------------------------------------------------------------------
    Wie alt bist du?


### **使用约束波束搜索**

但是如果我们想要一个正式的表达而不是非正式的表达呢？如果我们已经先验地知道输出中必须包含什么，我们该如何 _将其_ 注入到输出中呢？

我们可以通过 `model.generate()` 的 `force_words_ids` 参数来实现这一功能，代码如下:

```python
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"

force_words = ["Sie"]

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

outputs = model.generate(
    input_ids,
    force_words_ids=force_words_ids,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 *'-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    Output:
    ----------------------------------------------------------------------------------------------------
    Wie alt sind Sie?


如你所见，现在我们能用我们对输出的先验知识来指导文本的生成。以前我们必须先生成一堆候选输出，然后手动从中挑选出符合我们要求的输出。现在我们可以直接在生成阶段做到这一点。

## **例 2: 析取式约束**

在上面的例子中，我们知道需要在最终输出中包含哪些单词。这方面的一个例子可能是在神经机器翻译过程中结合使用字典。

但是，如果我们不知道要使用哪种 _词形_呢，我们可能希望使用单词 `rain` 但对其不同的词性没有偏好，即 `["raining", "rained", "rains", ...]` 是等概的。更一般地，很多情况下，我们可能并不刻板地希望 _逐字母一致_ ，此时我们希望划定一个范围由模型去从中选择最合适的。

支持这种行为的约束叫 _**析取式约束 (Disjunctive Constraints)**_ ，其允许用户输入一个单词列表来引导文本生成，最终输出中仅须包含该列表中的 _至少一个_ 词即可。

下面是一个混合使用上述两类约束的例子:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

force_word = "scared"
force_flexible = ["scream", "screams", "screaming", "screamed"]

force_words_ids = [
    tokenizer([force_word], add_prefix_space=True, add_special_tokens=False).input_ids,
    tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
]

starting_text = ["The soldiers", "The child"]

input_ids = tokenizer(starting_text, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    force_words_ids=force_words_ids,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 *'-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode(outputs[1], skip_special_tokens=True))

```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

    Output:
    ----------------------------------------------------------------------------------------------------
    The soldiers, who were all scared and screaming at each other as they tried to get out of the
    The child was taken to a local hospital where she screamed and scared for her life, police said.


如你所见，第一个输出里有 `"screaming"` ，第二个输出里有 `"screamed"` ，同时它们都原原本本地包含了 `"scared"` 。注意，其实 `["screaming", "screamed", ...]` 列表中不必一定是同一单词的不同词形，它可以是任何单词。使用这种方式，可以满足我们只需要从候选单词列表中选择一个单词的应用场景。

## **传统波束搜索**

以下是传统 **波束搜索** 的一个例子，摘自之前的 [博文](https://huggingface.co/blog/zh/how-to-generate):

![波束搜索](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/beam_search.png)

与贪心搜索不同，波束搜索会保留更多的候选词。上图中，我们每一步都展示了 3 个最可能的预测词。

在 `num_beams=3` 时，我们可以将第 1 步波束搜索表示成下图:

![波束搜索第 1 步](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/beam_1.jpg)

波束搜索不像贪心搜索那样只选择 `"The dog"` ，而是允许将 `"The nice"` 和 `"The car"` _留待进一步考虑_ 。

下一步，我们会为上一步创建的三个分支分别预测可能的下一个词。

![波束搜索第 2 步](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/beam_2.jpg)

虽然我们 _考查_ 了明显多于 `num_beams` 个候选词，但在每步结束时，我们只会输出 `num_beams` 个最终候选词。我们不能一直分叉，那样的话， `beams` 的数目将在 $n$ 步后变成 $\text{beams}^{n}$ 个，最终变成指数级的增长 (当波束数为 $10$ 时，在 $10$ 步之后就会变成 $10,000,000,000$ 个分支！)。

接着，我们重复上述步骤，直到满足中止条件，如生成 `<eos>` 标记或达到 `max_length` 。整个过程可以总结为: 分叉、排序、剪枝，如此往复。

## **约束波束搜索**

约束波束搜索试图通过在每一步生成过程中 _注入_所需词来满足约束。

假设我们试图指定输出中须包含短语 `"is fast"` 。

在传统波束搜索中，我们在每个分支中找到 `k` 个概率最高的候选词，以供下一步使用。在约束波束搜索中，除了执行与传统波束搜索相同的操作外，我们还会试着把约束词加进去，以 _看看我们是否能尽量满足约束_。图示如下:

![约束搜索第 1 步](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/cbeam_1.jpg)

上图中，我们最终候选词除了包括像 `"dog"` 和 `"nice"` 这样的高概率词之外，我们还把 `"is"` 塞了进去，以尽量满足生成的句子中须含 `"is fast"` 的约束。

第二步，每个分支的候选词选择与传统的波束搜索大部分类似。唯一的不同是，与上面第一步一样，约束波束搜索会在每个新分叉上继续强加约束，把满足约束的候选词强加进来，如下图所示:

![约束搜索第 2 步](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/cbeam_2.jpg)

### **组 (Banks)**

在讨论下一步之前，我们停下来思考一下上述方法的缺陷。

在输出中野蛮地强制插入约束短语 `is fast` 的问题在于，大多数情况下，你最终会得到像上面的 `The is fast` 这样的无意义输出。我们需要解决这个问题。你可以从 `huggingface/transformers` 代码库中的这个 [问题](https://github.com/huggingface/transformers/issues/14081#issuecomment-1004479944) 中了解更多有关这个问题及其复杂性的深入讨论。

组方法通过在满足约束和产生合理输出两者之间取得平衡来解决这个问题。

我们把所有候选波束按照其 `满足了多少步约束`分到不同的组中，其中组 $n$ 里包含的是 _**满足了 $n$ 步约束的波束列表**_ 。然后我们按照顺序轮流选择各组的候选波束。在上图中，我们先从组 2 (Bank 2) 中选择概率最大的输出，然后从组 1 (Bank 1) 中选择概率最大的输出，最后从组 0 (Bank 0) 中选择最大的输出; 接着我们从组 2 (Bank 2) 中选择概率次大的输出，从组 1 (Bank 1) 中选择概率次大的输出，依此类推。因为我们使用的是 `num_beams=3`，所以我们只需执行上述过程三次，就可以得到 `["The is fast", "The dog is", "The dog and"]`。

这样，即使我们 _强制_ 模型考虑我们手动添加的约束词分支，我们依然会跟踪其他可能更有意义的高概率序列。尽管 `The is fast` 完全满足约束，但这并不是一个有意义的短语。幸运的是，我们有 `"The dog is"` 和 `"The dog and"` 可以在未来的步骤中使用，希望在将来这会产生更有意义的输出。

图示如下 (以上例的第 3 步为例):

![约束搜索第 3 步](https://raw.githubusercontent.com/huggingface/blog/main/assets/53_constrained_beam_search/cbeam_3.jpg)

请注意，上图中不需要强制添加 `"The is fast"`，因为它已经被包含在概率排序中了。另外，请注意像 `"The dog is slow"` 或 `"The dog is mad"` 这样的波束实际上是属于组 0 (Bank 0) 的，为什么呢？因为尽管它包含词 `"is"` ，但它不可用于生成 `"is fast"` ，因为 `fast` 的位子已经被 `slow` 或 `mad` 占掉了，也就杜绝了后续能生成 `"is fast"` 的可能性。从另一个角度讲，因为 `slow` 这样的词的加入，该分支 _满足约束的进度_ 被重置成了 0。

最后请注意，我们最终生成了包含约束短语的合理输出: `"The dog is fast"` ！

起初我们很担心，因为盲目地添加约束词会导致出现诸如 `"The is fast"` 之类的无意义短语。然而，使用基于组的轮流选择方法，我们最终隐式地摆脱了无意义的输出，优先选择了更合理的输出。

## **关于 `Constraint` 类的更多信息及自定义约束**

我们总结下要点。每一步，我们都不断地纠缠模型，强制添加约束词，同时也跟踪不满足约束的分支，直到最终生成包含所需短语的合理的高概率序列。

在实现时，我们的主要方法是将每个约束表示为一个 `Constraint` 对象，其目的是跟踪满足约束的进度并告诉波束搜索接下来要生成哪些词。尽管我们可以使用 `model.generate()` 的关键字参数 `force_words_ids` ，但使用该参数时后端实际发生的情况如下:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PhrasalConstraint

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"

constraints = [
    PhrasalConstraint(
        tokenizer("Sie", add_special_tokens=False).input_ids
    )
]

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    constraints=constraints,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 *'-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

    Output:
    ----------------------------------------------------------------------------------------------------
    Wie alt sind Sie?

你甚至可以定义一个自己的约束并将其通过 `constraints` 参数输入给 `model.generate()` 。此时，你只需要创建 `Constraint` 抽象接口类的子类并遵循其要求即可。你可以在 [此处](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_constraints.py) 的 `Constraint` 定义中找到更多信息。

我们还可以尝试其他一些有意思的约束 (尚未实现，也许你可以试一试！) 如  `OrderedConstraints` 、 `TemplateConstraints` 等。目前，在最终输出中约束短语间是无序的。例如，前面的例子一个输出中的约束短语顺序为 `scared -> screaming` ，而另一个输出中的约束短语顺序为 `screamed -> scared` 。 如果有了 `OrderedConstraints`， 我们就可以允许用户指定约束短语的顺序。 `TemplateConstraints` 的功能更小众，其约束可以像这样:

```python
starting_text = "The woman"
template = ["the", "", "School of", "", "in"]

possible_outputs == [
   "The woman attended the Ross School of Business in Michigan.",
   "The woman was the administrator for the Harvard School of Business in MA."
]
```

或是这样:

```python
starting_text = "The woman"
template = ["the", "", "", "University", "", "in"]

possible_outputs == [
   "The woman attended the Carnegie Mellon University in Pittsburgh.",
]
impossible_outputs == [
  "The woman attended the Harvard University in MA."
]
```

或者，如果用户不关心两个词之间应该隔多少个词，那仅用 `OrderedConstraint` 就可以了。

## **总结**

约束波束搜索为我们提供了一种将外部知识和需求注入文本生成过程的灵活方法。以前，没有一个简单的方法可用于告诉模型 1. 输出中需要包含某列表中的词或短语，其中 2. 其中有一些是可选的，有些必须包含的，这样 3. 它们可以最终生成至在合理的位置。现在，我们可以通过综合使用 `Constraint` 的不同子类来完全控制我们的生成！

该新特性主要基于以下论文:

- [Guided Open Vocabulary Image Captioning with Constrained Beam Search](https://arxiv.org/pdf/1612.00576.pdf)
- [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](https://arxiv.org/abs/1804.06609)
- [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](https://aclanthology.org/N19-1090/)
- [Guided Generation of Cause and Effect](https://arxiv.org/pdf/2107.09846.pdf)

与上述这些工作一样，还有许多新的研究正在探索如何使用外部知识 (例如 KG (Knowledge Graph) 、KB (Knowledge Base) ) 来指导大型深度学习模型输出。我们希望约束波束搜索功能成为实现此目的的有效方法之一。

感谢所有为此功能提供指导的人: Patrick von Platen 参与了从 [初始问题](https://github.com/huggingface/transformers/issues/14081) 讨论到 [最终 PR](https://github.com/huggingface/transformers/pull/15761) 的全过程，还有 Narsil Patry，他们二位对代码进行了详细的反馈。

_本文使用的图标来自于 <a href="https://www.flaticon.com/free-icons/shorthand" title="shorthand icons">Freepik - Flaticon</a>。_