---
title: "聊天模板：无声性能杀手的终结" 
thumbnail: /blog/assets/chat-templates/thumbnail.png
authors:
- user: rocketknight1
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 聊天模板

> _一个幽灵，格式不正确的幽灵，在聊天模型中游荡！_

## 太长不看版

现存的聊天模型使用的训练数据格式各各不同，我们需要用这些格式将对话转换为单个字符串并传给分词器。如果我们在微调或推理时使用的格式与模型训练时使用的格式不同，通常会导致严重的、无声的性能下降，因此匹配训练期间使用的格式极其重要！ Hugging Face 分词器新增了 `chat_template` 属性，可用于保存模型训练时使用的聊天格式。此属性包含一个 Jinja 模板，可将对话历史记录格式化为正确的字符串。请参阅 [技术文档](https://huggingface.co/docs/transformers/main/en/chat_templated)，以了解有关如何在代码中编写和应用聊天模板。

## 引言

如果你熟悉 🤗 transformers 库，你可能写过如下代码:

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
```

通过从同一个 checkpoint 中加载分词器和模型，可以确保对输入字符串使用的分词方法符合模型预期。如果你从另一个模型中选择分词器，则其分词结果很可能会完全不同，此时模型的性能就会受到严重损害。这种现象叫 **分布漂移 (distribution shift)**: 模型一直从一种分布学习 (即训练分词器)，突然，数据分布变成了另一个不同的分布。

无论你是微调模型还是直接用它进行推理，让这种分布上的变化尽可能小，并保持提供的输入尽可能与训练时的输入一致总是一个好主意。对于常规语言模型，做到这一点相对容易 - 只需从同一检查点加载分词器和模型，就可以了。

然而，对于聊天模型来说，情况有点不同。这是因为“聊天”不仅仅是直接对单个文本字符串进行分词 - 它需要对一系列消息进行分词。每个消息都包含一个 `角色` 及其 `内容` ，其内容是消息的实际文本。最常见的，角色是“用户”(用于用户发送的消息) 、“助理”(用于模型生成的响应)，以及可选的“系统”(指在对话开始时给出的高级指令)。

干讲可能有点抽象，下面我们给出一个示例聊天，把问题具象化:

```python
[
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"}
]
```

此消息序列需要先转换为一个文本字符串，然后才能对其进行分词以输入给模型。但问题是，转换方法有很多！例如，你可以将消息列表转换为“即时消息”格式:

```
User: Hey there!
Bot: Nice to meet you!
```

或者你可以添加特殊词元来指示角色:

```
[USER] Hey there! [/USER]
[ASST] Nice to meet you! [/ASST]
```

抑或你可以添加词元以指示消息之间的边界，而将角色信息作为字符串插入:

```
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
```

方法多种多样，但没有哪种方法是最好的或是最正确的。因此，不同的模型会采用截然不同的格式进行训练。上面这些例子不是我编造的，它们都是真实的，并且至少被一个现存模型使用过！但是，一旦模型接受了某种格式的训练，你需要确保未来的输入使用相同的格式，否则就可能会出现损害性能的分布漂移。

## 模板: 一种保存格式信息的方式

当前的状况是: 如果幸运的话，你需要的格式已被正确记录在模型卡中的某个位置; 如果不幸的话，它不在，那如果你想用这个模型的话，只能祝你好运了; 在极端情况下，我们甚至会将整个提示格式放在 [相应模型的博文](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) 中，以确保用户不会错过它！但即使在最好的情况下，你也必须找到模板信息并在微调或推理流水线中手动将其写进代码。我们认为这是一个特别危险的做法，因为使用错误的聊天格式是一个 **静默错误** - 一旦出了错，不会有显式的失败或 Python 异常来告诉你出了什么问题，模型的表现只会比用正确格式时差多了，但很难调试其原因！

这正是 **聊天模板** 旨在解决的问题。聊天模板是一个 [Jinja 模板字符串](https://jinja.palletsprojects.com/en/3.1.x/)，你可以使用分词器保存和加载它。聊天模板包含了将聊天消息列表转换为模型所需的、格式正确的输入字符串所需要的全部信息。下面是三个聊天模板字符串，分别对应上文所述的三种消息格式:

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "User : " }}
    {% else %}
        {{ "Bot : " }}
    {{ message['content'] + '\n' }}
{% endfor %}
```

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "[USER]" + message['content'] + " [/USER]" }}
    {% else %}
        {{ "[ASST]" + message['content'] + " [/ASST]" }}
    {{ message['content'] + '\n' }}
{% endfor %}
```

```jinja
"{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
"{% endfor %}"
```

如果你不熟悉 Jinja，我强烈建议你花点时间研究下这些模板字符串及其相应的模板输出，看看你是否可以弄清楚这些模板如何将消息列表转换为格式化的消息字符串！其语法在很多方面与 Python 非常相似。

## 为什么要使用模板？

如果你不熟悉 Jinja，一开始上手可能会有点困惑，但我们在实践中发现 Python 程序员可以很快上手它。在开发此功能的过程中，我们考虑了其他方法，例如允许用户按角色指定消息的前缀和后缀。我们发现该方法会变得令人困惑且笨重，而且它非常不灵活，以至于对一些模型而言，我们得需要一些巧妙的变通才行。而另一方面，模板功能强大到足以完全支持我们所知的所有消息格式。

## 为什么要这样做呢？为什么大家不统一到一个标准格式呢？

好主意！不幸的是，为时已晚，因为现有的多个重要模型已经基于迥异的聊天格式进行了训练。

然而，我们仍然可以稍微缓解下这个问题。我们认为最接近“标准”的格式是 OpenAI 创建的 [ChatML 格式](https://github.com/openai/openai-python/blob/main/chatml.md)。如果你正在训练新的聊天模型，并且此格式适合你，我们建议你使用它并给分词器添加特殊的 `<|im_start|>` 和 `<|im_end|>` 词元。它的优点是角色非常灵活，因为角色只是作为字符串插入，而不是特定的角色词元。如果你想使用这个，它是上面的第三个模板，你可以简单地使用一行代码进行设置:

```py
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

不过，除了格式林立的现状之外，还有第二个不硬设标准格式的原因 - 我们预计模板将广泛用于多种类型模型的预处理，包括那些可能与标准聊天操作迥异的模型。硬设标准格式限制了模型开发人员使用此功能完成我们尚未想到的任务的能力，而模板则为用户和开发人员提供了最大的自由度。甚至可以在模板中加入逻辑检查和判断，这是目前任何默认模板中都没有深入使用的功能，但我们希望它能成为喜欢冒险的用户手中的利刃。我们坚信，开源生态系统应该让你能够做你想做的事，而不是命令你做什么。

## 模板如何工作？

聊天模板是 **分词器** 的一部分，因为它们履行与分词器相同的角色: 存储有关如何预处理数据的信息，以确保你以与训练时相同的格式将数据提供给模型。我们的设计使得用户非常容易将模板信息添加到现有分词器并将其保存或上传到 Hub。

在有聊天模板这个功能之前，聊天格式信息都存储在 **类级别** - 这意味着，例如，所有 LLaMA checkpoint 都将使用同一个硬设在 `transformers` 的 LLaMA 模型类代码中的聊天格式。为了向后兼容，目前具有自定义聊天格式方法的模型类也已被赋予了 **默认聊天模板**。

在类级别设置默认聊天模板，用于告诉 `ConversationPipeline` 等类在模型没有聊天模板时如何格式化输入，这样做 **纯粹是为了向后兼容**。我们强烈建议你在任何聊天模型上显式设置聊天模板，即使默认聊天模板是合适的。这可以确保默认聊天模板中的任何未来的更改或弃用都不会破坏你的模型。尽管我们将在可预见的将来保留默认聊天模板，但我们希望随着时间的推移将所有模型转换为显式聊天模板，届时默认聊天模板可能会被完全删除。

有关如何设置和应用聊天模板的详细信息，请参阅 [技术文档](https://huggingface.co/docs/transformers/main/en/chat_templated)。

## 我该如何开始使用模板？

很简单！如果分词器设置了 `chat_template` 属性，则它已准备就绪。你可以在 `ConversationPipeline` 中使用该模型和分词器，也可以调用 `tokenizer.apply_chat_template()` 来格式化聊天以进行推理或训练。请参阅我们的 [开发者指南](https://huggingface.co/docs/transformers/main/en/chat_templated) 或 [如何应用聊天模板的文档](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) 以了解更多！

如果分词器没有 `chat_template` 属性，它可能仍然可以工作，但它将使用该模型类的默认聊天模板。正如我们上面提到的，这是脆弱的，并且当类模板与模型实际训练的内容不匹配时，它同样会导致静默错误。如果你想使用没有 `chat_template` 的 checkpoint，我们建议检查模型卡等文档以确保使用正确的格式，然后为该格式添加正确的 `chat_template` 。即使默认聊天模板是正确的，我们也建议这样做 - 它可以使模型面向未来，并且还可以清楚地表明该模板是存在的且是适用的。

即使不是你的 checkpoint，你也可以通过提交 [合并请求 (pull request) ](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) 的方式为其添加 `chat_template` 。仅需将 `tokenizer.chat_template` 属性设置为 Jinja 模板字符串。完成后，推送更改就可以了！

如果你想在你的聊天应用中使用某 checkpoint，但找不到有关其使用的聊天格式的任何文档，你可能应该在 checkpoint 上提出问题或联系其所有者！一旦你弄清楚模型使用的格式，请提交一个 PR 以添加合适的 `chat_template` 。其他用户将会非常感激你的贡献！

## 总结: 模板理念

我们认为模板是一个非常令人兴奋的新特性。除了解决大量无声的、影响性能的错误之外，我们认为它们还开辟了全新的方法和数据模式。但最重要的也许是，它们还代表了一种理念转变: 从核心 `transformers` 代码库中挪出一个重要功能，并将其转移到各自模型的仓库中，用户可以自由地做各种奇怪、狂野抑或奇妙的事情。我们迫不及待想看看你会发现哪些用途！