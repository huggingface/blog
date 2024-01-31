---
title: "在 Transformers 中使用对比搜索生成可媲美人类水平的文本🤗"
thumbnail: /blog/assets/115_introducing_contrastive_search/thumbnail.png
authors:
- user: GMFTBY
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 在 Transformers 中使用对比搜索生成可媲美人类水平的文本🤗


---

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/115_introducing_contrastive_search.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 1. 引言

自然语言生成 (即文本生成) 是自然语言处理 (NLP) 的核心任务之一。本文将介绍神经网络文本生成领域当前最先进的解码方法 **对比搜索 (Contrastive Search)**。提出该方法的论文 _“A Contrastive Framework for Neural Text Generation”_ 最初发表于 NeurIPS 2022 ([[论文]](https://arxiv.org/abs/2202.06417)、[[官方实现]](https://github.com/yxuansu/SimCTG))。此后， _“Contrastive Search Is What You Need For Neural Text Generation”_ 的作者又进一步证明了对比搜索可以用 **现有的** 语言模型在 **16** 种语言上生成可媲美人类水平的文本 ([[论文]](https://arxiv.org/abs/2210.14140)、[[官方实现]](https://github.com/yxuansu/Contrastive_Search_Is_What_You_Need))。

**[备注]** 对于不熟悉文本生成的用户，请参阅 [此博文](https://huggingface.co/blog/how-to-generate) 了解更多详情。

---

<span id='demo'/>

### 2. Hugging Face 🤗 对比搜索演示

目前，🤗 `transformers` 的 PyTorch 和 TensorFlow 后端均支持对比搜索。你可以在 [该 Colab notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/115_introducing_contrastive_search.ipynb) 中根据不同的后端选择相应的部分来探索该方法，文章顶部也有该 notebook 链接。我们还构建了这个不错的 [演示应用](https://huggingface.co/spaces/joaogante/contrastive_search_generation)，用它可以直观地比较对比搜索与其他流行的解码方法 (例如波束搜索、top-k 采样 <a href='#references'>[3]</a> 以及核采样 <a href='#references'>[4]</a>)。

---

<span id='installation'/>

### 3. 环境安装

在进行后续实验前，我们要先安装最新的 `transformers` 库，如下:

```shell
pip install torch
pip install "transformers==4.24.0"
```

---

<span id='problems_of_decoding_methods'/>

### 4. 现有解码方法存在的问题

解码方法可以分为两类: (i) 确定性方法，(ii) 随机方法。下面我们分别对两者进行讨论！

<span id='deterministic_methods'/>

#### 4.1. 确定性方法

确定性方法，如贪心搜索和波束搜索，通过在语言模型输出的所有候选补全词中选择概率最高的词来生成最终文本。然而，正如之前研究 <a href='#references'>[3]</a><a href='#references'>[4]</a> 指出的，确定性方法通常会导致 _模型退化_，即生成的文本不自然且包含不必要的重复。

下面，我们看一个用 GPT-2 模型和贪心搜索生成文本的例子。

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
input_ids = tokenizer('DeepMind Company is', return_tensors='pt').input_ids
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

output = model.generate(input_ids, max_length=128)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

<details open>
<summary><b> 模型输出: </b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a leading AI research company, with a focus on deep learning and deep learning-based systems.

The company's research is focused on the development of deep learning-based systems that can learn from large amounts of data, and that can be used to solve real-world problems.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies for the UK's National Health Service.

DeepMind's research is also used by the UK government to develop new technologies
----------------------------------------------------------------------------------------------------
```
</details>

**[备注]** 我们可以看到，贪心搜索生成的结果中有明显的重复。

<span id='stochastic_methods'/>

#### 4.2. 随机方法

为了解决确定性方法带来的问题，随机方法通过在解码过程中引入随机性来生成文本。常用的两种随机方法是 (i) top-k 采样 <a href='#references'>[3]</a> 和 (ii) 核采样 (也称为 top-p 采样) <a href='#references'>[4]</a>。

下面，我们给出用 GPT-2 模型和核采样 (p=0.95) 生成文本的示例。

```python
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
input_ids = tokenizer('DeepMind Company is', return_tensors='pt').input_ids
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

torch.manual_seed(0.)
output = model.generate(input_ids, do_sample=True, max_length=128, top_p=0.95, top_k=0)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

<details open>
<summary><b> 模型输出: </b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a leading provider of AI-based research, development, and delivery of AI solutions for security, infrastructure, machine learning, communications, and so on."

'AI is not journalism'

Worse still was the message its researchers hoped would reach the world's media — that it was not really research, but rather a get-rich-quick scheme to profit from living forces' ignorance.

"The thing is, we know that people don't consciously assess the value of the others'
information. They understand they will get the same on their own."

One example? Given the details of today
----------------------------------------------------------------------------------------------------
```

</details>

**[备注]** 虽然核采样可以生成没有重复的文本，但生成文本的语义一致性并不是很好。例如，生成的短语 _‘AI is not journalism’_ 与给定的上文即 _‘DeepMind Company’_ 不一致。

我们注意到，这种语义不一致的问题可以通过降低温度 (temperature) 来部分解决。然而，降低温度会使核采样更接近贪心搜索，这其实就变成了贪心搜索和核采样之间的权衡。一般来讲，要找到一个既能避免贪心搜索又能避免核采样陷阱的快捷且与模型无关的温度相当有挑战。

---

<span id='contrastive_search'/>

### 5. 对比搜索

本节我们来详细介绍一种新的解码方法， _ **对比搜索**_。
<span id='contrastive_objective'/>

#### 5.1. 解码目标

给定前缀文本 $x_{< t}$，我们按如下公式选择输出词元 $x_{t}$:
<center class="half">
    <img src="/blog/assets/115_introducing_contrastive_search/formulation.png" width="750"/>
</center>

上式中， $V^{(k)}$ 是语言模型输出概率分布 $p_{\theta}(v|x_{< t})$ 中 k 个概率最大的候选词元的集合。第一项，即 _模型置信度 (model confidence)_，是语言模型预测的每个候选词元 $v$ 的概率。第二项， _退化惩罚 (degeneration penalty)_，用于度量 $v$ 与上文 $x_{< t}$ 中每个词元的相异度，其中函数 $s(\cdot, \cdot)$ 用于计算每两个词元间的余弦相似度。更具体地说，退化惩罚被定义为 $v$ 的向量表征 $h_{v}$ 与其上文 $x_ {< t}$ 中每个词元的向量表征间余弦相似度的最大值。这里，候选词元的向量表征 $h_{v}$ 是在给定 $x_{< t}$ 和  $v$ 的条件下将二者连接起来输入给语言模型，然后由语言模型计算出来的。直观上，如果 $v$ 的退化惩罚较大意味着它与上文更相似 (在表示空间中)，因此更有可能导致模型退化问题。超参数 $\alpha$ 用于在这两项中折衷。当 $\alpha=0$ 时，对比搜索退化为纯贪心搜索。

**[备注]** 在生成输出时，对比搜索同时考虑 (i) 语言模型预测的概率，以保持生成文本和前缀文本之间的语义连贯性; (ii) 与上文的相似性以避免模型退化。

<span id='contrastive_generation'/>

#### 5.2. 使用对比搜索生成文本

下面，我们使用与 <a href='#deterministic_methods'> 第 4.1 节 </a> 和 <a href='#stochastic_methods'> 第 4.2 节 </a> 中相同的前缀文本 (即 _“DeepMind Company is”_ )，并使用对比搜索生成文本 (取 k=4、$\alpha=0.6$)。为了充分展示对比搜索的卓越能力，我们让语言模型生成一个 **512** 词元的 **长**文档，如下:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
model.eval()

# prepare the prefix
prefix_text = r'DeepMind Company is'
input_ids = tokenizer(prefix_text, return_tensors='pt').input_ids

# generate the result with contrastive search
output = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=512)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

参数设置如下:
- `--top_k`: 对比搜索中的超参 $k$。
- `--penalty_alpha`: 对比搜索中的超参 $\alpha$。

<details open>
<summary><b> 模型输出: </b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
DeepMind Company is a leader in artificial intelligence (AI). We have a long history of working with companies such as Google, Facebook, Amazon, and Microsoft to build products that improve people's lives, and today we are excited to announce that DeepMind's AlphaGo program has won the game of Go, becoming the first program to defeat a professional Go player.

The victory is a testament to the power of deep learning, and to the incredible work of our research team, which has been at the forefront of AI research for the past five years. AlphaGo is one of the most advanced Go programs ever created, and its performance is an important step towards the goal of human-level AI.

"This is the culmination of a decade of hard work," said Andy Ng, co-founder and CTO of DeepMind. "We are thrilled to have achieved this milestone and look forward to continuing to develop AI that can be used in a wide range of applications and to help people live better lives."

DeepMind's work on Go began in 2010, when it began to train a neural network to play Go using millions of games played by top Go players around the world. Since then, the team has refined the algorithm, adding more and more layers of reinforcement learning to make it better at recognizing patterns and making decisions based on those patterns. In the past year and a half, the team has made significant progress in the game, winning a record-tying 13 games in a row to move into the top four of the world rankings.

"The game of Go is a complex game in which players have to be very careful not to overextend their territory, and this is something that we have been able to improve over and over again," said Dr. Demis Hassabis, co-founder and Chief Scientific Officer of DeepMind. "We are very proud of our team's work, and we hope that it will inspire others to take the next step in their research and apply the same techniques to other problems."

In addition to the win in Go, DeepMind has also developed an AI system that can learn to play a number of different games, including poker, Go, and chess. This AI system, called Tarsier, was developed in partnership with Carnegie Mellon University and the University of California, Berkeley, and is being used to teach computer vision and machine learning to identify objects in images and recognize speech in natural language. Tarsier has been trained to play the game of Go and other games on a
----------------------------------------------------------------------------------------------------
```

</details>

**[备注]** 我们看到生成的文本质量非常高。整个文档语法流畅，语义连贯。同时，生成的文本也很好地保持了事实的正确性。例如，在第一段中，它正确阐述了 _“AlphaGo”_ 作为 _“第一个击败职业围棋选手的程序”_ 这一事实。

<span id='contrastive_visual_demonstration'/>

#### 5.3. 对比搜索的结果可视化

为了更好地理解对比搜索的工作原理，我们对贪心搜索 (<a href='#deterministic_methods'> 第 4.1 节 </a>) 和对比搜索进行了直观比较。具体来说，我们分别将贪心搜索和对比搜索生成的词元相似度矩阵可视化。两个词元之间的相似性被定义为它们的向量表征 (即最后一个转换器层的隐藏状态) 之间的余弦相似性。贪心搜索 (上) 和对比搜索 (下) 的结果如下图所示。

<center class="half">
    <img src="/blog/assets/115_introducing_contrastive_search/greedy_search_visualization.png" width="400"/>
    <img src="/blog/assets/115_introducing_contrastive_search/contrastive_search_visualization.png" width="400"/>
</center>

**[备注]** 从贪心搜索的结果中，我们看到非对角线的相似度很高，这清楚地表明贪心搜索产生了重复。相反，在对比搜索的结果中，高相似度分数主要出现在对角线上，这证明我们成功解决了退化问题。对比搜索的这一优良特性是通过在解码过程中引入退化惩罚 (参见 <a href='#contrastive_objective'> 第 5.1 节 </a>) 来实现的。

---

<span id='more_examples'/>

### 6. 更多的生成示例

在本节中，我们提供了更多的生成示例来比较不同的解码方法。

<span id='gpt2_example_one'/>

#### 6.1. 示例一: GPT-2

在这部分中，我们使用 GPT-2 生成文本，其前缀文本来自 [OpenAI 发布 GPT-2 的博客文章](https://openai.com/blog/better-language-models/)。

> _In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English._

<details open>
<summary><b> 加载语言模型并准备前缀文本: </b></summary>

```python
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

prefix_text = r"In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
input_ids = tokenizer(prefix_text, return_tensors='pt').input_ids
```

</details>
<span id='gpt2_greedy_example_one'/>

##### 6.1.1. 使用贪心搜索生成文本

<details>
<summary><b> 代码: [点击展开]</b></summary>

```python
output = model.generate(input_ids, max_length=512)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details>
<summary><b> 模型输出: [点击展开]</b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, led by Dr. David R. Williams of the University of California, Santa Cruz, discovered the unicorns in the Andes Mountains of Peru. The area is known for its unique geology and is home to a number of rare species of animals.

The researchers found the unicorns in the Andes Mountains of Peru.

"We were surprised to find that the unicorns were able to communicate with each other," Williams said. "We were also surprised to find that they were able to communicate in English."

The researchers believe that the unicorns are descendants of the ancient Incas, who lived in the area around 2,000 years ago.

"The Incas were the first people to use the Andes Mountains as a refuge from the harsh climate of the Andes," Williams said. "They were also the first people to use the Andes Mountains as a place to hunt and gather food."

The researchers believe that the unicorns are descendants of the ancient Incas, who lived in the area around 2,000 years ago.

"The Incas were the first people to use the Andes Mountains as a refuge from the harsh climate of the Andes," Williams said. "They were also the first people to use the Andes Mountains as a place to hunt and gather food."

The researchers believe that the unicorns are descendants of the ancient Incas, who lived in the area around 2,000 years ago.

"The Incas were the first people to use the Andes Mountains as a refuge from the harsh climate of the Andes," Williams said. "They were also the first people to use the Andes Mountains as a place to hunt and gather food."

The researchers believe that the unicorns are descendants of the ancient Incas, who lived in the area around 2,000 years ago.

"The Incas were the first people to use the Andes Mountains as a refuge from the harsh climate of the Andes," Williams said. "They were also the first people to use the Andes Mountains as a place to hunt and gather food."

The researchers believe that the unicorns are descendants of the ancient Incas, who lived in the area around 2,000 years ago
----------------------------------------------------------------------------------------------------
```
</details>

<span id='gpt2_nucleus_example_one'/>

##### 6.1.2. 使用核采样生成文本

<details>

<summary><b> 代码: [点击展开]</b></summary>

```python
torch.manual_seed(0.)
output = model.generate(input_ids, do_sample=True, max_length=512, top_p=0.95, top_k=0)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details>

<summary><b> 模型输出: [点击展开]</b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English. The study was published in the Journal of Zoology in March 2016.

Polygynous mammals such as unicorns have remained largely unknown to science. Professor Gustavo Giacota, from the University of Oxford who led the study, said that they had been documented as far as Eastern Siberia in Russia, but had only been seen a handful of times in the Gobi Desert.

Tiny animals with pale and shiny coats live in the presence of human beings and are hardly likely to be victims of any cruelty. However, there is some evidence of the condition occurring in both humans and animals in remote regions, which might have similarities to "black moles" that coexist on the skin.

It is thought that Unicorns could be inside themselves, that they have different scents depending on their current environment, or just fall out and there are plenty of legends of how they have survived. Experts speculate that the moths and other animals could be remnants of the Yezidi Isis and Charon, which literally is both the word which means great bird, and the Greek word for sound. It is said that the Isis and Charon taught their young the use of voice in the form of calling out to others.

The scientists think that it could be ancient folklore that has survived and is no longer attributed to a real entity
----------------------------------------------------------------------------------------------------
```

</details>

<span id='gpt2_contrastive_example_one'/>

##### 6.1.3. 使用对比搜索生成文本

<details open>

<summary><b> 代码: </b></summary>

```python
output = model.generate(input_ids, max_length=512, penalty_alpha=0.6, top_k=4)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details open>

<summary><b> 模型输出: </b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

According to the BBC, a team of scientists led by Dr David MacKay, from the University of Bristol, spent two years searching for the unicorn herd, which they discovered during a survey of the area.

"It's a very rare find," MacKay told the BBC. "There are a few in the Himalayas, but this is the first time we've been able to find one in such a remote area."

The team was surprised to find a herd of unicorns living in a region that has been known to be a hotbed of poaching, with many of the animals poached for their horns, which are used in traditional Chinese medicine to treat everything from rheumatism to cancer.

"We knew that the area was rich in rhino horn, but we had no idea how many there were, or what they were doing there," MacKay said. "This is an area of high poaching pressure, and we wanted to find out what was going on."

In order to do so, the team used GPS collars to track the animals as they moved around the mountain and the surrounding area. The GPS data was then compared with information gathered from local villagers, who had a wealth of information about the animals' movements, including where they were eating, what they were doing at night, and how much time they spent in the mountains each day.

After analyzing the data, the team determined that the herd consisted of at least three species of unicorns, including a male and two females. One of the females was the mother of the male, and the other two were her daughters. All three had the same horn color, which is believed to be a sign of purity in the animal kingdom.

While the discovery is exciting, it's not the first time scientists have discovered an animal that speaks English. Last year, scientists discovered a species of porcupine that can be heard by humans, and has been dubbed "Porcupine Man" for his ability to converse with the human race.
----------------------------------------------------------------------------------------------------
```

</details>

<span id='opt_example_two'/>

#### 6.2. 示例二: OPT

本节中，我们使用 Meta 最近发布的 OPT 模型 <a href='#references'>[5]</a>，并从著名的 ResNet 论文 <a href='#references'>[6]</a> 摘要中选取前两个句子来作为前缀文本。

> Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.

<details open>

<summary><b> 加载语言模型并准备前缀文本: </b></summary>

```python
import torch
from transformers import AutoTokenizer, OPTForCausalLM
model_name = r'facebook/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)

prefix_text = r"Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."
input_ids = tokenizer(prefix_text, return_tensors='pt').input_ids
```

</details>

<span id='opt_greedy_example_two'/>

##### 6.2.1. 使用贪心搜索生成文本

<details>

<summary><b> 代码: [点击展开]</b></summary>

```python
output = model.generate(input_ids, max_length=256)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details>

<summary><b> 模型输出: [点击展开]</b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We show that the residual learning framework can be used to train deep neural networks that are significantly more difficult to train than those used previously. We also show that the residual learning framework can be used to train deep neural networks that are significantly more difficult to train than those used previously.

The paper presents a new residual learning framework for deep neural networks that is based on the concept of residuals. The residuals are the residuals of the network that are not used in the training process. The residuals are computed by taking the residuals of the network that are used in the training process and subtracting the residuals of the network that are not used in the training process. The residuals are then used to train the network. The residuals are computed by taking the residuals of the network that are used in the training process and subtracting the residuals of the network that are not used in the training process. The residuals are then used to train the network. The residuals are computed by taking the residuals of the network that are used in the training process and subtracting the residuals of the
----------------------------------------------------------------------------------------------------
```

</details>

<span id='opt_greedy_example_two'/>

##### 6.2.2. 使用核采样生成文本

<details>

<summary><b> 代码: [点击展开]</b></summary>

```python
torch.manual_seed(0.)
output = model.generate(input_ids, do_sample=True, max_length=256, top_p=0.95, top_k=0)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details>

<summary><b> 模型输出: [点击展开]</b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. The theory focuses on several aspects of learning, including the dynamics of replicative and non-replicative aspects of learning. This framework emphasizes learning by entropy. New randomized algorithms enable training networks with residual learning, so that deep networks can be deployed as reliably and as efficiently as their more conventional counterparts.
----------------------------------------------------------------------------------------------------
```
</details>

<span id='opt_contrastive_example_two'/>

##### 6.2.3. 使用对比搜索生成文本

<details open>

<summary><b> 代码: </b></summary>

```python
output = model.generate(input_ids, max_length=256, penalty_alpha=0.6, top_k=6)
print("Output:\n" + 100 *'-')
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("" + 100 *'-')
```

</details>

<details open>

<summary><b> 模型输出: </b></summary>

```
Output:
----------------------------------------------------------------------------------------------------
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.

In this paper, we propose a model-based residual learning (MBRL) framework that is based on neural networks trained on data that is sparse in terms of dimensionality (e.g., 1, 2, 3, etc.). The network parameters are chosen such that there is a high probability of convergence, i.e., the number of iterations is large enough to minimize the variance of the residuals. This is achieved by training the network on a set of training data, in which the data is sparse in terms of dimensionality, and then discarding the nonparametric part of the data after training is complete.

We show that MBRL outperforms other methods for deep reinforcement learning (RL) and deep convolutional neural networks (CNNs) by a factor of at least 2. In addition, we show that, compared to CNNs, MBRL performs better in two-dimensional (2D) and three-dimensional (3D) cases.
----------------------------------------------------------------------------------------------------
```

</details>

---

<span id='resources'/>

### 7. 更多资源

有关对比搜索的更多详细信息，请查看我们的论文和代码，如下:

- **A Contrastive Framework for Neural Text Generation**: [论文](https://arxiv.org/abs/2202.06417)、[官方实现](https://github.com/yxuansu/SimCTG)
- **Contrastive Search Is What You Need For Neural Text Generation**: [论文](https://arxiv.org/abs/2210.14140)、[官方实现](https://github.com/yxuansu/Contrastive_Search_Is_What_You_Need)

---

<span id='citation'/>

### 8. 引用

```bibtex
@inproceedings{su2022a,
   title={A Contrastive Framework for Neural Text Generation},
   author={Yixuan Su and Tian Lan and Yan Wang and Dani Yogatama and Lingpeng Kong and Nigel Collier},
   booktitle={Advances in Neural Information Processing Systems},
   editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
   year={2022},
   url={https://openreview.net/forum?id=V88BafmH9Pj}
}

@article{su2022contrastiveiswhatyouneed,
  title={Contrastive Search Is What You Need For Neural Text Generation},
  author={Su, Yixuan and Collier, Nigel},
  journal={arXiv preprint arXiv:2210.14140},
  year={2022}
}
```

---

<span id='references'/>

## 参考文献
> [1] Su et al., 2022 ["A Contrastive Framework for Neural Text Generation"](https://arxiv.org/abs/2202.06417), NeurIPS 2022

> [2] Su and Collier, 2022 ["Contrastive Search Is What You Need For Neural Text Generation"](https://arxiv.org/abs/2210.14140), Arxiv 2022

> [3] Fan et al., 2018 ["Hierarchical Neural Story Generation"](https://arxiv.org/abs/1805.04833), ACL 2018

> [4] Holtzman et al., 2020 ["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751), ICLR 2020

> [5] Zhang et al., 2022 ["OPT: Open Pre-trained Transformer Language Models"](https://arxiv.org/abs/2205.01068), Arxiv 2022

> [6] He et al., 2016 ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385), CVPR 2016

---

_- 本文由 Yixuan Su 和 Tian Lan 撰写_

---

<span id='acknowledgements'/>

## 致谢

我们要感谢 Joao Gante ([@joaogante](https://huggingface.co/joaogante))、Patrick von Platen ([@patrickvonplaten](https://huggingface.co/patrickvonplaten)) 和 Sylvain Gugger ([@sgugger](https://github.com/sgugger))，感谢他们在我们将本文中的对比搜索集成进 `transformers` 库的过程中给予的帮助和指导。
