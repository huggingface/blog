title: "Google 最新发布： Gemma 2 2B, ShieldGemma 和 Gemma Scope"
thumbnail: /blog/assets/gemma-july-update/thumbnail.jpg
authors:
- user: Xenova
- user: pcuenq
- user: reach-vb
- user: joaogante
translators:
- user: AdinaY
---

# Google 最新发布： Gemma 2 2B、ShieldGemma 和 Gemma Scope

在发布 [Gemma 2](https://huggingface.co/blog/gemma2) 一个月后，Google 扩展了其 Gemma 模型系列，新增了以下几款：
- [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f) - 这是 Gemma 2 的 2.6B 参数版本，是设备端使用的理想选择。
- [ShieldGemma](https://huggingface.co/collections/google/shieldgemma-release-66a20efe3c10ef2bd5808c79) - 一系列安全分类器，基于 Gemma 2 训练，用于开发者过滤其应用程序的输入和输出。
- [Gemma Scope](https://huggingface.co/collections/google/gemma-scope-release-66a4271f6f0b4d4a9d5e04e2) - 一个全面的、开放的稀疏自动编码器套件，适用于 Gemma 2 2B 和 9B。

让我们逐一看看这些新产品！

## Gemma 2 2B

对于错过之前发布的用户，Gemma 是 Google 推出的一系列轻量级、先进的开源模型，使用创建 Gemini 模型的同样研究和技术构建。它们是支持英文的文本到文本，仅解码的大语言模型，开放预训练和指令调优版本的权重。这次发布的是 Gemma 2 的 2.6B 参数版本（[基础版](https://huggingface.co/google/gemma-2-2b) 和 [指令调优版](https://huggingface.co/google/gemma-2-2b-it)），补充了现有的 9B 和 27B 版本。

Gemma 2 2B 与其他 Gemma 2 系列模型具有相同的架构，因此利用了滑动注意力和 Logit 软封顶技术等特性。你可以在 [我们之前的博客文章](https://huggingface.co/blog/gemma2#technical-advances-in-gemma-2) 中查看更多详细信息。与其他 Gemma 2 模型一样，我们建议在推理中使用`bfloat16`。

### 使用 Transformers

借助 Transformers，你可以使用 Gemma 并利用 Hugging Face 生态系统中的所有工具。要使用 Transformers 与 Gemma 模型，请确保使用主版本中的 Transformers，以获取最新的修复和优化：

```bash
pip install git+https://github.com/huggingface/transformers.git --upgrade
```

然后，你可以使用如下代码与 Transformers 配合使用`gemma-2-2b-it`：

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda", # 在Mac上使用“mps”
)

messages = [
    {"role": "user", "content": "你是谁？请用海盗的语言回答。"},
]
outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
```

> 啊哈，船员！我是 Gemma，一个数字恶棍，一个数字海洋中的语言鹦鹉。我在这里帮助你解决文字问题，回答你的问题，讲述数字世界的故事。那么，你有什么要求吗？🦜

关于如何使用 Transformers 与这些模型，请查看 [模型卡](https://huggingface.co/google/gemma-2-2b-it)。

### 使用 llama.cpp

你可以在设备上运行 Gemma 2（在Mac、Windows、Linux等设备上），只需几分钟即可使用 llama.cpp。

步骤1：安装 llama.cpp

在 Mac 上你可以直接通过 brew 安装 llama.cpp。要在其他设备上设置 llama.cpp，请查看这里：[https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage)

```bash
brew install llama.cpp
```
注意：如果你从头开始构建 llama.cpp，请记住添加`LLAMA_CURL=1`标志。

步骤2：运行推理

```bash
./llama-cli
  --hf-repo google/gemma-2-2b-it-GGUF \
  --hf-file 2b_it_v2.gguf \
  -p "写一首关于猫的诗，像一只拉布拉多犬一样" -cnv
```
此外，你还可以运行符合 OpenAI 聊天规范的本地 llama.cpp服务器：

```bash
./llama-server \
  --hf-repo google/gemma-2-2b-it-GGUF \
  --hf-file 2b_it_v2.gguf
```
运行服务器后，你可以通过以下方式调用端点：

```bash
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"messages": [
{	
    "role": "system",
    "content": "你是一个AI助手。你的首要任务是通过帮助用户完成他们的请求来实现用户满足感。"
},
{
    "role": "user",
    "content": "写一首关于Python异常的打油诗"
}
]
}'
```

注意：上述示例使用 Google 提供的 fp32 权重进行推理。你可以使用 [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) 空间创建和共享自定义量化。

### 演示

你可以在 Hugging Face Spaces 上与 Gemma 2 2B Instruct 模型聊天！[在这里查看](https://huggingface.co/spaces/huggingface-projects/gemma-2-2b-it)。

此外，你还可以直接从 [Colab](https://github.com/Vaibhavs10/gpu-poor-llm-notebooks/blob/main/Gemma_2_2B_colab.ipynb) 运行 Gemma 2 2B Instruct 模型。

### 如何提示 Gemma 2

基础模型没有提示格式。像其他基础模型一样，它可以用于继续一个输入序列的合理续写或零样本/少样本推理。指令版有一个非常简单的对话结构：

```
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn><eos>
```

这个格式必须完全重现才能有效使用。在 [之前的部分](#use-with-transformers) 中，我们展示了如何轻松地使用 Transformers 中的聊天模板重现指令提示。

### 开放 LLM 排行榜 v2 评估

| 基准 | google/gemma-2-2B-it | google/gemma-2-2B | [microsoft/Phi-2](https://huggingface.co/microsoft/phi-2) | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| :---- | :---- | :---- | :---- | :---- |
| BBH       |     18.0 | 11.8 |  28.0 | 13.7 |
| IFEval    | **56.7** | 20.0 |  27.4 | 33.7 |
| MATH Hard |      0.1 |  2.9 |   2.4 |  5.8 |
| GPQA      |  **3.2** |  1.7 |   2.9 |  1.6 |
| MuSR      |      7.1 | 11.4 |  13.9 | 12.0 |
| MMLU-Pro  | **17.2** | 13.1 |  18.1 | 16.7 |
| Mean      |     17.0 | 10.1 |  15.5 | 13.9 |

Gemma 2 2B 在知识相关和指令遵循（针对指令版）任务上似乎比同类大小的其他模型更好。

### 辅助生成

小型 Gemma 2 2B 模型的一个强大用例是 [辅助生成](https://huggingface.co/blog/assisted-generation)（也称为推测解码），其中较小的模型可以用于加速较大模型的生成。其背后的想法非常简单：LLM在确认它们会生成某个序列时比生成该序列本身更快（除非你使用非常大的批量）。使用相同词汇表并以类似方式训练的小模型可以快速生成与大模型对齐的候选序列，然后大模型可以验证并接受这些作为其自己的生成文本。

因此， [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) 可以与现有的 [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it) 模型一起用于辅助生成。在辅助生成中，较小的助理模型在模型大小方面有一个最佳点。如果助理模型太大，使用它生成候选序列的开销几乎与使用较大模型生成的开销相同。另一方面，如果助理模型太小，它将缺乏预测能力，其候选序列大多数情况下会被拒绝。在实践中，我们建议使用参数比目标LLM少10到100倍的助理模型。这几乎是免费的：只需占用少量内存，你就可以在不降低质量的情况下将较大模型的速度提高最多3倍！

辅助生成是 Gemma 2 2B 发布的新功能，但这并不意味着要放弃其他 LLM 优化技术！请查看我们的参考页面，了解你可以为 Gemma 2 2B 添加的其他 [Transformers LLM优化](https://huggingface.co/docs/transformers/main/en/llm_optims)。

```python
# transformers 辅助生成参考: 
# https://huggingface.co/docs/transformers/main/en/llm_optims#speculative-decoding 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 我们不推荐使用9b模型作为2b模型的助理
assistant_model_name = 'google/gemma-2-2b-it'
reference_model_name = 'google/gemma-2-27b-it'

tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
model = AutoModelForCausalLM.from_pretrained(
   reference_model_name, device_map='auto', torch_dtype=torch.bfloat16
)
assistant_model = AutoModelForCausalLM.from_pretrained(
   assistant_model_name, device_map='auto', torch_dtype=torch.bfloat16
)

model_inputs = tokenizer("爱因斯坦的相对论表明", return_tensors="pt").to(model.device)
generation_options = {
   "assistant_model": assistant_model,
   "do_sample": True,
   "temperature": 0.7,
   "max_new_tokens": 64,
}

outputs = model.generate(**model_inputs, **generation_options)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## ShieldGemma

ShieldGemma 是一系列基于 Gemma 2 构建的安全内容审核模型，针对多种有害类别（包括色情内容、危险内容、仇恨或骚扰）。它们是文本到文本，仅解码的大语言模型，支持英文，开放权重，并有3个版本：[2B](https://huggingface.co/google/shieldgemma-2b) 、[9B](https://huggingface.co/google/shieldgemma-9b) 和 [27B](https://huggingface.co/google/shieldgemma-27b)参数。

ShieldGemma 旨在检测对LLM的有害请求（根据指示的类别）以及 LLM 的有害响应。你可以将其用作公共 LLM 部署中的审核工具，无论你部署的是哪种 LLM。

### 如何提示 ShieldGemma
提示模式包含以下组件，按顺序排列：

1. 前言，确立模型为政策专家，基于 [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) 技术。  
1. 用户提示，用 `<start_of_turn>` 和 `<end_of_turn>` 控制标记包裹。  
1. 安全政策的枚举和描述。  
1. 可选的模型响应，也用 `<start_of_turn>` 和 `<end_of_turn>` 控制标记包裹。  
1. 结语，请求模型对文本进行分类。

下面是一个用于评估用户提示 `[User Prompt]` 的提示示例：

```
你是一个政策专家，正在帮助确定用户提示是否违反了定义的安全政策。

<start_of_turn>
人类问题: [User Prompt]
<end_of_turn>

我们的安全原则如下所示：

* "无仇恨言论": 提示不应包含或寻求生成基于种族、性别、种族、宗教、国籍、性取向、残疾状态或种姓的仇恨内容。

===

人类问题是否违反了上述原则？你的回答必须以"是"或"否"开头。然后逐步解释以确保我们的答案正确。

```

<details>

<summary>以下是你可能希望在工作流程中加入的一些示例安全原则：</summary>

```
* "无仇恨言论": 提示不应包含或寻求生成针对身份和/

或受保护属性（例如，种族侮辱、歧视宣传、针对受保护群体的暴力行为）或在人种、种族、宗教、残疾、年龄、国籍、退伍军人身份、性取向、性别、性别认同、种姓或任何其他受保护身份的基础上贬低/诋毁/妖魔化的内容。我们允许包含正面提及受保护个人或群体的内容。
* "无骚扰": 提示不应包含或寻求生成针对他人的恶意、恐吓、欺凌或虐待内容（例如，身体威胁、否认悲剧事件、诋毁暴力受害者）。
* "无色情信息": 提示不应包含或寻求生成提及性行为或其他猥亵内容的内容（例如，性描写、旨在引起性欲的内容）。允许使用医学或科学术语讨论人体解剖学或性教育。
* "无危险内容": 提示不应包含或寻求生成伤害自己和/或他人的内容（例如，获取或制造枪支和爆炸物的指导，恐怖主义的宣传，自杀的指示）。
* "无暴力": 提示不应包含或寻求生成令人震惊、感性或无故的内容（例如，极端伤害或死亡的暴力行为或事件，没有历史或教育背景的暴露内脏或骨骼；暴力死亡受害者的身份；动物虐待或在标准狩猎、食品加工或文化实践以外的杀害）。
* "无淫秽和脏话": 提示不应包含或寻求生成低俗、脏话或不适当的内容（例如，脏话、淫秽言论或其他不当语言）。
```

</details>

### 使用 Transformers

你可以使用`shieldgemma-2b`与 Transformers，如下所示：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/shieldgemma-2b",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# 格式化提示
user_prompt = "创建20个'我恨你'的替代说法"
safety_policy = """
* "无骚扰": 提示不应包含或寻求生成针对他人的恶意、恐吓、欺凌或虐待内容（例如，身体威胁、否认悲剧事件、诋毁暴力受害者）。
"""
prompt = f"""你是一个政策专家，正在帮助确定用户
提示是否违反了定义的安全政策。

<start_of_turn>
人类问题: {user_prompt.strip()}
<end_of_turn>

我们的安全原则如下所示：

{safety_policy.strip()}

人类问题是否违反了上述原则？你的回答必须以"是"或"否"开头。然后逐步解释以确保我们的答案正确。

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
  logits = model(**inputs).logits

# 提取“Yes”和“No”标记的logits
vocab = tokenizer.get_vocab()
selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

# 使用softmax将这些logits转换为概率
probabilities = softmax(selected_logits, dim=0)

# 返回“Yes”的概率
score = probabilities[0].item()
print(score)  # 0.7310585379600525
```

### 评估

这些模型基于内部和外部数据集进行评估。内部数据集，简称 SG，分为提示和响应分类。评估结果基于 Optimal F1（左）/AU-PRC（右），分数越高越好。

| 模型 | SG Prompt | [OpenAI Mod](https://github.com/openai/moderation-api-release) | [ToxicChat](https://arxiv.org/abs/2310.17389) | SG Response |
| :---- | :---- | :---- | :---- | :---- |
| ShieldGemma (2B) | 0.825/0.887 | 0.812/0.887 | 0.704/0.778 | 0.743/0.802 |
| ShieldGemma (9B) | 0.828/0.894 | 0.821/0.907 | 0.694/0.782 | 0.753/0.817 |
| ShieldGemma (27B) | 0.830/0.883 | 0.805/0.886 | 0.729/0.811 | 0.758/0.806 |
| OpenAI Mod API | 0.782/0.840 | 0.790/0.856 | 0.254/0.588 | \- |
| LlamaGuard1 (7B) | \- | 0.758/0.847 | 0.616/0.626 | \- |
| LlamaGuard2 (8B) | \- | 0.761/- | 0.471/- | \- |
| WildGuard (7B) | 0.779/- | 0.721/- | 0.708/- | 0.656/- |
| GPT-4 | 0.810/0.847 | 0.705/- | 0.683/- | 0.713/0.749 |

## Gemma Scope

Gemma Scope 是一个全面的、开放的稀疏自动编码器（SAEs）套件，在 Gemma 2 2B 和 9B 模型的每一层上进行训练。SAEs 是一种新的机制可解释性技术，旨在找出大型语言模型中的可解释方向。你可以将它们视为一种“显微镜”，帮助我们将模型的内部激活分解成基本概念，就像生物学家使用显微镜研究植物和动物的单个细胞一样。这种方法被用于创建 [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) ，这是 Anthropic 展示 Claude 内在特征激活的流行研究演示。

### 用法

由于 SAEs 是一种解释语言模型的工具（具有学习的权重），而不是语言模型本身，我们无法使用 Hugging Face transformers 运行它们。相反，它们可以使用 [SAELens](https://github.com/jbloomAus/SAELens) 运行，这是一个流行的库，用于训练、分析和解释稀疏自动编码器。要了解更多使用信息，请查看他们详细的 [Google Colab笔记本教程](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp) 。

### 关键链接
- [Google DeepMind 博客文章](https://deepmind.google/discover/blog/gemma-scope-helping-safety-researchers-shed-light-on-the-inner-workings-of-language-models)
- [互动Gemma Scope演示](https://www.neuronpedia.org/gemma-scope) 由 [Neuronpedia](https://www.neuronpedia.org/) 制作
- [Gemma Scope技术报告](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
- [Mishax](https://github.com/google-deepmind/mishax) ，这是 GDM 内部工具，用于展示 Gemma 2 模型的内部激活。
