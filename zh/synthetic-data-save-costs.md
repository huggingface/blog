---
title: "合成数据：利用开源技术节约资金、时间和减少碳排放" 
thumbnail: /blog/assets/176_synthetic-data-save-costs/thumbnail.png
authors:
- user: MoritzLaurer
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 合成数据: 利用开源技术节约资金、时间和减少碳排放 <!-- omit in toc -->

## 简单概括 <!-- omit in toc -->

你应该使用自己的模型，还是使用 LLM API？创建你自己的模型可以让你完全控制，但需要数据收集、训练和部署方面的专业知识。LLM API 使用起来更简单，但会将数据发送给第三方，并对提供商有强烈依赖。这篇博客让你可以将 LLM 的便利性与定制模型的控制性和效率相结合。

在一个关于识别新闻报道中投资者情绪的案例研究中，我们展示了如何使用开源 LLM 创建合成数据，并在几个步骤中训练你的定制模型。我们定制的 RoBERTa 模型可以分析大型新闻数据集，与 GPT4 相比性能一致都是 (94% acc 和 0.94 的 F1 macro)，我们只需 2.7 美元，排碳 0.12kg，延迟 0.13s ; 而 GPT4 要费 3061 美元，排碳约 735 到 1100 kg ，延迟多秒。这里提供了 [notebooks](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) 方便你用于自己的研究。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/table_pros_cons.png" alt="table_pros_cons" width=95%>
</p>

## 目录 <!-- omit in toc -->

- [1. 问题：你的使用案例没有数据](#1-问题-你的使用案例没有数据)
- [2. 解决方案：合成数据来高效蒸馏学生模型](#2-解决方案-合成数据来高效蒸馏学生模型)
- [3. 案例分析：监控金融情绪 ](#3-案例分析-监控金融情绪 )
  - [3.1 给 LLM 提示来标注你的数据](#31-给-LLM-提示来标注你的数据)
  - [3.2 将开源模型与专有模型进行比较](#32-将开源模型与专有模型进行比较)
  - [3.3 理解并验证你合成的数据](#33-理解并验证你合成的数据)
  - [3.4 使用 AutoTrain 调整你高效、专业的模型](#34-使用-AutoTrain-调整你高效、专业的模型)
  - [3.5 不同方法的利弊](#35-不同方法的利弊)
- [结论](#结论)

## 1. 问题: 你的使用案例没有数据

想象一下你的老板让你去建一个你公司的情感分析系统。你可以在 Hugging Face Hub 上找到 100,000+ 个数据集，这其中包含标题含有 “sentiment” 的字段的数据集， Twitter 上的情感数据集、诗歌中的情感或以希伯来语的情感数据集。这很好，但比如你在一家金融机构工作并且你追踪你投资组合中特定品牌的市场情绪，那么你可能会发现上面这些数据集没有一个有用的。虽然机器学习需要处理数百万的任务公司，但正巧别人已经收集并发布你公司的这个案例的数据集的可能性机会几乎为零。

由于对特定领域的数据集和模型的缺失，许多人尝试用更加通用的 LLM。这些模型都非常大和通用，以至于它们可以开箱即用，并实现令人印象深刻的准确度。它们的易于使用的 API 消除了对微调和对部署的专业知识的需求。但他们最主要的缺点是大小和可控性: 大小超过十亿到万亿的参数运行在计算集群中，控制权只在少数的几家公司手中。

## 2. 解决方案: 合成数据来高效蒸馏学生模型

在 2023 年，一个东西根本的改变了机器学习的蓝图，LLM 开始达到和人类数据标注者相同的水平。现在有大量的证据表明，最好的 LLM 比众包工人更胜一筹，并且在创建高质量 (合成的) 数据中部分达到了专家水平 (例如 [Zheng et al. 2023](https://arxiv.org/pdf/2306.05685.pdf), [Gilardi et al. 2023](https://arxiv.org/pdf/2303.15056.pdf), [He et al. 2023](https://arxiv.org/pdf/2303.16854.pdf))。这一进展的重要性怎么强调都不为过。创建定制模型的关键瓶颈在于招募和协调人工工作者以创造定制训练数据的资金、时间和专业知识需求。随着大型语言模型 (LLMs) 开始达到人类水平，高质量的数据标注现在可以通过 API 获得; 可复制的注释指令可以作为提示 prompt 发送; 合成数据几乎可以立即返回，唯一的瓶颈就剩计算能力了。

在 2024 年，这种方法将变得具有商业可行性，并提升开源对大中小企业的重要性。在 2023 年的大部分时间里，由于 LLM API 提供商的限制性商业条款，LLMs 的商业用途在标注数据方面被阻止。随着像 [Mistral](https://mistral.ai/) 的 [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 这样的模型的推出，LLM 数据标注和合成数据现在对商业用途开放。[Mixtral 的表现与 GPT3.5 相当](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)，并且由于它的 Apache 2.0 许可证，其合成数据输出可以作为商业用例中较小、专业化的模型 (“学生”) 的训练数据。这篇博客提供了一个示例，这将显著加快你自己的定制模型的创建速度，同时大幅降低长期推理成本。

## 3. 案例分析: 监控金融情绪

想象你是一个数据科学家，正在为一家大型投资公司工作。你的任务是监控经济新闻情绪，以帮助公司做出投资决策。最近，你有两个主要选择:

1. 你可以微调你自己的模型。这需要编写标注指令，创建标注接口，招人，引入质量保证措施以处理低质量数据，在这个数据上微调模型，并部署。
2. 或者，你可以按照指令将数据发送到 LLM API。你完全跳过微调和部署步骤，将数据分析过程简化为编写指令 (提示)，然后发送给 API 背后的“LLM 标注器”。在这种情况下，LLM API 就是你的最终推理解决方案，你直接使用 LLM 的输出进行分析。

尽管选项 2 在推理时间上更贵，并且需要你发送敏感数据到第三方，但选项 2 比选项 1 更容易设置，因此被许多开发人员使用。

在 2024 年，合成数据将提供第三个选项: 结合选项 1 的成本效益与选项 2 的简易性。你可以使用一个 LLM (老师模型) 去标注一个你的小数据样本，并在这个数据集上微调一个小的，高效的语言模型 (学生模型)。这种方法可以在几步内执行完成。

### 3.1 给 LLM 提示来标注你的数据

我们使用 [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) 情感数据集作为示例，但你可以将代码适配到任何其他用例。financial_phrasebank 任务是一个 3 类分类任务，其中 16 位专家从投资者视角对芬兰公司金融新闻中的句子进行“积极”/“消极”/“中性”标注 ( [Malo et al. 2013](https://arxiv.org/pdf/1307.5336.pdf) )。例如，数据集中包含这样一句话: “对于 2010 年最后一个季度，Componenta 的净销售额翻倍，达到 1.31 亿欧元，而去年同期为 7600 万欧元”，标注者从投资者视角将其归类为“积极”。

我们首先安装一些必需的库。

```python
!pip install datasets # for loading the example dataset
!pip install huggingface_hub # for secure token handling
!pip install requests # for making API requests
!pip install scikit-learn # for evaluation metrics
!pip install pandas # for post-processing some data
!pip install tqdm # for progress bars
```

然后，我们可以下载带有专家标注的示例数据集。

```python
from datasets import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree", split='train')

# create a new column with the numeric label verbalised as label_text (e.g. "positive" instead of "0")
label_map = {
    i: label_text
    for i, label_text in enumerate(dataset.features["label"].names)
}

def add_label_text(example):
    example["label_text"] = label_map[example["label"]]
    return example

dataset = dataset.map(add_label_text)

print(dataset)
# Dataset({
# features: ['sentence', 'label', 'label_text'],
# num_rows: 2264
#})
```

现在我们写一个短的标注指令，针对 `financial_phrasebank` 任务，并将其格式化为一个 LLM 提示。这个提示类似于你通常提供给众包工人的指令。

```python
prompt_financial_sentiment = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to analyze the sentiment in the TEXT below from an investor perspective and label it with only one the three labels:
positive, negative, or neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company.

Do not provide any explanations and only respond with one of the labels as one word: negative, positive, or neutral

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
Label: positive
Text: The company generated net sales of 11.3 million euro this year.
Label: neutral
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
Label: negative

Your TEXT to analyse:
TEXT: {text}
Label: """
```

这个标注指令现在可以被传递给 LLM API。对于这个例子，我们使用免费 Hugging Face [无服务的推理 API](https://huggingface.co/docs/api-inference/index)。这个 API 是测试流行模型的理想选择。请注意，如果你发送次数过多，尤其是分享给过多用户，你可能会遇到速率限制。对于更大的工作流，我们推荐创建一个 [专用推理端点](https://huggingface.co/docs/inference-endpoints/index)。专用推理端点对于你自己的付费 API 尤为重要，特别是你可以灵活的控制开和关。

我们登录 `huggingface_hub` 库，简单安全的填入我们的 API token。或者，你也可以定义你自己的 token 作为环境变量。(详情可以参考 [文档](https://huggingface.co/docs/huggingface_hub/quick-start#authentication))。

```python
# you need a huggingface account and create a token here: https://huggingface.co/settings/tokens
# we can then safely call on the token with huggingface_hub.get_token()
import huggingface_hub
huggingface_hub.login()
```

我么定义一个简单的 `generate_text` 函数，用于发送我们的提示 prompt 和数据到 API。

```python
import os
import requests

# Choose your LLM annotator
# to find available LLMs see: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.list_deployed_models
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

# docs on different parameters: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
generation_params = dict(
    top_p=0.90,
    temperature=0.8,
    max_new_tokens=128,
    return_full_text=False,
    use_cache=False
)

def generate_text(prompt=None, generation_params=None):
    payload = {
        "inputs": prompt,
        "parameters": {**generation_params}
    }
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {huggingface_hub.get_token()}"},
        json=payload
    )
    return response.json()[0]["generated_text"]
```

作为 LLM 可能不会总是返回标签相同的标准化格式，我们还可以定义一个短 `clean_output` 函数，将 LLM 从字符串输出映射到我们的三个可能标签。

```python
labels = ["positive", "negative", "neutral"]

def clean_output(string, random_choice=True):
    for category in labels:
        if category.lower() in string.lower():
            return category
    # if the output string cannot be mapped to one of the categories, we either return "FAIL" or choose a random label
    if random_choice:
        return random.choice(labels)
    else:
        return "FAIL"
```

我们现在可以将我们的文本发送给 LLM 进行标注。下面的代码将每一段文本发送到 LLM API，并将文本输出映射到我们的三个清晰类别。注意: 在实际操作中，逐个文本迭代并将它们分别发送到 API 是非常低效的。API 可以同时处理多个文本，你可以异步地批量向 API 发送文本来显著加快 API 调用速度。你可以在本博客的 [复现仓库](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) 中找到优化后的代码。

```python
output_simple = []
for text in dataset["sentence"]:
    # add text into the prompt template
    prompt_formatted = prompt_financial_sentiment.format(text=text)
    # send text to API
    output = generate_text(
        prompt=prompt_formatted, generation_params=generation_params
    )
    # clean output
    output_cl = clean_output(output, random_choice=True)
    output_simple.append(output_cl)
```

基于这个输出，我么可以计算指标来查看模型在不对其进行训练的情况下是否准确地完成了任务。

```python
from sklearn.metrics import classification_report

def compute_metrics(label_experts, label_pred):
    # classification report gives us both aggregate and per-class metrics
    metrics_report = classification_report(
        label_experts, label_pred, digits=2, output_dict=True, zero_division='warn'
    )
    return metrics_report

label_experts = dataset["label_text"]
label_pred = output_simple

metrics = compute_metrics(label_experts, label_pred)
```

基于简单的提示 prompt，LLM 正确分类了 91.6% 的文本 (0.916 准确率和 0.916 F1 macro)。考虑到它没有训练来完成这个具体任务，这相当不错。

我们通过使用两个简单的提示 Prompt 技巧来进一步提升精度: 思维链 COT 和  自我一致 SC。CoT 要求模型首先对正确的标签进行推理，然后再做出标注决策，而不是立即决定正确的标签。SC 意味着多次向同一个 LLM 发送相同文本的相同提示。SC 有效地为 LLM 提供了针对每段文本的多条不同的推理路径，如果 LLM 回应“积极”两次和“中性”一次，我们选择多数 (“积极”) 作为正确的标签。这是我们为 CoT 和 SC 更新的提示:

```python
prompt_financial_sentiment_cot = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to briefly analyze the sentiment in the TEXT below from an investor perspective and then label it with only one the three labels:
positive, negative, neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company.

You first reason step by step about the correct label and then return your label.

You ALWAYS respond only in the following JSON format: {{"reason": "...", "label": "..."}}
You only respond with one single JSON response.

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
JSON response: {{"reason": "An increase in operating profit is positive for investors", "label": "positive"}}
Text: The company generated net sales of 11.3 million euro this year.
JSON response: {{"reason": "The text only mentions financials without indication if they are better or worse than before", "label": "neutral"}}
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
JSON response: {{"reason": "A decrease in profit is negative for investors", "label": "negative"}}

Your TEXT to analyse:
TEXT: {text}
JSON response: """
```

这是一个 JSON 提示，我们要求 LLM 返回一个结构化的 JSON 字符串，其中 “reason” 作为一个键，“label” 作为另一个键。JSON 的主要优点是我们可以将其解析为 Python 字典，然后提取 “label” 。如果我们想了解 LLM 选择这个标签的原因，我们也可以提取 “reason”。

`process_output_cot` 函数解析 LLM 返回的 JSON 字符串，并且如果 LLM 没有返回有效的 JSON，它会尝试使用上面定义的 `clean_output` 函数通过简单的字符串匹配来识别标签。

```python
import ast

def process_output_cot(output):
    try:
        output_dic = ast.literal_eval(output)
        return output_dic
    except Exception as e:
        # if json/dict parse fails, do simple search for occurance of first label term
        print(f"Parsing failed for output: {output}, Error: {e}")
        output_cl = clean_output(output, random_choice=False)
        output_dic = {"reason": "FAIL", "label": output_cl}
        return output_dic
```

现在，我们可以使用上面新的提示重复使用我们的 `generate_text` 函数，用 `process_output_cot` 处理 JSON 的 COT 输出，并且为了 SC 多次发送每个提示。

```python
self_consistency_iterations = 3

output_cot_multiple = []
for _ in range(self_consistency_iterations):
    output_lst_step = []
    for text in tqdm(dataset["sentence"]):
        prompt_formatted = prompt_financial_sentiment_cot.format(text=text)
        output = generate_text(
            prompt=prompt_formatted, generation_params=generation_params
        )
        output_dic = process_output_cot(output)
        output_lst_step.append(output_dic["label"])

    output_cot_multiple.append(output_lst_step)
```

对于每段文本，我们现在的 LLM 标注器有了三次尝试来识别正确标签，并采用了三种不同的推理路径。下面的代码从这三条路径中选择了多数标签。

```python
import pandas as pd
from collections import Counter

def find_majority(row):
    # Count occurrences
    count = Counter(row)
    # Find majority
    majority = count.most_common(1)[0]
    # Check if it's a real majority or if all labels are equally frequent
    if majority[1] > 1:
        return majority[0]
    else: # in case all labels appear with equal frequency
        return random.choice(labels)

df_output = pd.DataFrame(data=output_cot_multiple).T

df_output['label_pred_cot_multiple'] = df_output.apply(find_majority, axis=1)
```

现在，我们可以比较我们的改进的 LLM 标签与专家标签，并计算指标。

```python
label_experts = dataset["label_text"]
label_pred_cot_multiple = df_output['label_pred_cot_multiple']

metrics_cot_multiple = compute_metrics(label_experts, label_pred_cot_multiple)
```

CoT 和 SC 将性能提升到了 94.0% 的准确率和 0.94 的 F1 macro。通过给模型时间来考虑其标签决策，并给予它多次尝试，我们提升了性能。请注意，CoT 和 SC 需要额外的计算资源。我们本质上是在用计算资源购买标注的准确性。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/fig_mixtral.png" alt="fig_mixtral" width=95%>
</p>

现在，我们通过这些简单的 LLM API 调用创建了一个合成训练数据集。我们在做出标签决策之前，让 LLM 尝试了三种不同的推理路径来标注每段文本。结果是，这些标签与人类专家的高度一致，并且我们得到了一个高质量的数据集，可以用来训练更高效、更专业的模型。

```python
df_train = pd.DataFrame({
    "text": dataset["sentence"],
    "labels": df_output['label_pred_cot_multiple']
})

df_train.to_csv("df_train.csv")
```

请注意，在这篇博客文章的 [完整复现脚本](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) 中，我们还将仅基于专家标注创建一个测试集，以评估所有模型的质量。所有指标始终基于这个人类专家测试集。

### 3.2 将开源模型与专有模型进行比较

使用开源的 Mixtral 模型创建的这种数据的主要优势在于，这些数据在商业上完全可用，且没有法律上的不确定性。例如，使用 OpenAI API 创建的数据受 [OpenAI 商业条款](https://openai.com/policies/business-terms) 的约束，这些条款明确禁止将模型输出用于训练与他们的产品和服务竞争的模型。这些条款的法律价值和意义尚不明确，但它们为使用 OpenAI 模型合成的数据训练模型的商业使用引入了法律上的不确定性。任何使用合成数据训练的更小、更高效的模型都可能被视为竞争者，因为它减少了对 API 服务的依赖。

开源的 Mistral 的 `Mixtral-8x7B-Instruct-v0.1` 与 OpenAI 的 GPT3.5 和 GPT4 之间合成的数据质量如何比较呢？我们使用 `gpt-3.5-turbo-0613` 和 `gpt-4-0125-preview` 运行了上述相同的流程和提示，并在下表中报告了结果。我们看到，Mixtral 在这个任务上的表现优于 GPT3.5，并且与 GPT4 相当，这取决于提示类型。(我们没有显示新版本的 gpt-3.5-turbo-0125 的结果，因为不知何故，这个模型的表现比旧版本的默认 gpt-3.5-turbo-0613 要差)。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/fig_mixtral_gpt.png" alt="fig_mixtral_gpt" width=95%>
</p>

请注意，这并不意味着 Mixtral 总是比 GPT3.5 更好，与 GPT4 相当。GPT4 在多个基准测试上的表现更好。主要想表达的是，开源模型现在可以创建高质量的合成数据。

### 3.3 理解并验证你合成的数据

所有这些在实践中意味着什么呢？到目前为止，结果只是由一些黑盒 LLM 标注的数据。我们只能计算指标，因为我们有来自示例数据集的专家标注的参考数据。如果在真实世界的场景中没有专家标注，我们如何信任 LLM 的标注呢？

在实践中，无论你使用哪种标注器 (人类标注或 LLM )，你只能信任你自己验证过的数据。指令/提示总是包含一定程度的模糊性。即使是一个完美智能的标注也可能犯错误，并且在面对通常模糊的现实世界数据时，必须做出不明确的决定。

幸运的是，随着近年来开源工具的出现，数据验证变得更加简单: [Argilla](https://argilla.io/) 提供了一个免费的界面，用于验证和清理非结构化的 LLM 输出; [LabelStudio](https://labelstud.io/) 使你能够以多种方式标注数据; [CleanLab](https://cleanlab.ai/) 提供了一个用于标注和自动清理结构化数据的界面; 对于快速和简单的验证，仅在简单的 Excel 文件中标注也可能是可以的。

花些时间标注文本，以了解数据和其模糊性，这是非常重要的。你会很快发现模型犯了一些错误，但也会有几个例子，正确的标签是不明确的，有些文本你更同意 LLM 的决定，而不是创建数据集的专家。这些错误和模糊性是数据集创建的正常部分。实际上，只有极少数现实世界的任务中，人类专家的基线是完全一致的。这是一个古老的见解，最近被机器学习文献“重新发现”，即人类数据是一个有缺陷的金标准 ([Krippendorf 2004](https://books.google.de/books/about/Content_Analysis.html?id=q657o3M3C8cC&redir_esc=y), [Hosking et al. 2024](https://arxiv.org/pdf/2309.16349.pdf))。

在标注界面不到一个小时的时间里，我们更好地了解了我们的数据并纠正了一些错误。然而，为了可复现性，以及展示纯粹合成数据的质量，我们在下一步继续使用未清理的 LLM 标注。

### 3.4 使用 AutoTrain 调整你高效、专业的模型

到目前为止，我们已经经历了一个标准的流程，即通过 API 提示 LLM 并验证输出。现在，进入一个额外的步骤，以实现显著的资源节约: 我们将在 LLM 的合成数据上微调一个更小、更高效和专业化的 LM。这个过程也被称为“蒸馏”，其中较大模型的输出 (“教师”) 用于训练一个较小的模型 (“学生”)。虽然这听起来很复杂，但它本质上只意味着我们使用数据集中的原始 `text` ，并将 LLM 的预测作为我们微调的 `labels` 。如果你以前训练过分类器，你知道，使用 `transformers` 、 `sklearn` 或其他库，你只需要这两个列来训练一个分类器。

我们使用 Hugging Face 的 [AutoTrain](https://huggingface.co/autotrain) 解决方案使这个过程更加简单。AutoTrain 是一个无代码界面，它使你能够上传一个带有标记数据的 `.csv` 文件，该服务然后使用它为你自动微调模型。这消除了为训练你自己的模型编写代码或深入微调专业知识的需求。

在 Hugging Face 网站上，我们首先在顶部点击 “Spaces”，然后点击 “Create new Space”。然后选择 “Docker”>“AutoTrain” 并选择一个小型 A10G GPU，每小时成本为 $1.05。AutoTrain 的空间将然后初始化。然后，我们可以通过界面上传我们的合成训练数据和专家测试数据，并调整不同的字段，如下面的截图所示。填写所有内容后，我们可以点击 “Start Training”，并在 Space 的日志中跟踪训练过程。仅在 1811 个数据点上训练一个小型的 RoBERTa-base 模型 (~0.13 B 参数) 非常快，可能不需要超过几分钟。一旦训练完成，模型将自动上传到你的 HF 个人资料。一旦训练完成，space 就会停止，整个过程最多应该需要 15 分钟，成本不到 $1。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/autotrain.png" alt="autotrain" width=95%>
</p>

如果你愿意，你也可以完全在你自己的硬件上本地使用 AutoTrain，请参阅我们的 [文档](https://huggingface.co/docs/autotrain/index)。高级用户当然总是可以编写自己的训练脚本，但对于这些默认的超参数，AutoTrain 的结果对于许多分类任务来说应该足够了。

我们最终微调的约 0.13B 参数的 RoBERTa-base 模型与更大的 LLM 相比表现如何？下图显示，在 1811 个文本上微调的自定义模型达到了 94% 的准确率，与它的老师 Mixtral 和 GPT4 一样！一个小型模型当然无法与一个更大型的 LLM 出厂即战，但通过在一些高质量数据上进行微调，它可以达到在它所专长的任务上与大型 LLM 相同的性能水平。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/fig_mixtral_gpt_roberta.png" alt="fig_mixtral_gpt_roberta" width=95%>
</p>

### 3.5 不同方法的利弊

我们在开始时讨论的三种方法的总体优缺点是什么:(1) 手动创建你自己的数据和模型，(2) 仅使用 LLM API，或者 (3) 使用 LLM API 创建用于专业模型的合成数据？下面的表格显示了不同因素之间的权衡，我们将在下面根据我们的示例数据集讨论不同的指标。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/176_synthetic-data-save-costs/table_pros_cons.png" alt="table_pros_cons" width=95%>
</p>

让我们从任务性能开始。如上所示，专业模型与更大型的 LLM 表现相当。微调后的模型只能执行我们训练它执行的特定任务，但它在执行这个特定任务方面表现非常好。要创建更多训练数据来将模型适应到新的领域或更复杂的任务是轻而易举的。多亏了 LLM 的合成数据，由于缺乏专业数据而导致的低性能不再是问题。

其次，计算成本和推理速度。实际中的主要计算成本将是推理，即在训练后运行模型。假设在你的生产用例中，你需要在给定时间段内处理 100 万句话。我们的微调 RoBERTa-base 模型在一个带有 16GB RAM 的小型 T4 GPU 上运行效率很高，在 [推理端点](https://ui.endpoints.huggingface.co/) 上的成本为每小时 $0.6。它具有 0.13s 的延迟和每秒 61 句话的吞吐量 ( `batch_size=8` )。这使得处理 100 万句话的总成本为 $2.7。

使用 GPT 模型，我们可以通过计算 token 来计算推理成本。处理 100 万句话的 toekn 将花费 GPT3.5 约  $153，GPT4 约  $3061。这些模型的延迟和吞吐量更加复杂，因为它们根据一天中的当前服务器负载而变化。任何使用 GPT4 的人都清楚，延迟通常可以是多秒，并且受到速率限制。请注意，速度是任何 LLM (API) 的问题，包括开源 LLM。许多生成型 LLM 由于过大而无法快速运行。

训练计算成本往往不太相关，因为 LLM 可以不经过微调直接使用，且小型模型的微调成本相对较小 (微调 RoBERTa-base 的成本不到 $1)。只有在需要将大型生成型 LLM 专门化以执行特定生成任务时，才需要投资从头开始预训练模型。当微调一个更大的生成型 LLM 以使其适应特定生成任务时，训练成本可能变得相关。

第三，在时间和专业知识方面的投资。这是 LLM API 的主要优势。与手动收集数据、微调定制模型和部署相比，向 API 发送指令要容易得多。这正是使用 LLM API 创建合成数据变得重要的地方。创建良好的训练数据变得显著更容易。然后，微调和部署可以由 AutoTrain 等服务和专业推理端点处理。

第四，控制。这可能是 LLM API 的主要缺点。按设计，LLM API 使你依赖于 LLM API 提供商。你需要将敏感数据发送到别人的服务器，并且你无法控制系统的可靠性和速度。自己训练模型可以让你选择如何和在哪里部署它。

最后，环境影响。由于缺乏有关模型架构和硬件基础设施的信息，很难估计 GPT4 等封闭模型的能源消耗和二氧化碳排放。我们找到的 [最佳 (但非常粗略) 估计](https://towardsdatascience.com/chatgpts-energy-use-per-query-9383b8654487) 显示，GPT4 查询的能源消耗约为 0.0017 至 0.0026 千瓦时。这将是分析 100 万句话的大致 1700 至 2600 千瓦时。根据 [EPA 二氧化碳当量计算器](https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator)，这相当于 0.735 至 1.1 公吨二氧化碳，或平均汽车行驶 1885 至 2883 英里。请注意，实际二氧化碳排放可以根据 LLM 特定计算区域的能源混合而有很大差异。与我们的自定义模型相比，这个估计要容易得多。使用自定义模型分析 100 万句话，在一个 T4 GPU 上大约需要 4.52 小时，在 US East N. Virginia 的 AWS 服务器上，这相当于大约 0.12 公斤二氧化碳 (见 [ML CO2 Impact calculator](https://mlco2.github.io/impact/))。与具有 (据称) 8x220B 参数的通用 LLM 相比，运行一个专门化的模型 (约 0.13B 参数) 的效率低下得多。

## 结论

我们已经展示了使用 LLM 创建合成数据来训练一个更小、更高效的模型的巨大好处。虽然这个例子只处理了投资者情绪分类，但同样的流程可以应用于许多其他任务，从其他分类任务 (例如，客户意图检测或有害内容检测)，到 token 分类 (例如，命名实体识别或 PII 检测)，或生成任务 (例如，总结或问答)。

在 2024 年，公司创建自己的高效模型、控制自己的数据和基础设施、减少二氧化碳排放、节省计算成本和时间，而不必妥协准确性的难度从未如此之低。

现在，亲自动手尝试一下！你可以在本博客文章中找到所有数字的完整复现代码，以及更高效的异步函数和批量 API 调用的代码，在 [复现仓库](https://github.com/MoritzLaurer/synthetic-data-blog/tree/main) 中。我们邀请你复制并适配我们的代码以应用于你的用例！