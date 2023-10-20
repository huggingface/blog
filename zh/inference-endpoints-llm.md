---
title:  用 Hugging Face 推理端点部署 LLM
thumbnail: /blog/assets/155_inference_endpoints_llm/thumbnail.jpg
authors:
- user: philschmid
translators:
- user: innovation64
- user: zhongdongy
  proofreader: true
---

# 用 Hugging Face 推理端点部署 LLM


开源的 LLM，如 [Falcon](https://huggingface.co/tiiuae/falcon-40b)、[(Open-)LLaMA](https://huggingface.co/openlm-research/open_llama_13b)、[X-Gen](https://huggingface.co/Salesforce/xgen-7b-8k-base)、[StarCoder](https://huggingface.co/bigcode/starcoder) 或 [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base)，近几个月来取得了长足的进展，能够在某些用例中与闭源模型如 ChatGPT 或 GPT4 竞争。然而，有效且优化地部署这些模型仍然是一个挑战。

在这篇博客文章中，我们将向你展示如何将开源 LLM 部署到 [Hugging Face Inference Endpoints](https://ui.endpoints.huggingface.co/)，这是我们的托管 SaaS 解决方案，可以轻松部署模型。此外，我们还将教你如何流式传输响应并测试我们端点的性能。那么，让我们开始吧！

1. [怎样部署 Falcon 40B instruct 模型](#1-how-to-deploy-falcon-40b-instruct)
2. [测试 LLM 端点](#2-test-the-llm-endpoint)
3. [用 javascript 和 python 进行流响应传输](#3-stream-responses-in-javascript-and-python)

在我们开始之前，让我们回顾一下关于推理端点的知识。

## 什么是 Hugging Face 推理端点

[Hugging Face 推理端点](https://ui.endpoints.huggingface.co/) 提供了一种简单、安全的方式来部署用于生产的机器学习模型。推理端点使开发人员和数据科学家都能够创建 AI 应用程序而无需管理基础设施: 简化部署过程为几次点击，包括使用自动扩展处理大量请求，通过缩减到零来降低基础设施成本，并提供高级安全性。

以下是 LLM 部署的一些最重要的特性:

1. [简单部署](https://huggingface.co/docs/inference-endpoints/index): 只需几次点击即可将模型部署为生产就绪的 API，无需处理基础设施或 MLOps。
2. [成本效益](https://huggingface.co/docs/inference-endpoints/autoscaling): 利用自动缩减到零的能力，通过在端点未使用时缩减基础设施来降低成本，同时根据端点的正常运行时间付费，确保成本效益。
3. [企业安全性](https://huggingface.co/docs/inference-endpoints/security): 在仅通过直接 VPC 连接可访问的安全离线端点中部署模型，由 SOC2 类型 2 认证支持，并提供 BAA 和 GDPR 数据处理协议，以增强数据安全性和合规性。
4. [LLM 优化](https://huggingface.co/text-generation-inference): 针对 LLM 进行了优化，通过自定义 transformers 代码和 Flash Attention 来实现高吞吐量和低延迟。
5. [全面的任务支持](https://huggingface.co/docs/inference-endpoints/supported_tasks): 开箱即用地支持 🤗 Transformers、Sentence-Transformers 和 Diffusers 任务和模型，并且易于定制以启用高级任务，如说话人分离或任何机器学习任务和库。

你可以在 [https://ui.endpoints.huggingface.co/](https://ui.endpoints.huggingface.co/) 开始使用推理端点。

## 1. 怎样部署 Falcon 40B instruct

要开始使用，你需要使用具有文件付款方式的用户或组织帐户登录 (你可以在 **[这里](https://huggingface.co/settings/billing)** 添加一个)，然后访问推理端点 **[https://ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co/endpoints)**。

然后，点击“新建端点”。选择仓库、云和区域，调整实例和安全设置，并在我们的情况下部署 `tiiuae/falcon-40b-instruct` 。

![Select Hugging Face Repository](https://huggingface.co/blog/assets/155_inference_endpoints_llm/repository.png "Select Hugging Face Repository")

推理端点会根据模型大小建议实例类型，该类型应足够大以运行模型。这里是 `4x NVIDIA T4` GPU。为了获得 LLM 的最佳性能，请将实例更改为 `GPU [xlarge] · 1x Nvidia A100` 。

_注意: 如果无法选择实例类型，则需要 [联系我们](mailto:api-enterprise@huggingface.co?subject=Quota%20increase%20HF%20Endpoints&body=Hello,%0D%0A%0D%0AI%20would%20like%20to%20request%20access/quota%20increase%20for%20{INSTANCE%20TYPE}%20for%20the%20following%20account%20{HF%20ACCOUNT}.) 并请求实例配额。_

![Select Instance Type](https://huggingface.co/blog/assets/155_inference_endpoints_llm/instance-selection.png "Select Instance Type")

然后，你可以点击“创建端点”来部署模型。10 分钟后，端点应该在线并可用于处理请求。

## 2. 测试 LLM 端点

端点概览提供了对推理小部件的访问，可以用来手动发送请求。这使你可以使用不同的输入快速测试你的端点并与团队成员共享。这些小部件不支持参数 - 在这种情况下，这会导致“较短的”生成。

![Test Inference Widget](https://huggingface.co/blog/assets/155_inference_endpoints_llm/widget.png "Test Inference Widget")

该小部件还会生成一个你可以使用的 cURL 命令。只需添加你的 `hf_xxx` 并进行测试。

```python
curl https://j4xhm53fxl9ussm8.us-east-1.aws.endpoints.huggingface.cloud \
-X POST \
-d '{"inputs":"Once upon a time,"}' \
-H "Authorization: Bearer <hf_token>" \
-H "Content-Type: application/json"
```

你可以使用不同的参数来控制生成，将它们定义在有效负载的 `parameters` 属性中。截至目前，支持以下参数:

- `temperature`: 控制模型中的随机性。较低的值会使模型更确定性，较高的值会使模型更随机。默认值为 1.0。
- `max_new_tokens`: 要生成的最大 token 数。默认值为 20，最大值为 512。
- `repetition_penalty`: 控制重复的可能性。默认值为 `null` 。
- `seed`: 用于随机生成的种子。默认值为 `null` 。
- `stop`: 停止生成的 token 列表。当生成其中一个 token 时，生成将停止。
- `top_k`: 保留概率最高的词汇表 token 数以进行 top-k 过滤。默认值为 `null` ，禁用 top-k 过滤。
- `top_p`: 保留核心采样的参数最高概率词汇表 token 的累积概率，默认为 `null`
- `do_sample`: 是否使用采样; 否则使用贪婪解码。默认值为 `false` 。
- `best_of`: 生成 best_of 序列并返回一个最高 token 的 logprobs，默认为 `null` 。
- `details`: 是否返回有关生成的详细信息。默认值为 `false` 。
- `return_full_text`: 是否返回完整文本或仅返回生成部分。默认值为 `false` 。
- `truncate`: 是否将输入截断到模型的最大长度。默认值为 `true` 。
- `typical_p`: token 的典型概率。默认值为 `null` 。
- `watermark`: 用于生成的水印。默认值为 `false` 。

## 3. 用 javascript 和 python 进行流响应传输

使用 LLM 请求和生成文本可能是一个耗时且迭代的过程。改善用户体验的一个好方法是在生成 token 时将它们流式传输给用户。下面是两个使用 Python 和 JavaScript 流式传输 token 的示例。对于 Python，我们将使用 [Text Generation Inference 的客户端](https://github.com/huggingface/text-generation-inference/tree/main/clients/python)，对于 JavaScript，我们将使用 [HuggingFace.js 库](https://huggingface.co/docs/huggingface.js/main/en/index)。

### 使用 Python 流式传输请求

首先，你需要安装 `huggingface_hub` 库:

```python
pip install -U huggingface_hub
```

我们可以创建一个 `InferenceClient` ，提供我们的端点 URL 和凭据以及我们想要使用的超参数。

```python
from huggingface_hub import InferenceClient

# HF Inference Endpoints parameter
endpoint_url = "https://YOUR_ENDPOINT.endpoints.huggingface.cloud"
hf_token = "hf_YOUR_TOKEN"

# Streaming Client
client = InferenceClient(endpoint_url, token=hf_token)

# generation parameter
gen_kwargs = dict(
    max_new_tokens=512,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repetition_penalty=1.02,
    stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
)
# prompt
prompt = "What can you do in Nuremberg, Germany? Give me 3 Tips"

stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)

# yield each generated token
for r in stream:
    # skip special tokens
    if r.token.special:
        continue
    # stop if we encounter a stop sequence
    if r.token.text in gen_kwargs["stop_sequences"]:
        break
    # yield the generated token
    print(r.token.text, end = "")
    # yield r.token.text
```

将 `print` 命令替换为 `yield` 或你想要将 token 流式传输到的函数。

![Python Streaming](assets/155_inference_endpoints_llm/python-stream.gif Python Streaming)

### 使用 Javascript 流式传输请求

首先你需要安装 `@huggingface/inference` 库

```python
npm install @huggingface/inference
```

我们可以创建一个 `HfInferenceEndpoint` ，提供我们的端点 URL 和凭据以及我们想要使用的超参数。

```jsx
import { HfInferenceEndpoint } from '@huggingface/inference'

const hf = new HfInferenceEndpoint('https://YOUR_ENDPOINT.endpoints.huggingface.cloud', 'hf_YOUR_TOKEN')

//generation parameter
const gen_kwargs = {
  max_new_tokens: 512,
  top_k: 30,
  top_p: 0.9,
  temperature: 0.2,
  repetition_penalty: 1.02,
  stop_sequences: ['\nUser:', '<|endoftext|>', '</s>'],
}
// prompt
const prompt = 'What can you do in Nuremberg, Germany? Give me 3 Tips'

const stream = hf.textGenerationStream({ inputs: prompt, parameters: gen_kwargs })
for await (const r of stream) {
  // # skip special tokens
  if (r.token.special) {
    continue
  }
  // stop if we encounter a stop sequence
  if (gen_kwargs['stop_sequences'].includes(r.token.text)) {
    break
  }
  // yield the generated token
  process.stdout.write(r.token.text)
}
```

将 `process.stdout` 调用替换为 `yield` 或你想要将 token 流式传输到的函数。

![Javascript Streaming](https://huggingface.co/blog/assets/155_inference_endpoints_llm/js-stream.gif "Javascript Streaming")

## 结论

在这篇博客文章中，我们向你展示了如何使用 Hugging Face 推理端点部署开源 LLM，如何使用高级参数控制文本生成，以及如何将响应流式传输到 Python 或 JavaScript 客户端以提高用户体验。通过使用 Hugging Face 推理端点，你可以只需几次点击即可将模型部署为生产就绪的 API，通过自动缩减到零来降低成本，并在 SOC2 类型 2 认证的支持下将模型部署到安全的离线端点。

---

感谢你的阅读！如果你有任何问题，请随时在 [Twitter](https://twitter.com/_philschmid) 或 [LinkedIn](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/) 上联系我。