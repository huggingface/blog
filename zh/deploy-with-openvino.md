---
title: Optimize and deploy with Optimum-Intel and OpenVINO GenAI
authors:
- user: AlexKoff88
  guest: true
  org: Intel
- user: MrOpenVINO
  guest: true
  org: Intel
- user: katuni4ka
  guest: true
  org: Intel
- user: sandye51
  guest: true
  org: Intel
- user: raymondlo84
  guest: true
  org: Intel
- user: helenai
  guest: true
  org: Intel
- user: echarlaix
translators:
- user: Zipxuan
---

# 使用Optimum-Intel和OpenVINO GenAI优化和部署模型

在端侧部署Transformer模型需要仔细考虑性能和兼容性。Python虽然功能强大，但对于部署来说有时并不算理想，特别是在由C++主导的环境中。这篇博客将指导您如何使用Optimum-Intel和OpenVINO™ GenAI来优化和部署Hugging Face Transformers模型，确保在最小依赖性的情况下进行高效的AI推理。

## 目录
- [为什么使用OpenVINO来进行端侧部署](#为什么使用OpenVINO来进行端侧部署)
- [第一步：创建环境](#第一步创建环境)
- [第二步：将模型导出为OpenVINO IR](#第二步将模型导出为openvino-ir)
- [第三步：模型优化](#第三步模型优化)
- [第四步：使用OpenVINO GenAI API进行部署](#第四步使用openvino-genai-api进行部署)
- [结论](#结论)

## 为什么使用OpenVINO来进行端侧部署
OpenVINO™ 最初是作为 C++ AI 推理解决方案开发的，使其非常适合在端侧设备部署中，其中最小化依赖性至关重要。随着引入 GenAI API，将大型语言模型（LLMs）集成到 C++ 或 Python 应用程序中变得更加简单，其特性旨在简化部署并提升性能。

## 第一步：创建环境

### 预先准备

开始之前，请确保您的环境已正确配置了Python和C++。安装必要的Python包：
```sh
pip install --upgrade --upgrade-strategy eager optimum[openvino]
```

以下是本文中使用的具体包：
```
transformers==4.44
openvino==24.3
openvino-tokenizers==24.3
optimum-intel==1.20
lm-eval==0.4.3
```

有关GenAI C++库的安装，请按照[此处](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html)的说明进行操作。

## 第二步：将模型导出为OpenVINO IR

Hugging Face 和 Intel 的合作促成了 [Optimum-Intel](https://huggingface.co/docs/optimum/en/intel/index) 项目。该项目旨在优化 Transformers 模型在 Intel 硬件上的推理性能。Optimum-Intel 支持 OpenVINO 作为推理后端，其 API 为各种基于 OpenVINO 推理 API 构建的模型架构提供了封装。这些封装都以 `OV` 前缀开头，例如 `OVModelForCausalLM`。除此之外，它与 🤗 Transformers 库的 API 类似。

要将 Transformers 模型导出为 OpenVINO 中间表示（IR），可以使用两种方法：可以使用 Python 的 `.from_pretrained()` 方法或 Optimum 命令行界面（CLI）。以下是使用这两种方法的示例：
### 使用 Python API
```python
from optimum.intel import OVModelForCausalLM

model_id = "meta-llama/Meta-Llama-3.1-8B"
model = OVModelForCausalLM.from_pretrained(model_id, export=True)
model.save_pretrained("./llama-3.1-8b-ov")
```

### 使用命令行 (CLI)
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B ./llama-3.1-8b-ov
```

./llama-3.1-8b-ov 文件夹将包含 .xml 和 bin IR 模型文件以及来自源模型的所需配置文件。🤗 tokenizer 也将转换为 openvino-tokenizers 库的格式，并在同一文件夹中创建相应的配置文件。

## 第三步：模型优化

在资源受限的端侧设备上运行大型语言模型（LLMs）时，模型优化是一个极为重要的步骤。仅量化权重是一种主流方法，可以显著降低延迟和模型占用空间。Optimum-Intel 通过神经网络压缩框架（NNCF）提供了仅量化权重（weight-only quantization）的功能，该框架具有多种专为LLMs设计的优化技术：从无数据（data-free）的 INT8 和 INT4 权重量化到数据感知方法，如 [AWQ](https://huggingface.co/docs/transformers/main/en/quantization/awq)、[GPTQ](https://huggingface.co/docs/transformers/main/en/quantization/gptq)、量化scale估计、混合精度量化等。默认情况下，超过十亿参数的模型的权重会被量化为 INT8 精度，这在准确性方面是安全的。这意味着上述导出步骤会生成具有8位权重的模型。然而，4位整数的仅量化权重允许实现更好的准确性和性能的权衡。

对于 `meta-llama/Meta-Llama-3.1-8B` 模型，我们建议结合 AWQ、量化scale估计以及使用反映部署用例的校准数据集进行混合精度 INT4/INT8 权重的量化。与导出情况类似，在将4比特仅量化权重应用于LLM模型时有两种选项：

### 使用Python API
- 在 `.from_pretrained()` 方法中指定 `quantization_config` 参数。在这种情况下，应创建 `OVWeightQuantizationConfig` 对象，并将其设置为该参数，如下所示：
```python
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
quantization_config = OVWeightQuantizationConfig(bits=4, awq=True, scale_estimation=True, group_size=64, dataset="c4")
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, quantization_config=quantization_config)
model.save_pretrained("./llama-3.1-8b-ov")
```

### 使用命令行 (CLI)
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B --weight-format int4 --awq --scale-estimation --group-size 64 --dataset wikitext2 ./llama-3.1-8b-ov
```


## 第四步：使用OpenVINO GenAI API进行部署
在转换和优化之后，使用OpenVINO GenAI部署模型非常简单。OpenVINO GenAI中的LLMPipeline类提供了Python和C++ API，支持各种文本生成方法，并具有最小的依赖关系。


### Python API的例子
```python
import argparse
import openvino_genai

device = "CPU"  # GPU can be used as well
pipe = openvino_genai.LLMPipeline(args.model_dir, device)
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
print(pipe.generate(args.prompt, config))
```

为了运行这个示例，您需要在Python环境中安装最小的依赖项，因为OpenVINO GenAI旨在提供轻量级部署。您可以将OpenVINO GenAI包安装到相同的Python环境中，或者创建一个单独的环境来比较应用程序的占用空间：
```sh
pip install openvino-genai==24.3
```

### C++ API的例子

让我们看看如何使用OpenVINO GenAI C++ API运行相同的流程。GenAI API的设计非常直观，并提供了与 🤗 Transformers API 无缝迁移的功能。

> **注意**：在下面的示例中，您可以为 "device" 变量指定环境中的任何其他可用设备。例如，如果您正在使用带有集成显卡的Intel CPU，则尝试使用 "GPU" 是一个不错的选择。要检查可用设备，您可以使用 ov::Core::get_available_devices 方法（参考 [query-device-properties](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html)）。

```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
   std::string model_path = "./llama-3.1-8b-ov";
   std::string device = "CPU"  // GPU can be used as well
   ov::genai::LLMPipeline pipe(model_path, device);
   std::cout << pipe.generate("What is LLM model?", ov::genai::max_new_tokens(256));
}
```

### 自定义生成配置
`LLMPipeline` 还允许通过 `ov::genai::GenerationConfig` 来指定自定义生成选项：
```cpp
ov::genai::GenerationConfig config;
config.max_new_tokens = 256;
std::string result = pipe.generate(prompt, config);
```

使用LLMPipeline，用户不仅可以轻松利用各种解码算法，如 Beam Search，还可以像下面的示例中那样构建具有 Streamer 的交互式聊天场景。此外，用户可以利用LLMPipeline的增强内部优化，例如利用先前聊天历史的KV缓存减少提示处理时间，使用 chat 方法：start_chat() 和 finish_chat()（参考 [using-genai-in-chat-scenario](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html#using-genai-in-chat-scenario)）。


```cpp
ov::genai::GenerationConfig config;
config.max_new_tokens = 100;
config.do_sample = true;
config.top_p = 0.9;
config.top_k = 30;

auto streamer = [](std::string subword) {
    std::cout << subword << std::flush;
    return false;
};

// Since the streamer is set, the results will
// be printed each time a new token is generated.
pipe.generate(prompt, config, streamer);
```

最后你可以看到如何在聊天场景下使用LLMPipeline：
```cpp
pipe.start_chat()
for (size_t i = 0; i < questions.size(); i++) {
   std::cout << "question:\n";
   std::getline(std::cin, prompt);

   std::cout << pipe.generate(prompt) << std::endl;
}
pipe.finish_chat();
```

## 结论
Optimum-Intel和OpenVINO™ GenAI的结合为在端侧部署Hugging Face模型提供了强大而灵活的解决方案。通过遵循这些步骤，您可以在Python可能不是理想选择的环境中实现优化的高性能AI推理，以确保您的应用在Intel硬件上平稳运行。

## 其他资源
1. 您可以在这个 [教程](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) 中找到更多详细信息。
2. 要构建上述的C++示例，请参考这个 [文档](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2024/3/src/docs/BUILD.md)。
3. [OpenVINO文档](docs.openvino.ai)
4. [Jupyter笔记本](https://docs.openvino.ai/2024/learn-openvino/interactive-tutorials-python.html)
5. [Optimum文档](https://huggingface.co/docs/optimum/main/en/intel/index)

![OpenVINO GenAI C++聊天演示](https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_genai_workflow/demo.gif)
