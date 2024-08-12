---
title: Optimize and deploy 🤗 Transformer models with Optimum-Intel and OpenVINO Gen.AI
authors:
- user: AlexKoff88
  guest: true
- user: MrOpenVINO
  guest: true
- user: katuni4ka
  guest: true
- user: sandye51
  guest: true
- user: raymondlo84
  guest: true
- user: helenai
  guest: true
- user: sayakpaul
- user: echarlaix
---

# Optimize and deploy 🤗 Transformer models with Optimum-Intel and OpenVINO Gen.AI

When it comes to the Edge or Client deployment of the Transformers models Python is not always a suitable solution for this purpose. Many ISV's applications are written in C++ and require model inference API to be also in C++. Another aspect of such deployment is the application footprint which also should be minimized to simplify SW installation and update. [OpenVINO Toolkit](https://docs.openvino.ai/) initially emerged as a C++ AI inference solution that has bindings to popular programming languages such as Python or Java. It continues to be popular for Edge and Client deployment with the minimum dependencies on 3rd party SW libraries.

Recently, OpenVINO introduced [Generative AI (Gen.AI) API](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) designed to simplify integration of Generative AI model inference into C++ (or Python) application with the minimum external dependencies. LLM inference is the first feature in Gen.AI API that is currently available. OpenVINO Gen.AI SW package is supplied with [OpenVINO Tokenizers](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/ov-tokenizers.html) library required for text tokenization-detokenization. It also contains various [examples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples): from naive LLM decoder-based inference to speculative decoding.

In this blog post, we will outline the LLM deployment steps that include exporting model from Transformers library to OpenVINO Intermediate Representation (IR) using [Optimum-Intel](https://huggingface.co/docs/optimum/en/intel/index), model optimization with [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf), and deployment with new Gen.AI API. We will guide the user through all these steps and highlight changes of basic model KPIs, namely accuracy and performance.

## Pre-requisites

Python and C++ environments are required to run the examples below.

Python requirements:
- transformers==4.44
- openvino==24.3
- optimum-intel==1.20
- lm-eval==0.4.3

For Gen.AI C++ libraries installation follow the instruction [here](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-genai.html).


## Exporting model from 🤗 Transformers to OpenVINO

🤗 and Intel have a long story of collaboration and [Optimum-Intel](https://huggingface.co/docs/optimum/en/intel/index) project is a part of this story. It is designed to optimize Transformers models for inference on Intel HW. Optimum-Intel supports OpenVINO as an inference backend and its API has wrappers for various model architectures built on top of OpenVINO inference API. All of these wrappers start from `OV` prefix, for example, `OVModelForCausalLM`. Otherwise, it is similar to the API of 🤗 Transformers library.

To export 🤗 Transformers model to OpenVINO IR one case use two options: `.from_pretrained()` API method of the Optimum-Intel class or the command-line interface (CLI).
Further, we will use the recent Llama 3.1 8B decoder model as an example.
The export with the former option looks as follows:
```python
from optimum.intel import OVModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)
model.save_pretrained("./llam-3.1-8b-ov")
```

Alternatively, the same can be done with CLI as follows:
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B ./llam-3.1-8b-ov
```

The `./llam-3.1-8b-ov` folder will contain `.xml` and `bin` IR model files and required configuration files that come from the source model. If `openvino-tokenizers` library is installed the export process will also convert 🤗 tokenizer to the format of OpenVINO Tokinizers library and create corresponding configuration files in the same folder.

## Model Optimization

Model optimization is a mandatory step when it comes to LLM deployment. And weight-only quantization is a mainstream approach that significantly reduces latency and model footprint. Optimum-Intel provides weight-only quantization capabilities by means of NNCF which has a variety of optimization techniques designed specifically for LLMs: from data-free INT8 and INT4 weight quantization to data-aware methods such as AWQ, GPTQ, quantization scale estimation, mixed-precision quantization.
By default, weights of the models that are larger than one billion parameters are quantized to INT8 precision which is safe in terms of accuracy. It means that the export steps described above lead to the model with 8-bit weights. However, 4-bit integer weight-only quantization allows achieving a better accuracy-performance trade-off. 

For `meta-llama/Meta-Llama-3.1-8B` model we recommend stacking AWQ, quantization scale estimation along with mixed-precision INT4/INT8 quantization of weights using a calibration dataset that reflects a deployment use case. As in the case of export, there are two options on how to apply 4-bit weight-only quantization to LLM model:
- Specify `quantization_config` parameter in the `.from_pretrained()` method. In this case `OVWeightQuantizationConfig` object should be created and set to this parameter as follows:
```python
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
quantization_config = OVWeightQuantizationConfig(awq=True, scale_estimation=True, group_size=64, dataset="c4")
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, quantization_config=quantization_config)
model.save_pretrained("./llam-3.1-8b-ov")
```

- Use command-line options to enable 4-bit weight-only quantization:
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B --weight-format int4 --awq --scale-estimation --group-size 64 --dataset wikitext2 ./llam-3.1-8b-ov
```

>**Note**: The model optimization process can take time as it and applies several methods subsequently and uses model inference over the specified dataset.

Model optimization with API is more flexible as it allows using custom datasets that can be passed as an iterable object, for example, and instance of `Dataset` object of 🤗 library or just a list of strings.

Weight quantization usually introduces some degradation of the accuracy metric. To compare optimized and source models we report Word Perplexity metric measured on the [Wikitext](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) project which support both 🤗 Transformers and Optimum-Intel models out-of-the-box.

TODO compare metrics of the PyTorch, OV 8-bit and OV 4-bit models. 
| Model                        | PPL PyTorch FP32 | OpenVINO INT8 | OpenVINO INT4 |
| :--------------------------- | :--------------: | :-----------: | :-----------: |
| meta-llama/Meta-Llama-3.1-8B |   7.3366         | 7.3463        | 7.8288        | 

## Deploy model with OpenVINO Gen.AI C++ API

In this part, we will show how to run an optimized model with OpenVINO Gen.AI C++ API. The Gen.AI API is designed to be intuitive and provides a seamless migration from 🤗 Transformers API. The basic concept of this API is `ov::genai::LLMPipeline` class. Its instance can be created directly from the folder with the converted model. It will automatically load the main model, tokenizer, detokenizer, and the default generation configuration. The basic inference with `LLMPipeline` looks as follows:
```cpp
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
   std::string model_path = "./llam-3.1-8b-ov";
   ov::genai::LLMPipeline pipe(model_path, "CPU");
   std::cout << pipe.generate("What is LLM model?", ov::genai::max_new_tokens(256));
}
```

`LLMPipeline` also allows specifying custom generation options by means of `ov::genai::GenerationConfig`. It also supports streaming and chat scenarios as well as Beam Search. You can find more details in this [tutorial](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html).
