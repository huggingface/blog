---
title: Optimize and deploy 🤗 Transformer models with Optimum-Intel and OpenVINO Gen.AI
thumbnail: /blog/assets/train_optimize_sd_intel/thumbnail.png
authors:
- user: AlexKoff88
  guest: true
- user: MrOpenVINO
  guest: true
- user: helenai
  guest: true
- user: sayakpaul
- user: echarlaix
---

# Optimize and deploy 🤗 Transformer models with Optimum-Intel and OpenVINO Gen.AI

When it comes to the Edge or Client deployment of the Transformers models Python is not always a suitable solution for this purpose. Many ISV's applications are written in C++ and require model inference API to be also in C++. Another aspect of such deployment is the application footprint that also should be minimized to simplify SW installation and update. [OpenVINO Toolkit](https://docs.openvino.ai/) initially emerged as a C++ AI inference solution that has bindings to the popular programming languages such as Python or Java. It continues being popular for Edge and Client deployment with the minimum dependencies on 3rd party SW libraries.

Recently, OpenVINO introduced [Generative AI (Gen.AI) API](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html) designed to simplify integration of Generative AI model inference into C++ (or Python) application with the minium external dependencies. LLM inference is the first feature in Gen.AI API that is currently available. OpenVINO Gen.AI SW package is supplied with [OpenVINO Tokenizers](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/ov-tokenizers.html) library required for text tokenization-detokenization. It also contains various [examples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples): from naive LLM decoder-based inference to speculative decoding.

In this blog post, we will outline the LLM deployment steps that include exporting model from Transformers library to OpenVINO Intermediate Representation (IR) using [Optimum-Intel](https://huggingface.co/docs/optimum/en/intel/index), model optimization with [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf), and deployment with new Gen.AI API. We will guide user through all these step and highlight changes of basic model KPIs, namely accuracy and peformance.

## Exporting model from 🤗 Transformers to OpenVINO

🤗 and Intel have a long story of collaboration and [Optimum-Intel](https://huggingface.co/docs/optimum/en/intel/index) project is a part of this story. It is designed to optimize Transformers models for inference on Intel HW. Optimum-Intel supports OpenVINO as an inference backend and its API has wrappers various model architectures built on top of OpenVINO inference API. All of these wrappers start from `OV` prefix, for example, `OVModelForCausalLM`. Otherwise, it is similar to the API of 🤗 Transformers library.

To export 🤗 Transformers model to OpenVINO IR one case use two options: `.from_pretrained()` API method of the Optimum-Intel class or the command-line interface (CLI).
Further, we will use recent Llama 3.1 8B decoder model as an example.
The export with former option looks like as follows:
```python
from optimum.intel import OVModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)
model.save_pretrained("./llam-3.1-8b-ov")
```

Alternativly, the same can be don with CLI as follows:
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3-8B ./llam-3.1-8b-ov
```

The `./llam-3.1-8b-ov` folder will contain `.xml` and `bin` IR model files and required configuration files that come from the source model. If `openvino-tokenizers` library is installed the export process will also convert 🤗 tokenizer to the formant of OpenVINO Tokinizers library and create corresponding configuration files in the same folder.

## Model Optimization

Model optimization is a mandatory step when it comes to LLM deployment. And weight-only quantization is a mainstream approach which allows reducing latency and model footprint significantly. Optimum-Intel provides weight-only quantization capabilities by means of NNCF which has a veriety of optimization techneques designed specifically for LLMs: from data-free INT8 and INT4 weight quantization to data-aware methods such as AWQ, GPTQ, quantization scale estimation, mixed-precision quantization.
By default, weights of the models that are larger than one billion parameters are quantized to INT8 precision which is safe in terms of accuracy. It means that the export steps described above lead to the model with 8-bit weights. However, 4-bit interger weight-only quantization allows achieving a better accuracy-perforance trade-off. 

For `meta-llama/Meta-Llama-3.1-8B` model we recommend stacking AWQ, quantization scale estimation along with mixed-precision INT4/INT8 quantization of weights using a calibration dataset that reflects a deployment use case. As in case of export, there are two options how to apply 4-bit weight-only quantizaiton to LLM model:
- Specify `quantization_config` parameter in the `.from_pretrained()` method. In this case `OVWeightQuantizationConfig` object should be created and set to this parameter as follows:
```
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
quantization_config = OVWeightQuantizationConfig(awq=True, scale_estimation=True, group_size=64, dataset="c4")
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, quantization_config=quantization_config)
model.save_pretrained("./llam-3.1-8b-ov")
```

- Use command-line options to enable 4-bit weight-only quantization:
```sh
optimum-cli export openvino -m meta-llama/Meta-Llama-3-8B --weight-format int4 --awq --scale-estimation --group-size 64 --dataset c4 ./llam-3.1-8b-ov
```

Model optimization with API is more flexible as it allows using custom datasets that cat be passed as an iterable object, for example, and instance of `Dataset` object of 🤗 library or just a list of strings.
