---
title: "Accelerating over 110,000 Hugging Face models with ONNX Runtime"
thumbnail: /blog/assets/163_getting_most_out_of_llms/optimize_llm.png
authors:
- user: Sophie Schoenmeyer
  guest: true
- user: Morgan Funtowicz
---

# Accelerating over 110,000 Hugging Face models with ONNX Runtime

There are currently over 320,000 models on Hugging Face (HF), and this number continues to grow every day.
Only about 6,000 of these models have an indication of ONNX support in the HF Model Hub, but over 130,000 support the ONNX format.

ONNX models can be accelerated with ONNX Runtime (ORT), which works cross-platform and provides coverage for many cloud models and large language models.
Updating the HF Model Hub with more accurate information about ONNX coverage will ensure that users can leverage all the benefits of ORT when deploying HF models.

This blog post will provide an overview of HF model architectures with ORT support,
discuss ORT coverage for cloud models and large language models,
and provide next steps for increasing the number of ONNX models listed in the HF Model Hub.


## HF ORT Support Overview
Hugging Face provides a list of supported model architectures in its [Transformers documentation](https://huggingface.co/docs/transformers/index).

Model architectures are groups of models with similar operators, meaning that if one model within a model architecture is supported by ONNX,
the other models in the architecture are supported by ONNX as well (with rare exceptions).

Models in the HF Model Hub can be filtered by model architecture using search queries (e.g., the number of models from the BERT model architecture
can be found using https://huggingface.co/models?other=bert).

ORT supports model architectures where: (1) one or more models in the model architecture have ONNX listed as a library in the HF [Model Hub](https://huggingface.co/models?library=onnx&sort=trending),
(2) the model architecture is supported by the Optimum API (more information [here](https://huggingface.co/docs/optimum/exporters/onnx/overview)),
or (3) the model architecture is supported by Transformers.js (more information [here](https://huggingface.co/docs/transformers.js/index)).

ORT can greatly improve performance for some of the most popular models in the HF Model Hub.

Using ORT instead of PyTorch can improve average latency per inference,
a measure of how quickly data is received, with an up to 50.10% gain over PyTorch for the whisper-large model and an up to
74.30% gain over PyTorch for the whisper-tiny model


These benchmark results were run with FP32 on an A100 40GB device. For CPU benchmarks, an AMD EPYC 7V12 64-core processor was used.
The top 30 HF model architectures are all supported by ORT, and over 90 HF model architectures in total boast ORT support.
Any gaps in ORT coverage generally represent less popular model architectures.

The following table includes a list of the top 11 model architectures, all of which are convertible to ONNX using the Hugging Face Optimum API,
along with the corresponding number of models uploaded to HF (as of the date this post was published).
These numbers will continue to grow over time, as will the list of supported model architectures.


## Large Language Models (LLMs)
ONNX Runtime also supports many increasingly popular large language model (LLM) architectures, including most of those available in the HF Model Hub.
These model architectures include the following, all of which are convertible to ONNX using the Hugging Face Optimum API:

For more detailed tracking and evaluation of recently released LLMs from the community, see HF’s [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).


## Azure ML Cloud Models
Models accelerated by ONNX Runtime can be easily deployed to the cloud through Azure ML, which improves time to value, streamlines MLOps, and provides built-in security.
Azure ML (AML) also publishes a curated model list that is updated regularly and includes some of the most popular models at the moment.
Of the models on this list that are available in the HF Model Hub, over 84% have HF Optimum ONNX support. 
Six of the remaining models are of the recently released llama2 model architecture, for which support is still in the works but will be available very soon.


## Next Steps
The top priority moving forward is to add as many ONNX models as possible to the HF Model Hub so these models are easily accessible to the community.
We are currently in the process of identifying a scalable way to run the Optimum API and working with the HF team directly to increase the number of models indicated to have ONNX support in the HF Model Hub.

We also encourage members of the community to add their own ONNX models to HF, as over 100,000 models in the HF Model Hub have ONNX support that is not indicated.

The simplest way for a user to export PyTorch models hosted on the HF Model Hub to ONNX is the ONNX Export Space. 
Instructions for this tool can be found [here](https://huggingface.co/spaces/onnx/export). 
Additional information and instructions for other ONNX export options can be found [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model)