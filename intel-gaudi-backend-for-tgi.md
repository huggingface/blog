---
title: "🚀 Accelerating LLM Inference with TGI on Intel Gaudi"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: baptistecolle
---

# 🚀 Accelerating LLM Inference with TGI on Intel Gaudi

We're excited to announce the native integration of Intel Gaudi hardware support directly into Text Generation Inference (TGI), our production-ready serving solution for Large Language Models (LLMs). This integration brings the power of Intel's specialized AI accelerators to our high-performance inference stack, enabling more deployment options for the open-source AI community 🎉

## ✨ What's New? 

We've fully integrated Gaudi support into TGI's main codebase in PR [#3091](https://github.com/huggingface/text-generation-inference/pull/3091). Previously, we maintained a separate fork for Gaudi devices at [tgi-gaudi](https://github.com/huggingface/tgi-gaudi). This was cumbersome for users and prevented us from supporting the latest TGI features at launch. Now using the new [TGI multi-backend architecture](https://huggingface.co/blog/tgi-multi-backend), we support Gaudi directly on TGI – no more finicking on a custom repository 🙌

This integration supports Intel's full line of Gaudi hardware:
- Gaudi1 💻: Available on [AWS EC2 DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/)
- Gaudi2 💻💻: Available on [Intel Dev Cloud](https://ai.cloud.intel.com/)
- Gaudi3 💻💻💻: Available on [Intel Dev Cloud](https://ai.cloud.intel.com/) and [IBM Cloud](https://www.ibm.com/cloud)

## 🌟 Why This Matters 

The Gaudi backend for TGI provides several key benefits:
- Hardware Diversity 🔄: More options for deploying LLMs in production beyond traditional GPUs
- Cost Efficiency 💰: Gaudi hardware often provides compelling price-performance for specific workloads
- Production-Ready ⚙️: All the robustness of TGI (dynamic batching, streamed responses, etc.) now available on Gaudi
- Model Support 🤖: Run popular models like Llama 3.1, Mixtral, Mistral, and more on Gaudi hardware
- Advanced Features 🔥: Support for multi-card inference (sharding), vision-language models, and FP8 precision

## 🚦 Getting Started with TGI on Gaudi 

The easiest way to run TGI on Gaudi is to use our official Docker image. You need to run the image on a Gaudi hardware machine. Here's a basic example to get you started: 

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct 
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run 
hf_token=YOUR_HF_ACCESS_TOKEN

docker run --runtime=habana --cap-add=sys_nice --ipc=host \
 -p 8080:80 \
 -v $volume:/data \
 -e HF_TOKEN=$hf_token \
 -e HABANA_VISIBLE_DEVICES=all \
 ghcr.io/huggingface/text-generation-inference:3.2.1-gaudi \
 --model-id $model 
```

Once the server is running, you can send inference requests: 

```bash
curl 127.0.0.1:8080/generate
 -X POST
 -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32}}'
 -H 'Content-Type: application/json'
```

For comprehensive documentation on using TGI with Gaudi, including how-to guides and advanced configurations, refer to the new dedicated [Gaudi backend documentation](https://huggingface.co/docs/text-generation-inference/backends/gaudi).

## 🎉 Top features

We have optimized the following models for both single and multi-card configurations. This means these models run as fast as possible on Intel Gaudi. We've specifically optimized the modeling code to target Intel Gaudi hardware, ensuring we offer the best performance and fully utilize Gaudi's capabilities:

- Llama 3.1 (8B and 70B)
- Mistral (7B)
- Mixtral (8x7B)
- CodeLlama (13B)
- Falcon (180B)
- Qwen2 (72B) 
- Starcoder and Starcoder2 
- Gemma (7B) 
- Llava-v1.6-Mistral-7B 

Furthermore, we also support all models implemented in the [Transformers library](https://huggingface.co/docs/transformers/index), providing a [fallback mechanism](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) that ensures you can still run any model on Gaudi hardware even if it's not yet specifically optimized.

🏃‍♂️ We also offer many advanced features on Gaudi hardware, such as FP8 quantization thanks to [Intel Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html), enabling even greater performance optimizations.

## 💪 Getting Involved 

We invite the community to try out TGI on Gaudi hardware and provide feedback. The full documentation is available in the [TGI Gaudi backend documentation](https://huggingface.co/docs/text-generation-inference/backends/gaudi). 📚 If you're interested in contributing, check out our contribution guidelines or open an issue with your feedback on GitHub. 🤝 By bringing Intel Gaudi support directly into TGI, we're continuing our mission to provide flexible, efficient, and production-ready tools for deploying LLMs. We're excited to see what you'll build with this new capability! 🎉