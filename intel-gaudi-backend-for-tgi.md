---
title: "🚀 Accelerating LLM Inference with TGI on Intel Gaudi"
thumbnail: /blog/assets/intel-gaudi-backend-for-tgi/tgi-gaudi-thumbnail.png
authors:
- user: baptistecolle
- user: regisss
- user: IlyasMoutawwakil
- user: echarlaix
- user: kding1
  guest: true
  org: Intel
---

# 🚀 Accelerating LLM Inference with TGI on Intel Gaudi

We're excited to announce the native integration of Intel Gaudi hardware support directly into Text Generation Inference (TGI), our production-ready serving solution for Large Language Models (LLMs). This integration brings the power of Intel's specialized AI accelerators to our high-performance inference stack, enabling more deployment options for the open-source AI community 🎉

## ✨ What's New? 

We've fully integrated Gaudi support into TGI's main codebase in PR [#3091](https://github.com/huggingface/text-generation-inference/pull/3091). Previously, we maintained a separate fork for Gaudi devices at [tgi-gaudi](https://github.com/huggingface/tgi-gaudi). This was cumbersome for users and prevented us from supporting the latest TGI features at launch. Now using the new [TGI multi-backend architecture](https://huggingface.co/blog/tgi-multi-backend), we support Gaudi directly on TGI – no more finicking on a custom repository 🙌

This integration supports Intel's full line of [Gaudi hardware](https://www.intel.com/content/www/us/en/developer/platform/gaudi/develop/overview.html):
- Gaudi1 💻: Available on [AWS EC2 DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/)
- Gaudi2 💻💻: Available on [Intel Tiber AI Cloud](https://ai.cloud.intel.com/) and [Denvr Dataworks](https://www.denvrdata.com/guadi2)
- Gaudi3 💻💻💻: Available on [Intel Tiber AI Cloud](https://ai.cloud.intel.com/), [IBM Cloud](https://www.ibm.com/cloud) and from OEM such as [Dell](https://www.dell.com/en-us/lp/intel-gaudi), [HP](https://www.hpe.com/us/en/compute/proliant-xd680.html) and [Supermicro](https://www.supermicro.com/en/accelerators/intel)

You can also find more information on Gaudi hardware on [Intel's Gaudi product page](https://www.intel.com/content/www/us/en/developer/platform/gaudi/develop/overview.html)

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
- Llama 3.3 (70B)
- Llama 3.2 Vision (11B)
- Mistral (7B)
- Mixtral (8x7B)
- CodeLlama (13B)
- Falcon (180B)
- Qwen2 (72B) 
- Starcoder and Starcoder2 
- Gemma (7B) 
- Llava-v1.6-Mistral-7B 
- Phi-2

🏃‍♂️ We also offer many advanced features on Gaudi hardware, such as FP8 quantization thanks to [Intel Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html), enabling even greater performance optimizations.

✨ Coming soon! We're excited to expand our model lineup with cutting-edge additions including DeepSeek-r1/v3, QWen-VL, and more powerful models to power your AI applications! 🚀

## 💪 Getting Involved 

We invite the community to try out TGI on Gaudi hardware and provide feedback. The full documentation is available in the [TGI Gaudi backend documentation](https://huggingface.co/docs/text-generation-inference/backends/gaudi). 📚 If you're interested in contributing, check out our contribution guidelines or open an issue with your feedback on GitHub. 🤝 By bringing Intel Gaudi support directly into TGI, we're continuing our mission to provide flexible, efficient, and production-ready tools for deploying LLMs. We're excited to see what you'll build with this new capability! 🎉
