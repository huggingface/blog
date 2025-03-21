# 🚀 Intel Gaudi Meets Hugging Face: Supercharging Text Generation Inference

We’re thrilled to announce that **Intel Gaudi accelerators** are now integrated into Hugging Face’s [**Text Generation Inference (TGI)**](https://github.com/huggingface/text-generation-inference) project! This collaboration brings together the power of Intel’s high-performance AI hardware and Hugging Face’s state-of-the-art NLP software stack, enabling faster, more efficient, and scalable text generation for everyone.

Whether you’re building chatbots, generating creative content, or deploying large language models (LLMs) in production, this integration unlocks new possibilities for performance and cost-efficiency. Let’s dive into the details!

---

## 🤖 What is Text Generation Inference (TGI)?

Hugging Face’s **Text Generation Inference** is an open-source project designed to make deploying and serving large language models (LLMs) for text generation as seamless as possible. It powers popular models like GPT, T5, and BLOOM, providing features like:

- **High-performance inference**: Optimized for low latency and high throughput.
- **Scalability**: Built to handle large-scale deployments.
- **Ease of use**: Simple APIs and integrations for developers.

With TGI, you can deploy LLMs in production with confidence, knowing you’re leveraging the latest advancements in inference optimization.

---

## 🚀 Introducing Intel Gaudi Accelerators

Intel Gaudi accelerators are designed to deliver exceptional performance for AI workloads, particularly in training and inference for deep learning models. With features like high memory bandwidth, efficient tensor processing, and scalability across multiple devices, Gaudi accelerators are a perfect match for demanding NLP tasks like text generation.

By integrating Gaudi into TGI, we’re enabling users to:

- **Reduce inference costs**: Gaudi’s efficiency translates to lower operational expenses.
- **Scale seamlessly**: Handle larger models and higher request volumes with ease.
- **Achieve faster response times**: Optimized hardware for faster text generation.

---

## 🛠️ How It Works

The integration of Intel Gaudi into TGI leverages the **Habana SynapseAI SDK**, which provides optimized libraries and tools for running AI workloads on Gaudi hardware. Here’s how it works under the hood:

1. **Model Optimization**: TGI now supports Gaudi’s custom kernels and optimizations, ensuring that text generation models run efficiently on Gaudi accelerators.
2. **Seamless Deployment**: With just a few configuration changes, you can deploy your favorite Hugging Face models on Gaudi-powered infrastructure.
3. **Scalable Inference**: Gaudi’s architecture allows for multi-device setups, enabling you to scale inference horizontally as your needs grow.

---

## 🚀 Getting Started with TGI on Intel Gaudi

Ready to try it out? Here’s a quick guide to deploying a text generation model on Intel Gaudi using TGI:

### Step 1: Build tgi-gaudi image
Ensure you have access to a Gaudi accelerator and install the required dependencies:

```bash
# Build Text Generation Inference image with Gaudi support
git clone https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference/backends/gaudi
make image
```

### Step 2: Deploy Your Model
Use the TGI CLI to deploy a model on Gaudi:

```bash
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_HOME=<your huggingface home directory>
HF_TOKEN=<your huggingface token>
docker run -it  -p 8080:80 \
   --runtime=habana \
   -v $HF_HOME:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e PREFILL_BATCH_BUCKET_SIZE=16 \
   -e BATCH_BUCKET_SIZE=16 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=128 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   tgi-gaudi:latest --model-id $MODEL \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 65536 --max-batch-size 64 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 256
```

### Step 3: Generate Text
Send requests to your deployed model using the TGI API:

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
    -H 'Content-Type: application/json'
```

---

## 📊 Performance Benchmarks
### Step 1: Deploy meta-llama/Meta-Llama-3.1-8B-Instruct
According to the workload you are running, adjust the parameters such as max_batch_size, max_input_length, etc., and then deploy the model.
```bash
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
HF_HOME=<your huggingface home directory>
HF_TOKEN=<your huggingface token>

docker run -it  -p 8080:80 \
   --runtime=habana \
   -v $HF_HOME:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e PREFILL_BATCH_BUCKET_SIZE=16 \
   -e BATCH_BUCKET_SIZE=16 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=128 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   tgi-gaudi:latest --model-id $MODEL \
   --max-input-length 512 --max-total-tokens 1536 \
   --max-batch-prefill-tokens 131072 --max-batch-size 256 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Step2: Run the inference-benchmarker
```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=<your huggingface token>
RESULT=<directory of the result data>
docker run \
        --rm \
        -it \
        --net host \
        --cap-add=sys_nice \
        -v $RESULT:/opt/inference-benchmarker/results \
        -e "HF_TOKEN=$HF_TOKEN" \
        ghcr.io/huggingface/inference-benchmarker:latest \
        inference-benchmarker \
        --tokenizer-name "$MODEL" \
        --profile fixed-length \
        --url http://localhost:8080
```

| Benchmark          | QPS        | E2E Latency (avg) | TTFT (avg) | ITL (avg) | Throughput         | Error Rate | Successful Requests | Prompt tokens per req (avg) | Decoded tokens per req (avg) |
|--------------------|------------|-------------------|------------|-----------|--------------------|------------|---------------------|-----------------------------|------------------------------|
| warmup             | 0.13 req/s | 7.68 sec          | 171.90 ms  | 9.40 ms   | 104.11 tokens/sec  | 0.00%      | 3/3                 | 200.00                      | 800.00                       |
| throughput         | 4.30 req/s | 25.65 sec         | 656.54 ms  | 32.96 ms  | 3253.86 tokens/sec | 0.00%      | 518/518             | 200.00                      | 756.78                       |
| constant@0.52req/s | 0.49 req/s | 8.14 sec          | 175.20 ms  | 10.30 ms  | 379.90 tokens/sec  | 0.00%      | 57/57               | 200.00                      | 774.46                       |
| constant@1.03req/s | 0.97 req/s | 8.81 sec          | 175.69 ms  | 11.42 ms  | 730.78 tokens/sec  | 0.00%      | 114/114             | 200.00                      | 756.49                       |
| constant@1.55req/s | 1.42 req/s | 11.53 sec         | 179.17 ms  | 15.00 ms  | 1078.02 tokens/sec | 0.00%      | 168/168             | 200.00                      | 757.45                       |
| constant@2.06req/s | 1.86 req/s | 13.47 sec         | 179.41 ms  | 17.53 ms  | 1408.29 tokens/sec | 0.00%      | 219/219             | 200.00                      | 758.79                       |
| constant@2.58req/s | 2.11 req/s | 20.39 sec         | 183.98 ms  | 26.50 ms  | 1611.33 tokens/sec | 0.00%      | 252/252             | 200.00                      | 763.28                       |
| constant@3.10req/s | 2.24 req/s | 31.81 sec         | 191.35 ms  | 41.75 ms  | 1701.30 tokens/sec | 0.00%      | 265/265             | 200.00                      | 759.18                       |
| constant@3.61req/s | 2.36 req/s | 36.68 sec         | 285.70 ms  | 47.94 ms  | 1796.89 tokens/sec | 0.00%      | 283/283             | 200.00                      | 759.86                       |
| constant@4.13req/s | 2.67 req/s | 35.12 sec         | 306.93 ms  | 46.37 ms  | 2004.35 tokens/sec | 0.00%      | 311/311             | 200.00                      | 751.81                       |
| constant@4.64req/s | 2.79 req/s | 33.87 sec         | 315.08 ms  | 44.51 ms  | 2102.61 tokens/sec | 0.00%      | 332/332             | 200.00                      | 754.55                       |
| constant@5.16req/s | 3.01 req/s | 32.82 sec         | 317.72 ms  | 42.91 ms  | 2280.97 tokens/sec | 0.00%      | 355/355             | 200.00                      | 758.23                       |
---

## 🌟 What’s Next?

This integration is just the beginning of our collaboration with Intel. We’re excited to continue working together to bring even more optimizations and features to the Hugging Face ecosystem. Stay tuned for updates on:

- **Support for more models**: Expanding Gaudi compatibility to additional architectures.
- **Enhanced tooling**: Improved developer experience for deploying on Gaudi.
- **Community contributions**: Open-source contributions to make Gaudi accessible to everyone.

---

## 🎉 Join the Revolution

We can’t wait to see what you build with Hugging Face’s Text Generation Inference and Intel Gaudi accelerators. Whether you’re a researcher, developer, or enterprise, this integration opens up new possibilities for scaling and optimizing your text generation workflows.

Try it out today and let us know what you think! Share your feedback, benchmarks, and use cases with us on [GitHub](https://github.com/huggingface/text-generation-inference) or [Twitter](https://twitter.com/huggingface).

Happy text generating! 🚀