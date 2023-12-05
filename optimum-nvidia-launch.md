---
title: "Optimum-NVIDIA Unlocking blazingly fast LLM inference in just 1 line of code" 
authors:
- user: laikh-nvidia
  guest: true
- user: mfuntowicz
---


# Optimum-NVIDIA
## Unlock blazingly fast LLM inference in just 1 line of code

Large Language Models (LLMs) have revolutionized natural language processing and are increasingly deployed to solve complex problems at scale. 
Achieving optimal performance with these models is notoriously challenging due to their unique and intense computational demands. 
Optimized performance of LLMs is incredibly valuable for end users looking for a snappy and responsive experience as well as for scaled deployments where improved throughput translates to dollars saved.

That's where Optimum-NVIDIA  comes in. 
Optimum-NVIDIA dramatically accelerates LLM inference through an extremely simple API. 
By changing just a single line of code, you can unlock up to **$X$ faster** inference on the NVIDIA platform


### How to run
You can start running LLaMA with blazingly fast inference speeds in just 3 lines of code with a pipeline from Optimum-NVIDIA. 
If you already set up a pipeline from Hugging Face’s transformers library to run LLaMA, you just need to modify a single line of code to unlock peak performance!

```diff
- from transformers.pipelines import pipeline
+ from optimum.nvidia.pipelines import pipeline
```

```python
# everything else is the same as in transformers!
pipe = pipeline('text-generation', 'meta-llama/Llama-2-7b-chat-hf', use_fp8=True)
pipe("Describe a real-world application of AI in sustainable energy.")
```
The pipeline interface is great to get up and running quickly, but power users who want fine-grained control (e.g. setting sampling parameters) can use the Model API. 
You can also enable FP8 with a single flag, which allows you to run a bigger model on a single GPU, at faster speeds, and without sacrificing accuracy.


```diff
- from transformers import AutoTokenizer
+ from optimum.nvidia import AutoModelForCausalLM, QuantConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
  "meta-llama/Llama-2-13b-chat-hf",
+ use_fp8=True,  
)
```

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=quantization_config)

model_inputs = tokenizer(["How is autonomous vehicle technology transforming the future of transportation and urban planning?"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

The FP8 flag (shown above) uses a predefined calibration strategy by default, though you can provide your own calibration dataset and customized tokenization to tailor the quantization to your use case.
For more details, check out our [documentation](https://huggingface.co/docs/optimum/main/en/nvidia/index).


### Performance Evaluation

When evaluating the performance of an LLM, we consider 2 metrics: First Token Latency and Throughput. 
First Token Latency (also known as Time to First Token or prefill latency) measures how long you wait from the time you enter your prompt to begin receiving your output, so this metric can tell you how responsive the model will feel. 
Optimum-NVIDIA delivers up to XX faster First Token Latency compared to the framework:


Throughput, on the other hand, measures how fast the model can generate tokens, and is particularly relevant when you want to batch generations together. 
While there are a few ways to calculate throughput, we adopt a standard method to divide the end-to-end latency by the total sequence length, including both input and output tokens summed over all batches. 
Optimum-NVIDIA delivers up to XX better throughput compared to the framework:

Initial evaluations of the recently announced NVIDIA H200 Tensor Core GPU show up to 2x faster inference for LLaMA models compared to an H100.
As H200s become more readily available, we will share performance data for Optimum-NVIDIA on an H200.


### Next steps

Optimum-NVIDIA currently provides peak performance for the LLaMAForCausalLM architecture + task, so any [LLaMA-based model](https://huggingface.co/models?other=llama,llama2), 
including fine-tuned versions, should work with Optimum-NVIDIA out-of-the-box today. 


We are actively expanding support to include other text generation model architectures and other tasks like feature extraction to enable exciting applications like Retrieval Augmented Generation, all from within Hugging Face.
We continue to push the boundaries of performance and plan to incorporate cutting-edge optimization techniques like In-Flight Batching to improve throughput when streaming prompts and INT4 quantization to run even bigger models on a single GPU. 
