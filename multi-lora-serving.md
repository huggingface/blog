---
title: "TGI Multi-LoRA: Deploy Once, Serve 30 Models" 
thumbnail: /blog/assets/multi-lora-serving.png
authors:
- user: derek-thomas
- user: dmaniloff
- user: drbh
---

# TGI Multi-LoRA: Deploy Once, Serve 30 models

Are you tired of the complexity and expense of managing multiple AI models? **What if you could deploy once and serve 30 models?** In today's ML world, organizations looking to leverage the value of their data will likely end up in a _fine-tuned world_, building a multitude of models, each one highly specialized for a specific task. But how can you keep up with the hassle and cost of deploying a model for each use case? The answer is Multi-LoRA serving.


## Motivation

As an organization, building a multitude of models via fine-tuning makes sense for multiple reasons. 


- **Performance -** There is [compelling evidence](https://huggingface.co/papers/2405.09673) that smaller, specialized models outperform their larger, general-purpose counterparts on the tasks that they were trained on. Predibase [[5]](#5) showed that you can get better performance than GPT-4 using task-specific LoRAs with a base like [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main).


- **Adaptability -** Models like Mistral or Llama are extremely versatile. You can pick one of them as your base model and build many specialized models, even when the [downstream tasks are very different](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4). Also, note that you aren't locked in as you can easily swap that base and fine-tune it with your data on another base (more on this later).


- **Independence -** For each task that your organization cares about, different teams can work on different fine tunes, allowing for independence in data preparation,  configurations, evaluation criteria, and cadence of model updates.


- **Privacy -** Specialized models offer flexibility with training data segregation and access restrictions to different users based on data privacy requirements. Additionally, in cases where running models locally is important, a small model can be made highly capable for a specific task while keeping its size small enough to run on device. 

In summary, fine-tuning enables organizations to unlock the value of their data, and this advantage becomes especially significant, even game-changing, when organizations use highly specialized data that is uniquely theirs.

So, where is the catch? Deploying and serving Large Language Models (LLMs) is challenging in many ways. Cost and operational complexity are key considerations when deploying a single model, let alone _n_ models. This means that, for all its glory, fine-tuning complicates LLM deployment and serving even further.

That is why today we are super excited to introduce TGI's latest feature - **Multi-LoRA serving**.


## Background on LoRA

LoRA, which stands for [Low-Rank Adaptation](https://huggingface.co/papers/2106.09685), is a technique to fine-tune large pre-trained models efficiently. The core idea is to adapt large pre-trained models to specific tasks without needing to retrain the entire model, but only a small set of parameters called adapters. These adapters typically only add about 1% of storage and memory overhead compared to the size of the pre-trained LLM while maintaining the quality compared to fully fine-tuned models. 

The obvious benefit of LoRA is that it makes fine-tuning a lot cheaper by reducing memory needs. It also [reduces catastrophic forgetting](https://huggingface.co/papers/2405.09673) and works better with [small datasets](https://huggingface.co/blog/peft).

<video style="width: auto; height: auto;" controls autoplay muted loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/LoRA.webm">
  Your browser does not support the video tag.
</video>

|                            |
|----------------------------|
| *Figure 1: LoRA Explained* |

During training, LoRA freezes the original weights `W` and fine-tunes two small matrices, `A` and `B`, making fine-tuning much more efficient. With this in mind, we can see in _Figure 1_ how LoRA works during inference. We take the output from the pre-trained model `Wx`, and we add the Low Rank _adaptation_ term `BAx` [[6]](#6).


## Multi-LoRA Serving

Now that we understand the basic idea of model adaptation introduced by LoRA, we are ready to delve into multi-LoRA serving. The concept is simple: given one base pre-trained model and many different tasks for which you have fine-tuned specific LoRAs, multi-LoRA serving is a mechanism to dynamically pick the desired LoRA based on the incoming request.

<video style="width: auto; height: auto;" controls autoplay muted loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/MultiLoRA.webm">
  Your browser does not support the video tag.
</video>

|                                  |
|----------------------------------|
| *Figure 2: Multi-LoRA Explained* |

_Figure 2_ shows how this dynamic adaptation works. Each user request contains the input `x` along with the id for the corresponding LoRA for the request (we call this a heterogeneous batch of user requests). The task information is what allows TGI to pick the right LoRA adapter to use. 

Multi-LoRA serving enables you to deploy the base model just once. And since the LoRA adapters are small, you can load many adapters. Note the exact number will depend on your available GPU resources and what model you deploy. What you end up with is effectively equivalent to having multiple fine-tuned models in one single deployment.

LoRAs (the adapter weights) can vary based on rank and quantization, but they are generally quite tiny. Let's get a quick intuition of how small these adapters are: [predibase/magicoder](https://huggingface.co/predibase/magicoder/tree/main) is 13.6MB, which is less than 1/1000th the size of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main), which is 14.48GB. In relative terms, loading 30 adapters into RAM results in only a 3% increase in VRAM. Ultimately, this is not an issue for most deployments. Hence, we can have one deployment for many models.


# How to Use

## Gather LoRAs

First, you need to train your LoRA models and export the adapters. You can find a [guide here](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods) on fine-tuning LoRA adapters. Do note that when you push your fine-tuned model to the Hub, you only need to push the adapter, not the full merged model. When loading a LoRA adapter from the Hub, the base model is inferred from the adapter model card and loaded separately again. For deeper support, please check out our [Expert Support Program](https://huggingface.co/support). The real value will come when you create your own LoRAs for your specific use cases.


### Low Code Teams

For some organizations, it may be hard to train one LoRA for every use case, as they may lack the expertise or other resources. Even after you choose a base and prepare your data, you will need to keep up with the latest techniques, explore hyperparameters, find optimal hardware resources, write the code, and then evaluate. This can be quite a task, even for experienced teams. 

AutoTrain can lower this barrier to entry significantly. AutoTrain is a no-code solution that allows you to train machine learning models in just a few clicks. There are a number of ways to use AutoTrain. In addition to [locally/on-prem](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file#local-installation) we have:


| AutoTrain Environment                                                                                                          | Hardware Details             | Code Requirement | Notes                                     |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ---------------- | ----------------------------------------- |
| [Hugging Face Space](https://huggingface.co/login?next=%2Fspaces%2Fautotrain-projects%2Fautotrain-advanced%3Fduplicate%3Dtrue) | Variety of GPUs and hardware | No code          | Flexible and easy to share                |
| [DGX cloud](https://huggingface.co/blog/train-dgx-cloud)                                                                       | Up to 8xH100 GPUs            | No code          | Better for large models                   |
| [Google Colab](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain.ipynb)       | Access to a T4 GPU           | Low code         | Good for small loads and quantized models |


## Deploy
For our examples, we will use a couple of the excellent adapters featured in [LoRA Land from Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4):

- [predibase/customer_support](https://huggingface.co/predibase/customer_support) is trained on
  the [Gridspace-Stanford Harper Valley speech dataset](https://github.com/cricketclub/gridspace-stanford-harper-valley)
  which enhances its ability to understand and respond to customer service interactions accurately. This improves the
  model's performance in tasks such as speech recognition, emotion detection, and dialogue management, leading to more
  efficient and empathetic customer support.

- [predibase/magicoder](https://huggingface.co/predibase/magicoder) is trained
  on [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) which is
  a code instruction dataset that is synthetically generated.

### TGI

There is already a lot of good information on [how to deploy TGI](https://github.com/huggingface/text-generation-inference). Deploy like you normally would, but ensure that you:

1. Use a TGI version newer or equal to `v2.1.1`
2. Deploy your base: `mistralai/Mistral-7B-v0.1`
3. Add the `LORA_ADAPTERS` env var during deployment
    * Example: `LORA_ADAPTERS=predibase/customer_support,predibase/magicoder` 

```bash
model=mistralai/Mistral-7B-v0.1
# share a volume with the Docker container to avoid downloading weights every run
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.1.1 \
    --model-id $model \
    --lora-adapters=predibase/customer_support,predibase/magicoder
```


### Inference Endpoints GUI

[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/en/index) allows you to have access to deploy any Hugging Face model on many [GPUs and alternative Hardware types](https://huggingface.co/docs/inference-endpoints/en/pricing#gpu-instances) across AWS, GCP, and Azure all in a few clicks! In the GUI, it's easy to deploy. Under the hood, we use TGI by default for text generation (though you have the [option](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container) to use any image you choose).

To use Multi-LoRA serving on Inference Endpoints, you just need to go to your [dashboard](https://ui.endpoints.huggingface.co/), then:

1. Choose your base model: `mistralai/Mistral-7B-v0.1`
2. Choose your `Cloud` | `Region` | `HW`
    * Ill use `AWS` | `us-east-1` | `Nvidia L4`
3. Select Advanced Configuration
    * You should see `text generation` already selected
    * You can configure based on your needs
4. Add `LORA_ADAPTERS=predibase/customer_support,predibase/magicoder` in Environment Variables
5. Finally `Create Endpoint`!

Note that this is the minimum, but you should configure the other settings as you desire.


| ![multi-lora-inference-endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-inference-endpoints.png) |
|-------------------------------------------------|
| *Figure 3: Multi-LoRA Inference Endpoints* |


| ![multi-lora-inference-endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-inference-endpoints.png) |
|-------------------------------------------------|
| *Figure 4: Multi-LoRA Inference Endpoints 2* |

### Inference Endpoints Code

Maybe some of you are [musophobic](https://en.wikipedia.org/wiki/Fear_of_mice_and_rats) and don't want to use your mouse, we don’t judge. It’s easy enough to automate this in code and only use your keyboard. 

```python
from huggingface_hub import create_inference_endpoint

# Custom Docker image details
custom_image = {
    "health_route": "/health",
    "url": "ghcr.io/huggingface/text-generation-inference:2.1.1",  # This is the min version
    "env": {
        "LORA_ADAPTERS": "predibase/customer_support,predibase/magicoder",  # Add adapters here
        "MAX_BATCH_PREFILL_TOKENS": "2048",  # Set according to your needs
        "MAX_INPUT_LENGTH": "1024", # Set according to your needs
        "MAX_TOTAL_TOKENS": "1512", # Set according to your needs
        "MODEL_ID": "/repository"
    }
}

# Creating the inference endpoint
endpoint = create_inference_endpoint(
    name="mistral-7b-multi-lora",
    repository="mistralai/Mistral-7B-v0.1",
    framework="pytorch",
    accelerator="gpu",
    instance_size="x1",
    instance_type="nvidia-l4",
    region="us-east-1",
    vendor="aws",
    min_replica=1,
    max_replica=1,
    task="text-generation",
    custom_image=custom_image,
)
endpoint.wait()

print("Your model is ready to use!")
```

It took ~3m40s for this configuration to deploy. Note for more models it will take longer. Do make a [github issue](https://github.com/huggingface/text-generation-inference/issues) if you are facing issues with load time!

## Consume

When you consume your endpoint, you will need to specify your `adapter_id`. Here is a cURL example:

```bash
curl 127.0.0.1:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "Hello who are you?",
  "parameters": {
    "max_new_tokens": 40,
    "adapter_id": "predibase/customer_support"
  }
}'
```

Alternatively, here is an example using [InferenceClient](https://huggingface.co/docs/huggingface_hub/guides/inference) from the wonderful [Hugging Face Hub Python library](https://huggingface.co/docs/huggingface_hub/index). Do make sure you are using `huggingface-hub>=0.24.0` and that you are [logged in](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) if necessary.

```python
from huggingface_hub import InferenceClient

tgi_deployment = "127.0.0.1:3000"
client = InferenceClient(tgi_deployment)
response = client.text_generation(
    prompt="Hello who are you?",
    max_new_tokens=40,
    adapter_id='predibase/customer_support',
)
```

## Practical Considerations


### Cost

We are not the first to climb this summit, as discussed [below](#Acknowledgements). The team behind LoRAX, Predibase, has an excellent [write up](https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in). Do check it out, as this section is based on their work. 

| ![multi-lora-cost](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-cost.png) |
|-------------------------------------------------|
| *Figure 5: Multi-LoRA Cost* For TGI, I deployed [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as a base on nvidia-l4, which has a [cost](https://huggingface.co/docs/inference-endpoints/en/pricing#gpu-instances) of $0.8/hr on [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/en/index). I was able to get 75 requests/s with an average of 450 input tokens and 234 output tokens and adjusted accordingly for GPT3.5 Turbo.|



One of the big benefits of Multi-LoRA serving is that **you don’t need to have multiple deployments for multiple models**, and ultimately this is much much cheaper. This should match your intuition as multiple models will need all the weights and not just the small adapter layer. As you can see in _Figure 5_, even when we add many more models with TGI Multi-LoRA the cost is the same per token. The cost for TGI dedicated scales as you need a new deployment for each fine-tuned model.


## Usage Patterns

| ![multi-lora-serving-pattern](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-lora-serving/multi-lora-serving-pattern.png) |
|-------------------------------------------------|
| *Figure 6: Multi-LoRA Serving Pattern* |

One real-world challenge when you deploy multiple models is that you will have a strong variance in your usage patterns. Some models might have low usage; some might be bursty, and some might be high frequency. This makes it really hard to scale, especially when each model is independent. There are a lot of “rounding” errors when you have to add another GPU, and that adds up fast. In an ideal world, you would maximize your GPU utilization per GPU and not use any extra. You need to make sure you have access to enough GPUs, knowing some will be idle, which can be quite tedious. 

When we consolidate with Multi-LoRA, we get much more stable usage. We can see the results of this in _Figure 6_ where the Multi-Lora Serving pattern is quite stable even though it consists of more volatile patterns. By consolidating the models, you allow much smoother usage and more manageable scaling. Do note that these are just illustrative patterns, but think through your own patterns and how Multi-LoRA can help. Scale 1 model, not 30!


## Changing the base model

What happens in the real world with AI moving at breakneck speeds? What if you want to choose a different/newer model as your base? While our examples use [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as a base model, there are other bases like Mistral's [v0.3](https://ubiops.com/function-calling-deploy-the-mistral-7b-v03/) which supports [function calling](https://ubiops.com/function-calling-deploy-the-mistral-7b-v03/), and altogether different model families like Llama 3. In general, we expect new base models that are more efficient and more performant to come out all the time.

But worry not! It is easy enough to re-train the LoRAs if you have a _compelling reason_ to update your base model. Training is relatively cheap; in fact [Predibase found](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4) it costs only ~$8.00 to train each one. The amount of code changes is minimal with modern frameworks and common engineering practices:

* Keep the notebook/code used to train your model
* Version control your datasets
* Keep track of the configuration used
* Update with the new model/settings


## Conclusion

Multi-LoRA serving represents a transformative approach in the deployment of AI models, providing a solution to the cost and complexity barriers associated with managing multiple specialized models. By leveraging a single base model and dynamically applying fine-tuned adapters, organizations can significantly reduce operational overhead while maintaining or even enhancing performance across diverse tasks. **AI Directors we ask you to be bold, choose a base model and embrace the Multi-LoRA paradigm,** the simplicity and cost savings will pay off in dividends. Let Multi-LoRA be the cornerstone of your AI strategy, ensuring your organization stays ahead in the rapidly evolving landscape of technology.


## Acknowledgements

Implementing Multi-LoRA serving can be really tricky, but due to awesome work by [punica-ai](https://github.com/punica-ai/punica) and the [lorax](https://github.com/predibase/lorax) team, optimized kernels and frameworks have been developed to make this process more efficient. TGI leverages these optimizations in order to provide fast and efficient inference with multiple LoRA models.

Special thanks to the Punica, LoRAX, and S-LoRA teams for their excellent and open work in multi-LoRA serving. 

## References

* <a id="1">[1]</a> : Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham, [LoRA Learns Less and Forgets Less](https://huggingface.co/papers/2405.09673), 2024
* <a id="2">[2]</a>  : Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685), 2021
* <a id="3">[3]</a>  : Sourab Mangrulkar, Sayak Paul, [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft), 2023
* <a id="4">[4]</a>  : Travis Addair, Geoffrey Angus, Magdy Saleh, Wael Abid, [LoRAX: The Open Source Framework for Serving 100s of Fine-Tuned LLMs in Production](https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in), 2023
* <a id="5">[5]</a>  : Timothy Wang, Justin Zhao, Will Van Eaton, [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4), 2024
* <a id="6">[6]</a>  : Punica: Serving multiple LoRA finetuned LLM as one: [https://github.com/punica-ai/punica](https://github.com/punica-ai/punica)
