---
title: "Fine-tune Any LLM from the Hugging Face Hub with Together AI"
thumbnail: /blog/assets/197_together-ft/thumbnail.png
authors:
- user: artek0chumak
  guest: true
  org: togethercomputer
- user: timofeev1995
  guest: true
  org: togethercomputer
- user: zainh
  guest: true
  org: togethercomputer
- user: mryab
  guest: true
  org: togethercomputer
---

# Fine-tune Any LLM from the Hugging Face Hub with Together AI

The pace of AI development today is breathtaking. Every single day, hundreds of new models appear on the [Hugging Face Hub](https://huggingface.co/models), some are specialized variants of popular base models like Llama or Qwen, others feature novel architectures or have been trained from scratch for specific domains. Whether it's a medical AI trained on clinical data, a coding assistant optimized for a particular programming language, or a multilingual model fine-tuned for specific cultural contexts, the Hugging Face Hub has become the beating heart of open-source AI innovation.

But here's the challenge: finding an amazing model is just the beginning. What happens when you discover a model that's 90% perfect for your use case, but you need that extra 10% of customization? Traditional fine-tuning infrastructure is complex, expensive, and often requires significant DevOps expertise to set up and maintain.

This is exactly the gap that Together AI and Hugging Face are bridging today. We're announcing a powerful new capability that makes the entire Hugging Face Hub available for fine-tuning using Together AI's infrastructure. Now, any compatible LLM on the Hub, whether it's from Meta or an individual contributor, can be fine-tuned with the same ease and reliability you expect from Together's platform.🚀

## **Getting Started in 5 Minutes**

Here's all it takes to start fine-tuning a HF model on the Together AI platform:

```python
#pip install together

from together import Together

client = Together(api_key="your-api-key")

file_upload = client.files.upload("sft_examples.jsonl", check=True)

# Fine-tune any compatible HF model
job = client.fine_tuning.create(
    model="togethercomputer/llama-2-7b-chat",  # Base model for configuration
    from_hf_model="HuggingFaceTB/SmolLM2-1.7B-Instruct",  # Your chosen HF model
    training_file=file_upload.id,
    n_epochs=3,
    learning_rate=1e-5,
    hf_api_token="hf_***",  # for private repos
    hf_output_repo_name="my-username-org/SmolLM2-1.7B-FT"  # to upload model back to hub
)

print(f"Training job started: {job.id}")
```

That's it! Your model will be trained on Together's infrastructure and can be deployed for inference, downloaded or even uploaded back to the Hub! For private repositories, simply add your HF token with `hf_api_token="hf_xxxxxxxxxxxx"`.

## How It Works:

As seen in the example above, when you fine-tune a Hugging Face model on Together AI, you actually specify two models:

- **Base Model** (`model` parameter): A model from Together's [official catalog](https://docs.together.ai/docs/fine-tuning-models) that provides the infrastructure configuration, training optimizations, and inference setup
- **Custom Model** (`from_hf_model`parameter): Your actual Hugging Face model that gets fine-tuned

Think of the base model as a "training template." It tells our system how to optimally allocate GPU resources, configure memory usage, set up the training pipeline, and prepare the model for inference. Your custom model should have a similar architecture, approximate size, and sequence length to the base model for optimal results.

As seen in the example above, if you want to fine-tune `HuggingFaceTB/SmolLM2-1.7B-Instruct` (which uses Llama architecture), you'd use `togethercomputer/llama-2-7b-chat` as your base model template, because they share the same underlying architecture.

The integration works bidirectionally. Together AI can pull any compatible public model from the Hugging Face Hub for training, and with the proper API tokens, it can access private repositories as well. After training, your fine-tuned model can be automatically pushed back onto the Hub if you've specified `hf_output_repo_name`, making it available for sharing with your team or the broader community.

In general, all `CausalLM` models under 100B params will work. For a comprehensive walkthrough on how to choose base and custom models and much more read out full [guide](https://docs.together.ai/docs/fine-tuning-byom)!

## What This Means for Developers?

This integration solves a real problem many of us have faced: finding a great model on Hugging Face but not having the infrastructure to actually fine-tune it for your specific needs. Now you can go from discovering a promising model to having a customized version running in production with just a few API calls.

The big win here is removing friction. Instead of spending days setting up training infrastructure or being limited to whatever models are officially supported by various platforms, you can now experiment with any compatible model from the Hub. Found a specialized coding model that's close to what you need? Train it on your data!📈

For teams, this means faster iteration cycles. You can test multiple model approaches quickly, build on community innovations, and even use your own fine-tuned models as starting points for further customization. 

## **How Teams Are Using This Feature?**

Beta users and early adopters of this feature are already seeing results across diverse use cases.

[**Slingshot AI**](https://www.together.ai/customers/slingshot-ai) has integrated this capability directly into their model development pipeline. Rather than being limited to Together's model catalog, they can now run parts of the training pipeline on their own infrastructure, upload those models to the Hub, and then perform continued fine-tuning of those models using the Together AI fine-tuning platform. This has dramatically accelerated their development cycles and allowed them to experiment with cutting-edge architectures as soon as they're published on the Hub.

[**Parsed**](https://www.together.ai/blog/fine-tune-small-open-source-llms-outperform-closed-models) has demonstrated the power of this approach in their work showing how small, well-tuned open-source models can outperform much larger closed models. By fine-tuning models on carefully curated datasets, they've achieved superior performance while maintaining cost efficiency and full control over their models.

**Common usage we're seeing from other customers include:**

- *Domain Adaptation*: Taking general-purpose models and specializing them for industries like healthcare, finance, or legal work. Teams discover models that already have some domain knowledge and use Together's infrastructure to adapt them to their specific data and requirements.
- *Iterative Model Improvement*: Starting with a community model, fine-tuning it, then using that result as the starting point for further refinement. This creates a compound improvement effect that would be difficult to achieve starting from scratch.
- *Community Model Specialization*: Leveraging models that have already been optimized for specific tasks (like coding, reasoning, or multilingual capabilities) and further customizing them for proprietary use cases.
- *Architecture Exploration*: Quickly testing newer architectures and model variants as they're released, without waiting for them to be added to official platforms.

The most significant advantage teams report is *speed to value*. Instead of spending weeks setting up training infrastructure or months training models from scratch, they can identify promising starting points from the community and have specialized models running in production within days.

Cost efficiency is another major benefit. By starting with models that already have relevant capabilities, teams need fewer training epochs and can use smaller datasets to achieve their target performance, dramatically reducing compute costs.

Perhaps most importantly, this approach gives teams access to the collective intelligence of the open-source community. Every breakthrough, every specialized adaptation, every novel architecture becomes a potential starting point for their own work.

## Show Us What You Build!🔨

As expected with a large feature of this nature, we're actively improving the experience based on real usage, so your feedback directly shapes the platform! 

Start with our [implementation guide](https://docs.together.ai/docs/fine-tuning-byom) for examples and troubleshooting tips. If you run into issues or want to share what you're building, hop into our [Discord](https://discord.gg/9Rk6sSeWEG), our team is on there and the community is pretty active about helping each other out.

If you have any feedback about fine-tuning at Together AI or want to explore it for your tasks in more depth, feel free to [reach out to us](https://www.together.ai/contact)!
