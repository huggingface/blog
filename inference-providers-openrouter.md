---
title: "Hugging Face and OpenRouter join forces to make open models easy to run"
thumbnail: /blog/assets/inference-providers-openrouter/thumbnail.jpg
authors:
  - user: celinah
  - user: victor
  - user: kramp
  - user: Wauplin
  - user: pierric
  - user: coyotte508
  - user: julien-c
---

# Hugging Face and OpenRouter join forces to make open models easy to run

We're excited to announce that Hugging Face and OpenRouter are partnering to make open models as easy to run as closed ones. This partnership is a natural fit: [OpenRouter](https://openrouter.ai) has the best expertise in inference, and Hugging Face is the home of open AI models. Every HF model that has OpenRouter support will now run with your Hugging Face token, same API, no code change.

OpenRouter now brings the best inference for open models to Hugging Face builders. With one Hugging Face token, you get serverless inference through [Inference Providers](https://huggingface.co/docs/inference-providers), powered by OpenRouter, in addition to storage with [Storage Buckets](https://huggingface.co/storage), compute with [Jobs](https://huggingface.co/docs/hub/jobs-overview) and dedicated inference with [Inference Endpoints](https://huggingface.co/docs/inference-endpoints). PRO, Team and Enterprise plans include monthly credits for compute and inference.

Let's go 🚀

The router supports OpenAI's Chat Completions and Responses APIs and Anthropic's Messages API for every chat model, whatever the provider behind it. Pick your coding agent or SDK and a model, then copy the steps, or copy the whole setup as a prompt for your agent:

<iframe src="https://huggingface-inference-providers-setup.static.hf.space" width="100%" height="560" frameborder="0"></iframe>

📚 Setup guides for [Claude Code](https://huggingface.co/docs/inference-providers/integrations/claude-code), [Codex](https://huggingface.co/docs/inference-providers/integrations/codex), [OpenCode](https://huggingface.co/docs/inference-providers/integrations/opencode), [Pi](https://huggingface.co/docs/inference-providers/integrations/pi) and [Hermes Agent](https://huggingface.co/docs/inference-providers/integrations/hermes-agent)

## OpenRouter to power serverless inference on the Hub

OpenRouter serves more than 800 trillion tokens a month to over 10 million users, across 600+ models from 80+ providers. Its routing is based on live measurements of every provider across that traffic, and requests to language models on Inference Providers now go through it.

OpenRouter just [rebuilt its provider routing](https://openrouter.ai/blog/announcements/how-we-rebuilt-provider-routing/): instead of only looking at the lowest list price, it sends each request to the provider with the best mix of price, speed and reliability.

It reacts in real time: when a provider slows down or starts failing, traffic moves away from it until it recovers, and a failed try is retried on the next best provider inside the same call. In OpenRouter's measurements, the new routing cut capacity errors by 46% and improved time to first token by 16.9% and throughput by 16.3%.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/routing@2x.gif" alt="Default routing retries on another provider when the first one is rate limited, :cheapest picks the lowest price and :fastest the highest throughput"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/routing-dark@2x.gif" alt="Default routing retries on another provider when the first one is rate limited, :cheapest picks the lowest price and :fastest the highest throughput"/>
</div>

Before, Inference Providers picked one provider for your request and returned its error if it failed. Now, the request is retried on another provider. You can still choose a policy or a provider with a suffix on the model id:

```text
zai-org/GLM-5.3            # default: best mix of price, speed and reliability
zai-org/GLM-5.3:fastest    # highest throughput
zai-org/GLM-5.3:cheapest   # lowest price
zai-org/GLM-5.3:novita     # only Novita
```

Routing is also cache-aware: a conversation stays on the provider that holds its prompt cache. To make it explicit, send a stable UUID per conversation in the `X-HF-Session-Id` header. 📚 [Prompt caching guide](https://huggingface.co/docs/inference-providers/prompt-caching)

## Pricing

Pricing doesn't change: you pay the provider's price on your Hugging Face bill, and Hugging Face doesn't add a markup on top of it. PRO users get $2 of credits every month, and Team and Enterprise organizations get $2 per seat. See [Pricing and Billing](https://huggingface.co/docs/inference-providers/pricing) for the details.

## Image, video and audio generation

Image, video and audio generation keep working as they do today, with the same token and the same API. You can also run tens of thousands of [LoRAs](https://huggingface.co/models?inference_provider=all&pipeline_tag=text-to-image&other=lora).

## See where your requests go

On model pages, the widget has a new Providers tab: it lists every provider OpenRouter can route the model to, with its input and output price, speed and uptime over the last 24 hours:

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/widget-providers@2x.png" alt="The Providers tab of the model page widget, with each provider's price, speed and uptime"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/widget-providers-dark@2x.png" alt="The Providers tab of the model page widget, with each provider's price, speed and uptime"/>
</div>

Your [usage overview](https://huggingface.co/settings/inference-providers/overview) breaks your spend down by the provider each request went to:

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/usage-by-provider@2x.png" alt="Daily Inference Providers spend, broken down by provider"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/usage-by-provider-dark@2x.png" alt="Daily Inference Providers spend, broken down by provider"/>
</div>

In your [settings](https://huggingface.co/settings/inference-providers), you can turn off any provider you don't want: the router never sends your requests there. Organizations have the same settings, and they apply to every request billed to the organization.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/settings-providers@2x.png" alt="Inference Providers settings, with a toggle for each provider"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/settings-providers-dark@2x.png" alt="Inference Providers settings, with a toggle for each provider"/>
</div>

## Breaking change: custom provider keys

Requests to language models now go through OpenRouter and are billed to your Hugging Face account or organization, so you can no longer use your own API key for a provider with these models. You can still pick a provider with a suffix on the model id (`model:provider`) or turn providers off in your settings, and your code stays the same. Custom keys for image and video providers keep working.

Very soon, you'll be able to connect your OpenRouter account instead.

## What's next

This is the first step of our partnership with OpenRouter, and more will follow. Next, you'll be able to connect your OpenRouter account, and more tasks will be supported.

Try a model in the [Playground](https://huggingface.co/playground), or read the [Inference Providers docs](https://huggingface.co/docs/inference-providers) to get started.
