---
title: "Hugging Face and OpenRouter join forces to make open-weight models easy to run"
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

# Hugging Face and OpenRouter join forces to make open-weight models easy to run

Hugging Face and OpenRouter are partnering to make open-weight models easy to run. Every major open-weight release is published on the Hugging Face Hub, and [OpenRouter](https://openrouter.ai) serves those models at scale, routing each request across 80+ providers based on live measurements of each one. One is where open-weight models are released, the other is where they run. Together, every open-weight model on the Hub that a provider serves can be called with your Hugging Face token, through the same API, with the best provider picked for each request.

[Inference Providers](https://huggingface.co/docs/inference-providers) already let you call models from many providers with one Hugging Face token. What changes is how requests to language models are routed: they now go through OpenRouter, which picks the provider for each call based on live price, speed and reliability, and retries on another provider if the first one fails. Your code stays the same.

## Run them with the tools you already use

The router supports OpenAI's Chat Completions and Responses APIs and Anthropic's Messages API for every chat model, whatever the provider behind it. Pick your coding agent or SDK and a model, then copy the steps, or copy the whole setup as a prompt for your agent:

<iframe src="https://huggingface-inference-providers-setup.static.hf.space" width="100%" height="560" frameborder="0"></iframe>

Setup guides for [Claude Code](https://huggingface.co/docs/inference-providers/integrations/claude-code), [Codex](https://huggingface.co/docs/inference-providers/integrations/codex), [OpenCode](https://huggingface.co/docs/inference-providers/integrations/opencode), [Pi](https://huggingface.co/docs/inference-providers/integrations/pi) and [Hermes Agent](https://huggingface.co/docs/inference-providers/integrations/hermes-agent)

The same Hugging Face token also gives you storage with [Storage Buckets](https://huggingface.co/storage), compute with [Jobs](https://huggingface.co/docs/hub/jobs-overview) and dedicated inference with [Inference Endpoints](https://huggingface.co/docs/inference-endpoints). PRO, Team and Enterprise plans include monthly credits for compute and inference.

## How open-weight models are used

On OpenRouter, open-weight models went from 26.8% of the tokens served in October 2025 to 71.7% at the end of September 2026, after peaking at 77.5% in July ([OpenRouter, October 2026](https://openrouter.ai/blog/insights/open-weight-vs-closed-weight-models/)). Over the same period, weekly open-weight volume grew more than 60-fold, against almost tenfold for closed models. This week, the most used model on OpenRouter is an open one, DeepSeek V4.1 Flash 1017, with 59.2 trillion tokens ([rankings](https://openrouter.ai/rankings)).

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/open-weight-share@2x.png" alt="Open-weight models' share of tokens on OpenRouter, from 26.8% in October 2025 to 71.7% in September 2026"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/open-weight-share-dark@2x.png" alt="Open-weight models' share of tokens on OpenRouter, from 26.8% in October 2025 to 71.7% in September 2026"/>
</div>

Price is a large part of it. Open-weight models cost about four times less on average: $0.28 per million input tokens and $1.62 per million output tokens, against $1.27 and $6.36 for closed models. The other part is competition. The same open weights are served by many providers: 31 for DeepSeek V4.1 Flash and 30 for GLM-5.3 Flash, where closed models have two to five. Providers compete on price and speed for the same model, and when one is slow or down, another one serves the same weights.


The best open-weight models still trail the best proprietary ones on the hardest benchmarks. On the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models), Kimi K3.1 Pro, the best open-weight model, scores 46 against 65 for Claude Fable 5.2, at about a seventh of the cost per task ($3.68 against $26.54). For a lot of workloads, that trade-off is worth it.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/intelligence-vs-cost@2x.png" alt="Artificial Analysis Intelligence Index against cost per task, for the top proprietary and open-weight models"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/intelligence-vs-cost-dark@2x.png" alt="Artificial Analysis Intelligence Index against cost per task, for the top proprietary and open-weight models"/>
</div>

## Routing powered by OpenRouter

OpenRouter serves more than 800 trillion tokens a month to over 10 million users, across 600+ models from 80+ providers. Its routing is based on live measurements of every provider across that traffic.

OpenRouter recently [rebuilt its provider routing](https://openrouter.ai/blog/announcements/how-we-rebuilt-provider-routing/): instead of only looking at the lowest list price, it sends each request to the provider with the best mix of price, speed and reliability.

When a provider slows down or starts failing, traffic moves away from it until it recovers, and a failed try is retried on the next best provider inside the same call. In OpenRouter's measurements, the new routing cut capacity errors by 46% and improved time to first token by 16.9% and throughput by 16.3%.

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

Routing is also cache-aware: a conversation stays on the provider that holds its prompt cache. To make it explicit, send a stable UUID per conversation in the `X-HF-Session-Id` header (see the [prompt caching guide](https://huggingface.co/docs/inference-providers/prompt-caching)).

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

## What's next

This is the first step of our partnership with OpenRouter. Next:

- Connect your OpenRouter account to Hugging Face.
- More tasks routed through OpenRouter.

Try a model in the [Playground](https://huggingface.co/playground), or read the [Inference Providers docs](https://huggingface.co/docs/inference-providers) to get started.
