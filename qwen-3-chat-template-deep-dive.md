---
title: "The 4 Things Qwen-3’s Chat Template Teaches Us"
thumbnail: /blog/assets/qwen-3-chat-template-deep-dive/thumbnail.png
authors:
- user: cfahlgren1
---

# The 4 Things Qwen-3’s Chat Template Teaches Us

_**What a boring Jinja snippet tells us about the new Qwen-3 model.**_

The new Qwen-3 model by [Qwen](https://huggingface.co/qwen) ships with a much more sophisticated chat template than its predecessors Qwen-2.5 and QwQ. By taking a look at the differences in the Jinja template, we can find interesting insights into the new model.

<h2 style="text-align: center; margin-bottom: 0.5rem; font-style: italic;">Chat Templates</h2>
<ul style="display: flex; justify-content: center; list-style: none; padding: 0; margin: 0;">
  <li style="margin-right: 1rem;"><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B?chat_template=default">Qwen-3 Chat Template</a></li>
  <li style="margin-right: 1rem;"><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct?chat_template=default">Qwen-2.5 Chat Template</a></li>
  <li><a href="https://huggingface.co/Qwen/QwQ-32B?chat_template=default">Qwen-QwQ Chat Template</a></li>
</ul>


## What is a Chat Template?

A [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) defines how conversations between users and models are structured and formatted. The template acts as a translator, converting a human-readable conversation: 

```js
  [
    { role: "user", content: "Hi there!" },
    { role: "assistant", content: "Hi there, how can I help you today?" },
    { role: "user", content: "I'm looking for a new pair of shoes." },
  ]
```

into a model friendly format:

```xml
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Hi there, how can I help you today?<|im_end|>
<|im_start|>user
I'm looking for a new pair of shoes.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

You can easily view the chat template for a given model on the Hugging Face model page.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/qwen-3-chat-template/qwen-3-chat-template.png)
<p style="text-align:center; font-style:italic; font-size:medium;">
  Chat Template for <a href="https://huggingface.co/Qwen/Qwen3-235B-A22B?chat_template=default" target="_blank"> Qwen/Qwen3-235B-A22B </a>
</p>

Let's dive into the Qwen-3 chat template and see what we can learn!
## 1. Reasoning doesn't have to be forced

_**and you can make it optional via a simple prefill...**_

Qwen-3 is unique in its ability to toggle reasoning via the `enable_thinking` flag. When set to false, the template inserts an empty `<think></think>` pair, telling the model to skip step‑by‑step thoughts. Earlier models baked the `<think>` tag into every generation, forcing chain‑of‑thought whether you wanted it or not.

```jinja
{# Qwen-3 #}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

QwQ for example, forces reasoning in every conversation.

```jinja
{# QwQ #}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
```

If the `enable_thinking` is true, the model is able to decide whether to think or not. 

You can test test out the template with the following code:

```js
import { Template } from "@huggingface/jinja";
import { downloadFile } from "@huggingface/hub";

const HF_TOKEN = process.env.HF_TOKEN;

const file = await downloadFile({
  repo: "Qwen/Qwen3-235B-A22B",
  path: "tokenizer_config.json",
  accessToken: HF_TOKEN,
});
const config = await file!.json();

const template = new Template(config.chat_template);
const result = template.render({
  messages,
  add_generation_prompt: true,
  enable_thinking: false,  
  bos_token: config.bos_token,
  eos_token: config.eos_token,
});
```

## 2. Context Management Should be Dynamic

_Qwen-3 utilizes a rolling checkpoint system, intelligently preserving or pruning reasoning blocks to maintain relevant context. Older models discarded reasoning prematurely to save tokens._ 

Qwen-3 introduces a "**_rolling checkpoint_**" by traversing the message list in reverse to find the latest user turn that wasn’t a tool call. For any assistant replies after that index it keeps the full `<think>` blocks; everything earlier is stripped out.

**Why this matters**:
- Keeps the active plan visible during a multi‑step tool call.
- Supports nested tool workflows without losing context.
- Saves tokens by pruning thoughts the model no longer needs.
- Prevents "stale" reasoning from bleeding into new tasks.

### Example

Here's an example of chain-of-thought preservation through tool calls with Qwen-3 and QwQ.
![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/qwen-3-chat-template/qwen-chat-output.png)
<p style="text-align:center; font-style:italic; font-size:medium;">
  Check out <a href="https://www.npmjs.com/package/@huggingface/jinja">@huggingface/jinja</a> for testing out the chat templates
</p>

## 3. Tool Arguments Need Better Serialization

Before, every `tool_call.arguments` field was piped through ` | tojson`, even if it was already a JSON‑encoded string—risking double‑escaping. Qwen‑3 checks the type first and only serializes when necessary.

```jinja
{# Qwen3 #}
{%- if tool_call.arguments is string %}
    {{- tool_call.arguments }}
{%- else %}
    {{- tool_call.arguments | tojson }}
{%- endif %}
```

## 4. There's No Need for a Default System Prompt

Like many models, the Qwen‑2.5 series has a default system prompt.

> You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

This is pretty common as it helps models respond to user questions like "Who are you?"

Qwen-3 and QwQ ship without this default system prompt. Despite this, the model can still accurately identify its creator if you ask it.

## Conclusion

Qwen-3 shows us that through the `chat_template` we can provide better flexibility, smarter context handling, and improved tool interaction. These improvements not only improve capabilities, but also make agentic workflows more reliable and efficent.
