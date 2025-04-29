---
title: "The 4 Things Qwen-3’s Chat Template Teaches Us"
thumbnail: /blog/assets/qwen-3-chat-template-deep-dive/thumbnail.jpg
authors:
- user: cfahlgren1
---

# The 4 Things Qwen-3’s Chat Template Teaches Us

_**What a boring Jinja snippet tells us about the new Qwen-3 model.**_

The new Qwen-3 model by [Qwen](https://huggingface.co/qwen) ships with a much more sophisticated chat template than it's predecessors Qwen-2.5 and QwQ. By taking a look at the differences in the Jinja template, we can find interesting insights into the new model.

<h2 style="text-align: center; margin-bottom: 0.5rem; font-style: italic;">Chat Templates</h2>
<ul style="display: flex; justify-content: center; list-style: none; padding: 0; margin: 0;">
  <li style="margin-right: 1rem;"><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B?chat_template=default">Qwen-3 Chat Template</a></li>
  <li style="margin-right: 1rem;"><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct?chat_template=default">Qwen-2.5 Chat Template</a></li>
  <li><a href="https://huggingface.co/Qwen/QwQ-32B?chat_template=default">Qwen-QwQ Chat Template</a></li>
</ul>

## 1. Reasoning doesn't have to be forced

_**and you can do it via a simple prefill...**_

Qwen-3 is unique in it's ability to toggle reasoning via the `enable_thinking` flag. When set to false, the template inserts an empty <think></think> pair, telling the model to skip step‑by‑step thoughts. Earlier models baked the <think> tag into every generation, forcing chain‑of‑thought whether you wanted it or not.

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
_More often than not it will think._

## 2. Context Management Should be Dynamic

_Qwen-3 utilizes a rolling checkpoint system, intelligently preserving or pruning reasoning blocks to maintain relevant context. Older models discarded reasoning prematurely to save tokens._ 

Qwen-3 introduces a "**_rolling checkpoint_**" by traversing the message list in reverse to find the latest user turn that wasn’t a tool echo. For any assistant replies after that index it keeps the full `<think>` blocks; everything earlier is stripped out.

**Why this matters**:
- Keeps the active plan visible during a multi‑step tool call.
- Supports nested tool workflows without losing context.
- Saves tokens by pruning thoughts the model no longer needs.
- Prevents "stale" reasoning from bleeding into new tasks.

### Example

Here's an example of chain-of-thought preservation through tool calls with Qwen-3 and QwQ.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/648a374f00f7a3374ee64b99/7OWKkRuO9Qc2L48LYjxVf.png)
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

## 4. Default Prompts Should be Optional

Qwen‑2.5 automatically inserted a default Alibaba system prompt:

> You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Unlike Qwen-2.5, which automatically inserted a default Alibaba system prompt, Qwen-3 (alongside QwQ) omits any default prompts, allowing developers full control over the model's persona.

## Conclusion

Qwen-3 shows us that through the `chat_template` we can provide better flexibility, smarter context handling, and improved tool interaction. These improvements not only improve capabilities, but also make agentic workflows more reliable and efficent.
