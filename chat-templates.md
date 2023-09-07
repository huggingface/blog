---
title: "Chat Templates: An End to the Silent Performance Killer" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: rocketknight1
---

# Chat Templates

<!-- {blog_metadata} -->
<!-- {authors} -->


> *A spectre is haunting chat models - the spectre of incorrect formatting!*

## tl;dr

Chat models have been trained with very different formats for converting conversations into a single tokenizable string. Using a format different from the format a model was trained with will usually cause severe, silent performance degradation, so matching the format used during training is extremely important! Hugging Face tokenizers now have a `chat_template` attribute that can be used to save the chat format the model was trained with. This attribute contains a Jinja template that converts conversation histories into a correctly formatted string. Please see the technical documentation [link when it's available] for information on how to write and apply chat templates in your code.

## Introduction

If you're familiar with the Transformers library, you've probably written code like this:

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
```
By loading the tokenizer and model from the same checkpoint, you ensure that inputs are tokenized
in the way that the model expects. If you pick a tokenizer from a different model, the input tokenization
might be completely different, and the result will be that your model's performance will be seriously damaged. The term for this is a **distribution shift** - the model has been learning data from one distribution (the tokenization it was trained with), and suddenly it has shifted to a completely different one. 

Whether you're fine-tuning a model or using it directly for inference, it's always a good idea to minimize these distribution shifts and keep the input you give it as similar as possible to the input it was trained on. With regular language models, it's relatively easy to do that - simply load your tokenizer and model from the same checkpoint, and you're good to go. 

With chat models, however, it's a bit different. This is because "chat" is not just a single string of text that can be straightforwardly tokenized - it's a sequence of messages, each of which contains a `role`, like `"user"` or `"assistant"` or `"system"`, as well as `content`, which is the actual text of the message. Here's an example chat:
```python
[
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"}
]
```

This sequence of messages needs to be converted into a text string before it can be tokenized and used as input to a model. The problem, though, is that there are many ways to do this conversion! You could, for example, convert the list of messages into an "instant messenger" format:
```
User: Hey there!
Bot: Nice to meet you!
```
Or you could add special tokens, to indicate the roles:
```
[USER] Hey there! [/USER]
[ASST] Nice to meet you! [/ASST]
```
Or you could add tokens to indicate the boundaries between messages, but insert the role information as a string:
```
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
```
There are lots of ways to do this, and none of them is obviously the best or correct way to do it. As a result, different models have been trained with wildly different formatting. I didn't make these examples up; they're all real and being used by at least one active model! But once a model has been trained with a certain format, you really want to ensure that future inputs use the same format, or else you could get a performance-destroying distribution shift.

## Templates: A way to save format information

Right now, if you're lucky, the format you need is correctly documented somewhere in the model card. If you're unlucky, it isn't, so good luck if you want to use that model. Even in the best-case scenario, though, you have to locate the template information and manually code it up in your fine-tuning or inference pipeline. We think this is an especially dangerous issue because using the wrong chat format is a **silent error** - you won't get a loud failure or a Python exception to tell you something is wrong, the model will just perform much worse than it would have with the right format, and it'll be very difficult to debug the cause!

This is the problem that **chat templates** aim to solve. Chat templates are Jinja template strings that are saved and loaded with your tokenizer, and that contain all the information needed to turn a list of chat messages into a correctly formatted input for your model. Here are three chat template strings, corresponding to the three message formats above:

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "User : " }}
    {% else %}
        {{ "Bot : " }}
    {{ message['content'] + '\n' }}
{% endfor %}
```
```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "[USER] " + message['content'] + " [/USER]" }}
    {% else %}
        {{ "[ASST] " + message['content'] + " [/ASST]" }}
    {{ message['content'] + '\n' }}
{% endfor %}
```
```
"{% for message in messages %}"  
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"  
"{% endfor %}"
```

If you're unfamiliar with Jinja, I strongly recommend that you take a moment to look at these template strings, and their corresponding template outputs, and see if you can convince yourself that you understand how the template turns a list of messages into a formatted string! The syntax is very similar to Python in a lot of ways.

## Why templates?

Although Jinja can be confusing at first if you're unfamiliar with it, in practice we find that Python programmers can pick it up quickly. During development of this feature, we considered other approaches, such a limited system that allows users to specify per-role prefixes and suffixes for messages. We found that this could become confusing and unwieldy, and was so inflexible that hacky workarounds were needed for several models. Templating, on the other hand, is powerful enough to cleanly support all of the message formats that we're aware of.

## Why bother doing this? Why not just pick a standard format?

This is an excellent idea! Unfortunately, it's too late, because multiple important models have already been trained with very different chat formats.

However, we can still mitigate this problem a bit. We think the closest thing to a 'standard' for formatting is the [ChatML format](https://github.com/openai/openai-python/blob/main/chatml.md) created by OpenAI. If you're training a new model for chat, and this format is suitable for you, we recommend using it and adding `<|im_start|>` and `<|im_end|>` tokens to your tokenizer. It has the advantage of being very flexible with roles, as the role is just inserted as a string rather than having specific role tokens.

There's also a second reason not to hardcode a standard format, though, beyond the proliferation of existing formats - we expect that templates will be broadly useful in preprocessing for many types of model, including those that might be doing very different things from standard chat. Hardcoding a standard format limits the ability of model developers to use this feature to do things we haven't even thought of yet, whereas templating gives users and developers maximum freedom. We strongly believe that the open-source ecosystem should enable you to do what you want, not dictate to you what you're permitted to do.

## How do templates work?

Chat templates are part of the **tokenizer**, because they fulfill the same role as tokenizers do: They store information about how data is preprocessed, to ensure that you feed data to the model in the same format that it saw during training. We have designed it to be very easy to add template information to an existing tokenizer and save it or upload it to the Hub. 

Before chat templates, chat formatting information was stored at the **class level** - this meant that, for example, all LLaMA checkpoints would get the same chat formatting, using code that was hardcoded in `transformers` for the LLaMA model class. For backward compatibility, model classes that had custom chat format methods have been given **default chat templates** instead.

Default chat templates are also set at the class level, and tell classes like `ConversationPipeline` how to format inputs when the model does not have a chat template. We're doing this **purely for backwards compatibility** - we highly recommend that you explicitly set a chat template on any chat model, even when the default chat template is appropriate. This ensures that any future changes or deprecations in the default chat template don't break your model. Although we will be keeping default chat templates for the forseeable future, we hope to transition all models to explicit chat templates over time, at which point the default chat templates may be removed entirely.

For information about how to set and apply chat templates, please see the technical documentation [link when it's available].

## How do I get started with templates?

Easy! If a tokenizer has the `chat_template` attribute set, it's ready to go. You can use that model and tokenizer in `ConversationPipeline`, or you can call `tokenizer.apply_chat_template()` to format chats for inference or training.

If a tokenizer doesn't have a `chat_template` attribute, it might still work, but it will use the default chat template set for that model class. This is fragile, as we mentioned above, and it's also a source of silent bugs when the class template doesn't match what the model was actually trained with. If you want to use a checkpoint that doesn't have a `chat_template`, we recommend checking docs like the model card to verify what the right format is, and then adding a correct `chat_template`for that format. We recommend doing this even if the default chat template is correct - it future-proofs the model, and also makes it clear that the template is present and suitable. 

You can add a `chat_template` even for checkpoints that you're not the owner of, by opening a [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions). If you'd like to use a checkpoint for chat but you can't find any documentation on the chat format it used, you should probably open an issue on the checkpoint or ping the owner!

## Conclusion: Template philosophy

We think templates are a very exciting change. In addition to resolving a huge source of silent, performance-killing bugs, we think they open up completely new approaches and data modalities. Perhaps most importantly, they also represent a philosophical shift: They take a big function out of the core `transformers` codebase and move it into individual model repos, where users have the freedom to do weird and wild and wonderful things. We're excited to see what uses you find for them!