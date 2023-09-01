## Chat Templates

> *A spectre is haunting chat models - the spectre of incorrect formatting!*
> \- Karl Marx (probably)

### Summary for the impatient
Chat models have been trained with very different formats for converting chat messages into a tokenizable string. Using a format different from the format a model was trained with will usually cause severe, silent performance degradation, so matching the format used during training is extremely important! Hugging Face tokenizers now have a `chat_template` attribute that can be used to save the chat format the model was trained with. This attribute contains a Jinja template that converts lists of chat messages with `role` and `content` keys into a correctly formatted string. They also have an `apply_chat_template` method which uses that template to convert lists of messages into inputs that are ready to pass to your model.

### Introduction

If you're familiar with the Transformers library, you've probably written code like this:

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
```
By loading the tokenizer and model from the same checkpoint, you ensure that inputs are tokenized
in the way that the model expects. If you pick a tokenizer from a different model, the input tokenization
might be completely different, and the result will be that your model's performance will be seriously damaged. We call this a **distribution shift** - the model has been learning data from one distribution (the tokenization it was trained with), and suddenly it has shifted to a completely different one. 

Whether you're fine-tuning a model or using it directly for inference, it's always a good idea to minimize these distribution shifts and keep the input you give it as similar as possible to the input it was trained on. With regular language models, it's relatively easy to do that - simply load your tokenizer and model from the same checkpoint, and you're good to go. 

With chat models, however, it's a bit different. This is because "chat" is not just a single string of text that can be straightforwardly tokenized - it's a sequence of messages, each of which contains a `role`, like `"user"` or `"assistant"`, as well as `content`, which is the actual text of the message. This sequence of messages needs to be converted into a text string before it can be tokenized and used as input to a model.

The problem, though, is that there are many ways to do this conversion! You could, for example, convert the list of messages into an "instant messenger" format:
```
User: Hey there!
Bot: Nice to meet you!
```
Or you could add special tokens, to indicate the roles:
```
[USER] Hey there! [/USER]
[BOT] Nice to meet you! [/BOT]
```
Or you could add tokens to indicate the boundaries between messages, but insert the role information as a string:
```
<|im_start|>user
Hey there!<|im_end|>
<|im_start|>bot
Nice to meet you!<|im_end|>
```
There are lots of ways to do this, and none of them is obviously the best or correct way to do it. As a result, different models have been trained with wildly different formatting. But once a model has been trained with a certain format, you really want to ensure that future inputs use the same format, or else we could get a performance-destroying distribution shift, just like the one we described above. 

### Templates: A way to save format information

At the time of writing this blog, chat formats are quite chaotic. If you're lucky, the format you need is correctly documented somewhere in the model card. If you're unlucky, it isn't, so good luck if you want to use that model without suffering a catastrophic distribution shift. Even in the best-case scenario, though, you have to locate the template information and manually code it up in your fine-tuning or inference pipeline.

This is the problem that **chat templates** aim to solve. Chat templates are Jinja template strings that are saved and loaded with your tokenizer, and that contain all the information needed to turn a list of chat messages into a correctly formatted input for your model. Here are three chat template strings, corresponding to the three message formats above:

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "User :" }}
    {% else %}
        {{ "Bot :" }}
    {{ message['content'] + '\n' }}
{% endfor %}
```
```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ "[USER] " + message['content'] + " [/USER]" }}
    {% else %}
        {{ "[BOT] " + message['content'] + " [/BOT]" }}
    {{ message['content'] + '\n' }}
{% endfor %}
```
```
"{% for message in messages %}"  
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"  
"{% endfor %}"
```

If you're unfamiliar with Jinja, I strongly recommend that you take a moment to look at these template strings, and their corresponding template outputs, and see if you can convince yourself that you understand how the template turns a list of messages into a formatted string! The syntax is very similar to Python in a lot of ways.

### Saving, loading and using templates

### How to choose a template


