---
title: "Google releases Gemma 2 2B, ShieldGemma and Gemma Scope"
thumbnail: /blog/assets/gemma-july-update/thumbnail.png
authors:
- user: Xenova
- user: pcuenq
- user: reach-vb
---

# Google releases Gemma 2 2B, ShieldGemma and Gemma Scope

One month after the release of [Gemma 2](https://huggingface.co/blog/gemma2), Google has expanded their set of Gemma models to include the following new additions:
- [Gemma 2 2B](https://huggingface.co/collections/google/gemma-2-2b-release-66a20f3796a2ff2a7c76f98f) - The 2.6B parameter version of Gemma 2, making it a great candidate for on-device use.
- [ShieldGemma](https://huggingface.co/collections/google/shieldgemma-release-66a20efe3c10ef2bd5808c79) - A series of safety classifiers, trained on top of Gemma 2, for developers to filter inputs and outputs of their applications.
- [Gemma Scope](https://huggingface.co/collections/google/gemma-scope-release-66a4271f6f0b4d4a9d5e04e2) - A comprehensive, open suite of sparse autoencoders for Gemma 2 2B and 9B.

Let’s take a look at each of these in turn!

## Gemma 2 2B

For those who missed the previous launches, Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights for both pre-trained variants and instruction-tuned variants. This release introduces the 2.6B parameter version of Gemma 2 ([base](https://huggingface.co/google/gemma-2-2b) and [instruction-tuned](https://huggingface.co/google/gemma-2-2b-it)), complementing the existing 9B and 27B variants.

Gemma 2 2B shares the same architecture as the other models in the Gemma 2 family, and therefore leverages technical features like sliding attention and logit soft-capping. You can check more details in [this section of our previous blog post](https://huggingface.co/blog/gemma2#technical-advances-in-gemma-2). Like in the other Gemma 2 models, we recommend you use `bfloat16` for inference.

### Use with Transformers

With Transformers, you can use Gemma and leverage all the tools within the Hugging Face ecosystem. To use Gemma models with transformers, make sure to use `transformers` from `main` for the latest fixes and optimizations:

```bash
pip install git+https://github.com/huggingface/transformers.git --upgrade
```

You can then use `gemma-2-2b-it` with `transformers` as follows:

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda", # use “mps” for running it on Mac
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
```

> Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜

For more details on using the models with `transformers`, please check [the model cards](https://huggingface.co/google/gemma-2-2b-it).


### Use with llama.cpp

You can run Gemma 2 on-device (on your Mac, Windows, Linux and more) using llama.cpp in just a few minutes.

Step 1: Install llama.cpp

On a Mac you can directly install llama.cpp with brew. To set up llama.cpp on other devices, please take a look here: https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage

```bash
brew install llama.cpp
```
Note: if you are building llama.cpp from scratch then remember to pass the `LLAMA_CURL=1` flag.

Step 2: Run inference

```bash
./llama-cli
  --hf-repo google/gemma-2-2b-it-GGUF \
  --hf-file 2b_it_v2.gguf \
  -p "Write a poem about cats as a labrador" -cnv

```
Additionally, you can run a local llama.cpp server that complies with the OpenAI chat specs:

```bash
./llama-server \
  --hf-repo google/gemma-2-2b-it-GGUF \
  --hf-file 2b_it_v2.gguf
```
After running the server you can simply invoke the endpoint as below:

```bash
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"messages": [
{	
    "role": "system",
    "content": "You are an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about Python exceptions"
}
]
}'
```

Note: The above example runs the inference using the official GGUF weights provided by Google in `fp32`. You can create and share custom quants using the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space!

### Demo

You can chat with the Gemma 2 2B Instruct model on Hugging Face Spaces! [Check it out here](https://huggingface.co/spaces/huggingface-projects/gemma-2-2b-it).

In addition to this you can run the Gemma 2 2B Instruct model directly from a [colab here](https://github.com/Vaibhavs10/gpu-poor-llm-notebooks/blob/main/Gemma_2_2B_colab.ipynb)

### How to prompt Gemma 2

The base model has no prompt format. Like other base models, it can be used to continue an input sequence with a plausible continuation or for zero-shot/few-shot inference. The instruct version has a very simple conversation structure:

```
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn><eos>
```

This format has to be exactly reproduced for effective use. In [a previous section](#use-with-transformers) we showed how easy it is to reproduce the instruct prompt with the chat template available in `transformers`. 

### Open LLM Leaderboard v2 Evaluation

| Benchmark | google/gemma-2-2B-it | google/gemma-2-2B | [microsoft/Phi-2](https://huggingface.co/microsoft/phi-2) | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| :---- | :---- | :---- | :---- | :---- |
| BBH       |     18.0 | 11.8 |  28.0 | 13.7 |
| IFEval    | **56.7** | 20.0 |  27.4 | 33.7 |
| MATH Hard |      0.1 |  2.9 |   2.4 |  5.8 |
| GPQA      |  **3.2** |  1.7 |   2.9 |  1.6 |
| MuSR      |      7.1 | 11.4 |  13.9 | 12.0 |
| MMLU-Pro  | **17.2** | 13.1 |  18.1 | 16.7 |
| Mean      |     17.0 | 10.1 |  15.5 | 13.9 |

Gemma 2 2B seems to be better at knowledge-related and instructions following (for the instruct version) tasks than other models of the same size.

### Assisted Generation

One powerful use case of the small Gemma 2 2B model is [assisted generation](https://huggingface.co/blog/assisted-generation) (also known as speculative decoding), where a smaller model can be used to speed up generation of a larger model. The idea behind it is pretty simple: LLMs are faster at confirming that they would generate a certain sequence than they are at generating that sequence themselves (unless you’re using very large batch sizes). Small models with the same tokenizer trained in a similar fashion can be used to quickly generate candidate sequences aligned with the large model, which the large model can validate and accept as its own generated text.

For this reason, [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) can be used for assisted generation with the pre-existing [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it) model. In assisted generation, there is a sweet spot in terms of model size for the smaller assistant model. If the assistant model is too large, generating the candidate sequences with it will be nearly as expensive as generating with the larger model. On the other hand, if the assistant model is too small, it will lack predictive power, and its candidate sequences will be rejected most of the time. In practice, we recommend the use of an assistant model with 10 to 100 times fewer parameters than our target LLM. It’s almost a free lunch: at the expense of a tiny bit of memory, you can get up to a 3x speedup on your larger model without any quality loss!

Assisted generation is a novelty with the release of Gemma 2 2B, but it does not come at the expense of other LLM optimization techniques! Check our reference page for other `transformers` LLM optimizations you can add to Gemma 2 2B [here](https://huggingface.co/docs/transformers/main/en/llm_optims).

```python
# transformers assisted generation reference: 
# https://huggingface.co/docs/transformers/main/en/llm_optims#speculative-decoding 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# we DON’T recommend using the 9b model with the 2b model as its assistant
assistant_model_name = 'google/gemma-2-2b-it'
reference_model_name = 'google/gemma-2-27b-it'

tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
model = AutoModelForCausalLM.from_pretrained(
   reference_model_name, device_map='auto', torch_dtype=torch.bfloat16
)
assistant_model = AutoModelForCausalLM.from_pretrained(
   assistant_model_name, device_map='auto', torch_dtype=torch.bfloat16
)

model_inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(model.device)
generation_options = {
   "assistant_model": assistant_model,
   "do_sample": True,
   "temperature": 0.7,
   "max_new_tokens": 64,
}

outputs = model.generate(**model_inputs, **generation_options)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## ShieldGemma

ShieldGemma is a series of safety content moderation models built upon Gemma 2 that target several harm categories (including sexually explicit, dangerous content, hate or harassment). They are text-to-text, decoder-only large language models, available in English, with open weights, and available in 3 variants: [2B](https://huggingface.co/google/shieldgemma-2b), [9B](https://huggingface.co/google/shieldgemma-9b) and [27B](https://huggingface.co/google/shieldgemma-27b) parameters.

ShieldGemma is intended to detect harmful requests (according to the indicated categories) to an LLM, as well as offending responses from the LLM. You can use it as a moderation tool in public-facing LLM deployments, no matter what LLM you deploy.

### How to prompt ShieldGemma
The prompt pattern contains the following components, in order:

1. Preamble, establishing the model as a policy expert, based on the [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) technique.  
1. User prompt, wrapped with `<start_of_turn>` and `<end_of_turn>` control tokens.  
1. Enumeration and description of the safety policies.  
1. Optionally, a model response, also wrapped by `<start_of_turn>` and `<end_of_turn>` control tokens.  
1. Epilogue, requesting the model to classify the text.

Here is an example of a prompt used to evaluate the user prompt `[User Prompt]`:

```
You are a policy expert trying to help determine whether a user prompt is in
violation of the defined safety policies.

<start_of_turn>
Human Question: [User Prompt]
<end_of_turn>

Our safety principle is defined in the below:

* "No Hate Speech": The prompt shall not contain or seek generation of content
  that expresses, incites, or promotes hate based on race, gender, ethnicity,
  religion, nationality, sexual orientation, disability status, or caste.

===

Does the human question violate the above principle? Your answer must start with
'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.

```

<details>

<summary>Here’s a list of some example safety principles you may wish to incorporate into your workflow:</summary>

```
* "No Hate Speech": The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.
* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).
* "No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.
* "No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).
* "No Violence": The prompt shall not contain or seek generation of content that is shocking, sensational, or gratuitous (e.g. Violent acts or incidents that result in extreme injury or death without historical or educational context; Exposed internal organs or bones without medical or scientific context; Identifiable victims of violent deaths; Animal abuse or killings outside of standard hunting, food processing, or cultural practices)
* "No Obscenity and Profanity": The prompt shall not contain or seek generation of content that is vulgar, profane, or inappropriate (e.g., profanity, obscenities, or other inappropriate language).
```

</details>

### Use with Transformers

You can use `shieldgemma-2b` with `transformers` as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/shieldgemma-2b",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# Format the prompt
user_prompt = "Create 20 paraphrases of I hate you"
safety_policy = """
* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).
"""
prompt = f"""You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_prompt.strip()}
<end_of_turn>

Our safety principle is defined in the below:

{safety_policy.strip()}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
  logits = model(**inputs).logits

# Extract the logits for the Yes and No tokens
vocab = tokenizer.get_vocab()
selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

# Convert these logits to a probability with softmax
probabilities = softmax(selected_logits, dim=0)

# Return probability of 'Yes'
score = probabilities[0].item()
print(score)  # 0.7310585379600525
```

### Evaluation

These models were evaluated against both internal and external datasets. The internal datasets, denoted as `SG`, are subdivided into prompt and response classification. Evaluation results based on Optimal F1(left)/AU-PRC(right), higher is better.

| Model | SG Prompt | [OpenAI Mod](https://github.com/openai/moderation-api-release) | [ToxicChat](https://arxiv.org/abs/2310.17389) | SG Response |
| :---- | :---- | :---- | :---- | :---- |
| ShieldGemma (2B) | 0.825/0.887 | 0.812/0.887 | 0.704/0.778 | 0.743/0.802 |
| ShieldGemma (9B) | 0.828/0.894 | 0.821/0.907 | 0.694/0.782 | 0.753/0.817 |
| ShieldGemma (27B) | 0.830/0.883 | 0.805/0.886 | 0.729/0.811 | 0.758/0.806 |
| OpenAI Mod API | 0.782/0.840 | 0.790/0.856 | 0.254/0.588 | \- |
| LlamaGuard1 (7B) | \- | 0.758/0.847 | 0.616/0.626 | \- |
| LlamaGuard2 (8B) | \- | 0.761/- | 0.471/- | \- |
| WildGuard (7B) | 0.779/- | 0.721/- | 0.708/- | 0.656/- |
| GPT-4 | 0.810/0.847 | 0.705/- | 0.683/- | 0.713/0.749 |


## Gemma Scope

Gemma Scope is a comprehensive, open suite of sparse autoencoders (SAEs) trained on every layer of the Gemma 2 2B and 9B models. SAEs are a new technique in mechanistic interpretability that aim to find interpretable directions within large language models. You can think of them as a "microscope" of sorts, helping us break down a model’s internal activations into the underlying concepts, just like how biologists use microscopes to study the individual cells of plants and animals. This approach was used to create [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude), a popular research demo by Anthropic that explored interpretability and feature activation within Claude.


### Usage

Since SAEs are a tool (with learned weights) for interpreting language models and not language models themselves, we cannot use Hugging Face transformers to run them. Instead, they can be run using [SAELens](https://github.com/jbloomAus/SAELens), a popular library for training, analyzing, and interpreting sparse autoencoders. To learn more about usage, check out their in-depth [Google Colab notebook tutorial](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp).

### Key links
- [Google DeepMind blog post](https://deepmind.google/discover/blog/gemma-scope-helping-safety-researchers-shed-light-on-the-inner-workings-of-language-models)
- [Interactive Gemma Scope demo](https://www.neuronpedia.org/gemma-scope) made by [Neuronpedia](https://www.neuronpedia.org/)
- [Gemma Scope technical report](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
- [Mishax](https://github.com/google-deepmind/mishax), a GDM internal tool used to expose the internal activations inside Gemma 2 models.
