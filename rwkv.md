---
title: "Introducing RWKV - An RNN with the advantages of a transformer" 
thumbnail: /blog/assets/142_rwkv/rwkv_thumbnail.png
authors:
- user: BLinkDL
  guest: true
- user: Hazzzardous
  guest: true
- user: sgugger
- user: ybelkada
---

# Introducing RWKV - An RNN with the advantages of a transformer


ChatGPT and chatbot-powered applications have captured significant attention in the Natural Language Processing (NLP) domain. The community is constantly seeking strong, reliable and open-source models for their applications and use cases. 
The rise of these powerful models stems from the democratization and widespread adoption of transformer-based models, first introduced by Vaswani et al. in 2017. These models significantly outperformed previous SoTA NLP models based on Recurrent Neural Networks (RNNs), which were considered dead after that paper.
Through this blogpost, we will introduce the integration of a new architecture, RWKV, that combines the advantages of both RNNs and transformers, and that has been recently integrated into the Hugging Face [transformers](https://github.com/huggingface/transformers) library.

### Overview of the RWKV project

The RWKV project was kicked off and is being led by [Bo Peng](https://github.com/BlinkDL), who is actively contributing and maintaining the project. The community, organized in the official discord channel, is constantly enhancing the project’s artifacts on various topics such as performance (RWKV.cpp, quantization, etc.), scalability (dataset processing & scrapping) and research (chat-fine tuning, multi-modal finetuning, etc.). The GPUs for training RWKV models are donated by Stability AI.

You can get involved by joining the [official discord channel](https://discord.gg/qt9egFA7ve) and learn more about the general ideas behind RWKV in these two blogposts: https://johanwind.github.io/2023/03/23/rwkv_overview.html / https://johanwind.github.io/2023/03/23/rwkv_details.html 

### Transformer Architecture vs RNNs

The RNN architecture is one of the first widely used Neural Network architectures for processing a sequence of data, contrary to classic architectures that take a fixed size input. It takes as input the current “token” (i.e. current data point of the datastream), the previous “state”, and computes the predicted next token, and the predicted next state. The new state is then used to compute the prediction of the next token, and so on.
A RNN can be also used in different “modes”, therefore enabling the possibility of applying RNNs on different scenarios, as denoted by [Andrej Karpathy’s blogpost](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), such as one-to-one (image-classification), one-to-many (image captioning), many-to-one (sequence classification), many-to-many (sequence generation), etc.

| ![rnn_diagram](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/RNN-scheme.png) |
|:--:|
| <b>Overview of possible configurations of using RNNs. Source: <a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/" rel="noopener" target="_blank" >Andrej Karpathy's blogpost</a>  </b>|

Because RNNs use the same weights to compute predictions at every step, they struggle to memorize information for long-range sequences due to the vanishing gradient issue. Efforts have been made to address this limitation by introducing new architectures such as LSTMs or GRUs. However, the transformer architecture proved to be the most effective thus far in resolving this issue.

In the transformer architecture, the input tokens are processed simultaneously in the self-attention module. The tokens are first linearly projected into different spaces using the query, key and value weights. The resulting matrices are directly used to compute the attention scores (through softmax, as shown below), then multiplied by the value hidden states to obtain the final hidden states. This design enables the architecture to effectively mitigate the long-range sequence issue, and also perform faster inference and training compared to RNN models. 

| ![transformer_diagram](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/transformer-scheme.png) |
|:--:|
| <b>Formulation of attention scores in transformer models. Source: <a href="https://jalammar.github.io/illustrated-transformer/" rel="noopener" target="_blank" >Jay Alammar's blogpost</a>  </b>|

| ![rwkv_attention_formula](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/RWKV-formula.png)|
|:--:|
| <b>Formulation of attention scores in RWKV models. Source: <a href="https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-formula.png" rel="noopener" target="_blank" >RWKV blogpost</a>  </b>|

During training, Transformer architecture has several advantages over traditional RNNs and CNNs. One of the most significant advantages is its ability to learn contextual representations. Unlike the RNNs and CNNs, which process input sequences one word at a time, Transformer architecture processes input sequences as a whole. This allows it to capture long-range dependencies between words in the sequence, which is particularly useful for tasks such as language translation and question answering.

During inference, RNNs have some advantages in speed and memory efficiency. These advantages include simplicity, due to needing only matrix-vector operations, and memory efficiency, as the memory requirements do not grow during inference. Furthermore, the computation speed remains the same with context window length due to how computations only act on the current token and the state.

## The RWKV architecture

RWKV is inspired by [Apple’s Attention Free Transformer](https://machinelearning.apple.com/research/attention-free-transformer). The architecture has been carefully simplified and optimized such that it can be transformed into an RNN. In addition, a number of tricks has been added such as `TokenShift` & `SmallInitEmb` (the list of tricks is listed in [the README of the official GitHub repository](https://github.com/BlinkDL/RWKV-LM/blob/main/README.md#how-it-works)) to boost its performance to match GPT. Without these, the model wouldn't be as performant.
For training, there is an infrastructure to scale the training up to 14B parameters as of now, and some issues have been iteratively fixed in RWKV-4 (latest version as of today), such as numerical instability.

### RWKV as a combination of RNNs and transformers

How to combine the best of transformers and RNNs? The main drawback of transformer-based models is that it can become challenging to run a model with a context window that is larger than a certain value, as the attention scores are computed simultaneously for the entire sequence. 

RNNs natively support very long context lengths - only limited by the context length seen in training, but this can be extended to millions of tokens with careful coding. Currently, there are RWKV models trained on a context length of 8192 (`ctx8192`) and they are as fast as `ctx1024` models and require the same amount of RAM.

The major drawbacks of traditional RNN models and how RWKV is different:

1. Traditional RNN models are unable to utilize very long contexts (LSTM can only manage ~100 tokens when used as a LM). However, RWKV can utilize thousands of tokens and beyond, as shown below: 

| ![rwkv_loss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/RWKV-loss.png) |
|:--:|
| <b>LM loss with respect to different context lengths and model sizes. Source: <a href="https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-ctxlen.png" rel="noopener" target="_blank" >RWKV original repository</a>  </b>|

2. Traditional RNN models cannot be parallelized when training. RWKV is similar to a “linearized GPT” and it trains faster than GPT.

By combining both advantages into a single architecture, the hope is that RWKV can grow to become more than the sum of its parts.

### RWKV attention formulation

The model architecture is very similar to classic transformer-based models (i.e. an embedding layer, multiple identical layers, layer normalization, and a Causal Language Modeling head to predict the next token). The only difference is on the attention layer, which is completely different from the traditional transformer-based models.

To gain a more comprehensive understanding of the attention layer, we recommend to delve into the detailed explanation provided in [a blog post by Johan Sokrates Wind](https://johanwind.github.io/2023/03/23/rwkv_details.html).

### Existing checkpoints

#### Pure language models: RWKV-4 models

Most adopted RWKV models range from ~170M parameters to 14B parameters. According to the RWKV overview [blog post](https://johanwind.github.io/2023/03/23/rwkv_overview.html), these models have been trained on the Pile dataset and evaluated against other SoTA models on different benchmarks, and they seem to perform quite well, with very comparable results against them.

| ![rwkv_loss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/RWKV-eval.png) |
|:--:|
| <b>RWKV-4 compared to other common architectures. Source: <a href="https://johanwind.github.io/2023/03/23/rwkv_overview.html" rel="noopener" target="_blank" >Johan Wind's blogpost</a>  </b>|


#### Instruction Fine-tuned/Chat Version: RWKV-4 Raven

Bo has also trained a “chat” version of the RWKV architecture, the RWKV-4 Raven model. It is a RWKV-4 pile (RWKV model pretrained on The Pile dataset) model fine-tuned on ALPACA, CodeAlpaca, Guanaco, GPT4All, ShareGPT and more. The model is available in multiple versions, with models trained on different languages (English only, English + Chinese + Japanese, English + Japanese, etc.) and different sizes (1.5B parameters, 7B parameters, 14B parameters). 

All the HF converted models are available on Hugging Face Hub, in the [`RWKV` organization](https://huggingface.co/RWKV).

## 🤗 Transformers integration

The architecture has been added to the `transformers` library thanks to [this Pull Request](https://github.com/huggingface/transformers/pull/22797). As of the time of writing, you can use it by installing `transformers` from source, or by using the `main` branch of the library. The architecture is tightly integrated with the library, and you can use it as you would any other architecture.

Let us walk through some examples below.

### Text Generation Example

To generate text given an input prompt you can use `pipeline` to generate text:

```python
from transformers import pipeline

model_id = "RWKV/rwkv-4-169m-pile"

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

pipe = pipeline("text-generation", model=model_id)
print(pipe(prompt, max_new_tokens=20))
>>> [{'generated_text': '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.\n\nThe researchers found that the dragons were able to communicate with each other, and that they were'}]
```

Or you can run and start from the snippet below:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=20)
print(tokenizer.decode(output[0].tolist()))
>>> In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.\n\nThe researchers found that the dragons were able to communicate with each other, and that they were
```

### Use the raven models (chat models)

You can prompt the chat model in the alpaca style, here is an example below:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "RWKV/rwkv-raven-1b5"

model = AutoModelForCausalLM.from_pretrained(model_id).to(0)
tokenizer = AutoTokenizer.from_pretrained(model_id)

question = "Tell me about ravens"
prompt = f"### Instruction: {question}\n### Response:"

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"], max_new_tokens=100)

print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
>>> ### Instruction: Tell me about ravens
### Response: RAVENS are a type of bird that is native to the Middle East and North Africa. They are known for their intelligence, adaptability, and their ability to live in a variety of environments. RAVENS are known for their intelligence, adaptability, and their ability to live in a variety of environments. They are known for their intelligence, adaptability, and their ability to live in a variety of environments.
```

According to Bo, better instruction techniques are detailed in [this discord message (make sure to join the channel before clicking)](https://discord.com/channels/992359628979568762/1083107245971226685/1098533896355848283)

| ![discord_message](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/142_rwkv/RWKV%20instructions.png) |

### Weights conversion

Any user could easily convert the original RWKV weights to the HF format by simply running the conversion script provided in the `transformers` library. First, push the "raw" weights to the Hugging Face Hub (let's denote that repo as `RAW_HUB_REPO`, and the raw file `RAW_FILE`), then run the conversion script:

```bash
python convert_rwkv_checkpoint_to_hf.py --repo_id RAW_HUB_REPO --checkpoint_file RAW_FILE --output_dir OUTPUT_DIR
```

If you want to push the converted model on the Hub (let's say, under `dummy_user/converted-rwkv`), first forget to log in with `huggingface-cli login` before pushing the model, then run:

```bash
python convert_rwkv_checkpoint_to_hf.py --repo_id RAW_HUB_REPO --checkpoint_file RAW_FILE --output_dir OUTPUT_DIR --push_to_hub --model_name dummy_user/converted-rwkv
```

## Future work

### Multi-lingual RWKV

Bo is currently working on a multilingual corpus to train RWKV models. Recently a new multilingual tokenizer [has been released](https://twitter.com/BlinkDL_AI/status/1649839897208045573).

### Community-oriented and research projects 

The RWKV community is very active and working on several follow up directions, a list of cool projects can be find in a [dedicated channel on discord (make sure to join the channel before clicking the link)](https://discord.com/channels/992359628979568762/1068563033510653992). 
There is also a channel dedicated to research around this architecure, feel free to join and contribute!

### Model Compression and Acceleration

Due to only needing matrix-vector operations, RWKV is an ideal candidate for non-standard and experimental computing hardware, such as photonic processors/accelerators.

Therefore, the architecture can also naturally benefit from classic acceleration and compression techniques (such as [ONNX](https://github.com/harrisonvanderbyl/rwkv-onnx), 4-bit/8-bit quantization, etc.), and we hope this will be democratized for developers and practitioners together with the transformers integration of the architecture.

RWKV can also benefit from the acceleration techniques proposed by [`optimum`](https://github.com/huggingface/optimum) library in the near future.
Some of these techniques are highlighted in the [`rwkv.cpp` repository](https://github.com/saharNooby/rwkv.cpp) or [`rwkv-cpp-cuda` repository](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda).

## Acknowledgements

The Hugging Face team would like to thank Bo and RWKV community for their time and for answering our questions about the architecture. We would also like to thank them for their help and support and we look forward to see more adoption of RWKV models in the HF ecosystem.
We also would like to acknowledge the work of [Johan Wind](https://twitter.com/johanwind) for his blogpost on RWKV, which helped us a lot to understand the architecture and its potential.
And finally, we would like to highlight anf acknowledge the work of [ArEnSc](https://github.com/ArEnSc) for starting over the initial `transformers` PR.
Also big kudos to [Merve Noyan](https://huggingface.co/merve), [Maria Khalusova](https://huggingface.co/MariaK) and [Pedro Cuenca](https://huggingface.co/pcuenq) for kindly reviewing this blogpost to make it much better!

## Citation

If you use RWKV for your work, please use [the following `cff` citation](https://github.com/BlinkDL/RWKV-LM/blob/main/CITATION.cff).
