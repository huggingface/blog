---
title: "Parameter-Efficient Fine-Tuning using 🤗 PEFT"
thumbnail: /blog/assets/130_peft/thumbnail.png
authors:
- user: smangrul
- user: sayakpaul
---

<h1>Parameter-Efficient Fine-Tuning using 🤗 PEFT</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->

# Motivation
Large Language Models (LLMs) based on Transformers architectures like GPT, T5 and BERT have achieved state-of-the-art results in various Natural Language Processing (NLP) tasks and have also started foraying in other domains such as Computer Vision (CV) and Audio. The conventional paradigm is large-scale pretraining on generic web-scale data followed by finetuning to downstream tasks. Finetuning of these pretrained LLMs on downstream datasets which results in huge perforamnce gains when compared to using the pretrained LLMs out-of-the-box. However, as models get larger and larger, full finetuning becomes infeasible to train on consumer hardware and storing and deploying fine-tuned models independently for each downstream task becomes very expensive. Parameter-Efficient Fine-tuning (PEFT) approaches are meant to address this problem.

PEFT approaches only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs. This also overcomes the issues of catastrophic forgetting benaviour observed in full finetuning of LLMs. PEFT approaches have also shown to be better than fine-tuning in low-data regime and generalizes better to out-of-domain scenarios. It also helps in portability wherein users can tune models using PEFT methods to get tiny checkpoints worth few MBs compared to the large checkpoints of full fine-tuning, e.g., `bigscience/mt0-xxl` takes up 40GB of storage and full fine-tuning will lead to 40GB checkpoint for each downstream dataset wheread using PEFT methods it would be few MBs for each downstream dataset all the while achieving comparable performance to full fine-tuning. **In short, PEFT approaches enables you to get perfoemance comparable to full fine-tuning while only having small number of trainable parameters.**  

Hugging Face is excited to introduce 🤗 PEFT library which provides latest Parameter-Efficient Fine-tuning techniques seamlessly integrated with 🤗 Transformers and 🤗 Accelerate. This enables using most popular and performant models from 🤗 Transformers coupled with the simplicity and sclability of 🤗 Accelerate. Below are the currently supported PEFT methods with more coming soon:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 
4. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf) 

## Use Cases

We focus on the plethora of interesting use-cases in the [README](https://github.com/huggingface/peft#use-cases) of the library. Highlighting below few of the more interesting ones:

1. Using 🤗 PEFT LoRA for tuning `bigscience/T0_3B` model (3 Billion parameters) on consumer hardware with 16GB GPU using
🤗 Accelerate's DepSpeed integration: [peft_lora_seq2seq_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). This means you can tune such large LLMs in Google Colab.

2. Taking previous example a notch up by enabling INT8 tuning of `OPT-6.7b` model (6.7 Billion parameters) in Google Colab
using 🤗 PEFT LoRA and bitsandbytes: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing)

3. Stable Diffusion Dreambooth tuning using 🤗 PEFT on consumer hardware. Try out the 🤗 Gradio Space which should run seamlessly on a T4 instance (16GB GPU): [smangrul/peft-lora-sd-dreambooth](https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth).

![peft lora dreambooth gradio space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/peft_lora_dreambooth_gradio_space.png)

## Training your own model using 🤗 PEFT

Let's consider the case of tuning `bigscience/mt0-large` using LoRA.  

1. Let's get the necessary imports

```diff
from transformers import AutoModelForSeq2SeqLM
+ from peft import get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
```

2. Creating config corresponding to the PEFT method
```
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
```

3. Wrapping base 🤗 Transformers model by calling `get_peft_model`
```diff
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
+ model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

That's it! Rest of the training loop remains same. 

4. When you are ready to save the model for inference, just do the following. For exmaple, you can find the `bigscience/T0_3B` tuned using LoRA on `twitter_complaints` raft dataset here: [smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM](https://huggingface.co/smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM). Notice that it has 2 files `adapter_config.json` and `adapter_model.bin` with the latter being just `19MB`.  
```
model.save_pretrained("output_dir") # model.push_to_hub("my_awesome_peft_model")
```

5. To load it for inference, follow the below snippet:
```diff
from transformers import AutoModelForSeq2SeqLM
+ from peft import PeftModel, PeftConfig

peft_model_id = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
+ model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = model.to(device)
model.eval()
inputs = tokenizer("Tweet text : @HondaCustSvc Your customer service has been horrible during the recall process. I will never purchase a Honda again. Label :", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
# 'complaint'
```

## Next steps
We've released PEFT as an efficient way of tuning large LLMs on downstream tasks and domains saving a lot of compute and storage while achieving comparable performance to full finetuning. In the coming months, we'll be exploring more PEFT methods such as (IA)3 and bottleneck adapters. Also focus on new use cases such as INT8 training of `wishper-large` model in Colab and tuning of RLHF compnents such as policy and ranker using PEFT approaches. 

In the meantime, we're excited to see how industry practitioners apply PEFT to their use cases - if you have any questions or feedback, open an issue on our [GitHub repo](https://github.com/huggingface/peft) 🤗.

Happy Parameter-Efficient Fine-Tuning!