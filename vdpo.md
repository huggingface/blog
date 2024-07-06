---
title: 'Direct Preference Optimization for Visual Language Models with TRL'
thumbnail: /blog/assets/vdpo/thumbnail.png
authors:
- user: qgallouedec
- user: vwxyzjn
- user: merve
---

# Direct Preference Optimization for Visual Language Models with TRL

Training models to understand and predict human preferences can be incredibly complex. Traditional methods, like supervised fine-tuning, often require assigning specific labels to data, which can be difficult, especially for nuanced tasks. There is an alternative approach that can simplify this process and yield more accurate results: preference optimization. By focusing on comparing and ranking candidate answers rather than assigning fixed labels, preference optimization allows models to capture the subtleties of human judgment more effectively.

Preference optimization is widely used for finetuning language models, but it can also be applied to visual language models (VLM).
We are excited to announce that the [TRL](https://huggingface.co/docs/trl/index) library now supports direct preference optimization for VLMs. This article will guide you through the process of training VLMs using TRL and direct preference optimization.

## Dataset

Preference optimisation requires data that captures user preferences. For example, you need to have samples like the following:

![Example Image](https://datasets-server.huggingface.co/assets/openbmb/RLAIF-V-Dataset/--/fb08536fc84ca3c8b5aed0bc72b1130b37c7a91e/--/default/train/1/image/image.jpg?Expires=1720283142&Signature=OIttYvFmQtbx6qqxuWi67Y07VxAddRL4dDXjTto-oBT2TPPYJCcttKmLCfmNq2upWmWB~rvxXmcfSXWgMr3uOY6Kp5-dl2vBLO3MjIuDnncyc1sAyFC891BH-PqfeuB2sz6d-JLQLAlL7fBcT5-0WUtbA2fhoep5eqoZcu3As-a0xYvHNKa2W5hNQQxmmIYchY2F7YaFeGzn2r7FM8NIZVbyJRedQ7YSblFitJPvbIu73FBSpAEVrLkVt6WiTdngsqy3GUshAW7JNIIJIurSu51mNuEg8HyFlmBdGOwhk9s9AazS39zt4nowz2snsnTnG53U4GneQpiPo7fMLbZmXg__&Key-Pair-Id=K3EI6M078Z3AC3)

**❔ Question**: _How many families?_

- **❌ Rejected:** _The image does not provide any information about families._
- **✅ Chosen:** _The image shows a Union Organization table setup with 18,000 families._

Note that the chosen message is not necessarily correct. For example, the chosen response that says 18,000 families is still wrong, but it's less wrong compared to the rejected response.

For this blog post, we'll use the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) dataset, which contains 83k+ rows annotated in this way. Let's have a look at what this dataset looks like:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:1%]")
>>> sample = dataset[1]
>>> sample["image"].show()
>>> sample["question"]
'how many families?'
>>> sample["rejected"]
'The image does not provide any information about families.'
>>> sample["chosen"]
'The image shows a Union Organization table setup with 18,000 families.'
```

## Formatting the Dataset

Since this is an interaction, we need to format the entire dataset in the form of a chat. To do this...

```python
from datasets import features
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

def format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    max_size = processor.image_processor.size["longest_edge"]
    example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

# Apply the formatting function to the dataset
dataset = dataset.map(format, remove_columns=dataset.column_names)

# Make sure that the images are decoded, it prevent from storing bytes
f = dataset.features
f["images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
dataset = dataset.cast(f)
```

Our dataset is now formatted. Let's have a look at the first example:

```python
>>> dataset[1]
{'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=L size=980x812 at 0x154505570>],
 'prompt': 'User:<image>how many families?<end_of_utterance>\n',
 'rejected': 'Assistant: The image does not provide any information about families.<end_of_utterance>\n',
 'chosen': 'Assistant: The image shows a Union Organization table setup with 18,000 families.<end_of_utterance>\n'}
```

Warm up your GPUs, the dataset is ready for training!

## Training

In this section, we embark on training Idefics2 using the DPO implementation of TRL with our formatted dataset. Before delving into the training process, we'll first ensure everything fits smoothly into memory.

### Choosing the parameters

I myself have a GPU with 80GB of memory. Is this enough to train my Idefics2-8b model? To answer this question, we need to calculate the memory requirements for training. Here's a rough estimate.
The components to consider are:

- The model to train
- The reference model
- The gradients
- The optimizer states

--------------------------------------------

- Les paramètres du modèle à entrainer
- Les paramètres du modèle de référence
- Les gradients
- Les états de l'optimiseur

Il y a d'autre chose à considérer, mais nous faisons ici un calcul rapide qui nous permet d'avoir une appriximation de la taille requises.

Dans notre cas, le calcul est le suivant:

| Composant           |                                             | Calcul                                 | Mémoire en Gb |
| ------------------- | ------------------------------------------- | -------------------------------------- | ------------- |
| Modèle à entrainer  | Nombre de paramètres \( \times \) Precision | 8b \( \times \) 4 bytes (float32)      | 32            |
| Modèle de référence | Même que le modèle à entrainer              | 8b \( \times \) 4 bytes                | 32            |
| Gradients           | Même que le modèle à entrainer              | 8b \( \times \) 4 bytes                | 32            |
| Optimiseur          | 2* modèle à entrainer (AdamW)               | 2 \( \times \) 8B \( \times \) 4 bytes | 64            |
| **Total**           |                                             |                                        | **160**       |



Aïe, 160 est bien au deussus de ce que mon hardware peut contenir (80Gb). Il va falloir réduire la quantité de mémoire utilisée.

Precision mixte
[Explication de la précision mixte]

Parameters efficient training

Work has shown that it is possible to reduce the search set to a small set of parameters. In particular, LoRA  is a method that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while keeping the original weights frozen. This significantly decreases the storage needs for large language models adapted to specific tasks.
LoRA is integrated in PEFT and you can set it up in no time:

```diff
+ from peft import get_peft_model
  from transformers import AutoModelForVision2Seq
+ from trl get_peft_config
  from trl import ModelConfig

  model_id = "HuggingFaceM4/idefics2-8b"
  model_config = ModelConfig(
      model_id,
+     lora_target_modules="all-linear",
+     use_peft=True,
  )
  model = AutoModelForVision2Seq.from_pretrained(model_id)
+ peft_config = get_peft_config(model_config)
+ model = get_peft_model(model, peft_config)
```

Regardons dans quelle mesure l'utilisation de LoRA permet de réduire le nombre de parametres à entrainer:

```python
>>> model.print_trainable_parameters()
trainable params: 55,348,736 || all params: 8,458,116,848 || trainable%: 0.6543860411799315
```

You pass from 8b to 55m, which is a huge improvement: in term of memory, you go from 32Gb to 0.2Gb

Re-faisons le bilan approximatif de mémoire requis:

| Composant           |                                             | Calcul                                 | Mémoire en Gb |
| ------------------- | ------------------------------------------- | -------------------------------------- | ------------- |
| Modèle à entrainer  | Nombre de paramètres \( \times \) Precision | 8B \( \times \) 2 bytes (bfloat16)     | 16            |
| Modèle de référence | Même que le modèle à entrainer              | 8B \( \times \) 2 bytes                | 16            |
| Gradients           | Même que le modèle à entrainer              | 55m \( \times \) 2 bytes               | 0.2            |
| Optimiseur          | 2* modèle à entrainer                       | 2 \( \times \) 55m \( \times \) 2 bytes | 0.4            |
| **Total**           |                                             |                                        | **32.6**       |

Cette fois, il nous faut autour de 30 Gb de mémoire pour finetuner notre idefics8 ce qui bien plus raisonable, mais surtout, qui fit dans mon GPU!


Putting all together, the training parameters are:

```python
from peft import get_peft_model
from transformers import AutoModelForVision2Seq
from trl get_peft_config
from trl import ModelConfig

model_id = "HuggingFaceM4/idefics2-8b"
model_config = ModelConfig(model_id, lora_target_modules="all-linear", use_peft=True)
model = AutoModelForVision2Seq.from_pretrained(model_id)
peft_config = get_peft_config(model_config)
model = get_peft_model(model, peft_config)
```

To train a model using TRL, you need to define a `TRLTrainer` and pass it the `TRL`.

```python
from transformers import TrainingArguments
model_ref = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b")
trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
)
```


- Training (curves)
- Evulation (embed compraing space, metrics, with polar plot)
- Conclusion
