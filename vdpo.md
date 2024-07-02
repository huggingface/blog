---
title: 'Direct Preference Optimization for Visual Language Models with TRL'
thumbnail: /blog/assets/vdpo/thumbnail.png
authors:
- user: qgallouedec
---

# Direct Preference Optimization for Visual Language Models with TRL

It is now possible to fine-tune visual language models using the [TRL](https://huggingface.co/docs/trl/index) library.
This article will guide you through the process of direct preference optimization for visual language models using TRL.

## Dataset

Preference optimisation requires data that captures user preferences. For example, you need to have samples like the following:

![Example Image](image.jpg)

**❔ Question**: _What is the setting or environment in which the image takes place?_

- **❌ Rejected:** _The image is set in an open area with train tracks, grassy fields, and trees in the background._
- **✅ Chosen:** _The image depicts a train traveling on a track through a countryside setting with tall grass, trees, and power lines in the background._

For this blog post, we'll use the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) dataset, which contains 33k+ rows annotated in this way. Let's have a look at what this dataset looks like:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("openbmb/RLAIF-V-Dataset")
>>> sample = dataset["train"][0]
>>> sample["image"].show()
>>> sample["question"]
'Who is more likely to use these tools a leather crafter or a paper crafter?'
>>> sample["chosen"]
'A leather crafter is more likely to use these tools. The image shows various crafting tools, including scissors and a hole punch, which are commonly used in leatherworking projects. Leather is a material that requires cutting, shaping, and precise hole-punching techniques to create desired designs or patterns. In contrast, paper crafters typically use different types of tools, such as adhesives, decorative papers, or specialized cutting machines like the Silhouette Cameo, for their projects.'
>>> sample["rejected"]
'A leather crafter is more likely to use these tools as they consist of a hole punch, scissors, and a knife. These items are typically used in crafting projects involving fabric or leather materials for various designs and patterns. Paper crafters may also benefit from some of these tools, but their primary focus would be on paper-related projects, which might require different types of tools such as paper cutters or scrapbooking supplies.'
```

## Formatting the Dataset

Since this is an interaction, we need to format the entire dataset in the form of a chat. To do this...
We also need to ensure that the size are at most 640x640 (otherwise we resize)

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
    example["image"].thumbnail((640, 640)) # Make sure the image is at most 640x640
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

column_names = dataset.column_names
dataset = dataset.map(format, writer_batch_size=10, remove_columns=column_names)

# Make sure that the images are decoded, it prevent from storing bytes
f = dataset.features
f["images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
dataset = dataset.cast(f)
```

```python
>>> dataset[0]
{'prompt': 'User:<image>Who is more likely to use these tools a leather crafter or a paper crafter?<end_of_utterance>\n',
 'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x16EF92860>],
 'chosen': 'Assistant: A leather crafter is more likely to use these tools. The image shows various crafting tools, including scissors and a hole punch, which are commonly used in leatherworking projects. Leather is a material that requires cutting, shaping, and precise hole-punching techniques to create desired designs or patterns. In contrast, paper crafters typically use different types of tools, such as adhesives, decorative papers, or specialized cutting machines like the Silhouette Cameo, for their projects.<end_of_utterance>\n',
 'rejected': 'Assistant: A leather crafter is more likely to use these tools as they consist of a hole punch, scissors, and a knife. These items are typically used in crafting projects involving fabric or leather materials for various designs and patterns. Paper crafters may also benefit from some of these tools, but their primary focus would be on paper-related projects, which might require different types of tools such as paper cutters or scrapbooking supplies.<end_of_utterance>\n'}
```

The dataset is now formatted in the required chat format, and ready for training.

## Training

### Choosing the parameters

At this stage, we need to define the training parameters.
The first question is about the GPU memory.
En grosse approximation, l'entrainement va necessiter le stockage de:

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
