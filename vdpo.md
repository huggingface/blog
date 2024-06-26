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

```python
from datasets import load_dataset, features
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

def format(example):
    images = [example["image"]]
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
    return {"images": images, "prompt": prompt, "chosen": chosen, "rejected": rejected}


dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:5%]")
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
 'rejected': 'Assistant: A leather crafter is more likely to use these tools as they consist of a hole punch, scissors, and a knife. These items are typically used in crafting projects involving fabric or leather materials for various designs and patterns. Paper crafters may also benefit from some of these tools, but their primary focus would be on paper-related projects, which might require different types of tools such as paper cutters or scrapbooking supplies.<end_of_utterance>\n', }
```

The dataset is now formatted in the required chat format, and ready for training.
