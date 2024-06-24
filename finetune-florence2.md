---
title: "Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models" 
thumbnail: /blog/assets/182_finetune-florence/thumbnail.png
authors:
- user: andito
- user: merve
- user: SkalskiP
  guest: true

---

# Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models

Florence 2, released by Microsoft in June 2024, is a foundation vision-language model. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks.

Florence supports captioning, object detection, OCR, and more out of the box. However, your task might not be supported, or you might need to control the model's output for your task. That's when you will need to fine-tune the model.

In this blog, we focus on fine-tuning Florence on DocVQA since the authors report that Florence 2 can perform visual question answering, but their released model didn't include this capability.

## Pre-training Details and Architecture

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/florence-2.png" alt="VLM Structure" style="width: 90%; height: auto;"><br>
 <em>Florence-2 Architecture</em>
</p>

Regardless of the computer vision task being performed, Florence-2 formulates the problem as a sequence-to-sequence task. Florence-2 takes an image and text as input and generates text as output.
The model has a simple structure. It utilizes a DaViT vision encoder to convert images into visual embeddings and BERT to convert text prompts into text and location embeddings. The resulting embeddings are then processed by a standard encoder-decoder transformer architecture, generating text and location tokens.
Florence-2's strength doesn't stem from its architecture, but from the massive dataset it was pre-trained on. The authors noted that leading computer vision datasets typically contain limited information - WIT only includes image/caption pairs, [SA-1B](https://ai.meta.com/datasets/segment-anything/) only contains images and associated segmentation masks. Therefore, they decided to build a new FLD-5B dataset containing a wide range of information about each image - boxes, masks, captions, and grounding.
The dataset creation process was largely automated. The authors used off-the-shelf task-specific models and a set of heuristics and quality checks to clean the obtained results. The result was a new dataset containing over 5 billion annotations for 126 million images, which was used to pre-train the Florence-2 model.
## Original performance on VQA

We experimented with various methods to adapt the model for VQA (Visual Question Answering) responses. The most effective approach we found was region-to-description prompting, though it doesn't fully align with VQA tasks. Captioning provides descriptive information about the image but doesn't allow for direct question input.
We also tested several "unsupported" prompts such as "<VQA>", "<vqa>", and "<Visual question answering>". Unfortunately, these attempts yielded unusable results.

## Performance on DocVQA after fine-tuning

We measure performance using the [Levenshtein's similarity](https://en.wikipedia.org/wiki/Levenshtein_distance), the standard metric for this dataset. Initially, before fine-tuning, the similarity between the model's predictions and the ground truth on the validation dataset was 0, as the model's outputs were not close to the ground truth. After fine-tuning with the training set for seven epochs, the similarity score on the validation set improved to 57.0.
We created a 🤗 [space](https://huggingface.co/spaces/andito/Florence-2-DocVQA) to demo the fine-tuned model. While the model performs well for DocVQA, it still struggles with more general document understanding and isn't very chatty. However, it clearly accomplishes the tasks, demonstrating the significant potential for fine-tuning Florence-2 for downstream tasks. To create a great VQA model, we suggest further fine-tuning Florence-2 using [the cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), and already provide the implemented code on our GitHub page.

To give a solid example, below we provide two inference results before and after fine-tuning. You can also try the model [here](https://huggingface.co/spaces/andito/Florence-2-DocVQA).

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/before-ft.png" alt="VLM Structure" style="width: 90%; height: auto;"><br>
 <em>Before Fine-tuning</em>
</p>

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/after-ft.png" alt="VLM Structure" style="width: 90%; height: auto;"><br>
 <em>After Fine-tuning</em>
</p>

## Fine-tuning Details
The authors utilized a batch size of 2048 for their base model and 3072 for the large model. In our experiments, we adapted this approach to a lower-resource setup. Specifically, we froze the vision encoder and used a batch size of 6 on a single A100 GPU in Colab, or a batch size of 1 with a T4. The authors also reported an improvement in performance when fine-tuning with an unfrozen image encoder compared to freezing it.
In parallel, we conducted an experiment with more resources, fine-tuning the entire model with a batch size of 64. This training process took 70 minutes on a cluster equipped with 8 H100 GPUs.
[Maybe we could give links to all the models here]
In every case, we found a small learning rate of 1e-6 to be beneficial for training. With larger learning rates the model will quickly overfit the training set.
## Code Walkthrough
You can find the notebook that includes the notebook [here](https://colab.research.google.com/drive/1hKDrJ5AH_o7I95PtZ9__VlCTNAo1Gjpf?usp=sharing). We will fine-tune [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) checkpoint on [DocVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) dataset.
Let's start by installing the dependencies.

```python
!pip install -q datasets flash_attn timm einops
```

Load the DocVQA dataset from Hugging Face Hub.

```python
import torch from datasets import load_dataset 
data = load_dataset("HuggingFaceM4/DocumentVQA")
```

We can load the model using `AutoModelForCausalLM` and the processor using `AutoProcessor` classes of transformers library. Note that we need to pass `trust_remote_code` as `True` since this model is not a transformers model.

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6').to(device) 
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')
```

Let's do inference with our dataset first to see how the model performs already with our dataset before fine-tuning.


```python
# a document
image = data['train'][3]['image']

prompt = "DocVQA" + "What do you see in this image?" # task prefix with question

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device) 

generated_ids = model.generate( input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3 ) generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

print(processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height)))
# {'DocVQA': '499150498'}
```

We need to construct our dataset. Note how we are adding a new task prefix `<DocVQA>` before the question when constructing the prompt.

```python
import torch from torch.utils.data import Dataset 

class DocVQADataset(Dataset): 

    def __init__(self, data): 
        self.data = data
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx):
        example = self.data[idx] question = "<DocVQA>" + example['question'] 
        first_answer = example['answers'][0]
        image = example['image']  
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image
```

Let's get to fine-tuning. We will create our dataset, the data collator, and start training. In A100 with 40GB memory, we can fit in 6 examples. If you're training on T4, you can use batch size of 1.

```python
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import (AdamW, AutoProcessor, get_scheduler)

def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers 

train_dataset = DocVQADataset(data['train'])
val_dataset = DocVQADataset(data['validation']) 
batch_size = 6
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
```

We can train the model now.

```python
epochs = 2
optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = epochs * len(train_loader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

for epoch in range(epochs): 
    model.train() 
    train_loss = 0
    i = -1
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        i += 1
        inputs, answers = batch
        input_ids = inputs["input_ids"] pixel_values = inputs["pixel_values"] 
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
            inputs, answers = batch
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

      print(val_loss / len(val_loader))
```

You can save the model and processor by calling `save_pretrained()` on both objects. The resulting model is here. 

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>

<gradio-app theme_mode="light" src="https://andito-Florence-2-DocVQA.hf.space"></gradio-app>

## Useful Resources

- [Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [Notebook for Florence-2 Inference](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)
- [Florence-2 DocVQA Demo](https://huggingface.co/spaces/andito/Florence-2-DocVQA)
- [Florence-2 Demo](https://huggingface.co/spaces/gokaygo)
