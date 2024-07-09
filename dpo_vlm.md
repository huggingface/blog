---
title: 'Preference Optimization for Vision Language Models'
thumbnail: /blog/assets/dpo_vlm/thumbnail.png
authors:
- user: qgallouedec
- user: vwxyzjn
- user: merve
---

# Preference Optimization for Vision Language Models with TRL

Training models to understand and predict human preferences can be incredibly complex. Traditional methods, like supervised fine-tuning, often require assigning specific labels to data, which is not cost-efficient, especially for nuanced tasks. Preference optimization is an alternative approach that can simplify this process and yield more accurate results. By focusing on comparing and ranking candidate answers rather than assigning fixed labels, preference optimization allows models to capture the subtleties of human judgment more effectively.

Preference optimization is widely used for fine-tuning language models, but it can also be applied to vision language models (VLM).
We are excited to announce that the [TRL](https://huggingface.co/docs/trl/index) library now supports direct preference optimization for VLMs. This article will guide you through the process of training VLMs using TRL and direct preference optimization.

## Dataset

Preference optimization requires data that captures user preferences. For example, you need to have samples like the following:

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

# Make sure that the images are decoded, it prevents from storing bytes.
# More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
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

In this section, we embark on training Idefics2 using the DPO implementation of TRL with our formatted dataset. Before looking into the training process, we'll first ensure everything fits smoothly into memory.

### Choosing the parameters

I myself have a GPU with 80GB of memory. Is this enough to train my Idefics2-8b model? To answer this question, we need to calculate the memory requirements for training. Here are the calculations steps to get a rough estimate:

Let \( N \) be the number of parameters, \( P \) the precision. We need to consider the following components:

- **Model to train**: \( N \times P \)
- **Reference model**: we use the same model as the one to train, so it also requires \( N \times P \)
- **Gradients**: we train the whole model, and each parameter requires a gradient, so it requires \( N \times P \)
- **Optimizer states**: we use [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), which requires 2 times the number of parameters for the first and second moments: \( 2 \times N \times P \)

Idefics2-8b has 8 billion parameters, and we use float32 precision which requires 4 bytes. So the total memory required is:

| Component        | Calculation                           | Memory (GB) |
| ---------------- | ------------------------------------- | ----------- |
| Model to train   | \( 8 \times 10^9 \times 4 \)          | 32          |
| Reference model  | \( 8 \times 10^9 \times 4 \)          | 32          |
| Gradients        | \( 8 \times 10^9 \times 4 \)          | 32          |
| Optimizer states | \( 2 \times 8 \times 10^9 \times 4 \) | 64          |
| **Total**        |                                       | **160**     |

This is way above my GPU's memory capacity. We need to reduce the memory usage.
To do this, we will use two techniques: quantization and LoRA.

#### Quantization

Quantization is a technique that reduces the precision of the model's weights and activations. We can use `bfloat16` precision instead of `float32`, which requires 2 bytes instead of 4. To do this, we need to load the model with the `bfloat16` precision:

```python
import torch
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16)
```

To use `bfloat16` precision for the optimizer, we also need to set `--bf16` in the training arguments.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(..., bf16=True)
```

#### LoRA

LoRA is a method that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while keeping the original weights frozen. This significantly decreases the storage needs for large language models adapted to specific tasks. LoRA is integrated in PEFT and you can set it up in no time:

```diff
+ from peft import get_peft_model
  from transformers import AutoModelForVision2Seq
+ from trl import get_peft_config
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

How much LoRA reduces the number of trainable parameters?

```python
>>> model.print_trainable_parameters()
trainable params: 55,348,736 || all params: 8,458,116,848 || trainable%: 0.6543860411799315
```

It reduces the number of trainable parameters from 8 billion to 55 million, which is a huge gap.

Now that we have reduced the memory requirements, let's recalculate the memory needed:

| Component        | Calculation                           | Memory (GB) |
| ---------------- | ------------------------------------- | ----------- |
| Model to train   | \( 8 \mathrm{G} \times 2 \)           | 16          |
| Reference model  | \( 8 \mathrm{G} \times 2 \)           | 16          |
| Gradients        | \( 55 \mathrm{M} \times 2 \)          | 0.2         |
| Optimizer states | \( 2 \times 55 \mathrm{M} \times 2 \) | 0.4         |
| **Total**        |                                       | **32.6**    |

This time, we need around 30GB of memory to finetune our Idefics2, which is much more reasonable and fits within my GPU!

#### What about the batch size?

Our memory calculation isn't exact as it doesn't account for activations, which are the intermediate outputs of the network layers. Activations depend on the batch size, which affects the overall memory usage. To choose an appropriate training batch size, a practical approach is to start with your desired batch size (e.g., 64). You will likely encounter an OOM error with this size. If so, halve the batch size until the memory fits within your GPU. For each reduction by half, double the gradient accumulation steps to maintain the same effective batch size. In our case, we end up with a batch size of 2 and gradient accumulation steps of 32.

#### Summary: training script

Now, we've set up the model, the dataset, and the training parameters, and we're ready to train our model. Putting all together, the training script looks like this:

```python
# dpo_idefics2-8b.py
from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


def main():
    # Load the model and processor
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

    # Load the dataset
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")

    def format(example):
        # Prepare the input for the chat template
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        # Apply the chat template
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)
        # Resize the image to ensure it fits within the maximum allowable
        # size of the processor to prevent OOM errors.
        max_size = processor.image_processor.size["longest_edge"] // 2
        example["image"].thumbnail((max_size, max_size))
        return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

    # Apply the formatting function to the dataset
    dataset = dataset.map(format, remove_columns=dataset.column_names, num_proc=32)

    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    # Train the model
    training_args = DPOConfig(
        output_dir="idefics2-8b-dpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        dataset_num_proc=32,
        dataloader_num_workers=32,
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,  # not needed when using peft
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        peft_config=LoraConfig(target_modules="all-linear"),
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

Let's run and wait... 🚀

```sh
accelerate launch dpo_idefics2-8b.py
```

### Results

A few hours later, the training is complete. Let's have a look at the training curves:

![Learning curves](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/learning_curves.png)

In DPO, we focus on some metrics to determine if the training went well.

- **The accuracy**: the percentage of training samples for which the model has a higher chance of outputting the chosen answer than the rejected answer. We can see that the accuracy is increasing, which is a good sign.
- **The rewards**: the reward are somewhat related to the probability of an answer being chosen. We refer the reader to the TODO for more details. We expect the reward of the chosen answer to be higher than the rejected answer. To check this, we can watch the _reward margin_ which is the difference between the rewards of the chosen answer and the rejected answer. We can see that the reward margin is increasing, which is a also good sign.

### Evaluation

Now that the model is trained, we can get a sense of how well it performs by evaluating it on some examples. We can use the following script to evaluate the model:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b").to("cuda")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
model.load_adapter("HuggingFaceH4/idefics2-8b-dpo-rlaif-v-v0.3")  # <-- Load the adapter we've just trained

# Process
user_message = "How many families?"
image_path = "image.jpg"
data = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_message}]}]
prompts = processor.apply_chat_template(data, add_generation_prompt=True)
images = [Image.open(image_path)]
inputs = processor(prompts, images, return_tensors="pt", padding=True)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response_text)  # 18,000,000 families
```

### Does it still hallucinate?

As mentionned earlier, the dataset used is meant to reduce hallucination. Have we succeeded in this task? To check this, we can use the AMBER benchmark which is a dataset specifically designed to evaluate hallucination in VLMs. We report also results for other models for comparison.

|                  | Accuracy | F1   |
| ---------------- | -------- | ---- |
| VCD              | 71.8     | 74.9 |
| Less-is-more     | 72.4     | 75.8 |
| OPERA            | 75.2     | 78.3 |
| LURE             | 73.5     | 77.7 |
| QWEN-VL          | 81.9     | 86.4 |
| LLaVA-NeXT       | 81.4     | 85.4 |
| MiniGemini       | 82.6     | 87.6 |
| GPT-4o           | 84.5     | 85.3 |
| Idefics2         | 85.8     | 89.1 |
| **Idefics2+DPO** | 85.9     | 89.4 |

Overall, the model seems to hallucinate a bit less. The training seems to have been successful!

As an illustration, here are some examples where the model hallucinates less:

| Image | Question | Idefics2 | Idefics2+DPO |
| ----- | -------- | -------- | ------------ |
| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_2.jpg) | Are there two ships in this image? | Yes | No |
| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_7.jpg) | Is the ground uneven in this image? | No | Yes |
| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_111.jpg) | Is there one shovel in this image? | Yes | No |

### Finetuning Llava 1.5, PaliGemma and others

At the time of writing, the DPO implementation in TRL supports Idefics2, Llava 1.5 and PaliGemma, and we're working on adding support for more models. To finetune these models the easier is to use the [example script](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_visual.py) provided in the TRL repository. For example, to finetune PaliGemma, you can use the following command:

```sh
accelerate launch examples/scripts/dpo_visual.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path google/paligemma-3b-pt-224 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_paligemma_rlaif-v \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules=all-linear
```

You can also refer to the [smol-vision](https://github.com/merveenoyan/smol-vision) project where you'll find more examples and scripts to finetune vision models with DPO.
