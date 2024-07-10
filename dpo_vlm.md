---
title: 'Preference Optimization for Vision Language Models'
thumbnail: /blog/assets/dpo_vlm/thumbnail.png
authors:
- user: qgallouedec
- user: vwxyzjn
- user: merve
- user: kashif
---

# Preference Optimization for Vision Language Models with TRL

Training models to understand and predict human preferences can be incredibly complex. Traditional methods, like supervised fine-tuning, often require assigning specific labels to data, which is not cost-efficient, especially for nuanced tasks. Preference optimization is an alternative approach that can simplify this process and yield more accurate results. By focusing on comparing and ranking candidate answers rather than assigning fixed labels, preference optimization allows models to capture the subtleties of human judgment more effectively.

Preference optimization is widely used for fine-tuning language models, but it can also be applied to vision language models (VLM).
We are excited to announce that the **[TRL](https://huggingface.co/docs/trl/index) library now supports direct preference optimization (DPO) for VLMs**. This article will guide you through the process of training VLMs using TRL and DPO.

## Preference dataset

Preference optimization requires data that captures user preferences. In the binary choice setting, each example consists of a prompt, and two candidate answers: one that is chosen and one that is rejected. The model's goal is to learn to predict the chosen answer over the rejected one.
For example, you need to have samples like the following:

<figure class="image table text-center m-0 w-full">
  <img  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/how-many-families.jpg"></img>
  <figcaption>Image from <a href="https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset">openbmb/RLAIF-V-Dataset</a></figcaption>
</figure>

**❔ Question**: _How many families?_

- **❌ Rejected:** _The image does not provide any information about families._
- **✅ Chosen:** _The image shows a Union Organization table setup with 18,000 families._

Note that the chosen message is not necessarily correct. For example, the chosen response that says 18,000 families is still wrong, but it's less wrong compared to the rejected response.

For this blog post, we'll be using the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), which includes over 83,000 annotated rows. Let's take a closer look at the dataset:

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

Our model requires both text and images as input, so the first step is to format the dataset to fit this requirement. The data should be structured to simulate a conversation between a user and an assistant. The user provides a prompt that includes an image and a question, while the assistant responds with an answer. Here's how this formatting is done:

```python
from datasets import features
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

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

# Apply the formatting function to the dataset,
# remove columns to end up with only "images", "prompt", "chosen", "rejected" columns
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

For the sake of the example, we'll be training the [Idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) model, but note that the DPO implementation in TRL supports other models like [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224). More information in Section [Finetuning Llava 1.5, PaliGemma and others](#finetuning-llava-15-paligemma-and-others). Before looking into the training process, we'll first ensure everything fits smoothly into memory.

### How much memory do I need?

I have a GPU with 80GB of VRAM. Is it enough to train my Idefics2-8b model? Here are the calculation steps to get a rough estimate of the memory needed.

Let \\( N \\) be the number of parameters, \\( P \\) the precision. The following components will have to fit together in memory:

- **Model to train**: \\( N \times P \\)
- **Reference model**: the reference model is the same as the model to train, so it also requires \\( N \times P \\)
- **Gradients**: we train the whole model, and each parameter requires a gradient, so it requires \\( N \times P \\)
- **Optimizer states**: we use [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), which requires two states per parameter, so it requires \\( 2 \times N \times P \\)

Idefics2-8b has 8 billion parameters, and we use `float32` precision which requires 4 bytes per float. So the total memory required is:

| Component        | Calculation                           | Memory     |
| ---------------- | ------------------------------------- | ---------- |
| Model to train   | \\( 8 \times 10^9 \times 4 \\)          | 32 GB      |
| Reference model  | \\( 8 \times 10^9 \times 4 \\)          | 32 GB      |
| Gradients        | \\( 8 \times 10^9 \times 4 \\)          | 32 GB      |
| Optimizer states | \\( 2 \times 8 \times 10^9 \times 4 \\) | 64 GB      |
| **Total**        |                                       | **160 GB** |

This is way above my GPU's memory capacity. Fortunately, by applying techniques such as quantization and LoRA, we can significantly reduce the memory requirements and make the training feasible. Let's see how to do this.

### Quantization

Quantization is a technique that reduces the precision of the model's weights and activations. Switching from `float32` to `bfloat16` precision halves the storage requirement per parameter from 4 bytes to 2 bytes. This optimization conserves memory while also accelerating computations, ensuring high performance with minimal compromise.
To implement `bfloat16` precision in the model:

```python
import torch
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16)
```

`bfloat16` precision can also be applied to the optimizer by setting `bf16=True` in the training arguments:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(..., bf16=True)
```

### LoRA

[LoRA](https://arxiv.org/abs/2106.09685) is a method that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while keeping the original weights frozen. This significantly decreases the storage needs for LLM adapted to specific tasks. LoRA is integrated in [PEFT](https://github.com/huggingface/peft) and you can set it up in no time:

```diff
  from transformers import AutoModelForVision2Seq
+ from peft import get_peft_model, LoraConfig

  model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b")
+ peft_config = LoraConfig(target_modules="all-linear")
+ model = get_peft_model(model, peft_config)
```

PEFT acts like a wrapper (called _adaptater_) around the model. This is the adapter that will be trained while the inner model is kept frozen. How much does LoRA reduce the number of trainable parameters?

```python
>>> model.print_trainable_parameters()
trainable params: 55,348,736 || all params: 8,458,116,848 || trainable%: 0.6543860411799315
```

It reduces the number of trainable parameters from 8 billion to 55 million, which is a huge gap, and it will significantly reduce the memory requirements.

### The new memory requirements after quantization and LoRA

Now that we have reduced the memory requirements, let's recalculate the memory needed:

| Component        | Calculation                           | Memory      |
| ---------------- | ------------------------------------- | ----------- |
| Model to train   | \\( 8 \mathrm{G} \times 2 \\)           | 16  GB      |
| Reference model  | \\( 8 \mathrm{G} \times 2 \\)           | 16  GB      |
| Gradients        | \\( 55 \mathrm{M} \times 2 \\)          | 0.1 GB      |
| Optimizer states | \\( 2 \times 55 \mathrm{M} \times 2 \\) | 0.2 GB      |
| **Total**        |                                       | **32.3 GB** |

This time, we need around 32GB of memory to finetune our Idefics2-8b model, which is much more reasonable and fits within my GPU!

For additional information on optimizing memory usage using LoRA and QLoRA, refer to the [PEFT documentation](https://huggingface.co/docs/peft/en/index) or [LoRA and QLoRA Google's recommendations for LLMs](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/lora-qlora).

### What about the batch size?

Our memory calculation isn't exact as it doesn't account for activations. Activations are the intermediate outputs of the network layers and their memory requirements depend on the model structure and batch size. Precisely calculating the memory needed for activations is challenging, so we'll rely on empirical observations.

To choose an appropriate training batch size (`per_device_train_batch_size`), start with your desired batch size (e.g., 64). This will likely result in an out-of-memory (OOM) error. If it does, reduce the batch size by half and double the gradient accumulation steps (`gradient_accumulation_steps`) to maintain the same effective batch size. Repeat this process until the memory fits within your GPU. In our case, we end up with a batch size of 2 and gradient accumulation steps of 32.

An additional optimization is to use gradient checkpointing (`gradient_checkpointing`) to reduce the memory needed for activations. This technique trades off compute for memory by recomputing parts of the network during the backward pass. It can be enabled by setting `gradient_checkpointing=True` in the training arguments.

### Summary: complete training script

Now that we've set up the model, dataset, and training parameters, we're ready to train. Here's how to put everything together in a script, including some additional elements to speed up processing, like `dataset_num_proc` and `dataloader_num_workers`:

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
        dataset_num_proc=32,  # tokenization will use 32 processes
        dataloader_num_workers=32,  # data loading will use 32 workers
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

## Results

A few hours later, the training is complete. Let's take a look at the training curves:

![Learning curves](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/learning_curves.png)

In DPO, we focus on several metrics to assess the quality of the training:

- **Accuracy**: This metric indicates the percentage of training samples where the model is more likely to output the chosen answer rather than the rejected answer. We can see an increase in accuracy, which is a positive sign.
- **Rewards**: Rewards are related to the probability of an answer being chosen. For more details, refer to [DPO paper, Section 5](https://arxiv.org/abs/2305.18290). We expect the reward for the chosen answer to be higher than for the rejected answer. To verify this, we look at the _reward margin_, which is the difference between the rewards for the chosen and rejected answers. An increasing reward margin, as observed here, is also a good sign.

## Evaluation

### Inference

With the model training complete, the next step is to evaluate its performance on some examples. This will give us a sense of how well the model has learned and how effectively it can make predictions. Here’s a script to help you evaluate the model and analyze its performance on a set of test examples:

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b").to("cuda")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
model.load_adapter("HuggingFaceH4/idefics2-8b-dpo-rlaif-v-v0.3")  # <-- Load the adapter we've just trained

# Process
user_message = ...
image_path = ...
data = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_message}]}]
prompts = processor.apply_chat_template(data, add_generation_prompt=True)  # add_generation_prompt=True to end the prompt with "ASSISTANT:"
images = [Image.open(image_path)]
inputs = processor(prompts, images, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response_text)
```

As mentioned above, the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) is designed to reduce hallucinations. But has the fine-tuning actually reduced hallucinations? To find out, we can use the [AMBER benchmark](https://arxiv.org/abs/2311.07397), a dataset specifically created to evaluate hallucinations in VLMs. We will report the results for Idefics2 and Idefics2+DPO on the discriminative task and compare them with other models for reference.

|                  | Accuracy | F1       |
| ---------------- | -------- | -------- |
| GPT-4o           | 88.8     | 91.6     |
| **Idefics2+DPO** | **85.9** | **89.4** |
| Idefics2         | 85.8     | 89.1     |
| GPT-4v           | 83.4     | 87.4     |
| MiniGemini       | 82.6     | 87.6     |
| LLaVA-NeXT       | 81.4     | 85.4     |
| QWEN-VL          | 81.9     | 86.4     |
| LURE             | 73.5     | 77.7     |
| OPERA            | 75.2     | 78.3     |
| Less-is-more     | 72.4     | 75.8     |
| VCD              | 71.8     | 74.9     |

Overall, the fine-tuned model seems to hallucinate a bit less. The training seems to have been successful!

Here are some cherry-picked examples to illustrate the model's performance:

| Image                                                                                                                  | Question                            | Idefics2 | Idefics2+DPO |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | -------- | ------------ |
| ![AMBER_2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_2.jpg)     | Are there two ships in this image?  | Yes      | No           |
| ![AMBER_111](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_111.jpg) | Is the ground uneven in this image? | No       | Yes          |
| ![AMBER_7](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo_vlm/AMBER_7.jpg)     | Is there one shovel in this image?  | Yes      | No           |

Try it yourself and see how the model performs on your own examples!

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"></script>

<gradio-app theme_mode="light" space="HuggingFaceH4/compare_idefics-8b-dpo"></gradio-app>

## Finetuning Llava 1.5, PaliGemma and others

At the time of writing, the DPO implementation in TRL supports Idefics2, Llava 1.5, and PaliGemma, with ongoing efforts to add support for more models. The easiest way to fine-tune these models is to use the [example script](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_visual.py) provided in the TRL repository. For example, to finetune PaliGemma, you can use the following command:

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

You can find a detailed focus on PaliGemma finetuning in the [smol-vision](https://github.com/merveenoyan/smol-vision) project.

🚀🚀 Now you have everything you need to start fine-tuning your own VLMs with DPO. Share your findings, models, and datasets with the community!
