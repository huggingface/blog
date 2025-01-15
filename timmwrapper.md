---
title: "Timm is Now a First-Class Citizen in the 🤗 Transformers Library"
thumbnail: /blog/assets/timmwrapper/thumbnail.png
authors:
- user: ariG23498
---

# Timm is Now a First-Class Citizen in the 🤗 Transformers Library

Ever wanted to leverage the vast collection of state-of-the-art models from the `timm` library using
the familiar 🤗 `transformers` ecosystem? Enter the `TimmWrapper`—a simple, yet powerful tool that unlocks this potential.

In this post, we’ll cover:
- How the `TimmWrapper` works and why it’s a game-changer.
- How to integrate `timm` models with 🤗 `transformers`.
- Practical examples: pipelines, quantization, fine-tuning, and more.

> [!NOTE]
> To follow along with this blog post, install the latest version of `transformers` by running:
> ```bash
> pip install -Uq transformers
> ```

> [!NOTE]  
> **Check out the full repository for all code examples and notebooks:**  
> 🔗 [TimmWrapper Examples](https://github.com/your-username/timmwrapper-examples)  


## What is `timm`?

The PyTorch Image Models (`timm`) library offers a rich collection of state-of-the-art computer vision models,
along with useful layers, utilities, optimizers, and data augmentations. It’s a go-to resource for
image classification, object detection, segmentation, and more.

With pre-trained models covering a wide range of architectures, `timm` simplifies the workflow for
computer vision practitioners.

## Why Use the `TimmWrapper`?

While 🤗 `transformers` supports several vision models, `timm` offers an even broader collection,
including many mobile-friendly and efficient models not natively supported.

The `TimmWrapper` bridges this gap, bringing the best of both worlds:
- ✅ **Pipeline API Support**: Easily plug any `timm` model into the `transformers` pipeline for streamlined inference.
- 🧩 **Compatibility with Auto Classes**: While `timm` models aren’t natively compatible with `transformers` Auto classes, the `TimmWrapper` makes them work seamlessly.
- ⚡ **Quick Quantization**: With just ~5 lines of code, you can quantize `timm` models for efficient inference.
- 🎯 **Fine-Tuning with Trainer API**: Fine-tune `timm` models using the `Trainer` API and even integrate with adapters like LoRA.
- 🚀 **Torch Compile for Speed**: Leverage `torch.compile` to optimize inference time.


## Pipeline API: Using timm Models for Image Classification

One of the standout features of the `TimmWrapper` is that it allows you to leverage the 🤗 **`pipeline` API**
to perform image classification with any `timm` model. The **`pipeline` API** abstracts away a lot of
complexity, making it easy to load a pre-trained model, perform inference, and view results with minimal lines of code.

Let’s see how to use the `TimmWrapper` with the **MobileNetV4** model from `timm`.

```python
from transformers import pipeline
import requests

# Load the image classification pipeline with a timm model
image_classifier = pipeline(model="timm/mobilenetv4_conv_medium.e500_r256_in1k")

# URL of the image to classify
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"

# Perform inference
outputs = image_classifier(url)

# Print the results
for output in outputs:
    print(f"Label: {output['label'] :20} Score: {output['score'] :0.2f}")
```

**Outputs**:

```bash
Device set to use cpu
Label: tabby, tabby cat     Score: 0.69
Label: tiger cat            Score: 0.21
Label: Egyptian cat         Score: 0.02
Label: bee                  Score: 0.00
Label: marmoset             Score: 0.00
```

## Gradio Integration: Building a Food Classifier Demo 🍣  

Want to quickly create an interactive web app for image classification? **Gradio** makes it simple to build a user-friendly interface with minimal code. Let's combine **Gradio** with the `pipeline` API to classify food images using a fine-tuned timm ViT model (we will cover fine tuning in a later section).

Here’s how you can set up a quick demo:

```python
import gradio as gr
from transformers import pipeline

# Load the image classification pipeline
pipe = pipeline(
    "image-classification",
    model="ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.ft_food101"
)

# Define a simple function for Gradio
def classify(image):
    return pipe(image)[0]["label"]

# Build the Gradio interface
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=[["./sushi.png", "sushi"]]
)

# Launch the app
demo.launch()
```

Here’s a live example hosted on Hugging Face Spaces. You can test it directly in your browser!  

<iframe  
  src="https://huggingface.co/spaces/ariG23498/food-classification"  
  frameborder="0"  
  width="100%"  
  height="560px"  
></iframe>  


## Auto Classes: Simplifying Model Loading 🧩  

The 🤗 `transformers` library provides **Auto Classes** to abstract away the complexity of loading models and processors. With the **`TimmWrapper`**, you can use **`AutoModelForImageClassification`** and **`AutoImageProcessor`** to load any `timm` model effortlessly.

Here’s a quick example:

```python
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from transformers.image_utils import load_image

# Load an image
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"
image = load_image(image_url)

# Use Auto classes to load a timm model
checkpoint = "timm/mobilenetv4_conv_medium.e500_r256_in1k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

# Check the types
print(type(image_processor))  # TimmWrapperImageProcessor
print(type(model))            # TimmWrapperForImageClassification
```

## Quantizing timm Models with TimmWrapper ⚡️

Quantization is a powerful technique to **reduce model size and speed up inference**, especially for deployment on resource-constrained devices. With the **`TimmWrapper`** in 🤗 `transformers`, you can quantize any `timm` model to **8-bit precision** with just a few lines of code using **`BitsAndBytesConfig`** from `bitsandbytes`.

Here’s how simple it is to quantize a model using `TimmWrapper`:

```python
from transformers import TimmWrapperForImageClassification, BitsAndBytesConfig

# Load the model
checkpoint = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"

# Define 8-bit quantization configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the quantized model
model_8bit = TimmWrapperForImageClassification.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
).eval()
```

```python
original_footprint = model.get_memory_footprint()
quantized_footprint = model_8bit.get_memory_footprint()

print(f"Original model size: {original_footprint / 1e6:.2f} MB")
print(f"Quantized model size: {quantized_footprint / 1e6:.2f} MB")
print(f"Reduction: {(original_footprint - quantized_footprint) / original_footprint * 100:.2f}%")
```

**Output:**  
```
Original model size: 346.27 MB  
Quantized model size: 88.20 MB  
Reduction: 74.53%  
``` 

Quantized models perform **almost identically** to full-precision models during inference:

| **Model**         | **Label**                | **Accuracy** |
|-------------------|-------------------------|-----------|
| Original Model    | Remote Control, Remote   | 0.35%     |
| Quantized Model   | Remote Control, Remote   | 0.33%     |

## Supervised Fine-Tuning with `TimmWrapper` 🏋️‍♂️

Fine-tuning a `timm` model with the **`Trainer` API** from 🤗 `transformers` is **straightforward and highly flexible**. You can fine-tune your model on custom datasets using the `Trainer` class, which handles the training loop, logging, and evaluation. Additionally, you can enhance your fine-tuning process using **LoRA (Low-Rank Adaptation)** to train efficiently with fewer parameters.

This section gives a **quick overview** of both standard fine-tuning and LoRA fine-tuning, with links to the complete code.


### Standard Fine-Tuning with `Trainer` API 

The `Trainer` API makes it easy to set up training with minimal code. For an end to end example for fine tuning, one can visit [TimmWrapper Examples](https://github.com/your-username/timmwrapper-examples).

Here's an outline of what a fine-tuning setup looks like:

```python
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    load_best_model_at_end=True,
    push_to_hub=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
```

💡 **Model Example:**  
Fine-tuned ViT on **Food-101**: [**`vit_base_patch16_224.augreg2_in21k_ft_in1k.ft_food101`**](https://huggingface.co/ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.ft_food101)


### LoRA Fine-Tuning for Efficient Training 💡 

LoRA (Low-Rank Adaptation) allows you to **fine-tune large models efficiently** by training only a few additional parameters. You can fine-tune a `timm` model using LoRA with the **PEFT** library. For an end to end example for LoRA fine tuning, one can visit [TimmWrapper Examples](https://github.com/your-username/timmwrapper-examples).

Here’s how you can set it up:

```python
from peft import LoraConfig, get_peft_model

# Load the model
model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)

# Define the LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["qkv"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["head"],
)

# Wrap the model with PEFT
lora_model = get_peft_model(model, lora_config)

# Print trainable parameters
lora_model.print_trainable_parameters()
```

🧪 **Trainable Parameters with LoRA:**  
```bash
trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.77%
```

💾 **Model Example:**  
LoRA Fine-Tuned ViT on **Food-101**: [**`vit_base_patch16_224.augreg2_in21k_ft_in1k.lora_ft_food101`**](https://huggingface.co/ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.lora_ft_food101)

### Inference with LoRA Fine-Tuned Model 🎯  

Once the model is fine-tuned, you can use it for inference as shown below:

```python
from transformers import AutoImageProcessor
from peft import PeftModel

# Load the LoRA fine-tuned model
inference_model = PeftModel.from_pretrained(model, "ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.lora_ft_food101")

# Process the input image
image_processor = AutoImageProcessor.from_pretrained("ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k")
inputs = image_processor(image, return_tensors="pt")

# Make predictions
with torch.no_grad():
    logits = inference_model(**inputs).logits
    prediction = logits.argmax(-1).item()

# Display the result
plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {model.config.id2label[prediction]}")
plt.show()
```

![image of sushi with prediction from a fine tuned model](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/timmwrapper/prediction-food.png)

## Torch Compile: Instant Speedup 🚀  

With **`torch.compile`** in PyTorch 2.0, you can achieve **faster inference** by compiling your model with just one line of code. Here's a quick benchmark to compare inference time with and without `torch.compile` using the `TimmWrapper`.

```python
# Load the model and input
model = TimmWrapperForImageClassification.from_pretrained(checkpoint).to(device)
processed_input = image_processor(image, return_tensors="pt").to(device)

# Benchmark function
def run_benchmark(model, input_data, runs=300):
    model.eval()
    with torch.no_grad():
        times = [time.time() - time.time() for _ in range(runs)]
        for i in range(runs):
            start = time.time()
            _ = model(**input_data)
            times[i] = time.time() - start
    return sum(times) / runs

# Run benchmarks
time_no_compile = run_benchmark(model, processed_input)
compiled_model = torch.compile(model).to(device)
time_compile = run_benchmark(compiled_model, processed_input)

# Results
print(f"Without torch.compile: {time_no_compile:.4f} s")
print(f"With torch.compile: {time_compile:.4f} s")
```

![compile timing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/timmwrapper/compile.png)

## Wrapping Up

The TimmWrapper in 🤗 transformers library opens new doors for leveraging state-of-the-art vision models with minimal effort. Whether you're looking to fine-tune, quantize, or simply run inference, this integration provides a unified API to streamline your workflow.

By combining timm's model repository with the extensive ecosystem of transformers, you get the best of both worlds. Start exploring today and unlock new possibilities in computer vision!