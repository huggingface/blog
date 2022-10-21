---
title: "OpenVINO quantization and inference with Optimum Intel"
# thumbnail: 
---

<h1>OpenVINO quantization and inference with Optimum Intel</h1>


## Case study: Quantizing a ViT with Optimum Intel OpenVINO support

In this example, we will run post-training static quantization on a ViT model fine-tuned for classification. Quantization is a process that shrinks memory and compute requirements by reducing the bit width of model parameters. Reducing the number of bits means that the resulting model requires less memory storage and that operations like matrix multiplication can be performed much faster with integer arithmetic.

First, to install all the required libraries :

```
pip install optimum-intel[openvino,nncf] # Should be replaced to pip install optimum[openvino,nncf] after optimum release
```


```python
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from optimum.intel.openvino import OVConfig, OVQuantizer

model_id = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# Load the default quantization configuration detailing the quantization we wish to apply
quantization_config = OVConfig()
# Instantiate our OVQuantizer using the desired configuration
quantizer = OVQuantizer.from_pretrained(model)
```

Post-training static quantization introduces an additional calibration step where data is fed through the network in order to compute the activations quantization parameters. In order to create the calibration dataset, we can use the quantizer `get_calibration_dataset()` method. The dataset chosen is the training split of the Food-101 dataset, which consists of 101 food categories, and the number of calibration amples is set to `300`.

```python
# Create the calibration dataset used to perform static quantization
calibration_dataset = quantizer.get_calibration_dataset(
    "food101",
    num_samples=300,
    dataset_split="train",
)
```

Then we define the torchvision transforms to be applied to each image.


```python
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_val_transforms = Compose(
    [
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ]
)
def val_transforms(example_batch):
    example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
    return example_batch

calibration_dataset.set_transform(val_transforms)
```

We can now use the quantizer `quantize()` method which will apply quantization and export the resulting quantized model to the OpenVINO IR format.

```python
save_dir = "./quantized_model"

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
  
# Apply static quantization and export the resulting quantized model to OpenVINO IR format
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    data_collator=collate_fn,
    remove_unused_columns=False,
    save_directory=save_dir,
)
# Save the tokenizer
feature_extractor.save_pretrained(save_dir)
```

After applying quantization on our model, we can then easily load it with our `OVModelForXxx` classes, and create pipelines to run inference with OpenVINO Runtime.

```python
from transformers import pipeline
from optimum.intel.openvino import OVModelForImageClassification

ov_model = OVModelForImageClassification.from_pretrained(save_dir)
pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
outputs = pipe(url)
```



OpenVINO Runtime can be used to perform inference on the corresponding devices :

| Device type      | Supported devices                                                           |
| :--------------- | :-------------------------------------------------------------------------- |
| CPU              |6th to 13th generation Intel Core processors                                 |
| CPU              |Intel Xeon Scalable processors                                               |
| CPU              |Intel Pentium processor N4200/5, N3350/5, N3450/5 with Intel HD Graphics     |
| CPU              |Intel Atom processors with Intel Streaming SIMD Extensions 4.2 (Intel SSE4.2)|
| GPU              |Intel Processor Graphics, including Intel HD Graphics and Intel Iris Graphics|
