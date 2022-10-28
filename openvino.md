---
title: "Shrink and accelerate your models with Optimum and Intel OpenVINO"
thumbnail: /blog/assets/113_openvino/thumbnail.png
---

<h1>Shrink and accelerate your models with Optimum and Intel OpenVINO</h1>

<div class="blog-metadata">
    <small>Published July 20, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/openvino.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/echarlaix">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1615915889033-6050eb5aeb94f56898c08e57.jpeg?w=200&h=200&f=face" title="Ella Charlaix">
        <div class="bfc">
            <code>echarlaix</code>
            <span class="fullname">Ella Charlaix</span>
        </div>
    </a>
    <a href="https://twitter.com/julsimon">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1633343465505-noauth.jpeg?w=128&h=128&f=face" title="Julien Simon">
        <div class="bfc">
            <code>juliensimon</code>
            <span class=fullname">Julien Simon</span>
        </div>
    </a>
</div>

![image](assets/113_openvino/thumbnail.png)

Last July, we [announced](https://huggingface.co/blog/intel) that Intel and Hugging Face would collaborate on building state-of-the-art yet simple hardware acceleration tools for Transformer models.
​
Today, we are very happy to announce that we added Intel [OpenVINO](https://docs.openvino.ai/latest/index.html) to Optimum Intel. You can now easily perform inference with OpenVINO Runtime on a variety of Intel processors  ([see](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) the full list of the supported device) using Transformers models which can be hosted either on the Hugging Face hub or locally.  You can also easily quantize your models with the OpenVINO Neural Network Compression Framework ([NNCF](https://github.com/openvinotoolkit/nncf)), and shrink their size and prediction latency in minutes. ​

This first release is based on OpenVINO 2022.2.0 and enables inference for a large quantity of PyTorch models using our [OVModels](https://huggingface.co/docs/optimum/intel/inference).

Post-training static quantization and quantization aware training can be applied on many encoder models (BERT, DistilBERT, etc.). More encoder models will be supported after OpenVINO's next release. Currently the quantization of Encoder Decoder models is not enabled. This restriction should be lifted with our integration of the next OpenVINO release.

​Let us show you how to get started in minutes!​

## Quantizing a Vision Transformer with Optimum Intel and OpenVINO
​
In this example, we will run post-training static quantization on a Vision Transformer (ViT) [model](https://huggingface.co/juliensimon/autotrain-food101-1471154050) fine-tuned for image classification on the [food101](https://huggingface.co/datasets/food101) dataset.
​
Quantization is a process that shrinks memory and compute requirements by reducing the bit width of model parameters. Reducing the number of bits means that the resulting model requires less memory at inference time, and that operations like matrix multiplication can be performed faster thanks to integer arithmetic.

First, let's create a virtual environment and install all dependencies.​

```bash
virtualenv openvino
source openvino/bin/activate
pip install pip --upgrade
pip install optimum[openvino,nncf] torchvision evaluate
```

Next, moving to a Python environment, we import the appropriate modules and download the original model as well as its feature extractor.
​
```python
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
​
model_id = "juliensimon/autotrain-food101-1471154050"
model = AutoModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
```
​
Post-training static quantization requires a calibration step where data is fed through the network in order to compute the quantized activation parameters. Here, we take 300 samples from the original dataset to build the calibration dataset.
​
```python
from optimum.intel.openvino import OVQuantizer
​
quantizer = OVQuantizer.from_pretrained(model)
calibration_dataset = quantizer.get_calibration_dataset(
    "food101",
    num_samples=300,
    dataset_split="train",
)
```

As usual with image datasets, we need to apply the same transforms that were used at training time. We use the preprocessing defined in the feature extractor. We also define a data collation function to feed the model batches of properly formatted tensors.
​

```python
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
​
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
​
calibration_dataset.set_transform(val_transforms)
​
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```


For our first try, we use the default configuration for quantization. You can also specify the number of samples to use during the calibration step, which is by default 300.

```python
from optimum.intel.openvino import OVConfig
​
quantization_config = OVConfig()
quantization_config.compression["initializer"]["range"]["num_init_samples"] = 300
```


We're now ready to quantize the model. The `OVQuantizer.quantize()` method quantizes the model and exports it to the OpenVINO Intermediate Representation (IR) format. The resulting graph is represented with two files: an XML file describing the network topology and a binary file describing the weights. The resulting model can run on any target Intel device.
​

```python
save_dir = "quantized_model"

# Apply static quantization and export the resulting quantized model to OpenVINO IR format
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    data_collator=collate_fn,
    remove_unused_columns=False,
    save_directory=save_dir,
)
feature_extractor.save_pretrained(save_dir)
```

A minute or two later, the model has been quantized. We can then easily load it with our [`OVModelForXxx`](https://huggingface.co/docs/optimum/intel_inference#optimum-inference-with-openvino) classes, the equivalent of the Transformers [`AutoModelForXxx`](https://huggingface.co/docs/transformers/main/en/autoclass_tutorial#automodel) classes found in the `transformers` library. Likewise, we can create [pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) and run inference with [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html). An important thing to mention is that the model is compiled just before the first inference, which will inflate the latency of the first inference.
​
```python
from transformers import pipeline
from optimum.intel.openvino import OVModelForImageClassification
​
ov_model = OVModelForImageClassification.from_pretrained(save_dir)
ov_pipe = pipeline("image-classification", model=ov_model, feature_extractor=feature_extractor)
outputs = ov_pipe("http://farm2.staticflickr.com/1375/1394861946_171ea43524_z.jpg")
print(outputs)
```

​To verify that quantization did not have a negative impact on accuracy, we applied an evaluation step to compare the accuracy of the original model with its quantized counterpart.
We evaluate both models on a subset of the dataset (taking only 20% of the evaluation dataset). We observed no loss in accuracy with both models having an accuracy of **87.6**.

```python
from datasets import load_dataset
from evaluate import evaluator

# We run the evaluation step on 20% of the evaluation dataset
eval_dataset = load_dataset("food101", split="validation").select(range(5050))
eval = evaluator("image-classification")

ov_eval_results = eval.compute(
    model_or_pipeline=ov_pipe,
    data=eval_dataset,
    metric="accuracy",
    label_mapping=ov_pipe.model.config.label2id,
)

trfs_pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
trfs_eval_results = eval.compute(
    model_or_pipeline=trfs_pipe,
    data=eval_dataset,
    metric="accuracy",
    label_mapping=trfs_pipe.model.config.label2id,
)
print(trfs_eval_results, ov_eval_results)
```

Looking at the quantized model, we see that its memory size decreased by **4x** from 344MB to 90MB. Running a quick benchmark (on a m6i.4xlarge EC2 instance) on 5050 image predictions, we also see a speedup in latency of **2x**, from 93ms to 48ms per sample. That's not bad for a few lines of code!

You can find the resulting model hosted on the Hugging Face hub. To load it, you can easily do as follows:
```python
from optimum.intel.openvino import OVModelForImageClassification
​
ov_model = OVModelForImageClassification.from_pretrained("echarlaix/vit-food101-int8")
```

## Now it's your turn
​
As you can see, it's pretty easy to shrink and accelerate your models with Optimum Intel and OpenVINO. If you'd like to get started, please visit the [Optimum Intel](https://github.com/huggingface/optimum-intel) repository, and don't forget to give it a star :star:. You'll also find additional examples [there](https://huggingface.co/docs/optimum/intel/optimization_ov). If you'd like to dive deeper into OpenVINO, the Intel [documentation](https://docs.openvino.ai/latest/index.html) has you covered.
​
Give it a try and let us know what you think. We'd love to hear your feedback on the Hugging Face [forum](https://discuss.huggingface.co/c/optimum), and please feel free to request features or file issues on [Github](https://github.com/huggingface/optimum-intel).
​
Have fun with 🤗 Optimum Intel, and thank you for reading.
​

## Appendix

OpenVINO Runtime can be used to perform inference on the corresponding devices :

| Device type      | Supported devices                                                           |
| :--------------- | :-------------------------------------------------------------------------- |
| CPU              |6th to 13th generation Intel Core processors                                 |
| CPU              |Intel Xeon Scalable processors                                               |
| CPU              |Intel Pentium processor N4200/5, N3350/5, N3450/5 with Intel HD Graphics     |
| CPU              |Intel Atom processors with Intel Streaming SIMD Extensions 4.2 (Intel SSE4.2)|
| GPU              |Intel Processor Graphics, including Intel HD Graphics and Intel Iris Graphics|
