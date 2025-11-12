---
title: Fine-Tune a Semantic Segmentation Model with a Custom Dataset
thumbnail: /blog/assets/56_fine_tune_segformer/thumb.png
authors:
- user: tobiasc
  guest: true
- user: nielsr
---

# Fine-Tune a Semantic Segmentation Model with a Custom Dataset


<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/56_fine_tune_segformer.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**This guide shows how you can fine-tune Segformer, a state-of-the-art semantic segmentation model. Our goal is to build a model for a pizza delivery robot, so it can see where to drive and recognize obstacles 🍕🤖. We'll first use an available segmentation dataset from the 🤗 hub. Then we'll fine-tune a pre-trained SegFormer model by using [`🤗 transformers`](https://huggingface.co/transformers), an open-source library that offers easy-to-use implementations of state-of-the-art models. Along the way, you'll learn how to work with the Hugging Face Hub, the largest open-source catalog of models and datasets.**

Semantic segmentation is the task of classifying each pixel in an image. You can see it as a more precise way of classifying an image. It has a wide range of use cases in fields such as medical imaging and autonomous driving. For example, for our pizza delivery robot, it is important to know exactly where the sidewalk is in an image, not just whether there is a sidewalk or not.

Because semantic segmentation is a type of classification, the network architectures used for image classification and semantic segmentation are very similar. In 2014, [a seminal paper](https://arxiv.org/abs/1411.4038) by Long et al. used convolutional neural networks for semantic segmentation. More recently, Transformers have been used for image classification (e.g. [ViT](https://huggingface.co/blog/fine-tune-vit)), and now they're also being used for semantic segmentation, pushing the state-of-the-art further.

[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) is a model for semantic segmentation introduced by Xie et al. in 2021. It has a hierarchical Transformer encoder that doesn't use positional encodings (in contrast to ViT) and a simple multi-layer perceptron decoder. SegFormer achieves state-of-the-art performance on multiple common datasets. Let's see how our pizza delivery robot performs for sidewalk images.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Pizza delivery robot segmenting a scene" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/pizza-scene.png"></medium-zoom>
</figure>

Let's get started by installing the necessary dependencies. Because we're going to push our dataset and model to the Hugging Face Hub, we need to install [Git LFS](https://git-lfs.github.com/) and log in to Hugging Face.

The installation of `git-lfs` might be different on your system. Note that Google Colab has Git LFS pre-installed.

```bash
pip install -q transformers datasets evaluate segments-ai
apt-get install git-lfs
git lfs install
huggingface-cli login
```

## 1. Create/choose a dataset

The first step in any ML project is assembling a good dataset. In order to train a semantic segmentation model, we need a dataset with semantic segmentation labels. We can either use an existing dataset from the Hugging Face Hub, such as [ADE20k](https://huggingface.co/datasets/scene_parse_150), or create our own dataset by annotating images with corresponding segmentation maps.

For our pizza delivery robot, we could use an existing autonomous driving dataset such as [CityScapes](https://www.cityscapes-dataset.com/) or [BDD100K](https://bdd100k.com/). However, these datasets were captured by cars driving on the road. Since our delivery robot will be driving on the sidewalk, there will be a mismatch between the images in these datasets and the data our robot will see in the real world. 

We don't want our delivery robot to get confused, so we have created our own semantic segmentation dataset using images captured on sidewalks. It's available at [segments/sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic). This can be done using annotation platforms like [CVAT](https://www.cvat.ai/) or[Segments.ai](https://segments.ai/).

### Use a dataset from the Hub

We'll load the full labeled sidewalk dataset here. Note that you can check out the examples [directly in your browser](https://huggingface.co/datasets/segments/sidewalk-semantic). 

```python
hf_dataset_identifier = "segments/sidewalk-semantic"
```

## 2. Load and prepare the Hugging Face dataset for training

Now that we've created a new dataset and pushed it to the Hugging Face Hub, we can load the dataset in a single line.


```python
from datasets import load_dataset

ds = load_dataset(hf_dataset_identifier)
```

Let's shuffle the dataset and split the dataset in a train and test set.


```python
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
```

We'll extract the number of labels and the human-readable ids, so we can configure the segmentation model correctly later on.


```python
import json
from huggingface_hub import hf_hub_download

repo_id = f"datasets/{hf_dataset_identifier}"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
```

### Image processor & data augmentation

A SegFormer model expects the input to be of a certain shape. To transform our training data to match the expected shape, we can use `SegFormerImageProcessor`. We could use the `ds.map` function to apply the image processor to the whole training dataset in advance, but this can take up a lot of disk space. Instead, we'll use a *transform*, which will only prepare a batch of data when that data is actually used (on-the-fly). This way, we can start training without waiting for further data preprocessing.

In our transform, we'll also define some data augmentations to make our model more resilient to different lighting conditions. We'll use the [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) function from `torchvision` to randomly change the brightness, contrast, saturation, and hue of the images in the batch.


```python
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor

processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)
```

## 3. Fine-tune a SegFormer model

### Load the model to fine-tune

The SegFormer authors define 5 models with increasing sizes: B0 to B5. The following chart (taken from the original paper) shows the performance of these different models on the ADE20K dataset, compared to other models.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="SegFormer model variants compared with other segmentation models" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/segformer.png"></medium-zoom>
  <figcaption><a href="https://arxiv.org/abs/2105.15203">Source</a></figcaption>
</figure>

Here, we'll load the smallest SegFormer model (B0), pre-trained on ImageNet-1k. It's only about 14MB in size!
Using a small model will make sure that our model can run smoothly on our pizza delivery robot.


```python
from transformers import SegformerForSemanticSegmentation

pretrained_model_name = "nvidia/mit-b0" 
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)
```

### Set up the Trainer

To fine-tune the model on our data, we'll use Hugging Face's [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer). We need to set up the training configuration and an evalutation metric to use a Trainer.

First, we'll set up the [`TrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). This defines all training hyperparameters, such as learning rate and the number of epochs, frequency to save the model and so on. We also specify to push the model to the hub after training (`push_to_hub=True`) and specify a model name (`hub_model_id`).


```python
from transformers import TrainingArguments

epochs = 50
lr = 0.00006
batch_size = 2

hub_model_id = "segformer-b0-finetuned-segments-sidewalk-2"

training_args = TrainingArguments(
    "segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="end",
)
```

Next, we'll define a function that computes the evaluation metric we want to work with. Because we're doing semantic segmentation, we'll use the [mean Intersection over Union (mIoU)](https://huggingface.co/spaces/evaluate-metric/mean_iou), directly accessible in the [`evaluate` library](https://huggingface.co/docs/evaluate/index). IoU represents the overlap of segmentation masks. Mean IoU is the average of the IoU of all semantic classes. Take a look at [this blogpost](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) for an overview of evaluation metrics for image segmentation.

Because our model outputs logits with dimensions height/4 and width/4, we have to upscale them before we can compute the mIoU.


```python
import torch
from torch import nn
import evaluate

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=0,
        reduce_labels=processor.do_reduce_labels,
    )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics
```

Finally, we can instantiate a `Trainer` object.


```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

Now that our trainer is set up, training is as simple as calling the `train` function. We don't need to worry about managing our GPU(s), the trainer will take care of that.


```python
trainer.train()
```

When we're done with training, we can push our fine-tuned model and the image processor to the Hub.

This will also automatically create a model card with our results. We'll supply some extra information in `kwargs` to make the model card more complete.


```python
kwargs = {
    "tags": ["vision", "image-segmentation"],
    "finetuned_from": pretrained_model_name,
    "dataset": hf_dataset_identifier,
}

processor.push_to_hub(hub_model_id)
trainer.push_to_hub(**kwargs)
```

## 4. Inference

Now comes the exciting part, using our fine-tuned model! In this section, we'll show how you can load your model from the hub and use it for inference. 

However, you can also try out your model directly on the Hugging Face Hub, thanks to the cool widgets powered by the [hosted inference API](https://api-inference.huggingface.co/docs/python/html/index.html). If you pushed your model to the Hub in the previous step, you should see an inference widget on your model page. You can add default examples to the widget by defining example image URLs in your model card. See [this model card](https://huggingface.co/tobiasc/segformer-b0-finetuned-segments-sidewalk/blob/main/README.md) as an example.

<figure class="image table text-center m-0 w-full">
    <video 
        alt="The interactive widget of the model"
        style="max-width: 70%; margin: auto;"
        autoplay loop autobuffer muted playsinline
    >
      <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/widget.mp4" poster="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/widget-poster.png" type="video/mp4">
  </video>
</figure>

### Use the model from the Hub

We'll first load the model from the Hub using `SegformerForSemanticSegmentation.from_pretrained()`.

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained(f"{hf_username}/{hub_model_id}")
```

Next, we'll load an image from our test dataset.


```python
image = test_ds[0]['pixel_values']
gt_seg = test_ds[0]['label']
image
```

To segment this test image, we first need to prepare the image using the image processor. Then we forward it through the model.

We also need to remember to upscale the output logits to the original image size. In order to get the actual category predictions, we just have to apply an `argmax` on the logits.


```python
from torch import nn

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# First, rescale logits to original image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (height, width)
    mode='bilinear',
    align_corners=False
)

# Second, apply argmax on the class dimension
pred_seg = upsampled_logits.argmax(dim=1)[0]
```

Now it's time to display the result. We'll display the result next to the ground-truth mask.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(1,1,1,1)" alt="SegFormer prediction vs the ground truth" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/output.png"></medium-zoom>
</figure>

What do you think? Would you send our pizza delivery robot on the road with this segmentation information?

The result might not be perfect yet, but we can always expand our dataset to make the model more robust. We can now also go train a larger SegFormer model, and see how it stacks up.

## 5. Conclusion

That's it! You now know how to create your own image segmentation dataset and how to use it to fine-tune a semantic segmentation model.

We introduced you to some useful tools along the way, such as:


*   [Segments.ai](https://segments.ai) for labeling your data
*   [🤗 datasets](https://huggingface.co/docs/datasets/) for loading and sharing a dataset
*   [🤗 transformers](https://huggingface.co/transformers) for easily fine-tuning a state-of-the-art segmentation model
*   [Hugging Face Hub](https://huggingface.co/docs/hub/main) for sharing our dataset and model, and for creating an inference widget for our model


We hope you enjoyed this post and learned something. Feel free to share your own model with us on Twitter ([@TobiasCornille](https://twitter.com/tobiascornille), [@NielsRogge](https://twitter.com/nielsrogge), and [@huggingface](https://twitter.com/huggingface)).
