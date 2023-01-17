---
title: Towards Universal Image Segmentation with Mask2Former
thumbnail: /blog/assets/56_fine_tune_segformer/thumb.png
---

<h1>
	Towards Universal Image Segmentation with Mask2Former
</h1>

<div class="blog-metadata">
    <small>Published March 17, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/mask2former.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/nielsr">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/48327001?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>nielsr</code>
            <span class="fullname">Niels Rogge</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<a target="_blank" href="https://colab.research.google.com/drive/1MdkavsjGHYcuGyjmsf9wmeAK3WvtYLty?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**This guide introduces Mask2Former, a state-of-the-art neural network for image segmentation. The model is now available in [`🤗 transformers`](https://huggingface.co/transformers), an open-source library that offers easy-to-use implementations of state-of-the-art models. Along the way, you'll learn about the difference between the various forms of image segmentation.**

## Image segmentation

Image segmentation is the task of identifying different "segments" in an image, like people or cars. More technically, image segmentation is the task of grouping pixels with different semantics. Refer to the HuggingFace [task page](https://huggingface.co/tasks/image-segmentation) for a brief introduction.

Over the years, people have come up with different ways of performing image segmentation. They can largely be split into 3 subtasks, called instance, semantic and panoptic segmentation:

- **instance segmentation** is the task of identifying different "instances", like individual people, in an image. Instance segmentation is very similar to object detection, except that we'd like to output a set of binary segmentation masks, rather than bounding boxes, with corresponding class labels. Instances are oftentimes also called "objects" or "things". Note that individual instances may overlap.
- **semantic segmentation** is the task of identifying different "semantic categories", like "person" or "sky" in an image. Contrary to instance segmentation, no distinction is made between individual instances of a given semantic category; one just likes to come up with a mask for the "person" category, rather than for the individual people for example. Semantic categories which don't have individual instances, like "sky" or "grass", are oftentimes referred to as "stuff", to make the distinction with "things" (great names, huh?). Note that no overlap between semantic categories is possible, as each pixel belongs to one category.
- **panoptic segmentation**, introduced in 2018 by [Kirillov et al.](https://arxiv.org/abs/1801.00868), aims to unify instance and semantic segmentation, by making models simply identify a set of "segments", each with a corresponding binary mask and class label. Segments can be both "things" or "stuff". Unlike in instance segmentation, no overlap between different segments is possible.

Over the last years, researchers have come up with several architectures that were typically very tailored to either instance, semantic or panoptic segmentation. Instance and panoptic segmentation were typically solved by outputting a set of binary masks + corresponding labels (very similar to object detection, except that one outputs a binary mask instead of a bounding box per instance). This is oftentimes called "binary mask classification". Semantic segmentation on the other hand was typically solved by making models output a single "segmentation map" with one label per pixel. Hence, semantic segmentation was treated as a "per-pixel classification" problem. An example of a popular semantic segmentation model which adopts this paradigm is [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer), on which we wrote an extensive blog post [here](https://huggingface.co/blog/fine-tune-segformer).

## Universal image segmentation

Luckily, since around 2020, people started to come up with models that can solve all 3 tasks (instance, semantic and panoptic segmentation) with a unified architecture, using the same paradigm. This started with [DETR](https://huggingface.co/docs/transformers/model_doc/detr), which was the first model that solved panoptic segmentation using a "binary mask classification" paradigm, by treating "things" and "stuff" classes in a unified way. The key innovation was to have a Transformer decoder come up with a set of binary masks + classes in a parallel way. This was then improved in the [MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer) paper, which showed that the "binary mask classification" paradigm also works really well for semantic segmentation.

Later on, Mask2Former extended this to instance segmentation by further improving the neural network architecture. Hence, we've evolved from separate architectures to what researchers now refer to as "universal image segmentation" architectures, capable of solving any image segmentation task. Interestingly, these universal models all adopt the "mask classification" paradigm, discarding the "per-pixel classification" paradigm entirely.

Note that Mask2Former still needs to be trained on each task separately to obtain state-of-the-art results. This has already been improved by [OneFormer](https://arxiv.org/abs/2211.06220), which obtains state-of-the-art performance on all 3 tasks by only training on a panoptic version of the dataset, by adding a text encoder to condition the model on either "instance", "semantic" or "panoptic". This model will also become available in [`🤗 transformers`](https://huggingface.co/transformers), but comes with greater latency.

## Inference with Mask2Former in Transformers

Usage of Mask2Former is pretty straightforward, and exactly the same as its predecessor MaskFormer. One prepares an image for the model using the image processor, and then forwards the `pixel_values` through the model. The model outputs a set of binary masks and corresponding class logits. Next, these can be postprocessed using the image processor to get the final instance, semantic or panoptic segmentation predictions.

Below, we load a Mask2Former checkpoint from the hub trained on the COCO panoptic dataset (note that the authors released no less than [30 checkpoints](https://huggingface.co/models?other=mask2former)). Hence we use the `post_process_panoptic_segmentation` method to get the final predicted panoptic segmentation.

```
from transformers import AutoImageProcessor, Mask2FormerForUniversalImageSegmentation
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalImageSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
```

In panoptic segmentation, the final `predicted_panoptic_segmentation` contains 2 things: a segmentation map, as well as a corresponding `segments_info`, which contains more information about the individual segments of the map (such as their class ID). Here one can see that in panoptic segmentation, no overlap between segments is possible due to the fact that ground truths are stored in a single segmentation map.

For visualization, we refer the reader to the [demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former/Inference_with_Mask2Former.ipynb), which also illustrates semantic segmentation.

## Fine-tuning Mask2Former in Transformers

For fine-tuning Mask2Former on a custom dataset for either instance, semantic and panoptic segmentation, we refer the reader to the [demo notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Mask2Former). Note that fine-tuning Mask2Former has the exact same API as fine-tuning MaskFormer. Hence, upgrading from MaskFormer to Mask2Former is easy!

# Conclusion

That's it! You now know about the difference between instance, semantic and panoptic segmentation, as well as how to use "universal architectures" such as Mask2Former using the [🤗 transformers](https://huggingface.co/transformers) library.

We hope you enjoyed this post and learned something. Feel free to let us know whether you are satisfied with the results you get when fine-tuning Mask2Former.