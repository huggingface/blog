---
title: Universal Image Segmentation with Mask2Former and OneFormer
thumbnail: /blog/assets/127_mask2former/thumbnail.png
---

<h1>
	Universal Image Segmentation with Mask2Former and OneFormer
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
    <a href="/shivi">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/73357305?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>nielsr</code>
            <span class="fullname">Shivalika Singh</span>
        </div>
    </a>
    <a href="/adirik">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/8944735?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>nielsr</code>
            <span class="fullname">Alara Dirik</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<a target="_blank" href="https://colab.research.google.com/drive/1MdkavsjGHYcuGyjmsf9wmeAK3WvtYLty?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**This guide introduces Mask2Former and OneFormer, 2 state-of-the-art neural networks for image segmentation. The models are now available in [`🤗 transformers`](https://huggingface.co/transformers), an open-source library that offers easy-to-use implementations of state-of-the-art models. Along the way, you'll learn about the difference between the various forms of image segmentation.**

## Image segmentation

Image segmentation is the task of identifying different "segments" in an image, like people or cars. More technically, image segmentation is the task of grouping pixels with different semantics. Refer to the Hugging Face [task page](https://huggingface.co/tasks/image-segmentation) for a brief introduction.

Image segmentation can largely be split into 3 subtasks - instance, semantic and panoptic segmentation - with numerous methods and model architectures to perform each subtask.

- **instance segmentation** is the task of identifying different "instances", like individual people, in an image. Instance segmentation is very similar to object detection, except that we'd like to output a set of binary segmentation masks, rather than bounding boxes, with corresponding class labels. Instances are oftentimes also called "objects" or "things". Note that individual instances may overlap.
- **semantic segmentation** is the task of identifying different "semantic categories", like "person" or "sky" of each pixel in an image. Contrary to instance segmentation, no distinction is made between individual instances of a given semantic category; one just likes to come up with a mask for the "person" category, rather than for the individual people for example. Semantic categories which don't have individual instances, like "sky" or "grass", are oftentimes referred to as "stuff", to make the distinction with "things" (great names, huh?). Note that no overlap between semantic categories is possible, as each pixel belongs to one category.
- **panoptic segmentation**, introduced in 2018 by [Kirillov et al.](https://arxiv.org/abs/1801.00868), aims to unify instance and semantic segmentation, by making models simply identify a set of "segments", each with a corresponding binary mask and class label. Segments can be both "things" or "stuff". Unlike in instance segmentation, no overlap between different segments is possible.

The figure below illustrates the difference between the 3 subtasks (taken from [this blog post](https://www.v7labs.com/blog/panoptic-segmentation-guide)).

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/127_mask2former/semantic_vs_semantic_vs_panoptic.png" alt="drawing" width=500>
</p>

Over the last years, researchers have come up with several architectures that were typically very tailored to either instance, semantic or panoptic segmentation. Instance and panoptic segmentation were typically solved by outputting a set of binary masks + corresponding labels per object instance (very similar to object detection, except that one outputs a binary mask instead of a bounding box per instance). This is oftentimes called "binary mask classification". Semantic segmentation on the other hand was typically solved by making models output a single "segmentation map" with one label per pixel. Hence, semantic segmentation was treated as a "per-pixel classification" problem. Popular semantic segmentation models which adopt this paradigm are [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer), on which we wrote an extensive [blog post](https://huggingface.co/blog/fine-tune-segformer), and [UPerNet](https://huggingface.co/docs/transformers/main/en/model_doc/upernet).

## Universal image segmentation

Luckily, since around 2020, people started to come up with models that can solve all 3 tasks (instance, semantic and panoptic segmentation) with a unified architecture, using the same paradigm. This started with [DETR](https://huggingface.co/docs/transformers/model_doc/detr), which was the first model that solved panoptic segmentation using a "binary mask classification" paradigm, by treating "things" and "stuff" classes in a unified way. The key innovation was to have a Transformer decoder come up with a set of binary masks + classes in a parallel way. This was then improved in the [MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer) paper, which showed that the "binary mask classification" paradigm also works really well for semantic segmentation.

[Mask2Former](https://huggingface.co/docs/transformers/main/model_doc/mask2former) extends this to instance segmentation by further improving the neural network architecture. Hence, we've evolved from separate architectures to what researchers now refer to as "universal image segmentation" architectures, capable of solving any image segmentation task. Interestingly, these universal models all adopt the "mask classification" paradigm, discarding the "per-pixel classification" paradigm entirely. A figure illustrating Mask2Former's architecture is depicted below (taken from the [original paper](https://arxiv.org/abs/2112.01527)).

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mask2former_architecture.jpg" alt="drawing" width=500>
</p>

In short, an image is first sent through a backbone (which, in the paper could be either [ResNet](https://huggingface.co/docs/transformers/model_doc/resnet) or [Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin)) to get a list of low-resolution feature maps. Next, these feature maps are enhanced using a pixel decoder module to get high-resolution features. Finally, a Transformer decoder takes in a set of queries and transforms them into a set of binary mask and class predictions, conditioned on the pixel decoder's features.

Note that Mask2Former still needs to be trained on each task separately to obtain state-of-the-art results. This has been improved by the [OneFormer](https://arxiv.org/abs/2211.06220) model, which obtains state-of-the-art performance on all 3 tasks by only training on a panoptic version of the dataset (!), by adding a text encoder to condition the model on either "instance", "semantic" or "panoptic" inputs. This model is also as of today [available in 🤗 transformers](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer). It's even more accurate than Mask2Former, but comes with greater latency due to the additional text encoder. See the figure below for an overview of OneFormer. It leverages either Swin Transformer or the new [DiNAT](https://huggingface.co/docs/transformers/model_doc/dinat) model as backbone.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png" alt="drawing" width=500>
</p>

## Inference with Mask2Former and OneFormer in Transformers

Usage of Mask2Former and OneFormer is pretty straightforward, and very similar to their predecessor MaskFormer. Let's instantiate a Mask2Former model from the hub trained on the COCO panoptic dataset, along with its processor. Note that the authors released no less than [30 checkpoints](https://huggingface.co/models?other=mask2former) trained on various datasets.

```
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
```

Next, let's load the familiar cats image from the COCO dataset, on which we'll perform inference.

```
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<img src="assets/78_annotated-diffusion/output_cats.jpeg" width="400" />

We prepare the image for the model using the image processor, and forward it through the model.

```
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
```

The model outputs a set of binary masks and corresponding class logits. The raw outputs of Mask2Former can be easily postprocessed using the image processor to get the final instance, semantic or panoptic segmentation predictions:

```
prediction = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
print(prediction.keys())
```
<div class="output stream stdout">

    Output:
    ----------------------------------------------------------------------------------------------------
    dict_keys(['segmentation', 'segments_info'])

</div>

In panoptic segmentation, the final `prediction` contains 2 things: a `segmentation` map of shape (height, width) where each value encodes the instance ID of a given pixel, as well as a corresponding `segments_info`. The  `segments_info` contains more information about the individual segments of the map (such as their class / category ID). Note that Mask2Former outputs binary mask proposals of shape (96, 96) for efficiency and the `target_sizes` argument is used to resize the final mask to the original image size.

Let's visualize the results:

```
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    ax.legend(handles=handles)

draw_panoptic_segmentation(**panoptic_segmentation)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/127_mask2former/cats_panoptic_result.png" width="400" />

Here, we can see that the model is capable of detecting the individual cats and remotes in the image. Semantic segmentation on the other hand would just create a single mask for the "cat" category.

To perform inference with OneFormer, which has an identical API except that it also takes an additional text prompt as input, we refer to the [demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer).

## Fine-tuning Mask2Former and OneFormer in Transformers

For fine-tuning Mask2Former/OneFormer on a custom dataset for either instance, semantic and panoptic segmentation, check out our [demo notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer/Fine-tuning). MaskFormer, Mask2Former and OneFormer share a similar API so upgrading from MaskFormer is easy and requires minimal changes.

The demo notebooks make use of `MaskFormerForInstanceSegmentation` to load the model whereas you'll have to switch to using either `Mask2FormerForUniversalSegmentation` or `OneFormerForUniversalSegmentation`. The image preprocessing part would not require any changes for Mask2Former, since MaskFormer and Mask2Former share a common image processor, that is, `MaskFormerImageProcessor`. You can also load the image processor using the `AutoImageProcessor` class which automatically takes care of loading the correct processor corresponding to your model. OneFormer on the other hand requires a `OneFormerProcessor`, which prepares the images, along with a text input, for the model.

# Conclusion

That's it! You now know about the difference between instance, semantic and panoptic segmentation, as well as how to use "universal architectures" such as Mask2Former and OneFormer using the [🤗 transformers](https://huggingface.co/transformers) library.

We hope you enjoyed this post and learned something. Feel free to let us know whether you are satisfied with the results when fine-tuning Mask2Former or OneFormer.

If you liked this topic and want to learn more, we recommend the following resources:

- Our demo notebooks for [MaskFormer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer), [Mask2Former](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former) and [OneFormer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OneFormer), which give a broader overview on inference (including visualization) as well as fine-tuning on custom data.
- The [live demo](https://huggingface.co/spaces/shi-labs/OneFormer) on the Hugging Face Hub which you can use to quickly try out OneFormer on sample inputs of your choice.