---
title: "New and Open-Source ViT and ALIGN from Kakao Brain" 
thumbnail: /blog//assets/130_vit_align/thumbnail.png
authors:
- user: Unso
- user: dylan-m
- user: jun-untitled
- user: adirik
---


# Kakao Brain’s Open Source ViT, ALIGN, and the new COYO text-image dataset

<!-- {blog_metadata} -->
<!-- {authors} -->

Kakao Brain and Hugging Face are excited to release a new open-source image-text dataset [COYO](https://github.com/kakaobrain/coyo-dataset) of 700 million pairs and two new visual language models trained on it, [ViT](https://github.com/kakaobrain/coyo-vit) and [ALIGN](https://github.com/kakaobrain/coyo-align). This is the first time ever the ALIGN model is made public for free and open-source use and the first release of ViT and ALIGN models that come with the train dataset.  

Kakao Brain’s ViT and ALIGN models follow the same architecture and hyperparamters as provided in the original respective Google models but are trained on the open source [COYO](https://github.com/kakaobrain/coyo-dataset) dataset. Google’s [ViT](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) and [ALIGN](https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html) models, while trained on huge datasets (ViT trained on 300 million images and ALIGN trained on 1.8 billion image-text pairs respectively), cannot be replicated because the datasets are not public. This contribution is particularly valuable to researchers who want to reproduce visual language modeling with access to the data as well. More detailed information on the Kakao ViT and ALIGN models can be found [here](https://huggingface.co/kakaobrain).  

This blog will introduce the new [COYO](https://github.com/kakaobrain/coyo-dataset) dataset, Kakao Brain's ViT and ALIGN models, and how to use them! Here are the main takeaways:

* First open-source ALIGN model ever! 
* First open ViT and ALIGN models that have been trained on an open-source dataset [COYO](https://github.com/kakaobrain/coyo-dataset)
* Kakao Brain's ViT and ALIGN models perform on-par with the Google versions
* ViT demo is available on HF! You can play with the ViT demo online with image samples of your own choice!


## Performance Comparison

Kakao Brain's released ViT and ALIGN models perform on par and sometimes better than what Google has reported about their implementation. Kakao Brain's `ALIGN-B7-Base` model, while trained on a much fewer pairs (700 million pairs vs 1.8 billion), performs on par with Google's `ALIGN-B7-Base` on the Image KNN classification task and better on MS-COCO retrieval image-to-text, text-to-image tasks. Kakao Brain's `ViT-L/16` performs similarly to Google's `ViT-L/16` when evaluated on ImageNet and ImageNet-ReaL at model resolutions 384 and 512. This means the community can use Kakao Brain's ViT and ALIGN models to replicate Google's ViT and ALIGN releases especially when users require access to the training data. We are excited to see open-source and transparent releases of these model that perform on par with the state of the art!

<p>
<center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/130_vit_align/align.png" alt="align performance" width="430"/><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/130_vit_align/vit.png" alt="vit performance" width="430"/>
</center>
</p>

## COYO DATASET

<p>
<center>
<img src="https://cdn.shopify.com/s/files/1/0190/8574/products/Art_Riley-Monterey_Fishing_Fleet_1_grande.jpg?v=1479962684" width="290" /><img src="https://api.time.com/wp-content/uploads/2015/03/168951187.jpg" width="330" /><img src="https://s.yimg.com/ny/api/res/1.2/mOZe9uKtwugmPrqeXBlxFg--/YXBwaWQ9aGlnaGxhbmRlcjt3PTk2MDtoPTYzMA--/https://s.yimg.com/uu/api/res/1.2/JuTSVK74cI8II09Q75uzGA--~B/aD01MjU7dz04MDA7YXBwaWQ9eXRhY2h5b24-/https://media.zenfs.com/en/reuters.com/15941d3b47960da80f8033f4ddf9da64" width="330" />
</center>
</p>

What's special about these model releases is that the models are trained on the free and accessible COYO dataset. [COYO](https://github.com/kakaobrain/coyo-dataset#dataset-preview) is an image-text dataset of 700 million pairs similar to Google's `ALIGN 1.8B` image-text dataset which is a collection of "noisy" alt-text and image pairs from webpages, but open-source. `COYO-700M` and `ALIGN 1.8B` are "noisy" because minimal filtering was applied. `COYO` is similar to the other open-source image-text dataset, `LAION` but with the following differences. While `LAION` 2B is a much larger dataset of 2 billion English pairs, compared to `COYO`’s 700 million pairs, `COYO` pairs come with more metadata that give users more flexibility and finer-grained control over usage. The following table shows the differences: `COYO` comes equipped with aesthetic scores for all pairs, more robust watermark scores, and face count data. 


| COYO | LAION 2B| ALIGN 1.8B |
| :----: | :----: | :----: |
| Image-text similarity score calculated with CLIP ViT-B/32 and ViT-L/14 models, they are provided as metadata but nothing is filtered out so as to avoid possible elimination bias | Image-text similarity score provided with CLIP (ViT-B/32) - only examples above threshold 0.28 | Minimal, Frequency based filtering | 
| NSFW filtering on images and text | NSFW filtering on images | [Google Cloud API](https://cloud.google.com/vision) |
| Face recognition (face count) data provided as meta-data | No face recognition data | NA | 
| 700 million pairs all English | 2 billion English| 1.8 billion | 
| From CC 2020 Oct - 2021 Aug| From CC 2014-2020|  NA |
|Aesthetic Score | Aesthetic Score Partial | NA| 
|More robust Watermark score | Watermark Score |  NA| 
|Hugging Face Hub | Hugging Face Hub | Not made public |  
| English | English | English? | 
                                                                                                  

## How ViT and ALIGN work

So what do these models do? Let's breifly discuss how the ViT and ALIGN models work.

ViT -- Vision Transformer -- is a vision model [proposed by Google in 2020](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) that resembles the text Transformer architecture. 
It is a new approach to vision, distinct from convolutional neural nets (CNNs) that have dominated vision tasks since 2012's AlexNet. It is upto four times more computationally efficient than similarly performing CNNs and domain agnostic. ViT takes as input an image which is broken up into a sequence of image patches - just as the text Transformer takes as input a sequence of text -  and given position embeddings to each patch to learn the image structure. ViT performance is notable in particular for having an excellent performance-compute trade-off. While some of Google's ViT models are open-source, the JFT-300 million image-label pair dataset they were trained on has not been released publicly. While Kakao Brain's trained on [COYO-Labeled-300M](https://github.com/kakaobrain/coyo-dataset/tree/main/subset/COYO-Labeled-300M), which has been released publicly, and released ViT model performs similarly on various tasks, its code, model, and training data(COYO-Labeled-300M) are made entirely public for reproducibility and open science.

<p>
<center>
<img src="https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif" width="700" />
</center>
</p>
<p>
<center>
<em>A Visualization of How ViT Works from <a href="https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html">Google Blog</a></em>
</center>
</p>

 [Google then introduced ALIGN](https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html) -- a Large-scale Image and Noisy Text Embedding model in 2021 -- a visual-language model trained on "noisy" text-image data for various vision and cross-modal tasks such as text-image retrieval. ALIGN has a simple dual-encoder architecture trained on image and text pairs, learned via a contrastive loss function. ALIGN's "noisy" training corpus is notable for balancing scale and robustness. Previously, visual language representational learning had been trained on large-scale datasets with manual labels, which require extensive preprocessing. ALIGN's corpus uses the image alt-text data, text that appears when the image fails to load, as the caption to the image -- resulting in an inevitably noisy, but much larger (1.8 billion pair) dataset that allows ALIGN to perform at SoTA levels on various tasks. Kakao Brain's ALIGN is the first open-source version of this model, trained on the `COYO` dataset and performs better than Google's reported results.

<p>
<center>
<img src="https://1.bp.blogspot.com/-M5VbNqegBqM/YJqTsnf1JzI/AAAAAAAAHlk/UKkhs1XFelQ8gnKPINyD7z8H4wg3J9EzACLcBGAsYHQ/s1449/image4.png" width="700" />
</center>
</p>
<p>
<center>
<em>ALIGN Model from <a href="https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html">Google Blog</a>
</em>
</center>
<p>


## How to use the COYO dataset
To use the `COYO` dataset, refer to the [COYO github page](https://github.com/kakaobrain/coyo-dataset/tree/main/download).

## How to use ViT and ALIGN from the Hub
Let’s go ahead and experiment with the new ViT and ALIGN models. First, let’s install 🤗Transformers: `pip install transformers` and get started with ViT for image classification by importing the modules and libraries we will use.

```
import requests
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
```

Next, we will download a random image of two cats from the COCO dataset and preprocess the image to transform it to the input format expected by the model. To do this, we can conveniently use the corresponding preprocessor class (`ViTProcessor`). To initialize the model and the preprocessor, we will use one of the [Kakao Brain ViT repos](https://huggingface.co/models?search=kakaobrain/vit) on the hub. Note that initializing the preprocessor from a repository ensures that the preprocessed image is in the expected format required by that specific pretrained model.

```
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('kakaobrain/vit-large-patch16-384')
model = ViTForImageClassification.from_pretrained('kakaobrain/vit-large-patch16-384')
```

The rest is simple, we will forward preprocess the image and use it as input to the model to retrive the class logits. The Kakao Brain ViT image classification models are trained on ImageNet labels and outputs logits of shape (batch_size, 1000).
```
# preprocess image or list of images
inputs = processor(images=image, return_tensors="pt")

# inference
with torch.no_grad():
    outputs = model(**inputs)

# apply SoftMax to logits to compute the probability of each class
preds = torch.nn.functional.softmax(outputs.logits)

# print the top 5 class predictions and their probabilities
top_class_preds = torch.argsort(preds, descending=True)[:5]

for c in top_class_preds:
    print(f"{model.config.id2label[c.item()]} with probability {round(preds[c.item()].item(), 4)}")

>>> remote control, remote with probability 0.8224
>>> tabby, tabby cat with probability 0.0658
>>> tiger cat with probability 0.0656
>>> Egyptian cat with probability 0.0389
>>> lynx, catamount with probability 0.0011
```

And we are done! If you want to experiment more with the Kakao Brain ViT model, head over to its [Space](https://huggingface.co/spaces/adirik/kakao-brain-vit) on the 🤗 Hub.
<center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/130_vit_align/vit_demo.png" alt="vit performance" width="900"/>
</center>

Let's move on to experimenting with ALIGN, which can be used to retrieve multi-modal embeddings of texts or images or to perform zero-shot image classification. ALIGN's transformers implementation and usage is similar to [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip). To get started, we will first download the pretrained model and its processor, which can preprocess both the images and texts such that they are in the expected format to be fed into the vision and text encoders of ALIGN. Once again, let's import the modules we will use and initialize the preprocessor and the model.

```
import requests
from PIL import Image
import torch
from transformers import AlignProcessor, AlignModel


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
model = AlignModel.from_pretrained('kakaobrain/align-base')
```

We will start with zero-shot image classification first. To do this, we will suppy candidate labels (free-form text) and use AlignModel to find out which description better describes the image. We will first preprocess both the image and text inputs and feed the preprocessed input to the AlignModel.

```
candidate_labels = ['an image of a cat', 'an image of a dog']

inputs = processor(images=image, text=candidate_labels, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image  

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)  
print(probs)
```

Done, easy as that. We can also use the vision and text encoders of ALIGN separately to retrieve multi-modal embeddings. These embeddings can then be used to train models for various downstream tasks such as object detection, image segmentation and image captioning. Let's see how we can retrieve these embeddings using `AlignTextModel` and `AlignVisionModel`.


