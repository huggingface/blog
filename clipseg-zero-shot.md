---
title: Zero-shot image segmentation with CLIPSeg
thumbnail: /blog/assets/123_clipseg-zero-shot/thumb.png
---

<h1>
	Using CLIPSeg with Hugging Face Transformers
</h1>

<div class="blog-metadata">
    <small>Published December 23, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/clipseg-zero-shot.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/segments-tobias">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/89590365?v=4" title="Gravatar">
        <div class="bfc">
            <code>segments-tobias</code>
            <span class="fullname">Tobias Cornille</span>
            <span class="bg-gray-100 dark:bg-gray-700 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
    </a>
    <a href="/nielsr">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/48327001?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>nielsr</code>
            <span class="fullname">Niels Rogge</span>
        </div>
    </a>
</div>

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<a target="_blank" href="https://colab.research.google.com/drive/1x1xHpQT4IjqhB6qX9m05uCmcpD3thF7x?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**This guide shows how you can use CLIPSeg, a zero-shot image segmentation model, using [`🤗 transformers`](https://huggingface.co/transformers). CLIPSeg creates rough segmentation masks that can be used for robot perception, image inpainting, and many other tasks. If you need more precise segmentation masks, we’ll show how you can refine the results of CLIPSeg on [Segments.ai](https://segments.ai/?utm_source=hf&utm_medium=blog&utm_campaign=clipseg).**

Image segmentation is a well-known task within the field of computer vision. It allows a computer to not only know what is in an image (classification), where objects are in the image (detection), but also what the outlines of those objects are. Knowing the outlines of objects is essential in fields such as robotics and autonomous driving. For example, a robot has to know the shape of an object to grab it correctly. Segmentation can also be combined with [image inpainting](https://t.co/5q8YHSOfx7) to allow users to describe which part of the image they want to replace.

One limitation of most image segmentation models is that they only work with a fixed list of categories. For example, you cannot simply use a segmentation model trained on oranges to segment apples. To teach the segmentation model an additional category, you have to label data of the new category and train a new model, which can be costly and time-consuming. But what if there was a model that can already segment almost any kind of object, without any further training? That’s exactly what CLIPSeg, a zero-shot segmentation model, achieves.

Currently, CLIPSeg still has its limitations. For example, the model uses images of 352 x 352 pixels, so the output is quite low-resolution. This means we cannot expect pixel-perfect results when we work with images from modern cameras. If we want more precise segmentations, we can fine-tune a state-of-the-art segmentation model, as shown in [our previous blog post](https://huggingface.co/blog/fine-tune-segformer). In that case, we can still use CLIPSeg to generate some rough labels, and then refine them in a labeling tool such as [Segments.ai](http://Segments.ai). Before we describe how to do that, let’s first take a look at how CLIPSeg works.

## CLIP: the magic model behind CLIPSeg

[CLIP](https://openai.com/blog/clip/), which stands for Contrastive Language–Image Pre-training, is a model developed by OpenAI in 2021. You can give CLIP an image or a piece of text, and CLIP will output an abstract *representation* of your input. This abstract representation, also called an *embedding*, is really just a vector (a list of numbers). You can think of this vector as a point in high-dimensional space. CLIP is trained so that the representations of similar pictures and texts are similar as well. This means that if we input an image and a text description that fits that image, the representations of the image and the text will be similar (i.e. the high-dimensional points will be close together).

At first, this might not seem very useful, but it is actually very powerful. As an example, let’s take a quick look at how CLIP can be used to classify images without ever having been trained on that task. To classify an image, we input the image and the different categories we want to choose from to CLIP (e.g. we input an image and the words “apple”, “orange”, …).  CLIP then gives us back an embedding of the image and of each category. Now, we simply have to check which category embedding is closest to the embedding of the image, et voilà! Feels like magic, doesn’t it? 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Overview of the CLIPSeg model" src="assets/123_clipseg-zero-shot/clip-tv-example.png"></medium-zoom>
  <figcaption>Example of image classification using CLIP (<a href="https://openai.com/blog/clip/">source</a>).</figcaption>
</figure>

What’s more, CLIP is not only useful for classification, but it can also be used for image search (can you see how this is similar to classification?), text-to-image models (DALL-E 2 is powered by CLIP), object detection (OWL-ViT), and most importantly for us: image segmentation. Now you see why CLIP was truly a breakthrough in machine learning.

The reason why CLIP works so well is that the model was trained on a huge dataset of images with text captions. The dataset contained a whopping 400 million image-text pairs taken from the internet. These images contain a wide variety of objects and concepts, and CLIP is great at creating a representation for each of them.

## CLIPSeg: image segmentation with CLIP

[CLIPSeg](https://arxiv.org/abs/2112.10003) is a model that uses CLIP representations to create image segmentation masks. It was published by Timo Lüddecke and Alexander Ecker. They achieved zero-shot image segmentation by training a Transformer-based decoder on top of the CLIP model, which is kept frozen. The decoder takes in the CLIP representation of an image, and the CLIP representation of the the thing you want to segment. Using these two inputs, the CLIPSeg decoder creates a binary segmentation mask. To be more precise, the decoder doesn’t only use the final CLIP representation of the image we want to segment, but it also uses the outputs of some of the layers of CLIP. 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Overview of the CLIPSeg model" src="assets/123_clipseg-zero-shot/clipseg-overview.png"></medium-zoom>
  <figcaption><a href="https://arxiv.org/abs/2112.10003">Source</a></figcaption>
</figure>

The decoder is trained on the [PhraseCut dataset](https://arxiv.org/abs/2008.01187), which contains over 340,000 phrases with corresponding image segmentations. The authors also experimented with various augmentations to expand the size of the dataset. The goal here is not only to be able to segment the categories that are present in the dataset, but also to segment unseen categories. Experiments indeed show that the decoder can generalize to unseen categories.

 

One interesting feature of CLIPSeg is that we input a CLIP embedding to specify which thing we want to segment. This CLIP embedding can come from a word or a piece of text, but also from another image. This means we can use an example image to segment objects that might be hard to describe (e.g. a specific logo). The authors call this “visual prompting”, and the paper contains some tips on improving the effectiveness of this technique. They find that cropping the query image so that it only contains the object you want to segment, helps a lot. Blurring and darkening the background of the query image also helps a little bit.

## Using CLIPSeg with Hugging Face Transformers

Using Hugging Face Transformers, you can easily download and run a
pre-trained CLIPSeg model on your images. Let's start by installing
transformers.

```python
!pip install -q transformers
```

To download the model, simply instantiate it.

```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
```

Now we can load an image to try out the segmentation. We\'ll choose a
picture of a delicious breakfast taken by [Calum
Lewis](https://unsplash.com/@calumlewis).

```python
from PIL import Image
import requests

url = "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<figure class="image table text-center m-0 w-6/12">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/73d97c93dc0f5545378e433e956509b8acafb8d9.png"></medium-zoom>
</figure>

## Text prompting

Let's start by defining some text categories we want to segment.

```python
prompts = ["cutlery", "pancakes", "blueberries", "orange juice"]
```

Now that we have our inputs, we can process them and input them to the
model.

```python
import torch

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**inputs)
preds = outputs.logits.unsqueeze(1)
```

Finally, let's visualize the output.

```python
import matplotlib.pyplot as plt

_, ax = plt.subplots(1, len(prompts) + 1, figsize=(3*(len(prompts) + 1), 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
[ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)];
```

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/14c048ea92645544c1bbbc9e55f3c620eaab8886.png"></medium-zoom>
</figure>

## Visual prompting

As mentioned before, we can also use images as the input prompts (i.e.
in place of the category names). This can be especially useful if it\'s
not easy to describe the thing you want to segment. For this example,
we\'ll use a picture of a coffee cup taken by [Daniel
Hooper](https://unsplash.com/@dan_fromyesmorecontent).

```python
url = "https://unsplash.com/photos/Ki7sAc8gOGE/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTJ8fGNvZmZlJTIwdG8lMjBnb3xlbnwwfHx8fDE2NzExOTgzNDQ&force=true&w=640"
prompt = Image.open(requests.get(url, stream=True).raw)
prompt
```

<figure class="image table text-center m-0 w-6/12">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/7931f9db82ab07af7d161f0cfbfc347645da6646.png"></medium-zoom>
</figure>

We can now process the input image and prompt image and input them to
the model.

```python
encoded_image = processor(images=[image], padding="max_length", return_tensors="pt")
encoded_prompt = processor(images=[prompt], padding="max_length", return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values)
preds = outputs.logits.unsqueeze(1)
preds = torch.transpose(preds, 0, 1)
```

```python
_, ax = plt.subplots(1, 2, figsize=(6, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
ax[1].imshow(torch.sigmoid(preds[0]))
```

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/fbde45fc65907d17de38b0db3eb262bdec1f1784.png"></medium-zoom>
</figure>

Let's try one last time by using the visual prompting tips described in
the paper, i.e. cropping the image and darkening the background.

```python
url = "https://i.imgur.com/mRSORqz.jpg"
prompt_alt = Image.open(requests.get(url, stream=True).raw)
prompt_alt
```

<figure class="image table text-center m-0 w-6/12">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/915a97da22131e0ab6ff4daa78ffe3f1889e3386.png"></medium-zoom>
</figure>

```python
encoded_prompt_alt = processor(images=[prompt_alt], padding="max_length", return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt_alt.pixel_values)
preds = outputs.logits.unsqueeze(1)
preds = torch.transpose(preds, 0, 1)
```

```python
_, ax = plt.subplots(1, 2, figsize=(6, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
ax[1].imshow(torch.sigmoid(preds[0]))
```

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/7f75badfc245fc3a75e0e05058b8c4b6a3a991fa.png"></medium-zoom>
</figure>

In this case, the result is pretty much the same. This is probably
because the coffee cup was already separated well from the background in
the original image.

## Using CLIPSeg to pre-label images on Segments.ai

As you can see, the results from CLIPSeg are a little fuzzy and very
low-res. If we want to obtain better results, you can fine-tune a
state-of-the-art segmentation model, as explained in [our previous
blogpost](https://huggingface.co/blog/fine-tune-segformer). To finetune
the model, we\'ll need labeled data. In this section, we\'ll show you
how you can use CLIPSeg to create some rough segmentation masks and then
refine them on
[Segments.ai](https://segments.ai/?utm_source=hf&utm_medium=colab&utm_campaign=clipseg),
the best labeling platform for image segmentation.

First, create an account at
[https://segments.ai/join](https://segments.ai/join?utm_source=hf&utm_medium=colab&utm_campaign=clipseg)
and install the Segments Python SDK. Then you can initialize the
Segments.ai Python client using an API key. This key can be found on
[the account page](https://segments.ai/account).

```python
!pip install -q segments-ai
```

```python
from segments import SegmentsClient
from getpass import getpass

api_key = getpass('Enter your API key: ')
segments_client = SegmentsClient(api_key)
```

Next, let\'s load an image from a dataset using the Segments client.
We\'ll use the [a2d2 self-driving
dataset](https://www.a2d2.audi/a2d2/en.html). You can also create your
own dataset by following [these
instructions](https://docs.segments.ai/tutorials/getting-started).

```python
samples = segments_client.get_samples("admin-tobias/clipseg")

# Use the last image as an example
sample = samples[1]
image = Image.open(requests.get(sample.attributes.image.url, stream=True).raw)
image
```

<figure class="image table text-center m-0 w-9/12">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/a0ca3accab5a40547f16b2abc05edd4558818bdf.png"></medium-zoom>
</figure>

We also need to get the category names from the dataset attributes.

```python
dataset = segments_client.get_dataset("admin-tobias/clipseg")
category_names = [category.name for category in dataset.task_attributes.categories]
```

Now we can use CLIPSeg on the image as before. This time, we\'ll also
scale up the outputs so that they match the input image\'s size.

```python
from torch import nn

inputs = processor(text=category_names, images=[image] * len(category_names), padding="max_length", return_tensors="pt")

# predict
with torch.no_grad():
  outputs = model(**inputs)

# resize the outputs
preds = nn.functional.interpolate(
    outputs.logits.unsqueeze(1),
    size=(image.size[1], image.size[0]),
    mode="bilinear"
)
```

And we can visualize the results again.

```python
len_cats = len(category_names)
_, ax = plt.subplots(1, len_cats + 1, figsize=(3*(len_cats + 1), 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len_cats)];
[ax[i+1].text(0, -15, category_name) for i, category_name in enumerate(category_names)];
```

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/7782da300097ce4dcb3891257db7cc97ccf1deb3.png"></medium-zoom>
</figure>

Now we have to combine the predictions to a single segmentated image.
We\'ll simply do this by taking the category with the greatest sigmoid
value for each patch. We\'ll also make sure that all the values under a
certain threshold do not count.

```python
threshold = 0.1

flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

# Initialize a dummy "unlabeled" mask with the threshold
flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

# Get the top mask index for each pixel
inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))
```

Let\'s quickly visualize the result.

```python
plt.imshow(inds)
```

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/b92dc12452108a0b2769ddfc1d7f79909e65144b.png"></medium-zoom>
</figure>

Lastly, we can upload the prediction to Segments.ai. To do that, we\'ll
first convert the bitmap to a png file, then we\'ll upload this file to
the Segments, and finally we\'ll add the label to the sample.

```python
from segments.utils import bitmap2file
import numpy as np

inds_np = inds.numpy().astype(np.uint32)
unique_inds = np.unique(inds_np).tolist()
f = bitmap2file(inds_np, is_segmentation_bitmap=True)

asset = segments_client.upload_asset(f, "clipseg_prediction.png")

attributes = {
      'format_version': '0.1',
      'annotations': [{"id": i, "category_id": i} for i in unique_inds if i != 0],
      'segmentation_bitmap': { 'url': asset.url },
  }

segments_client.add_label(sample.uuid, 'ground-truth', attributes)
```

If you take a look at the [uploaded prediction on
Segments.ai](https://segments.ai/admin-tobias/clipseg/samples/71a80d39-8cf3-4768-a097-e81e0b677517/ground-truth),
you can see that it\'s not perfect. However, you can manually correct
the biggest mistakes, and then you can use the corrected dataset to
train a better model than CLIPSeg.

<figure class="image table text-center m-0 w-9/12">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Colab output" src="assets/123_clipseg-zero-shot/segments-thumbs.png"></medium-zoom>
</figure>

## Conclusion

CLIPSeg is a zero-shot segmentation model that works with both text and image prompts. The model adds a decoder to CLIP and can segment almost anything. However, the output segmentation masks are still very low-res for now, so you’ll probably still want to fine-tune a different segmentation model.

If you’re interested in learning how to fine-tune a state-of-the-art segmentation model, check out our previous blog post: [https://huggingface.co/blog/fine-tune-segformer](https://huggingface.co/blog/fine-tune-segformer)