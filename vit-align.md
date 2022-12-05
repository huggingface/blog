---
title: "New and Open-Source ViT and ALIGN from Kakao Brain" 
thumbnail: /blog/assets/120_vit_align/thumbnail.png
---

# New and Open-Source ViT and ALIGN from Kakao Brain

<div class="blog-metadata">
    <small>Published Dec 05, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/vit-align.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/unso"> 
        <img class="avatar avatar-user" src="https://scholar.googleusercontent.com/citations?view_op=medium_photo&user=-I2EZeEAAAAJ&citpid=8" title="Gravatar">
        <div class="bfc">
            <code>unso</code>
            <span class="fullname">Unso Jo</span>
        </div>
    </a>
</div>



# Kakao Brain’s Open Source ViT, ALIGN, and the new COYO text-image dataset

Kakao Brain and Hugging Face are excited to release a new open-source image-text dataset [COYO](coyo github link) of 700 million pairs and two new visual language models trained on it, [ViT](hub link) and [ALIGN](hub link). Kakao brain’s initiative sets itself apart from Google's ViT and ALIGN releases in its dedication to open-sourcing the dataset along with the models. In particular, this is the first time ever the ALIGN model is made public for free and open-source use. Google’s [ViT](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) and [ALIGN](https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html) models, while trained on huge datasets (ViT trained on 300 million images and ALIGN trained on 1.8 billion image-text pairs respectively), cannot be replicated because the datasets are not public. Kakao Brain’s ViT and ALIGN models follow the same architecture and hyperparamters as provided in the original [papers](google papers) respective Google models but are trained on the open source COYO dataset. This contribution is particularly valuable to researchers who want to reproduce visual language modeling with access to the data as well. This blog will introduce the new COYO dataset, Kakao Brain's ViT and ALIGN models, and how to use them in your work. More detailed information on the Kakao ViT and ALIGN models can be found [here](github vit link) and [here](github align link). 

Here are the main takeaways about our release:

* First open-source ALIGN model
* First open ViT and ALIGN models that have been trained on an open-source dataset COYO
* Both ViT and ALIGN models perform on-par with the Google evaluation reports

## Performance Comparison

Kakao Brain's released ViT and ALIGN models perform on par and sometimes better than what Google has reported. 
Kakao Brain's ALIGN-B7-Base model, while trained on a much fewer pairs (700 million pairs vs 1.8 billion), performs on par with Google's ALIGN-B7-Base on the Image KNN classification task and better on MsCOCO retrieval image-to-text, text-to-image tasks. Kakao Brain's ViT-L/16 performs similarly to Google's ViT-L/16 when evaluated on ImageNet and ImageNet-ReaL at model resolution 384 and 512. This means the community can use Kakao Brain's ViT and ALIGN models to replicate Google's ViT and ALIGN releases especially when users require access to the training data.


<img src="assets/120_vit_align/align.png" alt="align performance" width="500"/>
<img src="assets/120_vit_align/vit.png" alt="vit performance" width="500"/>



<!-- 
| Model  | Upstream Dataset | Resolution | ImageNet (downstream)| ImageNet-ReaL | Public | 
|:---------: | :--------------: | :------------:| :----: | :----: | :-----:|
| Google ViT-L/16 | JFT-300M   | 512   |  87.76 | 90.54 | X | 
| Kakao Brain ViT-L/16 | COYO-Labeled-300M| 512 | 87.24 (-0.52) | 90.03 (-0.51) | O | 
| Google ViT-L/16| JFT-300M     | 384     | 87.12 | 89.99  | X |
| Kakao Brain ViT-L/16 | COYO-Labeled-300M  | 384  | 86.72 (-0.40) | 89.84 (-0.15) | O |



|                                    |    Dataset    | ImageNet |      MsCOCO  |          |
|------------------------------------|:-------------:|:--------:|:--------:|:--------:|
|                                    |               |   KNN    |   I2T R@1  | T2I R@1  |
| Google ALIGN-B7-Base        |  ALIGN 1.8B   |   69.3   |      55.4   |   41.7   |
| **Kakao Brain COYO-ALIGN-B7-Base** | **COYO-700M** | **68.6** |  **61.2** | **43.1** |



REPLACE WITH GRAPHS -->





## COYO DATASET

COYO is an image-text dataset of 700 million pairs similar to Google's ALIGN 1.8B image-text dataset which is a collection of "noisy" alt-text and image pairs from webpages, but open-source. COYO and ALIGN 1.8B are "noisy" because minimal filtering was applied. COYO is similar to the other open-source image/text dataset, LAION but with the following differences. While LAION 2B is a much larger dataset of 2 billion English pairs, compared to COYO’s 700 million pairs, COYO pairs come with more metadata that give users more flexibility and finer-grained control over usage. The following table shows the differences: COYO comes equipped with aesthetic scores for all pairs, more robust watermark scores, and face count data. 


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


note: COYO Dataset example: images with caption
note: ViT demo examples and short description

<!-- 
| id            | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; url &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | text                                                                                                                                                                               | width | height | image_phash      | text_length | word_count | num_tokens_bert | num_tokens_gpt | num_faces | clip_similarity_vitb32 | clip_similarity_vitl14 | nsfw_score_opennsfw2 | nsfw_score_gantman | watermark_score | aesthetic_score_laion_v2 |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|--------|------------------|-------------|------------|-----------------|----------------|-----------|------------------------|------------------------|----------------------|--------------------|-----------------|--------------------------|
| 4896263451343 | <img src="https://cdn.shopify.com/s/files/1/0190/8574/products/Art_Riley-Monterey_Fishing_Fleet_1_grande.jpg?v=1479962684" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Fishing Fleet (Monterey), California art by Art Riley. HD giclee art prints for sale at CaliforniaWatercolor.com - original California paintings, & premium giclee prints for sale | 600   | 447    | bac58374982e0fc7 | 178         | 25         | 39              | 40             | 0         | 0.319336               | 0.248169               | 2.54512e-05          | 0.0293861          | 0.0406009       | 7.04812                  |
| 1425929344479 | <img src="https://www.ephotozine.com/resize/2018/07/xlrg/121543_1518912380.jpg?RTUdGk5cXyJFBQgJVANtcQlnYF8JERFaGwJRNQh6SlYUAEw1cmsCdg1hAWoxXFNGLSI=" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | The Gate by Pete2453                                                                                                                                                               | 600   | 347    | 8374726575bc0f8a | 20          | 4          | 6               | 6              | 0         | 0.24939                | 0.203735               | 6.97374e-06          | 0.00823276         | 0.0721415       | 6.98521                  |
| 7456063527931 | <img src="https://www.boredart.com//wp-content/uploads/2014/06/Beautiful-Pictures-From-the-Shores-of-the-Mythical-Land-421.jpg" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Beautiful Pictures From the Shores of the Mythical Land (42                                                                                                                        | 600   | 320    | 949d1fe559e2cc90 | 59          | 10         | 11              | 14             | 0         | 0.290771               | 0.179321               | 0.0130615            | 0.0178628          | 0.489642        | 6.94643                  |
| 3221225511175 | <img src="https://homesfeed.com/wp-content/uploads/2017/12/contemporary-expensive-lighting-fixtures-with-minimum-lighting.jpg" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | contemporary expensive lighting fixtures with minimum lighting                                                                                                                     | 800   | 499    | e5ea35075ab912c6 | 62          | 7          | 7               | 8              | 0         | 0.263916               | 0.217896               | 0.000990868          | 0.0137114          | 0.0960748       | 4.57594                  |
| 5626407855002 | <img src="https://api.time.com/wp-content/uploads/2015/03/168951187.jpg" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Nintendo Co.'s Super Mario is displayed on coffee mugs for sale at the Nintendo World store in New York, U.S., on Friday, May 17, 2013.                                            | 2000  | 1309   | 9311891e9437f4f3 | 135         | 27         | 37              | 35             | 0         | 0.400878               | 0.316650               | 0.00362968           | 0.0317519          | 0.0022693       | 6.324910                 |
| 1125282207474 | <img src="https://s.yimg.com/ny/api/res/1.2/mOZe9uKtwugmPrqeXBlxFg--/YXBwaWQ9aGlnaGxhbmRlcjt3PTk2MDtoPTYzMA--/https://s.yimg.com/uu/api/res/1.2/JuTSVK74cI8II09Q75uzGA--~B/aD01MjU7dz04MDA7YXBwaWQ9eXRhY2h5b24-/https://media.zenfs.com/en/reuters.com/15941d3b47960da80f8033f4ddf9da64" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | FILE PHOTO: A rainbow appears on the Auckland skyline featuring Sky Tower in New Zealand                                                                                           | 800   | 525    | 85b89c0166ee63be | 88          | 15         | 16              | 16             | 0         | 0.4453125              | 0.3505859              | 2.640485e-05         | 0.012074           | 0.0219129       | 5.294523                 |
| 1434519186493 | <img src="https://static.straitstimes.com.sg/s3fs-public/styles/article_pictrure_780x520_/public/articles/2013/07/24/CHINA24e_2x.jpg?itok=6ppRPBJs&timestamp=1436931188" width="400" />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | A man covers himself with algae as he poses for photographs on a beach in Qingdao, Shandong province on Tuesday, July 23, 2013. -- FILE PHOTO: REUTERS                             | 860   | 573    | f2c48dabbf93810a | 150         | 26         | 35              | 36             | 7         | 0.4165039              | 0.3427734              | 0.025009             | 0.01608            | 0.072775        | 6.833739                 |




|----------|  -------- | 
|  **id**  |   4896263451343   |
| **url** | <img src="https://cdn.shopify.com/s/files/1/0190/8574/products/Art_Riley-Monterey_Fishing_Fleet_1_grande.jpg?v=1479962684" width="400" /> | 
| **width**  |    3        |
| **height** |     3       | 
 -->



<table>
  <tr>
    <th>id</th>
    <td>4896263451343</td>
    <td>5626407855002</td>
    <td>1125282207474</td>
  </tr>
  <tr>
    <th>url</th>
    <td>
        <img src="https://cdn.shopify.com/s/files/1/0190/8574/products/Art_Riley-Monterey_Fishing_Fleet_1_grande.jpg?v=1479962684" width="400" />
    </td>
    <td>
        <img src="https://api.time.com/wp-content/uploads/2015/03/168951187.jpg" width="400" />
    </td>
    <td>
        <img src="https://s.yimg.com/ny/api/res/1.2/mOZe9uKtwugmPrqeXBlxFg--/YXBwaWQ9aGlnaGxhbmRlcjt3PTk2MDtoPTYzMA--/https://s.yimg.com/uu/api/res/1.2/JuTSVK74cI8II09Q75uzGA--~B/aD01MjU7dz04MDA7YXBwaWQ9eXRhY2h5b24-/https://media.zenfs.com/en/reuters.com/15941d3b47960da80f8033f4ddf9da64" width="400" />
    </td>

  </tr>
  <tr>
    <th>text</th>
    <td width="100">
        Fishing Fleet (Monterey), California art by Art Riley. 
        HD gicleeart prints for sale at CaliforniaWatercolor.com 
        - original California paintings, & premium giclee prints for sale
    </td>
    <td width="100">
      Nintendo Co.'s Super Mario is displayed on coffee mugs for sale at the Nintendo World store in New York, U.S., on Friday, May 17, 2013.
    </td>
    <td width="100">
      FILE PHOTO: A rainbow appears on the Auckland skyline featuring Sky Tower in New Zealand
    </td>

  </tr>
  <tr>
    <th>image_phash</th>
    <td>   </td>

  </tr>




</table>

image_phash      | text_length | word_count | num_tokens_bert | num_tokens_gpt | num_faces | clip_similarity_vitb32 | clip_similarity_vitl14 | nsfw_score_opennsfw2 | nsfw_score_gantman | watermark_score | aesthetic_score_laion_v2 |


## How ViT and ALIGN work

ViT -- Vision Transformer -- is a vision model [proposed by Google in 2020](link) that resembles the text Transformer architecture. 
It is a new approach to vision, distinct from convolutional neural nets (CNNs) that have dominated vision tasks since 2012's AlexNet. It is upto four times more computationally efficient than similarly performing CNNs and domain agnostic. ViT takes as input an image which is broken up into a sequence of image patches - just as the text Transformer takes as input a sequence of text -  and given position embeddings to each patch to learn the image structure. ViT performance is notable in particular for having an excellent performance-compute trade-off. While some of Google's ViT models are open-source, the JFT300 million image-label pair dataset they were trained on has not been released publicly. While Kakao Brain's trained and released ViT model performs similarly on various tasks, its code, model, and training data are made entirely public for reproducibility and open science.

<img src="assets/120_vit_align/vit_demo.png" alt="vit performance" width="900"/>



 [Google then introduced ALIGN](link) -- a Large-scale Image and Noisy Text Embedding model in 2021 -- a visual-language model trained on "noisy" text-image data for various vision and cross-modal tasks such as text-image retrieval. ALIGN has a simple dual-encoder architecture trained on image and text pairs, learned via a contrastive loss function. ALIGN's "noisy" training corpus is notable for balancing scale and robustness. Previously, visual language representational learning had been trained on large-scale datasets with manual labels, which require extensive preprocessing. ALIGN's corpus uses the image alt-text data, text that appears when the image fails to load, as the caption to the image -- resulting in an inevitably noisy, but much larger (1.8 billion pair) dataset that allows ALIGN to perform at SoTA levels on various tasks. Kakao Brain's ALIGN is the first open-source version of this model, trained on the COYO dataset and performs better than Google's reported results.





## How to use the COYO dataset

```python
from datasets import load_dataset
```

## how to use Vit and align from the hub



