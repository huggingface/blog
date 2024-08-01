---
title: "Introducing TextImage Augmentation for Document Images" 
thumbnail: "/blog/assets/185_albumentations/thumbnail.png"
authors:
- user: danaaubakirova
- user: Molbap
- user: Ternaus
guest: True
---
# Introducing Multimodal TextImage Augmentation for Document Images

In this blog post, we provide a tutorial on how to use a new data augmentation technique for document images, developed in collaboration with Albumentations AI.

## Motivation
Vision Language Models (VLMs) have an immense range of applications, but they often need to be fine-tuned to specific use-cases, particularly for datasets containing document images, i.e., images with high textual content. In these cases, it is crucial for text and image to interact with each other at all stages of model training, and applying augmentation to both modalities ensures this interaction. Essentially, we want a model to learn to read properly, which is challenging in the most common cases where data is missing.

Hence, the need for **effective data augmentation** techniques for document images became evident when addressing challenges in fine-tuning models with limited datasets. A common concern is that typical image transformations, such as resizing, blurring, or changing background colors, can negatively impact text extraction accuracy.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/po85g2Nu4-d2eHqJ0PMt4.png)

We recognized the need for data augmentation techniques that preserve the integrity of the text while augmenting the dataset. Such data augmentation can facilitate generation of new documents or modification of existing ones, while preserving their text quality.

## Introduction

To address this need, we introduce a **new data augmentation pipeline** developed in collaboration with [Albumentations AI](https://albumentations.ai). This pipeline handles both images and text within them, providing a comprehensive solution for document images. This class of data augmentation is *multimodal* as it modifies both the image content and the text annotations simultaneously.

As discussed in a previous [blog post](https://huggingface.co/blog/danaaubakirova/doc-augmentation), our goal is to test the hypothesis that integrating augmentations on both text and images during pretraining of VLMs is effective. Detailed parameters and use case illustrations can be found on the  [Albumentations AI Documentation](https://albumentations.ai/docs/examples/example_textimage/?h=textimage). Albumentations AI enables the dynamic design of these augmentations and their integration with other types of augmentations.

# Method 

To augment document images, we begin by randomly selecting lines within the document. A hyperparameter `fraction_range` controls the bounding box fraction to be modified.

Next, we apply one of several text augmentation methods to the corresponding lines of text, which are commonly utilized in text generation tasks. These methods include Random Insertion, Deletion, and Swap, and Stopword Replacement. 

After modifying the text, we black out parts of the image where the text is inserted and inpaint them, using the original bounding box size as a proxy for the new text's font size. The font size can be specified with the parameter `font_size_fraction_range`, which determines the range for selecting the font size as a fraction of the bounding box height. Note that the modified text and corresponding bounding box can be retrieved and used for training. This process results in a dataset with semantically similar textual content and visually distorted images.


## Main Features of the TextImage Augmentation

The library can be used for two main purposes:

1. **Inserting any text on the image**: This feature allows you to overlay text on document images, effectively generating synthetic data. By using any random image as a background and rendering completely new text, you can create diverse training samples. A similar technique, called SynthDOG, was introduced in the [OCR-free document understanding transformer](https://arxiv.org/pdf/2111.15664).

2. **Inserting augmented text on the image**: This includes the following text augmentations:
   - **Random deletion**: Randomly removes words from the text.
   - **Random swapping**: Swaps words within the text.
   - **Stop words insertion**: Inserts common stop words into the text.

Combining these augmentations with other image transformations from Albumentations allows for simultaneous modification of images and text. You can retrieve the augmented text as well.


## Installation


```python
from __future__ import annotations
```


```python
%load_ext autoreload
%autoreload 2
```


```python
!pip install -U pillow
```

    Requirement already satisfied: pillow in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (10.4.0)



```python
!pip install albumentations
```

    Requirement already satisfied: albumentations in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (1.4.12)
    Requirement already satisfied: numpy>=1.24.4 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (1.26.4)
    Requirement already satisfied: scipy>=1.10.0 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (1.13.0)
    Requirement already satisfied: scikit-image>=0.21.0 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (0.22.0)
    Requirement already satisfied: PyYAML in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (6.0.1)
    Requirement already satisfied: typing-extensions>=4.9.0 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (4.10.0)
    Requirement already satisfied: pydantic>=2.7.0 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (2.8.2)
    Requirement already satisfied: albucore>=0.0.13 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (0.0.13)
    Requirement already satisfied: eval-type-backport in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (0.2.0)
    Requirement already satisfied: opencv-python-headless>=4.9.0.80 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albumentations) (4.9.0.80)
    Requirement already satisfied: tomli>=2.0.1 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from albucore>=0.0.13->albumentations) (2.0.1)
    Requirement already satisfied: annotated-types>=0.4.0 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from pydantic>=2.7.0->albumentations) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from pydantic>=2.7.0->albumentations) (2.20.1)
    Requirement already satisfied: networkx>=2.8 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (3.1)
    Requirement already satisfied: pillow>=9.0.1 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (10.4.0)
    Requirement already satisfied: imageio>=2.27 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (2.33.1)
    Requirement already satisfied: tifffile>=2022.8.12 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (2023.4.12)
    Requirement already satisfied: packaging>=21 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (24.0)
    Requirement already satisfied: lazy_loader>=0.3 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from scikit-image>=0.21.0->albumentations) (0.3)



```python
import albumentations as A
import cv2
from matplotlib import pyplot as plt
import json
```

## Visualization


```python
def visualize(image):
    plt.figure(figsize=(20, 15))
    plt.axis('off')
    plt.imshow(image)
```

## Load data

Note that for this type of augmentation you can use the [IDL](https://huggingface.co/datasets/pixparse/idl-wds) and [PDFA](https://huggingface.co/datasets/pixparse/pdfa-eng-wds) datasets. They provide the bounding boxes of the lines that you want to modify. 
For this tutorial, we will focus on the sample from IDL dataset. 


```python
bgr_image = cv2.imread("examples/original/fkhy0236.tif")
image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
```


```python
with open("examples/original/fkhy0236.json") as f:
    labels = json.load(f)
```


```python
font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
```


```python
visualize(image)
```


    
![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/g3lYRSdMBazALttw7wDJ2.png)
    

We need to correctly preprocess the data, as the input format for the bounding boxes is the normalized Pascal VOC. Hence, we build the metadata as follows: 


```python
page = labels['pages'][0]

def prepare_metadata(page: dict, image_height: int, image_width: int) -> list:
    metadata = []

    for text, box in zip(page['text'], page['bbox']):
        left, top, width_norm, height_norm = box

        metadata.append({
            "bbox": [left, top, left + width_norm, top + height_norm],
            "text": text
        })
    
    return metadata
```


```python
image_height, image_width = image.shape[:2]
metadata = prepare_metadata(page, image_height, image_width)
```

## Random Swap


```python
transform = A.Compose([A.TextImage(font_path=font_path, p=1, augmentations=["swap"], clear_bg=True, font_color = 'red', fraction_range = (0.5,0.8), font_size_fraction_range=(0.8, 0.9))])
```


```python
transformed = transform(image=image, textimage_metadata=metadata)
```


```python
visualize(transformed["image"])
```


![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/k06LJuPRSRHGeGnpCj3XP.png)


## Random Deletion


```python
transform = A.Compose([A.TextImage(font_path=font_path, p=1, augmentations=["deletion"], clear_bg=True, font_color = 'red', fraction_range = (0.5,0.8), font_size_fraction_range=(0.8, 0.9))])
```


```python
transformed = transform(image=image, textimage_metadata=metadata)
```


```python
visualize(transformed['image'])
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/3Z_L4GTZMT5tvBYJSMOha.png)
    

## Random Insertion


```python
!pip install nltk
```

    Requirement already satisfied: nltk in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (3.8.1)
    Requirement already satisfied: click in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from nltk) (8.1.7)
    Requirement already satisfied: joblib in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from nltk) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from nltk) (2023.12.25)
    Requirement already satisfied: tqdm in /fsx/dana_aubakirova/miniconda/envs/myenv/lib/python3.11/site-packages (from nltk) (4.65.0)



```python
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /admin/home/dana_aubakirova/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
stops = stopwords.words('english')
```


```python
transform = A.Compose([A.TextImage(font_path=font_path, p=1, augmentations=["insertion"], stopwords = stops, clear_bg=True, font_color = 'red', fraction_range = (0.5,0.8), font_size_fraction_range=(0.8, 0.9))])
```


```python
transformed = transform(image=image, textimage_metadata=metadata)
```


```python
visualize(transformed['image'])
```


    
![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/QZKZP_VEzFhEV5GhykRlP.png)
    


## Can we combine with other transformations?

Let's define a complex transformation pipeline using `A.Compose`, which includes text insertion with specified font properties and stopwords, Planckian jitter, and affine transformations. Firstly, with `A.TextImage` we insert text into the image using specified font properties, with a clear background and red font color. The fraction and size of the text to be inserted are also specified.
Then with `A.PlanckianJitter` we alter the color balance of the image. Finally, using `A.Affine` we apply affine transformations, which can include scaling, rotating, and translating the image.

```python
transform_complex = A.Compose([A.TextImage(font_path=font_path, p=1, augmentations=["insertion"], stopwords = stops, clear_bg=True, font_color = 'red', fraction_range = (0.5,0.8), font_size_fraction_range=(0.8, 0.9)),
                               A.PlanckianJitter(p=1),
                               A.Affine(p=1)
                              ])
```


```python
transformed = transform_complex(image=image, textimage_metadata=metadata)
```


```python
visualize(transformed["image"])
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/-mDto1DdKHJXmzG2j9RzR.png)

# How to get the altered text? 

To extract the information on the bounding box indices where text was altered, along with the corresponding transformed text data
run the following cell. This data can be used effectively for training models to recognize and process text changes in images.


```python
transformed['overlay_data']
```




    [{'bbox_coords': (375, 1149, 2174, 1196),
      'text': "Lionberger, Ph.D., (Title: if Introduction to won i FDA's yourselves Draft Guidance once of the wasn't General Principles",
      'original_text': "Lionberger, Ph.D., (Title: Introduction to FDA's Draft Guidance of the General Principles",
      'bbox_index': 12,
      'font_color': 'red'},
     {'bbox_coords': (373, 1677, 2174, 1724),
      'text': "After off needn't were a brief break, ADC member mustn Jeffrey that Dayno, MD, Chief Medical Officer for at their Egalet",
      'original_text': 'After a brief break, ADC member Jeffrey Dayno, MD, Chief Medical Officer at Egalet',
      'bbox_index': 19,
      'font_color': 'red'},
     {'bbox_coords': (525, 2109, 2172, 2156),
      'text': 'll Brands recognize the has importance and of a generics ADF guidance to ensure which after',
      'original_text': 'Brands recognize the importance of a generics ADF guidance to ensure',
      'bbox_index': 23,
      'font_color': 'red'},
     {'bbox_coords': (526, 2772, 2173, 2820),
      'text': "couldn't potential routes of abuse. our Need for i will between a themselves broader, more flexible approach to generic",
      'original_text': 'potential routes of abuse. Need for a broader, more flexible approach to generic',
      'bbox_index': 31,
      'font_color': 'red'},
     {'bbox_coords': (367, 1214, 1972, 1262),
      'text': 'whom for than too Evaluation of Abuse yours Deterrence of both Generic Solid for Oral Opioid Drug Products).',
      'original_text': 'for Evaluation of Abuse Deterrence of Generic Solid Oral Opioid Drug Products).',
      'bbox_index': 13,
      'font_color': 'red'},
     {'bbox_coords': (2022, 261, 2452, 305),
      'text': 'by Washington, DC 20004',
      'original_text': 'Washington, DC 20004',
      'bbox_index': 6,
      'font_color': 'red'},
     {'bbox_coords': (523, 2837, 1912, 2885),
      'text': "ADF guidance doesn't shouldn't and product specific itself guidance ourselves that for generic ADF opioids.",
      'original_text': 'ADF guidance and product specific guidance for generic ADF opioids.',
      'bbox_index': 32,
      'font_color': 'red'},
     {'bbox_coords': (373, 1808, 877, 1846),
      'text': 'Generics was ADF Guidance."',
      'original_text': 'Generics ADF Guidance."',
      'bbox_index': 21,
      'font_color': 'red'},
     {'bbox_coords': (232, 161, 543, 219),
      'text': 'Deterrent only',
      'original_text': 'Deterrent',
      'bbox_index': 1,
      'font_color': 'red'},
     {'bbox_coords': (524, 2174, 2171, 2221),
      'text': "widespread access to you'll safe such shan and effective analgesics for wouldn other patients it who need them.",
      'original_text': 'widespread access to safe and effective analgesics for patients who need them.',
      'bbox_index': 24,
      'font_color': 'red'},
     {'bbox_coords': (373, 2006, 1158, 2054),
      'text': "hadn A aren't few ve highlights from his presentation:",
      'original_text': 'A few highlights from his presentation:',
      'bbox_index': 22,
      'font_color': 'red'},
     {'bbox_coords': (375, 1579, 523, 1623),
      'text': "didn't impact.",
      'original_text': 'impact.',
      'bbox_index': 18,
      'font_color': 'red'},
     {'bbox_coords': (235, 90, 450, 152),
      'text': 'Abuse with',
      'original_text': 'Abuse',
      'bbox_index': 0,
      'font_color': 'red'},
     {'bbox_coords': (374, 1082, 2175, 1131),
      'text': 's Pre Market Evaluation of Abuse-Deterrent Properties from of Opioid Drug at Products) are at and Robert them',
      'original_text': 'Pre Market Evaluation of Abuse-Deterrent Properties of Opioid Drug Products) and Robert',
      'bbox_index': 11,
      'font_color': 'red'},
     {'bbox_coords': (2087, 3304, 2468, 3339),
      'text': 'MNKOI 0001045819 its',
      'original_text': 'MNKOI 0001045819',
      'bbox_index': 34,
      'font_color': 'red'},
     {'bbox_coords': (525, 2640, 2172, 2689),
      'text': "to In its current form, the guidance does not they hers this adequately address what their is needed to y didn't",
      'original_text': 'In its current form, the guidance does not adequately address what is needed to',
      'bbox_index': 29,
      'font_color': 'red'},
     {'bbox_coords': (375, 1741, 2171, 1789),
      'text': 'Corporation delivered a presentation entitled "Brand these Industry Perspective nor then on the if d',
      'original_text': 'Corporation delivered a presentation entitled "Brand Industry Perspective on the',
      'bbox_index': 20,
      'font_color': 'red'},
     {'bbox_coords': (562, 583, 1987, 631),
      'text': 'PUBLIC MEETING ON PRE-MARKET is hasn EVALUATION OF than',
      'original_text': 'PUBLIC MEETING ON PRE-MARKET EVALUATION OF',
      'bbox_index': 7,
      'font_color': 'red'},
     {'bbox_coords': (525, 2375, 2173, 2421),
      'text': "Emphasized that it couldn don't won't is critical that the she generic shouldn product is weren no less abuse-deterrent",
      'original_text': 'Emphasized that it is critical that the generic product is no less abuse-deterrent',
      'bbox_index': 26,
      'font_color': 'red'}]



## Synthetic Data Generation

This augmentation method can be extended to the generation of synthetic data, as it enables the rendering of text on any background or template.

```python
template = cv2.imread('template.png')
image_template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
```


```python
transform = A.Compose([A.TextImage(font_path=font_path, p=1, clear_bg=True, font_color = 'red', font_size_fraction_range=(0.5, 0.7))])
```


```python
metadata = [{
    "bbox": [0.1, 0.4, 0.5, 0.48],
    "text": "Some smart text goes here.",
}, {
    "bbox": [0.1, 0.5, 0.5, 0.58],
    "text": "Hope you find it helpful.",
}]
```


```python
transformed = transform(image=image_template, textimage_metadata=metadata)
```


```python
visualize(transformed['image'])
```
   

![image/png](https://cdn-uploads.huggingface.co/production/uploads/640e21ef3c82bd463ee5a76d/guKKPs5P0-g8nX4XSGcLe.png)

    



```python

```
## Conclusion

In collaboration with Albumentations AI, we introduced TextImage Augmentation, a multimodal technique that modifies document images while along with the text. By combining text augmentations such as Random Insertion, Deletion, Swap, and Stopword Replacement with image modifications, this pipeline allows for the generation of diverse training samples. 

For detailed parameters and use case illustrations, refer to the [Albumentations AI Documentation](https://albumentations.ai/docs/examples/example_textimage/?h=textimage). We hope you find these augmentations useful for enhancing your document image processing workflows.

## References

```
@inproceedings{kim2022ocr,
  title={Ocr-free document understanding transformer},
  author={Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle={European Conference on Computer Vision},
  pages={498--517},
  year={2022},
  organization={Springer}
}
```
