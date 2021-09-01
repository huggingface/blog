---
title: Fine tuning CLIP with Remote Sensing (Satellite) images and captions
thumbnail: /blog/assets/25_clip-rsicd/clip_schematic.png
---

# Fine tuning CLIP with Remote Sensing (Satellite) images and captions

_by Artashes Arutiunian (@arampacha), Dev Vidhani (@devvidhani), Goutham Venkatesh (@goutham794), Mayank Bhaskar (@cataluna84), Ritobrata Ghosh (@ghosh-r), and Sujit Pal (@sujitpal)_


In July this year, [Hugging Face](https://huggingface.co/) organized a [Flax/JAX Community Week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax), during which the community was invited to submit projects to train Hugging Face transformer models in the areas of Natural Language Processing (NLP) and Computer Vision (CV) on Tensor Processing Units (TPU) using [Flax](https://github.com/google/flax), a neural network library and ecosystem for [JAX](https://github.com/google/jax), a linear algebra library (like numpy) which can do automatic differentiation ([Autograd](https://github.com/hips/autograd)) and can compile down to [XLA](https://www.tensorflow.org/xla). [Google Cloud](https://cloud.google.com/) were co-sponsors of this event.

They organized 3 days of lectures during which speakers from Hugging Face and Google Cloud talked about TPUs, JAX, Flax, and how to use them to train Hugging Face transformer models (recordings for [day #1](https://www.youtube.com/watch?v=fuAyUQcVzTY), [day #2](https://www.youtube.com/watch?v=__eG63ZP_5g), and [day #3](https://www.youtube.com/watch?v=ZCMOPkcTu3s) available). This was followed by around 10 days of actual work, during which time Google Cloud provided each team with a GCP instance with a TPU. Each team was expected to train one or more Hugging Face models using JAX/Flax and share them with the community via their [model hub](https://huggingface.co/models). In addition, teams were asked to provide a demo [Hugging Face spaces](https://huggingface.co/spaces) showcasing the capabilities of their model. Approximately 100 teams participated in the event, and it resulted in 170 models and 36 demo spaces.

Our team, like probably many others, is a distributed one, spanning 12 time zones. Our common thread is that we all belong to the [TWIML Slack Channel](https://twimlai.slack.com/), where we came together based on a shared interest in Artificial Intelligence (AI) and Machine Learning (ML) topics. 

We decided that we would fine tune the [CLIP Network from OpenAI](https://openai.com/blog/clip/) with satellite images and captions from the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). The CLIP network learns visual concepts by being trained with image and caption pairs in a self-supervised manner, by using text paired with images found across the Internet. During inference, the model can predict the most relevant image given a text description, or the most relevant text description given an image. CLIP is powerful enough to be used in zero-shot manner on everyday images. However, we felt that satellite images were sufficiently different from everyday images, that it would be useful to fine-tune CLIP with them. Our intuition turned out to be correct, as the evaluation results (described below) shows.

We think of our project more as an "applied" project, in the sense that it is more about providing a useful service than about significantly advancing the technological frontier. Our model can be used in applications to search through large collections of satellite images using text, or automatically find specific features in such images. Both these features can be invaluable for "digital assistant" type applications ranging from domains such as military / police surveillance to helping deal with natural disasters and climate change -- really any application where humans have to search through enormous quantities of image data. In a broader sense, it demonstrates that CLIP can be similarly fine-tuned for other domains also, such as Radiography or Pathology. You can read about our project on our [project page](https://github.com/arampacha/CLIP-rsicd), download our [trained model](https://huggingface.co/flax-community/clip-rsicd-v2) to use for inference on your own data, or see it in action on our [demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo).

Our project ended up placing third at the event, which was a very pleasant surprise. We are incredibly humbled and gratified that the judges saw the potential of our model. We are very grateful to the organizers for introducing us to Flax/JAX and for providing the resources to create and showcase our model. We are also very thankful to the other participants and organizers for sharing their knowledge and insights so freely, we have each benefited enormously from these interactions. 

In this post, we describe details of our training and evaluation process, and our plans for future work on this project.

## Training

### Dataset

Our model was fine-tuned primarily using the RSICD dataset, which is a set of about 10,000 images collected from Google Earth, Baidu Map, MapABC and Tianditu, and provided freely to the research community for advancement of remote sensing captioning via [Exploring Models and Data for Remote Sensing Image Caption Generation](https://arxiv.org/abs/1712.07835) (Lu et al, 2017). The images are provided as (224, 224) RGB images at various resolutions. Each image has up to 5 captions associated with it.

In addition, we used the [UCM Dataset](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA) and the [Sydney dataset](https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ) for training, The UCM dataset is based on the UC Merced Land Use dataset. It consists of 2100 images belonging to 21 classes (100 images per class). The dataset provides 5 captions for each image. The Sydney dataset contains images of Sydney, Australia from Google Earth. It contains 613 images belonging to 7 classes. Images are (500, 500) RGB, and provides 5 captions for each image.

### Model

Our model is just the fine tuned version of the original CLIP model shown below. Inputs to the model are a batch of captions and a batch of images passed through a text encoder and image encoder respectively. The training process uses contrastive learning to push closer images and their respective captions and to push apart images and captions for other images. In this way, images and captions that belong together get pushed closer together.

<img src="/blog/assets/25_clip_rsicd/clip_schematic.png"/>
<center><i>CLIP Training and Inference (Image Credit: CLIP: Connecting Text and Images (https://openai.com/blog/clip/))</i></center>

### Data Augmentation

Because our dataset was not very large, we used both image augmentation and text augmentation in an effort to regularize our dataset and prevent overfitting.

Image augmentation was done inline using built-in transforms from Pytorch's Torchvision package. The transformations used were Random Cropping, Random Resizing and Cropping, Color Jitter, and Random Horizontal and Vertical flipping.

Text augmentation was done, using backtranslation, to generate captions for images that had less than 5 unique captions per image. We used the [Marian MT family of models from Hugging Face](https://huggingface.co/transformers/model_doc/marian.html) to translate the existing captions into French, Spanish, Italian and Portugese and back to English to fill out the captions for these images.

As shown in these loss plots below, image augmentation reduced overfitting significantly, and text and image augmentation reduced overfitting even further.

<img src="/blog/assets/25_clip_rsicd/image-augment-loss.png"/>
<img src="/blog/assets/25_clip_rsicd/image-text-aug-loss.png"/>
<center><i>Evaluation and Training loss plots comparing (top) no augmentation vs image augmentation, and (bottom) image augmentation vs text+image augmentation</i></center>

## Evaluation

### Metrics

For the evaluation, we used a subset of the RSICD test set where image file names contained the category the image belonged to. We found 30 categories of images in this subset. Evaluation was done by comparing each image with a set of 30 caption sentences of the form `"An aerial photograph of {category}"`. The model produced a ranked list of the 30 captions, from most relevant to least relevant. Categories corresponding to captions with the top k scores (for k=1, 3, 5, and 10) were compared with the "label" category as indicated in the image name. The scores are averaged over the entire set of images used for evaluation and reported for various values of k, as shown below.

The `baseline` model represents the pre-trained `openai/clip-vit-base-path32` CLIP model. This model was fine tuned with captions and images from the RSICD dataset, which resulted in a significant performance boost, as shown below.

Our best model was trained with image and text augmentation, with batch size 1024 (128 spread across 8 TPU devices), and the Adam optimizer with learning rate 5e-6. Our second base model was trained with the same hyperparameters, except that we used the Adafactor optimizer with learning rate 1e-4. You can download either model from their model cards linked to in the table below.

| Model-name                               | k=1   | k=3   | k=5   | k=10  |
| ---------------------------------------- | ----- | ----- | ----- | ----- |
| baseline                                 | 0.572 | 0.745 | 0.837 | 0.939 |
| bs128x8-lr1e-4-augs/ckpt-2               | 0.819 | 0.950 | 0.974 | 0.994 |
| bs128x8-lr1e-4-imgaugs/ckpt-2            | 0.812 | 0.942 | 0.970 | 0.991 |
| [bs128x8-lr1e-4-imgaugs-textaugs/ckpt-4](https://huggingface.co/flax-community/clip-rsicd)<sup>2</sup>   | 0.843 | 0.958 | 0.977 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs/ckpt-8   | 0.831 | 0.959 | 0.977 | 0.994 |
| bs128x8-lr5e-5-imgaugs/ckpt-4            | 0.746 | 0.906 | 0.956 | 0.989 |
| bs128x8-lr5e-5-imgaugs-textaugs-2/ckpt-4 | 0.811 | 0.945 | 0.972 | 0.993 |
| bs128x8-lr5e-5-imgaugs-textaugs-3/ckpt-5 | 0.823 | 0.946 | 0.971 | 0.992 |
| bs128x8-lr5e-5-wd02/ckpt-4               | 0.820 | 0.946 | 0.965 | 0.990 |
| [bs128x8-lr5e-6-adam/ckpt-1](https://huggingface.co/flax-community/clip-rsicd-v2)<sup>1</sup> | **0.883** | **0.968** | **0.982** | **0.998** |


_1 - our best model, 2 - our second best model_

### Demo

You can access our [CLIP-RSICD Demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo) here. It uses our fine-tuned CLIP model to provide the following functionality. 

* Text to Image search
* Image to Image search
* Find text feature in image

For the first two features, the "image corpus" is composed of images from the RSICD test set. They are encoded using our best fine-tuned CLIP model and stored in a NMSLib index which allows Approximate Nearest Neighbor based retrieval. For text-to-image and image-to-image search respectively, the query text or image are encoded with our model and matched against the image vectors in the corpus. For the third service, we partition the incoming image into patches and encode them, then encode the text feature being looked for, and match the text vector with each of the image patch vectors, and return the probability of finding the feature in each patch.

## Future Work

We are grateful that we have been given an opportunity to further refine our model. Some ideas we have for future work are as follows:

1. Construct a sequence to sequence model using a CLIP encoder and a GPT-3 decoder and train it to caption images.
2. Fine tune the model on more image caption pairs from other datasets and investigate if we can improve its performance.
3. Investigate how fine-tuning affects the performance of model on non-RSICD image caption pairs.
4. Investigate the capability of the fine-tuned model to classify outside the categories it has been fine tuned on.
5. Evaluate the model using other criteria such as image classification.





