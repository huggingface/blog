---
title: Fine tuning CLIP with Remote Sensing (Satellite) images and captions
thumbnail: /blog/assets/30_clip_rsicd/clip_schematic.png
authors:
- user: arampacha
  guest: true
- user: devv
  guest: true
- user: goutham794
  guest: true
- user: cataluna84
  guest: true
- user: ghosh-r
  guest: true
- user: sujitpal
  guest: true
---

# Fine tuning CLIP with Remote Sensing (Satellite) images and captions

{blog_metadata}
{authors}


## Fine tuning CLIP with Remote Sensing (Satellite) images and captions

<img src="/blog/assets/30_clip_rsicd/clip-rsicd-header-image.png"/>

In July this year, [Hugging Face](https://huggingface.co/) organized a [Flax/JAX Community Week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md), and invited the community to submit projects to train Hugging Face [transformers](https://github.com/huggingface/transformers) models in the areas of Natural Language Processing (NLP) and Computer Vision (CV).

Participants used Tensor Processing Units (TPUs) with [Flax](https://github.com/google/flax) and [JAX](https://github.com/google/jax). JAX is a linear algebra library (like `numpy`) that can do automatic differentiation ([Autograd](https://github.com/hips/autograd)) and compile down to [XLA](https://www.tensorflow.org/xla), and Flax is a neural network library and ecosystem for JAX. TPU compute time was provided free by [Google Cloud](https://cloud.google.com/), who co-sponsored the event.

Over the next two weeks, teams participated in lectures from Hugging Face and Google, trained one or more models using JAX/Flax, shared them with the community, and provided a  [Hugging Face Spaces](https://huggingface.co/spaces) demo showcasing the capabilities of their model. Approximately 100 teams participated in the event, and it resulted in 170 models and 36 demos.

Our team, like probably many others, is a distributed one, spanning 12 time zones. Our common thread is that we all belong to the [TWIML Slack Channel](https://twimlai.slack.com/), where we came together based on a shared interest in Artificial Intelligence (AI) and Machine Learning (ML) topics. 

We fine-tuned the [CLIP Network from OpenAI](https://openai.comclip/) with satellite images and captions from the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). The CLIP network learns visual concepts by being trained with image and caption pairs in a self-supervised manner, by using text paired with images found across the Internet. During inference, the model can predict the most relevant image given a text description or the most relevant text description given an image. CLIP is powerful enough to be used in zero-shot manner on everyday images. However, we felt that satellite images were sufficiently different from everyday images that it would be useful to fine-tune CLIP with them. Our intuition turned out to be correct, as the evaluation results (described below) shows. In this post, we describe details of our training and evaluation process, and our plans for future work on this project.

The goal of our project was to provide a useful service and demonstrate how to use CLIP for practical use cases. Our model can be used by applications to search through large collections of satellite images using textual queries. Such queries could describe the image in totality (for example, beach, mountain, airport, baseball field, etc) or search or mention specific geographic or man-made features within these images. CLIP can similarly be fine-tuned for other domains as well, as shown by the [medclip-demo team](https://huggingface.co/spaces/flax-community/medclip-demo) for medical images.

The ability to search through large collections of images using text queries is an immensely powerful feature, and can be used as much for social good as for malign purposes. Possible applications include national defense and anti-terrorism activities, the ability to spot and address effects of climate change before they become unmanageable, etc. Unfortunately, this power can also be misused, such as for military and police surveillance by authoritarian nation-states, so it does raise some ethical questions as well.

You can read about the project on our [project page](https://github.com/arampacha/CLIP-rsicd), download our [trained model](https://huggingface.co/flax-community/clip-rsicd-v2) to use for inference on your own data, or see it in action on our [demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo).


### Training

#### Dataset

We fine-tuned the CLIP model primarily with the [RSICD dataset](https://github.com/201528014227051/RSICD_optimal). This dataset consists of about 10,000 images collected from Google Earth, Baidu Map, MapABC, and Tianditu. It is provided freely to the research community to advance remote sensing captioning via [Exploring Models and Data for Remote Sensing Image Caption Generation](https://arxiv.org/abs/1712.0783) (Lu et al, 2017). The images are (224, 224) RGB images at various resolutions, and each image has up to 5 captions associated with it.

<img src="/blog/assets/30_clip_rsicd/rsicd-images-sampling.png"/>
<center><i>Some examples of images from the RSICD dataset</i></center>

In addition, we used the [UCM Dataset](https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA) and the [Sydney dataset](https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ) for training, The UCM dataset is based on the UC Merced Land Use dataset. It consists of 2100 images belonging to 21 classes (100 images per class), and each image has 5 captions. The Sydney dataset contains images of Sydney, Australia from Google Earth. It contains 613 images belonging to 7 classes. Images are (500, 500) RGB and provides 5 captions for each image. We used these additional datasets because we were not sure if the RSICD dataset would be large enough to fine-tune CLIP.


#### Model

Our model is just the fine-tuned version of the original CLIP model shown below. Inputs to the model are a batch of captions and a batch of images passed through the CLIP text encoder and image encoder respectively. The training process uses [contrastive learning](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607) to learn a joint embedding representation of image and captions. In this embedding space, images and their respective captions are pushed close together, as are similar images and similar captions. Conversely, images and captions for different images, or dissimilar images and captions, are likely to be pushed further apart.

<img src="/blog/assets/30_clip_rsicd/clip_schematic.png"/>
<center><i>CLIP Training and Inference (Image Credit: CLIP: Connecting Text and Images (https://openai.comclip/))</i></center>


#### Data Augmentation

In order to regularize our dataset and prevent overfitting due to the size of the dataset, we used both image and text augmentation.

Image augmentation was done inline using built-in transforms from Pytorch's [Torchvision](https://pytorch.org/vision/stable/index.html) package. The transformations used were Random Cropping, Random Resizing and Cropping, Color Jitter, and Random Horizontal and Vertical flipping.

We augmented the text with backtranslation to generate captions for images with less than 5 unique captions per image. The [Marian MT]((https://huggingface.co/transformers/model_doc/marian.html)) family of models from Hugging Face was used to translate the existing captions into French, Spanish, Italian, and Portuguese and back to English to fill out the captions for these images.

As shown in these loss plots below, image augmentation reduced overfitting significantly, and text and image augmentation reduced overfitting even further.

<img src="/blog/assets/30_clip_rsicd/image-augment-loss.png"/>
<img src="/blog/assets/30_clip_rsicd/image-text-aug-loss.png"/>
<center><i>Evaluation and Training loss plots comparing (top) no augmentation vs image augmentation, and (bottom) image augmentation vs text+image augmentation</i></center>


### Evaluation

#### Metrics

A subset of the RSICD test set was used for evaluation. We found 30 categories of images in this subset. The evaluation was done by comparing each image with a set of 30 caption sentences of the form `"An aerial photograph of {category}"`. The model produced a ranked list of the 30 captions, from most relevant to least relevant. Categories corresponding to captions with the top k scores (for k=1, 3, 5, and 10) were compared with the category provided via the image file name. The scores are averaged over the entire set of images used for evaluation and reported for various values of k, as shown below.

The `baseline` model represents the pre-trained `openai/clip-vit-base-path32` CLIP model. This model was fine-tuned with captions and images from the RSICD dataset, which resulted in a significant performance boost, as shown below.

Our best model was trained with image and text augmentation, with batch size 1024 (128 on each of the 8 TPU cores), and the Adam optimizer with learning rate 5e-6. We trained our second base model with the same hyperparameters, except that we used the Adafactor optimizer with learning rate 1e-4. You can download either model from their model repos linked to in the table below.

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


#### Demo

You can access the [CLIP-RSICD Demo](https://huggingface.co/spaces/sujitpal/clip-rsicd-demo) here. It uses our fine-tuned CLIP model to provide the following functionality:

* Text to Image search
* Image to Image search
* Find text feature in image

The first two functionalities use the RSICD test set as its image corpus. They are encoded using our best fine-tuned CLIP model and stored in a [NMSLib](https://github.com/nmslib/nmslib) index which allows Approximate Nearest Neighbor based retrieval. For text-to-image and image-to-image search respectively, the query text or image are encoded with our model and matched against the image vectors in the corpus. For the third functionality, we divide the incoming image into patches and encode them, encode the queried text feature, match the text vector with each image patch vector, and return the probability of finding the feature in each patch.

### Future Work

We are grateful that we have been given an opportunity to further refine our model. Some ideas we have for future work are as follows:

1. Construct a sequence to sequence model using a CLIP encoder and a GPT-3 decoder and train it for image captioning.
2. Fine-tune the model on more image caption pairs from other datasets and investigate if we can improve its performance.
3. Investigate how fine-tuning affects the performance of model on non-RSICD image caption pairs.
4. Investigate the capability of the fine-tuned model to classify outside the categories it has been fine-tuned on.
5. Evaluate the model using other criteria such as image classification.





