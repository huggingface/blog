---
title: "A Deepdive into Aya Vision: Advancing the Frontier of Multilingual Multimodality"
thumbnail: /blog/assets/aya-vision/thumbnail.png
authors:
- user: saurabhdash
  guest: true
  org: CohereForAI
- user: olivernan
  guest: true
  org: CohereForAI
- user: ArashAhmadian
  guest: true
  org: CohereForAI
- user: johndang-cohere
  guest: true
  org: CohereForAI
---

# A Deepdive into Aya Vision: Advancing the Frontier of Multilingual Multimodality

With the release of the [Aya Vision family](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484), our new **8B** and **32B** parameter vision-language models (VLMs), we are addressing one of the biggest challenges in AI: *bringing multilingual performance to multimodal models*. 

Aya Vision is [Cohere For AI](https://cohere.com/research)'s latest open-weight multilingual and multimodal model family, designed to be a strong foundation for language and vision understanding across **23 languages**. It builds on the success of [Aya Expanse](https://huggingface.co/collections/CohereForAI/c4ai-aya-expanse-671a83d6b2c07c692beab3c3), state-of-the-art multilingual language models, and extends it using a combination of advanced techniques. These include synthetic annotations, scaling up multilingual data through translation and rephrasing, and multimodal model merging – key methods that improve both language and vision understanding in a multilingual setting. 

As a result, our models perform well in a variety of tasks, including image captioning, visual question answering, text generation, and translating both text and images into clear, natural-language text. We evaluated Aya Vision models on a set of datasets, including our new open-ended vision-language benchmark [AyaVisionBench](https://huggingface.co/datasets/CohereForAI/AyaVisionBench) and a multilingual version of Wild Vision Bench ([mWildVision](https://huggingface.co/datasets/CohereForAI/m-WildVision)) that is translated into 23 languages, which we release both of them for research. 

In pair-wise comparison, Aya Vision 32B outperforms models more than 2x of its size, such as Llama-3.2 90B Vision, Molmo 72B, and Qwen2.5-VL 72B by win rates ranging from 50% to 64% on AyaVisionBench and 52% to 72% on mWildVision average across 23 languages. 

![aya vision win rates](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/aya-vision-combined-win-rates.png)

Our compact and more efficient model Aya Vision 8B achieves the best performance in multilingual multimodal in its parameter class,  outperforming leading models such as Qwen2.5-VL 7B, Pixtral 12B, Gemini Flash 1.5 8B, Llama-3.2 11B Vision, Molmo-D 7B, and Pangea 7B by up to 79% win-rates on AyaVisionBench and 81% on mWildBench. 

![efficiency vs performance trade offs](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/efficiency-vs-performance.png)

We release both [8B](https://huggingface.co/CohereForAI/aya-vision-8b) and [32B](https://huggingface.co/CohereForAI/aya-vision-32b) models as open weights for the research community to further accelerate multilingual multimodal progress. In this blog post, we share the key technical details behind Aya Vision models

## Aya Vision Architecture and Training

![aya vision architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/image-8.png) 

For a high-performance vision-language model, it is important to process images with a*rbitrary resolutions*, especially high-resolution images. To enable this capability in Aya Vision, we dynamically resize and split any higher-resolution images into multiple tiles to generate rich image features from the image encoder. In Aya Vision models, we use the recently released [SigLIP2-patch14-384](https://huggingface.co/blog/siglip2) model as the initialization for the vision encoder. 

While dynamic resizing enables processing high-resolution images, it also leads to a larger number of image tokens passing through the vision-language connector and LLM decoder. To improve latency and throughput, we use a downsampling method called [Pixel Shuffle](https://arxiv.org/pdf/2404.16821), to compress the number of image tokens by 4x. After downsampling, image tokens are aligned to the language model input embeddings through a vision-language connector and passed to an LLM decoder.

For the text decoder, we use our multilingual language models. For Aya Vision 8B, we use an LLM that is initialized from [Cohere Command  R7B](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024) for improved instruction following and world knowledge and further post-trained using the [Aya Expanse recipe](https://huggingface.co/blog/aya-expanse) consisting of diverse multilingual data, model merging, and preference training. For Aya Vision 32B, we initialize the language model from [Aya Expanse 32B](https://huggingface.co/CohereForAI/aya-expanse-32b) based on its state-of-the-art multilingual performance.

### Training process

We trained Aya Vision models in **2 stages** – *vision-language alignment* and *supervised fine-tuning (SFT)*. In the vision-language alignment stage, only the vision-language connector is trained, while the vision encoder and the language model weights are kept frozen. This enables rudimentary vision-language understanding by mapping the image encoder features to the language model embedding space. In the SFT stage, we train both the connector and the language model on a diverse set of multimodal tasks in 23 languages.

![step by step improvement](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/step-by-step-improvement.png)

## Multimodal Data Enhancement and Expanding Language Coverage

![multilingual synthetic annotation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/image-9.png)

One of the biggest challenges in developing a multilingual vision-language model is ensuring strong performance across underrepresented languages. To address this, we first gather synthetic annotations using a diverse pool of high-quality datasets in English, which lay the basis for our multilingual multimodal annotation. Following the synthetic annotations of English datasets, we translated a large volume of the data into 23 languages. To avoid translation artefacts and maintain fluent textual characteristics with high precision in answers, we then rephrased translated prompt/generation pairs by matching them with the original high-quality synthetic samples, expanding language coverage where real-world datasets are scarce. This improves both linguistic fluency and alignment between vision and text, allowing Aya Vision to exhibit superior image understanding in multiple languages.

Our 8B model, when only supervised fine-tuned with original academic datasets, reaches a 40.9% win rate across 23 languages in AyaVisionBench against Pangea 7B, which is a multilingual VLM, whereas synthetic annotations and scaling up the multilingual data lead to a 58.1% win rate with a gain of 17.2%. This significant improvement showcases the impact of significant investment in multilingual data coverage.

## Multimodal Model Merging

![multimodal merging](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/image-10.png)

A state-of-the-art vision-language model should excel not only in image understanding but also in conversational context, where the model is expected to generate a high-quality response to both image and text inputs. To address this, inspired by our previous research on model merging, a technique that combines multiple trained models, we merge the base language model with the fine-tuned vision-language model. 

Model merging enhances the generative capabilities of our final model that leads to a 70% win rates across 23 languages on [AyaVisionBench](https://huggingface.co/datasets/CohereForAI/AyaVisionBench) against Pangea 7B, improving the multimodal win rate by 11.9% compared to the model before merging. 

Multimodal model merging also enables our Aya Vision models to excel in text-only tasks as measured in mArenaHard datasets compared with the other leading vision-language models. 

| ![stages](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/image-11.png) |
| :--: |
| Overview of the training pipeline for Aya Vision |

## Scaling up to 32B

Finally, we scale our recipe from 8B to 32B, resulting in the state-of-the-art open-weight multilingual vision-language model – Aya Vision 32B which shows significant improvements in win rates due to the stronger initialization of the text-backbone, and outperforms models more than 2x of its size, such as Llama-3.2 90B Vision, Molmo 72B, and Qwen2.5-VL 72B by win rates ranging from 49% to 63% on AyaVisionBench and 52% to 72% on mWildVision average across 23 languages.

![aya vision benchmark](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/aya-vision/aya-vision-bench.png)

## Aya Vision Benchmark – a multilingual evaluation data

Together with Aya Vision models, we also release a high-quality multilingual vision-language benchmark called [AyaVisionBench](https://huggingface.co/datasets/CohereForAI/AyaVisionBench), constructed based on real-world applications, covering 23 languages and 9 distinct task categories, with 135 image-question pairs per language. 

We make this evaluation set available to the research community to push forward multilingual multimodal evaluations. This dataset is designed to assess a model’s ability to perform a diverse range of vision-language tasks, including captioning, chart and figure understanding, identifying differences between two images, general visual question answering, OCR, document understanding, text transcription, reasoning involving logic and math, and converting screenshots to code. By incorporating multiple languages and task types, the dataset provides a broad and challenging evaluation framework for assessing cross-lingual and multimodal understanding.

To create this dataset, we first selected images from the [Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) held-out test set, a large collection derived from 50 high-quality datasets, ensuring they had not been seen during training. For each image, we then generated a corresponding question that explicitly required visual context for an answer. These questions were synthetically generated and subsequently refined through a two-stage verification process. First, human annotators reviewed and validated each question to ensure it was clear, relevant, and truly dependent on the image. This rigorous selection and validation process ensures that the dataset serves as a robust benchmark for evaluating vision-language models in multilingual and real-world settings.

## Designed for real-world applications

Communication happens in many forms and in many languages. With our leading research and development, we’ve released a model that facilitates connection, whether in text or visual, in 23 different languages today. 

Aya Vision has a wide range of practical applications, where one notable example is its [availability on WhatsApp](https://cohere.com/research/aya/whatsapp), one of the most broadly used communications platforms in the world. This allows a massive audience of global citizens who speak a multitude of languages to utilize the capabilities of Aya Vision on a platform they use to communicate every single day.

## Getting Started with Aya

To get started:

Download weights and datasets from the [Aya Vision collection](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484) on Hugging Face.  
Try Aya Vision using our [Hugging Face Space](https://huggingface.co/spaces/CohereForAI/aya_expanse) or text it on [Whatsapp](https://cohere.com/research/aya/whatsapp)  
Build on Aya using our [colab example](https://colab.research.google.com/drive/1jHYi8WVyRE6-imTRA37h_9txjrr8WNZd?usp=sharing).

Learn more about our ongoing efforts around multilingual.

## Acknowledgments  
This work wouldn’t have been possible without the core Aya Vision technical team:  

Saurabh Dash, Oliver Nan, John Dang, Arash Ahmadian Dehkordi, Shivalika Singh, Alejandro Salamanca, Bharat Venkitesh, Vlad Shmyhlo, Walter Beller-Morales, Jeremy Pekmez, Jason Ozuzu, Madeline Smith, Marzieh Fadaee, Manoj Govindassamy, Sudip Roy, Matthias Gallé, Beyza Ermis, Ahmet Üstün, Sara Hooker.

It also wouldn’t have been possible without the wider Cohere For AI and Cohere team who supported in many different ways. Special thanks to Sungjin Hong, Michael Kozakov, Pierre Richemond, Brittawnya Prince, Jim Payne, Kyle Lastovica, Jeff Colen, Jenna Cook, Viraat Aryabumi, Trent Fowler, Linus Chui, Meor Amer, Lucas Fayoux, Kyle Lastovica, Billy Trend, Acyr Locatelli, Morgan Norman, Florian Strub, Jon Ander Campos, Nick Frosst, Phil Blunsom, Aidan Gomez, Ivan Zhang.

Special thank you to Hugging Face for helping make this come together: Yoni Gozlan, Arthur Zucker, Pedro Cuenca, Aritra Roy Gosthipaty, Merve Noyan, Vaibhav Srivastav.

## References

[1] [*Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier*](https://arxiv.org/abs/2412.04261)  
[2] [Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages](https://arxiv.org/abs/2410.16153)  
[3] [WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences](https://arxiv.org/abs/2406.11069)  
[4] [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)  
[5] [What matters when building vision-language models?](https://arxiv.org/abs/2405.02246)  
[6] [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146)  
[7] [How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://arxiv.org/pdf/2404.16821)
