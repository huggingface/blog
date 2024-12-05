---
title: "Welcome PaliGemma 2 – New vision language models by Google"
thumbnail: /blog/assets/paligemma/Paligemma2.png
authors:
- user: merve
- user: andsteing
  guest: true
  org: google
- user: pcuenq
- user: ariG23498
---


# Welcome PaliGemma 2 – New vision language models by Google

We are excited to welcome Google's all-new vision language models, PaliGemma 2, a new iteration of PaliGemma. Like its predecessor, PaliGemma 2 uses the same powerful [SigLIP](https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba) for vision, but it upgrades to the latest Gemma 2 for the text decoder part.

PaliGemma 2 comes with new pre-trained (pt) models, in sizes of `3B`, `10B`, and `28B` parameters. All of them support various input resolutions: `224x224`, `448x448`, and `896x896`. These combinations provide a lot of flexibility for different use cases, so practitioners can choose the balance they need in the quality / efficiency space. In contrast, the previous PaliGemma was only available in the 3B variant.

The pre-trained models have been designed for easy fine-tuning to downstream tasks. The first PaliGemma was widely adopted by the community for multiple purposes. With the increased flexibility from the additional variants, combined with better pre-trained quality, we can’t wait to see what the community can do this time.

As an example, Google is also releasing some fine-tuned variants on the [DOCCI](https://huggingface.co/datasets/google/docci) dataset, demonstrating versatile and robust captioning capabilities that are long, nuanced and detailed. The fine-tuned DOCCI models are available for the 3B and 10B variants, and support input resolution of 448x448.


This release includes all the open model repositories, transformers integration, fine-tuning scripts, and a demo of a model we fine-tuned ourselves for visual question answering on the [VQAv2 dataset](https://huggingface.co/datasets/HuggingFaceM4/VQAv2).


- [Release collection](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)
  
- [Fine-tuning Script](https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb)
  
- [Demo for Fine-tuned Model](https://huggingface.co/spaces/merve/paligemma2-vqav2)
  
- [The technical report](https://huggingface.co/papers/2412.03555)


## Table of Content

* [Introducing PaliGemma 2](#introducing-paligemma-2)

* [Model Capabilities](#model-capabilities)

* [Demo](#demo)

* [How to Use with transformers](#how-to-use-with-transformers)

* [Fine-tuning](#fine-tuning)

* [Resources](#resources)

## Introducing PaliGemma 2

PaliGemma 2 is a new iteration of the [PaliGemma vision language model](https://huggingface.co/blog/paligemma) released by Google in May. PaliGemma 2 connects the powerful SigLIP image encoder with the [Gemma 2](https://huggingface.co/blog/gemma2) language model.

![PaliGemma2 Architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma2_arch.png)

The new models are based on the Gemma 2 2B, 9B, and 27B language models, resulting in the corresponding 3B, 10B, and 28B PaliGemma 2 variants, whose names take into account the additional parameters of the (compact) image encoder. As mentioned above, they support three different resolutions, providing great flexibility for fine-tuning to downstream tasks.

PaliGemma 2 is distributed under the Gemma license, which allows for redistribution, commercial use, fine-tuning and creation of model derivatives.

This release comes with the following checkpoints in `bfloat16` precision:

- 9 pre-trained models: 3B, 10B, and 28B with resolutions of `224x224`, `448x448`, and `896x896`.
  
- 2 models fine-tuned on DOCCI: Two models fine-tuned on the [DOCCI](https://huggingface.co/datasets/google/docci) dataset (image-text caption pairs), supporting the 3B and 10B PaliGemma 2 variants and input resolution of `448x448`.
  
## Model Capabilities

As seen with the previous PaliGemma release, the pre-trained (pt) models work great for further fine-tuning on downstream tasks. 

The pt models are pre-trained on the following data mixture. The diversity of the pre-training dataset allows fine-tuning on downstream tasks in similar domains to be carried out using comparatively fewer examples.

- **WebLI**: A web-scale multilingual image-text dataset built from the public web. A wide range of WebLI splits is used to acquire versatile model capabilities, such as visual semantic understanding, object localization, visually-situated text understanding, and multilinguality.
  
- **CC3M-35L:** Curated English image-alt_text pairs from webpages ([Sharma et al., 2018](https://aclanthology.org/P18-1238/)). To label this dataset, the authors used [Google Cloud Translation API](https://cloud.google.com/translate) to translate into 34 additional languages.
  
- **Visual Question Generation with Question Answering Validation (VQ2A):** An improved dataset for question answering. The dataset is translated into the same additional 34 languages, using the Google Cloud Translation API.
  
OpenImages: Detection and object-aware questions and answers (Piergiovanni et al. 2022) generated by handcrafted rules on the [OpenImages dataset](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).
- **WIT**: Images and texts collected from Wikipedia (Srinivasan et al., 2021). 

The PaliGemma 2 team internally fine-tuned the PT models on a wide variety of visual-language understanding tasks, and they provide benchmarks of these fine-tuned models [in the model card](https://huggingface.co/google/paligemma2-28b-pt-896#paligemma-2-results-by-model-resolution-and-size) and [the technical report](https://huggingface.co/papers/2412.03555).

PaliGemma 2 fine-tuned on the DOCCI dataset, can accomplish a wide range of captioning tasks, including text rendering, capturing spatial relations, and including world knowledge in captions.

You can find below the performance of the DOCCI fine-tuned PaliGemma 2 checkpoints, compared with other models (taken from Table 6 in [the technical report](https://huggingface.co/papers/2412.03555)).

|                 | #par    | #char   | #sent   | NES↓     |
|-----------------|---------|---------|---------|----------|
| MiniGPT-4       | 7B      | 484     | 5.6     | 52.3     |
| mPLUG-Owl2      | 8B      | 459     | 4.4     | 48.4     |
| InstructBLIP    | 7B      | 510     | 4.0     | 42.6     |
| LLaVA-1.5       | 7B      | 395     | 4.2     | 40.6     |
| VILA            | 7B      | 871     | 8.6     | 28.6     |
| PaliGemma       | 3B      | 535     | 8.9     | 34.3     |
| PaLI-5B         | 5B      | 1065    | 11.3    | 32.9     |
| **PaliGemma 2** | **3B**  | **529** | **7.7** | **28.4** |
| **PaliGemma 2** | **10B** | **521** | **7.5** | **20.3** |


- **#char**: Average number of characters in the generated caption.
- **#sent**: Average number of sentences.
- **NES**: Non entailment sentences (lower is better) that measure factual inaccuracies.

Below you can find some model outputs for the DOCCI checkpoint that showcase the versatility of the model.

<table><thead><tr><th width="70%">Input Image</th><th>Caption</th></tr></thead><tbody><tr> 
    <td style="vertical-align: middle;">
        <img width="512" src="https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/howto.jpg" width="300"/></td>
       <td style="vertical-align: middle;">
         A line graph shows the top-1 accuracy of the ImageNet model after fine-tuning. The graph shows four lines that are colored blue, orange, green, and black. The blue line is the lowest of the four lines, and it is
       </td>
     </tr>
     <tr>
       <td style="vertical-align: middle;">
        <img width="512" src="https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg" width="300"/></td>
       <td style="vertical-align: middle;">
         A close up view of a white piece of paper with black text on it. The paper is curved in the middle. The text on the paper is in a typewriter font. The top of the paper has the words "Ashley Hotel West Coast" on it. Underneath that is "WiFi Internet Service". Underneath that is "Username: fqpp". Underneath that is "Password: aaeu
       </td>
     </tr>
     <tr>
       <td style="vertical-align: middle;">
        <img width="512" src="https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/bowie.jpg" width="300"/></td>
       <td style="vertical-align: middle;">
         A mural of David Bowie's Ziggy Stardust look is painted on a white wall. The mural is of three faces side by side, each with red hair and blue lightning bolts painted over their eyes. The faces have blue eyeshadow, pink blush, and red lips. The face in the middle has a black square window above it with white text that reads "JAM" in blue. A silver car
       </td>
     </tr>
     <tr>
   <td style="vertical-align: middle;">
    <img width="512" src="https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/cups.jpg" width="300"/></td>
       <td style="vertical-align: middle;">
         A top-down view of a white marble counter with four coffee mugs on it. There are two gray ones on the left, and one is white on the bottom left. The one on the right is gray. There is a metal wire fruit basket on a wood stand in the top right corner with oranges in it. There is a clear glass pitcher with water in it on the left, and part
       </td>
     </tr>
     <tr>
   <td style="vertical-align: middle;">
    <img width="512" src="https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/ulges.jpg" width="300"/></td>
       <td style="vertical-align: middle;">
         A close up view of a white book with a blue strip at the bottom of it. The top half of the book is white. Black text is printed on the white portion of the book. The text reads "Visual Concept Learning from User-tagged Web Video". Underneath the black text is a white box with five small images inside of it. The image on the far left is of a person standing in a field of grass. The image to the right of that one is of a blue ocean
       </td>
     </tr>
   </table>
   
## Demo

For demonstration purposes, we in the Hugging Face team fine-tuned [PaliGemma 2 3B with 448x448 resolution](https://huggingface.co/google/paligemma2-3b-pt-448) on a small portion of the [VQAv2 dataset](https://huggingface.co/datasets/merve/vqav2-small). We used LoRA fine-tuning and PEFT, as explained later in the fine-tuning section. The demo below showcases the final result. Feel free to examine the code in [the Space](https://huggingface.co/spaces/merve/paligemma2-vqav2) to see how it works, or clone it to adapt to your own fine-tunes.
<figure class="image flex flex-col items-center text-center m-0 w-full">
  <video alt="paligemma.mp4" autoplay loop autobuffer muted playsinline>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma2.mp4" type="video/mp4">
  </video>
  <figcaption></figcaption>
</figure>

## How to Use with Transformers

You can run inference on the PaliGemma 2 models with 🤗 transformers, using the PaliGemmaForConditionalGeneration and AutoProcessor APIs. Until a PyPi version of transformers is released, you need to install it from the main branch as follows:

```bash
pip install git+https://github.com/huggingface/transformers
```

After that, you can run inference like this:

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

model_id = "google/paligemma2-10b-ft-docci-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<image>caption en"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
raw_image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")

inputs = processor(prompt, raw_image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
# A medium shot of two cats laying on a pile of brown fishing nets. The cat in the foreground is a gray tabby cat with white on its chest and paws. The cat is laying on its side with its head facing the bottom right corner of the image. The cat in the background is laying on its side with its head facing the top left corner of the image. The cat's body is
```


You can also use the transformers `bitsandbytes` integration to load the models with quantization. The following example uses 4-bit `nf4`:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = PaligemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0}
)
```

We quickly tested performance degradation in the presence of quantization by evaluating a 3B fine-tuned checkpoint on the [textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) dataset, using 224x224 input images. These are the results we got on the 5,000 entries of the validation set:

- `bfloat16`, no quantization: 60.04% accuracy.
- `8-bit`: 59.78%.
- `4-bit`, using the configuration from the snippet above: 58.72%.

These are very encouraging figures! Of course, quantization is most interesting for the larger checkpoints, we recommend you always measure results on the domains and tasks you’ll be using.

## Fine-tuning

If you have previously fine-tuned PaliGemma, the API to fine-tune PaliGemma 2 is the same, you can use your code out of the box. We provide a [fine-tuning script](https://github.com/merveenoyan/smol-vision/blob/main/paligemma.py) and [a notebook](https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb) for you to fine-tune the model, freeze parts of the model, or apply memory efficient fine-tuning techniques like LoRA or QLoRA.

We have LoRA-fine-tuned a PaliGemma 2 model on half of the VQAv2 validation split for demonstration purposes. This took half an hour on 3 A100s with 80GB VRAM. The model can be found [here](https://huggingface.co/merve/paligemma2-3b-vqav2), and [this is a Gradio demo that showcases it](https://huggingface.co/spaces/merve/paligemma2-vqav2).


## Conclusion

The new PaliGemma 2 release is even more exciting than the previous one, with various sizes fitting everyone’s needs and stronger pre-trained models. We are looking forward to seeing what the community will build!

We thank the Google team for releasing this amazing, and open, model family. Big thanks to [Pablo Montalvo](https://huggingface.co/Molbap) for integrating the model to transformers, and to [Lysandre](https://huggingface.co/lysandre), [Raushan](https://huggingface.co/RaushanTurganbay), [Arthur](https://huggingface.co/ArthurZ), [Yieh-Dar](https://huggingface.co/ydshieh) and the rest of the team for reviewing, testing, and merging in no time.

## Resources

- [Release collection](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)
- [PaliGemma blog post](https://huggingface.co/blog/paligemma)
- [Fine-tuning Script](https://github.com/merveenoyan/smol-vision/blob/main/Fine_tune_PaliGemma.ipynb)
- [Fine-tuned Model on VQAv2](https://huggingface.co/merve/paligemma2-3b-vqav2)
- [Demo for Fine-tuned Model](https://huggingface.co/spaces/merve/paligemma2-vqav2)
- [The technical report](https://huggingface.co/papers/2412.03555)

