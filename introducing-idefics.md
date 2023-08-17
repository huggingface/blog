---
title: "Introducing IDEFICS: An Open Reproduction of State-of-the-art Visual Langage Model"
thumbnail: /blog/assets/71_ethical-charter/thumbnail.jpg
authors:
- user: HugoLaurencon
- user: davanstrien
- user: stas
- user: Leyo
- user: SaulLu
- user: TimeRobber
- user: skaramcheti
- user: aps
- user: giadap
- user: VictorSanh
---

# Introducing IDEFICS: An Open Reproduction of State-of-the-art Visual Langage Model

<!-- {blog_metadata} -->
<!-- {authors} -->

<!-- TODO: recognize full list of co-authors -->

We are excited to announce the release of IDEFICS (**I**mage-aware **D**ecoder **E**nhanced à la **F**lamingo with **I**nterleaved **C**ross-attention**S**), an open-access visual language model. IDEFICS is based on [Flamingo](https://huggingface.co/papers/2204.14198), a state-of-the-art visual language model initially developed by DeepMind, which has not been released publicly. Similarly to GPT4, the model accepts arbitrary sequences of image and text inputs and produces text outputs. IDEFICS is built solely on publicly available data and models and comes in two variants: the base version and the instructed version.

We believe that the development of state-of-the-art AI models should be more transparent. Our goal with IDEFICS is to reproduce and provide the AI community with systems that match the capabilities of large proprietary models like Flamingo. As such, we take important steps contributing to bringing transparency to these AI systems: we use only publicly available data and we provide tooling to explore training datasets, we share [technical lessons and mistakes](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) of building such artifacts and assessed the model’s harmfulness by adversarially prompting it before releasing it. We hope that IDEFICS will serve as a solid foundation for more open research in multimodal AI systems while contributing to the open-science community.

Try out the [demo](TODO: link) and the [models](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)!

![Screenshot of IDEFICS generation for Woodstock of AI](TODO)

## What is IDEFICS?

IDEFICS is an 80 billion parameter multimodal model that accepts sequences of images and text as input, and generates coherent text as output. It can answer questions about images, describe visual content, create stories grounded in multiple images, etc.

IDEFICS is an open reproduction of Flamingo and is comparable in performance with the original closed-source model across various image-text understanding benchmarks. It comes in two variants - 80B parameters and 9B parameters.

![Plot comparing the performance of Flamingo, OpenFlamingo and IDEFICS](TODO)

We also provide fine-tuned versions [idefics-80B-instruct](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) and [idefics-9B-instruct](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct) adapted for conversational use cases.

## Training Data

IDEFICS was trained on a mixture of openly available datasets: Wikipedia, PMD and LAION, as well as a new 115B token dataset called [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) that we created. OBELICS consists of 141M interleaved image-text documents scraped from the web, containing 353M images.

We provide an [interactive visualization](https://atlas.nomic.ai/map/f2fba2aa-3647-4f49-a0f3-9347daeee499/ee4a84bd-f125-4bcc-a683-1b4e231cb10f) of OBELICS that allows exploring the content of the dataset with [Nomic AI](https://home.nomic.ai/).

The complete details of the IDEFICS architecture, training methodology, and evaluations, as well as information about the dataset are available in the [model card](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) and our [research paper](TODO).

## Ethical evaluation

At the outset of this project, through a set of discussions, the team developed an ethical charter (https://huggingface.co/blog/ethical-charter-multimodal) that would help steer decisions made during the project. This charter sets out values, including being self-critical, transparent, and fairness which we have sought to pursue in how we approached the project itself and the release of this model.

As part of the release process, we internally evaluated the model for potential biases by adversarially prompting the model with images and text that might elicit responses we do not want from the model (a process known as red teaming). We also carried out a more systematic evaluation of potential biases in the model.

Please try out IDEFICS with the demo (link) and let us know your feedback by using the community tab! We are committed to improving these models and making large multimodal AI models more accessible to everyone.

## Getting Started with IDEFICS

IDEFICS models are available on Hugging Face. Here is code sample to try it out:

```python
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
prompts = [
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser:",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "And who is that?<end_of_utterance>",

        "\nAssistant:",
    ],
]

# --batched mode
inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
# --single sample mode
# inputs = processor(prompts[0], return_tensors="pt").to(device)

# Generation args
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")
```
