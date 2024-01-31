---
title: "Ethics and Society Newsletter #4: Bias in Text-to-Image Models"
thumbnail: /blog/assets/152_ethics_soc_4/ethics_4_thumbnail.png
authors:
- user: sasha
- user: giadap
- user: nazneen
- user: allendorf
- user: irenesolaiman
- user: natolambert
- user: meg

---

# Ethics and Society Newsletter #4: Bias in Text-to-Image Models



**TL;DR: We need better ways of evaluating bias in text-to-image models**


## Introduction

[Text-to-image (TTI) generation](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads) is all the rage these days, and thousands of TTI models are being uploaded to the Hugging Face Hub. Each modality is potentially susceptible to separate sources of bias, which begs the question: how do we uncover biases in these models? In the current blog post, we share our thoughts on sources of bias in TTI systems as well as tools and potential solutions to address them, showcasing both our own projects and those from the broader community.

## Values and bias encoded in image generations

There is a very close relationship between [bias and values](https://www.sciencedirect.com/science/article/abs/pii/B9780080885797500119), particularly when these are embedded in the language or images used to train and query a given [text-to-image model](https://dl.acm.org/doi/abs/10.1145/3593013.3594095); this phenomenon heavily influences the outputs we see in the generated images. Although this relationship is known in the broader AI research field and considerable efforts are underway to address it, the complexity of trying to represent the evolving nature of a given population's values in a single model still persists. This presents an enduring ethical challenge to uncover and address adequately.

For example, if the training data are mainly in English they probably convey rather Western values. As a result we get stereotypical representations of different or distant cultures. This phenomenon appears noticeable when we compare the results of ERNIE ViLG (left) and Stable Diffusion v 2.1 (right) for the same prompt, "a house in Beijing":

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/ernie-sd.png" alt="results of ERNIE ViLG (left) and Stable Diffusion v 2.1 (right) for the same prompt, a house in Beijing" />
</p>

## Sources of Bias

Recent years have seen much important research on bias detection in AI systems with single modalities in both Natural Language Processing ([Abid et al., 2021](https://dl.acm.org/doi/abs/10.1145/3461702.3462624)) as well as Computer Vision ([Buolamwini and Gebru, 2018](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)). To the extent that ML models are constructed by people, biases are present in all ML models (and, indeed, technology in general). This can manifest itself by an over- and under-representation of certain visual characteristics in images (e.g., all images of office workers having ties), or the presence of cultural and geographical stereotypes (e.g., all images of brides wearing white dresses and veils, as opposed to more representative images of brides around the world, such as brides with red saris). Given that AI systems are deployed in sociotechnical contexts that are becoming widely deployed in different sectors and tools (e.g. [Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html), [Shutterstock](https://www.shutterstock.com/ai-image-generator)), they are particularly likely to amplify existing societal biases and inequities. We aim to provide a non-exhaustive list of bias sources below:

**Biases in training data:** Popular multimodal datasets such as [LAION-5B](https://laion.ai/blog/laion-5b/) for text-to-image, [MS-COCO](https://cocodataset.org/) for image captioning, and [VQA v2.0](https://paperswithcode.com/dataset/visual-question-answering-v2-0) for visual question answering, have been found to contain numerous biases and harmful associations ([Zhao et al 2017](https://aclanthology.org/D17-1323/), [Prabhu and Birhane, 2021](https://arxiv.org/abs/2110.01963), [Hirota et al, 2022](https://facctconference.org/static/pdfs_2022/facct22-3533184.pdf)), which can percolate into the models trained on these datasets. For example, initial results from the [Hugging Face Stable Bias project](https://huggingface.co/spaces/society-ethics/StableBias) show a lack of diversity in image generations, as well as a perpetuation of common stereotypes of cultures and identity groups. Comparing Dall-E 2 generations of CEOs (right) and managers (left), we can see that both are lacking diversity:

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/CEO_manager.png" alt="Dall-E 2 generations of CEOs (right) and managers (left)" />
</p>

**Biases in pre-training data filtering:** There is often some form of filtering carried out on datasets before they are used for training models; this introduces different biases. For instance, in their [blog post](https://openai.com/research/dall-e-2-pre-training-mitigations), the creators of Dall-E 2 found that filtering training data can actually amplify biases – they hypothesize that this may be due to the existing dataset bias towards representing women in more sexualized contexts or due to inherent biases of the filtering approaches that they use.

**Biases in inference:** The [CLIP model](https://huggingface.co/openai/clip-vit-large-patch14) used for guiding the training and inference of text-to-image models like Stable Diffusion and Dall-E 2 has a number of [well-documented biases](https://arxiv.org/abs/2205.11378) surrounding age, gender, and race or ethnicity, for instance treating images that had been labeled as `white`, `middle-aged`, and `male` as the default. This can impact the generations of models that use it for prompt encoding, for instance by interpreting unspecified or underspecified gender and identity groups to signify white and male.

**Biases in the models' latent space:** [Initial work](https://arxiv.org/abs/2302.10893) has been done in terms of exploring the latent space of the model and guiding image generation along different axes such as gender to make generations more representative (see the images below). However, more work is necessary to better understand the structure of the latent space of different types of diffusion models and the factors that can influence the bias reflected in generated images.


<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/fair-diffusion.png" alt="Fair Diffusion generations of firefighters." />
</p>

**Biases in post-hoc filtering:** Many image generation models come with built-in safety filters that aim to flag problematic content. However, the extent to which these filters work and how robust they are to different kinds of content is to be determined – for instance, efforts to [red-team the Stable Diffusion safety filter](https://arxiv.org/abs/2210.04610)have shown that it mostly identifies sexual content, and fails to flag other types violent, gory or disturbing content.

## Detecting Bias

Most of the issues that we describe above cannot be solved with a single solution – indeed, [bias is a complex topic](https://huggingface.co/blog/ethics-soc-2) that cannot be meaningfully addressed with technology alone. Bias is deeply intertwined with the broader social, cultural, and historical context in which it exists. Therefore, addressing bias in AI systems is not only a technological challenge but also a socio-technical one that demands multidisciplinary attention. However, a combination of approaches including tools, red-teaming and evaluations can help glean important insights that can inform both model creators and downstream users about the biases contained in TTI and other multimodal models.

We present some of these approaches below:

**Tools for exploring bias:** As part of the [Stable Bias project](https://huggingface.co/spaces/society-ethics/StableBias), we created a series of tools to explore and compare the visual manifestation of biases in different text-to-image models. For instance, the [Average Diffusion Faces](https://huggingface.co/spaces/society-ethics/Average_diffusion_faces) tool lets you compare the average representations for different professions and different models – like for 'janitor', shown below, for Stable Diffusion v1.4, v2, and Dall-E 2:

<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/average.png" alt="Average faces for the 'janitor' profession, computed based on the outputs of different text to image models." />
</p>


Other tools, like the [Face Clustering tool](https://hf.co/spaces/society-ethics/DiffusionFaceClustering) and the [Colorfulness Profession Explorer](https://huggingface.co/spaces/tti-bias/identities-colorfulness-knn) tool, allow users to explore patterns in the data and identify similarities and stereotypes without ascribing labels or identity characteristics. In fact, it's important to remember that generated images of individuals aren't actual people, but artificial creations, so it's important not to treat them as if they were real humans. Depending on the context and the use case, tools like these can be used both for storytelling and for auditing.

**Red-teaming:** ['Red-teaming'](https://huggingface.co/blog/red-teaming) consists of stress testing AI models for potential vulnerabilities, biases, and weaknesses by prompting them and analyzing results. While it has been employed in practice for evaluating language models (including the upcoming [Generative AI Red Teaming event at DEFCON](https://aivillage.org/generative%20red%20team/generative-red-team/), which we are participating in), there are no established and systematic ways of red-teaming AI models and it remains relatively ad hoc. In fact, there are so many potential types of failure modes and biases in AI models, it is hard to anticipate them all, and the [stochastic nature](https://dl.acm.org/doi/10.1145/3442188.3445922) of generative models makes it hard to reproduce failure cases. Red-teaming gives actionable insights into model limitations and can be used to add guardrails and document model limitations. There are currently no red-teaming benchmarks or leaderboards highlighting the need for more work in open source red-teaming resources. [Anthropic's red-teaming dataset](https://github.com/anthropics/hh-rlhf/tree/master/red-team-attempts) is the only open source resource of red-teaming prompts, but is limited to only English natural language text.

**Evaluating and documenting bias:** At Hugging Face, we are big proponents of [model cards](https://huggingface.co/docs/hub/model-card-guidebook) and other forms of documentation (e.g., [datasheets](https://arxiv.org/abs/1803.09010), READMEs, etc). In the case of text-to-image (and other multimodal) models, the result of explorations made using explorer tools and red-teaming efforts such as the ones described above can be shared alongside model checkpoints and weights. One of the issues is that we currently don't have standard benchmarks or datasets for measuring the bias in multimodal models (and indeed, in text-to-image generation systems specifically), but as more [work](https://arxiv.org/abs/2306.05949) in this direction is carried out by the community, different bias metrics can be reported in parallel in model documentation.

## Values and Bias

All of the approaches listed above are part of detecting and understanding the biases embedded in image generation models. But how do we actively engage with them?

One approach is to develop new models that represent society as we wish it to be. This suggests creating AI systems that don't just mimic the patterns in our data, but actively promote more equitable and fair perspectives. However, this approach raises a crucial question: whose values are we programming into these models? Values differ across cultures, societies, and individuals, making it a complex task to define what an "ideal" society should look like within an AI model. The question is indeed complex and multifaceted. If we avoid reproducing existing societal biases in our AI models, we're faced with the challenge of defining an "ideal" representation of society. Society is not a static entity, but a dynamic and ever-changing construct. Should AI models, then, adapt to the changes in societal norms and values over time? If so, how do we ensure that these shifts genuinely represent all groups within society, especially those often underrepresented?

Also, as we have mentioned in a [previous newsletter](https://huggingface.co/blog/ethics-soc-2#addressing-bias-throughout-the-ml-development-cycle), there is no one single way to develop machine learning systems, and any of the steps in the development and deployment process can present opportunities to tackle bias, from who is included at the start, to defining the task, to curating the dataset, training the model, and more. This also applies to multimodal models and the ways in which they are ultimately deployed or productionized in society, since the consequences of bias in multimodal models will depend on their downstream use. For instance, if a model is used in a human-in-the-loop setting for graphic design (such as those created by [RunwayML](https://runwayml.com/ai-magic-tools/text-to-image/)), the user has numerous occasions to detect and correct bias, for instance by changing the prompt or the generation options. However, if a model is used as part of a [tool to help forensic artists create police sketches of potential suspects](https://www.vice.com/en/article/qjk745/ai-police-sketches) (see image below), then the stakes are much higher, since this can reinforce stereotypes and racial biases in a high-risk setting.


<p align="center">
 <br>
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/152_ethics_soc_4/forensic.png" alt="Forensic AI Sketch artist tool developed using Dall-E 2." />
</p>

## Other updates

We are also continuing work on other fronts of ethics and society, including:

- **Content moderation:**
  - We made a major update to our [Content Policy](https://huggingface.co/content-guidelines). It has been almost a year since our last update and the Hugging Face community has grown massively since then, so we felt it was time. In this update we emphasize *consent* as one of Hugging Face's core values. To read more about our thought process, check out the [announcement blog](https://huggingface.co/blog/content-guidelines-update) **.**
- **AI Accountability Policy:**
  - We submitted a response to the NTIA request for comments on [AI accountability policy](https://ntia.gov/issues/artificial-intelligence/request-for-comments), where we stressed the importance of documentation and transparency mechanisms, as well as the necessity of leveraging open collaboration and promoting access to external stakeholders. You can find a summary of our response and a link to the full document [in our blog post](https://huggingface.co/blog/policy-ntia-rfc)!

## Closing Remarks

As you can tell from our discussion above, the issue of detecting and engaging with bias and values in multimodal models, such as text-to-image models, is very much an open question. Apart from the work cited above, we are also engaging with the community at large on the issues - we recently co-led a [CRAFT session at the FAccT conference](https://facctconference.org/2023/acceptedcraft.html) on the topic and are continuing to pursue data- and model-centric research on the topic. One particular direction we are excited to explore is a more in-depth probing of the [values](https://arxiv.org/abs/2203.07785) instilled in text-to-image models and what they represent (stay tuned!).

