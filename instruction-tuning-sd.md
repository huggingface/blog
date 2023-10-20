---
title: "Instruction-tuning Stable Diffusion with InstructPix2Pix" 
thumbnail: assets/instruction_tuning_sd/thumbnail.png
authors:
- user: sayakpaul
---

# Instruction-tuning Stable Diffusion with InstructPix2Pix


This post explores instruction-tuning to teach [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) to follow instructions to translate or process input images. With this method, we can prompt Stable Diffusion using an input image and an “instruction”, such as - *Apply a cartoon filter to the natural image*.

| ![schematic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/schematic.png) | 
|:--:|
| **Figure 1**: We explore the instruction-tuning capabilities of Stable Diffusion. In this figure, we prompt an instruction-tuned Stable Diffusion system with prompts involving different transformations and input images.  The tuned system seems to be able to learn these transformations stated in the input prompts. Figure best viewed in color and zoomed in. |

This idea of teaching Stable Diffusion to follow user instructions to perform **edits** on input images was introduced in [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800). We discuss how to extend the InstructPix2Pix training strategy to follow more specific instructions related to tasks in image translation (such as cartoonization) and low-level image processing (such as image deraining). We cover:

- [Introduction to instruction-tuning](#introduction-and-motivation)
- [The motivation behind this work](#introduction-and-motivation)
- [Dataset preparation](#dataset-preparation)
- [Training experiments and results](#training-experiments-and-results)
- [Potential applications and limitations](#potential-applications-and-limitations)
- [Open questions](#open-questions)

Our code, pre-trained models, and datasets can be found [here](https://github.com/huggingface/instruction-tuned-sd). 

## Introduction and motivation

Instruction-tuning is a supervised way of teaching language models to follow instructions to solve a task. It was introduced in [Fine-tuned Language Models Are Zero-Shot Learners](https://huggingface.co/papers/2109.01652) (FLAN) by Google. From recent times, you might recall works like [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [FLAN V2](https://huggingface.co/papers/2210.11416), which are good examples of how beneficial instruction-tuning can be for various tasks. 

The figure below shows a formulation of instruction-tuning (also called “instruction-finetuning”). In the [FLAN V2 paper](https://huggingface.co/papers/2210.11416), the authors take a pre-trained language model ([T5](https://huggingface.co/docs/transformers/model_doc/t5), for example) and fine-tune it on a dataset of exemplars, as shown in the figure below. 

| ![flan_schematic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/flan_schematic.png) |
|:--:|
| **Figure 2**: FLAN V2 schematic (figure taken from the FLAN V2 paper). |

With this approach, one can create exemplars covering many different tasks, which makes instruction-tuning a multi-task training objective: 

| **Input** | **Label** | **Task** |
|---|---|---|
| Predict the sentiment of the<br>following sentence: “The movie<br>was pretty amazing. I could not<br>turn around my eyes even for a<br>second.” | Positive | Sentiment analysis /<br>Sequence classification |
| Please answer the following<br>question. <br>What is the boiling point of<br>Nitrogen? | 320.4F | Question answering |
| Translate the following<br>English sentence into German: “I have<br>a cat.” | Ich habe eine Katze. | Machine translation |
| … | … | … |
| | | | |

Using a similar philosophy, the authors of FLAN V2 conduct instruction-tuning on a mixture of thousands of tasks and achieve zero-shot generalization to unseen tasks:

| ![flan_dataset_overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/flan_dataset_overview.png) | 
|:--:|
| **Figure 3**: FLAN V2 training and test task mixtures (figure taken from the FLAN V2 paper). |

Our motivation behind this work comes partly from the FLAN line of work and partly from InstructPix2Pix. We wanted to explore if it’s possible to prompt Stable Diffusion with specific instructions and input images to process them as per our needs. 

The [pre-trained InstructPix2Pix models](https://huggingface.co/timbrooks/instruct-pix2pix) are good at following general instructions, but they may fall short of following instructions involving specific transformations:

| ![cartoonization_results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/cartoonization_results.jpeg) |
|:--:|
| **Figure 4**: We observe that for the input images (left column), our models (right column) more faithfully perform “cartoonization” compared to the pre-trained InstructPix2Pix models (middle column). It is interesting to note the results of the first row where the pre-trained InstructPix2Pix models almost fail significantly. Figure best viewed in color and zoomed in. See original [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/Instruction-tuning-sd/cartoonization_results.png). |

But we can still leverage the findings from InstructPix2Pix to suit our customizations. 

On the other hand, paired datasets for tasks like [cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization), [image denoising](https://paperswithcode.com/dataset/sidd), [image deraining](https://paperswithcode.com/dataset/raindrop), etc. are available publicly, which we can use to build instruction-prompted datasets taking inspiration from FLAN V2. Doing so allows us to transfer the instruction-templating ideas explored in FLAN V2 to this work. 

## Dataset preparation

### Cartoonization

In our early experiments, we prompted InstructPix2Pix to perform cartoonization and the results were not up to our expectations. We tried various inference-time hyperparameter combinations (such as image guidance scale and the number of inference steps), but the results still were not compelling. This motivated us to approach the problem differently.

As hinted in the previous section, we wanted to benefit from both worlds:

**(1)** training methodology of  InstructPix2Pix and
**(2)** the flexibility of creating instruction-prompted dataset templates from FLAN. 

We started by creating an instruction-prompted dataset for the task of cartoonization. Figure 5 presents our dataset creation pipeline: 

| ![itsd_data_wheel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/itsd_data_wheel.png) |
|:--:|
| **Figure 5**: A depiction of our dataset creation pipeline for cartoonization (best viewed in color and zoomed in). |

In particular, we:

1. Ask [ChatGPT](https://openai.com/blog/chatgpt) to generate 50 synonymous sentences for the following instruction: "Cartoonize the image.” 
2. We then use a random sub-set (5000 samples) of the [Imagenette dataset](https://github.com/fastai/imagenette) and leverage a pre-trained [Whitebox CartoonGAN](https://github.com/SystemErrorWang/White-box-Cartoonization) model to produce the cartoonized renditions of those images. The cartoonized renditions are the labels we want our model to learn from. So, in a way, this corresponds to transferring the biases learned by the Whitebox CartoonGAN model to our model.  
3. Then we create our exemplars in the following format:

| ![cartoonization_dataset_overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/cartoonization_dataset_overview.png) |
|:--:|
| **Figure 6**: Samples from the final cartoonization dataset (best viewed in color and zoomed in). |

Our final dataset for cartoonization can be found [here](https://huggingface.co/datasets/instruction-tuning-vision/cartoonizer-dataset). For more details on how the dataset was prepared, refer to [this directory](https://github.com/huggingface/instruction-tuned-sd/tree/main/data_preparation). We experimented with this dataset by fine-tuning InstructPix2Pix and got promising results (more details in the “Training experiments and results” section). 

We then proceeded to see if we could generalize this approach to low-level image processing tasks such as image deraining, image denoising, and image deblurring. 

### Low-level image processing

We focus on the common low-level image processing tasks explored in [MAXIM](https://huggingface.co/papers/2201.02973). In particular, we conduct our experiments for the following tasks: deraining, denoising, low-light image enhancement, and deblurring. 

We took different number of samples from the following datasets for each task and constructed a single dataset with prompts added like so:

| **Task** | **Prompt** | **Dataset** | **Number of samples** |
|---|---|---|---|
| Deblurring | “deblur the blurry image” | [REDS](https://seungjunnah.github.io/Datasets/reds.html) (`train_blur`<br>and `train_sharp`) | 1200 |
| Deraining | “derain the image” | [Rain13k](https://github.com/megvii-model/HINet#image-restoration-tasks) | 686 |
| Denoising | “denoise the noisy image” | [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) | 8 |
| Low-light<br>image enhancement | "enhance the low-light image” | [LOL](https://paperswithcode.com/dataset/lol) | 23 |
| | | | |

Datasets mentioned above typically come as input-output pairs, so we do not have to worry about the ground-truth. Our final dataset is available [here](https://huggingface.co/datasets/instruction-tuning-vision/instruct-tuned-image-processing). The final dataset looks like so:

| ![low_level_img_proc_dataset_overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/low_level_img_proc_dataset_overview.png) |
|:--:|
| **Figure 7**: Samples from the final low-level image processing dataset (best viewed in color and zoomed in). |

Overall, this setup helps draw parallels from the FLAN setup, where we create a mixture of different tasks. This also helps us train a single model one time, performing well to the different tasks we have in the mixture. This varies significantly from what is typically done in low-level image processing. Works like MAXIM introduce a single model architecture capable of modeling the different low-level image processing tasks, but training happens independently on the individual datasets. 

## Training experiments and results

We based our training experiments on [this script](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py). Our training logs (including validation samples and training hyperparameters) are available on Weight and Biases:

- [Cartoonization](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/wszjpb1b) ([hyperparameters](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/wszjpb1b/overview?workspace=))
- [Low-level image processing](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/2kg5wohb) ([hyperparameters](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/2kg5wohb/overview?workspace=))

When training, we explored two options:

1. Fine-tuning from an existing [InstructPix2Pix checkpoint](https://huggingface.co/timbrooks/instruct-pix2pix)
2. Fine-tuning from an existing [Stable Diffusion checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5) using the InstructPix2Pix training methodology 

In our experiments, we found out that the first option helps us adapt to our datasets faster (in terms of generation quality).  

For more details on the training and hyperparameters, we encourage you to check out [our code](https://github.com/huggingface/instruction-tuned-sd) and the respective run pages on Weights and Biases. 

### Cartoonization results

For testing the [instruction-tuned cartoonization model](https://huggingface.co/instruction-tuning-sd/cartoonizer), we compared the outputs as follows:

| ![cartoonization_full_results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/cartoonization_full_results.png) |
|:--:|
| **Figure 8**: We compare the results of our instruction-tuned cartoonization model (last column) with that of a [CartoonGAN](https://github.com/SystemErrorWang/White-box-Cartoonization) model (column two) and the pre-trained InstructPix2Pix model (column three). It’s evident that the instruction-tuned model can more faithfully match the outputs of the CartoonGAN model. Figure best viewed in color and zoomed in. See original [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/Instruction-tuning-sd/cartoonization_full_results.png). |

To gather these results, we sampled images from the `validation` split of ImageNette. We used the following prompt when using our model and the pre-trained InstructPix2Pix model: *“Generate a cartoonized version of the image”.* For these two models, we kept the `image_guidance_scale` and `guidance_scale` to 1.5 and 7.0, respectively, and number of inference steps to 20. Indeed more experimentation is needed around these hyperparameters to study how they affect the results of the pre-trained InstructPix2Pix model, in particular. 

More comparative results are available [here](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/g6cvggw2). Our code for comparing these models is available [here](https://github.com/huggingface/instruction-tuned-sd/blob/main/validation/compare_models.py). 

Our model, however, [fails to produce](https://wandb.ai/sayakpaul/instruction-tuning-sd/runs/g6cvggw2) the expected outputs for the classes from ImageNette, which it has not seen enough during training. This is somewhat expected, and we believe this could be mitigated by scaling the training dataset. 

### Low-level image processing results

For low-level image processing ([our model](https://huggingface.co/instruction-tuning-sd/low-level-img-proc)), we follow the same inference-time hyperparameters as above: 

- Number of inference steps: 20
- Image guidance scale: 1.5
- Guidance scale: 7.0

For deraining, our model provides compelling results when compared to the ground-truth and the output of the pre-trained InstructPix2Pix model:

| ![deraining_results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/deraining_results.png) |
|:--:|
| **Figure 9**: Deraining results (best viewed in color and zoomed in). Inference prompt: “derain the image” (same as the training set). See original [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/Instruction-tuning-sd/deraining_results.png). |

However, for low-light image enhancement, it leaves a lot to be desired: 

| ![image_enhancement_results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/image_enhancement_results.png) |
|:--:|
| **Figure 10**: Low-light image enhancement results (best viewed in color and zoomed in). Inference prompt: “enhance the low-light image” (same as the training set). See original [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/Instruction-tuning-sd/image_enhancement_results.png). |

This failure, perhaps, can be attributed to our model not seeing enough exemplars for the task and possibly from better training. We notice similar findings for deblurring as well: 

| ![deblurring_results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/instruction-tuning-sd/deblurring_results.png) |
|:--:|
| **Figure 11**: Deblurring results (best viewed in color and zoomed in). Inference prompt: “deblur the image” (same as the training set). See original [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/Instruction-tuning-sd/deblurring_results.png). |

We believe there is an opportunity for the community to explore how much the task mixture for low-level image processing affects the end results. *Does increasing the task mixture with more representative samples help improve the end results?* We leave this question for the community to explore further. 

You can try out the interactive demo below to make Stable Diffusion follow specific instructions:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.29.0/gradio.js"></script>

<gradio-app theme_mode="light" src="https://instruction-tuning-sd-instruction-tuned-sd.hf.space"></gradio-app>

## Potential applications and limitations

In the world of image editing, there is a disconnect between what a domain expert has in mind (the tasks to be performed) and the actions needed to be applied in editing tools (such as [Lightroom](https://www.adobe.com/in/products/photoshop-lightroom.html)). Having an easy way of translating natural language goals to low-level image editing primitives would be a seamless user experience. With the introduction of mechanisms like InstructPix2Pix, it’s safe to say that we’re getting closer to that realm. 

However, challenges still remain:

- These systems need to work for large high-resolution original images.
- Diffusion models often invent or re-interpret an instruction to perform the modifications in the image space. For a realistic image editing application, this is unacceptable.

## Open questions

We acknowledge that our experiments are preliminary. We did not go deep into ablating the apparent factors in our experiments. Hence, here we enlist a few open questions that popped up during our experiments:

- ***What happens we scale up the datasets?*** How does that impact the quality of the generated samples? We experimented with a handful of examples. For comparison, InstructPix2Pix was trained on more than 30000 samples.
- ***What is the impact of training for longer, especially when the task mixture is broader?*** In our experiments, we did not conduct hyperparameter tuning, let alone an ablation on the number of training steps.
- ***How does this approach generalize to a broader mixture of tasks commonly done in the “instruction-tuning” world?*** We only covered four tasks for low-level image processing: deraining, deblurring, denoising, and low-light image enhancement. Does adding more tasks to the mixture with more representative samples help the model generalize to unseen tasks or, perhaps, a combination of tasks (example: “Deblur the image and denoise it”)?
- ***Does using different variations of the same instruction on-the-fly help improve performance?***  For cartoonization, we randomly sampled an instruction from the set of ChatGPT-generated synonymous instructions **during** dataset creation. But what happens when we perform random sampling during training instead?
    
  For low-level image processing, we used fixed instructions. What happens when we follow a similar methodology of using synonymous instructions for each task and input image?  
    
- ***What happens when we use ControlNet training setup, instead?***  [ControlNet](https://huggingface.co/papers/2302.05543) also allows adapting a pre-trained text-to-image diffusion model to be conditioned on additional images (such as semantic segmentation maps, canny edge maps, etc.). If you’re interested, then you can use the datasets presented in this post and perform ControlNet training referring to [this post](https://huggingface.co/blog/train-your-controlnet).

## Conclusion

In this post, we presented our exploration of “instruction-tuning” of Stable Diffusion. While pre-trained InstructPix2Pix are good at following general image editing instructions, they may break when presented with more specific instructions. To mitigate that, we discussed how we prepared our datasets for further fine-tuning InstructPix2Pix and presented our results. As noted above, our results are still preliminary. But we hope this work provides a basis for the researchers working on similar problems and they feel motivated to explore the open questions further. 

## Links

- Training and inference code: [https://github.com/huggingface/instruction-tuned-sd](https://github.com/huggingface/instruction-tuned-sd)
- Demo: [https://huggingface.co/spaces/instruction-tuning-sd/instruction-tuned-sd](https://huggingface.co/spaces/instruction-tuning-sd/instruction-tuned-sd)
- InstructPix2Pix: [https://huggingface.co/timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)
- Datasets and models from this post: [https://huggingface.co/instruction-tuning-sd](https://huggingface.co/instruction-tuning-sd)

*Thanks to [Alara Dirik](https://www.linkedin.com/in/alaradirik/) and [Zhengzhong Tu](https://www.linkedin.com/in/zhengzhongtu) for the helpful discussions. Thanks to [Pedro Cuenca](https://twitter.com/pcuenq?lang=en) and [Kashif Rasul](https://twitter.com/krasul?lang=en) for their helpful reviews on the post.*

## Citation

To cite this work, please use the following citation:

```bibtex
@article{
  Paul2023instruction-tuning-sd,
  author = {Paul, Sayak},
  title = {Instruction-tuning Stable Diffusion with InstructPix2Pix},
  journal = {Hugging Face Blog},
  year = {2023},
  note = {https://huggingface.co/blog/instruction-tuning-sd},
}
```