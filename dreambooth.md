---
title: Training Stable Diffusion with Dreambooth using Diffusers
thumbnail: /blog/assets/sd_dreambooth_training/thumbnail.jpg
---

<h1>
	Training Stable Diffusion with Dreambooth using 🧨 Diffusers
</h1>

<div class="blog-metadata">
    <small>Published November 3rd, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/stable_diffusion_dreambooth_training.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/valhalla">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/27137566?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>valhalla</code>
            <span class="fullname">Suraj Patil</span>
        </div>
    </a>
	 <a href="/pcuenq">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/1177582?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>pcuenq</code>
            <span class="fullname">Pedro Cuenca</span>
        </div>
    </a>
	 <a href="/NineOfNein">
        <img class="avatar avatar-user" src="https://pbs.twimg.com/profile_images/1262423730870988806/qA2FeHXF_400x400.jpg" width="100" title="Gravatar">
        <div class="bfc">
            <code>NineOfNein</code>
            <span class="fullname">Valentine Kozin</span>
            <span class="bg-gray-100 dark:bg-gray-700 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
    </a>
</div>

[Dreambooth](https://dreambooth.github.io/) is a technique to teach new concepts to Stable Diffusion using a specialized form of fine-tuning. Some people have been using it with a few of their photos to place themselves in fantastic situations, while others are using it to incorporate new styles. [🧨 Diffusers](https://github.com/huggingface/diffusers) provides a Dreambooth [training script](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). It doesn't take long to train, but it's hard to select the right set of hyperparameters and it's easy to overfit.

We conducted a lot of experiments to analyze the effect of different settings in Dreambooth. This post presents our findings and some tips to improve your results when fine-tuning Stable Diffusion with Dreambooth.

Before we start, please be aware that this method should never be used for malicious purposes, to generate harm in any way, or to impersonate people without their knowledge. Models trained with it are still bound by the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) that governs distribution of Stable Diffusion models.

_Note: a previous version of this post was published [as a W&B report](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)_.

## TL;DR: Recommended Settings

* Dreambooth tends to overfit quickly. To get good-quality images, we must find a 'sweet spot' between the number of training steps and the learning rate. We recommend using a low learning rate and progressively increasing the number of steps until the results are satisfactory.
* Dreambooth needs more training steps for faces. In our experiments, 800-1200 steps worked well when using a batch size of 2 and LR of 1e-6.
* Prior preservation is important to avoid overfitting when training on faces. For other subjects, it doesn't seem to make a huge difference.
* If you see that the generated images are noisy or the quality is degraded, it likely means overfitting. First, try the steps above to avoid it. If the generated images are still noisy, use the DDIM scheduler or run more inference steps (~100 worked well in our experiments).
* Training the text encoder in addition to the UNet has a big impact on quality. Our best results were obtained using a combination of text encoder fine-tuning, low LR, and a suitable number of steps. However, fine-tuning the text encoder requires more memory, so a GPU with at least 24 GB of RAM is ideal. Using techniques like 8-bit Adam, `fp16` training or gradient accumulation, it is possible to train on 16 GB GPUs like the ones provided by Google Colab or Kaggle.
* Fine-tuning with or without EMA produced similar results.

## Learning Rate Impact

Dreambooth overfits very quickly. To get good results, tune the learning rate and the number of training steps in a way that makes sense for your dataset. In our experiments (detailed below), we fine-tuned on four different datasets with high and low learning rates. In all cases, we got better results with a low learning rate.

## Experiments Settings

All our experiments were conducted using the [`train_dreambooth.py`](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) script with the `AdamW` optimizer on 2x 40GB A100s. We used the same seed and kept all hyperparameters equal across runs, except LR, number of training steps and the use of prior preservation.

For the first 3 examples (various objects), we fine-tuned the model with a batch size of 4 (2 per GPU) for 400 steps. We used a high learning rate of `5e-6` and a low learning rate of `2e-6`. No prior preservation was used.

The last experiment attempts to add a human subject to the model. We used prior preservation with a batch size of 2 (1 per GPU), 800 and 1200 steps in this case. We used a high learning rate of `5e-6` and a low learning rate of `2e-6`.

Note that you can use 8-bit Adam, `fp16` training or gradient accumulation to reduce memory requirements and run similar experiments on GPUs with 16 GB of memory.

### Cat Toy

High Learning Rate (`5e-6`)

![Cat Toy, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/1_cattoy_hlr.jpg)

Low Learning Rate (`2e-6`)
![Cat Toy, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/2_cattoy_llr.jpg)

### Pighead

High Learning Rate (`5e-6`). Note that the color artifacts are noise remnants – running more inference steps could help resolve some of those details.
![Pighead, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/3_pighead_hlr.jpg)

Low Learning Rate (`2e-6`)
![Pighead, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/4_pighead_llr.jpg)

### Mr. Potato Head

High Learning Rate (`5e-6`). Note that the color artifacts are noise remnants – running more inference steps could help resolve some of those details.
![Potato Head, High Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/5_potato_hlr.jpg)

Low Learning Rate (`2e-6`)
![Potato Head, Low Learning Rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/6_potato_llr.jpg)

### Human Face

We tried to incorporate the Cosmo character from Seinfeld into Stable Diffusion. As previously mentioned, we trained for more steps with a smaller batch size. Even so, the results were not stellar. For the sake of brevity, we have omitted these sample images and defer the reader to the next sections, where face training became the focus of our efforts.

### Summary of Initial Results

To get good results training Stable Diffusion with Dreambooth, it's important to tune the learning rate and training steps for your dataset.

* High learning rates and too many training steps will lead to overfitting. The model will mostly generate images from your training data, no matter what prompt is used.
* Low learning rates and too few steps will lead to underfitting: the model will not be able to generate the concept we were trying to incorporate.

Faces are harder to train. In our experiments, a learning rate of `2e-6` with `400` training steps works well for objects but faces required `1e-6` (or `2e-6`) with ~1200 steps.

Image quality degrades a lot if the model overfits, and this happens if:
* The learning rate is too high.
* We run too many training steps.
* In the case of faces, when no prior preservation is used, as shown in the next section.

## Using Prior Preservation when training Faces

Prior preservation is a technique that uses additional images of the same class we are trying to train as part of the fine-tuning process. For example, if we try to incorporate a new person into the model, the _class_ we'd want to preserve could be _person_. Prior preservation tries to reduce overfitting by using photos of the new person and other people. The nice thing is that we can generate those additional class images using the Stable Diffusion model itself! The training script takes care of that automatically.

Prior preservation, 1200 steps, lr=`2e-6`.
![Faces, prior preservation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/7_faces_with_prior.jpg)

No prior preservation, 1200 steps, lr=`2e-6`.
![Faces, prior preservation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/8_faces_no_prior.jpg)

As you can see, results are better when prior preservation is used, but there are still noisy blotches. It's time for some additional tricks!

## Effect of Schedulers

In the previous examples, we used the `PNDM` scheduler to sample images during the inference process. We observed that when the model overfits, `DDIM` usually works much better than `PNDM` and `LMSDiscrete`. In addition, quality can be improved by running inference for more steps: 100 seems to be a good choice. The additional steps help resolve some of the noise patches into image details.

`PNDM`, Cosmo face
![PNDM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/9_cosmo_pndm.jpg)

`LMSDiscrete`, Cosmo face. Results are terrible!
![LMSDiscrete Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/a_cosmo_lmsd.jpg)

`DDIM`, Cosmo face. Much better
![DDIM Cosmo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/b_cosmo_ddim.jpg)

A similar behaviour can be observed for other subjects, although to a lesser extent.

`PNDM`, Potato Head
![PNDM Potato](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/c_potato_pndm.jpg)

`LMSDiscrete`, Potato Head
![LMSDiscrite Potato](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/d_potato_lmsd.jpg)

`DDIM`, Potato Head
![DDIM Potato](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/e_potato_ddim.jpg)

## Fine-tuning the Text Encoder

The original Dreambooth paper describes a method to fine-tune the UNet component of the model but keeps the text encoder frozen. However, we observed that fine-tuning the encoder produces better results. We experimented with this approach after seeing it used in other Dreambooth implementations, and the results are striking!

Frozen text encoder
![Frozen text encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/f_froxen_encoder.jpg)

Fine-tuned text encoder
![Fine-tuned text encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/g_unfrozen_encoder.jpg)

Fine-tuning the text encoder produces the best results, especially with faces. It generates more realistic images, it's less prone to overfitting and it also achieves better prompt interpretability, being able to handle more complex prompts.

## Epilogue: Textual Inversion + Dreambooth

We also ran a final experiment where we combined [Textual Inversion](https://textual-inversion.github.io) with Dreambooth. Both techniques have a similar goal, but their approaches are different.

In this experiment we first ran textual inversion for 2000 steps. From that model, we then ran Dreambooth for an additional 500 steps using a learning rate of `1e-6`. These are the results:

![Textual Inversion + Dreambooth](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dreambooth-assets/h_textual_inversion_dreambooth.jpg)

We think the results are much better than doing plain Dreambooth but not as good as when we fine-tune the whole text encoder. It seems to copy the style of the training images a bit more, so it could be overfitting to them. We didn't explore this combination further, but it could be an interesting alternative to improve Dreambooth and still fit the process in a 16GB GPU. Feel free to explore and tell us about your results!
