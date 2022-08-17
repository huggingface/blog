---
title: Stable Diffusion using 🧨diffusers
thumbnail: /blog/assets/78_annotated-diffusion/thumbnail.png
---

<h1>
	Stable Diffusion using 🧨diffusers
</h1>

<div class="blog-metadata">
    <small>Published August 17th, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/stable_diffusion_with_diffusers.md">
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
    <a href="/patrickvonplaten">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/23423619?v=4" width="100" title="Gravatar">
        <div class="bfc">
            <code>patrickvonplaten</code>
            <span class="fullname">Patrick von Platen</span>
        </div>
    </a>
    
</div>

<a target="_blank" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# **Stable Diffusion** 🎨 
*...using `🧨diffusers`*

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.

**The Stable Diffusion weights are currently only available to universities, academics, research institutions and independent researchers. Please request access applying to <a href="https://stability.ai/academia-access-form" target="_blank">this</a> form**

This Colab notebook shows how to use Stable Diffusion with the 🤗 Hugging Face [D🧨ffusers library](https://github.com/huggingface/diffusers). 

Let's get started!

## 1. How to use `StableDiffusionPipeline`

First, you'll see how to run text to image inference in just a few lines of code! 

### Setup

First, please make sure you are using a GPU runtime to run this notebook, so inference is much faster. If the following command fails, use the `Runtime` menu above and select `Change runtime type`.


```python
!nvidia-smi
```


Let's install the required dependacies.


```python
!pip install diffusers==0.2.2
!pip install transformers scipy ftfy
```


In order to use Stable Diffusion, you need to have access to the pre-trained weights. You can request access for research purposes using the form in [the model card](https://huggingface.co/CompVis/stable-diffusion). Run the following cell to authenticate against the 🤗 Hugging Face Hub once you get access.

If you don't have access yet, public weights are expected to be released in a few days, so check back soon!


```python
from huggingface_hub import notebook_login

notebook_login()
```


### StableDiffusionPipeline

`StableDiffusionPipeline` is an end-to-end inference pipeline that you can use to generate images from text with just a few lines of code.

This is how to create the pipeline and load the pre-trained weights.


```python
from diffusers import StableDiffusionPipeline

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-3-diffusers", use_auth_token=True)  
```

Using `autocast` will run inference faster because it uses half-precision.


```python
from torch import autocast

prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
  image = pipe(prompt, guidance_scale=7)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can do either save it such as:
image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with 
image
```

    
![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_12_1.png)
    



Running the above cell multiple times will give you a different image every time. If you want deterministic output you can pass a random seed to the pipeline. Every time you use the same seed you'll have the same image result.


```python
import torch

generator = torch.Generator("cuda").manual_seed(1024)

with autocast("cuda"):
  image = pipe(prompt, guidance_scale=7, generator=generator)["sample"][0]

image
```


![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_14_1.png)
    



You can change the number of inference steps using the `num_inference_steps` argument. In general, results are better the more steps you use. Stable Diffusion, being one of the latest models, works great with a relatively small number of steps, so we recommend to use the default of `50`. If you want faster results you can use a smaller number.

The following cell uses the same seed as before, but with fewer steps. Note how some details, such as the horse's head or the helmet, are less defined than in the previous image:


```python
import torch

generator = torch.Generator("cuda").manual_seed(1024)

with autocast("cuda"):
  image = pipe(prompt, guidance_scale=7, num_inference_steps=20, generator=generator)["sample"][0]

image
```


![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_16_1.png)
    



The other parameter in the pipeline call is `guidance_scale`. It is a way to increase the adherence to the conditional signal which in this case is text as well as overall sample quality. In simple terms classifier free guidance forces the generation to better match with the prompt. Numbers like `7` or `8.5` give good results, if you use a very large number the images might look good, but will be less diverse. 

You can learn about the technical details of this parameter in [the last section](https://colab.research.google.com/drive/1ALXuCM5iNnJDNW5vqBm5lCtUQtZJHN2f?authuser=1#scrollTo=UZp-ynZLrS-S) of this notebook.

### Generate a grid of images

Let's first write a helper function to display a grid of images. Just run the following cell to create the `image_grid` function, or disclose the code if you are interested in how it's done.


```python
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
```

To generate multiple images for the same prompt, we simply use a list with the same prompt repeated several times. We'll send the list to the pipeline instead of the string we used before.


```python
num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images

with autocast("cuda"):
  images = pipe(prompt, guidance_scale=7.5)["sample"]

grid = image_grid(images, rows=1, cols=3)
grid
```


![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_22_1.png)
    

And here's how to generate a grid of `n × m` images.


```python
num_cols = 3
num_rows = 4

prompt = ["a photograph of an astronaut riding a horse"] * num_cols

all_images = []
for i in range(num_rows):
  with autocast("cuda"):
    images = pipe(prompt, guidance_scale=7.5)["sample"]
  all_images.extend(images)

grid = image_grid(all_images, rows=num_rows, cols=num_cols)
grid
```

    
![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_24_4.png)
    



### Generate non-square images

Stable Diffusion produces images of `512 × 512` pixels by default. But it's very easy to override the default using the `height` and `width` arguments, so you can create rectangular images in portrait or landscape ratios.

These are some recommendations to choose good image sizes:
- Make sure `height` and `width` are both multiples of `8`.
- Going below 512 might result in lower quality images.
- Going over 512 in both directions will repeat image areas (global coherence is lost).
- The best way to create non-square images is to use `512` in one dimension, and a value larger than that in the other one.


```python
prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
  image = pipe(prompt, height=512, width=768, guidance_scale=7.5)["sample"][0]
image
```


![png](assets/98_stable_diffusion_with_diffusers/stable_diffusion_with_diffusers_26_1.png)
    



## 2. What is Stable Diffusion

Next, we go a bit more in-detail about how ***Stable diffusion*** works.



Diffusion models are machine learning systems that are trained to *denoise* random gaussian noise step by step, to get to a sample of interest, such as an *image*. For a more detailed overview of how they work, check [this colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb).

Stable Diffusion is based on a particular type of diffusion model called **Latent Diffusion**, proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). This section takes a look at how Latent Diffusion works.

Diffusion models have shown to achieve state-of-the-art results for generating image data. But one downside of diffusion models is that the reverse denoising process is slow. In addition, these models consume a lot of memory because they operate in pixel space, which becomes huge when generating high-resolution images. Therefore, it is challenging to train these models and also use them for inference.

<br>

Latent diffusion can reduce the memory and compute complexity by applying the diffusion process over a lower dimensional _latent_ space, instead of using the actual pixel space. This is the key difference between standard diffusion and latent diffusion models: **in latent diffusion the model is trained to generate latent (compressed) representations of the images.** 

There are two main components in latent diffusion.

1. An autoencoder (VAE).
2. A [UNet network](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=wW8o1Wp0zRkq) that combines the encoder and decoder layers of the autoencoder. 


**The autoencoder (VAE)**

The VAE model has two parts, an encoder and a decoder. The encoder is used to convert the image into a low dimensional latent representation. The decoder, conversely, transforms the latent representation back into an image. During latent diffusion _training_, the encoder is used to get the latent representations (_latents_) of the images for the forward diffusion process, which applies more and more noise at each step. During _inference_, the latents generated by the reverse diffusion process are converted back into images using the VAE decoder.

**Why is latent diffusion fast and efficient?**

Since latent diffusion operates on a low dimensional space, it greatly reduces the memory and compute requirements compared to pixel-space diffusion models. For example, the autoencoder used in Stable Diffusion has a reduction factor of 8. This means that an image of shape `(3, 512, 512)` becomes `(3, 64, 64)` in latent space, which requires `8 × 8 = 64` times less memory.

This is why it's possible to generate `512 × 512` images so quickly, even on 16GB Colab GPUs!

**Stable Diffusion**

Stable Diffusion is a text-conditioned latent diffusion model trained on the [LAION dataset](https://laion.ai).

Stable Diffusion has three core components:

1. An autoencoder which maps an image to a low-dimensional latent space, and the latent representation back to an image. In Stable Diffusion, the autoencoder is implemented in the `AutoencodeKL` model.
2. A text encoder model that converts text prompts to embeddings that can be passed as conditions to the UNet. In Stable Diffustion, the [CLIP model](https://openai.com/blog/clip/) is used for this purpose.
3. A UNet model trained to generate the low-dimentional latent space. Stable Diffusion uses cross-attention the the model blocks to condition on the text.


This is how the stable (latent) diffusion inference looks like:

<p align="center">
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png" alt="sd-pipeline" width="500"/>
</p>

After this brief introduction to Latent and Stable Diffusion, let's see how to make advanced use of 🤗 Hugging Face Diffusers!

## 3. How to write your own inference pipeline with `diffusers`

Finally, we show how you can create custom diffusion pipelines with `diffusers`.
This is often very useful to dig a bit deeper into certain functionalities of the system and to potentially switch out certain components. 

In this section, we will demonstrate how to use Stable Diffusion with a different scheduler, namely [Katherine Crowson's](https://github.com/crowsonkb) K-LMS scheduler that was added in [this PR](https://github.com/huggingface/diffusers/pull/185#pullrequestreview-1074247365).

Let's go through the `StableDiffusionPipeline` step by step to see how we could have written it ourselves.

We will start by loading the individual models involved.


```python
import torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
```

The [pre-trained model](https://huggingface.co/CompVis/stable-diffusion-v1-3-diffusers/tree/main) includes all the components required to setup a complete diffusion pipeline. They are stored in the following folders:
- `text_encoder`: Stable Diffusion uses CLIP, but other diffusion models may use other encoders such as `BERT`.
- `tokenizer`. It must match the one used by the `text_encoder` model.
- `scheduler`: The scheduling algorithm used to progressively add noise to the image during training.
- `unet`: The model used to generate the latent representation of the input.
- `vae`: Autoencoder module that we'll use to decode latent representations into real images.

We can load the components by referring to the folder they were saved, using the `subfolder` argument to `from_pretrained`.


```python
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-3-diffusers", subfolder="vae", use_auth_token=True)

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-3-diffusers", subfolder="unet", use_auth_token=True)
```


Now instead of loading the pre-defined scheduler, we load a K-LMS scheduler instead.


```python
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
```

Next we move the models to the GPU.


```python
%%capture
from torch import autocast

vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 
```

We now define the parameters we'll use to generate images.

Note that `guidance_scale` is defined analog to the guidance weight `w` of equation(2) in the [Imagen paper](https://arxiv.org/pdf/2205.11487.pdf). `guidance_scale == 1` corresponds to doing no classifier-free guidance.


```python
prompt = ["a photograph of an astronaut riding a horse"]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100            # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)   # Seed generator to create the inital latent noise

batch_size = len(prompt)
```

First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.


```python
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

We'll also get the unconditional text embeddings for classifier-free guidance, which are just the embeddings for the padding token (empty text). They need to have the same shape as the conditional `text_embeddings` (`batch_size` and `seq_length`)


```python
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
```

For classifier-free guidance, we need to do two forward passes. One with the conditioned input (`text_embeddings`), and another with the unconditional embeddings (`uncond_embeddings`). In practice, we can concatenate both into a single batch to avoid doing two forward passes.


```python
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

Generate the intial random noise.


```python
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)
```


```python
latents.shape
```




    torch.Size([1, 4, 64, 64])



Cool $64 \times 64$ is expected. The model will transform this latent representation (pure noise) into a `512 × 512` image later on.

The K-LMS scheduler needs to multiple the `latents` by its `sigma` values. Let's do this here


```python
latents = latents * scheduler.sigmas[0]
```

We are ready to write the denoising loop.


```python
from tqdm.auto import tqdm
from torch import autocast

scheduler.set_timesteps(num_inference_steps)

with autocast("cuda"):
  for i, t in tqdm(enumerate(scheduler.timesteps)):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

    # predict the noise residual
    with torch.no_grad():
      noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
```

We now use the `vae` to decode the generated `latents` back into the image.


```python
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
image = vae.decode(latents)
```

And finally, let's convert the image to PIL so we can display or save it.


```python
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]
```
