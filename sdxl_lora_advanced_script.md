<h1>LoRA training scripts of the world, unite!
</h1>


<h2>A community derived guide to some of the SOTA practices for SD-XL Dreambooth LoRA fine tuning
</h2>

<h3>tl;dr
</h3>

We combined the Pivotal Tuning technique used on Replicate's SDXL Cog trainer with the Prodigy optimizer used in the
Kohya trainer (plus a bunch of other optimizations) to achieve very good results on training a Dreambooth LoRA for SDXL.
[Check out the training script on diffusers]()🧨. [Try it out on Colab]().

If you want to skip the technical talk, you can use all the techniques in this blog
and [train on Hugging Face Spaces with a simple UI]() and curated parameters (that you can meddle with).

<h3>Overview
</h3>

Stable Diffusion XL (SDXL) models fine-tuned with LoRA dreambooth achieve incredible results at capturing new concepts using only a
handful of images, while simultaneously maintaining the aesthetic and image quality of SDXL and requiring relatively
little compute and resources. Check out some of the awesome SDXL
LoRAs [here](https://huggingface.co/spaces/multimodalart/LoraTheExplorer).  
In this blog, we'll review some of the popular practices and techniques to make your LoRA finetunes go brrr, and show how you
can use them now with diffusers!

Recap: LoRA (Low-Rank Adaptation) is a fine-tuning technique for Stable Diffusion models that makes slight
adjustments to the crucial cross-attention layers where images and prompts intersect. It achieves quality on par with
full fine-tuned models while being much faster and requiring less compute. To learn more on how LoRAs work please see
our previous post - [Using LoRA for Efficient Stable Diffusion Fine-Tuning](https://huggingface.co/blog/lora).

Contents:

1. Techniques/tricks
    1. Pivotal tuning
    2. Adaptive optimizers
    3. Recommended practices - Text encoder learning rate, custom captions, dataset repeats, min snr gamma, training set
       creation \

2. Experiments Settings and Results
3. Inference

**Acknowledgements** ❤️: 
The techniques showcased in this guide – training scripts, experiments and explorations – were inspired and built upon the 
contributions by [Simo Ryu](https://twitter.com/cloneofsimo), [cog-sdxl](https://github.com/replicate/cog-sdxl), 
[Kohya](https://twitter.com/kohya_tech/): [sd-scripts](https://github.com/kohya-ss/sd-scripts), [The Last Ben](https://twitter.com/__TheBen): [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion). Our most sincere gratitude to them and the rest of the community! 🙌 


<h2>Pivotal Tuning</h2>

[Pivotal Tuning](https://arxiv.org/abs/2106.05744) is a method that combines Textual Inversion with regular diffusion fine-tuning. For Dreambooth, it is
customary that you provide a rare token to be your trigger word, say "an sks dog", however, those tokens usually have
other semantic meaning associated with them and can affect your results. The sks example, popular in the community, is
actually associated with a weapons brand.

To tackle this issue, we insert new tokens into the text encoders of the model, instead of reusing existing ones.
We then optimize the newly-inserted token embeddings to represent the new concept: that is Textual Inversion –
we learn to represent the concept through new "words" in the embedding space. Once we obtain the new token and its
embeddings to represent it, we can train our Dreambooth LoRA with those token embeddings to get the best of both worlds.

**Training**

In our new training script, you can do textual inversion training by providing the following arguments

```
--train_text_encoder_ti
--train_text_encoder_ti_frac=0.5
--token_abstraction="TOK"
--num_new_tokens_per_abstraction=2
--adam_weight_decay_text_encoder
```

* `train_text_encoder_ti` activates training the embeddings of new concepts
* `train_text_encoder_ti_frac` and tells us when to stop optimizing such new concepts. Pivoting halfway is the default
  value in the cog sdxl example and we have seen good results with it. We encourage experimentation here.
* `token_abstraction` lets you choose a word to use in your instance prompt, validation prompt or custom captions to
  represent the new tokens being trained. Here we chose TOK. So, for example, "a photo of a TOK" as an instance prompt
  would then insert the newly trained tokens in place of `TOK`
* `num_new_tokens_per_abstraction` sets up how many new tokens to insert and train for each text encoder
  of the model. The default is set to 2, but you can increase that for complex concepts.
* `adam_weight_decay_text_encoder` This is used to set a different weight decay value for the text encoder parameters (
  different from the value used for the unet parameters).`

**Inference**

When doing pivotal tuning, besides the `*.safetensors` weights of your LoRA, you also get the `*.safetensors` embeddings
for the new tokens. In order to do inference with those we add 2 steps to how we would normally load a LoRA:

1. Download our trained embeddings from the hub

```py
import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")
# normal loading
```

2. Load the embeddings into the text encoders

```py
# download embeddings
embedding_path = hf_hub_download(repo_id="LinoyTsaban/web_y2k_lora", filename="embeddings.safetensors", repo_type="model")

# load embeddings to the text encoders
state_dict = load_file(embedding_path)

# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (CLIP ViT-G/14)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```

3. Load your LoRA and prompt it!

```py
pipe.load_lora_weights("LinoyTsaban/web_y2k_lora", weight_name="pytorch_lora_weights.safetensors")
prompt="a <s0><s1> webpage about an astronaut riding a horse"
images = pipe(
    prompt,
    cross_attention_kwargs={"scale": 0.8},
).images
# your output image
images[0]
```
![astronaut_web_y2k](.\assets\dreambooth_lora_sdxl\web_y2k_astronaut.png)


**Comfy UI / AUTOMATIC1111 Inference:**
The new script fully supports textual inversion loading with Comfy UI and AUTOMATIC1111 formats!
<h3>AUTOMATIC1111 / SD.Next</h3>
For AUTOMATIC1111/SD.Next we will load a LoRA and a textual embedding at the same time. For that you can download your `embeddings.safetensors` file from a trained model, rename it (to for example `y2k_emb.safetensors` and include it in your `embeddings` directory. You can also download `diffusers_lora_weights.safetensors`, rename it (to for example `y2k.safetensors`) and include it in your `models/lora` directory. 

You can then inference by prompting `a y2k_emb webpage about the movie Mean Girls <lora:y2k:0.9>`. You can use the `y2k_emb` token normally, including increasing its weight by doing `(y2k_emb:1.2)`. 

<h3>ComfyUI</h3>
For ComfyUI you will include your the trained `embeddings.safetensors` in the `models/embeddings` folder, rename it (to for example `y2k_emb.safetensors`) and use it in your prompts like `embedding:y2k_emb`. [Official guide for loading embeddings](https://comfyanonymous.github.io/ComfyUI_examples/textual_inversion_embeddings/). 

For the LoRA, you will include your trained `diffusers_lora_weights.safetensors` rename it (to for example `y2k.safetensors`) and include it in your `models/lora` directory. Then you will load the LoRALoader node and hook that up with your model and CLIP. [Official guide for loading LoRAs](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
```

```

<h2>Adaptive Optimizers</h2>

![optimization_gif](.\assets\dreambooth_lora_sdxl\optimization_gif.gif)

When training/fine-tuning a diffusion model (or any machine learning model for that matter), we use optimizers to guide
us towards the optimal path that leads to convergence of our training objective - a minimum point of our chosen loss
function that represents a state where the model learned what we are trying to teach it. The standard (and
state-of-the-art) choices for deep learning tasks are the Adam and AdamW optimizers.

However, they require the user to meddle a lot with the hyperparameters that pave the path to convergence (such as
learning rate, weight decay, etc.). This can result in time-consuming experiments that lead to suboptimal outcomes, and
even if you land on an ideal learning rate, it may still lead to convergence issues if the learning rate is constant
during training. Some parameters may benefit from more frequent updates to expedite convergence, while others may
require smaller adjustments to avoid overshooting the optimal value. To tackle this challenge, algorithms with adaptable
learning rates such as **Adafactor** and [**Prodigy**](https://github.com/konstmish/prodigy) have been introduced. These
methods optimize the algorithm's traversal of the optimization landscape by dynamically adjusting the learning rate for
each parameter based on their past gradients.

We chose to focus a bit more on Prodigy as we think it can be especially beneficial for Dreambooth LoRA training!

**Training**

```
--optimizer="prodigy"
```

When using prodigy it's generally good practice to set-

```
--learning_rate=1.0
```

Additional settings that are considered beneficial for diffusion models and specifically LoRA training are:

```
--prodigy_safeguard_warmup=True
--prodigy_use_bias_correction=True
--adam_beta1=0.9
# Note these are set to values different than the default:
--adam_beta2=0.99 
--adam_weight_decay=0.01
```

There are additional hyper-parameters you can adjust when training with prodigy
(like- `--prodigy_beta3`, `prodigy_decouple`, `prodigy_safeguard_warmup`), we will not delve into those in this post,
but you can learn more about them [here](https://github.com/konstmish/prodigy).

<h2>Additional Good Practices</h2>

Besides pivotal tuning and adaptive optimizers, here are some additional techniques that can impact the quality of your
trained LoRA, all of them have been incorporated into the new diffusers training script.

* <h3>separate learning rate for text-encoder and unet</h3>

  When optimizing the text encoder, it's been perceived by the community that setting different learning rates for it (
  versus the learning rate of the Unet) can lead to better quality results - specifically a **lower** learning rate for
  the text encoder as it tends to overfit _faster_.
    * The importance of different unet and text encoder learning rates is evident when performing pivotal tuning as
      well- in this case, setting a higher learning rate for the text encoder is perceived to be better.
    * Notice, however, that when using Prodigy (or adaptive optimizers in general) we start with an identical initial
      learning rate for all trained parameters, and let the optimizer work it's magic ✨

**Training**

```
--train_text_encoder
--learning_rate=1e-4 #unet
--text_encoder_lr=5e-5 
```

```
--train_text_encoder - Enables full text encoder training (i.e. Text Encoders weights are optimized vs. textual inversion (--train_text_encoder_ti) where we optimize embeddings but don't modify the weights)
If you wish the text encoder lr to always match --learning_rate, set --text_encoder_lr=None

```

* <h3>Custom Captioning</h3>

  While it is possible to achieve good results by training on a set of images all captioned with the same instance
  prompt, e.g. "photo of a <token> person" or "in the style of <token>" etc, using the same caption may lead to
  suboptimal results, depending on the complexity of the learned concept, how "familiar" the model is with the concept,
  and how well the training set captures it.
  ![meme](.\assets\dreambooth_lora_sdxl\custom_captions_meme.png)

**Training**
To use custom captioning, first ensure that you have the datasets library installed, otherwise you can install it by -

```
!pip install datasets
```

To load the custom captions we need our training set directory to follow the structure of a datasets ImageFolder,
containing both the images and the corresponding caption for each image.

* _Option 1_:
  You choose a dataset from the hub that already contains images and prompts - for example XXX. Now all you have to do
  is

```

--dataset_name= 
--caption_column=

```

* _Option 2_:
  You wish to use your own images and add captions to them. In that case, you can use [this colab notebook]() to
  automatically caption the images with BLIP, or you can manually create the captions in a metadata file. Then you
  follow up the same way, by specifying `--dataset_name` with your folder path, and `--caption_column` with the column
  name for the captions

* <h3>Min SNR Gamma</h3>
  Training diffusion models often suffers from slow convergence, partly due to conflicting optimization directions
  between timesteps. [Hang et al.](https://arxiv.org/abs/2303.09556) found a way to mitigate this issue by introducing
  the simple Min-SNR-gamma approach. This method adapts loss weights of timesteps based on clamped signal-to-noise
  ratios, which effectively balances the conflicts among timesteps.
    * For small datasets, the effects of Min-SNR weighting strategy might not appear to be pronounced, but for larger
      datasets, the effects will likely be more pronounced.
    * `snr vis`
      _find [this project on Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) that compares
      the loss surfaces of the following setups: snr_gamma set to 5.0, 1.0 and None._

**Training**

To use min snr gamma, set a value for:

```
--snr_gamma=5.0
```

By default `--snr_gamma=None`, I.e. not used. When enabling `--snr_gamma`, the recommended value is 5.0.

* <h3>Repeats</h3>
  This argument refers to the number of times an image from your dataset is repeated in the training set. This differs
  from epochs in that first the images are repeated, and only then shuffled.
![meme](.\assets\dreambooth_lora_sdxl\snr_gamma_effect.png)
**Training**

To enable repeats simply set an integer value > 1 as your repeats count-

```
--repeats
```

By default, --repeats=1, i.e. training set is not repeated

* <h3>Training Set Creation</h3>
    * As the popular saying goes - “Garbage in - garbage out” Training a good Dreambooth LoRA can be done easily using
      only a handful of images, but the quality of these images is very impactful on the fine tuned model.

    * Generally, when fine-tuning on an object/subject, we want to make sure the training set contains images that
      portray the object/subject in as many distinct ways we would want to prompt for it as possible.
    * For example, if my concept is this red backpack: (available
      in [google/dreambooth](https://huggingface.co/datasets/google/dreambooth) dataset)
    ![backpack_01](.\assets\dreambooth_lora_sdxl\dreambooth_backpack_01.jpg)
    * I would likely want to prompt it worn by people as well, so having examples like this: 
    ![backpack_02](.\assets\dreambooth_lora_sdxl\dreambooth_backpack_02.jpg)
    in the training set - that fits that scenario - will likely make it easier for the model to generalize to that 
      setting/composition during inference.

  _Specifically_ when training on _faces_, you might want to keep in mind the following things regarding your dataset:

    1. If possible, always choose **high resolution, high quality** images. Blurry or low resolution images can harm the
       tuning process.

    2. When training on faces, it is recommended that no other faces appear in the training set as we don't want to
       create an ambiguous notion of what is the face we're training on.
    3. **Close-up photos** are important to achieve realism, however good full-body shots should also be included to
       improve the ability to generalize to different poses/compositions.
    4. We recommend **avoiding photos where the subject is far away**, as most pixels in such images are not related to
       the concept we wish to optimize on, there's not much for the model to learn from these.
    5. Avoid repeating backgrounds/clothing/poses - aim for **variety** in terms of lighting, poses, backgrounds, and
       facial expressions. The greater the diversity, the more flexible and generalizable the LoRA would be.
    6. **Class images** -  \
       _real images for regularization VS model generated ones_ \
       When choosing class images, you can decide between synthetic ones (i.e. generated by the diffusion model) and
       real ones. In favor of using real images, we can argue they improve the fine-tuned model's realism. On the other
       hand, some will argue that using model generated images better serves the purpose of preserving the models <em>
       knowledge </em>of the class and general aesthetics.</code></strong>
    7. **Celebrity lookalike** - this is more a comment on the captioning/instance prompt used to train. Some fine
       tuners experienced improvements in their results when prompting with a token identifier + a public person that
       the base model knows about that resembles the person they trained on.

**Training** with prior preservation loss

```
--with_prior_preservation
--class_data_dir
--num_class_images
--class_prompt
```

`--with_prior_preservation` - enables training with prior preservation \
`--class_data_dir` - path to folder containing class images\
`—-num_class_images` - Minimal class images for prior preservation loss. If there are not enough images already present
in --class_data_dir, additional images will be sampled with -class_prompt

<h3> Experiments Settings and Results </h3> 
To explore the described methods, we experimented with different combinations of these techniques on 
different objectives (style tuning, faces and objects). 

In order to narrow down the infinite amount of hyperparameters values, we used some of the more popular and common 
configurations as starting points and tweaked our way from there. 

<h4> Huggy Dreambooth LoRA </h4> 
First, we were interested in fine-tuning a huggy LoRA which means 
both teaching an artistic style, and a specific character at the same time. 
For this example, we curated a high quality Huggy mascot dataset (using Chunte-Lee’s amazing artwork) containing 31 
images paired with custom captions.

![huggy_data_example](.\assets\dreambooth_lora_sdxl\huggy_dataset_example.png)
Configurations:
```
--train_batch_size = 1, 4
-repeats = 1,2
-learning_rate = 1.0 (Prodigy), 1e-4 (AdamW)
-text_encoder_lr = 1.0 (Prodigy), 3e-4, 5e-5 (AdamW)
-snr_gamma = None, 5.0 
-max_train_steps = 1000, 1500
-text_encoder_training = regular finetuning, pivotal tuning (textual inversion)

```
* Full Text Encoder Tuning VS Pivotal Tuning - we noticed pivotal tuning achieves results competitive or better 
  than full text encoder training and yet without optimizing the weights of the text_encoder.
* Min SNR Gamma
  * We compare between a [version1](https://wandb.ai/linoy/dreambooth-lora-sd-xl/runs/mvox7cqg?workspace=user-linoy) 
    trained without `snr_gamma`, and a [version2](https://wandb.ai/linoy/dreambooth-lora-sd-xl/runs/cws7nfzg?workspace=user-linoy) trained with `snr_gamma = 5.0`
Specifically we used the following arguments in both versions (and added `snr_gamma` to version 2)
``` 
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
--dataset_name="./huggy_clean" \
--instance_prompt="a TOK emoji"\
--validation_prompt="a TOK emoji dressed as Yoda"\
--caption_column="prompt" \
--mixed_precision="bf16" \
--resolution=1024 \
--train_batch_size=4 \
--repeats=1\
--report_to="wandb"\
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--learning_rate=1e-4 \
--text_encoder_lr=3e-4 \
--optimizer="adamw"\
--train_text_encoder_ti\
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--rank=32 \
--max_train_steps=1000 \
--checkpointing_steps=2000 \
--seed="0" \
```
![huggy_snr_example](.\assets\dreambooth_lora_sdxl\snr_comparison_huggy.png)


* AdamW vs Prodigy Optimizer
  * We compare between [version1](https://wandb.ai/linoy/dreambooth-lora-sd-xl/runs/uk8d6k6j?workspace=user-linoy) 
    trained with `optimizer=prodigy`, and [version2](https://wandb.ai/linoy/dreambooth-lora-sd-xl/runs/cws7nfzg?
    workspace=user-linoy) trained with `optimizer=adamW`. Both version were trained with pivotal tuning. 
  * When training with `optimizer=prodigy` we set the initial learning rate to be 1. For adamW we used the default 
    learning rates used for pivotal tuning in cog-sdxl (`1e-4`, `3e-4` for `learning_rate` and `text_encoder_lr` respectively) 
    as we were able to reproduce good 
    results with these settings. 
![huggy_optimizer_example](.\assets\dreambooth_lora_sdxl\adamw_prodigy_comparsion_huggy.png)
  * all other training parameters and settings were the same. Specifically: 
``` 
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="./huggy_clean" \
  --instance_prompt="a TOK emoji"\
  --validation_prompt="a TOK emoji dressed as Yoda"\
  --output_dir="huggy_v11" \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=4 \
  --repeats=1\
  --report_to="wandb"\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --train_text_encoder_ti\
  --lr_scheduler="constant" \
  --snr_gamma=5.0 \
  --lr_warmup_steps=0 \
  --rank=32 \
  --max_train_steps=1000 \
  --checkpointing_steps=2000 \
  --seed="0" \
``` 


<h4> Y2K Webpage LoRA </h4> 
Let's explore another example, this time training on a dataset composed of 27 screenshots of webpages from the 1990s 
and early 2000s that we (nostalgically) scraped from the internet.

This example showcases a slightly different behaviour than the previous. 
While in both cases we used approximately the same amount of images ~30, 
we noticed 


Configurations:
```
–rank = 4,16,32
-optimizer = prodigy, adamW
-repeats = 1,2,3
-learning_rate = 1.0 (Prodigy), 1e-4 (AdamW)
-text_encoder_lr = 1.0 (Prodigy), 3e-4, 5e-5 (AdamW)
-snr_gamma = None, 5.0 
-train_batch_size = 1, 2, 3, 4
-max_train_steps = 500, 1000, 1500
-text_encoder_training = regular finetuning, pivotal tuning
```
`*assets- Web Y2K*`

Face LoRA
* Linoy face Datasets 
  * v1 - 7 close up photos taken at the same time 
  * v1.5 - 16 close up photos taken at different occasions (changing backgrounds, lighting and outfits)
  * v2 - 13 close up photos and fully body shots taken at different occasions (changing backgrounds, lighting and outfits)
  
```
rank = 4,16,32
optimizer = prodigy, adamW
repeats = 1,2,3,4
learning_rate = 1.0 , 1e-4
text_encoder_lr = 1.0, 3e-4
snr_gamma = None, 5.0 
max_train_steps = 1000, 1500
text_encoder_training = regular finetuning, pivotal tuning
```

<h3> What’s next? </h3> 

🚀 More features coming soon!
We are working on adding even more control and flexibility to our advanced training script. Let us know what features
you find most helpful!

🤹 Multi concept LoRAs
A recent [work](https://ziplora.github.io/) of Shah et al. introduced ZipLoRAs - a method to 
merge independently trained style and subject LoRAs in order to achieve generation of any user-provided subject in 
any user-provided style. [mkshing](https://twitter.com/mk1stats) implemented an open source replication available 
[here](https://github.com/mkshing/ziplora-pytorch) and it uses the new and improved [script](https://github.com/mkshing/ziplora-pytorch/blob/main/train_dreambooth_lora_sdxl.py).
