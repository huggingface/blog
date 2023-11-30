<h1>LoRA training scripts of the world, unite!
</h1>


<h2>A community derived guide to some of the SOTA practices for SD-XL Dreambooth LoRA fine tuning
</h2>

<h3>tl;dr
</h3>

We combined the Pivotal Tuning technique used on Replicate's SDXL Cog trainer with the Prodigy optimizer used in the
Kohya trainer (plus a bunch of other optimizations) to achieve very good results on training a Dreambooth LoRA for SDXL.
[Check out the training script on diffusers]()🧨. [Try it out on Colab]().

If you want to skip the technical talk, you can use all the techniques in this blog and [train on Hugging Face Spaces
with a simple UI]() and curated parameters (that you can meddle with).

<h3>Overview
</h3>

LoRA dreambooth fine-tuned stable diffusion xl models achieved incredible results at capturing new concepts using only a
handful of images, while simultaneously maintaining the aesthetic and image quality of sd-xl and requiring relatively
little compute and resources. Check out some of the awesome SD-XL LoRAs [here](https://huggingface.co/spaces/multimodalart/LoraTheExplorer).  
In this blog, we'll review some of the popular practices and techniques to make your LoRA finetune go brrr, and how you
can use them now with diffusers!

Recap: LoRA (Low-Rank Adaptation) is a training technique for fine-tuning Stable Diffusion models by making slight
adjustments to the crucial cross-attention layers where images and prompts intersect - achieving quality on par with
full fine tuned models while being much faster and requiring less compute. To learn more on how LoRAs work please see
our blog post - Using LoRA for Efficient Stable Diffusion Fine-Tuning.

*insert credits to the guides used/contributors*

Contents:

1. `Techniques/tricks `
    1. `Pivotal tuning`
    2. `Adaptive optimizers `
    3. `Recommended practices - Text encoder learning rate, custom captions, dataset repeats, min snr gamma, training set creation \
       `
2. `Experiments Settings and Results`
3. `Inference `

<h2>Pivotal Tuning</h2>

Pivotal Tuning is a method that combines training Textual Inversion with regular training. For Dreambooth, it is
customary that you provide a rare token to be your trigger word, say "an sks dog", however, those tokens usually have
other semantic meaning associated with them and can affect your results. The sks example, popular in the community, is
actually associated with a weapons brand.

To tackle that issue with pivotal tuning, instead of re-using an existing token, we insert new tokens into the text
encoders of the model - and we optimize these token embeddings to represent the new concept: that is Textual Inversion -
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

* `train_text_encoder_ti` activates the training the embeddings of new concepts
* `train_text_encoder_ti_frac` and tells us when to stop optimizing such new concepts. Pivoting halfway is the default
  value in the cog sdxl example and we have seen good results with it. We encourage experimentation here.`
* `token_abstraction` lets you choose a word to use in your instance prompt, validation prompt or custom captions to
  represent the new tokens being trained. Here we chose TOK. So, for example, "a photo of a TOK" as an instance prompt
  would then insert the newly trained tokens in place of TOK`
* `num_new_tokens_per_abstraction` sets up how many new tokens to insert and train its embeddings for each text encoder
  of the model. The default is set to 2, but you can increase that for complex concepts.`
* `Adam_weight_decay_text_encoder` This is used to set a different weight decay value for the text encoder parameters (
  different from the value used for the unet parameters).`

**Inference**

When doing pivotal tuning, besides the `*.safetensors` weights of your LoRA, you also get the `*.safetensors` embeddings
for the new tokens. In order to do inference with those we add 2 steps to how we would load normally where we:

1. Download our trained embeddings from the hub

```
import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")
# normal LoRA loading
pipe.load_lora_weights("LinoyTsaban/linoy_v9", weight_name="pytorch_lora_weights.safetensors")
```

2. Load these into the text encoders

```
# download embeddings
embedding_path = hf_hub_download(repo_id="LinoyTsaban/web_y2k_v3", filename="embeddings.safetensors", repo_type="model")
# load embeddings to the text encoders
state_dict = load_file(embedding_path)

pipe.load_textual_inversion(state_dict["text_encoders_0"][0], token=["<s0>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["text_encoders_0"][1], token=["<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

pipe.load_textual_inversion(state_dict["text_encoders_1"][0], token=["<s0>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["text_encoders_1"][1], token=["<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```

3. Prompt your LoRA!

```
prompt="a <s0><s1> webpage about an astronaut riding a horse"
images = pipe(
    prompt,
    cross_attention_kwargs={"scale": 0.8},
).images
#your output image
images[0]
```

`<astronaut image>`

**Comfy UI / AUTOMATIC1111 Inference:**
The new script fully supports textual inversion loading with Comfy UI and AUTOMATIC1111 formats!

```

```

<h2>Adaptive Optimizers</h2>

`loss animation`

When training/fine-tuning a diffusion model (or any machine learning model for that matter), we use optimizers to guide
us towards the optimal path that leads to convergence of our training objective - a minimum point of our chosen loss
function that represents a state where the model learned what we are trying to teach it. The standard (and
state-of-the-art) choices for deep learning tasks are the Adam and AdamW optimizers.

However, they require the user to meddle a lot with the hyperparameters that pave the path to convergence (such as
learning rate, weight decay, etc.). This can result in time-consuming experiments that lead to suboptimal outcomes, and
even if you land on an ideal learning rate, it may still lead to convergence issues if the learning rate is constant
during training. Some parameters may benefit from more frequent updates to expedite convergence, while others may
require smaller adjustments to avoid overshooting the optimal value. To tackle this challenge, algorithms with adaptable
learning rates such as **Adafactor** and [**Prodigy**](https://github.com/konstmish/prodigy) have been introduced. These methods optimize the algorithm's traversal of
the optimization landscape by dynamically adjusting the learning rate for each parameter based on their past gradients.

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
(like- `--prodigy_beta3`, `prodigy_decouple`, `prodigy_safeguard_warmup`), 
we will not delve into those in this post, but you can learn more about them [here](https://github.com/konstmish/prodigy).   
