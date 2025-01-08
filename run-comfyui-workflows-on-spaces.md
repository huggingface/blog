---
title: "Run ComfyUI workflows for free with Gradio on Hugging Face Spaces"
thumbnail: /blog/assets/comfyui-to-gradio/cover.png
authors:
- user: multimodalart
- user: cbensimon
---

# Run ComfyUI workflows for free with Gradio on Hugging Face Spaces

Index:
- [Intro](#intro)
    - [Prerequisites](#prerequisites)
- [Exporting your ComfyUI workflow to run on pure Python](#1-exporting-your-comfyui-workflow-to-run-on-pure-python)
- [Create a Gradio app for your ComfyUI app](#2-create-a-gradio-app-for-the-exported-python)
- [Prepare it to run on Hugging Face Spaces](#3-preparing-it-to-run-hugging-face-spaces)
- [Exporting to Spaces and running on ZeroGPU](#4-exporting-to-spaces-and-running-on-zerogpu)
- [Conclusion](#5-conclusion)

## Intro

In this tutorial I will present a step-by-step guide on how to convert a complex ComfyUI workflow to a simple Gradio application, and how to deploy this application on Hugging Face Spaces ZeroGPU serverless structure, which allows for it to be deployed and run for free in a serverless manner. In this tutorial, we are going to work with [Nathan Shipley's Flux[dev] Redux + Flux[dev] Depth ComfyUI workflow](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68), but you can follow the tutorial with any workflow that you would like.

![comfy-to-gradio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/main_ui_conversion.png)

The tl;dr summary of what we will cover in this tutorial is: 

1. Export your ComfyUI workflow using [`ComfyUI-to-Python-Extension`](https://github.com/pydn/ComfyUI-to-Python-Extension);
2. Create a Gradio app for the exported Python;
3. Deploy it on Hugging Face Spaces with ZeroGPU;
4. Soon: this entire process will be automated;

### Prerequisites

- Knowing how to run ComfyUI: this tutorial requires you to be able to grab a ComfyUI workflow and run it on your machine, installing missing nodes and finding the missing models (we do plan to automate this step soon though);
- Getting the workflow you would like to export up and running (if you want to learn without a workflow in mind, feel free to get [Nathan Shipley's Flux[dev] Redux + Flux[dev] Depth ComfyUI workflow](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68) up and running);
- A little bit of coding knowledge: but I would encourage beginners to attempt to follow it, as it can be a really nice introduction to Python, Gradio and Spaces without too much prior programming knowledge needed. 

(If you are looking for an end-to-end "workflow-to-app" structure, without needing to setup and run Comfy or knowing coding, stay tuned on my profile on [Hugging Face](https://huggingface.co/multimodalart/) or [Twitter/X](https://twitter.com/multimodalart) as we plan to do this in early 2025!).

## 1. Exporting your ComfyUI workflow to run on pure Python

ComfyUI is awesome, but as the name indicates, it contains a UI. But Comfy is way more than a UI, it contains it's own backend that runs on Python. As we don't want to use Comfy's node-based UI for the purposes of this tutorial, we want to export the code to be ran on pure python.

Thankfully, [Peyton DeNiro](https://github.com/pydn) has created this incredible [ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension) for ComfyUI that will export any Comfy workflow to a python script that can run any workflow of ComfyUI with Python, not firing up the UI.

![comfy-to-gradio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/export_as_python_steps.png)

The easiest way to install the extension is to (1) search for `ComfyUI to Python Extension` in the Custom Nodes Manager Menu of the ComfyUI Manager extension and (2) install it, then, for the option to appear, you have to go on the (3) settings on the bottom right of the UI, (4) disable the new menu and hit (5) `Save as Script`. With that, you will end up with a Python script.

## 2. Create a Gradio app for the exported Python

Now that we have our Python script, it is time to create our Gradio app that will orchestrate it. Gradio is a python-native web-UI builder that allows us to create streamline applications. If you don't have it already, you can install it on your Python environment with `pip install gradio`

Now, we will have to re-arrange our python script a bit to create a UI for it.

> Tip: LLMs like ChatGPT, Claude, Qwen, Gemni, LLama 3, etc. know how to create Gradio apps. Pasting your exported Python script to it and asking it to create a Gradio app should work on a basic level, but you'd probably need to correct somethings with the knowledge you'll get in this tutorial, but here we are going to create the application ourselves.

Open the exported Python script and add an import for Gradio

```diff
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
+ import gradio as gr
```

Now, we need to think of the UI - from the complex ComfyUI workflow, which parameters we would like to expose in our UI. For the `Flux[dev] Redux + Flux[dev] Depth ComfyUI workflow`, I would like to expose: the prompt, the structure image, the style image, the depth strength (for the structure) and the style strength.

<video controls src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/inputs_list.mp4" title="Title"></video>
_Video illustrating what nodes will be exposed to the final user_

For that, a minimal Gradio app would be: 
```py
if __name__ == "__main__":
    # Comment out the main() call
    
    # Start your Gradio app
    with gr.Blocks() as app:
        # Add a title
        gr.Markdown("# FLUX Style Shaping")

        with gr.Row():
            with gr.Column():
                # Add an input
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                # Add a `Row` to include the groups side by side 
                with gr.Row():
                    # First group includes structure image and depth strength
                    with gr.Group():
                        structure_image = gr.Image(label="Structure Image", type="filepath")
                        depth_strength = gr.Slider(minimum=0, maximum=50, value=15, label="Depth Strength")
                    # Second group includes style image and style strength
                    with gr.Group():
                        style_image = gr.Image(label="Style Image", type="filepath")
                        style_strength = gr.Slider(minimum=0, maximum=1, value=0.5, label="Style Strength")
                
                # The generate button
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # The output image
                output_image = gr.Image(label="Generated Image")

            # When clicking the button, it will trigger the `generate_image` function, with the respective inputs
            # and the output an image
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt_input, structure_image, style_image, depth_strength, style_strength],
                outputs=[output_image]
            )
        app.launch(share=True)
```

But if you try to run it, it won't work yet, as now we need to set up this `generate_image` function by altering the `def main()` function of our exported python 

script:
```diff
- def main():
+ def generate_image(prompt, structure_image, style_image, depth_strength, style_strength)
```

And inside the function, we need to find the hard coded values of the nodes we want, and replace it with the variables we would like to control, such as:
```diff
loadimage_429 = loadimage.load_image(
-    image="7038548d-d204-4810-bb74-d1dea277200a.png"
+    image=structure_image
)
# ...
loadimage_440 = loadimage.load_image(
-    image="2013_CKS_01180_0005_000(the_court_of_pir_budaq_shiraz_iran_circa_1455-60074106).jpg"
+    image=style_image
)
# ...
fluxguidance_430 = fluxguidance.append(
-   guidance=15,
+   guidance=depth_strength,
    conditioning=get_value_at_index(cliptextencode_174, 0)
)
# ...
stylemodelapplyadvanced_442 = stylemodelapplyadvanced.apply_stylemodel(
-   strength=0.5,
+   strength=style_strenght,
    conditioning=get_value_at_index(instructpixtopixconditioning_431, 0),
    style_model=get_value_at_index(stylemodelloader_441, 0),
    clip_vision_output=get_value_at_index(clipvisionencode_439, 0),
)
# ...
cliptextencode_174 = cliptextencode.encode(
-   text="a girl looking at a house on fire",
+   text=prompt,   
    clip=get_value_at_index(cr_clip_input_switch_319, 0),
)
```

and for our output, we need to find the save image output node, and export its path, such as:
```diff
saveimage_327 = saveimage.save_images(
    filename_prefix=get_value_at_index(cr_text_456, 0),
    images=get_value_at_index(vaedecode_321, 0),
)
+ saved_path = f"output/{saveimage_327['ui']['images'][0]['filename']}"
+ return saved_path
```

Check out a video rundown of this modifications: 
<video controls src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/video_code_change.mp4" title="Title"></video>

Now, we should be ready to run the code! Save your python file as `app.py`, add it to the root of your ComfyUI folder and run it as

```shell
python app.py
```

And just like that, you should be able to run your Gradio app on http://0.0.0.0:7860
```shell
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://366fdd17b8a9072899.gradio.live
```

<video controls src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/comfy_local_running.mp4" title="Title"></video>

To debug this process, check [here](https://gist.github.com/apolinario/47a8503c007c5ae8494324bed9e158ce/revisions?diff=unified&w=47a8503c007c5ae8494324bed9e158ce#diff-faf377dc15b3371a15d2c4a03b4d012825533bd2fb2297852cb2244d07fe36eeL1) the diff between the original python file exported by `ComfyUI-to-Python-Extension` and the Gradio app. You can download both at that URL as well to check and compare with your own workflow.

That's it, congratulations! You managed to convert your ComfyUI workflow to a Gradio app. You can run it locally or even send the URL to a customer or friend, however, if you turn off your computer or if 72h pass, the temporary Gradio link will die. For a persistent structure for hosting the app - including allowing people to run it for free in a serverless manner, you can use Hugging Face Spaces. 

## 3. Preparing it to run Hugging Face Spaces

Now with our Gradio demo working, we may feel tempted to just hit an export button and get it working on Hugging Face Spaces, however, as we have all models loaded locally, if we just exported all our folder to Spaces, we would upload dozens of GB of models on Hugging Face, which is not supported, specially as all this models should have a mirror on Hugging Face. 

So, we need to first install `pip install huggingface_hub` if we don't have it already, and then we need to do the following on the top of our `app.py` file:

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="black-forest-labs/FLUX.1-Redux-dev", filename="flux1-redux-dev.safetensors", local_dir="models/style_models")
hf_hub_download(repo_id="black-forest-labs/FLUX.1-Depth-dev", filename="flux1-depth-dev.safetensors", local_dir="models/diffusion_models")
hf_hub_download(repo_id="Comfy-Org/sigclip_vision_384", filename="sigclip_vision_patch14_384.safetensors", local_dir="models/clip_vision")
hf_hub_download(repo_id="Kijai/DepthAnythingV2-safetensors", filename="depth_anything_v2_vitl_fp32.safetensors", local_dir="models/depthanything")
hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors", local_dir="models/vae/FLUX1")
hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", local_dir="models/text_encoders")
hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", local_dir="models/text_encoders/t5")
```

This will map all local models on ComfyUI to a Hugging Face version of them. Unfortunately, currently there is no way to automate this process, you gotta find the models of your workflow on Hugging Face and map it to the same ComfyUI folders that.

If you are running models that are not on Hugging Face, you need find a way to programmatically download them to the correct folder via Python code. This will run only once when the Hugging Face Space starts.

Now, we will do one last modification to the `app.py` file, which is to include the function decoration for ZeroGPU, which will let us do inference for free!
```diff
import gradio as gr
from huggingface_hub import hf_hub_download
+ import spaces
# ...
+ @spaces.GPU(duration=60) #modify the duration for the average it takes for your worflow to run, in seconds
def generate_image(prompt, structure_image, style_image, depth_strength, style_strength):
```

Check here the [diff](https://gist.github.com/apolinario/47a8503c007c5ae8494324bed9e158ce/revisions?diff=unified&w=47a8503c007c5ae8494324bed9e158ce#diff-faf377dc15b3371a15d2c4a03b4d012825533bd2fb2297852cb2244d07fe36eeL4) from the previous Gradio demo with the Spaces prepared changes.

## 4. Exporting to Spaces and running on ZeroGPU

Now that you have your code ready for Hugging Face Spaces, it's time to export your demo to run there.

### Fix requirements

Firstly, you need to modify your `requirements.txt` to include the requirements in the `custom_nodes` folder, to add append the requirements of the nodes you want to work for this workflow to the `requirements.txt` on the root folder, as Hugging Face Spaces can only deal with a single `requirements.txt` file.

You can see the illustration below. You need to do the same process for all `custom_nodes`: 
<video controls src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/illustrative_video.mp4" title="Title"></video>

Now we are ready! 

![create-space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/create_space.png)

1. Get to [https://huggingface.co](https://huggingface.co) and create a new Space.
2. Set its hardware to ZeroGPU (if you are a Hugging Face PRO subscriber) or set it to CPU basic if you are not a PRO user (you'll need an extra step at the end if you are not PRO).
    2.1 (If you prefer a dedicated GPU that you pay for, pick L4, L40S, A100 instead of ZeroGPU, that's a paid option)
3. Click the Files tab, Add `File > Upload Files`. Drag all your ComfyUI folder files **except** the `models` folder (if you attempt to upload the `models` folder, your upload will fail), that's why we need part 3.
4. Click the `Commit changes to main` button on the bottom of the page and wait for everything to upload
5. If you are using gated models, like FLUX, you need to include a Hugging Face token to the settings. First, create a token with `read` access to all the gated models you need [here](https://huggingface.co/settings/tokens), then go to the `Settings` page of your Space and create a new secret named `HF_TOKEN` with the value of the token you have just created.

![variables-and-secrets](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/comfyu-to-gradio/variables_and_secrets.png)

### Move models outside the decorated function (ZeroGPU only)

Your demo should already be working, however, in the current setup, the models will be fully loaded from disk to GPU every time you run it. To make use of the serverless ZeroGPU efficiency, we will need to move all model declarations outside the decorated function to the global context of Python. Let's edit the `app.py` function to do that.

```diff
@@ -4,6 +4,7 @@
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
+from comfy import model_management
import spaces

hf_hub_download(repo_id="black-forest-labs/FLUX.1-Redux-dev", filename="flux1-redux-dev.safetensors", local_dir="models/style_models")
@@ -109,6 +110,62 @@

from nodes import NODE_CLASS_MAPPINGS

+intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
+dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
+dualcliploader_357 = dualcliploader.load_clip(
+    clip_name1="t5/t5xxl_fp16.safetensors",
+    clip_name2="clip_l.safetensors",
+    type="flux",
+)
+cr_clip_input_switch = NODE_CLASS_MAPPINGS["CR Clip Input Switch"]()
+cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
+loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
+imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
+getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
+vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
+vaeloader_359 = vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")
+vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
+unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
+unetloader_358 = unetloader.load_unet(
+    unet_name="flux1-depth-dev.safetensors", weight_dtype="default"
+)
+ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
+randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
+fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
+depthanything_v2 = NODE_CLASS_MAPPINGS["DepthAnything_V2"]()
+downloadandloaddepthanythingv2model = NODE_CLASS_MAPPINGS[
+    "DownloadAndLoadDepthAnythingV2Model"
+]()
+downloadandloaddepthanythingv2model_437 = (
+    downloadandloaddepthanythingv2model.loadmodel(
+        model="depth_anything_v2_vitl_fp32.safetensors"
+    )
+)
+instructpixtopixconditioning = NODE_CLASS_MAPPINGS[
+    "InstructPixToPixConditioning"
+]()
+text_multiline_454 = text_multiline.text_multiline(text="FLUX_Redux")
+clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
+clipvisionloader_438 = clipvisionloader.load_clip(
+    clip_name="sigclip_vision_patch14_384.safetensors"
+)
+clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
+stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
+stylemodelloader_441 = stylemodelloader.load_style_model(
+    style_model_name="flux1-redux-dev.safetensors"
+)
+text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
+emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
+cr_conditioning_input_switch = NODE_CLASS_MAPPINGS[
+    "CR Conditioning Input Switch"
+]()
+cr_model_input_switch = NODE_CLASS_MAPPINGS["CR Model Input Switch"]()
+stylemodelapplyadvanced = NODE_CLASS_MAPPINGS["StyleModelApplyAdvanced"]()
+basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
+basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
+samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
+vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
+saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
+imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()

@@ -117,75 +174,6 @@
def generate_image(prompt, structure_image, style_image, depth_strength, style_strength):
    import_custom_nodes()
    with torch.inference_mode():
-        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
         intconstant_83 = intconstant.get_value(value=1024)

         intconstant_84 = intconstant.get_value(value=1024)

-        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
-        dualcliploader_357 = dualcliploader.load_clip(
-            clip_name1="t5/t5xxl_fp16.safetensors",
-            clip_name2="clip_l.safetensors",
-            type="flux",
-        )
-
-        cr_clip_input_switch = NODE_CLASS_MAPPINGS["CR Clip Input Switch"]()
         cr_clip_input_switch_319 = cr_clip_input_switch.switch(
             Input=1,
             clip1=get_value_at_index(dualcliploader_357, 0),
             clip2=get_value_at_index(dualcliploader_357, 0),
         )

-        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
         cliptextencode_174 = cliptextencode.encode(
             text=prompt,
             clip=get_value_at_index(cr_clip_input_switch_319, 0),
         )

         cliptextencode_175 = cliptextencode.encode(
             text="purple", clip=get_value_at_index(cr_clip_input_switch_319, 0)
         )

-        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
         loadimage_429 = loadimage.load_image(image=structure_image)

-        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
         imageresize_72 = imageresize.execute(
             width=get_value_at_index(intconstant_83, 0),
             height=get_value_at_index(intconstant_84, 0),
             interpolation="bicubic",
             method="keep proportion",
             condition="always",
             multiple_of=16,
             image=get_value_at_index(loadimage_429, 0),
         )

-        getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
         getimagesizeandcount_360 = getimagesizeandcount.getsize(
             image=get_value_at_index(imageresize_72, 0)
         )

-        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
-        vaeloader_359 = vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")

-        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
         vaeencode_197 = vaeencode.encode(
             pixels=get_value_at_index(getimagesizeandcount_360, 0),
             vae=get_value_at_index(vaeloader_359, 0),
         )

-        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
-        unetloader_358 = unetloader.load_unet(
-            unet_name="flux1-depth-dev.safetensors", weight_dtype="default"
-        )

-        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
         ksamplerselect_363 = ksamplerselect.get_sampler(sampler_name="euler")

-        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
         randomnoise_365 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

-        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
         fluxguidance_430 = fluxguidance.append(
             guidance=15, conditioning=get_value_at_index(cliptextencode_174, 0)
         )

-        downloadandloaddepthanythingv2model = NODE_CLASS_MAPPINGS[
-            "DownloadAndLoadDepthAnythingV2Model"
-        ]()
-        downloadandloaddepthanythingv2model_437 = (
-            downloadandloaddepthanythingv2model.loadmodel(
-                model="depth_anything_v2_vitl_fp32.safetensors"
-            )
-        )

-        depthanything_v2 = NODE_CLASS_MAPPINGS["DepthAnything_V2"]()
         depthanything_v2_436 = depthanything_v2.process(
             da_model=get_value_at_index(downloadandloaddepthanythingv2model_437, 0),
             images=get_value_at_index(getimagesizeandcount_360, 0),
         )

-        instructpixtopixconditioning = NODE_CLASS_MAPPINGS[
-            "InstructPixToPixConditioning"
-        ]()
         instructpixtopixconditioning_431 = instructpixtopixconditioning.encode(
             positive=get_value_at_index(fluxguidance_430, 0),
             negative=get_value_at_index(cliptextencode_175, 0),
             vae=get_value_at_index(vaeloader_359, 0),
             pixels=get_value_at_index(depthanything_v2_436, 0),
         )

-        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
-        clipvisionloader_438 = clipvisionloader.load_clip(
-            clip_name="sigclip_vision_patch14_384.safetensors"
-        )

         loadimage_440 = loadimage.load_image(image=style_image)

-        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
         clipvisionencode_439 = clipvisionencode.encode(
             crop="center",
             clip_vision=get_value_at_index(clipvisionloader_438, 0),
             image=get_value_at_index(loadimage_440, 0),
         )

-        stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
-        stylemodelloader_441 = stylemodelloader.load_style_model(
-            style_model_name="flux1-redux-dev.safetensors"
-        )
-
-        text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
         text_multiline_454 = text_multiline.text_multiline(text="FLUX_Redux")

-        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
-        cr_conditioning_input_switch = NODE_CLASS_MAPPINGS[
-            "CR Conditioning Input Switch"
-        ]()
-        cr_model_input_switch = NODE_CLASS_MAPPINGS["CR Model Input Switch"]()
-        stylemodelapplyadvanced = NODE_CLASS_MAPPINGS["StyleModelApplyAdvanced"]()
-        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
-        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
-        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
-        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
-        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
-        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()

         emptylatentimage_10 = emptylatentimage.generate(
             width=get_value_at_index(imageresize_72, 1),
             height=get_value_at_index(imageresize_72, 2),
             batch_size=1,
         )
```

Additionally, in order to pre-load the models we need to use the ComfyUI `load_models_gpu` function, which will include, from the above pre-loaded model, all the models that were loaded (a good rule of thumb, is checking which from the above load a `*.safetensors` file)
```py
from comfy import model_management

#Add all the models that load a safetensors file
model_loaders = [dualcliploader_357, vaeloader_359, unetloader_358, clipvisionloader_438, stylemodelloader_441, downloadandloaddepthanythingv2model_437]

# Check which models are valid and how to best load them
valid_models = [
    getattr(loader[0], 'patcher', loader[0]) 
    for loader in model_loaders
    if not isinstance(loader[0], dict) and not isinstance(getattr(loader[0], 'patcher', None), dict)
]

#Finally loads the models
model_management.load_models_gpu(valid_models)
```

[Check the diff](https://gist.github.com/apolinario/47a8503c007c5ae8494324bed9e158ce/revisions#diff-faf377dc15b3371a15d2c4a03b4d012825533bd2fb2297852cb2244d07fe36eeL6) to understand precisely what changes

### If you are not a PRO subscriber (skip this step if you are)

If are not a Hugging Face PRO subscriber, you need to apply for a ZeroGPU grant, visit the Settings page of your Space and apply for a grant. Request ZeroGPU. I will grant everybody that requests a ZeroGPU grant for ComfyUI backends. 

### The demo is running

The demo we have built with this tutorial is live on Hugging Face Spaces. Come play with it here: [https://huggingface.co/spaces/multimodalart/flux-style-shaping](https://huggingface.co/spaces/multimodalart/flux-style-shaping)

## 5. Conclusion

😮‍💨, that's all! I know it is a bit of work, but the reward is an easy way to share your workflow with a simple UI and free inference to everyone! As mentioned before, the goal is to automate and streamline this process as much as possible in early 2025! Happy holidays 🎅✨