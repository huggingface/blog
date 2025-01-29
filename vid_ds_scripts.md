# Build awesome datasets for video generation

Tooling for image generation datasets is well established, with [`img2dataset`](https://github.com/rom1504/img2dataset) covering large scale and various community guides, scripts and UIs covering the small scale.

Our goal is to make tooling for video generation datasets as established by creating open video dataset scripts suited for small scale, with [`video2dataset`](https://github.com/iejMac/video2dataset) covering large scale.

*“If I have seen further it is by standing on the shoulders of giants”*

In this post, we will provide a overview of the tooling we are developing to make it easy for the community to build their own datasets for fine-tuning video generation models. If you cannot wait to get started already, we welcome you to check out the codebase [here](https://github.com/huggingface/video-dataset-scripts/tree/main/video_processing). 

**Table of contents**

TODO

## Tooling

Typically, video generation is conditioned on natural language text prompts such as: "A cat walks on the grass, realistic style". Then in a video, there are a number of qualitative aspects for controllability, like so:

* Motion
* Aesthetics
* Presence of watermarks

Video generation models are only as good as the data they are trained on. Therefore, these aspects become crucial when curating the datasets for training/fine-tuning.

Our 3 stage pipeline draws inspiration from works like [Stable Video Diffusion](https://arxiv.org/abs/2311.15127), [LTX-Video](https://arxiv.org/abs/2501.00103), and their data pipelines.

### Stage 1 (Acquisition)

Like [`video2dataset`](https://github.com/iejMac/video2dataset) we opt to use [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for downloading videos.

We create a script `Video to Scenes` to split long videos into short clips.

### Stage 2 (Pre-processing/filtering)

#### Extracted frames

- detect watermarks with [LAION-5B-WatermarkDetection](https://github.com/LAION-AI/LAION-5B-WatermarkDetection)
- predict an aesthetic score with [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

#### Entire video

- predict a motion score with [OpenCV](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

### Stage 3 (Processing)

Florence-2 [`microsoft/Florence-2-large`](http://hf.co/microsoft/Florence-2-large) to run <CAPTION>, <DETAILED_CAPTION>, <DENSE_REGION_CAPTION> and <OCR_WITH_REGION>

We can bring in any other captioner in this regard. We can also caption the entire video (e.g., with a model like [Qwen2.5](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl)) as opposed to captioning individual frames.

## Filtering examples

In the [dataset](https://huggingface.co/datasets/finetrainers/crush-smol) for the model [`finetrainers/crush-smol-v0`](https://hf.co/finetrainers/crush-smol-v0) we filtered on `pwatermark < 0.1` and `aesthetic > 5.5`. This highly restrictive filtering resulted in 47 videos out of 1493 total. 

Let's review the example frames from `pwatermark` - two with text have scores of 0.69 and 0.61, the "toy car with a bunch of mice in it" scores 0.60 then 0.17 as the toy car is crushed. All example frames were filtered by `pwatermark < 0.1`. `pwatermark` is effective at detecting text/watermarks however the score gives no indication whether it is a text overlay or a toy car's license plate. Our filtering required all scores to be below the threshold, an average across frames would be a more effective strategy for `pwatermark` with a threshold of around 0.2 - 0.3.

Let's review the example frames from aesthetic scores - the pink castle initially scores 5.5 then 4.44 as it is crushed, the action figure scores lower at 4.99 dropping to 4.84 as it is crushed and the shard of glass scores low at 4.04. Again fitlering required all scores to be below the threshold, in this case using the aesthetic score from the first frame only would be a more effective strategy. 

If we review [`finetrainers/crush-smol`](https://huggingface.co/datasets/finetrainers/crush-smol) we can notice that many of the objects being crushed are round or rectangular and colorful which is similar to our findings in the example frames. Aesthetic scores can be useful yet have a bias that will potentially filter out good data when used with extreme thresholds like > 5.5. It may be more effective as a filter for bad content than good with a minimum threshold of around 4.25 - 4.5.

We provide some visual examples for each filters as well as the captions from Florence-2.

### Watermarks

| **`pwatermark`** | **Image** |
|:----------:|:-----:|
| 0.69       | ![19s8CRUVf3E-Scene-022_0.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/19s8CRUVf3E-Scene-022_0.jpg) |
| 0.61       | ![19s8CRUVf3E-Scene-010_0.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/19s8CRUVf3E-Scene-010_0.jpg) |
| 0.60       | ![-IvRtqwaetM-Scene-003_0.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-003_0.jpg) |
| 0.17       | ![-IvRtqwaetM-Scene-003_1.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-003_1.jpg) |


### Aesthetics

| **Aesthetic** | **Image** |
|:---------:|:-----:|
| 5.50      | ![-IvRtqwaetM-Scene-036_0.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-036_0.jpg) |
| 4.44      | ![-IvRtqwaetM-Scene-036_1.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-036_1.jpg) |
| 4.99      | ![-IvRtqwaetM-Scene-046_0.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-046_0.jpg) |
| 4.87      | ![-IvRtqwaetM-Scene-046_1.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-046_1.jpg) |
| 4.84      | ![-IvRtqwaetM-Scene-046_2.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-046_2.jpg) |
| 4.04      | ![19s8CRUVf3E-Scene-015_1.jpg](Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/19s8CRUVf3E-Scene-015_1.jpg) |

### OCR/Caption

<table>
    <tr>
        <th>Image</th>
        <th>Caption</th>
        <th>Detailed Caption</th>
    </tr>
    <tr>
        <td>
            <img src="Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/-IvRtqwaetM-Scene-003_0%201.jpg" 
                 alt="Toy Car with Mice">
        </td>
        <td>A toy car with a bunch of mice in it.</td>
        <td>The image shows a blue toy car with three white mice sitting in the back of it, driving down a road with a green wall in the background.</td>
    </tr>
</table>

<table>
    <tr>
        <th>With OCR labels</th>
        <th>With OCR and region labels</th>
    </tr>
    <tr>
        <td>
            <img src="Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/image_with_ocr_labels.jpg" 
                 alt="OCR labels">
        </td>
        <td>
            <img src="Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/image_with_ocr_and_region_labels.jpg" 
                 alt="OCR and region labels">
        </td>
    </tr>
</table>


## Putting this tooling to use 👨‍🍳

We have created various datasets with the tooling in an attempt to generate cool video effects, similar to the [Pika Effects](https://pikartai.com/effects/):

- [Cakeify](https://huggingface.co/datasets/finetrainers/cakeify-smol)
- [Crush](https://huggingface.co/datasets/finetrainers/crush-smol)

We then used these datasets to fine-tune the [CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5b) model using [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers). Below is an example output from [`finetrainers/crush-smol-v0`](https://huggingface.co/finetrainers/crush-smol-v0): 


<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="demo4.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="Video%20dataset%20scripts%201881384ebcac804cb2f2cba5736f085a/assets_output_0.mp4" type="video/mp4">
  </video>
  <figcaption>Prompt: <i>DIFF_crush A red candle is placed on a metal platform, and a large metal cylinder descends from above, flattening the candle as if it were under a hydraulic press. The candle is crushed into a flat, round shape, leaving a pile of debris around it.</i></figcaption>
</figure>

## Call to action

We hope this tooling gives you a headstart to create small and high-quality video datasets for your own custom applications. We will continue to add more useful filters to the repository, so, please keep an eye out. Your contributions are also more than welcome 🤗