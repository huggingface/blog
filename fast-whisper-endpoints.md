---
title: "Blazingly fast whisper transcriptions with Inference Endpoints" 
thumbnail: /blog/assets/fast-whisper-endpoints/thumbnail.png
authors:
- user: mfuntowicz
- user: freddyaboulton
- user: Steveeeeeeen
- user: reach-vb
- user: erikkaum
- user: michellehbn
---

Today we are happy to introduce a new blazing fast OpenAI Whisper deployment option on [Inference Endpoints](https://endpoints.huggingface.co). It provides up to 8x performance improvements compared to the previous version, and makes everyone one click away from deploying dedicated, powerful transcription models in a cost-effective way, leveraging the amazing work done by the AI community.


Through this release, we would like to make Inference Endpoints more community-centric and allow anyone to come and contribute to create incredible inference deployments on the Hugging Face Platform. Along with the community, we would like to propose optimized deployments for a wide range of tasks through the use of awesome and available open-source technologies.

The unique position of Hugging Face, at the heart of the Open-Source AI Community, working hand-in-hand with individuals, institutions and industrial partners, makes it the most heterogeneous platform when it comes to deploying AI models for inference on a wide variety of hardware and software.

## Inference Stack

The new Whisper endpoint leverages amazing open-source community projects. Inference is powered by [the vLLM project](https://github.com/vllm-project/vllm), which provides efficient ways of running AI models on various hardware families – especially, but not limited to, NVIDIA GPUs. We use the vLLM implementation of OpenAI's Whisper model, allowing us to enable further, lower-level optimizations down the software stack. 

In this initial release, we are targeting NVIDIA GPUs with compute capabilities 8.9 or better (Ada Lovelace), like L4 & L40s, which unlocks a wide range of software optimizations:
- PyTorch compilation (torch.compile)
- CUDA graphs
- float8 KV cache
Compilation with `torch.compile` generates optimized kernels in a Just-In-Time (JIT) fashion, which can modify the computational graph, reorder operations, call specialized methods, and more. 

CUDA graphs record the flow of sequential operations, or kernels, happening on the GPU, and attempts to group them as bigger chunks of work units to execute on the GPU. This grouping operation reduces data movements, synchronizations, and GPU scheduling overhead by executing a single, much larger work unit, rather than multiple smaller ones.

Last but not least, we are dynamically quantizing activations to reduce the memory requirement incurred by the KV cache(s). The computations are done in half precision, bfloat16 in this case, and the outputs are being stored in reduced precision (1 byte for float8 vs 2 bytes for bfloat16) which allows us to store more elements in the KV cache, increasing the cache hit rate.

There are many ways to continue pushing this and we are gearing up to work hand in hand with the community to improve it!

## Benchmarks

Whisper Large V3 shows a nearly 8x improvement in RTFx, enabling much faster inference with no loss in transcription quality.

We evaluated the transcription quality and runtime efficiency of several Whisper-based models—Whisper Large V3, Whisper Large V3-Turbo, and Distil-Whisper Large V3.5—and compared them against their implementations on the Transformers library to assess both accuracy and decoding speed under identical conditions.

We computed Word Error Rate (WER) across 8 standard datasets from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), including AMI, GigaSpeech, LibriSpeech (Clean and Other), SPGISpeech, Tedlium, VoxPopuli, and Earnings22. These datasets span diverse domains and recording conditions, ensuring a robust evaluation of generalization and real-world transcription quality. WER measures transcription accuracy by calculating the percentage of words that are incorrectly predicted (via insertions, deletions, or substitutions); lower WER indicates better performance. All three Whisper variants maintain WER performance comparable to their Transformer baselines.
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/endpoints/fast-whisper-endpoints-bench-1.png" alt="Word Error Rate comparison" style="display: block; margin-left: auto; margin-right: auto;">

To assess inference efficiency, we sampled from the rev16 long-form dataset, which contains audio segments over 45 minutes in length—representative of real transcription workloads such as meetings, podcasts, or interviews. We measured the Real-Time Factor (RTFx), defined as the ratio of audio duration to transcription time, and averaged it across samples. All models were evaluated in `bfloat16` on a single L4 GPU, using consistent decoding settings (language, beam size, and batch size).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/endpoints/fast-whisper-endpoints-bench-2.png" alt="Real-Time Factor comparison" style="display: block; margin-left: auto; margin-right: auto;">

## How to deploy

You can deploy your own ASR inference pipeline via [Hugging Face Endpoints](https://endpoints.huggingface.co/catalog?task=automatic-speech-recognition). Endpoints allows anyone willing to deploy AI models into production ready environments to do so by filling in a few parameters.
It also features the most complete fleet of AI hardware available on the market to suit your need for cost and performance. 
All of this directly from where the AI community is being built.
To get started, nothing easier, simply choose the model you want to deploy: 

<video alt="Fast Whisper Deployment" autoplay loop autobuffer playsinline>
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/endpoints/fast-whisper-endpoints-deploy-flow.mp4" type="video/mp4">
</video>

## Inference

Running inference on the deployed model endpoint can be done in just a few lines of code in Python, you can also use the same structure in Javascript or any other language you're comfortable with.

Here's a small snippet to test the deployed checkpoint quickly.

``` python
import requests

ENDPOINT_URL = "https://<your‑hf‑endpoint>.cloud/api/v1/audio/transcriptions"  # 🌐 replace with your URL endpoint
HF_TOKEN     = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"                              # 🔑 replace with your HF token
AUDIO_FILE   = "sample.wav"                                                    # 🔊 path to your local audio file

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

with open(AUDIO_FILE, "rb") as f:
    files = {"file": f.read()}

response = requests.post(ENDPOINT_URL, headers=headers, files=files)
response.raise_for_status()

print("Transcript:", response.json()["text"])
```
## FastRTC Demo

With this blazing fast endpoint, it’s possible to build real-time transcription apps. Try out this [example](https://huggingface.co/spaces/freddyaboulton/really-fast-whisper) built with [FastRTC](https://fastrtc.org). Simply speak into your microphone and see your speech transcribed in real time!

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.36.1/gradio.js"
></script>

<gradio-app theme_mode="light" space="freddyaboulton/really-fast-whisper"></gradio-app>


Spaces can easily be duplicated so please feel free to duplicate away. All of the above is made available for community use on the Hugging Face Hub in our dedicated HF Endpoints organization. Open issues, suggest use-cases and contribute here:  [hfendpoints-images (Inference Endpoints Images)](https://huggingface.co/spaces/hfendpoints-images/README/discussions/2) 🚀
