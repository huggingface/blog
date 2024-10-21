---
title: "Transformers.js v3: WebGPU Support, New Models & Tasks, and More…"
thumbnail: /blog/assets/transformersjs-v3/thumbnail.png
authors:
  - user: xenova
---

# Transformers.js v3: WebGPU Support, New Models & Tasks, and More…

After more than a year of development, we're excited to announce the release of 🤗 Transformers.js v3!

Highlights include:

- [WebGPU support (up to 100x faster than WASM!)](#webgpu-support)
- [New quantization formats (dtypes)](#new-quantization-formats-dtypes)
- [A total of 120 supported architectures](#120-supported-architectures)
- [25 new example projects and templates](#example-projects-and-templates)
- [Over 1200 pre-converted models on the Hugging Face Hub](#over-1200-pre-converted-models)
- [Node.js (ESM + CJS), Deno, and Bun compatibility](#nodejs-esm--cjs-deno-and-bun-compatibility)
- [A new home on GitHub and NPM](#a-new-home-on-npm-and-github)

## Installation

You can get started by installing [Transformers.js v3](https://huggingface.co/docs/transformers.js) from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:

```bash
npm i @huggingface/transformers
```

Then, importing the library with

```js
import { pipeline } from "@huggingface/transformers";
```

or via a CDN,

```js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";
```

For more information, check out the [documentation](http://hf.co/docs/transformers.js).

## WebGPU support

### What is WebGPU?

WebGPU is a new web standard for accelerated graphics and compute. The [API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) enables web developers to use the underlying system's GPU to carry out high-performance computations directly in the browser. WebGPU is the successor to [WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API) and provides significantly better performance, because it allows for more direct interaction with modern GPUs. Lastly, it supports general-purpose GPU computations, which makes it just perfect for machine learning!

> [!WARNING]  
> As of October 2024, global WebGPU support is around 70% (according to [caniuse.com](https://caniuse.com/webgpu)), meaning some users may not be able to use the API.
>
> If the following demos do not work in your browser, you may need to enable it using a feature flag:
>
> - Firefox: with the `dom.webgpu.enabled` flag (see [here](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#:~:text=tested%20by%20Firefox.-,WebGPU%20API,-The%20WebGPU%20API)).
> - Safari: with the `WebGPU` feature flag (see [here](https://webkit.org/blog/14879/webgpu-now-available-for-testing-in-safari-technology-preview/)).
> - Older Chromium browsers (on Windows, macOS, Linux): with the `enable-unsafe-webgpu` flag (see [here](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips)).

### Usage in Transformers.js v3

Thanks to our collaboration with [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), enabling WebGPU acceleration is as simple as setting `device: 'webgpu'` when loading a model. Let's see some examples!

**Example:** Compute text embeddings on WebGPU ([demo](https://v2.scrimba.com/s06a2smeej))

```js
import { pipeline } from '@huggingface/transformers';

// Create a feature-extraction pipeline
const extractor = await pipeline(
  'feature-extraction',
  'mixedbread-ai/mxbai-embed-xsmall-v1',
  { device: 'webgpu' },
});

// Compute embeddings
const texts = ['Hello world!', 'This is an example sentence.'];
const embeddings = await extractor(texts, { pooling: 'mean', normalize: true });
console.log(embeddings.tolist());
// [
//   [-0.016986183822155, 0.03228696808218956, -0.0013630966423079371, ... ],
//   [0.09050482511520386, 0.07207386940717697, 0.05762749910354614, ... ],
// ]
```

**Example:** Perform automatic speech recognition with OpenAI whisper on WebGPU ([demo](https://v2.scrimba.com/s0oi76h82g))

```js
import { pipeline } from "@huggingface/transformers";

// Create automatic speech recognition pipeline
const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-tiny.en",
  { device: "webgpu" },
);

// Transcribe audio from a URL
const url =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
const output = await transcriber(url);
console.log(output);
// { text: ' And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.' }
```

**Example:** Perform image classification with MobileNetV4 on WebGPU ([demo](https://v2.scrimba.com/s0fv2uab1t))

```js
import { pipeline } from "@huggingface/transformers";

// Create image classification pipeline
const classifier = await pipeline(
  "image-classification",
  "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
  { device: "webgpu" },
);

// Classify an image from a URL
const url =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg";
const output = await classifier(url);
console.log(output);
// [
//   { label: 'tiger, Panthera tigris', score: 0.6149784922599792 },
//   { label: 'tiger cat', score: 0.30281734466552734 },
//   { label: 'tabby, tabby cat', score: 0.0019135422771796584 },
//   { label: 'lynx, catamount', score: 0.0012161266058683395 },
//   { label: 'Egyptian cat', score: 0.0011465961579233408 }
// ]
```

## New quantization formats (dtypes)

Before Transformers.js v3, we used the `quantized` option to specify whether to use a quantized (q8) or full-precision (fp32) variant of the model by setting `quantized` to `true` or `false`, respectively. Now, we've added the ability to select from a much larger list with the `dtype` parameter.

The list of available quantizations depends on the model, but some common ones are: full-precision (`"fp32"`), half-precision (`"fp16"`), 8-bit (`"q8"`, `"int8"`, `"uint8"`), and 4-bit (`"q4"`, `"bnb4"`, `"q4f16"`).

<p align="center">
    <picture> 
        <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/dtypes-dark.jpg" style="max-width: 100%;">
        <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/dtypes-light.jpg" style="max-width: 100%;">
        <img alt="Available dtypes for mixedbread-ai/mxbai-embed-xsmall-v1" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/dtypes-dark.jpg" style="max-width: 100%;">
    </picture>
  <a href="https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/tree/main/onnx">(e.g., mixedbread-ai/mxbai-embed-xsmall-v1)</a>
</p>

### Basic usage

**Example:** Run Qwen2.5-0.5B-Instruct in 4-bit quantization ([demo](https://v2.scrimba.com/s0dlcpv0ci))

```js
import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen2.5-0.5B-Instruct",
  { dtype: "q4", device: "webgpu" },
);

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Tell me a funny joke." },
];

// Generate a response
const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
```

### Per-module dtypes

Some encoder-decoder models, like Whisper or Florence-2, are extremely sensitive to quantization settings: especially of the encoder. For this reason, we added the ability to select per-module dtypes, which can be done by providing a mapping from module name to dtype.

**Example:** Run Florence-2 on WebGPU ([demo](https://v2.scrimba.com/s0pdm485fo))

```js
import { Florence2ForConditionalGeneration } from "@huggingface/transformers";

const model = await Florence2ForConditionalGeneration.from_pretrained(
  "onnx-community/Florence-2-base-ft",
  {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      encoder_model: "q4",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
  },
);
```

<p align="middle">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/florence-2-webgpu.gif" alt="Florence-2 running on WebGPU" />
</p>

<details>
<summary>
See full code example
</summary>

```js
import {
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
} from "@huggingface/transformers";

// Load model, processor, and tokenizer
const model_id = "onnx-community/Florence-2-base-ft";
const model = await Florence2ForConditionalGeneration.from_pretrained(
  model_id,
  {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      encoder_model: "q4",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
  },
);
const processor = await AutoProcessor.from_pretrained(model_id);
const tokenizer = await AutoTokenizer.from_pretrained(model_id);

// Load image and prepare vision inputs
const url =
  "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg";
const image = await RawImage.fromURL(url);
const vision_inputs = await processor(image);

// Specify task and prepare text inputs
const task = "<MORE_DETAILED_CAPTION>";
const prompts = processor.construct_prompts(task);
const text_inputs = tokenizer(prompts);

// Generate text
const generated_ids = await model.generate({
  ...text_inputs,
  ...vision_inputs,
  max_new_tokens: 100,
});

// Decode generated text
const generated_text = tokenizer.batch_decode(generated_ids, {
  skip_special_tokens: false,
})[0];

// Post-process the generated text
const result = processor.post_process_generation(
  generated_text,
  task,
  image.size,
);
console.log(result);
// { '<MORE_DETAILED_CAPTION>': 'A green car is parked in front of a tan building. The building has a brown door and two brown windows. The car is a two door and the door is closed. The green car has black tires.' }
```

</details>

## 120 supported architectures

This release increases the total number of supported architectures to 120 (see [full list](https://huggingface.co/docs/transformers.js/index#models)), spanning a wide range of input modalities and tasks. Notable new names include: Phi-3, Gemma & Gemma 2, LLaVa, Moondream, Florence-2, MusicGen, Sapiens, Depth Pro, PyAnnote, and RT-DETR.

<p align="middle">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/architectures.png" alt="Bubble diagram of new architectures in Transformers.js v3" />
</p>

<details>

<summary>List of new models</summary>

1. **[Cohere](https://huggingface.co/docs/transformers/main/model_doc/cohere)** (from Cohere) released with the paper [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/) by Cohere.
1. **[Decision Transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer)** (from Berkeley/Facebook/Google) released with the paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
1. **Depth Pro** (from Apple) released with the paper [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073) by Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun.
1. **Florence2** (from Microsoft) released with the paper [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) by Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan.
1. **[Gemma](https://huggingface.co/docs/transformers/main/model_doc/gemma)** (from Google) released with the paper [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/) by the Gemma Google team.
1. **[Gemma2](https://huggingface.co/docs/transformers/main/model_doc/gemma2)** (from Google) released with the paper [Gemma2: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/google-gemma-2/) by the Gemma Google team.
1. **[Granite](https://huggingface.co/docs/transformers/main/model_doc/granite)** (from IBM) released with the paper [Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler](https://arxiv.org/abs/2408.13359) by Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D. Cox, Rameswar Panda.
1. **[GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit)** (from UCSD, NVIDIA) released with the paper [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
1. **[Hiera](https://huggingface.co/docs/transformers/model_doc/hiera)** (from Meta) released with the paper [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/pdf/2306.00989) by Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
1. **JAIS** (from Core42) released with the paper [Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models](https://arxiv.org/pdf/2308.16149) by Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, Lalit Pradhan, Zain Muhammad Mujahid, Massa Baali, Xudong Han, Sondos Mahmoud Bsharat, Alham Fikri Aji, Zhiqiang Shen, Zhengzhong Liu, Natalia Vassilieva, Joel Hestness, Andy Hock, Andrew Feldman, Jonathan Lee, Andrew Jackson, Hector Xuguang Ren, Preslav Nakov, Timothy Baldwin, Eric Xing.
1. **[LLaVa](https://huggingface.co/docs/transformers/model_doc/llava)** (from Microsoft Research & University of Wisconsin-Madison) released with the paper [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) by Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee.
1. **[MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer)** (from Meta and UIUC) released with the paper [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) by Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.
1. **[MusicGen](https://huggingface.co/docs/transformers/model_doc/musicgen)** (from Meta) released with the paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.
1. **MobileCLIP** (from Apple) released with the paper [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049) by Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel.
1. **[MobileNetV1](https://huggingface.co/docs/transformers/model_doc/mobilenet_v1)** (from Google Inc.) released with the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.
1. **[MobileNetV2](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2)** (from Google Inc.) released with the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.
1. **MobileNetV3** (from Google Inc.) released with the paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) by Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.
1. **MobileNetV4** (from Google Inc.) released with the paper [MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518) by Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard.
1. **Moondream1** released in the repository [moondream](https://github.com/vikhyat/moondream) by vikhyat.
1. **OpenELM** (from Apple) released with the paper [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](https://arxiv.org/abs/2404.14619) by Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, Mohammad Rastegari.
1. **[Phi3](https://huggingface.co/docs/transformers/main/model_doc/phi3)** (from Microsoft) released with the paper [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219) by Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, Xiren Zhou.
1. **[PVT](https://huggingface.co/docs/transformers/main/model_doc/pvt)** (from Nanjing University, The University of Hong Kong etc.) released with the paper [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf) by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.
1. **PyAnnote** released in the repository [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) by Hervé Bredin.
1. **[RT-DETR](https://huggingface.co/docs/transformers/model_doc/rt_detr)** (from Baidu), released together with the paper [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) by Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen.
1. **Sapiens** (from Meta AI) released with the paper [Sapiens: Foundation for Human Vision Models](https://arxiv.org/pdf/2408.12569) by Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, Shunsuke Saito.
1. **[ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)** (from Meta AI) released with the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
1. **[ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn)** (from Meta AI) released with the paper [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas.

</details>

## Example projects and templates

As part of the release, we've published 25 new example projects and templates, primarily focused on showcasing WebGPU support! This includes demos like [Phi-3.5 WebGPU](https://github.com/huggingface/transformers.js-examples/tree/main/phi-3.5-webgpu) and [Whisper WebGPU](https://github.com/xenova/whisper-web/tree/experimental-webgpu), as shown below.

> [!NOTE]  
> We're in the process of moving all our example projects and demos to https://github.com/huggingface/transformers.js-examples, so stay tuned for updates on this!

| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/phi-3.5-webgpu.gif" style="max-height: 500px;" alt="Phi-3.5 running on WebGPU" /> | <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/whisper-turbo-webgpu.gif" style="max-height: 500px;" alt="Whisper Turbo running on WebGPU" /> |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

## Over 1200 pre-converted models

As of today's release, the community has converted over 1200 models to be compatible with Transformers.js! You can find the full list of available models [here](https://hf.co/models?library=transformers.js).

If you'd like to convert your own models or fine-tunes, you can use our [conversion script](https://github.com/huggingface/transformers.js/blob/main/scripts/convert.py) as follows:

```sh
python -m scripts.convert --quantize --model_id <model_name_or_path>
```

After uploading the generated files to the Hugging Face Hub, remember to add the `transformers.js` tag so others can easily find and use your model!

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/models-dark.jpg" style="max-width: 100%;">
        <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/models-light.jpg" style="max-width: 100%;">
        <img alt="Available Transformers.js models" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformersjs-v3/models-dark.jpg" style="max-width: 100%;">
    </picture>
</p>

## Node.js (ESM + CJS), Deno, and Bun compatibility

Transformers.js v3 is now compatible with the three most popular server-side JavaScript runtimes:

| Runtime                        | Description                                                                                                                                 | Examples                                                                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Node.js](https://nodejs.org/) | A widely-used JavaScript runtime built on Chrome's V8. It has a large ecosystem and supports a wide range of libraries and frameworks.      | [ESM Example](https://github.com/huggingface/transformers.js-examples/tree/main/node-esm) / [CJS Example](https://github.com/huggingface/transformers.js-examples/tree/main/node-cjs) |
| [Deno](https://deno.com/)      | A modern runtime for JavaScript and TypeScript that is secure by default. It uses ES modules and even features experimental WebGPU support. | [Deno Example](https://github.com/huggingface/transformers.js-examples/tree/main/deno-embed)                                                                                          |
| [Bun](https://bun.sh/)         | A fast JavaScript runtime optimized for performance. It features a built-in bundler, transpiler, and package manager.                       | [Bun Example](https://github.com/huggingface/transformers.js-examples/tree/main/bun)                                                                                                  |

## A new home on NPM and GitHub

Finally, we're delighted to announce that Transformers.js will now be published under the official Hugging Face organization on NPM as [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers) (instead of
[`@xenova/transformers`](https://www.npmjs.com/package/@xenova/transformers), which was used for v1 and v2).

We've also moved the repository to the official Hugging Face organization on GitHub (https://github.com/huggingface/transformers.js), which will be our new home — come say hi!

This is a significant milestone and we're extremely grateful to the community for helping us achieve this long-term goal! None of this would be possible without all of you... thank you! 🤗
