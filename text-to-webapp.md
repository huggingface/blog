---
title: "Using AI to generate web apps"
thumbnail: /blog/assets/153_text_to_webapp/thumbnail.jpg
authors:
- user: jbilcke-hf
---

# Using AI to generate web apps

<!-- {blog_metadata} -->
<!-- {authors} -->

As more code generation models become publicly available, it is now possible to do text-to-web and even text-to-app in ways that we couldn't imagine before.

This tutorial presents a direct approach to AI web content generation by streaming and rendering the content all in one go.

**Try the live demo here!** →  **[Webapp Factory](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-wizardcoder)**

![main_demo.gif](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/153_text_to_webapp/main_demo.gif)

## Using LLM in Node apps

While we usually think of Python for everything related to AI and ML, the web development community relies heavily on JavaScript and Node.

Here are some ways you can use large language models on this platform.

### By running a model locally

Various approaches exist to running LLM in Javascript, from using [ONNX](https://www.npmjs.com/package/onnxruntime-node) to converting code to [WASM](https://blog.mithrilsecurity.io/porting-tokenizers-to-wasm/) and calling external processes written in other languages.

Some of those techniques are now used in ready-to-use NPM libraries:

- Using AI/ML libraries such as [transformers.js](https://huggingface.co/docs/transformers.js/index) (which supports [code generation](https://huggingface.co/docs/transformers.js/api/models#codegenmodelgenerateargs-codepromiseampltanyampgtcode))
- Using dedicated LLM libraries such as [llama-node](https://github.com/Atome-FE/llama-node) (or [web-llm](https://github.com/mlc-ai/web-llm) for the browser)
- Using Python libraries through a bridge such as [Pythonia](https://www.npmjs.com/package/pythonia)

However, running large language models in such an environment can be pretty resource-intensive, especially if you are not able to use hardware acceleration.

### By using an API

Today, various cloud providers propose commercial APIs to use language models.

Hugging Face offers a free [Inference API](https://huggingface.co/docs/api-inference/index) to allow anyone to use small to medium-sized models from the community.

We also have an [Inference Endpoints API](https://huggingface.co/inference-endpoints) for those who require larger models or custom inference code.

These two APIs can be used from Node using the [Hugging Face Inference API library](https://www.npmjs.com/package/@huggingface/inference) on NPM.

💡 Top performing models generally require a lot of memory (32 Gb, 64 Gb or more) and hardware acceleration to get good latency (see [the benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)). But we are also seeing a trend of models shrinking in size while keeping relatively good results on some tasks, with requirements as low as 16 Gb or even 8 Gb of memory.

## Architecture

We are going to use NodeJS to create our generative AI web server.

The model will be [WizardCoder-15B](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) running on the Inference Endpoints API, but feel free to try with another model and stack.

If you are interested in other solutions, here are some pointers to alternative implementations:

- Using the Inference API: [code](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-any-model/tree/main) and [space](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-any-model)
- Using a Python module from Node: [code](https://huggingface.co/spaces/jbilcke-hf/template-node-ctransformers-express/tree/main) and [space](https://huggingface.co/spaces/jbilcke-hf/template-node-ctransformers-express)
- Using llama-node (llama cpp): [code](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-llama-node/tree/main)

## Initializing the project

First, we need to setup a new Node project (you can clone [this template](https://github.com/jbilcke-hf/template-node-express/generate) if you want to).

```html
git clone https://github.com/jbilcke-hf/template-node-express tutorial
cd tutorial
nvm use
npm install
```

Then, we can install the Hugging Face Inference client:

```html
npm install @huggingface/inference
```

And set it up in `src/index.mts``:

```javascript
import { HfInference } from '@huggingface/inference'

// to keep your API token secure, in production you should use something like:
// const hfi = new HfInference(process.env.HF_API_TOKEN)
const hfi = new HfInference('** YOUR TOKEN **')
```

## UsiConfiguring the Inference Endpoint


💡 **Note:** If you do not wish to pay for an endpoint instance for the purpose of this tutorial, you can skip this step and instead look at [the following code example](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-any-model/blob/main/src/index.mts) that uses our free Inference API (please note that this will only work with smaller models perhaps less capable of code generation).



To deploy a new endpoint you can go to the [endpoint creation page](https://ui.endpoints.huggingface.co/new).

You will have to select `WizardCoder` in the **Model Repository** dropdown and make sure that a GPU instance large enough is selected:

![new_endpoint.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/153_text_to_webapp/new_endpoint.jpg)

Once your endpoint is created, you can copy the URL from [this page](https://ui.endpoints.huggingface.co):

![deployed_endpoints.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/153_text_to_webapp/deployed_endpoints.jpg)

Configure the client to use it:

```javascript
const hf = hfi.endpoint('** URL TO YOUR ENDPOINT **')
```

You can now tell the inference client to use our private endpoint and call our model:

```javascript
const { generated_text } = await hf.textGeneration({
  inputs: 'a simple "hello world" html page: <html><body>'
});
```

## Generating the HTML stream

t is now time to return some HTML to the web client when they visit a URL, say `/app`

To achieve this, we will use Express.js and create an endpoint in which we will stream the results from the Hugging Face Inference API.


```javascript
import express from 'express'

import { HfInference } from '@huggingface/inference'

const hfi = new HfInference('** YOUR TOKEN **')
const hf = hfi.endpoint('** URL TO YOUR ENDPOINT **')

const app = express()
```

As we do not have any UI for the moment, the interface will be a simple URL parameter for the prompt:

```javascript
app.get('/', async (req, res) => {

  // send the beginning of the page to the browser (the rest will be generated by the AI)
  res.write('<html><head></head><body>')

  const inputs = `# Task
Generate ${req.query.prompt}
# Out
<html><head></head><body>`

  for await (const output of hf.textGenerationStream({
    inputs,
    parameters: {
      max_new_tokens: 1000,
      return_full_text: false,
    }
  })) {
    // stream the result to the browser
    res.write(output.token.text)

    // also print to the console for debugging
    process.stdout.write(output.token.text)
  }

  req.end()
})

app.listen(3000, () => { console.log('server started') })
```

Start your web server:

```bash
npm run start
```

and open `https://localhost:3000?prompt=some%20prompt`. You should see some primitive HTML content after a few moments.

## Tuning the prompt

Each language model reacts differently to prompting. For WizardCoder, simple instructions often work the best:

```javascript
const inputs = `# Task
Generate ${req.query.prompt}
# Orders
Write application logic inside a JS <script></script> tag.
Use a central layout to wrap everything in a <div class='flex flex-col items-center'>
# Out
<html><head></head><body>`
```

### Using Tailwind

Tailwind is a popular CSS framework for styling content, and WizardCoder is good at it out of the box.

This allows code generation to create style on the go without generating a stylesheet at the beginning or the end of the page (which would make the page feel stuck).

To improve results, we can also guide the model by showing the way (`<body class="p-4 md:p-8">`).

```javascript
const inputs = `# Task
Generate ${req.query.prompt}
# Orders
You must use TailwindCSS utility classes (Tailwind is already injected in the page).
Write application logic inside a JS <script></script> tag!.
Use a central layout to wrap everything in a <div class='flex flex-col items-center'>
# Out
<html><head></head><body class="p-4 md:p-8">`
```

### Preventing hallucination

It can be difficult to reliably prevent hallucinations and failures (such as parroting back the whole instructions, or writing “lorem ipsum” placeholder text) on light models dedicated to code generation, compared to larger general-purpose models, but we can try to mitigate it.

You can try to use an imperative tone and repeat the instructions, an efficient way can also be to show the way by giving a part of the output in English:

```javascript
const inputs = `# Task
Generate ${req.query.prompt}
# Orders
Never repeat those instructions, instead write the final code!
You must use TailwindCSS utility classes (Tailwind is already injected in the page)!
Write application logic inside a JS <script></script> tag!
This is not a demo app, so you MUST use English, no Latin! Write in English! 
Use a central layout to wrap everything in a <div class='flex flex-col items-center'>
# Out
<html><head><title>App</title></head><body class="p-4 md:p-8">`
```

## Adding support for images

We now have a system that can generate HTML, CSS and JS code, but it is prone to hallucinating broken URLs when asked to produce images.

Luckily we have a lot of options to choose from when it comes to image generation models!

→ The fastest way to get started is to call a Stable Diffusion model using our free [Inference API](https://huggingface.co/docs/api-inference/index) with one of the [public models](https://huggingface.co/models?other=stable-diffusion) available on the hub:

```javascript
app.get('/image', async (req, res) => {
  const blob = await hf.textToImage({
    inputs: `${req.query.caption}`,
    model: 'stabilityai/stable-diffusion-2-1'
  })
  const buffer = Buffer.from(await blob.arrayBuffer())
  res.setHeader('Content-Type', blob.type)
  res.setHeader('Content-Length', buffer.length)
  res.end(buffer)
})
```

To instruct WizardCoder to use this endpoint, adding the following line to the prompt was enough to achieve the desired effect (you may have to tweak it for other models):

```
To generate images from captions call the /image API: <img src="/image?caption=photo of something in some place" />
```

But you can try to ask for more specific ways of prompting images:

```
Only generate a few images and use descriptive photo captions with at least 10 words!
```

![preview_image.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/153_text_to_webapp/preview_image.jpg)

## Adding some UI

[Alpine.js](https://alpinejs.dev/) is a minimalist framework that allows us to create interactive UIs without any setup, build pipeline, JSX processing etc.

Everything is done within the page, making it a great candidate to create the UI of a quick demo.

Here is a static HTML page that you can put in `/public/index.html`:

```html
<html>
  <head>
    <title>Tutorial</title>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body>
    <div class="flex flex-col space-y-3 p-8" x-data="{ draft: '', prompt: '' }">
      <textarea
	      name="draft"
	      x-model="draft"
	      rows="3"
	      placeholder="Type something.."
	      class="font-mono"
	     ></textarea> 
      <button
        class="bg-green-300 rounded p-3"
        @click="prompt = draft">Generate</button>
      <iframe :src="`/app?prompt=${prompt}`"></iframe>
    </div>
  </body>
</html>
```

To make this work, you will have to make some changes:

```javascript
...

// going to localhost:3000 will load the file from /public/index.html
app.use(express.static('public'))

// we changed this from '/' to '/app'
app.get('/app', async (req, res) => {
   ...
```

## Final result

Look at the space repository for [more complete exemple](https://huggingface.co/spaces/jbilcke-hf/webapp-factory-wizardcoder/blob/main/public/index.html) of UI.

this final space makes use of Daisy UI to improve HTML generation, by trying to use a more compact representation of design system classes (eg. `<button class="btn" />`).

![main_demo.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/153_text_to_webapp/main_demo.jpg)