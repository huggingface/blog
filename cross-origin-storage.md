---
title: "Experimenting with the proposed Cross-Origin Storage API in Transformers.js"
thumbnail: /blog/assets/cross-origin-storage/thumbnail.jpg
authors:
- user: tomayac
  guest: true
  org: google
---

# Experimenting with the proposed Cross-Origin Storage API in Transformers.js

(This is a guest post by Developer Relations Engineer [Thomas Steiner](https://blog.tomayac.com/) from the Chrome team at Google.)

Transformers.js provides Web developers with a simple way to use the power of transformers in their Web apps through task-specific pipelines. To run inference in the browser, developers create an instance of [`pipeline()`](https://huggingface.co/docs/transformers.js/en/api/pipelines) and specify a task they want to use the pipeline for. As a concrete example, the following snippet shows how to set up an automatic speech recognition (ASR) pipeline.

```js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

const asr = await pipeline(
  'automatic-speech-recognition',
  'Xenova/whisper-tiny.en',
  { device: 'webgpu' },
);
const result = await asr('jfk.wav');
console.log(result);
```

![A minimalistic example of the automatic speech recognition pipeline.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/87a91qnbicf.png)

## The cache challenge

You will notice in the source code that I specified [`Xenova/whisper-tiny.en`](https://huggingface.co/Xenova/whisper-tiny.en) as the model, which is a very decent choice for common English automatic speech recognition tasks. In fact, it's even _the_ default model according to the Transformers.js [default model resolution](https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/pipelines/index.js), as per the linked [excerpt](https://github.com/huggingface/transformers.js/blob/bc9cf7400f4f2c8695016699f879e31026ff0313/packages/transformers/src/pipelines/index.js#L151-L158).

### Model resources

When you [run this example in the browser](https://googlechrome.github.io/samples/transformersjs-automatic-speech-recognition/index.html), Transformers.js automatically takes care of downloading and caching the relevant model resources and Wasm files. The following screenshot shows the Chrome DevTools [Cache storage](https://developer.chrome.com/docs/devtools/storage/cache) section after visiting the app. When you reload the page, the resources are served from the [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache), and the model returns results almost instantly.

![The Chrome DevTools Cache storage section showing Whisper AI model resources and Wasm runtime files after visiting the app.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/otd8tt1gusb.png)

However, `Xenova/whisper-tiny.en` being a popular model (and, as mentioned before, even being _the_ ASR default model in Transformers.js), you can well imagine that more than just one app that you visit would use it. To simulate this situation, here's the same example app from before, but served from a [different origin](https://rawcdn.rawgit.net/GoogleChrome/samples/c4192bd7a3c66fc288a7b22b77acb935df00b8a1/transformersjs-automatic-speech-recognition/index.html). When you visit this different origin app, rather than being usable almost instantly, the browser instead has to download and cache all the model resources again, even if they're byte-by-byte the same as before. Even in this toy example, this adds up to 177 MB of duplicate download and storage, as you can examine in the **Storage** section of the Chrome DevTools [Application panel](https://developer.chrome.com/docs/devtools/application#open_the_application_panel). You can imagine that this quickly adds up.

![The Chrome DevTools Storage overview showing 177 MB of used storage.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/9byoniem0pw.png)

### Wasm runtime resources

But it gets worse. Let's add a second pipeline to the toy example: sentiment analysis. Sentiment analysis [by default](https://github.com/huggingface/transformers.js/blob/bc9cf7400f4f2c8695016699f879e31026ff0313/packages/transformers/src/pipelines/index.js#L65) uses the [`Xenova/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english) model. By not specifying the model, Transformers.js' default model resolution automatically picks it for you.

```js
const classifier = await pipeline('sentiment-analysis');
const sentiment = await classifier(result.text);
console.log(sentiment);
```

![image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/le7l1km7o4g.png)

Two entirely different AI models, but they depend on the same 4,733 kB `ort-wasm-simd-threaded.asyncify.wasm` WebAssembly (Wasm) runtime file [from the underlying ONNX Runtime library](https://onnxruntime.ai/docs/api/js/interfaces/Env.WasmFilePaths.html#wasm) that Transformers.js is built on top of. Open the [extended demo on a different origin](https://rawcdn.rawgit.net/GoogleChrome/samples/d47114a15637383015c274e7bdcd81e1a17b0ccf/transformersjs-automatic-speech-recognition/index2.html), and you will notice in the [**Network** tab](https://developer.chrome.com/docs/devtools/network#load) how also the Wasm runtime gets downloaded and cached again.

![Chrome DevTools Network panel showing the download of the Wasm runtime resource.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/pz12g20fqeg.png)

So even if you run apps that don't share the same AI models, your browser still makes redundant requests for shared Wasm resources you already have, and on top of that also caches them again, which consumes space on your hard disk.

### Cache isolation

#### AI model resources serving

By default, **AI model resources** come from the [Hugging Face Hub](https://huggingface.co/docs/hub/en/models-the-hub), and ultimately the Hugging Face CDN. The browser makes a request for a resource like [`https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json) which then gets redirected to the final CDN URL like [`https://huggingface.co/api/resolve-cache/models/Xenova/distilbert-base-uncased-finetuned-sst-2-english/0b6928efcb76139cae2c6881d49cda67fe119f42/config.json?%2FXenova%2Fdistilbert-base-uncased-finetuned-sst-2-english%2Fresolve%2Fmain%2Fconfig.json=&etag=%223c36342ef1f74de2797d667c68c6b7b988d0b87c%22`](https://huggingface.co/api/resolve-cache/models/Xenova/distilbert-base-uncased-finetuned-sst-2-english/0b6928efcb76139cae2c6881d49cda67fe119f42/config.json?%2FXenova%2Fdistilbert-base-uncased-finetuned-sst-2-english%2Fresolve%2Fmain%2Fconfig.json=&etag=%223c36342ef1f74de2797d667c68c6b7b988d0b87c%22) in this case.

#### Wasm runtime resources serving

The **Wasm runtime resources** are served from the [jsDelivr CDN](https://www.jsdelivr.com/) by default. For example, `ort-wasm-simd-threaded.asyncify.wasm` comes from [`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm`](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm) at the time of this writing.

Now you may say that if different apps, even though running on different origins, in the end serve their resources from the same CDN URLs, caching shouldn't be a problem, as long as the final URLs are the same. Unfortunately, this is not how caching works in browsers for a long time. The article [Gaining security and privacy by partitioning the cache](https://developer.chrome.com/blog/http-cache-partitioning) goes into all the details, but essentially, **caches are isolated by origin** to prevent timing attacks: the time a website takes to respond to HTTP requests can reveal that the browser has accessed the same resource in the past, which makes the browser vulnerable to security and privacy leaks.

#### Chrome's implementation

The concrete implementation may vary by browser, but in Chrome, cached resources are keyed using a Network Isolation Key in addition to the **resource URL**. The Network Isolation Key is composed of the **top-level site** and the **current-frame site**. Take the previous toy examples hosted on the origins `https://googlechrome.github.io` and `https://rawcdn.rawgit.net`. If they both use the Wasm runtime from `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm`, their cache keys will look like in the following table.

<table>
  <thead>
    <tr>
      <th colspan="2">Network Isolation Key</th>
      <th rowspan="2"><strong>Resource URL</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Top-level site</strong></td>
      <td><strong>Current-frame site</strong></td>
    </tr>
    <tr>
      <td><p><pre>
https://googlechrome.github.io
</pre></p></td>
      <td><p><pre>
https://googlechrome.github.io
</pre></p></td>
      <td><p><pre>
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm
</pre></p></td>
    </tr>
    <tr>
      <td><p><pre>
https://rawcdn.rawgit.net
</pre></p></td>
      <td><p><pre>
https://rawcdn.rawgit.net
</pre></p></td>
      <td><p><pre>
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm
</pre></p></td>
    </tr>
  </tbody>
</table>

So even if the resource URLs are exactly the same, since the Network Isolation Keys don't match, there's no cache hit, which means duplicate download and duplicate storage. This is the challenge that the Cross-Origin Storage proposal aims to tackle.

## Enter the Cross-Origin Storage API

> **💡 Note:** The Cross-Origin Storage API is an early-stage proposal that isn't final. While the proposed API is not yet natively implemented in any browser, you don't have to wait to experiment with it. Install the [Cross-Origin Storage extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih) to inject the `navigator.crossOriginStorage` polyfill on all pages and test the complete flow.

The proposed **[Cross-Origin Storage](https://github.com/WICG/cross-origin-storage) (COS) API** introduces a dedicated `navigator.crossOriginStorage` interface through which web apps can store and retrieve large files across origin boundaries, identified not by a URL, but by a cryptographic hash.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/klwb5fryaa.png" alt="The Cross-Origin Storage API logo: a stylized walking person, as typically encountered on crosswalk signs." width="200" height="200">

That last point about cryptographic hashes is key. Because COS identifies files by their **hash** rather than by their URL or origin, the same `ort-wasm-simd-threaded.asyncify.wasm` Wasm runtime you downloaded while visiting `https://googlechrome.github.io` is recognized as identical to the one `https://rawcdn.rawgit.net` is about to request, no matter where either of the two origins fetched it from. See the following code snippet that illustrates the basic flow.

```js
const hash = {
  algorithm: 'SHA-256',
  value: '8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4',
};

try {
  const handle = await navigator.crossOriginStorage.requestFileHandle(hash);
  // Cache hit! Get the file as a Blob and use it directly.
  const fileBlob = await handle.getFile();
} catch {
  // Cache miss. Download from network, then store for next time.
  const fileBlob = await fetch('https://cdn.jsdelivr.net/.../ort-wasm-simd-threaded.asyncify.wasm')
    .then(r => r.blob());
  const handle = await navigator.crossOriginStorage.requestFileHandle(
    hash,
    { create: true, origins: '*' },
  );
  const writableStream = await handle.createWritable();
  await writableStream.write(fileBlob);
  await writableStream.close();  
}
```

If the resource is in COS, you get back a [`FileSystemFileHandle`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle) from which you can read the blob directly via [`getFile()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle/getFile) (the resulting [`File`](https://developer.mozilla.org/en-US/docs/Web/API/File) inherits from [`Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob)). If the resource is not in COS, you fall back to the network, and write the resource into COS for the next app that needs it, which could be your app, or another unrelated app, potentially on a completely different origin.

The API is deliberately shaped after the [File System Standard](https://fs.spec.whatwg.org/)'s [`FileSystemDirectoryHandle.getFileHandle()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemDirectoryHandle/getFileHandle) you likely are familiar with from the [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) (OPFS) API. The `hash` parameter plays the same role as the `name` parameter in OPFS: uniquely identifying a resource. The `options.create` flag works the same way: absent or `false` for read-only access, `true` when you intend to write.

### Control who can read what

Not every resource should be globally shared. COS gives developers precise control over visibility through the `origins` option when storing a file.

* Setting `origins: '*'` makes a file **globally available**. Any origin can find it by hash. This is the right choice for AI model resources or the Wasm runtime in the Transformers.js example: the whole point is that every app on the Web benefits from a single cached copy.
* Passing a specific list of origins, like `origins: ['https://write.example.com', 'https://calculate.example.com']`, **restricts** access to those sites. This works well for proprietary resources shared across a company's own properties that shouldn't be discoverable by anyone else, like a proprietary proofreading AI model used in a commercial office suite.
* Omitting `origins` entirely makes the file available only to **[same-site](https://web.dev/articles/same-site-same-origin#same-site-cross-site) origins**. This is a sensible default for resources shared across all of an organization's subdomains, but not intended to cross organizational boundaries.

One important rule: visibility can be upgraded but never downgraded. If a file is already globally available, a later attempt to store it with a restricted `origins` list is silently ignored. This prevents a malicious actor from re-storing a public resource and narrowing its availability. The reverse is possible: a file initially stored with a restricted `origins` list can later be made more permissive. Any site, not just the original storer, can call `requestFileHandle()` for the same hash (hashes are not a secret) with `create: true` and a broader `origins` value, and given the browser verifies the hash matches, the resource becomes available to the wider audience from that point on. Note that the upgrading site **must** still write the full file through the returned handle. This requirement exists to prevent sites from exploiting the upgrade path as a side-channel to detect whether a particular file was already stored in COS.

### Integrity by design

A subtle but important property of COS is that the browser **verifies the hash** when you write a file. If the data you write doesn't match the declared hash, the write fails with an error. This makes integrity checking automatic: an app reading a file from COS can be confident it's getting exactly the bytes it expected. The same guarantee it would have had if it had computed the hash itself after a network download.

This turns out to be doubly useful in the Transformers.js scenario. Today, after downloading model weights, most apps have no practical way to verify that the CDN served the right bytes. With COS, every file in the store is implicitly verified on write, no matter where it came from, the official Hugging Face CDN or a random site's self-hosted mirror.

### Privacy without sacrificing utility

Of course a cross-origin shared cache raises the same question as the partitioned HTTP cache in reverse: if any site can probe for the presence of a file by hash, couldn't an attacker learn something about the user's browsing history by checking whether, say, a game engine Wasm module is cached?

COS addresses this through two complementary mechanisms:

-  First, the `origins` field: proprietary resources that shouldn't be globally probeable simply shouldn't be stored with `origins: '*'`, which, through **developer education**, developers are encouraged to consider whenever it makes sense.
-  Second, **availability gating**: even for globally declared files, the browser may suppress confirmation of a file's presence if it hasn't been encountered across a sufficient number of distinct origins. A file that only appears on one or two sites could still serve as a cross-site identifier, so the browser may return an error as if the file weren't there at all, regardless of what's physically on disk. On the Chrome team, we are conscious of the possible privacy leaks uncommon resources could cause and plan generally to mitigate it through restricting which exact resources can be cached. The concrete mitigations are still being fleshed out.

Crucially, this means an error is not a definitive answer. It might mean "not stored", or it might mean "stored, but the browser isn't telling you". Apps should always handle it the same way: fall back to the network.

### What this means for the Transformers.js example

Going back to the toy examples from before: the `ort-wasm-simd-threaded.asyncify.wasm` runtime weighs in at 4,733 kB and is shared by every Transformers.js-powered app regardless of which AI model it uses. With COS, the first app to load it downloads it once and stores it under its SHA-256 hash with `origins: '*'`. Every subsequent app, whether on `https://googlechrome.github.io`, on `https://rawcdn.rawgit.net`, or any other origin, finds it in COS immediately. The 177 MB of duplicate Whisper model weights? Same story: `Xenova/whisper-tiny.en` gets downloaded once, recognized by hash the second time around, and served from COS in milliseconds. And of course, the same also happens for  `Xenova/distilbert-base-uncased-finetuned-sst-2-english`.

Transformers.js itself is already piloting the COS API at the library level. [Pull request #1549](https://github.com/huggingface/transformers.js/pull/1549) introduced an experimental COS cache backend behind an opt-in flag. Enabling it takes a single line before you set up your pipeline:

```js
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

// 👇 Opt in to the experimental Cross-Origin Storage cache backend.
env.experimental_useCrossOriginStorage = true;

const asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en', { device: 'webgpu' });
const result = await asr('jfk.wav');
console.log(result);
```

Note the `experimental_` prefix on the flag. It's intentional and signals that the underlying browser API has not yet been standardized and may change without a major version bump. With that flag set, Transformers.js resolves the SHA-256 hash for each [Xet-tracked](https://huggingface.co/docs/hub/en/xet/index) model file (the large ONNX weight files) by fetching the raw Xet pointer ([example raw pointer file](https://huggingface.co/Xenova/whisper-tiny.en/raw/main/onnx/decoder_model.onnx)) and extracting its `oid sha256:` field. It then uses that hash as the key for `navigator.crossOriginStorage`. If the model is already in COS (because another site stored it there first), it's served instantly without a network round-trip. If not, it falls back to a regular download and stores the result in COS for the next caller. With the toy example, the advantage in practice is that `Xenova/whisper-tiny.en` and `Xenova/distilbert-base-uncased-finetuned-sst-2-english` (and of course `ort-wasm-simd-threaded.asyncify.wasm`) only ever need to cross the ether once, regardless of how many different origins ask for them.

### Model flexibility

The toy example works just fine with `Xenova/whisper-tiny.en`, but of course you possibly wouldn't say no if the user already has [any of the other Whisper variants](https://huggingface.co/Xenova/models?search=whisper) in their COS cache. For example, the user might already have [`Xenova/whisper-large-v3`](https://huggingface.co/Xenova/whisper-large-v3), which, as the name suggests, is a lot larger than the tiny variant. Transformers.js's [Model Registry](https://huggingface.co/docs/transformers.js/en/api/utils/model_registry) makes being flexible about your models possible. If you know your app's needs can be addressed by, for example, any of `Xenova/whisper-tiny.en`, `whisper-medium.en`, or `Xenova/whisper-large-v3`, you can check the registry for the associated files for each model, probe for their existence in the COS cache (which may partially or completely contain the model resources you need), and then take a decision what model to choose eventually. The [`ModelRegistry.is_cached()`](https://huggingface.co/docs/transformers.js/en/api/utils/model_registry#modelregistryiscachedmodelid-options--promise--boolean-) API directly integrates with COS (and of course the Cache API), so this operation is really ergonomic.

### Try it today

The COS API is not yet natively implemented in any browser, but you don't have to wait to experiment with it. Install the [Cross-Origin Storage extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih) to inject the `navigator.crossOriginStorage` polyfill on all pages and test the complete flow. Check out the [source code of the extension](https://github.com/web-ai-community/cross-origin-storage-extension) and follow the [usage instructions](https://github.com/web-ai-community/cross-origin-storage-extension?tab=readme-ov-file#usage) to get started.

![Chrome Web Store page for the Cross-Origin Storage extension.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/0q3rowmy67ta.png)

With the extension installed, try the full end-to-end experience right now: open the first [toy example with COS enabled](https://googlechrome.github.io/samples/transformersjs-automatic-speech-recognition/index3.html), let it load `Xenova/whisper-tiny.en`, then open the [toy example with COS enabled from the second origin](https://rawcdn.rawgit.net/GoogleChrome/samples/1e4f2b8c10adc394352c6ec8327bb503bac7aba1/transformersjs-automatic-speech-recognition/index3.html). Instead of the 177 MB re-download you saw before, the model is served from COS in milliseconds. When you open the extension's popup window, you can see COS in action. If you **View by Resource**, you can see the resource with the SHA-256 hash `950978b1dbcbf250335358c1236053ba19a7f7849b33dc777f4421b72b7626fa` shared across `https://googlechrome.github.io` and `https://rawcdn.rawgit.net`.  It may not be obvious, but as you can verify by comparing the SHA-256 hash on Hugging Face, you're looking at [`https://huggingface.co/Xenova/whisper-tiny.en/blob/main/onnx/decoder_model_merged.onnx`](https://huggingface.co/Xenova/whisper-tiny.en/blob/main/onnx/decoder_model_merged.onnx). For now, the extension is mostly aimed at power users like you. Once implemented in the browser, there will be a friendlier integration in the browser's **Settings** page. The screenshot below shows the extension's popup window with the **View by Resource** tab active, where you can see the shared resource with its hash and the two origins that have it in their COS cache.

![A resource seen in the Cross-Origin Storage extension, showing it's shared between two origins.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/usg5dq7dhm.png)

## Call to action

If you're building your own Transformers.js app, the call to action is simple: add `env.experimental_useCrossOriginStorage = true` before your first `pipeline()` call, install the extension, and watch the duplicate downloads disappear from your Network tab. Every site that opts in makes the experience faster and cheaper for every other site's users. Opting in is completely risk-free: if the COS API isn't supported because the user doesn't have the COS extension installed, the code just falls back to the default path (the [Web Cache](https://developer.mozilla.org/en-US/docs/Web/API/Cache) API).

Transformers.js is not alone in experimenting with COS. [WebLLM](https://webllm.mlc.ai/) (opt-in, see [documentation](https://webllm.mlc.ai/docs/user/advanced_usage.html#using-cross-origin-storage-cache)) and [wllama](https://github.com/ngxson/wllama) (automatic, see [PR](https://github.com/ngxson/wllama/pull/248)) are likewise excited about this proposed API.

On the Chrome team, we're [considering implementing the COS API](https://chromestatus.com/feature/5163371507875840) natively in the browser. As an early stage proposal, we welcome feedback on the API, and the shape of the proposal itself. The [Cross-Origin Storage repository](https://github.com/WICG/cross-origin-storage) is the place to file issues, [express support](https://github.com/WICG/cross-origin-storage/labels/expression%20of%20support), or open PRs.
