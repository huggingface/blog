---
title: 'Sentence Transformers in the Hugging Face Hub'
# thumbnail: /blog/assets/22_few_shot_learning_gpt_neo_and_inference_api/thumbnail.png
---

<h1>
    Sentence Transformers in the Hugging Face Hub
</h1>

<div class="blog-metadata">
    <small>Published June 17, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/sentence-transformers-in-the-hub.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/osanseviero">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1622621035602-6032802e1f993496bc14d9e3.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>osanseviero</code>
            <span class="fullname">Omar Sanseviero</span>
        </div>
    </a>
    <a href="/nreimers">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://aeiljuispo.cloudimg.io/v7/https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1596792577829-5eff4688ff69163f6f59e66c.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>nreimers</code>
            <span class="fullname">Nils Reimers</span>
        </div>
    </a>
</div>

Over the past few weeks, we've built collaborations with many Open Source frameworks in the machine learning ecosystem. One that gets us particularly excited is Sentence Transformers.

[Sentence Transformers](https://github.com/UKPLab/sentence-transformers) is a framework for sentence, paragraph and image embeddings. This allows to derive semantically meaningful embeddings (1) which is useful for applications such as semantic search or multi-lingual zero shot classification. As part of Sentence Transformers [v2 release](TODO add link), there are a lot of cool new features:

- Sharing your models in the Hub easily.
- Widgets and Inference API for sentence embeddings and sentence similarity.
- Better sentence-embeddings models available ([benchmark](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models) and [models](https://huggingface.co/sentence-transformers) in the Hub).

With over 90 pretrained Sentence Transformers models for more than 100 languages in the Hub, anyone can benefit from them and easily use them. Pre-trained models can be loaded and used directly with few lines of code:

```python
from sentence_transformers import SentenceTransformer
sentences = ["Hello World", "Hallo Welt"]

# TODO: update with sentence-transformers org model once they are updated
model = SentenceTransformer('osanseviero/full-sentence-distillroberta3') 
embeddings = model.encode(sentences)
print(embeddings)
```

But not only this. People will probably want to either demo their models or play with other models easily, so we're happy to announce the release of two new widgets in the Hub! The first one is the `feature-extraction` widget which shows the sentence embedding.

<!-- Hackiest hack ever for the draft -->
<div><a class="text-xs block mb-3 text-gray-300" href="/osanseviero/full-sentence-distillroberta3"><code>osanseviero/full-sentence-distillroberta3</code></a>
					<div class="p-5 shadow-sm rounded-xl bg-white max-w-md"><div class="SVELTE_HYDRATER " data-props="{&quot;apiUrl&quot;:&quot;https://api-inference.huggingface.co&quot;,&quot;model&quot;:{&quot;author&quot;:&quot;osanseviero&quot;,&quot;autoArchitecture&quot;:&quot;AutoModel&quot;,&quot;branch&quot;:&quot;main&quot;,&quot;cardData&quot;:{&quot;tags&quot;:[&quot;feature-extraction&quot;],&quot;widget&quot;:[{&quot;text&quot;:&quot;Hello world&quot;}]},&quot;cardSource&quot;:true,&quot;config&quot;:{&quot;model_type&quot;:&quot;distilbert&quot;},&quot;pipeline_tag&quot;:&quot;feature-extraction&quot;,&quot;library_name&quot;:&quot;transformers&quot;,&quot;mask_token&quot;:&quot;[MASK]&quot;,&quot;modelId&quot;:&quot;osanseviero/full-sentence-distillroberta3&quot;,&quot;private&quot;:false,&quot;tags&quot;:[&quot;pytorch&quot;,&quot;distilbert&quot;,&quot;transformers&quot;,&quot;feature-extraction&quot;],&quot;tag_objs&quot;:[{&quot;id&quot;:&quot;feature-extraction&quot;,&quot;label&quot;:&quot;Feature Extraction&quot;,&quot;type&quot;:&quot;pipeline_tag&quot;},{&quot;id&quot;:&quot;pytorch&quot;,&quot;label&quot;:&quot;PyTorch&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;transformers&quot;,&quot;label&quot;:&quot;Transformers&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;distilbert&quot;,&quot;label&quot;:&quot;distilbert&quot;,&quot;type&quot;:&quot;other&quot;}],&quot;widgetData&quot;:[{&quot;text&quot;:&quot;Hello world&quot;}]},&quot;shouldUpdateUrl&quot;:false}" data-target="InferenceWidget"><div class="flex flex-col w-full max-w-full
	"> <div class="font-semibold flex items-center mb-2"><div class="text-lg flex items-center"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" class="-ml-1 mr-1 text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="M11 15H6l7-14v8h5l-7 14v-8z" fill="currentColor"></path></svg>
			Hosted inference API</div> <a target="_blank" href="/docs"><svg class="ml-1.5 text-sm text-gray-400 hover:text-black" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M17 22v-8h-4v2h2v6h-3v2h8v-2h-3z" fill="currentColor"></path><path d="M16 8a1.5 1.5 0 1 0 1.5 1.5A1.5 1.5 0 0 0 16 8z" fill="currentColor"></path><path d="M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4z" fill="currentColor"></path></svg></a></div> <div class="flex items-center text-sm text-gray-500 mb-1.5"><div class="inline-flex items-center"><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M27 3H5a2 2 0 0 0-2 2v22a2 2 0 0 0 2 2h22a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm0 2v4H5V5zm-10 6h10v7H17zm-2 7H5v-7h10zM5 20h10v7H5zm12 7v-7h10v7z"></path></svg> <span>Feature Extraction</span></div> <div class="ml-auto"></div></div> <form><div class="flex h-10"><input class="form-input-alt flex-1 rounded-r-none " placeholder="Your sentence here..." required="" type="text"> <button class="btn-widget w-24 h-10 px-5 rounded-l-none border-l-0 " type="submit">Compute</button></div></form> <div class="mt-1.5"><div class="text-gray-400 text-xs">This model is currently loaded and running on the Inference API.</div> </div>   <div class="mt-auto pt-4 flex items-center text-xs text-gray-500"><button class="flex items-center cursor-not-allowed text-gray-300" disabled=""><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path><path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path><path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path></svg>
		JSON Output</button> <button class="flex items-center ml-auto"><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M22 16h2V8h-8v2h6v6z" fill="currentColor"></path><path d="M8 24h8v-2h-6v-6H8v8z" fill="currentColor"></path><path d="M26 28H6a2.002 2.002 0 0 1-2-2V6a2.002 2.002 0 0 1 2-2h20a2.002 2.002 0 0 1 2 2v20a2.002 2.002 0 0 1-2 2zM6 6v20h20.001L26 6z" fill="currentColor"></path></svg>
		Maximize</button></div> </div></div></div>
				</div>

But seeing a bunch of numbers might not be very useful to you (unless you're able to  understand the embeddings from a quick look, which would be impressive!). We're also introducing a new widget for a common use case of Sentence Transformers: computing sentence similarity.

<!-- Hackiest hack ever for the draft -->
<div><a class="text-xs block mb-3 text-gray-300" href="/osanseviero/full-sentence-distillroberta2"><code>osanseviero/full-sentence-distillroberta2</code></a>
					<div class="p-5 shadow-sm rounded-xl bg-white max-w-md"><div class="SVELTE_HYDRATER " data-props="{&quot;apiUrl&quot;:&quot;https://api-inference.huggingface.co&quot;,&quot;model&quot;:{&quot;author&quot;:&quot;osanseviero&quot;,&quot;autoArchitecture&quot;:&quot;AutoModel&quot;,&quot;branch&quot;:&quot;main&quot;,&quot;cardData&quot;:{&quot;tags&quot;:[&quot;sentence-transformers&quot;,&quot;sentence-similarity&quot;]},&quot;cardSource&quot;:true,&quot;config&quot;:{&quot;architectures&quot;:[&quot;RobertaModel&quot;],&quot;model_type&quot;:&quot;roberta&quot;},&quot;pipeline_tag&quot;:&quot;sentence-similarity&quot;,&quot;library_name&quot;:&quot;sentence-transformers&quot;,&quot;mask_token&quot;:&quot;<mask>&quot;,&quot;modelId&quot;:&quot;osanseviero/full-sentence-distillroberta2&quot;,&quot;private&quot;:false,&quot;tags&quot;:[&quot;pytorch&quot;,&quot;jax&quot;,&quot;roberta&quot;,&quot;sentence-transformers&quot;,&quot;sentence-similarity&quot;],&quot;tag_objs&quot;:[{&quot;id&quot;:&quot;sentence-similarity&quot;,&quot;label&quot;:&quot;Sentence Similarity&quot;,&quot;type&quot;:&quot;pipeline_tag&quot;},{&quot;id&quot;:&quot;pytorch&quot;,&quot;label&quot;:&quot;PyTorch&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;jax&quot;,&quot;label&quot;:&quot;JAX&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;sentence-transformers&quot;,&quot;label&quot;:&quot;Sentence Transformers&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;roberta&quot;,&quot;label&quot;:&quot;roberta&quot;,&quot;type&quot;:&quot;other&quot;}],&quot;widgetData&quot;:[{&quot;source_sentence&quot;:&quot;That is a happy person&quot;,&quot;sentences&quot;:[&quot;That is a happy dog&quot;,&quot;That is a very happy person&quot;,&quot;Today is a sunny day&quot;]}]},&quot;shouldUpdateUrl&quot;:false}" data-target="InferenceWidget"><div class="flex flex-col w-full max-w-full
	"> <div class="font-semibold flex items-center mb-2"><div class="text-lg flex items-center"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" class="-ml-1 mr-1 text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="M11 15H6l7-14v8h5l-7 14v-8z" fill="currentColor"></path></svg>
			Hosted inference API</div> <a target="_blank" href="/docs"><svg class="ml-1.5 text-sm text-gray-400 hover:text-black" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M17 22v-8h-4v2h2v6h-3v2h8v-2h-3z" fill="currentColor"></path><path d="M16 8a1.5 1.5 0 1 0 1.5 1.5A1.5 1.5 0 0 0 16 8z" fill="currentColor"></path><path d="M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4z" fill="currentColor"></path></svg></a></div> <div class="flex items-center text-sm text-gray-500 mb-1.5"><div class="inline-flex items-center"><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M30 15H17V2h-2v13H2v2h13v13h2V17h13v-2z"></path><path d="M25.586 20L27 21.414L23.414 25L27 28.586L25.586 30l-5-5l5-5z"></path><path d="M11 30H3a1 1 0 0 1-.894-1.447l4-8a1.041 1.041 0 0 1 1.789 0l4 8A1 1 0 0 1 11 30zm-6.382-2h4.764L7 23.236z"></path><path d="M28 12h-6a2.002 2.002 0 0 1-2-2V4a2.002 2.002 0 0 1 2-2h6a2.002 2.002 0 0 1 2 2v6a2.002 2.002 0 0 1-2 2zm-6-8v6h6.001L28 4z"></path><path d="M7 12a5 5 0 1 1 5-5a5.006 5.006 0 0 1-5 5zm0-8a3 3 0 1 0 3 3a3.003 3.003 0 0 0-3-3z"></path></svg> <span>Sentence Similarity</span></div> <div class="ml-auto"></div></div> <form class="flex flex-col space-y-2"><label class="block "> <span class="text-sm text-gray-500">Source Sentence</span> <input class="mt-1.5 form-input-alt block w-full " placeholder="Your sentence here..." type="text"></label> <label class="block "> <span class="text-sm text-gray-500">Sentences to compare to</span> <input class="mt-1.5 form-input-alt block w-full " placeholder="Your sentence here..." type="text"></label> <label class="block ">  <input class=" form-input-alt block w-full " placeholder="Your sentence here..." type="text"></label><label class="block ">  <input class=" form-input-alt block w-full " placeholder="Your sentence here..." type="text"></label> <button class="btn-widget w-full h-10 px-5" type="submit">Add Sentence</button> <button class="btn-widget w-24 h-10 px-5 " type="submit">Compute</button></form> <div class="mt-1.5"><div class="text-gray-400 text-xs">This model can be loaded on the Inference API on-demand.</div> </div>   <div class="mt-auto pt-4 flex items-center text-xs text-gray-500"><button class="flex items-center cursor-not-allowed text-gray-300" disabled=""><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path><path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path><path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path></svg>
		JSON Output</button> <button class="flex items-center ml-auto"><svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M22 16h2V8h-8v2h6v6z" fill="currentColor"></path><path d="M8 24h8v-2h-6v-6H8v8z" fill="currentColor"></path><path d="M26 28H6a2.002 2.002 0 0 1-2-2V6a2.002 2.002 0 0 1 2-2h20a2.002 2.002 0 0 1 2 2v20a2.002 2.002 0 0 1-2 2zM6 6v20h20.001L26 6z" fill="currentColor"></path></svg>
		Maximize</button></div> </div></div></div>
				</div>

Of course, on top of the widgets, we also provide API endpoints in our Inference API that you can use to programmatically call your models!

```python
import json
import requests

# TODO: Update to sentence-transformers org.
API_URL = "https://api-inference.huggingface.co/models/osanseviero/full-sentence-distillroberta2"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))

data = query(
	{
		"inputs": {
			"source_sentence": "That is a happy person",
			"sentences": [
				"That is a happy dog",
				"That is a very happy person",
				"Today is a sunny day"
			]
		}
	}
)
```

## Unleashing the Power of Sharing

So why is this powerful? In a matter of minutes, you can share your trained models with the whole community.

```python
from sentence_transformers import SentenceTransformer

# Load or train a model

model.push_to_hub("my_new_model")
```

Now you will have a [repository](https://huggingface.co/osanseviero/my_new_model) in the Hub which hosts your model. A model card was automatically created. It describes the architecture by listing the layers and shows how to use the model with both `Sentence Transformers` and `🤗 Transformers`. You can also try out the widget and use the Inference API straight away!

If this was not exciting enough, your models will also be easily discoverable by [filtering for all](https://huggingface.co/models?filter=sentence-transformers) `Sentence Transformers` models.

## What's next?

Moving forward, we want to make this integration even more useful. In our roadmap, we expect training and evaluation data to be included in the automatically created model card.

And what's next for you? We're very excited to see your contributions! If you already have a `Sentence Transformer` repo in the Hub, you can now enable the widget and Inference API by changing the model card metadata. We'll 

```yaml
---
tags:
- sentence-transformers
- sentence-similarity # Or feature-extraction!
---
```

If you don't have any model in the Hub and want to learn more about Sentence Transformers, head to [www.SBERT.net](https://www.sbert.net)!

## Would you like to integrate your library to the Hub?

This integration is possible thanks to the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library which has all our widgets and the API for all our supported libraries. If you would like to integrate your library to the Hub, we have a [guide](https://huggingface.co/docs/adding-a-library) for you!

## References

1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
