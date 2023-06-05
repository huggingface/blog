---
title: "Welcome fastText to the Hugging Face Hub"
thumbnail: /blog/assets/147_fasttext/thumbnail.png
authors:
- user: sheonhan
- user: juanpino
---

# Welcome fastText to the Hugging Face Hub

<!-- {blog_metadata} -->
<!-- {authors} -->

[fastText](https://fasttext.cc/) is a library for efficient learning of text representation and classification. [Open-sourced](https://fasttext.cc/blog/2016/08/18/blog-post.html) by Meta AI in 2016, fastText integrates key ideas that have been influential in natural language processing and machine learning over the past few decades: representing sentences using bag of words and bag of n-grams, using subword information, and utilizing a hidden representation to share information across classes. 

To speed up computation, fastText uses hierarchical softmax, capitalizing on the imbalanced distribution of classes. All these techniques offer users scalable solutions for text representation and classification.

Hugging Face is now hosting official mirrors of word vectors of all 157 languages and the latest model for language identification. This means that using Hugging Face, you can easily download and use the models with a few commands. 

### Finding models

Word vectors for 157 languages and the language identification model can be found in the [Meta AI](https://huggingface.co/facebook) org. For example, you can find the model page for English word vectors [here](https://huggingface.co/facebook/fasttext-en-vectors) and the language identification model [here](https://huggingface.co/facebook/fasttext-language-identification).


### Widgets
This integration includes support for text classification and feature extraction widgets. Try out the language identifcation widget below!

<div class="bg-white pb-1">
	<div class="SVELTE_HYDRATER contents"
		data-props="{&quot;apiUrl&quot;:&quot;https://api-inference.huggingface.co&quot;,&quot;model&quot;:{&quot;author&quot;:&quot;facebook&quot;,&quot;cardData&quot;:{&quot;license&quot;:&quot;cc-by-nc-4.0&quot;,&quot;library_name&quot;:&quot;fasttext&quot;,&quot;tags&quot;:[&quot;text-classification&quot;,&quot;language-identification&quot;]},&quot;cardExists&quot;:true,&quot;discussionsDisabled&quot;:false,&quot;downloads&quot;:0,&quot;downloadsAllTime&quot;:0,&quot;id&quot;:&quot;facebook/fasttext-language-identification&quot;,&quot;isLikedByUser&quot;:true,&quot;inference&quot;:true,&quot;lastModified&quot;:&quot;2023-06-04T22:25:21.000Z&quot;,&quot;likes&quot;:8,&quot;pipeline_tag&quot;:&quot;text-classification&quot;,&quot;library_name&quot;:&quot;fasttext&quot;,&quot;model-index&quot;:null,&quot;private&quot;:false,&quot;repoType&quot;:&quot;model&quot;,&quot;gated&quot;:false,&quot;pwcLink&quot;:{&quot;error&quot;:&quot;Unknown error, can't generate link to Papers With Code.&quot;},&quot;tags&quot;:[&quot;arxiv:1607.04606&quot;,&quot;arxiv:1802.06893&quot;,&quot;arxiv:1607.01759&quot;,&quot;arxiv:1612.03651&quot;,&quot;fasttext&quot;,&quot;text-classification&quot;,&quot;language-identification&quot;,&quot;license:cc-by-nc-4.0&quot;,&quot;has_space&quot;],&quot;tag_objs&quot;:[{&quot;id&quot;:&quot;text-classification&quot;,&quot;label&quot;:&quot;Text Classification&quot;,&quot;subType&quot;:&quot;nlp&quot;,&quot;type&quot;:&quot;pipeline_tag&quot;},{&quot;id&quot;:&quot;fasttext&quot;,&quot;label&quot;:&quot;fastText&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;language-identification&quot;,&quot;label&quot;:&quot;language-identification&quot;,&quot;type&quot;:&quot;other&quot;},{&quot;id&quot;:&quot;has_space&quot;,&quot;label&quot;:&quot;Has a Space&quot;,&quot;type&quot;:&quot;other&quot;},{&quot;id&quot;:&quot;arxiv:1607.04606&quot;,&quot;label&quot;:&quot;arxiv:1607.04606&quot;,&quot;type&quot;:&quot;arxiv&quot;},{&quot;id&quot;:&quot;arxiv:1802.06893&quot;,&quot;label&quot;:&quot;arxiv:1802.06893&quot;,&quot;type&quot;:&quot;arxiv&quot;},{&quot;id&quot;:&quot;arxiv:1607.01759&quot;,&quot;label&quot;:&quot;arxiv:1607.01759&quot;,&quot;type&quot;:&quot;arxiv&quot;},{&quot;id&quot;:&quot;arxiv:1612.03651&quot;,&quot;label&quot;:&quot;arxiv:1612.03651&quot;,&quot;type&quot;:&quot;arxiv&quot;},{&quot;id&quot;:&quot;license:cc-by-nc-4.0&quot;,&quot;label&quot;:&quot;cc-by-nc-4.0&quot;,&quot;type&quot;:&quot;license&quot;}],&quot;hasHandlerPy&quot;:false,&quot;widgetData&quot;:[{&quot;text&quot;:&quot;Welcome fastText to Hugging Face!&quot;}]},&quot;shouldUpdateUrl&quot;:true,&quot;includeCredentials&quot;:true,&quot;isLoggedIn&quot;:true,&quot;callApiOnMount&quot;:true}"
		data-target="InferenceWidget">
		<div class="flex flex-col w-full max-w-full ">
			<div class="font-semibold flex items-center mb-2">
				<div class="text-lg flex items-center"><svg xmlns="http://www.w3.org/2000/svg"
						xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img"
						class="-ml-1 mr-1 text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet"
						viewBox="0 0 24 24">
						<path d="M11 15H6l7-14v8h5l-7 14v-8z" fill="currentColor"></path>
					</svg>
					Hosted inference API</div> <a target="_blank" href="https://api-inference.huggingface.co/"><svg
						class="ml-1.5 text-sm text-gray-400 hover:text-black" xmlns="http://www.w3.org/2000/svg"
						xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img"
						width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32">
						<path d="M17 22v-8h-4v2h2v6h-3v2h8v-2h-3z" fill="currentColor"></path>
						<path d="M16 8a1.5 1.5 0 1 0 1.5 1.5A1.5 1.5 0 0 0 16 8z" fill="currentColor"></path>
						<path d="M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4z"
							fill="currentColor"></path>
					</svg></a>
			</div>
			<div class="flex items-center justify-between flex-wrap w-full max-w-full text-sm text-gray-500 mb-0.5"><a
					class="hover:underline" href="/tasks/text-classification" target="_blank"
					title="Learn more about text-classification">
					<div class="inline-flex items-center mr-2 mb-1.5"><svg class="mr-1"
							xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
							aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em"
							preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);">
							<circle cx="10" cy="20" r="2" fill="currentColor"></circle>
							<circle cx="10" cy="28" r="2" fill="currentColor"></circle>
							<circle cx="10" cy="14" r="2" fill="currentColor"></circle>
							<circle cx="28" cy="4" r="2" fill="currentColor"></circle>
							<circle cx="22" cy="6" r="2" fill="currentColor"></circle>
							<circle cx="28" cy="10" r="2" fill="currentColor"></circle>
							<circle cx="20" cy="12" r="2" fill="currentColor"></circle>
							<circle cx="28" cy="22" r="2" fill="currentColor"></circle>
							<circle cx="26" cy="28" r="2" fill="currentColor"></circle>
							<circle cx="20" cy="26" r="2" fill="currentColor"></circle>
							<circle cx="22" cy="20" r="2" fill="currentColor"></circle>
							<circle cx="16" cy="4" r="2" fill="currentColor"></circle>
							<circle cx="4" cy="24" r="2" fill="currentColor"></circle>
							<circle cx="4" cy="16" r="2" fill="currentColor"></circle>
						</svg> <span>Text Classification</span></div>
				</a>
				<div class="ml-auto flex gap-x-1">
					<div class="relative mb-1.5  false false">
						<div
							class="no-hover:hidden inline-flex justify-between w-32 rounded-md border border-gray-100 px-4 py-1">
							<div class="text-sm truncate">Examples</div> <svg
								class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform false"
								xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"
								aria-hidden="true">
								<path fill-rule="evenodd"
									d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
									clip-rule="evenodd"></path>
							</svg>
						</div>
						<div
							class="with-hover:hidden inline-flex justify-between w-32 rounded-md border border-gray-100 px-4 py-1">
							<div class="text-sm truncate">Examples</div> <svg
								class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform false"
								xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"
								aria-hidden="true">
								<path fill-rule="evenodd"
									d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
									clip-rule="evenodd"></path>
							</svg>
						</div>
					</div>
				</div>
			</div>
			<form> <label class="block "> <span
						class="  block overflow-auto resize-y py-2 px-3 w-full min-h-[42px] max-h-[500px] whitespace-pre-wrap inline-block border border-gray-200 rounded-lg shadow-inner outline-none focus:ring focus:ring-blue-200 focus:shadow-inner dark:bg-gray-925 svelte-1wfa7x9"
						role="textbox" contenteditable="" style="--placeholder: 'Your sentence here...';"
						spellcheck="false" dir="auto">Welcome fastText to Hugging Face!</span></label> <button
					class="btn-widget w-24 h-10 px-5 mt-2" type="submit">Compute</button></form>
			<div class="mt-2">
				<div class="text-gray-400 text-xs">Computation time on Intel Xeon 3rd Gen Scalable cpu: cached</div>
			</div>
			<div class="space-y-3.5 pt-4">
				<div class="flex items-start justify-between font-mono text-xs leading-none animate__animated animate__fadeIn transition duration-200 ease-in-out false"
					style="animation-delay: 0s;">
					<div class="flex-1">
						<div class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600"
							style="width: 80%;"></div> <span class="leading-snug">eng_Latn</span>
					</div> <span class="pl-2">1.000</span>
				</div>
				<div class="flex items-start justify-between font-mono text-xs leading-none animate__animated animate__fadeIn transition duration-200 ease-in-out false"
					style="animation-delay: 0.04s;">
					<div class="flex-1">
						<div class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600"
							style="width: 1%;"></div> <span class="leading-snug">kor_Hang</span>
					</div> <span class="pl-2">0.000</span>
				</div>
				<div class="flex items-start justify-between font-mono text-xs leading-none animate__animated animate__fadeIn transition duration-200 ease-in-out false"
					style="animation-delay: 0.08s;">
					<div class="flex-1">
						<div class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600"
							style="width: 1%;"></div> <span class="leading-snug">nno_Latn</span>
					</div> <span class="pl-2">0.000</span>
				</div>
				<div class="flex items-start justify-between font-mono text-xs leading-none animate__animated animate__fadeIn transition duration-200 ease-in-out false"
					style="animation-delay: 0.12s;">
					<div class="flex-1">
						<div class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600"
							style="width: 1%;"></div> <span class="leading-snug">zul_Latn</span>
					</div> <span class="pl-2">0.000</span>
				</div>
				<div class="flex items-start justify-between font-mono text-xs leading-none animate__animated animate__fadeIn transition duration-200 ease-in-out false"
					style="animation-delay: 0.16s;">
					<div class="flex-1">
						<div class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600"
							style="width: 1%;"></div> <span class="leading-snug">umb_Latn</span>
					</div> <span class="pl-2">0.000</span>
				</div>
			</div>
			<div class="mt-auto pt-4 flex items-center text-xs text-gray-500"><button class="flex items-center "><svg
						class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
						aria-hidden="true" focusable="false" role="img" width="1em" height="1em"
						preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);">
						<path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path>
						<path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path>
						<path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path>
					</svg>
					JSON Output</button> <button class="flex items-center ml-auto"><svg class="mr-1"
						xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true"
						focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet"
						viewBox="0 0 32 32">
						<path d="M22 16h2V8h-8v2h6v6z" fill="currentColor"></path>
						<path d="M8 24h8v-2h-6v-6H8v8z" fill="currentColor"></path>
						<path
							d="M26 28H6a2.002 2.002 0 0 1-2-2V6a2.002 2.002 0 0 1 2-2h20a2.002 2.002 0 0 1 2 2v20a2.002 2.002 0 0 1-2 2zM6 6v20h20.001L26 6z"
							fill="currentColor"></path>
					</svg>
					Maximize</button></div>
		</div>
	</div>
</div>


### How to use

Here is how to load and use a pre-trained vectors:

```python
>>> import fasttext
>>> from huggingface_hub import hf_hub_download

>>> model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
>>> model = fasttext.load_model(model_path)
>>> model.words

['the', 'of', 'and', 'to', 'in', 'a', 'that', 'is', ...]

>>> len(model.words)

145940

>>> model['bread']

array([ 4.89417791e-01,  1.60882145e-01, -2.25947708e-01, -2.94273376e-01,
       -1.04577184e-01,  1.17962055e-01,  1.34821936e-01, -2.41778508e-01, ...])
```

Here is how to use this model to query nearest neighbors of an English word vector:

```python
>>> import fasttext
>>> from huggingface_hub import hf_hub_download

>>> model_path = hf_hub_download(repo_id="facebook/fasttext-en-nearest-neighbors", filename="model.bin")
>>> model = fasttext.load_model(model_path)
>>> model.get_nearest_neighbors("bread", k=5)

[(0.5641006231307983, 'butter'), 
 (0.48875734210014343, 'loaf'), 
 (0.4491206705570221, 'eat'), 
 (0.42444291710853577, 'food'), 
 (0.4229326844215393, 'cheese')]
```

Here is how to use this model to detect the language of a given text:

```python
>>> import fasttext
>>> from huggingface_hub import hf_hub_download

>>> model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
>>> model = fasttext.load_model(model_path)
>>> model.predict("Hello, world!")

(('__label__eng_Latn',), array([0.81148803]))

>>> model.predict("Hello, world!", k=5)

(('__label__eng_Latn', '__label__vie_Latn', '__label__nld_Latn', '__label__pol_Latn', '__label__deu_Latn'), 
 array([0.61224753, 0.21323682, 0.09696738, 0.01359863, 0.01319415]))
```

## Would you like to integrate your library to the Hub?

This integration is possible thanks to our collaboration with [Meta AI](https://ai.facebook.com/) and the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library, which enables all our widgets and the API for all our supported libraries. If you would like to integrate your library to the Hub, we have a [guide](https://huggingface.co/docs/hub/models-adding-libraries) for you!
