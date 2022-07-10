---
title: 'Building a Playlist Generator with Sentence Transformers'
thumbnail: /blog/assets/87_playlist_generator_st/thumbnail.png
---

<h1>
    Building a Playlist Generator with Sentence Transformers
</h1>

<div class="blog-metadata">
    <small>Published July 13, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/playlist-generator-st.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/nimaboscarino"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1647889744246-61e6a54836fa261c76dc3760.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>nimaboscarino</code>
            <span class="fullname">Nima Boscarino</span>
        </div>
    </a>
</div>

A short while ago I published a [playlist generator](https://huggingface.co/spaces/NimaBoscarino/playlist-generator) that I’d built using **[Sentence Transformers](http://sbert.net)** and [Gradio](https://gradio.app), and I followed that up with a [reflection on how I try to use my projects as effective learning experiences](https://huggingface.co/blog/your-first-ml-project). But how did I actually *build* the playlist generator? In this post we’ll break down that project and look at two technical details: how the embeddings were generated, and how the *multi-step* Gradio demo was built.

As we’ve explored in [previous posts on the Hugging Face blog](https://huggingface.co/blog/getting-started-with-embeddings), Sentence Transformers (ST) is a library that gives us tools to generate sentence embeddings, which have a variety of uses. Since I had access to a dataset of song lyrics, I decided to leverage ST’s **[semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html)** functionality to generate playlists from a given text prompt. Specifically, the goal was to create an embedding from the prompt, use that embedding for a semantic search across a set of *pre-generated lyrics embeddings* to generate a relevant set of songs. This would all be wrapped up in a Gradio app using the new **[Blocks API](https://gradio.app/introduction_to_blocks/)**, hosted on Hugging Face Spaces.

We’ll be looking at a slightly advanced use of Gradio, so if you’re a beginner to the library I recommend exploring the [Gradio guides](https://gradio.app/guides/) and the [Introduction to Blocks](https://gradio.app/introduction_to_blocks/) before tackling the Gradio-specific parts of this post. Also, note that while I won’t be releasing the lyrics dataset, the **[lyrics embeddings are available on the Hugging Face Hub](https://huggingface.co/datasets/NimaBoscarino/playlist-generator)** for you to play around with. Let’s jump in! 🪂

## Sentence Transformers: Embeddings and Semantic Search

Embeddings are **key** in Sentence Transformers! We’ve learned about **[what embeddings are and how we generate them in a previous article](https://huggingface.co/blog/getting-started-with-embeddings)**, and I recommend checking that out before continuing with this post.

Sentence Transformers offers a large collection of pre-trained embedding models! It even includes tutorials for fine-tuning those models with our own training data, but for many use-cases (such semantic search over a corpus of song lyrics) the pre-trained models will perform excellently right out of the box. With so many embedding models available, though, how do we know which one to use?

[The ST documentation highlights many of the choices](https://www.sbert.net/docs/pretrained_models.html), along with their evaluation metrics and some descriptions of their intended use-cases. The **[MS MARCO models](https://www.sbert.net/docs/pretrained-models/msmarco-v5.html)** are trained on Bing search engine queries, but since they also perform well on other domains I decided any one of these could be a good choice for this project. All we need for the playlist generator is to find songs that have some semantic similarity, and since I don’t really care about hitting a particular performance metric I arbitrarily chose [sentence-transformers/msmarco-MiniLM-L-6-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3).

Each model in ST has a configurable [input sequence length](https://www.sbert.net/examples/applications/computing-embeddings/README.html#input-sequence-length) (up to a maximum), after which your inputs will be truncated. The model I chose had a max sequence length of 512 word pieces, which, as I found out, is often not enough to embed entire songs. Luckily, there’s an easy way for us to split lyrics into smaller chunks that the model can digest – verses! Once we’ve chunked our songs into verses and embedded each verse, we’ll find that the search works much better.

** TODO: Diagram with Figma **

To actually generate the embeddings, we can call the the `.encode()`  method on our Sentence Transformers model and pass it our list of strings. Then we can save our embeddings however we like – in this case I opted to pickle them. To be able to share our embeddings with others, we can even **[upload the Pickle file to a Hugging Face dataset](https://huggingface.co/blog/getting-started-with-embeddings#2-host-embeddings-for-free-on-the-hugging-face-hub)**.

```python
from sentence_transformers import SentenceTransformer
import pickle

embedder = SentenceTransformer('msmarco-MiniLM-L-6-v3')
verses = [...] # Load up your strings in a list
corpus_embeddings = embedder.encode(verses, show_progress_bar=True)

with open('verse-embeddings.pkl', "wb") as fOut:
        pickle.dump(corpus_embeddings, fOut)
```

The last thing we need to do now is actually use the embeddings for semantic search! The following code loads the embeddings, generates a new embedding for a given string, and runs a semantic search over the lyrics embeddings to find the closest hits. To make it easier to work with the results, I also like to put them into a Pandas DataFrame.

```python
from sentence_transformers import util
import pandas as pd

prompt_embedding = embedder.encode(prompt, convert_to_tensor=True)
hits = util.semantic_search(prompt_embedding, corpus_embeddings, top_k=20)
hits = pd.DataFrame(hits[0], columns=['corpus_id', 'score'])
# Note that "corpus_id" is the index of the verse for that embedding
# You can use the "corpus_id" to look up the original song
```

Since we’re searching for any verse that matches the text prompt, there’s a good chance that the semantic search will find multiple verses from the same song. One simple way to avoid duplicates is by increasing the number of verse embeddings that `util.semantic_search` fetches for us with the `top_k` parameter, and then we can drop duplicate songs as we map each verse embedding to its respective song. Experimentally, I found that when I set `top_k=20` I rarely get duplicates!

## Making a Multi-Step Gradio App

For the demo, I wanted users to enter a text prompt (or choose from some examples), and conduct a semantic search to find the top 9 most relevant songs. Then, users should be able to select from the resulting songs to be able to see the lyrics, which might give them some insight into why the particular songs were chosen. Here’s how we can do that!

At the top of the Gradio demo](https://huggingface.co/spaces/NimaBoscarino/playlist-generator/blob/main/app.py) we load the embeddings, mappings, and lyrics from Hugging Face datasets when the app starts up.

```python
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download
import os
import pickle
import pandas as pd

corpus_embeddings = pickle.load(open(hf_hub_download("NimaBoscarino/playlist-generator", repo_type="dataset", filename="verse-embeddings.pkl"), "rb"))
songs = pd.read_csv(hf_hub_download("NimaBoscarino/playlist-generator", repo_type="dataset", filename="songs_new.csv"))
verses = pd.read_csv(hf_hub_download("NimaBoscarino/playlist-generator", repo_type="dataset", filename="verses.csv"))

# I'm loading the lyrics from my private dataset, with my own API token
auth_token = os.environ.get("TOKEN_FROM_SECRET") 
lyrics = pd.read_csv(hf_hub_download("NimaBoscarino/playlist-generator-private", repo_type="dataset", filename="lyrics_new.csv", use_auth_token=auth_token))
```

The Gradio Blocks API lets you build *multi-step* interfaces, which means that you’re free to create quite complex sequences for your demos. We’ll take a look at some example code snippets here, but [check out the project code to see it all in action](https://huggingface.co/spaces/NimaBoscarino/playlist-generator/blob/main/app.py). For this project, we want users to choose a text prompt and then, after the semantic search is complete, users should have the ability to choose a song from the results to inspect the lyrics. With Gradio, this can be built iteratively by starting off with defining the initial input components and then registering a `click` event on the button. There’s also a Radio component, which will get updated to show the names of the songs for the playlist.

```python
song_prompt = gr.TextArea(
    value="Running wild and free",
    placeholder="Enter a song prompt, or choose an example"
)

fetch_songs = gr.Button(value="Generate Your Playlist!")

song_option = gr.Radio(interactive=True)

fetch_songs.click(
    fn=generate_playlist,
    inputs=[song_prompt],
    outputs=[song_option],
)
```

This way, when the button gets clicked, Gradio grabs the current value of the TextArea and passes it to a function, shown below:

```python
def generate_playlist(prompt):
    prompt_embedding = embedder.encode(prompt, convert_to_tensor=True)
    hits = util.semantic_search(prompt_embedding, corpus_embeddings, top_k=20)
		hits = pd.DataFrame(hits[0], columns=['corpus_id', 'score'])
		# ... code to map from the verse IDs to the song names
		song_names = ... # e.g. ["Thank U, Next", "Freebird", "La Cucaracha"]
    return (
        gr.Radio.update(label="Songs", interactive=True, choices=song_names)
    )
```

In that function, we use the text prompt to conduct the semantic search. As we see above, to push updates to the Gradio components in the app, our function just needs to return components created with the `.update()` method. Since we connected the `song_option` Radio component to `fetch_songs.click` with its `output` parameter, `generate_playlist` can control the choices for our Radio component!

We can even do something similar to the Radio component, in order to let users choose which song lyrics to view. [Visit the code on Hugging Face Spaces to see it in detail!](https://huggingface.co/spaces/NimaBoscarino/playlist-generator/blob/main/app.py)

## Some Thoughts

As we can see, Sentence Transformers and Gradio are great choices for this kind of project! ST has the utility functions that we need for quickly generating embeddings, as well as for running semantic search with minimal code. Having access to a large collection of pre-trained models is also extremely helpful, since we don’t need to create and train our own models for this kind of stuff. Building our demo in Gradio means we only have to focus on coding in Python, and [deploying Gradio projects to Hugging Face Spaces is also super simple](https://huggingface.co/docs/hub/spaces-sdks-gradio)!

There’s a ton of other stuff I wish I’d had the time to build into this project, such as these ideas that I might explore in the future:

- Integrating with Spotify to automatically generate a playlist, and maybe even using Spotify’s embedded player to let users immediately listen to the songs.
- Using the **[HighlightedText** Gradio component](https://gradio.app/docs/#highlightedtext) to identify the specific verse that was found by the semantic search.
- Creating some visualizations of the embedding space, like in [this Space by Radamés Ajna](https://huggingface.co/spaces/radames/sentence-embeddings-visualization).

While the song *lyrics* aren’t being released, I’ve **[published the verse embeddings along with the mappings to each song](https://huggingface.co/datasets/NimaBoscarino/playlist-generator)**, so you’re free to play around and get creative!

Remember to [drop by the Discord](https://huggingface.co/join/discord) to ask questions and share your work! I’m excited to see what you end up doing with Sentence Transformers embeddings 🤗

## Extra Resources

- [Getting Started With Embeddings](https://huggingface.co/blog/getting-started-with-embeddings) by Omar Espejel
    - [Or as a Twitter thread](https://twitter.com/osanseviero/status/1540993407883042816?s=20&t=4gskgxZx6yYKknNB7iD7Aw) by Omar Sanseviero
- [Hugging Face + Sentence Transformers docs](https://www.sbert.net/docs/hugging_face.html)
- [Gradio Blocks party](https://huggingface.co/Gradio-Blocks) - View some amazing community projects showcasing Gradio Blocks!