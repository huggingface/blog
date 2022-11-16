---
title: "A Complete Guide to Audio Datasets" 
thumbnail: /blog/assets/116_audio_datasets/thumbnail.jpg
---

# A Complete Guide to Audio Datasets

<div class="blog-metadata">
    <small>Published 21 November, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/audio-datasets.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/sanchit-gandhi"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1653243468328-61f91cf54a8e5a275b2b3e7c.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>sanchit-gandhi</code>
            <span class="fullname">Sanchit Gandhi</span>
        </div>
    </a>
</div>

It's well known that 🤗 Datasets provides the easiest access to numerous NLP datasets. What's less known is that the 
same is true for audio datasets: all of the most popular audio datasets can be downloaded and prepared with the 
same ease as their NLP counterparts.

🤗 Datasets is open-source library for downloading and preparing datasets of all domains. Its minimalistic API 
allows users to download and prepare datasets in just one line of Python code, with a suite of functions that 
enable for efficient pre-processing. The number of datasets available is unparalleled: all of the most popular 
audio datasets are available through 🤗 Datasets.

Not only this, but 🤗 Datasets comes prepared with multiple audio-specific features that make working 
with audio datasets easy for both researchers and practitioners alike. In this blog, we'll demonstrate how 🤗 Datasets 
is the number one place for downloading and preparing audio datasets. Carry on reading to find out how to load and 
prepare the most popular audio datasets in just one line of Python code!

## Load an Audio Dataset

One of the key defining features of 🤗 Datasets is the ability to download and prepare a dataset in just one line of 
Python code. This is made possible through the [`load_dataset`](https://huggingface.co/docs/datasets/loading#load) 
function. Conventionally, we'd have to download the raw data, extract it from its compressed format, and prepare individal 
samples and splits. Using `load_dataset`, all of the heavy lifting of loading a dataset is done under the hood.

including 
downloading the raw data, extracting it from compressed files, and finally preparing samples and splits.

Let's take the example of loading the [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) dataset from 
Speech Colab. GigaSpeech is a popular speech recognition datasets for benchmarking academic speech systems. It is one 
of many speech recognition datasets available through 🤗 Datasets. To load the GigaSpeech dataset, we simply have to 
specify the dataset's identifier to the `load_dataset` function. GigaSpeech comes in an array of different split sizes, 
ranging from `xs` (10 hours) to `xl` (10,000 hours). For the purpose of this tutorial, we'll load the smallest of these 
splits:

```python
from datasets import load_dataset

gigaspeech = load_dataset("speechcolab/gigaspeech", "xs")

print(gigaspeech)
```

**Print Output:**
```python
DatasetDict({
    train: Dataset({
        features: ['segment_id', 'speaker', 'text', 'audio', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path'],
        num_rows: 9389
    })
    validation: Dataset({
        features: ['segment_id', 'speaker', 'text', 'audio', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path'],
        num_rows: 6750
    })
    test: Dataset({
        features: ['segment_id', 'speaker', 'text', 'audio', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path'],
        num_rows: 25619
    })
})
```

And just like that we have the GigaSpeech dataset ready! There simply is no easier way of loading an audio dataset! We 
can see that we have the training, validation and test splits pre-partitioned, with the corresponding information for 
each.

The object `gigaspeech` returned by `load_dataset` is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict). 
We can treat it in much the same way as an ordinary Python dictionary. To get the train split, we simply have to pass 
the corresponding key to the `DatasetDict` object:

```python
print(gigaspeech["train"])
```

**Print Output:**
```python
Dataset({
    features: ['segment_id', 'speaker', 'text', 'audio', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path'],
    num_rows: 9389
})
```

We can go one level deeper and get the first item of the training split. Again, this is possible through simple Python 
indexing:
```python
print(gigaspeech["train"][0])
```

**Print Output:**
```python
{'segment_id': 'YOU0000000315_S0000660',
 'speaker': 'N/A', 
 'text': "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>", 
 'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/7f8541f130925e9b2af7d37256f2f61f9d6ff21bf4a94f7c1a3803ec648d7d79/xs_chunks_0000/YOU0000000315_S0000660.wav', 
           'array': array([0.0005188 , 0.00085449, 0.00012207, ..., 0.00125122, 0.00076294,
       0.00036621], dtype=float32), 
           'sampling_rate': 16000
           }, 
 'begin_time': 2941.889892578125, 
 'end_time': 2945.070068359375, 
 'audio_id': 'YOU0000000315', 
 'title': 'Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43', 
 'url': 'https://www.youtube.com/watch?v=zr2n1fLVasU', 
 'source': 2, 
 'category': 24, 
 'original_full_path': 'audio/youtube/P0004/YOU0000000315.opus'
 }
```

We can see that there are a number of features returned by the `DatasetDict`, including `segment_id`, `speaker`, `text`, 
`audio` and more. For speech recognition, we'll be concerned with the `text` and `audio` columns. 

