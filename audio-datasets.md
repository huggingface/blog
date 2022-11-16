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

## Easy to Load, Easy to Process

Loading a dataset with 🤗 Datasets is just half of the fun. We can now use the suite of tools available to efficiently 
pre-process our data ready for model inference or training. In this Section, we'll perform three stages of data 
pre-processing:
1. [Resampling the Audio Data](#resampling-the-audio-data)
2. [Pre-Processing Function](#pre-processing-function)
3. [Filtering Function](#filtering-function)

### 1. Resampling the Audio Data
The `load_dataset` function prepares audio samples with the sampling rate that they were published with. This is not 
always the sampling rate expected by our model. In this case, we need to _resample_ the audio to the correct sampling 
rate.

We can set the audio inputs to the correct sampling rate using 🤗 Dataset's [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column) 
method. This operation does not change the audio in-place, but rather signals to `datasets` to resample audio samples 
_on the fly_ the first time that they are loaded:

```python
from datasets import Audio

gigaspeech = gigaspeech.cast_column("audio", Audio(sampling_rate=8000))
```

Re-loading the first audio sample in the gigaspeech dataset will resample it to the desired sampling rate:

```python
print(gigaspeech["train"][0])
```
**Print Output:**
```python
{'segment_id': 'YOU0000000315_S0000660',
 'speaker': 'N/A', 
 'text': "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>", 
 'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/7f8541f130925e9b2af7d37256f2f61f9d6ff21bf4a94f7c1a3803ec648d7d79/xs_chunks_0000/YOU0000000315_S0000660.wav', 
           'array': array([ 0.00046338,  0.00034808, -0.00086153, ...,  0.00099299,
        0.00083484,  0.00080221], dtype=float32), 
           'sampling_rate': 8000
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
We can see that the sampling rate has been downsampled to 8kHz. The array values are also different, as we've now only 
got approximately one amplitude value for every two that we had before.

### 2. Pre-Processing Function
One of the hardest aspects of working with audio datasets is preparing the data into the right format for the model. 
Using 🤗 Dataset's [`map`](https://huggingface.co/docs/datasets/v2.6.1/en/process#map) method, we can write a function 
to pre-process a single sample of the dataset, and then handily apply it to every sample without any code changes!

Let's first load a processor object from 🤗 Transformers. This processor will take care of pre-processing the audio 
inputs and tokenising the text. The `AutoProcessor` class is used to load a pre-trained processor 
from a given model checkpoint. In the example, we load the processor from OpenAI's [Whisper small.en](https://huggingface.co/openai/whisper-small.en) 
checkpoint, but you can change this to any model identifier on the Hugging Face Hub:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
```

Great! Now we write a function that takes a single training sample and passes it through the `processor` to pre-process 
ready for our model. We'll also compute the input length of each audio sample, information that we'll need for our next 
data preparation step:

```python
def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch
```

We can apply the data preparation function to all of our training examples using dataset's `map` method:

```python
gigaspeech = gigaspeech.map(prepare_dataset, remove_columns=next(iter(gigaspeech.values())).column_names)
```

Here, we remove the columns were defined when we loaded the dataset (e.g. `file`, `chapter_id`, `speaker_id`, etc.), 
columns that we do not require for the speech recognition task.

### 3. Filtering Function

Prior to training, we might have a heuristic by which we want to filter our training data. For instance, we might want 
to filter any audio samples longer than 30s. We can do this in much the same way that we prepared our dataset 
for our model in the previous step. We start by writing a function that returns a boolean, indicating which samples are 
shorter than 30s (True) and which are longer (False):

```python
MAX_DURATION_IN_SECONDS = 30

def is_audio_length_in_range(input_length):
    return 0 < input_length < MAX_DURATION_IN_SECONDS
```

We can apply this filtering function to all of our training examples using 🤗 Dataset's [`filter`](https://huggingface.co/docs/datasets/process#select-and-filter) 
method, keeping all samples that are shorter than 30s (True) and discarding those that are longer (False):

```python
gigaspeech["train"] = gigaspeech["train"].filter(is_audio_length_in_range, input_columns=["input_length"])
```

## The Hub

## A Tour of Audio Datasets on The Hub

