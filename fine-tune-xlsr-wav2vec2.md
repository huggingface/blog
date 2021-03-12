---
title: "Fine-Tune XLSR-Wav2Vec2 for low-resource ASR with 🤗 Transformers"
thumbnail: /blog/assets/15_fine_tune_wav2vec2/wav2vec2.png
---

<h1>
    Fine-Tune XLSR-Wav2Vec2 for low-resource ASR with 🤗 Transformers
</h1>

<div class="blog-metadata">
    <small>Published March 12, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/fine-tune-wav2vec2-english">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/patrickvonplaten">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1584435275418-5dfcb1aada6d0311fd3d5448.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>patrickvonplaten</code>
            <span class="fullname">Patrick von Platen</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_🤗_Transformers.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Wav2Vec2 is a pretrained model for Automatic Speech Recognition (ASR) and was released in [September 2020](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) by Alexei Baevski, Michael Auli, and Alex Conneau.  Soon after the superior performance of Wav2Vec2 was demonstrated on the English ASR dataset LibriSpeech, *Facebook AI* presented XLSR-Wav2Vec2 (click [here](https://arxiv.org/abs/2006.13979)). XLSR stands for *cross-lingual  speech representations* and refers to XLSR-Wav2Vec2`s ability to learn speech representations that are useful across multiple languages.

Similar to Wav2Vec2, XLSR-Wav2Vec2 learns powerful speech representations from hundreds of thousands of hours of speech in more than 50 languages of unlabeled speech. Similar, to [BERT's masked language modeling](http://jalammar.github.io/illustrated-bert/), the model learns contextualized speech representations by randomly masking feature vectors before passing them to a transformer network.

![wav2vec2\_structure](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/xlsr_wav2vec2.png)

The authors show for the first time that massively pretraining an ASR model on cross-lingual unlabeled speech data, followed by language-specific fine-tuning on very little labeled data achieves state-of-the-art results. See Table 1-5 of the official [paper](https://arxiv.org/pdf/2006.13979.pdf).

In this notebook, we will give an in-detail explanation of how XLSR-Wav2Vec2's pretrained checkpoint can be fine-tuned on a low-resource ASR dataset of any language. Note that in this notebook, we will fine-tune XLSR-Wav2Vec2 without making use of a language model. It is much simpler and more efficient to use XLSR-Wav2Vec2 without a language model, but better results can be achieved by including a language model. 

For demonstration purposes, we fine-tune the [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on the low resource Turkish ASR dataset of [Common Voice](https://huggingface.co/datasets/common_voice) that contains just ~6h of validated training data.

XLSR-Wav2Vec2 is fine-tuned using Connectionist Temporal Classification (CTC), which is an algorithm that is used to train neural networks for sequence-to-sequence problems and mainly in Automatic Speech Recognition and handwriting recognition. 

I highly recommend reading the blog post [Sequence Modeling with CTC (2017)](https://distill.pub/2017/ctc/) very well-written blog post by Awni Hannun.

Before we start, let's install both `datasets` and `transformers` from master. Also, we need the `torchaudio` and `librosa` package to load audio files and the `jiwer` to evaluate our fine-tuned model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric ${}^1$.

```bash
!pip install git+https://github.com/huggingface/datasets.git
!pip install git+https://github.com/huggingface/transformers.git
!pip install torchaudio
!pip install librosa
!pip install jiwer
```

------------------------------------------------------------------------

\\({}^1\\) In the [paper](https://arxiv.org/pdf/2006.13979.pdf), the model was evaluated using the phoneme error rate (PER), 
but by far the most common metric in ASR is the word error rate (WER). 
To keep this notebook as general as possible we decided to evaluate the model using WER.

Prepare Data, Tokenizer, Feature Extractor
------------------------------------------

ASR models transcribe speech to text, which means that we both need a feature extractor that processes the speech signal to the model's input format, *e.g.* a feature vector, and a tokenizer that processes the model's output format to text. 

In 🤗 Transformers, the XLSR-Wav2Vec2 model is thus accompanied by both a tokenizer, called [Wav2Vec2CTCTokenizer](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#wav2vec2ctctokenizer), and a feature extractor, called [Wav2Vec2FeatureExtractor](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#wav2vec2featureextractor).

Let's start by creating the tokenizer responsible for decoding the model's predictions.

### Create XLSR-Wav2Vec2CTCTokenizer

The pretrained XLSR-Wav2Vec2 checkpoint maps the speech signal to a sequence of context representations as illustrated in the figure above. A fine-tuned XLSR-Wav2Vec2 checkpoint needs to map this sequence of context representations to its corresponding transcription so that a linear layer has to be added on top of the transformer block (shown in yellow). This linear layer is used to classifies each context representation to a token class analogous how, *e.g.*, after pretraining a linear layer is added on top of BERT's embeddings for further classification - *cf.* with *"BERT"* section of this [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder).

The output size of this layer corresponds to the number of tokens in the vocabulary, which does **not** depend on XLSR-Wav2Vec2's pretraining task, but only on the labeled dataset used for fine-tuning. So in the first step, we will take a look at Timit and define a vocabulary based on the dataset's transcriptions.

First, let's go to [Common Voice](https://commonvoice.mozilla.org/en) and pick your favorite language you would like to fine-tune XLSR-Wav2Vec2 on. For this notebook, we will use Turkish. 

**Note**: Most likely, the common voice link has expired. In this case, just go to [Common Voice's dataset website](https://commonvoice.mozilla.org/en/datasets), select your language, *e.g.* `Turkish`, enter your email address to get the "*Download*" button, click right, and click `Copy link address` to fill it in the cell below.

```bash
!wget "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-6.1-2020-12-11/tr.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3G23VQA5S%2F20210312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210312T064856Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEIj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDKZIPKux%2BmNHRNIsviKSBNDlKP4z8BJGh7lr6j6zkLhvCmkVA8f7Ot4KR52ZpaRK1eP0KRvKdrAvWdk4F0%2FWiPb5lRkDxtfA5gQE6ntJytUzRQfriVpYDHqSCGM7YEaG1ng1XK47dzO0%2FfcwpmyCfpRnZxw1gfs9oH%2BiFub530LRxeuHjeSDrbYT0hLPSXHQX5QRVVhyNLpqd%2F%2Bgqee5JzpCF5AyVS7%2Fz8YTwnz75IWD%2BoPwr%2FFhW0wiZoj4LP%2F6dVi0GhDSyh9w6n%2FthS%2FkSod1xYDUPpN0FFU1ZFYsaJjNr9VYdBhM%2FRUv0y2mXgSuMCviJrE7cv23wu7t4wTT3ELkAKDvMlGC3IgBqd1okQUp2AQRlNKsor38lcqBNZCJfg%2FdF%2Fve7T5NBi3FG0e7TpcnNyaVhL7efFaI2BuuQhzu%2FX7HR3Dy4P7Q5MBz6FFQiVPWq%2B9eOJY8FSP%2Fc%2B7XNIiQPPjwYem0nhM%2FPAfRBkysc9vLSOJU%2FlWlzlPVc94EQMjlYwj3aBjswi5t6A42%2FNxAvNvJrFHOL0Hk1xlVUzRSVNo66NTD5%2BvHcM6UStOjiCYruFCLHk3%2FuyKmLEr7dieX%2BMblLNBQ%2F1JWz1E4hTAc7tXnCkhOknJQ0d%2BfsbPT%2FoWW562oUWLEdrC38I5T7BbuyC0v9X2vzhMA7jVh9pJdl4%2Bk82yXowTwT2AFBYlBv6CpuWrtrnFtYOrgSGboYCG%2FKPWLrIIGMirfCq6%2F%2BD0PuXf%2FQiEZXz8lgNBNuqTVH7enGlPXfW91sRoGH%2FV15J0XwYM%3D&X-Amz-Signature=40b9453607def83d314725a8f9986ee17f50db185999e14a36817bff03f5a6a1&X-Amz-SignedHeaders=host"
```

Cool, let's copy the downloaded file name from the output of the cell above and unzip it.

```bash
!tar -xvzf "tr.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQ3GQRTO3G23VQA5S%2F20210312%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210312T064856Z&X-Amz-Expires=43200&X-Amz-Security-Token=FwoGZXIvYXdzEIj%2F%2F%2F%2F%2F%2F%2F%2F%2F%"
from datasets import load_dataset, load_metric
```

The data is now saved under ./cv-corpus-6.1-2020-12-11. We can leverage datasets now to preprocess the dataset.

Let's start by loading the dataset and taking a look at its structure. [Common Voice](https://huggingface.co/datasets/common_voice) has many different splits including `invalidated`, which refers to data that was not rated as "clean enough" to be considered useful. Because the Turkish dataset is so small, we will merge both the validation and training data into a training dataset and simply use the test data for validation.

```python
from datasets import load_dataset, load_metric

common_voice_train = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11", split="train+validation")
common_voice_test = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11", split="test")
```

Many ASR datasets only provide the target text, `'sentence'` for each audio file `'path'`. Common Voice actually provides much more information about each audio file, such as the `'accent'`, etc. However, we want to keep the notebook as general as possible, so that we will only consider the transcribed text for fine-tuning.

```python
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
```

Let's write a short function to display some random samples of the dataset and run it a couple of times to get a feeling for the transcriptions.

```python
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(common_voice_train.remove_columns(["path"]))
```

**Print Output:**

| Idx |  Sentence |
|----------|:-------------:|
|	1  | Jonuz, kısa süreli görevi kabul eden tek adaydı. | 
|	2  | Biz umudumuzu bu mücadeleden almaktayız. | 
|	3  | Sergide beş Hırvat yeniliği sergilendi. | 
|	4  | Herşey adıyla bilinmeli. | 
|	5  | Kuruluş özelleştirmeye hazır. | 
|	6  | Yerleşim yerlerinin manzarası harika. | 
|	7  | Olayların failleri bulunamadı. | 
|	8  | Fakat bu çabalar boşa çıktı. | 
|	9  | Projenin değeri iki virgül yetmiş yedi milyon avro. | 
| 10 | Büyük yeniden yapım projesi dört aşamaya bölündü. |

Alright! The transcriptions look fairly clean. Having translated the transcribed sentences (I'm sadly not a native speaker in Turkish), it seems that the language corresponds more to written text than noisy dialogue. This makes sense taking into account that [Common Voice](https://huggingface.co/datasets/common_voice) is a crowd-sourced read speech corpus.

We can see that the transcriptions contain some special characters, such as `,.?!;:`. Without a language model, it is much harder to classify speech chunks to such special characters because they don't really correspond to a characteristic sound unit. *E.g.*, the letter `"s"` has a more or less clear sound, whereas the special character `"."` does not.
Also in order to understand the meaning of a speech signal, it is usually not necessary to include special characters in the transcription.

In addition, we normalize the text to only have lower case letters and append a word separator token at the end.

```python
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

common_voice_train = common_voice_train.map(remove_special_characters, remove_columns=["sentence"])
common_voice_test = common_voice_test.map(remove_special_characters, remove_columns=["sentence"])
```

Let's take a look at the preprocessed transcriptions.

```python
show_random_elements(common_voice_train.remove_columns(["path"]))
```

**Print Output:**

| Idx |  Transcription     |
|----------|:-------------:|
| 1   | birisi beyazlar için dediler | 
| 2   | maktouf'un cezası haziran ayında sona erdi | 
| 3   | orijinalin aksine kıyafetler çıkarılmadı | 
| 4   | bunların toplam değeri yüz milyon avroyu buluyor | 
| 5   | masada en az iki seçenek bulunuyor | 
| 6   | bu hiç de haksız bir heveslilik değil | 
| 7   | bu durum bin dokuz yüz doksanlarda ülkenin bölünmesiyle değişti | 
| 8   | söz konusu süre altı ay | 
| 9   | ancak bedel çok daha yüksek olabilir | 
| 10  | başkent fira bir tepenin üzerinde yer alıyor | 

Good! This looks better. We have removed most special characters from transcriptions and normalized them to lower-case only.

In CTC, it is common to classify speech chunks into letters, so we will do the same here. 
Let's extract all distinct letters of the training and test data and build our vocabulary from this set of letters.

We write a mapping function that concatenates all transcriptions into one long transcription and then transforms the string into a set of chars. 
It is important to pass the argument `batched=True` to the `map(...)` function so that the mapping function has access to all transcriptions at once.

```python
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
```

Now, we create the union of all distinct letters in the training dataset and test dataset and convert the resulting list into an enumerated dictionary.

```python
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)
```

**Print Output:**

```bash
{
 ' ': 33,
 "'": 23,
 'a': 10,
 'b': 4,
 'c': 8,
 'd': 15,
 'e': 24,
 'f': 31,
 'g': 26,
 'h': 16,
 'i': 29,
 'j': 0,
 'k': 11,
 'l': 27,
 'm': 22,
 'n': 9,
 'o': 12,
 'p': 36,
 'q': 19,
 'r': 7,
 's': 28,
 't': 34,
 'u': 13,
 'v': 6,
 'w': 17,
 'x': 25,
 'y': 3,
 'z': 21,
 'â': 14,
 'ç': 20,
 'î': 5,
 'ö': 30,
 'ü': 18,
 'ğ': 1,
 'ı': 2,
 'ş': 35,
 '̇' : 32,
}
```

Cool, we see that all letters of the alphabet occur in the dataset (which is not really surprising) and we also extracted the special characters `" "` and `'`. Note that we did not exclude those special characters because: 

- The model has to learn to predict when a word is finished or else the model prediction would always be a sequence of chars which would make it impossible to separate words from each other.
- From the transcriptions above it seems that words that include an apostrophe, such as `maktouf'un` do exist in Turkish, so I decided to keep the apostrophe in the dataset. This might be a wrong assumption though.

One should always keep in mind that the data-preprocessing is a very important step before training your model. E.g., we don't want our model to differentiate between `a` and `A` just because we forgot to normalize the data. The difference between `a` and `A` does not depend on the "sound" of the letter at all, but more on grammatical rules - *e.g.* use a capitalized letter at the beginning of the sentence. So it is sensible to remove the difference between capitalized and non-capitalized letters so that the model has an easier time learning to transcribe speech. 

It is always advantageous to get help from a native speaker of the language you would like to transcribe to verify whether the assumptions you made are sensible, *e.g.* I should have made sure that keeping `'`, but removing other special characters is a sensible choice for Turkish.

To make it clearer that `" "` has its own token class, we give it a more visible character `|`. In addition, we also add an "unknown" token so that the model can later deal with characters not encountered in Timit's training set. 

```python
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
```

Finally, we also add a padding token that corresponds to CTC's "*blank token*". The "blank token" is a core component of the CTC algorithm. For more information, please take a look at the "Alignment" section [here](https://distill.pub/2017/ctc/).

```python
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))
```

**Print Output:**

```bash
    39
```

Cool, now our vocabulary is complete and consists of 39 tokens, which means that the linear layer that we will add on top of the pretrained XLSR-Wav2Vec2 checkpoint will have an output dimension of 39.

Let\'s now save the vocabulary as a json file.

```python
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
```

In a final step, we use the json file to instantiate an object of the
`Wav2Vec2CTCTokenizer` class.

```python
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
```
Next, we will create the feature extractor.

### Create XLSR-Wav2Vec2 Feature Extractor

Speech is a continuous signal and to be treated by computers, it first has to be discretized, which is usually called **sampling**. The sampling rate hereby plays an important role in that it defines how many data points of the speech signal are measured per second. Therefore, sampling with a higher sampling rate results in a better approximation of the *real* speech signal but also necessitates more values per second.

A pretrained checkpoint expects its input data to have been sampled more or less from the same distribution as the data it was trained on. The same speech signals sampled at two different rates have a very different distribution, *e.g.*, doubling the sampling rate results in data points being twice as long. Thus, 
before fine-tuning a pretrained checkpoint of an ASR model, it is crucial to verify that the sampling rate of the data that was used to pretrain the model matches the sampling rate of the dataset used to fine-tune the model.

XLSR-Wav2Vec2 was pretrained on the audio data of [Babel](https://huggingface.co/datasets/librispeech_asr), 
[Multilingual LibriSpeech (MLS)](https://ai.facebook.com/blog/a-new-open-data-set-for-multilingual-speech-research/), and [Common Voice](https://huggingface.co/datasets/common_voice). Most of those datasets were sampled at 16kHz, so that Common Voice, sampled at 48kHz, has to be downsampled to 16kHz for training. Therefore, we will have to downsample our fine-tuning data to 16kHz in the following.

A XLSR-Wav2Vec2 feature extractor object requires the following parameters to be instantiated:

- `feature_size`: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal ${}^2$.
- `sampling_rate`: The sampling rate at which the model is trained on.
- `padding_value`: For batched inference, shorter inputs need to be padded with a specific value
- `do_normalize`: Whether the input should be *zero-mean-unit-variance* normalized or not. Usually, speech models perform better when normalizing the input
- `return_attention_mask`: Whether the model should make use of an `attention_mask` for batched inference. In general, XLSR-Wav2Vec2 models should **always** make use of the `attention_mask`.

```python
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
```

Great, XLSR-Wav2Vec2's feature extraction pipeline is thereby fully defined!

To make the usage of XLSR-Wav2Vec2 as user-friendly as possible, the feature extractor and tokenizer are *wrapped* into a single `Wav2Vec2Processor` class so that one only needs a `model` and `processor` object.

```python
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

If one wants to re-use the just created processor and the fine-tuned
model of this notebook, one can mount his/her google drive to the
notebook and save all relevant files there. To do so, please uncomment
the following lines.

We will give the fine-tuned model the name `"wav2vec2-large-xlsr-turkish-demo"`.

```python
# from google.colab import drive
# drive.mount('/content/gdrive/')
```

```python
# processor.save_pretrained("/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo")
```

Next, we can prepare the dataset.

### Preprocess Data

So far, we have not looked at the actual values of the speech signal but just kept the path to its file in the dataset. `XLSR-Wav2Vec2` expects the audio file in the format of a 1-dimensional array, so in the first step, let's load all audio files into the dataset object.

Let's first check the serialization format of the downloaded audio files by looking at the first training sample.


```python
print(common_voice_train[0])
```

**Print Output:**

```bash
		{'path': '/content/cv-corpus-6.1-2020-12-11/tr/clips/common_voice_tr_21921195.mp3',
		 'text': 'pirin sözleri hâlâ yankılanıyor '}
```

Alright, the audio file is saved in the `.mp3` format. The `.mp3` format is usually not the easiest format to deal with. We found that the [`torchaudio`](https://pytorch.org/audio/stable/index.html) library works best for reading in `.mp3` data. 

An audio file usually stores both its values and the sampling rate with which the speech signal was digitalized. We want to store both in the dataset and write a `map(...)` function accordingly.

```python
import torchaudio

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)
```

Great, now we've successfully read in all the audio files, but since we know that Common Voice is sampled at 48kHz, we need to resample the audio files to 16kHz. 

Let's make use of the [`librosa`](https://github.com/librosa/librosa) library to downsample the data.

```python
import librosa
import numpy as np

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

common_voice_train = common_voice_train.map(resample, num_proc=4)
common_voice_test = common_voice_test.map(resample, num_proc=4)
```

This seemed to have worked! Let's listen to a couple of audio files to better understand the dataset and verify that the audio was correctly loaded. 


```python
import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(common_voice_train))

ipd.Audio(data=np.asarray(common_voice_train[rand_int]["speech"]), autoplay=True, rate=16000)
```

It can be heard, that the speakers change along with their speaking rate, accent, and background environment, etc. Overall, the recordings sound acceptably clear though, which is to be expected from a crowd-sourced read speech corpus.

Let's do a final check that the data is correctly prepared, by printing the shape of the speech input, its transcription, and the corresponding sampling rate.

```python
rand_int = random.randint(0, len(common_voice_train))

print("Target text:", common_voice_train[rand_int]["target_text"])
print("Input array shape:", np.asarray(common_voice_train[rand_int]["speech"]).shape)
print("Sampling rate:", common_voice_train[rand_int]["sampling_rate"])
```

**Print Output:**

```bash
	Target text: mali sorunlara rağmen bunu tekrar başardı 
	Input array shape: (116352,)
	Sampling rate: 16000
```

Finally, we can process the dataset to the format expected by the model for training. We will again make use of the `map(...)` function.

First, we check that the data samples have the same sampling rate of 16kHz.
Second, we extract the `input_values` from the loaded audio file. In our case, this includes only normalization, but for other speech models, this step could correspond to extracting, *e.g.* [Log-Mel features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). 
Third, we encode the transcriptions to label ids.

**Note**: This mapping function is a good example of how the `Wav2Vec2Processor` class should be used. In "normal" context, calling `processor(...)` is redirected to `Wav2Vec2FeatureExtractor`'s call method. When wrapping the processor into the `as_target_processor` context, however, the same method is redirected to `Wav2Vec2CTCTokenizer`'s call method.
For more information please check the [docs](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#transformers.Wav2Vec2Processor.__call__).


```python
def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)
```

Training
--------

The data is processed so that we are ready to start setting up the training pipeline. We will make use of 🤗's [Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) for which we essentially need to do the following:

- Define a data collator. In contrast to most NLP models, XLSR-Wav2Vec2 has a much larger input length than output length. *E.g.*, a sample of input length 50000 has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning XLSR-Wav2Vec2 requires a special padding data collator, which we will define below

- Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a `compute_metrics` function accordingly

- Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.

- Define the training configuration.

After having fine-tuned the model, we will correctly evaluate it on the test data and verify that it has indeed learned to correctly transcribe speech.


### Set-up Trainer

Let's start by defining the data collator. The code for the data collator was copied from [this example](https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81).

Without going into too many details, in contrast to the common data collators, this data collator treats the `input_values` and `labels` differently and thus applies to separate padding functions on them (again making use of XLSR-Wav2Vec2's context manager). This is necessary because in speech input and output are of different modalities meaning that they should not be treated by the same padding function.
Analogous to the common data collators, the padding tokens in the labels with `-100` so that those tokens are **not** taken into account when computing the loss.


```python
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
```

Let's initialize the data collator.

```python
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
```

Next, the evaluation metric is defined. As mentioned earlier, the
predominant metric in ASR is the word error rate (WER), hence we will
use it in this notebook as well.

```python
wer_metric = load_metric("wer")
```

The model will return a sequence of logit vectors:

$$ \mathbf{y}_1, \ldots, \mathbf{y}_m $$, 

with \\(\mathbf{y}_1 = f_{\theta}(x_1, \ldots, x_n)[0]\\) and \\(n >> m\\).

A logit vector \\( \mathbf{y}_1 \\) contains the log-odds for each word in the
vocabulary we defined earlier, thus \\(\text{len}(\mathbf{y}_i) =\\)
`config.vocab_size`. We are interested in the most likely prediction of
the model and thus take the `argmax(...)` of the logits. Also, we
transform the encoded labels back to the original string by replacing
`-100` with the `pad_token_id` and decoding the ids while making sure
that consecutive tokens are **not** grouped to the same token in CTC
style \\({}^1\\).

```python
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
```

Now, we can load the pretrained `XLSR-Wav2Vec2` checkpoint. The tokenizer's `pad_token_id` must be to define the model's `pad_token_id` or in the case of `Wav2Vec2ForCTC` also CTC's *blank token* ${}^2$. To save GPU memory, we enable PyTorch's [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) and also set the loss reduction to "*mean*".

Because the dataset is quite small (~6h of training data) and because Common Voice is quite noisy, fine-tuning Facebook's [wav2vec2-large-xlsr-53 checkpoint](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) seems to require some hyper-parameter tuning. Therefore, I had to play around a bit with different values for dropout, [SpecAugment](https://arxiv.org/abs/1904.08779)'s masking dropout rate, layer dropout, and the learning rate until training seemed to be stable enough. 

**Note**: When using this notebook to train XLSR-Wav2Vec2 on another language of Common Voice those hyper-parameter settings might not work very well. Feel free to adapt those depending on your use case.

from transformers import Wav2Vec2ForCTC

```python
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
```

**Log Output:**

```bash
    Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2vec2-large-xlsr-53 and are newly initialized: ['lm_head.weight', 'lm_head.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore. 
Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.

```python
model.freeze_feature_extractor()
```

In a final step, we define all parameters related to training. 
To give more explanation on some of the parameters:

- 	`group_by_length` makes training more efficient by grouping training samples of similar input length into one batch. This can significantly speed up training time by heavily reducing the overall number of useless padding tokens that are passed through the model
- 	`learning_rate` and `weight_decay` were heuristically tuned until fine-tuning has become stable. Note that those parameters strongly depend on the Timit dataset and might be suboptimal for other speech datasets.

For more explanations on other parameters, one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).

**Note**: If one wants to save the trained models in his/her google drive the commented-out `output_dir` can be used instead.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
  # output_dir="./wav2vec2-large-xlsr-turkish-demo",
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
)
```

Now, all instances can be passed to Trainer and we are ready to start
training!

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit_prepared["train"],
    eval_dataset=timit_prepared["test"],
    tokenizer=processor.feature_extractor,
)
```

------------------------------------------------------------------------

\\({}^1\\) To allow models to become independent of the speaker rate, in
CTC, consecutive tokens that are identical are simply grouped as a
single token. However, the encoded labels should not be grouped when
decoding since they don\'t correspond to the predicted tokens of the
model, which is why the `group_tokens=False` parameter has to be passed.
If we wouldn\'t pass this parameter a word like `"hello"` would
incorrectly be encoded, and decoded as `"helo"`.

\\({}^2\\) The blank token allows the model to predict a word, such as
`"hello"` by forcing it to insert the blank token between the two l\'s.
A CTC-conform prediction of `"hello"` of our model would be
`[PAD] [PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD]`.

### Training

Training will take between 180 and 240 minutes depending on the GPU allocated to this notebook. While the trained model yields somewhat satisfying results on *Common Voice*'s test data of Turkish, it is by no means an optimally fine-tuned model. The purpose of this notebook is to demonstrate how XLSR-Wav2Vec2's [checkpoint](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) can be fine-tuned on a low-resource ASR dataset.

In case you want to use this google colab to fine-tune your model, you should make sure that your training doesn't stop due to inactivity. A simple hack to prevent this is to paste the following code into the console of this tab (*right mouse click -> inspect -> Console tab and insert code*).

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```

```python
trainer.train()
```

**Print Output:**

| Step  | Training Loss  | Validation Loss | WER | Runtime | Samples per Second |
|---|---|---|---|---|---|
| 400  | 5.216200 | 2.705985 | 1.000000 | 204.768000 | 8.043000 |
| 800  | 0.804600 | 0.446772 | 0.589725 | 209.062900 | 7.878000 |
| 1200 | 0.244300 | 0.402035 | 0.531611 | 202.232700 | 8.144000 |
| 1600 | 0.151700 | 0.406878 | 0.508324 | 206.073700 | 7.992000 |
| 2000 | 0.107100 | 0.417614 | 0.495149 | 216.555500 | 7.605000 |
| 2400 | 0.078200 | 0.427536 | 0.480748 | 222.747800 | 7.394000 |
| 2800 | 0.071700 | 0.423708 | 0.463589 | 219.078800 | 7.518000 |

The training loss goes down and we can see that the WER on the test set also improves nicely. Because this notebook is just for demonstration purposes, we can stop here.

The resulting model of this notebook has been saved to [`patrickvonplaten/wav2vec2-large-xlsr-turkish-demo`](https://huggingface.co/patrickvonplaten/wav2vec2-large-xlsr-turkish-demo)

As a final check, let's load the model and verify that it indeed has learned to transcribe Turkish speech.

Let's first load the pretrained checkpoint.

```python
model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-turkish-demo").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-turkish-demo")
```

Now, we will just take the first example of the test set, run it through the model and take the `argmax(...)` of the logits to retrieve the predicted token ids.


```python
input_dict = processor(common_voice_test["input_values"][0], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0][0]
```

We adapted `common_voice_test` quite a bit so that the dataset instance does not contain the original sentence label anymore. Thus, we re-use the original dataset to get the label of the first example.

```python
common_voice_test_transcription = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11", split="test")
```

Finally, we can decode the example.

```python
print("Prediction:")
print(processor.decode(pred_ids[0]))

print("\nReference:")
print(common_voice_test_transcription["sentence"][0].lower())
```

**Print Output:**

| pred_str |  target_text |
|----------|:-------------:|
| hata küçük şeyler için birbüy bi şeyler kolaluyor ve yenekiçük şeyler için bir bimizi inciltiyoruz | hayatta küçük şeyleri kovalıyor ve yine küçük şeyler için birbirimizi incitiyoruz. |

Alright! The transcription can definitely be recognized from our prediction, but it is far from being perfect. Training the model a bit longer, spending more time on the data preprocessing, and especially using a language model for decoding would certainly improve the model's overall performance.

For a demonstration model on a low-resource language, the results are acceptable, however 🤗.
