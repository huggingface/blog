---
title: "BERT 101 - State Of The Art NLP Model Explained"
thumbnail: /blog/assets/52_bert_101/thumbnail.jpg
authors:
- user: britneymuller
---
<html itemscope itemtype="https://schema.org/FAQPage">
<h1>BERT 101 🤗 State Of The Art NLP Model Explained</h1>

{blog_metadata}

{authors}

<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

## What is BERT?

BERT, short for Bidirectional Encoder Representations from Transformers, is a Machine Learning (ML) model for natural language processing. It was developed in 2018 by researchers at Google AI Language and serves as a swiss army knife solution to 11+ of the most common language tasks, such as sentiment analysis and named entity recognition. 

Language has historically been difficult for computers to ‘understand’. Sure, computers can collect, store, and read text inputs but they lack basic language _context_.

So, along came Natural Language Processing (NLP): the field of artificial intelligence aiming for computers to read, analyze, interpret and derive meaning from text and spoken words. This practice combines linguistics, statistics, and Machine Learning to assist computers in ‘understanding’ human language.

Individual NLP tasks have traditionally been solved by individual models created for each specific task. That is, until— BERT!

BERT revolutionized the NLP space by solving for 11+ of the most common NLP tasks (and better than previous models) making it the jack of all NLP trades. 

In this guide, you'll learn what BERT is, why it’s different, and how to get started using BERT:

1. [What is BERT used for?](#1-what-is-bert-used-for)
2. [How does BERT work?](#2-how-does-bert-work)
3. [BERT model size & architecture](#3-bert-model-size--architecture)
4. [BERT’s performance on common language tasks](#4-berts-performance-on-common-language-tasks)
5. [Environmental impact of deep learning](#5-enviornmental-impact-of-deep-learning)
6. [The open source power of BERT](#6-the-open-source-power-of-bert)
7. [How to get started using BERT](#7-how-to-get-started-using-bert)
8. [BERT FAQs](#8-bert-faqs)
9. [Conclusion](#9-conclusion)

Let's get started! 🚀


## 1. What is BERT used for?

BERT can be used on a wide variety of language tasks:

-   Can determine how positive or negative a movie’s reviews are. [(Sentiment Analysis)](https://huggingface.co/blog/sentiment-analysis-python)
-   Helps chatbots answer your questions. [(Question answering)](https://huggingface.co/tasks/question-answering)
-   Predicts your text when writing an email (Gmail). [(Text prediction)](https://huggingface.co/tasks/fill-mask)
-   Can write an article about any topic with just a few sentence inputs. [(Text generation)](https://huggingface.co/tasks/text-generation)
-   Can quickly summarize long legal contracts. [(Summarization)](https://huggingface.co/tasks/summarization)
-   Can differentiate words that have multiple meanings (like ‘bank’) based on the surrounding text. (Polysemy resolution)

**There are many more language/NLP tasks + more detail behind each of these.**

***Fun Fact:*** You interact with NLP (and likely BERT) almost every single day! 

NLP is behind Google Translate, voice assistants (Alexa, Siri, etc.), chatbots, Google searches, voice-operated GPS, and more.

---

### 1.1 Example of BERT


BERT helps Google better surface (English) results for nearly all searches since November of 2020. 

Here’s an example of how BERT helps Google better understand specific searches like:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="BERT Google Search Example" src="assets/52_bert_101/BERT-example.png"></medium-zoom>
  <figcaption><a href="https://blog.google/products/search/search-language-understanding-bert/">Source</a></figcaption>
</figure>


Pre-BERT Google surfaced information about getting a prescription filled. 

Post-BERT Google understands that “for someone” relates to picking up a prescription for someone else and the search results now help to answer that.

---

## 2. How does BERT Work?

BERT works by leveraging the following:

### 2.1 Large amounts of training data

A massive dataset of 3.3 Billion words has contributed to BERT’s continued success. 

BERT was specifically trained on Wikipedia (\~2.5B words) and Google’s BooksCorpus (\~800M words). These large informational datasets contributed to BERT’s deep knowledge not only of the English language but also of our world! 🚀

Training on a dataset this large takes a long time. BERT’s training was made possible thanks to the novel Transformer architecture and sped up by using TPUs (Tensor Processing Units - Google’s custom circuit built specifically for large ML models). —64 TPUs trained BERT over the course of 4 days.

**Note:** Demand for smaller BERT models is increasing in order to use BERT within smaller computational environments (like cell phones and personal computers). [23 smaller BERT models were released in March 2020](https://github.com/google-research/bert). [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) offers a lighter version of BERT; runs 60% faster while maintaining over 95% of BERT’s performance.

### 2.2 What is a Masked Language Model?

MLM enables/enforces bidirectional learning from text by masking (hiding) a word in a sentence and forcing BERT to bidirectionally use the words on either side of the covered word to predict the masked word. This had never been done before!

**Fun Fact:** We naturally do this as humans! 

**Masked Language Model Example:**

Imagine your friend calls you while camping in Glacier National Park and their service begins to cut out. The last thing you hear before the call drops is:

<p class="text-center px-6">Friend: “Dang! I’m out fishing and a huge trout just [blank] my line!”</p>

Can you guess what your friend said?? 

You’re naturally able to predict the missing word by considering the words bidirectionally before and after the missing word as context clues (in addition to your historical knowledge of how fishing works). Did you guess that your friend said, ‘broke’? That’s what we predicted as well but even we humans are error-prone to some of these methods. 

**Note:** This is why you’ll often see a “Human Performance” comparison to a language model’s performance scores. And yes, newer models like BERT can be more accurate than humans! 🤯

The bidirectional methodology you did to fill in the [blank] word above is similar to how BERT attains state-of-the-art accuracy. A random 15% of tokenized words are hidden during training and BERT’s job is to correctly predict the hidden words. Thus, directly teaching the model about the English language (and the words we use). Isn’t that neat?

Play around with BERT’s masking predictions: 

<div class="bg-white pb-1">
    <div class="SVELTE_HYDRATER contents" data-props="{&quot;apiUrl&quot;:&quot;https://api-inference.huggingface.co&quot;,&quot;apiToken&quot;:&quot;&quot;,&quot;model&quot;:{&quot;branch&quot;:&quot;main&quot;,&quot;cardData&quot;:{&quot;language&quot;:&quot;en&quot;,&quot;tags&quot;:[&quot;exbert&quot;],&quot;license&quot;:&quot;apache-2.0&quot;,&quot;datasets&quot;:[&quot;bookcorpus&quot;,&quot;wikipedia&quot;]},&quot;cardError&quot;:{&quot;errors&quot;:[],&quot;warnings&quot;:[]},&quot;cardExists&quot;:true,&quot;config&quot;:{&quot;architectures&quot;:[&quot;BertForMaskedLM&quot;],&quot;model_type&quot;:&quot;bert&quot;},&quot;id&quot;:&quot;bert-base-uncased&quot;,&quot;lastModified&quot;:&quot;2021-05-18T16:20:13.000Z&quot;,&quot;pipeline_tag&quot;:&quot;fill-mask&quot;,&quot;library_name&quot;:&quot;transformers&quot;,&quot;mask_token&quot;:&quot;[MASK]&quot;,&quot;model-index&quot;:null,&quot;private&quot;:false,&quot;gated&quot;:false,&quot;pwcLink&quot;:{&quot;error&quot;:&quot;Unknown error, can&#39;t generate link to Papers With Code.&quot;},&quot;siblings&quot;:[{&quot;rfilename&quot;:&quot;.gitattributes&quot;},{&quot;rfilename&quot;:&quot;README.md&quot;},{&quot;rfilename&quot;:&quot;config.json&quot;},{&quot;rfilename&quot;:&quot;flax_model.msgpack&quot;},{&quot;rfilename&quot;:&quot;pytorch_model.bin&quot;},{&quot;rfilename&quot;:&quot;rust_model.ot&quot;},{&quot;rfilename&quot;:&quot;tf_model.h5&quot;},{&quot;rfilename&quot;:&quot;tokenizer.json&quot;},{&quot;rfilename&quot;:&quot;tokenizer_config.json&quot;},{&quot;rfilename&quot;:&quot;vocab.txt&quot;}],&quot;tags&quot;:[&quot;pytorch&quot;,&quot;tf&quot;,&quot;jax&quot;,&quot;rust&quot;,&quot;bert&quot;,&quot;fill-mask&quot;,&quot;en&quot;,&quot;dataset:bookcorpus&quot;,&quot;dataset:wikipedia&quot;,&quot;arxiv:1810.04805&quot;,&quot;transformers&quot;,&quot;exbert&quot;,&quot;license:apache-2.0&quot;,&quot;autonlp_compatible&quot;,&quot;infinity_compatible&quot;],&quot;tag_objs&quot;:[{&quot;id&quot;:&quot;fill-mask&quot;,&quot;label&quot;:&quot;Fill-Mask&quot;,&quot;subType&quot;:&quot;nlp&quot;,&quot;type&quot;:&quot;pipeline_tag&quot;},{&quot;id&quot;:&quot;pytorch&quot;,&quot;label&quot;:&quot;PyTorch&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;tf&quot;,&quot;label&quot;:&quot;TensorFlow&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;jax&quot;,&quot;label&quot;:&quot;JAX&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;rust&quot;,&quot;label&quot;:&quot;Rust&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;transformers&quot;,&quot;label&quot;:&quot;Transformers&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;dataset:bookcorpus&quot;,&quot;label&quot;:&quot;bookcorpus&quot;,&quot;type&quot;:&quot;dataset&quot;},{&quot;id&quot;:&quot;dataset:wikipedia&quot;,&quot;label&quot;:&quot;wikipedia&quot;,&quot;type&quot;:&quot;dataset&quot;},{&quot;id&quot;:&quot;en&quot;,&quot;label&quot;:&quot;en&quot;,&quot;type&quot;:&quot;language&quot;},{&quot;id&quot;:&quot;arxiv:1810.04805&quot;,&quot;label&quot;:&quot;arxiv:1810.04805&quot;,&quot;type&quot;:&quot;arxiv&quot;},{&quot;id&quot;:&quot;license:apache-2.0&quot;,&quot;label&quot;:&quot;apache-2.0&quot;,&quot;type&quot;:&quot;license&quot;},{&quot;id&quot;:&quot;bert&quot;,&quot;label&quot;:&quot;bert&quot;,&quot;type&quot;:&quot;other&quot;},{&quot;id&quot;:&quot;exbert&quot;,&quot;label&quot;:&quot;exbert&quot;,&quot;type&quot;:&quot;other&quot;},{&quot;id&quot;:&quot;autonlp_compatible&quot;,&quot;label&quot;:&quot;AutoNLP Compatible&quot;,&quot;type&quot;:&quot;other&quot;},{&quot;id&quot;:&quot;infinity_compatible&quot;,&quot;label&quot;:&quot;Infinity Compatible&quot;,&quot;type&quot;:&quot;other&quot;}],&quot;transformersInfo&quot;:{&quot;auto_model&quot;:&quot;AutoModelForMaskedLM&quot;,&quot;pipeline_tag&quot;:&quot;fill-mask&quot;,&quot;processor&quot;:&quot;AutoTokenizer&quot;},&quot;widgetData&quot;:[{&quot;text&quot;:&quot;Paris is the [MASK] of France.&quot;},{&quot;text&quot;:&quot;The goal of life is [MASK].&quot;}],&quot;likes&quot;:104,&quot;isLikedByUser&quot;:false},&quot;shouldUpdateUrl&quot;:true}" data-target="InferenceWidget">
        <div class="flex flex-col w-full max-w-full">
            <div class="font-semibold flex items-center mb-2">
                <div class="text-lg flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" class="-ml-1 mr-1 text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24">
                        <path d="M11 15H6l7-14v8h5l-7 14v-8z" fill="currentColor"></path>
                    </svg>
                    Hosted inference API
                </div>
                <a target="_blank" href="https://api-inference.huggingface.co/">
                    <svg class="ml-1.5 text-sm text-gray-400 hover:text-black" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32">
                        <path d="M17 22v-8h-4v2h2v6h-3v2h8v-2h-3z" fill="currentColor"></path>
                        <path d="M16 8a1.5 1.5 0 1 0 1.5 1.5A1.5 1.5 0 0 0 16 8z" fill="currentColor"></path>
                        <path d="M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4z" fill="currentColor"></path>
                    </svg>
                </a>
            </div>
            <div class="flex items-center justify-between flex-wrap w-full max-w-full text-sm text-gray-500 mb-0.5">
                <a class="hover:underline" href="/tasks/fill-mask" target="_blank" title="Learn more about fill-mask">
                    <div class="inline-flex items-center mr-2 mb-1.5">
                        <svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 18 19">
                            <path d="M12.3625 13.85H10.1875V12.7625H12.3625V10.5875H13.45V12.7625C13.4497 13.0508 13.335 13.3272 13.1312 13.5311C12.9273 13.735 12.6508 13.8497 12.3625 13.85V13.85Z"></path>
                            <path d="M5.8375 8.41246H4.75V6.23746C4.75029 5.94913 4.86496 5.67269 5.06884 5.4688C5.27272 5.26492 5.54917 5.15025 5.8375 5.14996H8.0125V6.23746H5.8375V8.41246Z"></path>
                            <path d="M15.625 5.14998H13.45V2.97498C13.4497 2.68665 13.335 2.4102 13.1312 2.20632C12.9273 2.00244 12.6508 1.88777 12.3625 1.88748H2.575C2.28666 1.88777 2.01022 2.00244 1.80633 2.20632C1.60245 2.4102 1.48778 2.68665 1.4875 2.97498V12.7625C1.48778 13.0508 1.60245 13.3273 1.80633 13.5311C2.01022 13.735 2.28666 13.8497 2.575 13.85H4.75V16.025C4.75028 16.3133 4.86495 16.5898 5.06883 16.7936C5.27272 16.9975 5.54916 17.1122 5.8375 17.1125H15.625C15.9133 17.1122 16.1898 16.9975 16.3937 16.7936C16.5975 16.5898 16.7122 16.3133 16.7125 16.025V6.23748C16.7122 5.94915 16.5975 5.6727 16.3937 5.46882C16.1898 5.26494 15.9133 5.15027 15.625 5.14998V5.14998ZM15.625 16.025H5.8375V13.85H8.0125V12.7625H5.8375V10.5875H4.75V12.7625H2.575V2.97498H12.3625V5.14998H10.1875V6.23748H12.3625V8.41248H13.45V6.23748H15.625V16.025Z"></path>
                        </svg>
                         <span>Fill-Mask</span>
                     </div>
                </a>
                <div class="relative mb-1.5 false false">
                <div class="no-hover:hidden inline-flex justify-between w-32 lg:w-44 rounded-md border border-gray-100 px-4 py-1">
                    <div class="text-sm truncate">Examples</div>
                    <svg class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform false" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </div>
                <div class="with-hover:hidden inline-flex justify-between w-32 lg:w-44 rounded-md border border-gray-100 px-4 py-1">
                    <div class="text-sm truncate">Examples</div>
                    <svg class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform false" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                    </svg>
                </div>
            </div>
        </div>
        <form>
            <div class="text-sm text-gray-500 mb-1.5">Mask token: 
                <code>[MASK]</code>
            </div>
            <label class="block ">
                <span class=" block overflow-auto resize-y py-2 px-3 w-full min-h-[42px] max-h-[500px] border border-gray-200 rounded-lg shadow-inner outline-none focus:ring-1 focus:ring-inset focus:ring-indigo-200 focus:shadow-inner dark:bg-gray-925 svelte-1wfa7x9" role="textbox" contenteditable style="--placeholder: 'Your sentence here...'"></span>
            </label>
            <button class="btn-widget w-24 h-10 px-5 mt-2" type="submit">Compute</button>
        </form>
        <div class="mt-2">
            <div class="text-gray-400 text-xs">This model can be loaded on the Inference API on-demand.</div>
        </div>
        <div class="mt-auto pt-4 flex items-center text-xs text-gray-500">
            <button class="flex items-center cursor-not-allowed text-gray-300" disabled>
                <svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);">
                    <path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path>
                    <path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path>
                    <path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path>
                </svg>
                JSON Output
            </button>
            <button class="flex items-center ml-auto">
                <svg class="mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32">
                    <path d="M22 16h2V8h-8v2h6v6z" fill="currentColor"></path>
                    <path d="M8 24h8v-2h-6v-6H8v8z" fill="currentColor"></path>
                    <path d="M26 28H6a2.002 2.002 0 0 1-2-2V6a2.002 2.002 0 0 1 2-2h20a2.002 2.002 0 0 1 2 2v20a2.002 2.002 0 0 1-2 2zM6 6v20h20.001L26 6z" fill="currentColor"></path>
                </svg>
                Maximize
            </button>
        </div>
    </div>
</div>

**Fun Fact:** Masking has been around a long time - [1953 Paper on Cloze procedure (or ‘Masking’)](https://psycnet.apa.org/record/1955-00850-001). 

### 2.3 What is Next Sentence Prediction?

NSP (Next Sentence Prediction) is used to help BERT learn about relationships between sentences by predicting if a given sentence follows the previous sentence or not. 

**Next Sentence Prediction Example:**

1. Paul went shopping. He bought a new shirt. (correct sentence pair)
2. Ramona made coffee. Vanilla ice cream cones for sale. (incorrect sentence pair)

In training, 50% correct sentence pairs are mixed in with 50% random sentence pairs to help BERT increase next sentence prediction accuracy.

**Fun Fact:** BERT is trained on both MLM (50%) and NSP (50%) at the same time.

### 2.4 Transformers

The Transformer architecture makes it possible to parallelize ML training extremely efficiently. Massive parallelization thus makes it feasible to train BERT on large amounts of data in a relatively short period of time. 

Transformers use an attention mechanism to observe relationships between words. A concept originally proposed in the popular [2017 Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper sparked the use of Transformers in NLP models all around the world.


<p align="center">

>Since their introduction in 2017, Transformers have rapidly become the state-of-the-art approach to tackle tasks in many domains such as natural language processing, speech recognition, and computer vision. In short, if you’re doing deep learning, then you need Transformers!

<p class="text-center px-6">Lewis Tunstall, Hugging Face ML Engineer & <a href="https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)">Author of Natural Language Processing with Transformers</a></p>
</p>

Timeline of popular Transformer model releases:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Transformer model timeline" src="assets/52_bert_101/transformers-timeline.png"></medium-zoom>
  <figcaption><a href="https://huggingface.co/course/chapter1/4">Source</a></figcaption>
</figure>

#### 2.4.1 How do Transformers work?

Transformers work by leveraging attention, a powerful deep-learning algorithm, first seen in computer vision models.

—Not all that different from how we humans process information through attention. We are incredibly good at forgetting/ignoring mundane daily inputs that don’t pose a threat or require a response from us. For example, can you remember everything you saw and heard coming home last Tuesday? Of course not! Our brain’s memory is limited and valuable. Our recall is aided by our ability to forget trivial inputs. 

Similarly, Machine Learning models need to learn how to pay attention only to the things that matter and not waste computational resources processing irrelevant information. Transformers create differential weights signaling which words in a sentence are the most critical to further process.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Encoder and Decoder" src="assets/52_bert_101/encoder-and-decoder-transformers-blocks.png"></medium-zoom>
</figure>

A transformer does this by successively processing an input through a stack of transformer layers, usually called the encoder. If necessary, another stack of transformer layers - the decoder - can be used to predict a target output. —BERT however, doesn’t use a decoder. Transformers are uniquely suited for unsupervised learning because they can efficiently process millions of data points.

Fun Fact: Google has been using your reCAPTCHA selections to label training data since 2011. The entire Google Books archive and 13 million articles from the New York Times catalog have been transcribed/digitized via people entering reCAPTCHA text. Now, reCAPTCHA is asking us to label Google Street View images, vehicles, stoplights, airplanes, etc. Would be neat if Google made us aware of our participation in this effort (as the training data likely has future commercial intent) but I digress..

<p class="text-center">
    To learn more about Transformers check out our <a href="https://huggingface.co/course/chapter1/1">Hugging Face Transformers Course</a>.
</p>

## 3. BERT model size & architecture

Let’s break down the architecture for the two original BERT models:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Original BERT models architecture" src="assets/52_bert_101/BERT-size-and-architecture.png"></medium-zoom>
</figure>


ML Architecture Glossary:

| ML Architecture Parts | Definition                                                                                                                                                        |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameters:           | Number of learnable variables/values available for the model.                                                                                                     |
| Transformer Layers:   | Number of Transformer blocks. A transformer block transforms a sequence of word representations to a sequence of contextualized words (numbered representations). |
| Hidden Size:          | Layers of mathematical functions, located between the input and output, that assign weights (to words) to produce a desired result.                               |
| Attention Heads:      | The size of a Transformer block.                                                                                                                                  |
| Processing:           | Type of processing unit used to train the model.                                                                                                                  |
| Length of Training:   | Time it took to train the model. 


Here’s how many of the above ML architecture parts BERTbase and BERTlarge has:


|           | Transformer Layers | Hidden Size | Attention Heads | Parameters | Processing | Length of Training |
|-----------|--------------------|-------------|-----------------|------------|------------|--------------------|
| BERTbase  | 12                 | 768         | 12              | 110M       | 4 TPUs     | 4 days             |
| BERTlarge | 24                 | 1024        | 16              | 340M       | 16 TPUs    | 4 days             |



Let’s take a look at how BERTlarge’s  additional layers, attention heads, and parameters have increased its performance across NLP tasks.

## 4. BERT's performance on common language tasks

BERT has successfully achieved state-of-the-art accuracy on 11 common NLP tasks, outperforming previous top NLP models, and is the first to outperform humans! 
But, how are these achievements measured?

### NLP Evaluation Methods: 

#### 4.1 SQuAD v1.1 & v2.0
[SQuAD](https://huggingface.co/datasets/squad) (Stanford Question Answering Dataset) is a reading comprehension dataset of around 108k questions that can be answered via a corresponding paragraph of Wikipedia text. BERT’s performance on this evaluation method was a big achievement beating previous state-of-the-art models and human-level performance:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="BERT's performance on SQuAD v1.1" src="assets/52_bert_101/BERTs-performance-on-SQuAD1.1.png"></medium-zoom>
</figure>

#### 4.2 SWAG
[SWAG](https://huggingface.co/datasets/swag) (Situations With Adversarial Generations) is an interesting evaluation in that it detects a model’s ability to infer commonsense! It does this through a large-scale dataset of 113k multiple choice questions about common sense situations. These questions are transcribed from a video scene/situation and SWAG provides the model with four possible outcomes in the next scene. The model then does its’ best at predicting the correct answer.

BERT out outperformed top previous top models including human-level performance:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Transformer model timeline" src="assets/52_bert_101/BERTs-performance-on-SWAG.png"></medium-zoom>
</figure>

#### 4.3 GLUE Benchmark
[GLUE](https://huggingface.co/datasets/glue) (General Language Understanding Evaluation) benchmark is a group of resources for training, measuring, and analyzing language models comparatively to one another. These resources consist of nine “difficult” tasks designed to test an NLP model’s understanding. Here’s a summary of each of those tasks:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Transformer model timeline" src="assets/52_bert_101/GLUE-Benchmark-tasks.png"></medium-zoom>
</figure>

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Transformer model timeline" src="assets/52_bert_101/BERTs-Performance-on-GLUE.png"></medium-zoom>
</figure>

While some of these tasks may seem irrelevant and banal, it’s important to note that these evaluation methods are _incredibly_ powerful in indicating which models are best suited for your next NLP application. 

Attaining performance of this caliber isn’t without consequences. Next up, let’s learn about Machine Learning's impact on the environment.


## 5. Enviornmental impact of deep learning

Large Machine Learning models require massive amounts of data which is expensive in both time and compute resources.

These models also have an environmental impact: 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Transformer model timeline" src="assets/52_bert_101/enviornmental-impact-of-machine-learning.png"></medium-zoom>
  <figcaption><a href="https://huggingface.co/course/chapter1/4">Source</a></figcaption>
</figure>

Machine Learning’s environmental impact is one of the many reasons we believe in democratizing the world of Machine Learning through open source! Sharing large pre-trained language models is essential in reducing the overall compute cost and carbon footprint of our community-driven efforts.


## 6. The open source power of BERT

Unlike other large learning models like GPT-3, BERT’s source code is publicly accessible ([view BERT’s code on Github](https://github.com/google-research/bert)) allowing BERT to be more widely used all around the world. This is a game-changer!

Developers are now able to get a state-of-the-art model like BERT up and running quickly without spending large amounts of time and money. 🤯 

Developers can instead focus their efforts on fine-tuning BERT to customize the model’s performance to their unique tasks. 

It’s important to note that [thousands](https://huggingface.co/models?sort=downloads&search=bert) of open-source and free, pre-trained BERT models are currently available for specific use cases if you don’t want to fine-tune BERT. 

BERT models pre-trained for specific tasks:

-   [Twitter sentiment analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)
-   [Analysis of Japanese text](https://huggingface.co/cl-tohoku/bert-base-japanese-char)
-   [Emotion categorizer (English - anger, fear, joy, etc.)](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
-   [Clinical Notes analysis](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
-   [Speech to text translation](https://huggingface.co/facebook/hubert-large-ls960-ft)
-   [Toxic comment detection](https://huggingface.co/unitary/toxic-bert?)

You can also find [hundreds of pre-trained, open-source Transformer models](https://huggingface.co/models?library=transformers&sort=downloads) available on the Hugging Face Hub.


## 7. How to get started using BERT

We've [created this notebook](https://colab.research.google.com/drive/1YtTqwkwaqV2n56NC8xerflt95Cjyd4NE?usp=sharing) so you can try BERT through this easy tutorial in Google Colab. Open the notebook or add the following code to your own. Pro Tip: Use (Shift + Click) to run a code cell.

Note: Hugging Face's [pipeline class](https://huggingface.co/docs/transformers/main_classes/pipelines) makes it incredibly easy to pull in open source ML models like transformers with just a single line of code.

### 7.1 Install Transformers

First, let's install Transformers via the following code:

```python
!pip install transformers
```

### 7.2 Try out BERT

Feel free to swap out the sentence below for one of your own. However, leave [MASK] in somewhere to allow BERT to predict the missing word

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Artificial Intelligence [MASK] take over the world.")
```

When you run the above code you should see an output like this:

```
[{'score': 0.3182411789894104,
  'sequence': 'artificial intelligence can take over the world.',
  'token': 2064,
  'token_str': 'can'},
 {'score': 0.18299679458141327,
  'sequence': 'artificial intelligence will take over the world.',
  'token': 2097,
  'token_str': 'will'},
 {'score': 0.05600147321820259,
  'sequence': 'artificial intelligence to take over the world.',
  'token': 2000,
  'token_str': 'to'},
 {'score': 0.04519503191113472,
  'sequence': 'artificial intelligences take over the world.',
  'token': 2015,
  'token_str': '##s'},
 {'score': 0.045153118669986725,
  'sequence': 'artificial intelligence would take over the world.',
  'token': 2052,
  'token_str': 'would'}]
```

Kind of frightening right? 🙃

### 7.3 Be aware of model bias

Let's see what jobs BERT suggests for a "man":

```python
unmasker("The man worked as a [MASK].")
```

When you run the above code you should see an output that looks something like:

```python
[{'score': 0.09747546911239624,
  'sequence': 'the man worked as a carpenter.',
  'token': 10533,
  'token_str': 'carpenter'},
 {'score': 0.052383411675691605,
  'sequence': 'the man worked as a waiter.',
  'token': 15610,
  'token_str': 'waiter'},
 {'score': 0.04962698742747307,
  'sequence': 'the man worked as a barber.',
  'token': 13362,
  'token_str': 'barber'},
 {'score': 0.037886083126068115,
  'sequence': 'the man worked as a mechanic.',
  'token': 15893,
  'token_str': 'mechanic'},
 {'score': 0.037680838257074356,
  'sequence': 'the man worked as a salesman.',
  'token': 18968,
  'token_str': 'salesman'}]
```

BERT predicted the man's job to be a Carpenter, Waiter, Barber, Mechanic, or Salesman

  Now let's see what jobs BERT suggesst for "woman"

```python
unmasker("The woman worked as a [MASK].")
```

You should see an output that looks something like:
```python
[{'score': 0.21981535851955414,
  'sequence': 'the woman worked as a nurse.',
  'token': 6821,
  'token_str': 'nurse'},
 {'score': 0.1597413569688797,
  'sequence': 'the woman worked as a waitress.',
  'token': 13877,
  'token_str': 'waitress'},
 {'score': 0.11547300964593887,
  'sequence': 'the woman worked as a maid.',
  'token': 10850,
  'token_str': 'maid'},
 {'score': 0.03796879202127457,
  'sequence': 'the woman worked as a prostitute.',
  'token': 19215,
  'token_str': 'prostitute'},
 {'score': 0.030423851683735847,
  'sequence': 'the woman worked as a cook.',
  'token': 5660,
  'token_str': 'cook'}]
```

BERT predicted the woman's job to be a Nurse, Waitress, Maid, Prostitute, or Cook displaying a clear gender bias in professional roles.

### 7.4 Some other BERT Notebooks you might enjoy:

[A Visual Notebook to BERT for the First Time](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)

[Train your tokenizer](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)

+Don't forget to checkout the [Hugging Face Transformers Course](https://huggingface.co/course/chapter1/1) to learn more 🎉


## 8. BERT FAQs

  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">Can BERT be used with PyTorch?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <div itemprop="text">
        Yes! Our experts at Hugging Face have open-sourced the <a href="https://www.google.com/url?q=https://github.com/huggingface/transformers">PyTorch transformers repository on GitHub</a>. 
        <br />
        <p>Pro Tip: Lewis Tunstall, Leandro von Werra, and Thomas Wolf also wrote a book to help people build language applications with Hugging Face called, <a href="https://www.google.com/search?kgmid=/g/11qh58xzh7&hl=en-US&q=Natural+Language+Processing+with+Transformers:+Building+Language+Applications+with+Hugging+Face">‘Natural Language Processing with Transformers’</a>.</p>
      </div>
    </div>
  </div>
  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">Can BERT be used with Tensorflow?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <div itemprop="text">
        Yes! <a href="https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/bert#transformers.TFBertModel">You can use Tensorflow as the backend of Transformers.</a>
    </div>
  </div>
</div>
<div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
  <h3 itemprop="name">How long does it take to pre-train BERT?</h3>
  <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
    <div itemprop="text">
        The 2 original BERT models were trained on 4(BERTbase) and 16(BERTlarge) Cloud TPUs for 4 days.
    </div>
  </div>
</div>
<div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
  <h3 itemprop="name">How long does it take to fine-tune BERT?</h3>
  <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
    <div itemprop="text">
        For common NLP tasks discussed above, BERT takes between 1-25mins on a single Cloud TPU or between 1-130mins on a single GPU.
    </div>
  </div>
</div>
<div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
  <h3 itemprop="name">What makes BERT different?</h3>
  <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
    <div itemprop="text">
        BERT was one of the first models in NLP that was trained in a two-step way: 
        <ol>
            <li>1. BERT was trained on massive amounts of unlabeled data (no human annotation) in an unsupervised fashion.</li>
            <li>2. BERT was then trained on small amounts of human-annotated data starting from the previous pre-trained model resulting in state-of-the-art performance.</li>
        </ol>
    </div>
  </div>
</div>
</html>

## 9. Conclusion

BERT is a highly complex and advanced language model that helps people automate language understanding. Its ability to accomplish state-of-the-art performance is supported by training on massive amounts of data and leveraging Transformers architecture to revolutionize the field of NLP. 

Thanks to BERT’s open-source library, and the incredible AI community’s efforts to continue to improve and share new BERT models, the future of untouched NLP milestones looks bright.

What will you create with BERT? 

Learn how to [fine-tune BERT](https://huggingface.co/docs/transformers/training) for your particular use case 🤗 