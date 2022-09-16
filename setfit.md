---
title: "SetFit: Efficient Few-Shot Learning Without Prompts"
thumbnail: /blog/assets/
---
<div class="blog-metadata">
    <small>Published September 16, 2022.</small>
   
<h1>SetFit: Efficient Few-Shot Learning Without Prompts</h1>

</div>

<div class="author-card">
    <a href="/unsojo">
        <div class="bfc">
            <code>unsojo</code>
            <span class="fullname">Unso Jo</span>
        </div>
  </a>
</div>




# SetFit: Efficient Few-Shot Learning Without Prompts
###### The Hugging Face and Intel team introduce a new prompt-free few-shot learning regime called Setfit, with accompanying [code](https://github.com/SetFit/setfit), longer [paper](), and datasets on [HF hub](https://huggingface.co/SetFit)

<kbd>
<img src="assets/gpt3_stone.png">
</kbd>


## Introducing SetFit
In a collaborative effort between Hugging Face and Intel, we introduce SetFit, a prompt-free few-shot regime made with practicality and efficiency in mind. Recent discussions in ML have focused on few-shot regimes, where only few (zero to a few dozen) data points are needed to extend language model applications to downstream classification tasks. Examples of such regimes include T-few, GPT-3, and ADAPET, but these methods sometimes require large, inaccessible compute and finicky manually crafted prompts. Setfit performs on-par or better than comparable models while being prompt-free and only requiring small models that can fit on commercial personal computers. To promote wide access we make our code and data available open-source. 


<kbd>
<img src="assets/incremental_graph.png">
</kbd>

## SetFit’s Few-Shot Performance
While run prompt-free and on much smaller base models, SetFit performs on par or better than state of the art few-shot regimes on a variety of benchmarks. On the RAFT, a few-shot benchmark dataset, (https://huggingface.co/spaces/ought/raft-leaderboard)  as of September 2022, SetFit Roberta (with the Roberta-Large ST base model) with 355 million parameters outforms PET and GPT-3 and comes just under average human performance and the 11 billion parameter T-few a model 30 times the size of SetFit Roberta. Footnote (There is no information on state of the art on the RAFT.) SetFit outperforms the human baseline on 7 of the 11 RAFT tasks. 

| Rank | Method | Accuracy | Model Size | 
| :------: | ------ | :------: | :------: | 
| 1 | YiWise | 76.8 | N/A |
| 2 | T-Few | 75.8 | 11B | 
| 4 | Human Baseline | 73.5 | N/A | 
| 6 | SetFit (Roberta Large) | 71.3 | 355M |
| 9 | PET | 69.6 | 235M |
| 11 | SetFit (MP-Net) | 66.9 | 110M |
| 12 | GPT-3 | 62.7 | 175 B |


On non-few-shot datasets, SetFit shows robustness across a variety of tasks. It outperforms PERFECT, T-FEW 3 billion, ADAPET and vanilla transformers, on most tasks on sentiment, emotion, counterfactual, and unwanted language classification at very few (n=8) and few (n=64) -shot learning situations. 

| Method | SST-5 | AmazonCF | SentEval | Emotion | EnronSpam | AGNews | Average |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| Finetune | 33.5 (2.1) | 9.2 (4.9) | 58.8 (6.3) | 28.7 (6.8) | 85.0 (6.0) | 81.7 (3.8) | 49.5 (5.0)|
| ADAPET | 42.1 (3.7) | 14.8 (10.3) | 88.9 (1.3) | 46.5 (13.5) | 75.1 (5.0) | 74.3 (15.0) | 57.0 (8.1)|
| PERFECT | 34.9 (3.1) | 18.1 (5.3) | 81.5 (8.6) | 26.4  | 75.9 | 64.2 | NA |
|T-Few 3B | NA | 19.0 (3.9) | NA | 57.5 (1.8) | 93.1 (1.6) | NA | NA | 
|SetFit| 43.6(3.0) | 40.3 (11.8) | 88.5 (1.9) | 48.8 (4.5) | 90.1 (3.4) | 82.9 (2.8) | 65.7 (4.6) | 

| Method | SST-5 | AmazonCF | SentEval | Emotion | EnronSpam | AGNews | Average |
| ---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| Finetune | 45.9 (6.9) | 52.8 (12.1) | 88.9 (1.9) | 65.0 (17.2) | 95.9 (0.8) | 88.4 (0.9) | 72.8 (6.6)|
| ADAPET | 49.9 (1.5) | 56.4 (14.3) | 88.5 (7.4) | 65.0 (17.2) | 93.0 (1.4) | 85.3 (10.4) | 74.9 (6.1)|
| PERFECT | 49.1 (0.7) | 65.1 (5.2) | 91.4  | 66.8  | 92.3 | 88.7 | NA |
|T-Few 3B | NA | 34.7 (4.5) | NA | 71.0 (1.1) | 97.0 (0.3) | NA | NA | 
|SetFit| 51.9 (0.6) | 61.9 (2.9) | 90.4 (0.6) | 76.2 (1.3) | 96.1 (0.8) | 88.0 (0.7) | 77.4 (1.2) | 




And just by switching out the base ST model to a multilingual one, SetFit can function seamlessly in multilingual contexts. In our experiments, SetFit’s performance shows promising results on classification in German, Japanese, Mandarin, French and Spanish, in both in-language and cross linguistic settings.

## What is SetFit?

The strength of SetFit is its efficiency and simplicity. SetFit first finetunes a Sentence Transformer (ST) model then trains a classifier head on the generated ST embeddings. 


SetFit takes advantage of sentence transformers’ ability to generate dense embeddings based on paired sentences. In the data input stage, it maximizes the limited labeled input data by contrastive training, where positive and negative pairs are created by in-class and out-class selection. The sentence transformer model then trains on these pairs (or triplets) and generates dense vectors per example. This is SetFit’s fine-tuning step. In the second step, the classification head, such as a logistic regression model, trains on the encoded embeddings with their respective class labels. At inference time, the unseen example passes through the fine-tuned ST, generating an embedding that when fed to the classification head outputs a class label prediction.

## How to use SetFit

```sh
pip install setfit
```
