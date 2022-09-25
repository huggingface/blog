---
title: "SetFit: Efficient Few-Shot Learning Without Prompts"
thumbnail: /blog/assets/103_setfit/intel_hf_logo.png
---
<div class="blog-metadata">
    <small>Published September 24, 2022.</small>
   
<h1>SetFit: Efficient Few-Shot Learning Without Prompts</h1>

</div>
<div class="author-card">
    <a href="/Unso">
        <img class="avatar avatar-user" src="https://scholar.googleusercontent.com/citations?view_op=medium_photo&user=-I2EZeEAAAAJ&citpid=8" title="Gravatar">
        <div class="bfc">
            <code>Unso</code>
            <span class="fullname">Unso Eun Seo Jo</span>
        </div>
    </a>
</div>


Few-shot learning with pretrained language models has emerged as a promising solution to every data scientist's nightmare: dealing with data that has few to no labels 😱.
Together with our research partners at [Intel Labs](https://www.intel.com/content/www/us/en/research/overview.html) and the [UKP Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp), we are excited to introduce SetFit: an efficient framework for few-shot fine-tuning of Sentence Transformers.
Compared to other few-shot learning methods, SetFit has several unique features:

<p>📈 <strong>High accuracy with little labeled data</strong>: SetFit achieves comparable (or better) results than current state-of-the-art methods for text classification. For example, with only 8 labelled examples per class on the CR sentiment dataset, SetFit is competitive with fine-tuning RoBERTa-large on the full training set of 3k examples. </p>

<p>🗣 <strong>No prompts or verbalisers</strong>: Current techniques for few-shot fine-tuning require handcrafted prompts or verbalisers to convert examples into a format that's suitable for the underlying language model. SetFit dispenses with prompts altogether by generating rich embeddings directly from text examples. </p>

<p>🏎 <strong>Fast to train</strong>: SetFit doesn't require large-scale models like T0 or GPT-3 to achieve high accuracy. As a result, it is typically an order of magnitude (or more) faster to train and run inference with. </p>

<p>🌎 <strong>Multilingual support</strong>: SetFit can be used with any Sentence Transformer on the Hub, which means you can classify text in multiple languages by simply switching the backbone to a multilingual checkpoint. </p>

For more details, you can check out our [paper](https://arxiv.org/abs/2209.11055), [data](https://huggingface.co/SetFit) and [code](https://github.com/huggingface/setfit). In this blog post, we'll explain how SetFit works and how to train your very own models. Let's dive in!

<p align="center">
    <img src="assets/103_setfit/setfit_curves.png" width=400>
</p>
<p align="center">
    Fig.1: SetFit and Fine-tuning performance on SentEval test set, a sentiment classification task
</p>

## How does it work?

SetFit is designed with efficiency and simplicity in mind. SetFit first finetunes a Sentence Transformer model then trains a classifier head on the embeddings generated from the finetuned Sentence Transformer. 

<p align="center">
    <img src="assets/103_setfit/setfit_diagram_process.png" width=700>
</p>
<p align="center">
    Fig.2: SetFit's 2-Part Process
</p>

SetFit takes advantage of sentence transformers’ ability to generate dense embeddings based on paired sentences. In the data input stage, it maximizes the limited labeled input data by contrastive training, where positive and negative pairs are created by in-class and out-class selection. The Sentence Transformer model then trains on these pairs (or triplets) and generates dense vectors per example. This is SetFit’s fine-tuning step. In the second step, the classification head, such as a logistic regression model, trains on the encoded embeddings with their respective class labels. At inference time, the unseen example passes through the fine-tuned Sentence Transformer, generating an embedding that when fed to the classification head outputs a class label prediction.

And just by switching out the base Sentence Transformer model to a multilingual one, SetFit can function seamlessly in multilingual contexts. In our experiments, SetFit’s performance shows promising results on classification in German, Japanese, Mandarin, French and Spanish, in both in-language and cross linguistic settings.


## Benchmarking SetFit

| Rank | Method | Accuracy | Model Size | 
| :------: | ------ | :------: | :------: | 
| 2 | T-Few | 75.8 | 11B | 
| 4 | Human Baseline | 73.5 | N/A | 
| 6 | SetFit (Roberta Large) | 71.3 | 355M |
| 9 | PET | 69.6 | 235M |
| 11 | SetFit (MP-Net) | 66.9 | 110M |
| 12 | GPT-3 | 62.7 | 175 B |

<h4>Table 1: RAFT performance leaderboard as of September 2022</h4>


Although based on much smaller models than existing few-shot methods, SetFit performs on par or better than state of the art few-shot regimes on a variety of benchmarks. On [RAFT](https://huggingface.co/spaces/ought/raft-leaderboard), a few-shot benchmark dataset as of September 2022, SetFit Roberta (using the Roberta-Large Sentence Transformer base model) with 355 million parameters outforms PET and GPT-3 and places just under average human performance and the 11 billion parameter T-few, a model 30 times the size of SetFit Roberta (Table 1). SetFit also outperforms the human baseline on 7 of the 11 RAFT tasks.



<p align="center">
    <img src="assets/103_setfit/three-tasks.png" width=700>
</p>
<p align="center">
    Fig.3: Comparing Setfit performance against other methods on 3 tasks
</p>



On other datasets, SetFit shows robustness across a variety of tasks. It outperforms PERFECT, T-Few 3B, ADAPET and fine-tuned vanilla transformers, on many tasks on sentiment, emotion, counterfactual, and unwanted language classification tasks at very few (n=8) and few (n=64) - shot learning scenarios. Figure 3 shows a comparison on three chosen tasks: Emotion, SentEval-CR, and Amazon Counterfactual. 





## Fast training and inference

<p align="center">
    <img src="assets/103_setfit/bars.png" width=400>
</p>
<p align="center">
    Fig.4: Comparing T-Few 3B and SetFit (MP-Net) train cost on AWS for N=8 and average accuracy across test tasks
</p>

Since SetFit achieves high accuracy with relatively small models, this makes it blazing fast to train on a single GPU like the ones found on Google Colab (in fact you can train SetFit on CPU in just a few minutes!). For instance, training SetFit on an NVIDIA V100 with 8 labeled examples takes just 30 seconds, at a cost of $0.025. By comparison, training T-Few 3B requires an NVIDIA A100 and takes 11 minutes, at a cost of around $0.7 for the same experiment - a factor of 28x more. As shown in Figure 4, this speed-up comes with comparable model performance. Similar gains are also achieved for inference - as we show in the paper, distilling the SetFit model can bring speed-ups of 123x 🤯.




## Training your own model

Using SetFit is as simple as just a few lines of code. There is need for hyperparameter searching or prompt-engineering.
s
To start using SetFit, first install it using pip. This will also install dependencies such as datasets and sentence-transformers. 
```sh
pip install setfit
```
We first import relevant functions. In particular we import SetFitModel and SetFitTrainer, two functions that have streamlined the SetFit procedure for us.
```sh
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit.modeling import SetFitModel
from setfit.trainer import SetFitTrainer
```
We download our dataset from the HF hub. In this case we start with the SentEval-CR task, the performance of which is shown in our Fig.1. 
```sh
# Load a dataset
dataset = load_dataset("SetFit/SentEval-CR")
```
Let's now select our N, the number of examples per class. In this case we make this 8. N will vary depending on your dataset constraints. 
```sh
# Select N examples per class (8 in this case)
train_ds = dataset["train"].shuffle(seed=42).select(range(8 * 2))
test_ds = dataset["test"]
```
We will load a pretrained Sentence Transformer model from the HF hub then create a SetFitTrainer. 
```sh
# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_epochs=20,
)
```
Training this is easy! We can then evaluate with train.evaluate().
```sh
# Train and evaluate!
trainer.train()
metrics = trainer.evaluate()
```

Remember to push your trained model to the HF hub :) 
```sh
# Push model to the Hub
# Make sure you're logged in with huggingface-cli login first
trainer.push_to_hub("my-awesome-setfit-model")
```
While this example showed how this can be done with one specific type of base model, any Sentence Transformer base model could be traded in here for different performance and tasks. For intance, using a multilingual Sentence Transformer base can extend few-shot to multilingual settings.























