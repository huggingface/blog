---
title: "Leveraging Hugging Face for complex text classification use cases"
thumbnail:blog/assets/78_ml_director_insights/blogthumbnail.png
authors:
- user: VioletteLepercq
- user: juliensimon
- user: florentgbelidji
- user: lnazarenko
- user: lsmith77

---
<html>
<head>
<style>
.grandmahugs {
  margin: 25px;
}
</style>
<h1>Leveraging Hugging Face for complex text classification use cases </h1>
<h2>The Success Story of Witty Works with the Hugging Face Expert Acceleration Program Customer.</h2>
<!-- {blog_metadata} -->
<!-- {authors} -->
</head>
<body>

_If you're interested in building ML solutions faster visit: [hf.co/support](https://huggingface.co/support) today!_

<img class="grandmahugs" style="float: left;" padding="5px" width="200" src="/blog/assets/78_ml_director_insights/Javier.png"></a>

### Business Context
As IT continues to evolve and reshape our world, creating a more diverse and inclusive environment within the industry is imperative. [Witty Works](https://www.witty.works/) was built in 2018 to address this challenge. Starting as a consulting company advising organizations on how to become more diverse, Witty Works first helped them write job ads using inclusive language. To scale this effort, in 2019 they built an app to assist users in writing inclusive job ads in English, French and German. They enlarged the scope rapidly with a writing assistant working as a browser extension which automatically fixes and explains potential bias in emails, Linkedin posts, job ads, etc. The aim was to offer a solution for both internal and external communication that fosters a cultural change by also providing micro-learning bites that explain the underlying bias of highlighted words and phrases.

<p align="center">
    <img src="blog/blob/main/assets/78_ml_director_insights/wittyworks.png"><br>
    <em>Example of suggestions by the writing assistant</em>
</p>

### First experiments 
Witty Works first chose a basic machine learning approach to build their assistant from scratch. Using transfer learning with pre-trained spaCy models, the assistant analyzed text and transformed words to lemmas for linguistic analysis, named entity recognition, etc. By detecting and filtering words according to a specific knowledge base, the assistant could highlight non-inclusive words and suggest alternatives. 

### Challenge
This basic approach worked very well for 85% of the words, but it often failed for context-dependent non-inclusive words.

```diff
Example of context dependent non-inclusive words: 
  Fossil fuels are not renewable resources. Vs He is an old fossil
  You will have a flexible schedule. Vs You should keep your schedule flexible.
```

### Solutions provided by the [Hugging Face Experts](https://huggingface.co/support)

- #### **Get guidance for deciding on the right ML approach.**
The Hugging Face Expert recommended that Witty Work switch from using word embeddings to contextualized embeddings. In this approach, the representation of each word in a sentence depends on its surrounding context. Then, Hugging Face Experts suggested the use of a [Sentence Transformers](https://www.sbert.net/) model to create a contextual embedding representation for each word within a sentence, and combines them to generate a unique numerical representation for the entire sentence. The resulting sentence embedding serves as input for an inclusivity classifier.

```diff
Elena Nazarenko, Lead Data Scientist at Witty Works: “We generate contextualized embedding vectors for every word depending on its sentence (BERT embedding). Then, we keep only the embedding for the “problem” word’s token, and calculate the smallest angle (cosine similarity).” 
```

To fine-tune a transformers-based classifier, such as a simple BERT model, Witty Works would have needed a substantial amount of annotated data. hundreds of samples for each category of flagged words would have been necessary. However, such an annotation process would have been costly and time-consuming, which Witty Works couldn’t afford. 

- #### **Get guidance on selecting the right ML library.**
The Hugging Face Expert suggested using the Sentence Transformers Fine-tuning library (aka [SetFit](https://github.com/huggingface/setfit)), an efficient framework for few-shot fine-tuning of Sentence Transformers models. Combining contrastive learning and sentence semantic similarity, SetFit achieves high accuracy on text classification tasks with very little labeled data. 

```diff
Julien Simon, Chief Evangelist at Hugging Face: “SetFit for text classification tasks is a great tool to add to the ML toolbox.” 
```

The Witty Works team found the performance was more than adequate with as little as 15 labeled examples per category of words.

```diff
Elena Nazarenko, Lead Data Scientist at Witty Works: “At the end of the day, we saved time and money by not creating this large data set.”
```

Reducing the number of sentences was important to ensure that model training remained fast and running the model is efficient. However, it is also important for another reason: Witty explicitly takes a highly supervised/rule based approach in order to [actively manage bias](https://www.witty.works/en/blog/is-chatgpt-able-to-generate-inclusive-language). This means reducing the number of sentences was also very important to reduce the effort in manually reviewing the training sentences.

- #### **Get guidance on selecting the right ML models.**
One major challenge for Witty Works was to deploy a model with very low latency. No one is expecting to wait for 3 minutes to get suggestions to improve one’s text! Both Hugging Face and Witty Works experimented with a few sentence transformers models and settled for [mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) combined with logistic regression and KNN. 
After a first test on Google Colab, the Hugging Face experts guided Witty Works on deploying the model on Azure. No optimization was necessary as the model was fast enough.

```diff
Elena Nazarenko, Lead Data Scientist at Witty Works: “Working with Hugging Face saved us a lot of time and money. One can feel lost when implementing complex text classification use cases. As it is one of the most popular tasks, there are a lot of models on the Hub. The Hugging Face experts guided me through the massive amount of transformer-based models to choose the best possible approach. Plus, I felt very well supported during the model deployment.”
```
  
- #### **Results and conclusion.**
The number of training sentences dropped from 100-200 per word to 15-20 per word. Witty Works achieved an accuracy of 0.92 and successfully deployed a custom model on Azure with minimal DevOps effort!

```diff
Lukas Kahwe Smith CTO & Co-founder of Witty Works: “Working on an IT project by oneself can be challenging and even if the EAP is a significant investment for a startup, it is the cheaper and most meaningful way to get a sparring partner.“
```

With the guidance of the Hugging Face experts, Witty Works saved time and money by implementing a new ML workflow in the Hugging Face way.

```diff
Julien Simon, Chief Evangelist at Hugging Face: “The Hugging way to build workflows: find open-source pre-trained models, evaluate them right away, see what works, see what does not. By iterating, you start learning  things immediately.” 
```
---

🤗   If you or your team are interested in accelerating your ML roadmap with Hugging Face Experts, please visit [hf.co/support](https://huggingface.co/support) to learn more.

</body>
</html>

