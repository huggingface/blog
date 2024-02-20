---
title: "Introducing the Private Hub: A New Way to Build With Machine Learning"
thumbnail: /blog/assets/92_introducing_private_hub/thumbnail.png
authors:
- user: federicopascual
---

# Introducing the Private Hub: A New Way to Build With Machine Learning


<script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>

<br>
<div style="background-color: #e6f9e6; padding: 16px 32px; outline: 2px solid; border-radius: 10px;">
  June 2023 Update:
  
  The Private Hub is now called <b>Enterprise Hub</b>.
  
  The Enterprise Hub is a hosted solution that combines the best of Cloud Managed services (SaaS) and Enterprise security. It lets customers deploy specific services like <b>Inference Endpoints</b> on a wide scope of compute options, from on-cloud to on-prem. It offers advanced user administration and access controls through SSO. 

<code style="color: red">We no longer offer Private Hub on-prem deployments as this experiment is now discontinued.</code> 
  
  Get in touch with our [Enterprise team](/support) to find the best solution for your company.
</div>

Machine learning is changing how companies are building technology. From powering a new generation of disruptive products to enabling smarter features in well-known applications we all use and love, ML is at the core of the development process.

But with every technology shift comes new challenges. 

Around [90% of machine learning models never make it into production](https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/). Unfamiliar tools and non-standard workflows slow down ML development. Efforts get duplicated as models and datasets aren't shared internally, and similar artifacts are built from scratch across teams all the time. Data scientists find it hard to show their technical work to business stakeholders, who struggle to share precise and timely feedback. And machine learning teams waste time on Docker/Kubernetes and optimizing models for production. 

With this in mind, we launched the [Private Hub](https://huggingface.co/platform) (PH), a new way to build with machine learning. From research to production, it provides a unified set of tools to accelerate each step of the machine learning lifecycle in a secure and compliant way. PH brings various ML tools together in one place, making collaborating in machine learning simpler, more fun and productive.

In this blog post, we will deep dive into what is the Private Hub, why it's useful, and how customers are accelerating their ML roadmaps with it.

Read along or feel free to jump to the section that sparks 🌟 your interest:

1. [What is the Hugging Face Hub?](#1-what-is-the-hugging-face-hub)
2. [What is the Private Hub?](#2-what-is-the-private-hub)
3. [How are companies using the Private Hub to accelerate their ML roadmap?](#3-how-are-companies-using-the-private-hub-to-accelerate-their-ml-roadmap)

Let's get started! 🚀

## 1. What is the Hugging Face Hub?

Before diving into the Private Hub, let's first take a look at the Hugging Face Hub, which is a central part of the PH.

The [Hugging Face Hub](https://huggingface.co/docs/hub/index) offers over 60K models, 6K datasets, and 6K ML demo apps, all open source and publicly available, in an online platform where people can easily collaborate and build ML together. The Hub works as a central place where anyone can explore, experiment, collaborate and build technology with machine learning.

On the Hugging Face Hub, you’ll be able to create or discover the following ML assets:

- [Models](https://huggingface.co/models): hosting the latest state-of-the-art models for NLP, computer vision, speech, time-series, biology, reinforcement learning, chemistry and more.
- [Datasets](https://huggingface.co/datasets): featuring a wide variety of data for different domains, modalities and languages.
- [Spaces](https://huggingface.co/spaces): interactive apps for showcasing ML models directly in your browser.

Each model, dataset or space uploaded to the Hub is a [Git-based repository](https://huggingface.co/docs/hub/repositories), which are version-controlled places that can contain all your files. You can use the traditional git commands to pull, push, clone, and/or manipulate your files. You can see the commit history for your models, datasets and spaces, and see who did what and when.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Commit history on a machine learning model" src="assets/92_introducing_private_hub/commit-history.png"></medium-zoom>
  <figcaption>Commit history on a model</figcaption>
</figure>

The Hugging Face Hub is also a central place for feedback and development in machine learning. Teams use [pull requests and discussions](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to support peer reviews on models, datasets, and spaces, improve collaboration and accelerate their ML work.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Pull requests and discussions on a model" src="assets/92_introducing_private_hub/pull-requests-and-discussions.png"></medium-zoom>
  <figcaption>Pull requests and discussions on a model</figcaption>
</figure>

The Hub allows users to create [Organizations](https://huggingface.co/docs/hub/organizations), that is, team accounts to manage models, datasets, and spaces collaboratively. An organization’s repositories will be featured on the organization’s page and admins can set roles to control access to these repositories. Every member of the organization can contribute to models, datasets and spaces given the right permissions. Here at Hugging Face, we believe having the right tools to collaborate drastically accelerates machine learning development! 🔥 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Organization in the Hub for BigScience" src="assets/92_introducing_private_hub/organizations.png"></medium-zoom>
  <figcaption>Organization in the Hub for <a href="https://huggingface.co/bigscience">BigScience</a></figcaption>
</figure>

Now that we have covered the basics, let's dive into the specific characteristics of models, datasets and spaces hosted on the Hugging Face Hub.

### Models

[Transfer learning](https://www.youtube.com/watch?v=BqqfQnyjmgg&ab_channel=HuggingFace) has changed the way companies approach machine learning problems. Traditionally, companies needed to train models from scratch, which requires a lot of time, data, and resources. Now machine learning teams can use a pre-trained model and [fine-tune it for their own use case](https://huggingface.co/course/chapter3/1?fw=pt) in a fast and cost-effective way. This dramatically accelerates the process of getting accurate and performant models.

On the Hub, you can find 60,000+ state-of-the-art open source pre-trained models for NLP, computer vision, speech, time-series, biology, reinforcement learning, chemistry and more. You can use the search bar or filter by tasks, libraries, licenses and other tags to find the right model for your particular use case:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="60,000+ models available on the Hub" src="assets/92_introducing_private_hub/models.png"></medium-zoom>
  <figcaption>60,000+ models available on the Hub</figcaption>
</figure>

These models span 180 languages and support up to 25 ML libraries (including Transformers, Keras, spaCy, Timm and others), so there is a lot of flexibility in terms of the type of models, languages and libraries.

Each model has a [model card](https://huggingface.co/docs/hub/models-cards), a simple markdown file with a description of the model itself. This includes what it's intended for, what data that model has been trained on, code samples, information on potential bias and potential risks associated with the model, metrics, related research papers, you name it. Model cards are a great way to understand what the model is about, but they also are useful for identifying the right pre-trained model as a starting point for your ML project:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Model card" src="assets/92_introducing_private_hub/model-card.png"></medium-zoom>
  <figcaption>Model card</figcaption>
</figure>

Besides improving models' discoverability and reusability, model cards also make it easier for model risk management (MRM) processes. ML teams are often required to provide information about the machine learning models they build so compliance teams can identify, measure and mitigate model risks. Through model cards, organizations can set up a template with all the required information and streamline the MRM conversations between the ML and compliance teams right within the models.

The Hub also provides an [Inference Widget](https://huggingface.co/docs/hub/models-widgets) to easily test models right from your browser! It's a really good way to get a feeling if a particular model is a good fit and something you wanna dive into:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Inference widget" src="assets/92_introducing_private_hub/inference-widget.png"></medium-zoom>
  <figcaption>Inference widget</figcaption>
</figure>

### Datasets

Data is a key part of building machine learning models; without the right data, you won't get accurate models. The 🤗 Hub hosts more than [6,000 open source, ready-to-use datasets for ML models](https://huggingface.co/datasets) with fast, easy-to-use and efficient data manipulation tools. Like with models, you can find the right dataset for your use case by using the search bar or filtering by tags. For example, you can easily find 96 models for sentiment analysis by filtering by the task "sentiment-classification":

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Datasets available for sentiment classification" src="assets/92_introducing_private_hub/filtering-datasets.png"></medium-zoom>
  <figcaption>Datasets available for sentiment classification</figcaption>
</figure>

Similar to models, datasets uploaded to the 🤗 Hub have [Dataset Cards](https://huggingface.co/docs/hub/datasets-cards#dataset-cards) to help users understand the contents of the dataset, how the dataset should be used, how it was created and know relevant considerations for using the dataset. You can use the [Dataset Viewer](https://huggingface.co/docs/hub/datasets-viewer) to easily view the data and quickly understand if a particular dataset is useful for your machine learning project:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Super Glue dataset preview" src="assets/92_introducing_private_hub/dataset-preview.png"></medium-zoom>
  <figcaption>Super Glue dataset preview</figcaption>
</figure>

### Spaces

A few months ago, we introduced a new feature on the 🤗 Hub called [Spaces](https://huggingface.co/spaces/launch). It's a simple way to build and host machine learning apps. Spaces allow you to easily showcase your ML models to business stakeholders and get the feedback you need to move your ML project forward. 

If you've been generating funny images with [DALL-E mini](https://huggingface.co/spaces/dalle-mini/dalle-mini), then you have used Spaces. This space showcase the [DALL-E mini model](https://huggingface.co/dalle-mini/dalle-mini), a machine learning model to generate images based on text prompts:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Space for DALL-E mini" src="assets/92_introducing_private_hub/dalle-mini.png"></medium-zoom>
  <figcaption>Space for DALL-E mini</figcaption>
</figure>

## 2. What is the Private Hub?

The [Private Hub](https://huggingface.co/platform) allows companies to use Hugging Face’s complete ecosystem in their own private and compliant environment to accelerate their machine learning development. It brings ML tools for every step of the ML lifecycle together in one place to make collaborating in ML simpler and more productive, while having a compliant environment that companies need for building ML securely:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="The Private Hub" src="assets/92_introducing_private_hub/private-hub.png"></medium-zoom>
  <figcaption>The Private Hub</figcaption>
</figure>

With the Private Hub, data scientists can seamlessly work with [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets) and other [open source libraries](https://github.com/huggingface) with models, datasets and spaces privately and securely hosted on your own servers, and get machine learning done faster by leveraging the Hub features:

- [AutoTrain](https://huggingface.co/autotrain): you can use our AutoML no-code solution to train state-of-the-art models, automatically fine-tuned, evaluated and deployed in your own servers.
- [Evaluate](https://huggingface.co/spaces/autoevaluate/model-evaluator): evaluate any model on any dataset on the Private Hub with any metric without writing a single line of code.
- [Spaces](https://huggingface.co/spaces/launch): easily host an ML demo app to show your ML work to business stakeholders, get feedback early and build faster.
- [Inference API](https://huggingface.co/inference-api): every private model created on the Private Hub is deployed for inference in your own infrastructure via simple API calls.
- [PRs and Discussions](https://huggingface.co/blog/community-update): support peer reviews on models, datasets, and spaces to improve collaboration across teams.

From research to production, your data never leaves your servers. The Private Hub runs in your own compliant server. It provides enterprise security features like security scans, audit trail, SSO, and control access to keep your models and data secure.

We provide flexible options for deploying your Private Hub in your private, compliant environment, including:

- **Managed Private Hub (SaaS)**: runs in segregated virtual private servers (VPCs) owned by Hugging Face. You can enjoy the full Hugging Face experience on your own private Hub without having to manage any infrastructure.

- **On-cloud Private Hub**: runs in a cloud account on AWS, Azure or GCP owned by the customer. This deployment option gives you full administrative control of the underlying cloud infrastructure and lets you achieve stronger security and compliance.

- **On-prem Private Hub**: on-premise deployment of the Hugging Face Hub on your own infrastructure. For customers with strict compliance rules and/or workloads where they don't want or are not allowed to run on a public cloud.

Now that we have covered the basics of what the Private Hub is, let's go over how companies are using it to accelerate their ML development. 

## 3. How Are Companies Using the Private Hub to Accelerate Their ML Roadmap?

[🤗 Transformers](https://github.com/huggingface/transformers) is one of the [fastest growing open source projects of all time](https://star-history.com/#tensorflow/tensorflow&nodejs/node&kubernetes/kubernetes&pytorch/pytorch&huggingface/transformers&Timeline). We now offer [25+ open source libraries](https://github.com/huggingface) and over 10,000 companies are now using Hugging Face to build technology with machine learning.

Being at the heart of the open source AI community, we had thousands of conversations with machine learning and data science teams, giving us a unique perspective on the most common problems and challenges companies are facing when building machine learning. 

Through these conversations, we discovered that the current workflow for building machine learning is broken. Duplicated efforts, poor feedback loops, high friction to collaborate across teams, non-standard processes and tools, and difficulty optimizing models for production are common and slow down ML development.

We built the Private Hub to change this. Like Git and GitHub forever changed how companies build software, the Private Hub changes how companies build machine learning:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Before and after using The Private Hub" src="assets/92_introducing_private_hub/before-and-after.png"></medium-zoom>
  <figcaption>Before and after using The Private Hub</figcaption>
</figure>

In this section, we'll go through a demo example of how customers are leveraging the PH to accelerate their ML lifecycle. We will go over the step-by-step process of building an ML app to automatically analyze financial analyst 🏦 reports. 

First, we will search for a pre-trained model relevant to our use case and fine-tune it on a custom dataset for sentiment analysis. Next, we will build an ML web app to show how this model works to business stakeholders. Finally, we will use the Inference API to run inferences with an infrastructure that can handle production-level loads. All artifacts for this ML demo app can be found in this [organization on the Hub](https://huggingface.co/FinanceInc).

### Training accurate models faster

#### Leveraging a pre-trained model from the Hub

Instead of training models from scratch, transfer learning now allows you to build more accurate models 10x faster ⚡️by fine-tuning pre-trained models available on the Hub for your particular use case. 

For our demo example, one of the requirements for building this ML app for financial analysts is doing sentiment analysis. Business stakeholders want to automatically get a sense of a company's performance as soon as financial docs and analyst reports are available.

So as a first step towards creating this ML app, we dive into the [🤗 Hub](https://huggingface.co/models) and explore what pre-trained models are available that we can fine-tune for sentiment analysis. The search bar and tags will let us filter and discover relevant models very quickly. Soon enough, we come across [FinBERT](https://huggingface.co/yiyanghkust/finbert-pretrain), a BERT model pre-trained on corporate reports, earnings call transcripts and financial analyst reports: 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Finbert model" src="assets/92_introducing_private_hub/finbert-pretrain.png"></medium-zoom>
  <figcaption>Finbert model</figcaption>
</figure>

We [clone the model](https://huggingface.co/FinanceInc/finbert-pretrain) in our own Private Hub, so it's available to other teammates. We also add the required information to the model card to streamline the model risk management process with the compliance team.

#### Fine-tuning a pre-trained model with a custom dataset

Now that we have a great pre-trained model for financial data, the next step is to fine-tune it using our own data for doing sentiment analysis!  

So, we first upload a [custom dataset for sentiment analysis](https://huggingface.co/datasets/FinanceInc/auditor_sentiment) that we built internally with the team to our Private Hub. This dataset has several thousand sentences from financial news in English and proprietary financial data manually categorized by our team according to their sentiment. This data contains sensitive information, so our compliance team only allows us to upload this data on our own servers. Luckily, this is not an issue as we run the Private Hub on our own AWS instance.

Then, we use [AutoTrain](https://huggingface.co/autotrain) to quickly fine-tune the FinBert model with our custom sentiment analysis dataset. We can do this straight from the datasets page on our Private Hub:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Fine-tuning a pre-trained model with AutoTrain" src="assets/92_introducing_private_hub/train-in-autotrain.png"></medium-zoom>
  <figcaption>Fine-tuning a pre-trained model with AutoTrain</figcaption>
</figure>

Next, we select "manual" as the model choice and choose our [cloned Finbert model](https://huggingface.co/FinanceInc/finbert-pretrain) as the model to fine-tune with our dataset:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Creating a new project with AutoTrain" src="assets/92_introducing_private_hub/autotrain-new-project.png"></medium-zoom>
  <figcaption>Creating a new project with AutoTrain</figcaption>
</figure>

Finally, we select the number of candidate models to train with our data. We choose 25 models and voila! After a few minutes, AutoTrain has automatically fine-tuned 25 finbert models with our own sentiment analysis data, showing the performance metrics for all the different models 🔥🔥🔥

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="25 fine-tuned models with AutoTrain" src="assets/92_introducing_private_hub/autotrain-trained-models.png"></medium-zoom>
  <figcaption>25 fine-tuned models with AutoTrain</figcaption>
</figure>

Besides the performance metrics, we can easily test the [fine-tuned models](https://huggingface.co/FinanceInc/auditor_sentiment_finetuned) using the inference widget right from our browser to get a sense of how good they are:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Testing the fine-tuned models with the Inference Widget" src="assets/92_introducing_private_hub/auto-train-inference-widget.png"></medium-zoom>
  <figcaption>Testing the fine-tuned models with the Inference Widget</figcaption>
</figure>

### Easily demo models to relevant stakeholders

Now that we have trained our custom model for analyzing financial documents, as a next step, we want to build a machine learning demo with [Spaces](https://huggingface.co/spaces/launch) to validate our MVP with our business stakeholders. This demo app will use our custom sentiment analysis model, as well as a second FinBERT model we fine-tuned for [detecting forward-looking statements](https://huggingface.co/FinanceInc/finbert_fls) from financial reports. This interactive demo app will allow us to get feedback sooner, iterate faster, and improve the models so we can use them in production. ✅

In less than 20 minutes, we were able to build an [interactive demo app](https://huggingface.co/spaces/FinanceInc/Financial_Analyst_AI) that any business stakeholder can easily test right from their browsers:

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Space for our financial demo app" src="assets/92_introducing_private_hub/financial-analyst-space.png"></medium-zoom>
  <figcaption>Space for our financial demo app</figcaption>
</figure>

If you take a look at the [app.py file](https://huggingface.co/spaces/FinanceInc/Financial_Analyst_AI/blob/main/app.py), you'll see it's quite simple: 

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Code for our ML demo app" src="assets/92_introducing_private_hub/spaces-code.png"></medium-zoom>
  <figcaption>Code for our ML demo app</figcaption>
</figure>

51 lines of code are all it took to get this ML demo app up and running! 🤯

### Scale inferences while staying out of MLOps

By now, our business stakeholders have provided great feedback that allowed us to improve these models. Compliance teams assessed potential risks through the information provided via the model cards and green-lighted our project for production. Now, we are ready to put these models to work and start analyzing financial reports at scale! 🎉

Instead of wasting time on Docker/Kubernetes, setting up a server for running these models or optimizing models for production, all we need to do is to leverage the [Inference API](https://huggingface.co/inference-api). We don't need to worry about deployment or scalability issues, we can easily integrate our custom models via simple API calls. 

Models uploaded to the Hub and/or created with AutoTrain are instantly deployed to production, ready to make inferences at scale and in real-time. And all it takes to run inferences is 12 lines of code! 

To get the code snippet to run inferences with our [sentiment analysis model](https://huggingface.co/FinanceInc/auditor_sentiment_finetuned), we click on "Deploy" and "Accelerated Inference":

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Leveraging the Inference API to run inferences on our custom model" src="assets/92_introducing_private_hub/deploy.png"></medium-zoom>
  <figcaption>Leveraging the Inference API to run inferences on our custom model</figcaption>
</figure>

This will show us the following code to make HTTP requests to the Inference API and start analyzing data with our custom model:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/FinanceInc/auditor_sentiment_finetuned"
headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Operating profit jumped to EUR 47 million from EUR 6.6 million",
})
```

With just 12 lines of code, we are up and running in running inferences with an infrastructure that can handle production-level loads at scale and in real-time 🚀. Pretty cool, right? 

## Last Words

Machine learning is becoming the default way to build technology, mostly thanks to open-source and open-science.

But building machine learning is still hard. Many ML projects are rushed and never make it to production. ML development is slowed down by non-standard workflows. ML teams get frustrated with duplicated work, low collaboration across teams, and a fragmented ecosystem of ML tooling.

At Hugging Face, we believe there is a better way to build machine learning. And this is why we created the [Private Hub](https://huggingface.co/platform). We think that providing a unified set of tools for every step of the machine learning development and the right tools to collaborate will lead to better ML work, bring more ML solutions to production, and help ML teams spark innovation.

Interested in learning more? [Request a demo](https://huggingface.co/platform#form) to see how you can leverage the Private Hub to accelerate ML development within your organization.
