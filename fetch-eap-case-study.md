---
title: "Fetch Consolidates AI Tools and Saves 30% Development Time with Hugging Face on AWS"
thumbnail: /blog/assets/78_ml_director_insights/fetch2.png
authors:
- user: Violette
---

# Fetch Consolidates AI Tools and Saves 30% Development Time with Hugging Face on AWS

_If you need support in using Hugging Face and AWS, please get in touch with us [here](https://huggingface.co/support#form) - our team will contact you to discuss your requirements!_

## Executive Summary

Fetch, a consumer rewards company, developed about 15 different artificial intelligence (AI) tools to help it receive, route, read, process, analyze, and store receipts uploaded by users. The company has more than 18 million active monthly users for its shopping rewards app. Fetch wanted to rebuild its AI-powered platform and, using Amazon Web Services (AWS) and with the support of AWS Partner Hugging Face, moved from using third-party applications to developing its own tools to gain better insights about customers. Consumers scan receipts—or forward electronic receipts—to receive rewards points for their purchases. Businesses can offer special rewards to users, such as extra points for purchasing a particular product. The company can now process more than 11 million receipts per day faster and gets better data.

## Fetch Needed a Scalable Way to Train AI Faster

[Fetch](https://fetch.com/)—formerly Fetch Rewards—has grown since its founding to serve 18 million active users every month who scan 11 million receipts every day to earn reward points. Users simply take a picture of their receipt and upload it using the company’s app. Users can also upload electronic receipts. Receipts earn points; if the receipt is from a brand partner of Fetch, it may qualify for promotions that award additional points. Those points can be redeemed for gift cards from a number of partners. But scanning is just the beginning. Once Fetch receives the receipts, it must process them, extracting data and analytics and filing the data and the receipts. It has been using artificial intelligence (AI) tools running on AWS to do that.

The company was using an AI solution from a third party to process receipts, but found it wasn’t getting the data insights it needed. Fetch’s business partners wanted information about how customers were engaging with their promotions, and Fetch didn’t have the granularity it needed to extract and process data from millions of receipts daily. “Fetch was using a third-party provider for its brain, which is scanning receipts, but scanning is not enough,” says Boris Kogan, computer vision scientist at Fetch. “That solution was a black box and we had no control or insight into what it did. We just got results we had to accept. We couldn’t give our business partners the information they wanted.”

Kogan joined Fetch tasked with the job of building thorough machine learning (ML) and AI expertise into the company and giving it full access to all aspects of the data it was receiving. To do this, he hired a team of engineers to bring his vision to life. “All of our infrastructure runs on AWS, we also rely on the AWS products to train our models,” says Kogan. “When the team started working on creating a brain of our own, of course, we first had to train our models and we did that on AWS. We allocated 12 months for the project and completed it in 8 month because we always had the resources we needed.”

#### "Hugging Face has democratized transformer models, models that were nearly impossible to train, and made them available to anyone. We couldn’t have done this without them.”

Sam Corzine, Machine Learning Engineer, Fetch


## Hugging Face Opens Up the Black Box

The Fetch team engaged with [AWS Partner](https://partners.amazonaws.com/partners/0010h00001jBrjVAAS/Hugging%20Face%20Inc) [Hugging Face](https://huggingface.co/) through the [Hugging Face Expert Acceleration Program](https://aws.amazon.com/marketplace/pp/prodview-z6gp22wkcvdt2/) on the AWS Marketplace to help Fetch unlock new tools to power processes after the scans had been uploaded. Hugging Face is a leader in open-source AI and provides guidance to enterprises on using AI. Many enterprises, including Fetch, use transformers from Hugging Face, which allow users to train and deploy open-source ML models in minutes. “Transformers is something that started with Hugging Face, and they're great at that,” says Kogan. The Fetch and Hugging Face teams worked to identify and train state-of-the-art document AI models, improving entity resolution and semantic search.

In this relationship, Hugging Face acted in an advisory capacity, transferring knowledge to help the Fetch engineers use its resources more effectively. “Fetch had a great team in place,” says Yifeng Yin, machine learning engineer at Hugging Face. “They didn't need us to come in and run the project or build it. They wanted to learn how to use Hugging Face to train the models they were building. We showed them how to use the resources, and they ran with it.” With Yifeng’s guidance, Fetch was able to cut its development time by 30 percent.

Because it was building its own AI and ML models to take over from the third-party ‘brain’, it needed to ensure a robust system that produced good results before switching over. Fetch required doing this without interrupting the flow of millions of receipts every day. “Before we rolled anything out, we built a shadow pipeline,” says Sam Corzine, lead machine learning engineer at Fetch. “We took all the things and reprocessed them in our new ML pipeline. We could do audits of everything. It was running full volume, reprocessing all of those 11 million receipts and doing analytics on them for quite a while before anything made it into the main data fields. The black box was still running the show and we were checking our results against it.” The solution uses [Amazon SageMaker](https://aws.amazon.com/sagemaker/)—which lets businesses build, train, and deploy ML models for any use case with fully managed infrastructure, tools, and workflows. It also uses [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) accelerators to deliver high performance at the lowest cost for deep learning (DL) inference applications.

## Fetch Grows AI Expertise, Cuts Latency by 50%, and Saves Costs

Fetch’s commitment to developing in-house ML and AI capabilities has resulted in several benefits, including some cost savings, but more important is the development of a service that better serves the needs of the customers. “With any app you have to give the customer a reason to keep coming back,” says Corzine. “We’ve improved responsiveness for customers with faster processing of uploads, cutting processing latency by 50 percent. If you keep customers waiting too long, they’ll disengage. And the more customers use Fetch, the better understanding we and our partners get about what’s important to them. By building our own models, we get details we never had before.”

The company can now train a model in hours instead of the days or weeks it used to take. Development time has also been reduced by about 30 percent. And while it may not be possible to put a number to it, another major benefit has been creating a more stable foundation for Fetch. “Relying on a third-party black box presented considerable business risk to us,” says Corzine. “Because Hugging Face existed and its community existed, we were able to use that tooling and work with that community. At the end of the day, we now control our destiny.”

Fetch is continuing to improve the service to customers and gain a better understanding of customer behavior now that it is an AI-first company, rather than a company that uses a third-party AI ‘brain’. “Hugging Face and AWS gave us the infrastructure and the resources to do what we need,” says Kogan. “Hugging Face has democratized transformer models, models that were nearly impossible to train, and made them available to anyone. We couldn’t have done this without them.”


_This article is a cross-post from an originally published post on February 2024 [on AWS's website](https://aws.amazon.com/fr/partners/success/fetch-hugging-face/)._
