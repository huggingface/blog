---
title: "Fetch Cuts ML Processing Latency by 50% Using Amazon SageMaker & Hugging Face"
thumbnail: /blog/assets/78_ml_director_insights/fetch.png
authors:
- user: VioletteLepercq
---

# Fetch Cuts ML Processing Latency by 50% Using Amazon SageMaker & Hugging Face


_This article is a cross-post from an originally published post on September 2023 [on AWS's website](https://aws.amazon.com/fr/solutions/case-studies/fetch-case-study/)._


## Overview

Consumer engagement and rewards company [Fetch](https://fetch.com/) offers an application that lets users earn rewards on their purchases by scanning their receipts. The company also parses these receipts to generate insights into consumer behavior and provides those insights to brand partners. As weekly scans rapidly grew, Fetch needed to improve its speed and precision.

On Amazon Web Services (AWS), Fetch optimized its machine learning (ML) pipeline using Hugging Face and [Amazon SageMaker ](https://aws.amazon.com/sagemaker/), a service for building, training, and deploying ML models with fully managed infrastructure, tools, and workflows. Now, the Fetch app can process scans faster and with significantly higher accuracy.


## Opportunity | Using Amazon SageMaker to Accelerate an ML Pipeline in 12 Months for Fetch

Using the Fetch app, customers can scan receipts, receive points, and redeem those points for gift cards. To reward users for receipt scans instantaneously, Fetch needed to be able to capture text from a receipt, extract the pertinent data, and structure it so that the rest of its system can process and analyze it. With over 80 million receipts processed per week—hundreds of receipts per second at peak traffic—it needed to perform this process quickly, accurately, and at scale.

In 2021, Fetch set out to optimize its app’s scanning functionality. Fetch is an AWS-native company, and its ML operations team was already using Amazon SageMaker for many of its models. This made the decision to enhance its ML pipeline by migrating its models to Amazon SageMaker a straightforward one.

Throughout the project, Fetch had weekly calls with the AWS team and received support from a subject matter expert whom AWS paired with Fetch. The company built, trained, and deployed more than five ML models using Amazon SageMaker in 12 months. In late 2022, Fetch rolled out its updated mobile app and new ML pipeline.

#### "Amazon SageMaker is a game changer for Fetch. We use almost every feature extensively. As new features come out, they are immediately valuable. It’s hard to imagine having done this project without the features of Amazon SageMaker.”

Sam Corzine, Machine Learning Engineer, Fetch


## Solution | Cutting Latency by 50% Using ML & Hugging Face on Amazon SageMaker GPU Instances

#### "Using the flexibility of the Hugging Face AWS Deep Learning Container, we could improve the quality of our models,and Hugging Face’s partnership with AWS meant that it was simple to deploy these models.”

Sam Corzine, Machine Learning Engineer, Fetch


Fetch’s ML pipeline is powered by several Amazon SageMaker features, particularly [Amazon SageMaker Model Training](https://aws.amazon.com/sagemaker/train/), which reduces the time and cost to train and tune ML models at scale, and [Amazon SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html), a simplified, managed experience to run data-processing workloads. The company runs its custom ML models using multi-GPU instances for fast performance. “The GPU instances on Amazon SageMaker are simple to use,” says Ellen Light, backend engineer at Fetch. Fetch trains these models to identify and extract key information on receipts that the company can use to generate valuable insights and reward users. And on Amazon SageMaker, Fetch’s custom ML system is seamlessly scalable. “By using Amazon SageMaker, we have a simple way to scale up our systems, especially for inference and runtime,” says Sam Corzine, ML engineer at Fetch. Meanwhile, standardized model deployments mean less manual work.

Fetch heavily relied on the ML training features of Amazon SageMaker, particularly its [training jobs](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html), as it refined and iterated on its models. Fetch can also train ML models in parallel, which speeds up development and deployments. “There’s little friction for us to deploy models,” says Alec Stashevsky, applied scientist at Fetch. “Basically, we don’t have to think about it.” This has increased confidence and improved productivity for the entire company. In one example, a new intern was able to deploy a model himself by his third day on the job.

Since adopting Amazon SageMaker for ML tuning, training, and retraining, Fetch has enhanced the accuracy of its document-understanding model by 200 percent. It continues to fine-tune its models for further improvement. “Amazon SageMaker has been a key tool in building these outstanding models,” says Quency Yu, ML engineer at Fetch. To optimize the tuning process, Fetch relies on [Amazon SageMaker Inference Recommender](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-recommender.html), a capability of Amazon SageMaker that reduces the time required to get ML models in production by automating load testing and model tuning.  

In addition to its custom ML models, Fetch uses [AWS Deep Learning Containers ](https://aws.amazon.com/machine-learning/containers/)(AWS DL Containers), which businesses can use to quickly deploy deep learning environments with optimized, prepackaged container images. This simplifies the process of using libraries from [Hugging Face Inc.](https://huggingface.co/)(Hugging Face), an artificial intelligence technology company and [AWS  Partner](https://partners.amazonaws.com/partners/0010h00001jBrjVAAS/Hugging%20Face%20Inc.). Specifically, Fetch uses the Amazon SageMaker Hugging Face Inference Toolkit, an open-source library for serving transformers models, and the Hugging Face AWS Deep Learning Container for training and inference. “Using the flexibility of the Hugging Face AWS Deep Learning Container, we could improve the quality of our models,” says Corzine. “And Hugging Face’s partnership with AWS meant that it was simple to deploy these models.”

For every metric that Fetch measures, performance has improved since adopting Amazon SageMaker. The company has reduced latency for its slowest scans by 50 percent. “Our improved accuracy also creates confidence in our data among partners,” says Corzine. With more confidence, partners will increase their use of Fetch’s solution. “Being able to meaningfully improve accuracy on literally every data point using Amazon SageMaker is a huge benefit and propagates throughout our entire business,” says Corzine.

Fetch can now extract more types of data from a receipt, and it has the flexibility to structure resulting insights according to the specific needs of brand partners. “Leaning into ML has unlocked the ability to extract exactly what our partners want from a receipt,” says Corzine. “Partners can make new types of offers because of our investment in ML, and that’s a huge
additional benefit for them.”

Users enjoy the updates too; Fetch has grown from 10 million to 18 million monthly active users since it released the new version. “Amazon SageMaker is a game changer for Fetch,” says Corzine. “We use almost every feature extensively. As new features come out, they are immediately valuable. It’s hard to imagine having done this project without the features of Amazon SageMaker.” For example, Fetch migrated from a custom shadow testing pipeline to [Amazon SageMaker shadow testing](https://aws.amazon.com/sagemaker/shadow-testing/)—which validates the performance of new ML models against production models to prevent outages. Now, shadow testing is more direct because Fetch can directly compare performance with production traffic.

## Outcome | Expanding ML to New Use Cases

The ML team at Fetch is continually working on new models and iterating on existing ones to tune them for better performance. “Another thing we like is being able to keep our technology stack up to date with new features of Amazon SageMaker,” says Chris Lee, ML developer at Fetch. The company will continue expanding its use of AWS to different ML use cases, such as fraud prevention, across multiple teams.

Already one of the biggest consumer engagement software companies, Fetch aims to continue growing. “AWS is a key part of how we plan to scale, and we’ll lean into the features of Amazon SageMaker to continue improving our accuracy,” says Corzine.

## About Fetch

Fetch is a consumer engagement company that provides insights on consumer purchases to brand partners. It also offers a mobile rewards app that lets users earn rewards on purchases through a receipt-scanning feature.


_If you need support in using Hugging Face on SageMaker for your company, please contact us [here](https://huggingface.co/support#form) - our team will contact you to discuss your requirements!_
