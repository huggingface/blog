---
title: "Rocket Money x Hugging Face: Scaling Volatile ML Models in Production​"
thumbnail: /blog/assets/78_ml_director_insights/rocketmoney.png
authors:
- user: nicokuzak
  guest: true
- user: ccpoirier
  guest: true
---

# Rocket Money x Hugging Face: Scaling Volatile ML Models in Production


#### "We discovered that they were not just service providers, but partners who were invested in our goals and outcomes” _- Nicolas Kuzak, Senior ML Engineer at Rocket Money._

## Scaling and Maintaining ML Models in Production Without an MLOps Team

We created [Rocket Money](https://www.rocketmoney.com/) (a personal finance app formerly known as Truebill) to help users improve their financial wellbeing. Users link their bank accounts to the app which then classifies and categorizes their transactions, identifying recurring patterns to provide a consolidated, comprehensive view of their personal financial life. A critical stage of transaction processing is detecting known merchants and services, some of which Rocket Money can cancel and negotiate the cost of for members. This detection starts with the transformation of short, often truncated and cryptically formatted transaction strings into classes we can use to enrich our product experience.

## The Journey Toward a New System

We first extracted brands and products from transactions using regular expression-based normalizers. These were used in tandem with an increasingly intricate decision table that mapped strings to corresponding brands. This system proved effective for the first four years of the company when classes were tied only to the products we supported for cancellations and negotiations. However, as our user base grew, the subscription economy boomed and the scope of our product increased, we needed to keep up with the rate of new classes while simultaneously tuning regexes and preventing collisions and overlaps. To address this, we explored various traditional machine learning (ML) solutions, including a bag of words model with a model-per-class architecture. This system struggled with maintenance and performance and was mothballed.

We decided to start from a clean slate, assembling both a new team and a new mandate. Our first task was to accumulate training data and construct an in-house system from scratch. We used Retool to build labeling queues, gold standard validation datasets, and drift detection monitoring tools. We explored a number of different model topologies, but ultimately chose a BERT family of models to solve our text classification problem. The bulk of the initial model testing and evaluation was conducted offline within our GCP warehouse. Here we designed and built the telemetry and system we used to measure the performance of a model with 4000+ classes.

## Solving Domain Challenges and Constraints by Partnering with Hugging Face

There are a number of unique challenges we face within our domain, including entropy injected by merchants, processing/payment companies, institutional differences, and shifts in user behavior. Designing and building efficient model performance alerting along with realistic benchmarking datasets has proven to be an ongoing challenge. Another significant hurdle is determining the optimal number of classes for our system - each class represents a significant amount of effort to create and maintain. Therefore, we must consider the value it provides to users and our business.

With a model performing well in offline testing and a small team of ML engineers, we were faced with a new challenge: seamless integration of that model into our production pipeline. The existing regex system processed more than 100 million transactions per month with a very bursty load, so it was crucial to have a high-availability system that could scale dynamically to load and maintain a low overall latency within the pipeline coupled with a system that was compute-optimized for the models we were serving. As a small startup at the time, we chose to buy rather than build the model serving solution. At the time, we didn’t have in-house model ops expertise and we needed to focus the energy of our ML engineers on enhancing the performance of the models within the product. With this in mind, we set out in search of the solution.

In the beginning, we auditioned a hand-rolled, in-house model hosting solution we had been using for prototyping, comparing it against AWS Sagemaker and Hugging Face’s new model hosting Inference API. Given that we use GCP for data storage and Google Vertex Pipelines for model training, exporting models to AWS Sagemaker was clunky and bug prone. Thankfully, the set up for Hugging Face was quick and easy, and it was able to handle a small portion of traffic within a week. Hugging Face simply worked out of the gate, and this reduced friction led us to proceed down this path.

After an extensive three-month evaluation period, we chose Hugging Face to host our models. During this time, we gradually increased transaction volume to their hosted models and ran numerous simulated load tests based on our worst-case scenario volumes. This process allowed us to fine-tune our system and monitor performance, ultimately giving us confidence in the inference API's ability to handle our transaction enrichment loads.

Beyond technical capabilities, we also established a strong rapport with the team at Hugging Face. We discovered they were not just service providers, but partners who were invested in our goals and outcomes. Early in our collaboration we set up a shared Slack channel which proved invaluable. We were particularly impressed by their prompt response to issues and proactive approach to problem-solving. Their engineers and CSMs consistently demonstrated their commitment in our success and dedication to doing things right. This gave us an additional layer of confidence when it was time to make the final selection.

## Integration, Evaluation, and the Final Selection

#### "Overall, the experience of working hand in hand with Hugging Face on model deployment has been enriching for our team and has instilled in us the confidence to push for greater scale"_- Nicolas Kuzak, Senior ML Engineer at Rocket Money._

Once the contract was signed, we began the migration of moving off our regex based system to direct an increasing amount of critical path traffic to the transformer model. Internally, we had to build some new telemetry for both model and production data monitoring. Given that this system is positioned so early in the product experience, any inaccuracies in model outcomes could significantly impact business metrics. We ran an extensive experiment where new users were split equally between the old system and the new model. We assessed model performance in conjunction with broader business metrics, such as paid user retention and engagement. The ML model clearly outperformed in terms of retention, leading us to confidently make the decision to scale the system - first to new users and then to existing users - ramping to 100% over a span of two months.

With the model fully positioned in the transaction processing pipeline, both uptime and latency became major concerns. Many of our downstream processes rely on classification results, and any complications can lead to delayed data or incomplete enrichment, both causing a degraded user experience.

The inaugural year of collaboration between Rocket Money and Hugging Face was not without its challenges. Both teams, however, displayed remarkable resilience and a shared commitment to resolving issues as they arose. One such instance was when we expanded the number of classes in our second production model, which unfortunately led to an outage. Despite this setback, the teams persevered, and we've successfully avoided a recurrence of the same issue. Another hiccup occurred when we transitioned to a new model, but we still received results from the previous one due to caching issues on Hugging Face's end. This issue was swiftly addressed and has not recurred. Overall, the experience of working hand in hand with Hugging Face on model deployment has been enriching for our team and has instilled in us the confidence to push for greater scale.

Speaking of scale, as we started to witness a significant increase in traffic to our model, it became clear that the cost of inference would surpass our projected budget. We made use of a caching layer prior to inference calls that significantly reduces the cardinality of transactions and attempts to benefit from prior inference. Our problem technically could achieve a 93% cache rate, but we’ve only ever reached 85% in a production setting. With the model serving 100% of predictions, we’ve had a few milestones on the Rocket Money side - our model has been able to scale to a run rate of over a billion transactions per month and manage the surge in traffic as we climbed to the #1 financial app in the app store and #7 overall, all while maintaining low latency.

## Collaboration and Future Plans

#### "The uptime and confidence we have in the HuggingFace Inference API has allowed us to focus our energy on the value generated by the models and less on the plumbing and day-to-day operation" _- Nicolas Kuzak, Senior ML Engineer at Rocket Money._

Post launch, the internal Rocket Money team is now focusing on both class and performance tuning of the model in addition to more automated monitoring and training label systems. We add new labels on a daily basis and encounter the fun challenges of model lifecycle management, including unique things like company rebranding and new companies and products emerging after Rocket Companies acquired Truebill in late 2021.

We constantly examine whether we have the right model topology for our problem. While LLMs have recently been in the news, we’ve struggled to find an implementation that can outperform our specialized transformer classifiers at this time in both speed and cost. We see promise in the early results of using them in the long tail of services (i.e. mom-and-pop shops) - keep an eye out for that in a future version of Rocket Money! The uptime and confidence we have in the HuggingFace Inference API has allowed us to focus our energy on the value generated by the models and less on the plumbing and day-to-day operation. With the help of Hugging Face, we have taken on more scale and complexity within our model and the types of value it generates. Their customer service and support have exceeded our expectations and they’re genuinely a great partner in our journey.

_If you want to learn how Hugging Face can manage your ML inference workloads, contact the Hugging Face team [here](https://huggingface.co/support#form/)._ 
