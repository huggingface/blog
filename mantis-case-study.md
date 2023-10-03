---
title: "Why we’re switching to Hugging Face Inference Endpoints, and maybe you should too"
thumbnail: /blog/assets/78_ml_director_insights/mantis1.png
authors:
- user: mattupson
  guest: true
---

# Why we’re switching to Hugging Face Inference Endpoints, and maybe you should too



Hugging Face recently launched [Inference Endpoints](https://huggingface.co/inference-endpoints); which as they put it: solves transformers in production. Inference Endpoints is a managed service that allows you to:

- Deploy (almost) any model on Hugging Face Hub
- To any cloud (AWS, and Azure, GCP on the way)
- On a range of instance types (including GPU)
- We’re switching some of our Machine Learning (ML) models that do inference on a CPU to this new service. This blog is about why, and why you might also want to consider it.

## What were we doing?

The models that we have switched over to Inference Endpoints were previously managed internally and were running on AWS [Elastic Container Service](https://aws.amazon.com/ecs/) (ECS) backed by [AWS Fargate](https://aws.amazon.com/fargate/). This gives you a serverless cluster which can run container based tasks. Our process was as follows:

- Train model on a GPU instance (provisioned by [CML](https://cml.dev/), trained with [transformers](https://huggingface.co/docs/transformers/main/))
- Upload to [Hugging Face Hub](https://huggingface.co/models)
- Build API to serve model [(FastAPI)](https://fastapi.tiangolo.com/)
- Wrap API in container [(Docker)](https://www.docker.com/)
- Upload container to AWS [Elastic Container Repository](https://aws.amazon.com/ecr/) (ECR)
- Deploy model to ECS Cluster

Now, you can reasonably argue that ECS was not the best approach to serving ML models, but it served us up until now, and also allowed ML models to sit alongside other container based services, so it reduced cognitive load.

## What do we do now?

With Inference Endpoints, our flow looks like this:

- Train model on a GPU instance (provisioned by  [CML](https://cml.dev/), trained with [transformers](https://huggingface.co/docs/transformers/main/))
- Upload to [Hugging Face Hub](https://huggingface.co/models)
- Deploy using Hugging Face Inference Endpoints.

So this is significantly easier. We could also use another managed service such as [SageMaker](https://aws.amazon.com/es/sagemaker/), [Seldon](https://www.seldon.io/), or [Bento ML](https://www.bentoml.com/), etc., but since we are already uploading our model to Hugging Face hub to act as a model registry, and we’re pretty invested in Hugging Face’s other tools (like transformers, and [AutoTrain](https://huggingface.co/autotrain)) using Inference Endpoints makes a lot of sense for us.


## What about Latency and Stability?

Before switching to Inference Endpoints we tested different CPU endpoints types using [ab](https://httpd.apache.org/docs/2.4/programs/ab.html).

For ECS we didn’t test so extensively, but we know that a large container had a latency of about ~200ms from an instance in the same region. The tests we did for Inference Endpoints we based on text classification model fine tuned on [RoBERTa](https://huggingface.co/roberta-base) with the following test parameters:

- Requester region: eu-east-1
- Requester instance size: t3-medium
- Inference endpoint region: eu-east-1
- Endpoint Replicas: 1
- Concurrent connections: 1
- Requests: 1000 (1000 requests in 1–2 minutes even from a single connection would represent very heavy use for this particular application)

The following table shows latency (ms ± standard deviation and time to complete test in seconds) for four Intel Ice Lake equipped CPU endpoints.

```bash
size   |  vCPU (cores) |   Memory (GB)  |  ECS (ms) |  🤗 (ms)
----------------------------------------------------------------------
small  |  1            |  2             |   _       | ~ 296   
medium |  2            |  4             |   _       | 156 ± 51 (158s)  
large  |  4            |   8            |   ~200    | 80 ± 30 (80s)   
xlarge |  8            | 16             |  _        | 43 ± 31 (43s)    
```
What we see from these results is pretty encouraging. The application that will consume these endpoints serves requests in real time, so we need as low latency as possible. We can see that the vanilla Hugging Face container was more than twice as fast as our bespoke container run on ECS — the slowest response we received from the large Inference Endpoint was just 108ms.

## What about the cost?

So how much does this all cost? The table below shows a price comparison for what we were doing previously (ECS + Fargate) and using Inference Endpoints.

```bash
size   |  vCPU         |   Memory (GB)  |  ECS      |  🤗       |  % diff
----------------------------------------------------------------------
small  |  1            |  2             |  $ 33.18  | $ 43.80   |  0.24
medium |  2            |  4             |  $ 60.38  | $ 87.61   |  0.31 
large  |  4            |  8             |  $ 114.78 | $ 175.22  |  0.34
xlarge |  8            | 16             |  $ 223.59 | $ 350.44  | 0.5 
```

We can say a couple of things about this. Firstly, we want a managed solution to deployment, we don’t have a dedicated MLOPs team (yet), so we’re looking for a solution that helps us minimize the time we spend on deploying models, even if it costs a little more than handling the deployments ourselves.

Inference Endpoints are more expensive that what we were doing before, there’s an increased cost of between 24% and 50%. At the scale we’re currently operating, this additional cost, a difference of ~$60 a month for a large CPU instance is nothing compared to the time and cognitive load we are saving by not having to worry about APIs, and containers. If we were deploying 100s of ML microservices we would probably want to think again, but that is probably true of many approaches to hosting.

## Some notes and caveats:

- You can find pricing for Inference Endpoints [here](https://huggingface.co/pricing#endpoints), but a different number is displayed when you deploy a new endpoint from the [GUI](https://ui.endpoints.huggingface.co/new). I’ve used the latter, which is higher.
- The values that I present in the table for ECS + Fargate are an underestimate, but probably not by much. I extracted them from the [fargate pricing page](https://aws.amazon.com/fargate/pricing/) and it includes just the cost of hosting the instance. I’m not including the data ingress/egress (probably the biggest thing is downloading the model from Hugging Face hub), nor have I included the costs related to ECR.

## Other considerations

### Deployment Options

Currently you can deploy an Inference Endpoint from the [GUI](https://ui.endpoints.huggingface.co/new) or using a [RESTful API](https://huggingface.co/docs/inference-endpoints/api_reference). You can also make use of our command line tool [hugie](https://github.com/MantisAI/hfie) (which will be the subject of a future blog) to launch Inference Endpoints in one line of code by passing a configuration, it’s really this simple:

```bash
hugie endpoint create example/development.json
```

For me, what’s lacking is a [custom terraform provider](https://www.hashicorp.com/blog/writing-custom-terraform-providers). It’s all well and good deploying an inference endpoint from a [GitHub action](https://github.com/features/actions) using hugie, as we do, but it would be better if we could use the awesome state machine that is terraform to keep track of these. I’m pretty sure that someone (if not Hugging Face) will write one soon enough — if not, we will.

### Hosting multiple models on a single endpoint

Philipp Schmid posted a really nice blog about how to write a custom [Endpoint Handler](https://www.philschmid.de/multi-model-inference-endpoints) class to allow you to host multiple models on a single endpoint, potentially saving you quite a bit of money. His blog was about GPU inference, and the only real limitation is how many models you can fit into the GPU memory. I assume this will also work for CPU instances, though I’ve not tried yet.

## To conclude…

We find Hugging Face Inference Endpoints to be a very simple and convenient way to deploy transformer (and [sklearn](https://huggingface.co/scikit-learn)) models into an endpoint so they can be consumed by an application. Whilst they cost a little more than the ECS approach we were using before, it’s well worth it because it saves us time on thinking about deployment, we can concentrate on the thing we want to: building NLP solutions for our clients to help solve their problems.

_If you’re interested in Hugging Face Inference Endpoints for your company, please contact us [here](https://huggingface.co/inference-endpoints/enterprise) - our team will contact you to discuss your requirements!_

_This article was originally published on February 15, 2023 [in Medium](https://medium.com/mantisnlp/why-were-switching-to-hugging-face-inference-endpoints-and-maybe-you-should-too-829371dcd330)._

