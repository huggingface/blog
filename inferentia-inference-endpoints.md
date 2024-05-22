---
title: "Deploy models on AWS Inferentia2 from Hugging Face" 
thumbnail: /blog/assets/inferentia-inference-endpoints/thumbnail.jpg
authors:
- user: jeffboudier
- user: philschmid
---


# Deploy models on AWS Inferentia2 from Hugging Face

![thumbnail](/blog/assets/inferentia-inference-endpoints/thumbnail.jpg)


[AWS Inferentia2](https://aws.amazon.com/machine-learning/inferentia/) is the latest AWS machine learning chip available through the [Amazon EC2 Inf2 instances](https://aws.amazon.com/ec2/instance-types/inf2/) on Amazon Web Services. Designed from the ground up for AI workloads, Inf2 instances offer great performance and cost/performance for production workloads.

We have been working for over a year with the product and engineering teams at AWS to make the performance and cost-efficiency of AWS Trainium and Inferentia chips available to Hugging Face users. Our open-source library <code>[optimum-neuron](https://huggingface.co/docs/optimum-neuron/index)</code> makes it easy to train and deploy Hugging Face models on these accelerators. You can read more about our work [accelerating transformers](https://huggingface.co/blog/accelerate-transformers-with-inferentia2), [large language models](https://huggingface.co/blog/inferentia-llama2) and [text-generation-inference](https://huggingface.co/blog/text-generation-inference-on-inferentia2) (TGI).

Today, we are making the power of Inferentia2 directly and widely available to Hugging Face Hub users.


# Enabling over 100,000 models on AWS Inferentia2 with Amazon SageMaker

A few months ago, we introduced a new way to deploy Large Language Models (LLMs) on SageMaker, with a new Inferentia/Trainium option for supported models, like Meta [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B?sagemaker_deploy=true). You can deploy a Llama3 model on Inferentia2 instances on SageMaker to serve inference at scale and benefit from SageMaker’s complete set of fully managed features for building and fine-tuning models, MLOps, and governance.

![catalog](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inferentia-inference-endpoints/sagemaker.png)

Today, we are expanding support for this deployment experience to over 100,000 public models available on Hugging Face, including 14 new model architectures (`albert`,`bert`,`camembert`,`convbert`,`deberta`,`deberta-v2`,`distilbert`,`electra`,`roberta`,`mobilebert`,`mpnet`,`vit`,`xlm`,`xlm-roberta`), and 6 new machine learning tasks (`text-classification`,`text-generation`,`token-classification`,`fill-mask`,`question-answering`,`feature-extraction`).

Following these simple code snippets, AWS customers will be able to easily deploy the models on Inferentia2 instances in Amazon SageMaker.


# Hugging Face Inference Endpoints introduces support for AWS Inferentia2

Another option to deploy models from the Hugging Face model hub, the is [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated). Today, we are happy to introduce new Inferentia 2 instances for Hugging Face Inference Endpoints. So, now when you find a model in Hugging Face you are interested in, you can deploy it in just a few clicks on Inferentia2. All you need to do is select the model you want to deploy, select the new Inf2 instance option under the Amazon Web Services instance configuration, and you’re off to the races.

For supported models like Llama 3, you can select 2 flavors:
* Inf2-small, with 2 cores and 16 GB memory ($0.75/hour) perfect for Llama 3 8B
* Inf2-xlarge, with 24 cores and 760 GB memory ($12/hour) perfect for Llama 3 70B

Hugging Face Inference Endpoints are billed by the second of capacity used, with cost scaling up with replica autoscaling, and down to zero with scale to zero - both automated and enabled with easy to use settings.

![catalog](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inferentia-inference-endpoints/create-endpoint.png)

Inference Endpoints uses [Text Generation Inference for Neuron](https://huggingface.co/blog/text-generation-inference-on-inferentia2) (TGI) to run Llama 3 on AWS Inferentia. TGI is a purpose-built solution for deploying and serving Large Language Models (LLMs) for production workloads at scale, supporting continuous batching, streaming and much more. In addition, LLMs deployed with Text Generation Inference are compatible with the OpenAI SDK Messages API, so if you already have Gen AI applications integrated with LLMs, you don’t need to change the code of your application, and just have to point to your new endpoint deployed with Hugging Face Inference Endpoints.

After you deploy your endpoint on Inferentia2, you can send requests using the Widget provided in the UI or the OpenAI SDK.


## Whats Next 

We are working hard to expand the scope of models enabled for deployment on AWS Inferentia2 with Hugging Face Inference Endpoints. Next, we want to add support for Diffusion and Embedding models, so you can generate images and build semantic search and recommendation systems leveraging the acceleration of AWS Inferentia2 and the ease of use of Hugging Face Inference Endpoints.  

In addition, we continue our work to improve performance for Text Generation Inference (TGI) on Neuronx, ensuring faster and more efficient LLM deployments on AWS Inferentia 2 in our open source libraries. Stay tuned for these updates as we continue to enhance our capabilities and optimize your deployment experience!
