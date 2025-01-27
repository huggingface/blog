---
title: "Hugging Face models in Amazon Bedrock" 
thumbnail: /blog/assets/bedrock-marketplace/thumbnail.png
authors:
- user: pagezyhf
- user: philschmid
- user: jeffboudier
- user: Violette
---

![logos](/blog/assets/bedrock-marketplace/thumbnail.png)

# Use Hugging Face models with Amazon Bedrock

We are excited to announce that popular open models from Hugging Face are now available on Amazon Bedrock in the new Bedrock Marketplace! AWS customers can now deploy [83 open models](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog) with Bedrock Marketplace to build their Generative AI applications.

Under the hood, Bedrock Marketplace model endpoints are managed by Amazon Sagemaker Jumpstart. With Bedrock Marketplace, you can now combine the ease of use of SageMaker JumpStart with the fully managed infrastructure of Amazon Bedrock, including compatibility with high-level APIs such as Agents, Knowledge Bases, Guardrails and Model Evaluations.

When registering your Sagemaker Jumpstart endpoints in Amazon Bedrock, you only pay for the Sagemaker compute resources and regular Amazon Bedrock APIs prices are applicable.

In this blog we will show you how to deploy [Gemma 2 27B Instruct](https://huggingface.co/google/gemma-2-27b-it) and use the model with Amazon Bedrock APIs. Learn how to:

1. Deploy Google Gemma 2 27B Instruct
2. Send requests using the Amazon Bedrock APIs
3. Clean Up

## Deploy Google Gemma 2 27B Instruct

There are two ways to deploy an open model to be used with Amazon Bedrock:

1. You can deploy your open model from the Bedrock Model Catalog.
2. You can deploy your open model with Amazon Jumpstart and register it with Bedrock.

Both ways are similar, so we will guide you through the Bedrock Model catalog.

To get started, in the Amazon Bedrock console, make sure you are in one of the 14 regions where the Bedrock Marketplace is available. Then, you choose [“Model catalog”](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog) in the “Foundation models” section of the navigation pane. Here, you can search for both serverless models and models available in Amazon Bedrock Marketplace. You filter results by “Hugging Face” provider and you can browse through the 83 open models available.

For example, let’s search and select Google Gemma 2 27B Instruct.

![model-catalog.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bedrock-marketplace/model-catalog.png)

Choosing the model opens the model detail page where you can see more information from the model provider such as highlights about the model, and usage including sample API calls.

On the top right, let’s click on Deploy.

![model-card.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bedrock-marketplace/model-card.png)

It brings you to the deployment page where you can select the endpoint name, the instance configuration and advanced settings related to networking configuration and service role used to perform the deployment in Sagemaker. Let’s use the default advanced settings and the recommended instance type.

You are also required to accept the End User License Agreement of the model provider.

On the bottom right, let’s click on Deploy.

![model-deploy.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bedrock-marketplace/model-deploy.png)

We just launched the deployment of  GoogleGemma 2 27B Instruct model on a ml.g5.48xlarge instance, hosted in your Amazon Sagemaker tenancy, compatible with Amazon Bedrock APIs!

The endpoint deployment can take several minutes. It will appear in the “Marketplace deployments” page, which you can find in the “Foundation models” section of the navigation pane.

## Use the model with Amazon Bedrock APIs

You can quickly test the model in the Playground through the UI.  However, to invoke the deployed model programmatically with any Amazon Bedrock APIs, you need to get the endpoint ARN.

From the list of managed deployments, choose your model deployment to copy its endpoint ARN.

![model-arn.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bedrock-marketplace/model-arn.png)

You can query your endpoint using the AWS SDK in your preferred language or with the AWS CLI.

Here is an example using Bedrock Converse API through the AWS SDK for Python (boto3):

```python
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")

# Add your bedrock endpoint arn here.
endpoint_arn = "arn:aws:sagemaker:<AWS::REGION>:<AWS::AccountId>:endpoint/<Endpoint_Name>"

# Base inference parameters to use.
inference_config = {
	"maxTokens": 256,
	"temperature": 0.1,
	"topP": 0.999,
}

# Additional inference parameters to use.
additional_model_fields = {"parameters": {"repetition_penalty": 0.9, "top_k": 250, "do_sample": True}}
response = bedrock_runtime.converse(
	modelId=endpoint_arn,
	messages=[
		{
			"role": "user",
			"content": [
				{
					"text": "What is Amazon doing in the field of generative AI?",
				},
			]
		},
		],
	inferenceConfig=inference_config,
	additionalModelRequestFields=additional_model_fields,
)
print(response["output"]["message"]["content"][0]["text"])
```

```python
"Amazon is making significant strides in the field of generative AI, applying it across various products and services. Here's a breakdown of their key initiatives:\n\n**1. Amazon Bedrock:**\n\n* This is their **fully managed service** that allows developers to build and scale generative AI applications using models from Amazon and other leading AI companies. \n* It offers access to foundational models like **Amazon Titan**, a family of large language models (LLMs) for text generation, and models from Cohere"
```

That’s it! If you want to go further, have a look at the [Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).

## Clean up

Don’t forget to delete your endpoint at the end of your experiment to stop incurring costs! At the top right of the page where you grab the endpoint ARN, you can delete your endpoint by clicking on “Delete”.