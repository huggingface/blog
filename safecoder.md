---
title: "Introducing SafeCoder" 
thumbnail: /blog/assets/159_safecoder/thumbnail.jpg
authors:
- user: jeffboudier
- user: philschmid
---

# Introducing SafeCoder


Today we are excited to announce SafeCoder - a code assistant solution built for the enterprise.

The goal of SafeCoder is to unlock software development productivity for the enterprise, with a fully compliant and self-hosted pair programmer. In marketing speak: “your own on-prem GitHub copilot”.

Before we dive deeper, here’s what you need to know:

- SafeCoder is not a model, but a complete end-to-end commercial solution
- SafeCoder is built with security and privacy as core principles - code never leaves the VPC during training or inference
- SafeCoder is designed for self-hosting by the customer on their own infrastructure
- SafeCoder is designed for customers to own their own Code Large Language Model

![example](/blog/assets/159_safecoder/coding-example.gif)


## Why SafeCoder?

Code assistant solutions built upon LLMs, such as GitHub Copilot, are delivering strong [productivity boosts](https://github.blog/2022-09-07-research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness/). For the enterprise, the ability to tune Code LLMs on the company code base to create proprietary Code LLMs improves reliability and relevance of completions to create another level of productivity boost. For instance, Google internal LLM code assistant reports a completion [acceptance rate of 25-34%](https://ai.googleblog.com/2022/07/ml-enhanced-code-completion-improves.html) by being trained on an internal code base.

However, relying on closed-source Code LLMs to create internal code assistants exposes companies to compliance and security issues. First during training, as fine-tuning a closed-source Code LLM on an internal codebase requires exposing this codebase to a third party. And then during inference, as fine-tuned Code LLMs are likely to “leak” code from their training dataset during inference. To meet compliance requirements, enterprises need to deploy fine-tuned Code LLMs within their own infrastructure - which is not possible with closed source LLMs.

With SafeCoder, Hugging Face will help customers build their own Code LLMs, fine-tuned on their proprietary codebase, using state of the art open models and libraries, without sharing their code with Hugging Face or any other third party. With SafeCoder, Hugging Face delivers a containerized, hardware-accelerated Code LLM inference solution, to be deployed by the customer directly within the Customer secure infrastructure, without code inputs and completions leaving their secure IT environment.

## From StarCoder to SafeCoder

At the core of the SafeCoder solution is the [StarCoder](https://huggingface.co/bigcode/starcoder) family of Code LLMs, created by the [BigCode](https://huggingface.co/bigcode) project, a collaboration between Hugging Face, ServiceNow and the open source community.

The StarCoder models offer unique characteristics ideally suited to enterprise self-hosted solution:

- State of the art code completion results - see benchmarks in the [paper](https://huggingface.co/papers/2305.06161) and [multilingual code evaluation leaderboard](https://huggingface.co/spaces/bigcode/multilingual-code-evals)
- Designed for inference performance: a 15B parameters model with code optimizations, Multi-Query Attention for reduced memory footprint, and Flash Attention to scale to 8,192 tokens context.
- Trained on [the Stack](https://huggingface.co/datasets/bigcode/the-stack), an ethically sourced, open source code dataset containing only commercially permissible licensed code, with a developer opt-out mechanism from the get-go, refined through intensive PII removal and deduplication efforts.

Note: While StarCoder is the inspiration and model powering the initial version of SafeCoder, an important benefit of building a LLM solution upon open source models is that it can adapt to the latest and greatest open source models available. In the future, SafeCoder may offer other similarly commercially permissible open source models built upon ethically sourced and transparent datasets as the base LLM available for fine-tuning.

## Privacy and Security as a Core Principle

For any company, the internal codebase is some of its most important and valuable intellectual property. A core principle of SafeCoder is that the customer internal codebase will never be accessible to any third party (including Hugging Face) during training or inference.

In the initial set up phase of SafeCoder, the Hugging Face team provides containers, scripts and examples to work hand in hand with the customer to select, extract, prepare, duplicate, deidentify internal codebase data into a training dataset to be used in a Hugging Face provided training container configured to the hardware infrastructure available to the customer.

In the deployment phase of SafeCoder, the customer deploys containers provided by Hugging Face on their own infrastructure to expose internal private endpoints within their VPC. These containers are configured to the exact hardware configuration available to the customer, including NVIDIA GPUs, AMD Instinct GPUs, Intel Xeon CPUs, AWS Inferentia2 or Habana Gaudi accelerators.

## Compliance as a Core Principle

As the regulation framework around machine learning models and datasets is still being written across the world, global companies need to make sure the solutions they use minimize legal risks.

Data sources, data governance, management of copyrighted data are just a few of the most important compliance areas to consider. BigScience, the older cousin and inspiration for BigCode, addressed these areas in working groups before they were broadly recognized by the draft AI EU Act, and as a result was [graded as most compliant among Foundational Model Providers in a Stanford CRFM study](https://crfm.stanford.edu/2023/06/15/eu-ai-act.html).

BigCode expanded upon this work by implementing novel techniques for the code domain and building The Stack with compliance as a core principle, such as commercially permissible license filtering, consent mechanisms (developers can [easily find out if their code is present and request to be opted out](https://huggingface.co/spaces/bigcode/in-the-stack) of the dataset), and extensive documentation and tools to inspect the [source data](https://huggingface.co/datasets/bigcode/the-stack-metadata), and dataset improvements (such as [deduplication](https://huggingface.co/blog/dedup) and [PII removal](https://huggingface.co/bigcode/starpii)).

All these efforts translate into legal risk minimization for users of the StarCoder models, and customers of SafeCoder. And for SafeCoder users, these efforts translate into compliance features: when software developers get code completions these suggestions are checked against The Stack, so users know if the suggested code matches existing code in the source dataset, and what the license is. Customers can specify which licenses are preferred and surface those preferences to their users.

## How does it work?

SafeCoder is a complete commercial solution, including service, software and support.

### Training your own SafeCoder model

StarCoder was trained in more than 80 programming languages and offers state of the art performance on [multiple benchmarks](https://huggingface.co/spaces/bigcode/multilingual-code-evals). To offer better code suggestions specifically for a SafeCoder customer, we start the engagement with an optional training phase, where the Hugging Face team works directly with the customer team to guide them through the steps to prepare and build a training code dataset, and to create their own code generation model through fine-tuning, without ever exposing their codebase to third parties or the internet.

The end result is a model that is adapted to the code languages, standards and practices of the customer. Through this process, SafeCoder customers learn the process and build a pipeline for creating and updating their own models, ensuring no vendor lock-in, and keeping control of their AI capabilities.

### Deploying SafeCoder

During the setup phase, SafeCoder customers and Hugging Face design and provision the optimal infrastructure to support the required concurrency to offer a great developer experience. Hugging Face then builds SafeCoder inference containers that are hardware-accelerated and optimized for throughput, to be deployed by the customer on their own infrastructure.

SafeCoder inference supports various hardware to give customers a wide range of options: NVIDIA Ampere GPUs, AMD Instinct GPUs, Habana Gaudi2, AWS Inferentia 2, Intel Xeon Sapphire Rapids CPUs and more.

### Using SafeCoder

Once SafeCoder is deployed and its endpoints are live within the customer VPC, developers can install compatible SafeCoder IDE plugins to get code suggestions as they work. Today, SafeCoder supports popular IDEs, including [VSCode](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode), IntelliJ and with more plugins coming from our partners.

## How can I get SafeCoder?

Today, we are announcing SafeCoder in collaboration with VMware at the VMware Explore conference and making SafeCoder available to VMware enterprise customers. Working with VMware helps ensure the deployment of SafeCoder on customers’ VMware Cloud infrastructure is successful – whichever cloud, on-premises or hybrid infrastructure scenario is preferred by the customer. In addition to utilizing SafeCoder, VMware has published a [reference architecture](https://www.vmware.com/content/dam/digitalmarketing/vmware/en/pdf/docs/vmware-baseline-reference-architecture-for-generative-ai.pdf) with code samples to enable the fastest possible time-to-value when deploying and operating SafeCoder on VMware infrastructure. VMware’s Private AI Reference Architecture makes it easy for organizations to quickly leverage popular open source projects such as ray and kubeflow to deploy AI services adjacent to their private datasets, while working with Hugging Face to ensure that organizations maintain the flexibility to take advantage of the latest and greatest in open-source models. This is all without tradeoffs in total cost of ownership or performance.

“Our collaboration with Hugging Face around SafeCoder fully aligns to VMware’s goal of enabling customer choice of solutions while maintaining privacy and control of their business data. In fact, we have been running SafeCoder internally for months and have seen excellent results. Best of all, our collaboration with Hugging Face is just getting started, and I’m excited to take our solution to our hundreds of thousands of customers worldwide,” says Chris Wolf, Vice President of VMware AI Labs. Learn more about private AI and VMware’s differentiation in this emerging space [here](https://octo.vmware.com/vmware-private-ai-foundation/).

---

If you’re interested in SafeCoder for your company, please contact us [here](mailto:api-enterprise@huggingface.co?subject=SafeCoder) - our team will contact you to discuss your requirements!
