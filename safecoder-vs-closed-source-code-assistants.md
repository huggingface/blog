---
title: "SafeCoder vs. Closed-source Code Assistants"
thumbnail: /blog/assets/safecoder-vs-closed-source-code-assistants/image.png
authors:
- user: juliensimon
---

# SafeCoder vs. Closed-source Code Assistants



For decades, software developers have designed methodologies, processes, and tools that help them improve code quality and increase productivity. For instance, agile, test-driven development, code reviews, and CI/CD are now staples in the software industry. 

In "How Google Tests Software" (Addison-Wesley, 2012), Google reports that fixing a bug during system tests - the final testing stage - is 1000x more expensive than fixing it at the unit testing stage. This puts much pressure on developers - the first link in the chain - to write quality code from the get-go. 

For all the hype surrounding generative AI, code generation seems a promising way to help developers deliver better code fast. Indeed, early studies show that managed services like [GitHub Copilot](https://github.blog/2023-06-27-the-economic-impact-of-the-ai-powered-developer-lifecycle-and-lessons-from-github-copilot) or [Amazon CodeWhisperer](https://aws.amazon.com/codewhisperer/) help developers be more productive.

However, these services rely on closed-source models that can't be customized to your technical culture and processes. Hugging Face released [SafeCoder](https://huggingface.co/blog/starcoder) a few weeks ago to fix this. SafeCoder is a code assistant solution built for the enterprise that gives you state-of-the-art models, transparency, customizability, IT flexibility, and privacy.

In this post, we'll compare SafeCoder to closed-source services and highlight the benefits you can expect from our solution.


## State-of-the-art models

SafeCoder is currently built on top of the [StarCoder](https://huggingface.co/blog/starcoder) models, a family of open-source models designed and trained within the [BigCode](https://huggingface.co/bigcode) collaborative project.

StarCoder is a 15.5 billion parameter model trained for code generation in over 80 programming languages. It uses innovative architectural concepts, like [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA), to improve throughput and reduce latency, a technique also present in the [Falcon](https://huggingface.co/blog/falcon) and adapted for [LLaMa 2](https://huggingface.co/blog/llama2) models.

StarCoder has an 8192-token context window, helping it take into account more of your code to generate new code. It can also do fill-in-the-middle, i.e., insert within your code, instead of just appending new code at the end.

Lastly, like [HuggingChat](https://huggingface.co/chat/), SafeCoder will introduce new state-of-the-art models over time, giving you a seamless upgrade path.

Unfortunately, closed-source code assistant services don't share information about the underlying models, their capabilities, and their training data. 

## Transparency

In line with the [Chinchilla Scaling Law](https://arxiv.org/abs/2203.15556v1), SafeCoder is a compute-optimal model trained on 1 trillion (1,000 billion) code tokens. These tokens are extracted from [The Stack](https://huggingface.co/datasets/bigcode/the-stack), a 2.7 terabyte dataset built from permissively licensed open-source repositories. 
All efforts are made to honor opt-out requests, and we built a [tool](https://huggingface.co/spaces/bigcode/in-the-stack) that lets repository owners check if their code is part of the dataset.

In the spirit of transparency, our [research paper](https://arxiv.org/abs/2305.06161) discloses the model architecture, the training process, and detailed metrics.

Unfortunately, closed-source services stick to vague information, such as "[the model was trained on] billions of lines of code." To the best of our knowledge, no metrics are available.

## Customization

The StarCoder models have been specifically designed to be customizable, and we have already built different versions:

* [StarCoderBase](https://huggingface.co/bigcode/starcoderbase): the original model trained on 80+ languages from The Stack.
* [StarCoder](https://huggingface.co/bigcode/starcoder): StarCoderBase further trained on Python.
* [StarCoder+](https://huggingface.co/bigcode/starcoderplus): StarCoderBase further trained on English web data for coding conversations.

We also shared the [fine-tuning code](https://github.com/bigcode-project/starcoder/) on GitHub.
 
Every company has its preferred languages and coding guidelines, i.e., how to write inline documentation or unit tests, or do's and don'ts on security and performance. With SafeCoder, we can help you train models that learn the peculiarities of your software engineering process.  Our team will help you prepare high-quality datasets and fine-tune StarCoder on your infrastructure. Your data will never be exposed to anyone.

Unfortunately, closed-source services cannot be customized.
 
## IT flexibility

SafeCoder relies on Docker containers for fine-tuning and deployment. It's easy to run on-premise or in the cloud on any container management service.

In addition, SafeCoder includes our [Optimum](https://github.com/huggingface/optimum) hardware acceleration libraries. Whether you work with CPU, GPU, or AI accelerators, Optimum will kick in automatically to help you save time and money on training and inference. Since you control the underlying hardware, you can also tune the cost-performance ratio of your infrastructure to your needs.

Unfortunately, closed-source services are only available as managed services.

## Security and privacy

Security is always a top concern, all the more when source code is involved. Intellectual property and privacy must be protected at all costs.

Whether you run on-premise or in the cloud, SafeCoder is under your complete administrative control. You can apply and monitor your security checks and maintain strong and consistent compliance across your IT platform.

SafeCoder doesn't spy on any of your data. Your prompts and suggestions are yours and yours only. SafeCoder doesn't call home and send telemetry data to Hugging Face or anyone else. No one but you needs to know how and when you're using SafeCoder. SafeCoder doesn't even require an Internet connection. You can (and should) run it fully air-gapped.

Closed-source services rely on the security of the underlying cloud. Whether this works or not for your compliance posture is your call. For enterprise users, prompts and suggestions are not stored (they are for individual users). However, we regret to point out that GitHub collects  ["user engagement data"](https://docs.github.com/en/copilot/overview-of-github-copilot/about-github-copilot-for-business) with no possibility to opt-out. AWS does the same by default but lets you [opt out](https://docs.aws.amazon.com/codewhisperer/latest/userguide/sharing-data.html).

## Conclusion

We're very excited about the future of SafeCoder, and so are our customers. No one should have to compromise on state-of-the-art code generation, transparency, customization, IT flexibility, security, and privacy. We believe SafeCoder delivers them all, and we'll keep working hard to make it even better.

If you’re interested in SafeCoder for your company, please [contact us](mailto:api-enterprise@huggingface.co). Our team will contact you shortly to learn more about your use case and discuss requirements.

Thanks for reading!



 
 
