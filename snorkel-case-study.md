---
title: "Snorkel AI x Hugging Face: unlock foundation models for enterprises"
thumbnail: /blog/assets/78_ml_director_insights/snorkel.png
authors:
- user: VioletteLepercq
---

# Snorkel AI x Hugging Face: unlock foundation models for enterprises



_This article is a cross-post from an originally published post on April 6, 2023 [in Snorkel's blog](https://snorkel.ai/snorkel-hugging-face-unlock-foundation-models-for-enterprise/), by Friea Berg ._


As OpenAI releases [GPT-4](https://openai.com/research/gpt-4) and Google debuts [Bard](https://gizmodo.com/google-bard-chatgpt-ai-rival-released-1850248162) in beta, enterprises around the world are excited to leverage the power of foundation models. As that excitement builds, so does the realization that most companies and organizations are not equipped to properly take advantage of foundation models.

Foundation models pose a unique set of challenges for enterprises. Their larger-than-ever size makes them difficult and expensive for companies to host themselves, and using off-the-shelf FMs for production use cases could mean poor performance or substantial governance and compliance risks.

Snorkel AI bridges the gap between foundation models and practical enterprise use cases and has [yielded impressive results](https://snorkel.ai/how-pixability-uses-foundation-models-to-accelerate-nlp-application-development-by-months/) for AI innovators like Pixability. We’re teaming with [Hugging Face](https://huggingface.co/), best known for its enormous repository of ready-to-use open-source models, to provide enterprises with even more flexibility and choice as they develop AI applications.

## Foundation models in Snorkel Flow

The Snorkel Flow development platform enables users to [adapt foundation models](https://snorkel.ai/snorkel-flow/foundation-model-development/) for their specific use cases. Application development begins by inspecting the predictions of a selected foundation model “out of the box” on their data. These predictions become an initial version of training labels for those data points. Snorkel Flow helps users to identify error modes in that model and correct them efficiently via [programmatic labeling](https://snorkel.ai/programmatic-labeling/), which can include updating training labels with heuristics or [prompts](https://snorkel.ai/combining-foundation-models-with-weak-supervision/). The base foundation model can then be fine-tuned on the updated labels and evaluated once again, with this iterative “detect and correct” process continuing until the adapted foundation model is sufficiently high quality to deploy.

Hugging Face helps enable this powerful development process by making more than 150,000 open-source models immediately available from a single source. Many of those models are specialized on domain-specific data, like the BioBERT and SciBERT models used to demonstrate [how ML can be used to spot adverse drug events](https://snorkel.ai/adverse-drug-events-how-to-spot-them-with-machine-learning/). One – or better yet, [multiple](https://snorkel.ai/combining-foundation-models-with-weak-supervision/) – specialized base models can give users a jump-start on initial predictions, prompts for improving labels, or fine-tuning a final model for deployment.

## How does Hugging Face help?

Snorkel AI’s partnership with Hugging Face supercharges Snorkel Flow’s foundation model capabilities. Initially we only made a small number of foundation models available. Each one required a dedicated service, making it prohibitively expensive and difficult for us to offer enterprises the flexibility to capitalize on the rapidly growing variety of models available. Adopting Hugging Face’s Inference Endpoint service enabled us to expand the number of foundation models our users could tap into while keeping costs manageable.

Hugging Face’s service allows users to create a model API in a few clicks and begin using it immediately. Crucially, the new service has “pause and resume” capabilities that allow us to activate a model API when a client needs it, and put it to sleep when they don’t.

"We were pleasantly surprised to see how straightforward Hugging Face Inference Endpoint service was to set up.. All the configuration options were pretty self-explanatory, but we also had access to all the options we needed in terms of what cloud to run on, what security level we needed, etc."

– Snorkel CTO  and Co-founder Braden Hancock

<iframe width="100%" style="aspect-ratio: 16 / 9;" src="https://www.youtube-nocookie.com/embed/woblG7iZPSw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## How does this help Snorkel customers?

Few enterprises have the resources to train their own foundation models from scratch. While many may have the in-house expertise to fine-tune their own version of a foundation model, they may struggle to gather the volume of data needed for that task. Snorkel’s data-centric platform for developing foundation models and alignment with leading industry innovators like Hugging Face help put the power of foundation models at our users’ fingertips.

#### "With Snorkel AI and Hugging Face Inference Endpoints, companies will accelerate their data-centric AI applications with open source at the core. Machine Learning is becoming the default way of building technology, and building from open source allows companies to build the right solution for their use case and take control of the experience they offer to their customers. We are excited to see Snorkel AI enable automated data labeling for the enterprise building from open-source Hugging Face models and Inference Endpoints, our machine learning production service.”

Clement Delangue, co-founder and CEO, Hugging Face

## Conclusion

Together, Snorkel and Hugging Face make it easier than ever for large companies, government agencies, and AI innovators to get value from foundation models. The ability to use Hugging Face’s comprehensive hub of foundation models means that users can pick the models that best align with their business needs without having to invest in the resources required to train them. This integration is a significant step forward in making foundation models more accessible to enterprises around the world.

_If you’re interested in Hugging Face Inference Endpoints for your company, please contact us [here](https://huggingface.co/inference-endpoints/enterprise) - our team will contact you to discuss your requirements!_

