---
title: "Ryght’s Journey to Empower Healthcare and Life Sciences with Expert Support from Hugging Face" 
thumbnail: /blog/assets/ryght-case-study/thumbnail.png
authors:
- user: andrewrreed
- user: johnnybio
  guest: true
  org: RyghtAI
---

# Ryght’s Journey to Empower Healthcare and Life Sciences with Expert Support from Hugging Face

> [!NOTE] This is a guest blog post by the Ryght team.

## Who is Ryght?

Ryght is building an enterprise-grade generative AI platform tailored for the healthcare and life sciences sectors. Today is their official launch of [Ryght Preview](https://www.ryght.ai/signup?utm_campaign=Preview%20Launch%20April%2016%2C%2024&utm_source=Huggging%20Face%20Blog%20-%20Preview%20Launch%20Sign%20Up), now publicly available for all.

Life science companies are amassing a wealth of data from diverse sources (lab data, EMR, genomics, claims, pharmacy, clinical, etc.), but analysis of that data is archaic, requiring large teams for everything from simple queries to developing useful ML models. There is huge demand for actionable knowledge to drive drug development, clinical trials, and commercial activity, and the rise of precision medicine is only accelerating this demand.

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ryght-case-study/click-through.gif" alt="Ryght Laptop" style="width: 90%; height: auto;"><br>
</p>

[Ryght’s](https://hubs.li/Q02sLGKL0) goal is to empower life science professionals to get the insights they need swiftly and securely. To do so, they’re building a SaaS platform that offers industry-specific AI copilots and custom built solutions for professionals and organizations to accelerate their research, analysis, and documentation across a variety of complex data sources.

Recognizing how fast paced and ever changing the AI landscape is, Ryght sought out Hugging Face as a technical advisory partner early in their journey via the [Expert Support Program](https://huggingface.co/support).

## Overcoming challenges, together

> ##### *Our partnership with Hugging Face's expert support has played a crucial role in expediting the development of our generative AI platform. The rapidly evolving landscape of AI has the potential to revolutionize our industry, and Hugging Face’s highly performant and enterprise-ready Text Generation Inference (TGI) and Text Embeddings Inference (TEI) services are game changers in their own right. - [Johnny Crupi, CTO](https://www.linkedin.com/in/johncrupi/) at [Ryght](http://www.ryght.ai/?utm_campaign=hf&utm_source=hf_blog)*

Ryght faced several challenges as they set out to build their generative AI platform.

### 1. The need to quickly upskill a team and stay informed in a highly dynamic environment
With AI and ML technologies advancing so quickly, ensuring that the team remains abreast of the latest techniques, tools, and best practices is critical. This continuous learning curve is steep and requires a concerted effort to stay informed.

Having access to Hugging Face’s team of experts who operate at the center of the AI ecosystem helps Ryght keep up with the latest developments and models that are relevant to their domain. This is achieved through open, asynchronous channels of communication, regular advisory meetings, and dedicated technical workshops.

### 2. Identifying the most [cost] effective ML approaches amidst the noisy sea of options
The AI field is bustling with innovation, leading to an abundance of tools, libraries, models, and methodologies. For a startup like Ryght, it's imperative to cut through this noise and identify which ML strategies are most applicable to their unique use cases in the life sciences sector. This involves not just understanding the current state of the art, but also looking ahead to which technologies will remain relevant and scalable for the future.

Hugging Face serves as a partner to Ryght’s technical team – assisting in solution design, proof-of-concept development, and production workload optimization. This includes tailored recommendations on libraries, frameworks, and models best fit for Ryght’s specific needs, along with demonstrable examples of how to use them. This guidance ultimately streamlines the decision-making process and reduces the time to development.

### 3. Requirement to develop performant solutions that emphasize security, privacy, and flexibility
Given the focus on enterprise-level solutions, Ryght prioritizes security, privacy, and governance. This necessitates a flexible architecture capable of interfacing with various large language models (LLMs) in real-time, a crucial feature for their life science-specific content generation and query handling.

Understanding the rapid innovation within the open-source community, especially regarding medical LLMs, they embraced an architectural approach that supports "pluggable" LLMs. This design choice allows them to seamlessly evaluate and integrate new or specialized medical LLMs as they emerge.

In Ryght’s platform, each LLM is registered and linked to one or more, customer-specific inference endpoints. This setup not only secures the connections, but also provides the ability to switch between different LLMs, offering unparalleled flexibility – a design choice that is made possible by the adoption of Hugging Face’s [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) and [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated).

In addition to TGI, Ryght has also integrated [Text Embeddings Inference (TEI)](https://huggingface.co/docs/text-embeddings-inference/en/index) into their ML platform. Serving open-source embedding models with TEI marks a significant improvement over relying solely on proprietary embeddings – enabling Ryght to benefit from faster inference speeds, the elimination of rate limit worries, and the flexibility to serve their own fine-tuned models, tailored to the unique requirements of the life sciences domain.

Catering to multiple customers simultaneously, their system is designed to handle high volumes of concurrent requests while maintaining low latency. Their embedding and inference services go beyond simple model invocation and encompass a suite of services adept at batching, queuing, and distributing model processing across GPUs. This infrastructure is critical to avoiding performance bottlenecks and ensuring users do not experience delays, thereby maintaining an optimal system response time.

## Conclusion
Ryght's strategic partnership with and integration of Hugging Face's ML services underscores their commitment to delivering cutting-edge solutions in healthcare and life sciences. By embracing a flexible, secure, and scalable architecture, they ensure that their platform remains at the forefront of innovation, offering their clients unparalleled service and expertise in navigating the complexities of modern medical domains. 

[Sign up for Ryght Preview](https://hubs.li/Q02sLFl_0), now publicly available to life sciences knowledge workers as a free, secure platform with frictionless onboarding. Ryght’s copilot library consists of a diverse collection of tools to accelerate information retrieval, synthesis and structuring of complex unstructured data, and document builders, taking what might have taken weeks to complete down to days or hours. To inquire about custom building and collaborations, [contact their team](https://hubs.li/Q02sLG9V0) of AI experts to discuss Ryght for Enterprise.

If you’re interested to know more about Hugging Face Expert Support, please [contact us here](https://huggingface.co/contact/sales?from=support) - our team will reach out to discuss your requirements!