---
title: "XLSCOUT Unveils ParaEmbed 2.0: a Powerful Embedding Model Tailored for Patents and IP with Expert Support from Hugging Face"
thumbnail: /blog/assets/xlscout-case-study/thumbnail.png
authors:
  - user: andrewrreed
  - user: Khushwant78
    guest: true
    org: xlscout-ai
---

# XLSCOUT Unveils ParaEmbed 2.0: a Powerful Embedding Model Tailored for Patents and IP with Expert Support from Hugging Face

> [!NOTE] This is a guest blog post by the XLSCOUT team.

[XLSCOUT](https://xlscout.ai/), a Toronto-based leader in the use of AI in intellectual property (IP), has developed a powerful proprietary embedding model called **ParaEmbed 2.0** stemming from an ambitious collaboration with Hugging Face’s Expert Support Program. The collaboration focuses on applying state-of-the-art AI technologies and open-source models to enhance the understanding and analysis of complex patent documents including patent-specific terminology, context, and relationships. This allows XLSCOUT’s products to offer the best performance for drafting patent applications, patent invalidation searches, and ensuring ideas are novel compared to previously available patents and literature.
By fine-tuning on high-quality, multi-domain patent data curated by human experts, ParaEmbed 2.0 boasts **a remarkable 23% increase in accuracy** compared to its predecessor, [ParaEmbed 1.0](https://xlscout.ai/pressrelease/xlscout-paraembed-an-embedding-model-fine-tuned-on-patent-and-technology-data-is-now-opensource-and-available-on-hugging-face), which was released in October 2023. With this advancement, ParaEmbed 2.0 is now able to accurately capture context and map patents against prior art, ideas, products, or standards with even greater precision.

## The journey towards enhanced patent analysis

Initially, XLSCOUT explored proprietary AI models for patent analysis, but found that these closed-source models, such as GPT-4 and text-embedding-ada-002, struggled to capture the nuanced context required for technical and specialized patent claims.
By integrating open-source models like BGE-base-v1.5, Llama 2 70B, Falcon 40B, and Mixtral 8x7B and fine-tuning on proprietary patent data with guidance from Hugging Face, XLSCOUT achieved more tailored and performant solutions. This shift allowed for a more accurate understanding of intricate technical concepts and terminologies, revolutionizing the analysis and understanding of technical documents and patents.

## Collaborating with Hugging Face via the Expert Support Program

The collaboration with Hugging Face has been instrumental in enhancing the quality and performance of XLSCOUT’s solutions. Here's a detailed overview of how this partnership has evolved and its impact:

1. **Initial development and testing:** XLSCOUT initially built and tested a custom TorchServe inference server on Google Cloud Platform (GCP) with Distributed Data Parallel (DDP) for multiple serving replicas. By integrating ONNX optimizations, they achieved a performance rate of approximately 300 embeddings per second.
2. **Enhanced model performance via fine-tuning:** Fine-tuning of an embedding model was performed using data curated by patent experts. This workflow not only enabled more precise and contextually relevant embeddings, but also significantly improved the performance metrics, ensuring higher accuracy in detecting relevant prior art.
3. **High throughput serving:** By leveraging Hugging Face’s [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated)s with built-in load balancing, XLSCOUT now serves embedding models with [Text Embedding Inference (TEI)](https://huggingface.co/docs/text-embeddings-inference/en/index) for a high throughput use case running successfully in production. The solution now achieves impressive performance, **delivering 2700 embeddings per second!**
4. **LLM prompting and inference:** The collaboration included efforts in LLM prompt engineering and inference, which enhanced the model's ability to generate accurate and context-specific patent drafts. Prompt engineering was employed for patent drafting use cases, ensuring that the prompts resulted in coherent, comprehensive, and legally-sound patent documents.
5. **Fine-tuning LLMs with instruction data:** Instruction data formatting and fine-tuning were implemented using models from Meta and Mistral. This fine-tuning allowed for even more precise and detailed generation of some parts of the patent drafting process, further improving the quality of the generated output.

The partnership with Hugging Face has been a game-changer for XLSCOUT, significantly improving the processing speed, accuracy, and overall quality of their LLM-driven solutions. This collaboration ensures that universities, law firms, and other clients benefit from cutting-edge AI technologies, driving efficiency and innovation in the patent landscape.

## XLSCOUT's AI-based IP Solutions

XLSCOUT provides state-of-the-art AI-driven solutions that significantly enhance the efficiency and accuracy of patent-related processes. Their solutions are widely leveraged by corporations, universities, and law firms to streamline various facets of IP workflows, from novelty searches and invalidation studies to patent drafting.

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/xlscout-solutions.png" alt="XLSCOUT Solutions" style="width: 90%; height: auto;"><br>
</p>

- **[Novelty Checker LLM](https://xlscout.ai/novelty-checker-llm):** Leverages cutting-edge LLMs and Generative AI to swiftly navigate through patent and non-patent literature to validate your ideas. It delivers a comprehensive list of ranked prior art references alongside a key feature analysis report. This tool enables inventors, researchers, and patent professionals to ensure that inventions are novel by comparing them against the extensive corpus of existing literature and patents.
- **[Invalidator LLM](https://xlscout.ai/invalidator-llm):** Utilizes advanced LLMs and Generative AI to conduct patent invalidation searches with exceptional speed and accuracy. It provides a detailed list of ranked prior art references and a key feature analysis report. This service is crucial for law firms and corporations to efficiently challenge and assess the validity of patents.
- **[Drafting LLM](https://xlscout.ai/drafting-llm):** Is an automated patent application drafting platform harnessing the power of LLMs and Generative AI. It generates precise and high-quality preliminary patent drafts, encompassing comprehensive claims, abstracts, drawings, backgrounds, and descriptions within a few minutes. This solution aids patent practitioners in significantly reducing the time and effort required to produce detailed and precise patent applications.

Corporations and universities benefit by ensuring that novel research outputs are appropriately protected, encouraging innovation, and filing high quality patents. Law firms utilize XLSCOUT’s solutions to deliver superior service to their clients, improving the quality of their patent prosecution and litigation efforts.

## A partnership for innovation

_“We are thrilled to collaborate with Hugging Face”_, said Mr. Sandeep Agarwal, CEO of XLSCOUT. _“This partnership combines the unparalleled capabilities of Hugging Face's open-source models, tools, and team with our deep expertise in patents. By fine-tuning these models with our proprietary data, we are poised to revolutionize how patents are drafted, analyzed, and licensed.”_
The joint efforts of XLSCOUT and Hugging Face involve training open-source models on XLSCOUT’s extensive patent data collection. This synergy harnesses the specialized knowledge of XLSCOUT and the advanced AI capabilities of Hugging Face, resulting in models uniquely optimized for patent research. Users will benefit from more informed decisions and valuable insights derived from complex patent documents.

## Commitment to innovation and future plans

As pioneers in the application of AI to intellectual property, XLSCOUT is dedicated to exploring new frontiers in AI-driven innovation. This collaboration marks a significant step towards bridging the gap between cutting-edge AI and real-world applications in IP analysis.
Together, XLSCOUT and Hugging Face are setting new standards in patent analysis, driving innovation, and shaping the future of intellectual property. We’re excited to continue this awesome journey together!
To learn more about Hugging Face’s Expert Support Program for your company, please [get in touch with us here](https://huggingface.co/support#form) - our team will contact you to discuss your requirements!
