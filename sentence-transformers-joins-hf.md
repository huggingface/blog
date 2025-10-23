---
title: "Sentence Transformers is joining Hugging Face!"
thumbnail: /blog/assets/sentence-transformers-joins-hf/thumbnail.png
authors:
- user: tomaarsen
---

# Sentence Transformers is joining Hugging Face!

Today, we are announcing that Sentence Transformers is transitioning from Iryna Gurevych’s [Ubiquitous Knowledge Processing (UKP) Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) at the TU Darmstadt to Hugging Face. Hugging Face's [Tom Aarsen](https://huggingface.co/tomaarsen) has already been maintaining the library since late 2023 and will continue to lead the project. At its new home, Sentence Transformers will benefit from Hugging Face's robust infrastructure, including continuous integration and testing, ensuring that it stays up-to-date with the latest advancements in Information Retrieval and Natural Language Processing.

[Sentence Transformers](https://sbert.net/) (a.k.a. SentenceBERT or SBERT) is a popular open-source library for generating high-quality embeddings that capture semantic meaning. Since its inception by Nils Reimers in 2019, Sentence Transformers has been widely adopted by researchers and practitioners for various natural language processing (NLP) tasks, including semantic search, semantic textual similarity, clustering, and paraphrase mining. After years of development and training by and for the community, [over 16,000 Sentence Transformers models are publicly available on the Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers), serving more than a million monthly unique users.

*"Sentence Transformers has been a huge success story and a culmination of our long-standing research on computing semantic similarities for the whole lab. Nils Reimers has made a very timely discovery and has produced not only outstanding research outcomes, but also a highly usable tool. This continues to impact generations of students and practitioners in natural language processing and AI. I would also like to thank all the users and especially the contributors, without whom this project would not be what it is today. And finally, I would like to thank Tom and Hugging Face for taking the project into the future."*

- **Prof. Dr. Iryna Gurevych**, Director of the Ubiquitous Knowledge Processing Lab, TU Darmstadt

*"We're thrilled to officially welcome Sentence Transformers into the Hugging Face family! Over the past two years, it’s been amazing to see this project grow to massive global adoption, thanks to the incredible foundation from the UKP Lab and the amazing community around it. This is just the beginning: we’ll keep doubling down on supporting its growth and innovation, while staying true to the open, collaborative spirit that made it thrive in the first place."*

- **Clem Delangue**, co-founder & CEO, Hugging Face

Sentence Transformers will remain a **community-driven**, **open-source** project, with the same **open-source license (Apache 2.0)** as before. Contributions from researchers, developers, and enthusiasts are welcome and encouraged. The project will continue to prioritize transparency, collaboration, and broad accessibility.

## Project History

The [Sentence Transformers library](https://github.com/UKPLab/sentence-transformers) was introduced in 2019 by Dr. Nils Reimers at the Ubiquitous Knowledge Processing (UKP) Lab at Technische Universität Darmstadt, under the supervision of Prof. Dr. Iryna Gurevych. Motivated by the limitations of standard BERT embeddings for sentence-level semantic tasks, [Sentence-BERT](https://arxiv.org/abs/1908.10084) used a Siamese network architecture to produce semantically meaningful sentence embeddings that could be efficiently compared using cosine similarity. Thanks to its modular, open-source design and strong empirical performance on tasks such as semantic textual similarity, clustering, and information retrieval, the library quickly became a staple in the NLP research toolkit, spawning a range of follow-up work and real-world applications that rely on high-quality sentence representations. 

In 2020, multilingual support was added to the library, extending sentence embeddings to more than **400 languages**. In 2021, with contributions from Nandan Thakur and Dr. Johannes Daxenberger, the library expanded to support pair-wise sentence scoring using Cross Encoder and Sentence Transformer models. Sentence Transformers was also integrated with the Hugging Face Hub (v2.0). For over four years, the UKP Lab team maintained the library as a community-driven open-source project and provided continued research-driven innovation. During this period, the project's development was supported by grants to Prof. Gurevych by the German Research Foundation (DFG), German Federal Ministry of Education and Research (BMBF), and Hessen State Ministry for Higher Education, Research and the Arts (HMWK).

In late 2023, Tom Aarsen from Hugging Face took over maintainership of the library, introducing modernized training for Sentence Transformer models ([v3.0](https://huggingface.co/blog/train-sentence-transformers)), as well as improvements of Cross Encoder ([v4.0](https://huggingface.co/blog/train-reranker)) and Sparse Encoder ([v5.0](https://huggingface.co/blog/train-sparse-encoder)) models.

## Acknowledgements

The Ubiquitous Knowledge Processing (UKP) Lab at Technische Universität Darmstadt, led by Prof. Dr. Iryna Gurevych, is internationally recognized for its research in natural language processing (NLP) and machine learning. The lab has a long track record of pioneering work in representation learning, large language models, and information retrieval, with numerous publications at leading conferences and journals. Beyond Sentence Transformers, the UKP Lab has developed a number of widely used datasets, benchmarks, and open-source tools that support both academic research and real-world applications.

Hugging Face would like to thank the UKP Lab and all past and present contributors, especially Dr. Nils Reimers and Prof. Dr. Iryna Gurevych, for their dedication to the project and for entrusting us with its maintenance and now stewardship. We also extend our gratitude to the community of researchers, developers, and practitioners who have contributed to the library's success through model contributions, bug reports, feature requests, documentation improvements, and real-world applications. We are excited to continue building on the strong foundation laid by the UKP Lab and to work with the community to further advance the capabilities of Sentence Transformers.

## Getting Started

For those new to Sentence Transformers or looking to explore its capabilities:

- **Documentation**: [https://sbert.net](https://sbert.net)  
- **GitHub Repository**: [https://github.com/huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers)  
- **Models on Hugging Face Hub**: [https://huggingface.co/models?library=sentence-transformers](https://huggingface.co/models?library=sentence-transformers)  
- **Quick Start Tutorial**: [https://sbert.net/docs/quickstart.html](https://sbert.net/docs/quickstart.html)