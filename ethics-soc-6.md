---
title: "Ethics and Society Newsletter #6: Building Better AI: The Importance of Data Quality"
thumbnail: assets/182_ethics-soc-6/thumbnail.png
authors:
- user: evijit
- user: frimelle
- user: yjernite
- user: meg
- user: irenesolaiman
- user: dvilasuero
- user: fdaudens
- user: BrigitteTousi
- user: giadap
- user: sasha
---


# Ethics and Society Newsletter #6: Building Better AI: The Importance of Data Quality


In February, Reddit announced a [new content partnership with Google](https://www.cnet.com/tech/services-and-software/reddits-60-million-deal-with-google-will-feed-generative-ai/) where they would provide data that would power the new Generative AI based search engine using Retrieval Augmented Generation (RAG). [That attempt did not go as planned](https://www.technologyreview.com/2024/05/31/1093019/why-are-googles-ai-overviews-results-so-bad), and soon, people were seeing recommendations like adding [glue to pizza](https://www.theverge.com/2024/6/11/24176490/mm-delicious-glue):


<p align="center">
  <img src="https://huggingface.co/datasets/society-ethics/dataqualityblog/resolve/main/glueonpizza.png" />
</p>

In the age of artificial intelligence, [massive amounts of data](https://arxiv.org/abs/2401.00676) fuel the growth and sophistication of machine learning models. But not all data is created equal; AI systems [require](https://dl.acm.org/doi/abs/10.1145/3394486.3406477) [high-quality](https://arxiv.org/abs/2212.05129) [data](https://proceedings.neurips.cc/paper/1994/hash/1e056d2b0ebd5c878c550da6ac5d3724-Abstract.html) to produce [high-quality](https://dl.acm.org/doi/abs/10.1145/3447548.3470817) [outputs](https://arxiv.org/abs/1707.02968).

So, what makes data "high-quality," and why is it crucial to prioritize data quality from the outset? Achieving data quality is not just a matter of accuracy or quantity; it requires a [holistic, responsible approach](https://huggingface.co/blog/ethics-soc-3) woven throughout the entire AI development lifecycle. As data quality has garnered [renewed ](https://twitter.com/Senseye_Winning/status/1791007128578322722)attention, we explore what constitutes "high quality" data, why prioritizing data quality from the outset is crucial, and how organizations can utilize AI for beneficial initiatives while mitigating risks to privacy, fairness, safety, and sustainability.

In this article, we first provide a high-level overview of the relevant concepts, followed by a more detailed discussion.

## What is Good, High-Quality Data?

**Good data isn't just accurate or plentiful; it's data fit for its intended purpose**. Data quality must be evaluated based on the specific use cases it supports. For instance, the pretraining data for a heart disease prediction model must include detailed patient histories, current health status, and precise medication dosages, but in most cases, should not require patients' phone numbers or addresses for privacy. [The key is to match the data to the needs of the task at hand](https://arxiv.org/pdf/2012.05345). From a policy standpoint, consistently advocating for [a safety-by-design approach](https://huggingface.co/blog/policy-blog) towards responsible machine learning is crucial. This includes taking thoughtful steps at the data stage itself. [Desirable aspects](https://www.iso.org/standard/35749.html) of data quality include (but are not limited to!): 



* **Relevance:** The data must be directly applicable and meaningful to the specific problem the AI model is trying to solve. Irrelevant data can introduce noise, i.e., random errors or irrelevant information in the data that can obscure the underlying patterns and lead to poor performance or unintended consequences. “Relevance” is [widely](https://books.google.com/books?hl=en&lr=&id=Vh29JasHbKAC&oi=fnd&pg=PA105&dq=data+quality+relevance&ots=qFosiBsUKf&sig=AS6vMhOPDjRgMO6CrRnWd6B3Iyk#v=onepage&q=data%20quality%20relevance&f=false) [recognized](https://cdn.aaai.org/Symposia/Fall/1994/FS-94-02/FS94-02-034.pdf) as [critical](https://ieeexplore.ieee.org/abstract/document/7991050) [across](https://openproceedings.org/2024/conf/edbt/tutorial-1.pdf) [work](https://link.springer.com/content/pdf/10.1023/A:1007612503587.pdf) [on](https://ai.stanford.edu/~ronnyk/ml94.pdf) data quality, as it provides for control over what a system may or may not do and helps optimize statistical estimates.
* **Comprehensiveness:** The data should capture the full breadth and diversity of the real-world scenarios the AI will encounter. Incomplete or narrow datasets can lead to biases and overlooked issues. This is also known as [“Completeness”](https://www.iso.org/standard/35749.html) in data quality work.
* **Timeliness:** Particularly for rapidly evolving domains, the data must be up-to-date and reflect the current state of affairs. Outdated information can render an AI system ineffective or even dangerous. This is also known as [“Currentness”](https://www.iso.org/standard/35749.html) and [“Freshness”](https://ieeexplore.ieee.org/abstract/document/9343076) in work on data quality.
* **Mitigation of Biases:** Collecting data brings with it biases in everything from the data sources to the collection protocols. Data selection work must therefore make every effort to avoid encoding unintended harmful biases, which can result in systems that exacerbate patterns of societal oppression, stereotypes, discrimination, and underrepresentation of marginalized groups. 

While we have focused on a subset of data quality measures, many more measures have been defined that are useful for machine learning datasets, such as [traceability and consistency](https://www.iso.org/standard/35749.html).

## Why Data Quality?

Investing in data quality is fundamental for improving AI model performance. In an era where AI and machine learning are increasingly integrated into decision-making processes, ensuring data quality is not just beneficial but essential. Properly curated data allows AI systems to function more effectively, accurately, and fairly. It supports the development of models that can handle diverse scenarios, promotes sustainable practices by optimizing resource usage, and upholds ethical standards by mitigating biases and enhancing transparency. Some key motivators of data quality:



* **Enhanced Model Outcomes:** High-quality data improves model performance by eliminating noise, correcting inaccuracies, and standardizing formats.
* **Robustness and Generalization:** Diverse, multi-source data prevents overfitting and ensures that models are robust across various real-world scenarios. Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization.
* **Efficiency:** High-quality data leads to more efficient, compact models that require fewer computational resources.
* **Representation and Inclusivity:** High-quality data should be representative and inclusive, which helps address biases, promote equity, and ensure the representation of diverse societal groups.
* **Governance and Accountability:** Practices such as transparency about data sources, preprocessing, and provenance ensure effective AI governance and accountability.
* **Scientific Reproducibility:** High-quality data is crucial for open science as it ensures the validity of the findings and facilitates reproducibility and further research. 

## What is the Process toward Data Quality?

The process toward high-quality datasets involves several key strategies. Meticulous data curation and preprocessing, such as deduplication, content filtering, and human feedback, e.g., through domain expertise and stakeholder feedback, are essential to maintain dataset relevance and accuracy to the task at hand. [Participatory data collection](https://en.unesco.org/inclusivepolicylab/node/1242) and [open community contributions](https://huggingface.co/blog/community-update) enhance representation and inclusivity. Establishing a robust data governance framework with clear policies, standards, and accountability ensures consistent data management. Regular quality assessments using metrics like accuracy and completeness help identify and rectify issues. Thorough documentation, including dataset cards, improves usability, collaboration, and transparency. Lastly, while synthetic data can be beneficial, it should be used alongside real-world data and validated rigorously to prevent biases and ensure model performance. Some approaches to data quality include:



* [Dataset Cards](https://huggingface.co/docs/hub/en/datasets-cards)
* [DataTrove](https://github.com/huggingface/datatrove)
* [Data is better together initiative](https://huggingface.co/DIBT) and human feedback collection with [Argilla](https://github.com/argilla-io/argilla)
* [Data measurement tool](https://huggingface.co/blog/data-measurements-tool)
* [Large-scale Near-deduplication Behind BigCode](https://huggingface.co/blog/dedup)
* Dataset examples: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS), [The Stack V2](https://huggingface.co/datasets/bigcode/the-stack-v2)
* [Policy Questions Blog 1: AI Data Transparency Remarks for NAIAC Panel](https://huggingface.co/blog/yjernite/naiac-data-transparency)
* [📚 Training Data Transparency in AI: Tools, Trends, and Policy Recommendations 🗳️](https://huggingface.co/blog/yjernite/data-transparency)

## Data Quality for Improving Model Performance

Investing in data quality is crucial for enhancing the performance of AI systems. Numerous studies have demonstrated that [better data quality directly correlates with improved model outcomes](https://aclanthology.org/2022.acl-long.577/#:~:text=Deduplication%20allows%20us%20to%20train,the%20same%20or%20better%20accuracy), as most recently seen in the [Yi 1.5 model release](https://x.com/Dorialexander/status/1789709739695202645).  Achieving high data quality involves meticulous data cleaning and preprocessing to remove noise, correct inaccuracies, fill in missing values, and standardize formats. Incorporating diverse, multi-source data prevents overfitting and exposes models to a wide range of real-world scenarios. 

The benefits of high-quality data extend beyond improved metrics. Cleaner, smaller datasets allow models to be more [compact and parameter-efficient](https://arxiv.org/abs/2203.15556), requiring fewer computational resources and energy for training and inference. 

## Data Quality for Improving Representation

Another crucial aspect of data quality is representation. Models are often trained on [training data that over-represents dominant groups and perspectives](https://www.image-net.org/update-sep-17-2019.php), resulting in [skewed object representations](https://www.washingtonpost.com/technology/interactive/2023/ai-generated-images-bias-racism-sexism-stereotypes/), imbalanced [occupational and location biases](https://arxiv.org/abs/2303.11408), or the [consistent depiction of harmful stereotypes](https://researchportal.bath.ac.uk/en/publications/semantics-derived-automatically-from-language-corpora-necessarily). This means including data from all groups in society and capturing a wide range of languages, especially in text data. Diverse representation helps mitigate cultural biases and improves model performance across different populations. An example of such a dataset is [CIVICS](https://arxiv.org/abs/2405.13974).

Participatory approaches are key to achieving this. [By involving a larger number of stakeholders in the data creation process](https://arxiv.org/pdf/2405.06346), we can ensure that the data used to train models is more inclusive. Initiatives like ["Data is Better Together"](https://huggingface.co/DIBT) encourage community contributions to datasets, enriching the diversity and quality of the data. Similarly, the [Masakhane project](https://www.masakhane.io/) focuses on creating datasets for African languages, such as [evaluation datasets](https://huggingface.co/datasets/masakhane/afrimgsm), which have been underrepresented in AI research. These efforts ensure that AI systems are more equitable and effective across different contexts and populations, ultimately fostering more inclusive technological development.

## Data Quality for Governance and Accountability

[Maintaining high data quality ](https://arxiv.org/abs/2206.03216)practices is essential for enabling effective governance and accountability of AI systems. Transparency about data sources, licenses, and any preprocessing applied is crucial. Developers should provide clear documentation around [data provenance](https://arxiv.org/abs/2310.16787), including where the data originated, how it was collected, and any transformations it underwent.

[This transparency](https://huggingface.co/blog/yjernite/data-transparency) empowers external audits and oversight, allowing for thorough examination and validation of the data used in AI models. Clear documentation and data traceability also help identify potential issues and implement mitigation strategies. This level of transparency is critical for building trust and facilitating responsible AI development, ensuring that AI systems operate ethically and responsibly.

## Data Quality for Adaptability and Generalizability

Another critical aspect is ensuring that [data reflects the diversity required for AI models to adapt and generalize across contexts](https://vitalab.github.io/article/2019/01/31/Diversity_In_Faces.html). This involves capturing a wide range of languages, cultures, environments, and edge cases representative of the real world. [Participatory data collection](https://en.unesco.org/inclusivepolicylab/node/1242) approaches involving impacted communities can enrich datasets and improve representation, ensuring robust and adaptable models.

[Continuously evaluating model performance across different demographics](https://arxiv.org/pdf/2106.07057) is key to identifying generalizability gaps. Achieving adaptable AI hinges on continuous data collection and curation processes that ingest real-world feedback loops. As new products are released or business landscapes shift, the [training data should evolve in lockstep](https://www.decube.io/post/data-freshness-concepts) to reflect these changes. Developers should implement [processes to identify data drifts and model performance drops](https://ieeexplore.ieee.org/document/4811799) compared to the current state, ensuring the AI models remain relevant and effective in changing environments.


## Data Quality for Scientific Reproducibility and Replicability

In the research realm, data quality has profound implications for the reproducibility and validity of findings. Poor quality training data can [undermine the integrity of experiments and lead to non-reproducible results](https://arxiv.org/abs/2307.10320). Stringent data quality practices, such as [meticulous documentation of preprocessing steps and sharing of datasets](https://nap.nationalacademies.org/read/25303/chapter/9#119), enable other researchers to scrutinize findings and build upon previous work.

Replicability, [defined as the process of arriving at the same scientific findings using new data](https://www.ncbi.nlm.nih.gov/books/NBK547546/#:~:text=B1%3A%20%E2%80%9CReproducibility%E2%80%9D%20refers%20to,findings%20as%20a%20previous%20study.), is a bit more nuanced. Sometimes, the non-replicability of a study can actually aid in scientific progress by [expanding research from a narrow applied field into broader areas](https://nap.nationalacademies.org/read/25303/chapter/9#chapter06_pz161-4). Regardless, replicability is also difficult without proper documentation of data collection procedures and training methodology, and the current [reproducibility and replicability crisis](https://arxiv.org/abs/2307.10320) in AI can be significantly ameliorated by high-quality, well-documented data. 

## High-Quality Data needs High-Quality Documentation

One of the crucial aspects for high-quality data, just as for code, is the thorough documentation of the data. Proper documentation enables users to understand the content and context of the data, facilitating better decision-making and enhancing the transparency and reliability of AI models. One of the innovative approaches to data documentation is using [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards), as offered by the Hugging Face hub. There are various methods to document data including [data statements](https://techpolicylab.uw.edu/data-statements/), [datasheets](https://www.fatml.org/media/documents/datasheets_for_datasets.pdf), [data nutrition labels](https://datanutrition.org/labels/), [dataset cards](https://aclanthology.org/2021.emnlp-demo.21/), and [dedicated research papers](https://nips.cc/Conferences/2023/CallForDatasetsBenchmarks). Usually these documentation methods cover data sources and composition of the dataset, processing steps, descriptive statistics including demographics represented in the dataset, and the original purpose of the dataset ([see for more details on the importance of data transparency](https://huggingface.co/blog/yjernite/naiac-data-transparency)). Data documentation, such as dataset cards, can help with:

* **Enhanced Usability:** By providing a clear and comprehensive overview of the dataset, dataset cards make it easier for users to understand and utilize the data effectively.
* **Improved Collaboration:** Detailed documentation fosters better communication and collaboration, as everyone has a shared understanding of the data.
* **Informed Decision-Making:** With access to detailed information about the data, users can make more informed decisions regarding its application and suitability for various tasks.
* **Transparency and Accountability:** Thorough documentation promotes transparency and accountability in data management, building trust among users and stakeholders.

## A Note on Synthetic Data

Synthetic data has emerged as a [cost-efficient alternative to real-world data](https://huggingface.co/blog/synthetic-data-save-costs), providing a scalable solution for training and testing AI models without the expenses and privacy concerns associated with collecting and managing large volumes of real data, as done for example in [Cosmopedia](https://huggingface.co/blog/cosmopedia). This approach enables organizations to generate diverse datasets tailored to specific needs, accelerating development cycles and reducing costs. However, it is crucial to be aware of the potential downsides. Synthetic data can inadvertently [introduce biases](https://facctconference.org/static/papers24/facct24-117.pdf) if the algorithms generating the data are themselves biased, [leading to skewed model outcome](https://facctconference.org/static/papers24/facct24-144.pdf)s. It is important to [mark model output as generated content](https://huggingface.co/blog/alicia-truepic/identify-ai-generated-content), e.g., by [watermarking](https://huggingface.co/blog/watermarking) [across](https://huggingface.co/blog/imatag-vch/stable-signature-bzh) [modalities](https://arxiv.org/abs/2401.17264) ([overview](https://huggingface.co/collections/society-ethics/provenance-watermarking-and-deepfake-detection-65c6792b0831983147bb7578)). Additionally, over-reliance on synthetic data can result in [model collapse](https://en.wikipedia.org/wiki/Model_collapse), where the model becomes overly tuned to the synthetic data patterns. Therefore, while synthetic data is a powerful tool, it should be used judiciously, complemented by real-world data and robust validation processes to ensure model performance and fairness.

## The Process Toward Data Quality

Ensuring high data quality is essential for developing effective and reliable AI models. Here are several strategies for approaching data quality, illustrated with examples from Hugging Face's practices.

A crucial aspect of data quality is filtering and deduplication. For instance, in creating large, high-quality datasets like [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Hugging Face employs tools such as [DataTrove](https://github.com/huggingface/datatrove). Filtering involves selecting only relevant and high-quality data, ensuring that the dataset is comprehensive without unnecessary noise. Deduplication removes redundant entries, which improves the efficiency and performance of AI models. This meticulous approach ensures that the dataset remains robust and relevant.

Responsible multi-modal data creation is another key area where Hugging Face has set an example. The [OBELICS dataset](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) showcases several best practices in this regard. One significant practice is opt-out filtering, where images that have been opted out of redistribution or model training are removed using APIs like Spawning. This respects the rights and preferences of content creators. Additionally, deduplication ensures that images appear no more than ten times across the dataset, reducing redundancy and ensuring diverse representation. Content filtering is also essential; employing open-source classifiers to detect and exclude NSFW content, and filtering images based on their URLs, maintains the dataset's appropriateness and relevance.

Handling diverse data types is yet another strategy employed by Hugging Face. In creating [The Stack V2](https://huggingface.co/datasets/bigcode/the-stack-v2), which covers a broad range of programming languages and frameworks, careful selection of repositories and projects was done to ensure diversity and comprehensiveness. Quality checks, both automated and manual, verify the syntactic correctness and functional relevance of the code in the dataset, maintaining its high quality - for example, the [efforts in deduplication in the BigCode project](https://huggingface.co/blog/dedup).  

Gathering human feedback using data labeling tools (like [Argilla](https://argilla.io/blog/launching-argilla-huggingface-hub/)) can have a significant impact on data quality, especially by including stakeholders in the data creation process. Examples of this include the [improvement of the UltraFeedback dataset through human curation](https://argilla.io/blog/notus7b/), leading to Notus, an improved version of the [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model, or the community efforts of the [Data is Better Together initiative](https://github.com/huggingface/data-is-better-together).  

Beyond these specific practices, there are general strategies that can ensure data quality. Establishing a robust data governance framework is foundational. This framework should include policies, standards, and processes for data management, with clearly defined roles and responsibilities to ensure accountability and maintain high standards. Regular quality assessments are also vital. These assessments, which can utilize metrics like accuracy, completeness, consistency, and validity, help identify and address issues early. Tools such as data profiling and statistical analysis can be instrumental in this process.

## Are you working on data quality? Share your tools and methods on Hugging Face Hub!
The most important part of Hugging Face is our community. If you're a researcher focused on improving data quality in machine learning, especially within the context of open science, we want to support and showcase your work!

Thanks for reading! 🤗

~ Avijit and Lucie, on behalf of the Ethics & Society regulars

If you want to cite this blog post, please use the following (authors in alphabetical order):


```
@misc{hf_ethics_soc_blog_6,
  author    = {Avijit Ghosh and Lucie-Aimée Kaffee},
  title     = {Hugging Face Ethics and Society Newsletter 6: Building Better AI: The Importance of Data Quality},
  booktitle = {Hugging Face Blog},
  year      = {2024},
  url       = {https://huggingface.co/blog/ethics-soc-6},
  doi       = {10.57967/hf/2610}
}
```
