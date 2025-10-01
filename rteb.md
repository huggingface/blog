---
title: "Introducing RTEB: A New Standard for Retrieval Evaluation"
thumbnail: /blog/assets/rteb/thumbnail.png
authors:
- user: fzliu
  guest: true
  org: MongoDB
- user: KennethEnevoldsen
  guest: true
  org: mteb
- user: Samoed
  guest: true
  org: mteb
- user: isaacchung
  guest: true
  org: mteb
- user: tomaarsen
  org: mteb
- user: fzoll
  guest: true
  org: MongoDB
---

# Introducing RTEB: A New Standard for Retrieval Evaluation

**TL;DR –** We’re excited to introduce the beta version of the Retrieval Embedding Benchmark (RTEB), a new benchmark designed to reliably evaluate the retrieval accuracy of embedding models for real-world applications. Existing benchmarks struggle to measure true generalization, while RTEB addresses this with a hybrid strategy of open and private datasets. Its goal is simple: to create a fair, transparent, and application-focused standard for measuring how models perform on data they haven’t seen before.

The performance of many AI applications, from RAG and agents to recommendation systems, is fundamentally limited by the quality of search and retrieval. As such, accurately measuring the retrieval quality of embedding models is a common pain point for developers. How do you *really* know how well a model will perform in the wild?

This is where things get tricky. The current standard for evaluation often relies on a model's "zero-shot" performance on public benchmarks. However, this is, at best, an approximation of a model's true generalization capabilities. When models are repeatedly evaluated against the same public datasets, a gap emerges between their reported scores and their actual performance on new, unseen data.

<figure class="image text-center" id="figure1">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rteb/rteb-public-vs-closed.png">
  <figcaption>Performance Discrepancy Between Public and Closed Datasets</figcaption>
</figure>

To address these challenges, we developed RTEB, a benchmark built to provide a reliable standard for evaluating retrieval models.

## Why Existing Benchmarks Fall Short

While the underlying evaluation methodology and metrics (such as NDCG@10) are well-known and robust, the integrity of existing benchmarks is often set back by the following issues:

**The Generalization Gap**. The current benchmark ecosystem inadvertently encourages "teaching to the test." When training data sources overlap with evaluation datasets, a model's score can become inflated, undermining a benchmark's integrity. This practice, whether intentional or not, is evident in the training datasets of several models. This creates a feedback loop where models are rewarded for memorizing test data rather than developing robust, generalizable capabilities.

Because of the above, models with a lower zero-shot score<a href="#footnote-1">[1]</a> may perform very well on the benchmark, without generalizing to new problems. For this reason, models with slightly lower benchmark performance and a higher zero-shot score are often recommended instead.

<figure class="image text-center" id="figure2">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rteb/mteb-zero-shot-models.png">
  <figcaption>From <a href="https://arxiv.org/abs/2506.21182">Chung et al. (2025)</a></figcaption>
</figure>

**Misalignment with Today’s AI Applications**. Many benchmarks are poorly aligned with the enterprise use cases that developers are building today. They often rely on academic datasets or on retrieval tasks derived from QA datasets, which, while useful in their own right, were not designed to evaluate retrieval and can fail to capture the distributional biases and complexities encountered in real-world retrieval scenarios. Benchmarks which do not possess these issues are often too narrow, focusing on a single domain like code retrieval, making them unsuitable for evaluating general-purpose models.

## Introducing RTEB

Today, we’re excited to introduce the **Retrieval Embedding Benchmark (RTEB)**. Its goal is to create a new, reliable, high-quality benchmark that measures the true retrieval accuracy of embedding models.

### A Hybrid Strategy for True Generalization

To combat benchmark overfitting, RTEB implements a hybrid strategy using both open and private datasets:

* **Open Datasets:** The corpus, queries, and relevance labels are fully public. This ensures transparency and allows any user to reproduce the results.
* **Private Datasets:** These datasets are kept private, and evaluation is handled by the MTEB maintainers to ensure impartiality. This setup provides a clear, unbiased measure of a model’s ability to generalize to unseen data. For transparency, we provide descriptive statistics, a dataset description, and sample `(query, document, relevance)` triplets for each private dataset.

This hybrid approach encourages the development of models with broad, robust generalization. A model with a significant performance drop between the open and the private datasets would suggest overfitting, providing a clear signal to the community. This is already apparent with some models, which show a notable drop in performance on RTEB's private datasets.

### Built for Real-World Domains

RTEB is designed with a particular emphasis on enterprise use cases. Instead of a complex hierarchy, it uses simple groups for clarity. A single dataset can belong to multiple groups (e.g., a German law dataset exists in both the "law" and "German" groups).

* **Multilingual in Nature:** The benchmark datasets cover 20 languages, from common ones like English or Japanese to rarer languages such as Bengali or Finnish.
* **Domain-Specific Focus:** The benchmark includes datasets from critical enterprise domains like law, healthcare, code, and finance.
* **Efficient Dataset Sizes:** Datasets are large enough to be meaningful (at least 1k documents and 50 queries) without being so large that they make evaluation time-consuming and expensive.
* **Retrieval-First Metric:** The default leaderboard metric is **NDCG@10**, a gold-standard measure for the quality of ranked search results.

A complete list of the datasets can be found below. We plan to continually update both the open as well as closed portion with different categories of datasets and actively encourage participation from the community; please open an issue on the [MTEB repository on GitHub](https://github.com/embeddings-benchmark/mteb/issues) if you would like to suggest other datasets.


<details>
  <summary>RTEB Datasets</summary>

#### Open

| Dataset | Dataset Groups | Open/Closed | Dataset URL | Repurposed from QA | Description and Reason for Inclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AILACasedocs | english, legal | Open | https://huggingface.co/datasets/mteb/AILA_casedocs | No | This dataset comprises approximately 3,000 Supreme Court of India case documents and is designed to evaluae the retrieval of relevant prior cases for given legal situations. It includes 50 queries, each outlining a specific scenario. We include this dataset in the benchmark because the documents are reasonably challenging, the queries are non-synthetic, and the labels are of high quality. |
| AILAStatutes | english, legal | Open | https://huggingface.co/datasets/mteb/AILA_statutes | No | The dataset comprises descriptions of 197 Supreme Court of India statutes, designed to facilitate the retrieval of relevant prior statutes for given legal situations. It includes 50 queries, each outlining a specific scenario. We include this dataset in the benchmark because the documents are reasonably challenging, the queries are non-synthetic, and the labels are of high quality. |
| LegalSummarization | english, legal | Open | https://huggingface.co/datasets/mteb/legal_summarization | No | The dataset comprises 446 pairs of legal text excerpts and their corresponding plain English summaries, sourced from reputable websites dedicated to clarifying legal documents. The summaries have been manually reviewed for quality, ensuring that the data is clean and suitable for evaluating legal retrieval. |
| LegalQuAD | german, legal | Open | https://huggingface.co/datasets/mteb/LegalQuAD | No | The corpus consists of 200 real-world legal documents and the query set consists of 200 questions pertaining to legal documents. |
| FinanceBench | english, finance | Open | https://huggingface.co/datasets/virattt/financebench | Yes | The FinanceBench dataset is derived from the PatronusAI/financebench-test dataset, containing only the PASS examples processed into a clean format for question-answering tasks in the financial domain. FinanceBench-rtl has been repurposed for retrieval. |
| HC3Finance | english, finance | Open | https://huggingface.co/datasets/Hello-SimpleAI/HC3 | No | The HC3 dataset comprises tens of thousands of comparison responses from both human experts and ChatGPT across various domains, including open-domain, financial, medical, legal, and psychological areas. The data collection process involved sourcing publicly available question-answering datasets and wiki texts, ensuring that the human answers were either expert-provided or high-quality user responses, thereby minimizing mislabeling and enhancing the dataset's reliability. |
| FinQA | english, finance | Open | https://huggingface.co/datasets/ibm/finqa | Yes | FinQA is a large-scale dataset with 2.8k financial reports for 8k Q&A pairs to study numerical reasoning with structured and unstructured evidence. |
| HumanEval | code | Open | https://huggingface.co/datasets/openai/openai_humaneval | Yes | The HumanEval dataset released by OpenAI includes 164 programming problems with a handwritten function signature, docstring, body, and several unit tests for each problem. The dataset was handcrafted by engineers and researchers at OpenAI. |
| MBPP | code | Open | https://huggingface.co/datasets/google-research-datasets/mbpp | Yes | The MBPP dataset consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases. As described in the paper, a subset of the data has been hand-verified by the dataset authors to ensure quality. |
| MIRACLHardNegatives | | Open | https://huggingface.co/datasets/mteb/miracl-hard-negatives | No | MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. |
| APPS | code, english | Open | https://huggingface.co/datasets/codeparrot/apps | Yes | APPS is a benchmark for code generation with 10000 problems. It can be used to evaluate the ability of language models to generate code from natural language specifications. To create the APPS dataset, the authors manually curated problems from open-access sites where programmers share problems with each other, including Codewars, AtCoder, Kattis, and Codeforces. |
| DS1000 | code, english | Open | https://huggingface.co/datasets/xlangai/DS-1000 | Yes | DS-1000 is a code generation benchmark with a thousand data science problems spanning seven Python libraries, such as NumPy and Pandas. It employs multi-criteria evaluation metrics, including functional correctness and surface-form constraints, resulting in a high-quality dataset with only 1.8% incorrect solutions among accepted Codex-002 predictions. |
| WikiSQL | code, english | Open | https://huggingface.co/datasets/Salesforce/wikisql | Yes | WikiSQL is a dataset comprising 80,654 hand-annotated examples of natural language questions and corresponding SQL queries across 24,241 tables from Wikipedia. |
| ChatDoctor_HealthCareMagic | english, healthcare | Open | https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k | No | The ChatDoctor-HealthCareMagic-100k dataset comprises 112,000 real-world medical question-and-answer pairs, providing a substantial and diverse collection of authentic medical dialogues. There is a slight risk to this dataset since there are grammatical inconsistencies in many of the questions and answers, but this can potentially help separate strong healthcare retrieval models from weak ones. |
| HC3 Medicine | english, healthcare | Open | https://huggingface.co/datasets/Hello-SimpleAI/HC3 | No | The HC3 dataset comprises tens of thousands of comparison responses from both human experts and ChatGPT across various domains, including open-domain, financial, medical, legal, and psychological areas. The data collection process involved sourcing publicly available question-answering datasets and wiki texts, ensuring that the human answers were either expert-provided or high-quality user responses, thereby minimizing mislabeling and enhancing the dataset's reliability. |
| HC3 French OOD | french, healthcare | Open | https://huggingface.co/datasets/almanach/hc3_french_ood | No | The HC3 dataset comprises tens of thousands of comparison responses from both human experts and ChatGPT across various domains, including open-domain, financial, medical, legal, and psychological areas. The data collection process involved sourcing publicly available question-answering datasets and wiki texts, ensuring that the human answers were either expert-provided or high-quality user responses, thereby minimizing mislabeling and enhancing the dataset's reliability. |
| JaQuAD | japanese | Open | https://huggingface.co/datasets/SkelterLabsInc/JaQuAD | Yes | The JaQuAD dataset comprises 39,696 human-annotated question-answer pairs based on Japanese Wikipedia articles, with 88.7% of the contexts sourced from curated high-quality articles. |
| Cure | english, healthcare | Open | https://huggingface.co/datasets/clinia/CUREv1 | No | |
| TripClick | english, healthcare | Open | https://huggingface.co/datasets/irds/tripclick | No | |
| FreshStack | english | Open | https://huggingface.co/papers/2504.13128 | No | |

#### Closed

| Dataset | Dataset Groups | Open/Closed | Dataset URL | Comments | Repurposed from QA | Description and Reason for Inclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| _GermanLegal1 | german, legal | Closed | | Yes | | This dataset is derived from real-world judicial decisions and employs a combination of legal citation matching and BM25 similarity. The BM25 baseline poses a slight risk as it biases the data outside of citation matching. A subset of the dataset was manually verified to ensure correctness and quality. |
| _JapaneseLegal1 | japanese, legal | Closed | | No | | This dataset comprises 8.75K deduplicated law records retrieved from the official Japanese government website e-Gov, ensuring authoritative and accurate content. Record titles are used as queries, while record bodies are used as documents. |
| _FrenchLegal1 | french, legal | Closed | | No | | This dataset comprises case laws from the French court \"Conseil d'Etat,\" systematically extracted from the OPENDATA/JADE repository, focusing on tax-related cases. Queries are the title of each document, ensuring that the labels are clean. |
| _EnglishFinance1 | english, finance | Closed | | Yes | | This retrieval dataset has been repurposed for retrieval from TAT-QA, a large-scale QA dataset using tabular and textual content. |
| _EnglishFinance4 | english, finance | Closed | | No | | This dataset is a combination of Stanford's Alpaca and FiQA with another 1.3k pairs custom generated using GPT3.5, and then further cleaned to ensure that the data quality is high. |
| _EnglishFinance2 | english, finance | Closed | | Yes | | This dataset is a finance-domain dataset that is composed of questions for each conversation turn based on simulated conversation flow. The curation is done by expert annotators, ensuring a reasonably high data quality. The questions are repurposed as queries, while the conversation block is repurposed as documents for retrieval. |
| _EnglishFinance3 | english, finance | Closed | | Yes | | This dataset is a collection of question-answer pairs curated to address various aspects of personal finance. |
| _Code1 | code | Closed | | No | | We extracted functions from GIthub repos. With syntactic parsing, doc strings and function signature are obtained from the functions. Only functions with docstrings are kept. Doc strings are used as queries, with function signature (which includes function name and argument names) removed to making the task harder. Each language is a subset with separate corpus. |
| _JapaneseCode1 | code, japanese | Closed | | No | | This is a subset of the CoNaLa challenge with Japanese questions. |
| _EnglishHealthcare1 | english, healthcare | Closed | | Yes | | This dataset comprises 2,019 question-answer pairs annotated by 15 experts, each holding at least a Master's degree in biomedical sciences. A medical doctor led the annotation team, verifying each question-answer pair to ensure data quality. |
| _GermanHealthcare1 | german, healthcare | Closed | | No | | This dataset comprises of 465 German-language medical dialogues between patients and healthcare assistants, each entry containing detailed patient descriptions and corresponding professional responses. We have manually verified a subset of the dataset for accuracy and data quality. |
| _German1 | german | Closed | | No | | This dataset is a dialogue summarization dataset derived from multiple public corpora, which have cleaned and preprocessed into a unified format. Each dialogue has been manually summarized and labeled with topics by annotators, ensuring high-quality and clean data. Dialog summaries are used as queries, while full dialogues are used as documents. |
| _French1 | french | Closed | | Yes | | This dataset comprises over 4118 French trivia question-answer pairs, each accompanied by relevant Wikipedia context. We have manually verified a subset of the dataset for accuracy and data quality. |

</details>

## Launching RTEB: A Community Effort

RTEB is launching today in beta. We believe building a robust benchmark is a community effort, and we plan to evolve RTEB based on feedback from developers and researchers alike. We encourage you to share your thoughts, suggest new datasets, find issues in existing datasets and help us build a more reliable standard for everyone. Please feel free to join the discussion or open an issue in the [MTEB repository on Github](https://github.com/embeddings-benchmark/mteb).

## Limitations and Future Work

To highlight areas for improvement we want to be transparent about RTEB's current limitations and our plans for the future.

* **Benchmark Scope:** RTEB is focused on realistic, retrieval-first use cases. Highly challenging synthetic datasets are not a current goal but could be added in the future.
* **Modality:** The benchmark currently evaluates text-only retrieval. We plan to incorporate text-image and other multimodal retrieval tasks in future releases.
* **Language Coverage:** We are actively working to expand our language coverage, particularly for major languages like Chinese and Arabic, as well as more low-resource languages. If you know of high-quality datasets that fits these criteria please let us know.
* **Repurposing of QA dataset**: About 50% of the current retrieval datasets are repurposed from QA datasets, which might lead to issues such as a strong lexical overlap between the question and the context, favoring models that rely on keyword matching over true semantic understanding.
* **Private datasets:** To test for generalization, we utilize private datasets that are only accessible to MTEB maintainers. To maintain fairness, all maintainers commit to not publishing models trained on these datasets and only testing on these private datasets through public channels, ensuring no company or individual receives unfair advantages.

Our goal is for RTEB to become a community-trusted standard for retrieval evaluation.

The RTEB leaderboard is available today on [Hugging Face](https://huggingface.co/spaces/mteb/leaderboard?benchmark_name=RTEB%28beta%29) as a part of the new Retrieval section on the MTEB leaderboard. We invite you to check it out, evaluate your models, and join us in building a better, more reliable benchmark for the entire AI community.

---

<span id="footnote-1">[1] Zero-shot score is the proportion of the evaluation set which the model provider have explicitly stated to have trained on. This typically only includes the training split.</span>
