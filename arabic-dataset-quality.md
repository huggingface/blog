---
title: "Assessing 331 Arabic NLP Datasets: The First Large-Scale Quality Audit of Masader"
thumbnail: /blog/assets/arabic-dataset-quality/thumbnail.png
authors:
- user: SalahAbdoNLP
  guest: true
---

# Assessing 331 Arabic NLP Datasets: The First Large-Scale Quality Audit of Masader

## TL;DR

We built an automated pipeline that assessed the quality of **331 Arabic NLP datasets** from the [Masader catalog](https://arbml.github.io/masader/), inspecting up to 500 samples per dataset. Each dataset received a score across 7 quality dimensions — accessibility, documentation, ethics, licensing, reproducibility, peer review, and data quality. The results: **35 excellent, 200 good, 79 acceptable, and 17 poor**. The full assessments are available as a [HuggingFace dataset](https://huggingface.co/datasets/SalahAbdoNLP/arabic-dataset-quality-assessments) and an [interactive Arabic-language browser](https://salah-sal.github.io/ar/datasets/).

## Why This Matters

Arabic is spoken by 400+ million people, yet Arabic NLP lags behind English in resources and tooling. The [Masader catalog](https://arbml.github.io/masader/) by [ARBML](https://github.com/ARBML) was a breakthrough — the first comprehensive index of 600+ Arabic datasets with 25+ metadata attributes. But Masader answers *what exists and where to find it*. It doesn't answer **how good each dataset actually is**.

Researchers face a common problem: you find a dataset on Masader, download it, and only then discover that 30% of samples are duplicated, the license is unclear, or the Arabic text is riddled with encoding artifacts. We wanted to solve this at scale — assess every freely accessible Arabic dataset on HuggingFace before anyone wastes time on a broken one.

## The Pipeline

We built a fully autonomous assessment system using Claude (Sonnet) running inside isolated Docker containers.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Host Machine (macOS)                           │
│  ┌──────────────────────────────────┐           │
│  │  assess_parallel.sh              │           │
│  │  find pending → xargs -P 20     │           │
│  └─────┬────────────────────────────┘           │
│        │ spawns 20 parallel workers             │
│  ┌─────▼─────┐ ┌──────────┐ ┌──────────┐       │
│  │ Docker #1 │ │ Docker #2│ │...Docker#20│      │
│  │ Ubuntu    │ │ Ubuntu   │ │ Ubuntu     │      │
│  │ Claude    │ │ Claude   │ │ Claude     │      │
│  │ Sonnet    │ │ Sonnet   │ │ Sonnet     │      │
│  └─────┬─────┘ └────┬─────┘ └─────┬─────┘      │
│        │ writes JSON │             │             │
│  ┌─────▼─────────────▼─────────────▼──────┐     │
│  │  output/  (331 assessment JSONs)       │     │
│  └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

Each Docker container:
1. Receives a dataset name and HuggingFace link
2. Loads up to 500 samples via `datasets` library in streaming mode
3. Runs embedded Python statistical analysis (duplication rates, encoding problems, text lengths, Latin punctuation in Arabic text)
4. Opens the paper link, documentation, and dataset card
5. Produces a structured Arabic JSON assessment

### Concurrency and Safety

- **Atomic locking**: Workers use `mkdir` as an atomic lock to prevent double-assessment
- **15-minute timeout**: Hard kill via `gtimeout` prevents hanging on large/broken datasets
- **Idempotent design**: If output JSON exists, skip — safe to restart after failures
- **No shared state**: Each container is fully isolated

## What We Measured

Each dataset is scored 0-100 on **7 quality dimensions**:

| Dimension | What It Measures |
|-----------|-----------------|
| **Accessibility** | Can the data be freely downloaded and loaded? |
| **Documentation** | Is there a paper, README, or dataset card? |
| **Ethics** | Are there ethical risks (bias, PII, harmful content)? |
| **Licensing** | Is the license clear and specified? |
| **Reproducibility** | Can results be reproduced from the provided data? |
| **Peer Review** | Was the dataset published at a reviewed venue? |
| **Data Quality** | Are the actual samples well-formed and correct? |

The overall score is a weighted combination, and datasets receive a grade:
- **ممتاز (Excellent)**: 80-100
- **جيد (Good)**: 60-79
- **مقبول (Acceptable)**: 40-59
- **ضعيف (Poor)**: 0-39

## Key Findings

### Grade Distribution

Out of 331 assessed datasets:

| Grade | Count | % |
|-------|-------|---|
| ممتاز (Excellent) | 35 | 10.6% |
| جيد (Good) | 200 | 60.4% |
| مقبول (Acceptable) | 79 | 23.9% |
| ضعيف (Poor) | 17 | 5.1% |

The mean score was **65.3/100** (median 68), which is encouraging — the majority of Arabic NLP datasets are in the "Good" range.

### Common Issues Found

Across 331 datasets, the most frequent problems were:

1. **Missing or unclear licenses** — The single biggest quality issue. Many datasets have "unknown" license or no license field at all.
2. **Sparse documentation** — Empty HuggingFace dataset cards, missing READMEs, papers behind paywalls.
3. **Duplication** — Several datasets had 10-30% duplicate samples that were not documented.
4. **Encoding artifacts** — Latin punctuation mixed into Arabic text (commas, semicolons, parentheses instead of their Arabic counterparts).
5. **Size discrepancies** — Some datasets claim sizes far larger than what's actually downloadable.

### Bright Spots

- Datasets from major initiatives (OPUS, WikiANN, mC4) consistently scored 75+
- Datasets with proper peer review (ACL/EMNLP venues) averaged 15 points higher than preprints
- The Arabic NLP community has significantly improved documentation over the past 3 years

## Prompt Engineering for Depth

An interesting technical challenge was getting consistent, deep assessments at scale. Our prompt went through three iterations:

| Version | Samples | Output Quality |
|---------|---------|---------------|
| v1 (English prompt) | ~10 | Superficial, missed real issues |
| v2 (Arabic prompt) | ~10 | Better language, still shallow |
| v3 (Arabic + embedded Python) | 500 | Deep analysis with statistical evidence |

The breakthrough was **embedding Python analysis scripts directly in the prompt**. Instead of asking the model to "check for duplicates," we gave it a script that calculates the exact duplication rate. This forced the model to report precise numbers rather than vague observations, and dramatically improved assessment quality.

## How to Use the Dataset

```python
from datasets import load_dataset

ds = load_dataset("SalahAbdoNLP/arabic-dataset-quality-assessments")

# Find the best datasets for sentiment analysis
sentiment = [d for d in ds["train"]
             if "sentiment analysis" in (d["tasks"] or [])
             and d["quality_score"] >= 70]
print(f"High-quality sentiment datasets: {len(sentiment)}")

# Get datasets where data doesn't match description
mismatched = [d for d in ds["train"]
              if d["data_matches_description"] == False]
print(f"Datasets with description mismatches: {len(mismatched)}")
```

You can also browse the assessments in Arabic at our [interactive datasets browser](https://salah-sal.github.io/ar/datasets/).

## Limitations

- **Automated assessments**: These are AI-generated evaluations, not human expert reviews. Scores are indicative, not definitive.
- **MSA-only scope**: We assessed free, MSA-focused datasets with HuggingFace links. Dialectal datasets, paid datasets, and those hosted only on LDC/ELRA were not included.
- **Point-in-time snapshot**: Assessments reflect dataset state as of February 2026. Datasets may be updated.
- **4 timeouts**: clartts, doclang, merged_arabic_corpus_of_isolated_words, and pm4bench timed out and are not included.

## What's Next

- **Community validation**: We invite Arabic NLP researchers to review and correct individual assessments
- **Quality scores in Masader**: We'd like to propose adding quality scores as a new column in the Masader catalog
- **Dialectal datasets**: Extend coverage to datasets in Egyptian, Gulf, Levantine, and Maghrebi Arabic
- **Longitudinal tracking**: Re-assess periodically to track quality improvements over time

## Acknowledgments

This work builds directly on the [Masader catalog](https://arbml.github.io/masader/) by [ARBML](https://github.com/ARBML). Masader is the foundation that made large-scale assessment possible — without their catalog of 600+ Arabic datasets with rich metadata annotations, this project could not exist. We thank Zaid Alyafeai, Maraim Masoud, Mustafa Ghaleb, Maged S. Al-Shaibani, and the 40+ community contributors who built and maintain Masader.

The assessment pipeline was built using [Claude](https://www.anthropic.com/) by Anthropic.

## Citation

```bibtex
@misc{alyafeai2021masader,
    title={Masader: Metadata Sourcing for Arabic Text and Speech Data Resources},
    author={Zaid Alyafeai and Maraim Masoud and Mustafa Ghaleb and Maged S. Al-shaibani},
    year={2021},
    eprint={2110.06744},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
