---
title: "Arabic Leaderboards: Introducing Arabic Instruction Following, Updating AraGen, and More"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_3c3h_aragen.png
authors:
- user: alielfilali01
  guest: true
  org: inceptionai
- user: SarahAlBarri
  guest: true
  org: inceptionai
- user: Arwa88
  guest: true
  org: inceptionai
- user: samta-kamboj
  guest: true
  org: inceptionai
- user: neha1710
  guest: true
  org: inceptionai
- user: preslavnakov
  guest: true
  org: MBZUAI
---

# Arabic Leaderboards: Introducing Arabic Instruction Following, Updating AraGen, and More


We have been working to enhance AI model evaluations within the Arabic language context. Previously, we introduced **AraGen**, one of the first generative Arabic leaderboards, serving as a benchmark for evaluating Arabic LLMs on generative tasks.  

As part of our ongoing efforts, we are excited to share the following updates:  

- **Arabic-Leaderboards Space**, launched in collaboration with **Mohammed bin Zayed University of Artificial Intelligence (MBZUAI)** to consolidate Arabic AI evaluations in one place. This platform currently supports **AraGen 2** and **Arabic Instruction Following**, with plans to expand to leaderboards for Arabic AI models across various modalities.
- **AraGen 03-25 release** with improvements and updated benchmark.  
- **Instruction Following leaderboard**, powered by **Arabic IFEval Benchmark**, the first publicly available benchmark for evaluating instruction-following capabilities in Arabic.  


<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script>
<gradio-app theme_mode="dark" space="inceptionai/Arabic-Leaderboards"></gradio-app>

The following sections provide details about each of these updates.

## Arabic-Leaderboards Space
**Arabic-Leaderboards** is a comprehensive and unified space for all Arabic evaluations and tasks. It is meant to serve as a central hub covering a broad spectrum of evaluations, for models across modalities. Currently, it has AraGen 2 and Arabic Instruction Following as live leaderboards. We plan to expand this space with more leaderboards and tasks for Arabic AI models across various modalities. 

We invite interested contributors to reach out to us through the community tab or directly in order to discuss how to integrate their work/leaderboards as additional tabs into this space.

## Latest Updates in AraGen Leaderboard

In December 2024, we introduced the AraGen Benchmark as the foundation for the AraGen Leaderboard. A key feature of this leaderboard is its dynamic nature, with evaluation datasets remaining private (blind testing) for three months to ensure fair and unbiased assessments. Adhering to the same philosophy, we are publicly releasing the **AraGen-12-24 benchmark**, along with all model responses evaluated by **Claude-3.5-Sonnet** following the **3C3H guidelines**.  

By sharing this benchmark and model responses, we aim to encourage the community to review them, identify any unexpected behaviors we may have missed and help us refine our evaluation framework.

AraGen leaderboard is updated with a new version of benchmark (AraGen-03-25).


<iframe
  src="https://huggingface.co/datasets/inceptionai/AraGen/embed/viewer/default/test"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>


### AraGen-03-25 Release

In this latest AraGen release, we have expanded the dataset to include 340 pairs of questions and answers, up from 279 in the previous version. The distribution remains relatively similar:
- **Question Answering:** ~200 pairs  
- **Reasoning:** 70 pairs  
- **Safety Questions:** 40 pairs  
- **Orthographic and Grammatical Analysis:** 30 pairs  

This allocation reflects the primary focus on question answering as the main use cases of any Language-Model/Chatbot/AI-Assistant, while still addressing other evaluation areas, particularly given the complexity of generating challenging queries in Arabic grammar and orthography.

![Tasks Distribution (%)](https://huggingface.co/spaces/inceptionai/Arabic-Leaderboards/raw/main/assets/pictures/03-25/PercentageDistributionOfTasks.png)

Additionally, we refined the **judge system prompt** to enhance clarity, even for smaller/weaker judge models. 


### Dynamic Evaluation and Ranking Analysis

Maintaining consistency and reliability in our benchmark and evaluation pipeline is crucial as we introduce dynamic evaluation cycles. To ensure this, we analyzed ranking variations among the top 10 models across different dataset versions and system prompt configurations.

#### Analysis of Ranking Changes

We analyzed model performance under two evaluation scenarios:

- Compared previous system prompt (SP1) vs. current system prompt (SP2) using the latest AraGen version (**AraGen-03-25**).
- Assessed the impact of updating both the dataset and judge system prompt.

The overall rankings were stable, with the top-performing model (*o1-2024-12-17*) consistently maintaining its lead. Notably, we observed a swap in rankings between two Claude models, underscoring the sensitivity of our evaluation approach, especially given their initially close scores.

The only significant change in rankings was for the *gpt-4o-2024-08-06* model, whose performance markedly improved with the updated dataset and prompt. This sudden jump is currently under investigation as part of our ongoing benchmarks-design research.

No major variations occurred solely due to changes in the system prompt, indicating good reproducibility as long as the same judge model (*claude-3.5-sonnet*) is used. However, we anticipate potential variations with smaller or weaker models as judges, where employing the second system prompt (SP2) may enhance consistency.

As a summary, the robust, consistently top-ranking performance of *o1-2024-12-17* reinforces its reliability for Arabic applications. While recent updates to the evaluation pipeline introduced minor ranking shifts, the overall framework remained stable, with top and bottom performers showing consistency. Many observed ranking adjustments likely reflect typical evaluation error margins due to minor score differences. Notably, the second-ranked model’s score significantly dropped from 78.74% (AraGen-12-24) to 57.38% (AraGen-03-25), clearly indicating that the updated AraGen dataset poses a more challenging benchmark aligned with current advancements in reasoning models.

<details>
  <summary>More Detailed Scores</summary>

###### Couple 1: System Prompt Effect (AraGen-03-25 SP1 vs. AraGen-03-25 SP2)

**Table 1. AraGen-03-25 (SP1) Rankings**

| **Rank** | Model Name                 | 3C3H Score | Correctness | Completeness | Conciseness | Helpfulness | Honesty | Harmlessness |
|----------|----------------------------|------------|-------------|--------------|-------------|-------------|---------|--------------|
| 1        | o1-2024-12-17              | 69.49%     | 74.90%      | 73.04%       | 47.11%      | 72.40%      | 74.56%  | 74.90%       |
| 2        | gpt-4o-2024-08-06          | 56.10%     | 61.96%      | 58.92%       | 34.22%      | 58.80%      | 60.81%  | 61.89%       |
| 3        | claude-3-5-sonnet-20241022 | 54.29%     | 59.31%      | 57.65%       | 34.31%      | 57.13%      | 58.01%  | 59.31%       |
| 4        | claude-3-7-sonnet-20250219 | 53.21%     | 59.31%      | 56.76%       | 28.53%      | 56.86%      | 58.53%  | 59.24%       |
| 5        | o3-mini-2025-01-31         | 51.65%     | 56.67%      | 54.31%       | 31.74%      | 54.46%      | 56.10%  | 56.59%       |
| 6        | deepseek-chat              | 47.82%     | 54.31%      | 52.35%       | 20.56%      | 51.94%      | 53.46%  | 54.31%       |
| 7        | claude-3-5-haiku-20241022  | 43.62%     | 48.14%      | 44.61%       | 28.92%      | 45.37%      | 46.57%  | 48.14%       |
| 8        | o1-mini-2024-09-12         | 43.60%     | 47.55%      | 47.06%       | 26.54%      | 46.35%      | 46.57%  | 47.55%       |
| 9        | Qwen/Qwen2.5-72B-Instruct  | 42.18%     | 48.63%      | 47.55%       | 16.03%      | 44.93%      | 47.38%  | 48.55%       |
| 10       | gpt-4o-mini-2024-07-18     | 40.96%     | 45.10%      | 44.02%       | 24.24%      | 43.19%      | 44.14%  | 45.10%       |

**Table 2. AraGen-03-25 (SP2) Rankings**

| **Rank** | Model Name                 | 3C3H Score | Correctness | Completeness | Conciseness | Helpfulness | Honesty | Harmlessness |
|----------|----------------------------|------------|-------------|--------------|-------------|-------------|---------|--------------|
| 1        | o1-2024-12-17              | 70.25%     | 75.88%      | 70.98%       | 51.25%      | 72.55%      | 75.25%  | 75.59%       |
| 2        | gpt-4o-2024-08-06          | 57.38%     | 63.14%      | 56.67%       | 39.95%      | 59.66%      | 61.79%  | 63.06%       |
| 3        | claude-3-7-sonnet-20250219 | 56.54%     | 62.25%      | 58.53%       | 34.49%      | 60.39%      | 61.40%  | 62.18%       |
| 4        | claude-3-5-sonnet-20241022 | 55.60%     | 60.49%      | 56.67%       | 39.14%      | 58.60%      | 58.50%  | 60.20%       |
| 5        | o3-mini-2025-01-31         | 51.63%     | 56.08%      | 52.35%       | 36.72%      | 53.53%      | 55.10%  | 56.00%       |
| 6        | deepseek-chat              | 51.00%     | 57.55%      | 53.92%       | 25.61%      | 54.95%      | 56.42%  | 57.55%       |
| 7        | claude-3-5-haiku-20241022  | 44.79%     | 48.92%      | 44.51%       | 32.40%      | 46.67%      | 47.38%  | 48.85%       |
| 8        | o1-mini-2024-09-12         | 43.78%     | 47.55%      | 46.76%       | 28.04%      | 46.27%      | 46.67%  | 47.40%       |
| 9        | Qwen/Qwen2.5-72B-Instruct  | 43.09%     | 48.82%      | 47.55%       | 19.73%      | 46.59%      | 47.11%  | 48.75%       |
| 10       | gpt-4o-mini-2024-07-18     | 40.62%     | 45.10%      | 40.88%       | 27.60%      | 42.06%      | 43.58%  | 44.51%       |



###### Couple 2: Dataset and Prompt Update Effect (AraGen-12-24 SP1 (old) vs. AraGen-03-25 SP2 (new))

**Table 3. AraGen-12-24 (SP1) Rankings**

| **Rank** | Model Name                     | 3C3H Score | Correctness | Completeness | Conciseness | Helpfulness | Honesty | Harmlessness |
|----------|--------------------------------|------------|-------------|--------------|-------------|-------------|---------|--------------|
| 1        | o1-2024-12-17                  | 82.67%     | 92.71%      | 92.47%       | 34.65%      | 91.19%      | 92.26%  | 92.71%       |
| 2        | claude-3-5-sonnet-20241022     | 78.74%     | 88.31%      | 87.81%       | 33.27%      | 86.97%      | 87.78%  | 88.31%       |
| 3        | claude-3-7-sonnet-20250219     | 77.71%     | 87.89%      | 87.77%       | 29.20%      | 86.27%      | 87.26%  | 87.89%       |
| 4        | gpt-4o-2024-08-06              | 73.89%     | 83.75%      | 82.91%       | 28.94%      | 80.99%      | 83.00%  | 83.75%       |
| 5        | deepseek-chat                  | 71.28%     | 81.89%      | 81.89%       | 21.13%      | 79.53%      | 81.32%  | 81.89%       |
| 6        | o3-mini-2025-01-31             | 70.91%     | 80.29%      | 79.21%       | 27.33%      | 78.38%      | 79.99%  | 80.29%       |
| 7        | claude-3-5-haiku-20241022      | 66.40%     | 74.43%      | 73.36%       | 30.56%      | 72.34%      | 73.30%  | 74.43%       |
| 8        | o1-mini-2024-09-12             | 64.95%     | 74.22%      | 74.22%       | 21.46%      | 72.24%      | 73.32%  | 74.22%       |
| 9        | gpt-4o-mini-2024-07-18         | 63.40%     | 72.10%      | 71.38%       | 22.98%      | 70.41%      | 71.41%  | 72.10%       |
| 10       | Qwen/Qwen2.5-72B-Instruct      | 62.58%     | 71.92%      | 71.80%       | 19.06%      | 69.86%      | 70.94%  | 71.92%       |

**Table 4. AraGen-03-25 (SP2) Rankings**

| **Rank** | Model Name                 | 3C3H Score | Correctness | Completeness | Conciseness | Helpfulness | Honesty | Harmlessness |
|----------|----------------------------|------------|-------------|--------------|-------------|-------------|---------|--------------|
| 1        | o1-2024-12-17              | 70.25%     | 75.88%      | 70.98%       | 51.25%      | 72.55%      | 75.25%  | 75.59%       |
| 2        | gpt-4o-2024-08-06          | 57.38%     | 63.14%      | 56.67%       | 39.95%      | 59.66%      | 61.79%  | 63.06%       |
| 3        | claude-3-7-sonnet-20250219 | 56.54%     | 62.25%      | 58.53%       | 34.49%      | 60.39%      | 61.40%  | 62.18%       |
| 4        | claude-3-5-sonnet-20241022 | 55.60%     | 60.49%      | 56.67%       | 39.14%      | 58.60%      | 58.50%  | 60.20%       |
| 5        | o3-mini-2025-01-31         | 51.63%     | 56.08%      | 52.35%       | 36.72%      | 53.53%      | 55.10%  | 56.00%       |
| 6        | deepseek-chat              | 51.00%     | 57.55%      | 53.92%       | 25.61%      | 54.95%      | 56.42%  | 57.55%       |
| 7        | claude-3-5-haiku-20241022  | 44.79%     | 48.92%      | 44.51%       | 32.40%      | 46.67%      | 47.38%  | 48.85%       |
| 8        | o1-mini-2024-09-12         | 43.78%     | 47.55%      | 46.76%       | 28.04%      | 46.27%      | 46.67%  | 47.40%       |
| 9        | Qwen/Qwen2.5-72B-Instruct  | 43.09%     | 48.82%      | 47.55%       | 19.73%      | 46.59%      | 47.11%  | 48.75%       |
| 10       | gpt-4o-mini-2024-07-18     | 40.62%     | 45.10%      | 40.88%       | 27.60%      | 42.06%      | 43.58%  | 44.51%       |


</details>


#### Analysis of 3C3H

As part of our December release, we introduced 3C3H as a new evaluation measure of the chat capability of models, aimed at assessing both the factuality and usability of LLMs’ answers. Over the past three months, we have observed some interesting findings, which we share in this section.

One emergent trend is that the various dimensions are almost perfectly correlated. In most cases, correct answers are scored as both highly helpful and harmless, while most models fail to maintain this correlation for the conciseness dimension. This generally reflects the way we train these models today, where increased helpfulness is often rewarded with higher verbosity. This trend has recently caught the attention of the research community, as exemplified by the release of OpenAI’s GPT-4.5 model. According to their [use cases section](https://openai.com/index/introducing-gpt-4-5/), answers from GPT-4.5 are more concise than those from GPT-4, while still being equally helpful.

![HeatMap for o1-2024-12-17](https://huggingface.co/spaces/inceptionai/Arabic-Leaderboards/raw/main/assets/pictures/03-25/o1-heatmap.png)


A model that stood out in this analysis is “silma-ai/SILMA-9B-Instruct-v1.0”, which exhibited a higher conciseness score compared to other open-weight models—even those with larger sizes. However, this gain in conciseness came at the cost of helpfulness and other dimensions when compared to its base model, “google/gemma-2-9b-it”. We believe that this analysis, along with optimizing for 3C3H, will enable the community to develop better models through curated datasets while maintaining the correlation across all dimensions.

![SILMA-9B-Instruct-v1.0 VS Gemma-2-9b-it HeatMaps](https://huggingface.co/spaces/inceptionai/Arabic-Leaderboards/raw/main/assets/pictures/03-25/silma-vs-gemma-heatmap.png)


This is an ongoing effort to better understand how these dimensions are interconnected and how various scenarios and training recipes affect this relationship. Below, we provide a space where you can generate heatmaps for any combination of models of your choice. We hope the community finds it helpful in spotting additional trends that we may not have noticed. Ultimately, we aim for this tool to foster more discussion about evaluation and 3C3H, serving as a resource for others’ work.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script>
<gradio-app theme_mode="dark" space="inceptionai/3C3H-HeatMap"></gradio-app>

We believe that one limitation of this analysis is the zeroing rule, whereby we do not evaluate the other dimensions if the answer is not correct. In the future, we plan to investigate further whether an answer can be helpful despite being incorrect, and how dimensions such as conciseness and harmlessness factor into this evaluation if the answer is not correct.

## Instruction Following Leaderboard

### What is Instruction Following as a Benchmark? 

One of the core capabilities of large language models (LLMs) is their ability to understand and follow human instructions. This skill is crucial for building reliable chatbots, virtual assistants, and AI systems that do what users ask. Without strong instruction-following, a model might generate correct information but in the wrong format, ignore user-specified constraints, or produce unwanted content. Instruction-Following benchmark is standardized, objective way to measure a model's instruction adherence and compare models fairly to drive improvements. 

### Dataset: Arabic IFEval 

Our work took inspiration from [IFEval](https://arxiv.org/abs/2311.07911) dataset. IFEval, originally introduced by Google, provides a structured benchmark designed explicitly to evaluate LLMs on their ability to follow verifiable instructions. It consists of prompts containing specific, objectively measurable commands such as “use exactly three bullet points,” “include the word ‘innovation’ twice,” or “limit your answer to 100 words.”  English IFEval dataset contains around 500 prompts covering 25 different types of such verifiable instructions. Evaluation within IFEval is conducted through Python functions that automatically verify whether instructions are followed or not avoiding the need for human evaluators or another AI judge, this makes the evaluations **reproducible and unbiased**. While IFEval dataset has become the standard for assessing English-LLMs, a similarly detailed and structured resource is absent for Arabic, leaving a gap in evaluating Arabic LLMs' instrcution-following capabilities.

Construction of our **Arabic IFEval** dataset began by carefully adapting approximately 300 prompts from the original English IFEval. This wasn't a straightforward, word-for-word translation; instead, we thoughtfully adjusted prompts to clearly reflect Arabic linguistic nuances and cultural contexts. Instructions that made little sense in Arabic, such as those involving English-specific vowel constraints, were either adapted to equivalent Arabic linguistic challenges or omitted entirely. Cultural references specific to English-speaking contexts were replaced with culturally relevant or Arabic-language equivalents to maintain contextual clarity. Additionally, we created unique Arabic-specific samples from scratch, specifically designed to emphasize distinctive Arabic phonetics, orthographic characteristics, and morphology, such as the careful use of diacritical marks (tashkīl), phonetic constraints like avoiding certain letters (e.g., writing without using the letter Alef (ا)), and leveraging root-based morphology to challenge models' word-selection abilities. All prompts underwent rigorous expert validation by  Arabic linguists and domain experts who ensured grammatical accuracy, cultural appropriateness, and unambiguous clarity of each instruction.

**Arabic IFEval** dataset is publicly available for the research community to utilize, test, and contribute to. It is available on Huggingface under [inceptionai/Arabic_IFEval](https://huggingface.co/datasets/inceptionai/Arabic_IFEval)

<details>
  <summary><strong>Sample I: Arabic IFEval</strong></summary>


**Prompt (Ar):**  
فسر كيف يمكن للتقنيات الحديثة مثل الذكاء الاصطناعي أن تسهم في الحفاظ على الأدب العربي، مع تضمين 12 كلمة تنتهي بأحد الحروف الرافسة (د، ذ، أ، ر، ز، و)، وأن تكون الإجابة مكتوبة بأسلوب موجز لا يتجاوز 120 كلمة. يجب أن لا تحتوي إجابتك على أي فواصل.

**Prompt Translation (En):**
Explain how modern technologies, such as artificial intelligence, can contribute to preserving Arabic literature. Your answer should include at least 12 words ending with one of these specific Arabic letters (د، ذ، أ، ر، ز، و), be concise, and should not exceed 120 words. Your response must not contain any commas.

**Instructions to follow:**  
- **Letter Frequency Constraint:** Include at least 12 words ending with one of the letters (د، ذ، أ، ر، ز، و).  
- **Punctuation Constraint:** Do not use commas.  
- **Length Constraint:** Write concisely, not exceeding 120 words.

**Example JSON Format:**
```json
{
  "key": 4767,
  "prompt": "فسر كيف يمكن للتقنيات الحديثة مثل الذكاء الاصطناعي أن تسهم في الحفاظ على الأدب العربي، مع تضمين 12 كلمة تنتهي بأحد الحروف الرافسة (د، ذ، أ، ر، ز، و)، وأن تكون الإجابة مكتوبة بأسلوب موجز لا يتجاوز 120 كلمة. يجب أن لا تحتوي إجابتك على أي فواصل.",
  "instruction_id_list": [
    "keywords:letter_list_freq",
    "punctuation:no_comma",
    "length_constraints:number_words"
  ],
  "kwargs": [
    {
      "letters": ["د", "ذ", "أ", "ر", "ز", "و"],
      "frequency": 12,
      "relation": "at least",
      "position": "end"
    },
    {},
    {
      "relation": "less than",
      "num_words": 500
    }
  ],
  "lang": ["ar"]
}
```
</details>

<details>
  <summary><strong>Sample II: Arabic IFEval</strong></summary>


**Prompt (Ar):**
اكتب قصة قصيرة عن الرقم 600، على أن يكتب الرقم في القصة بالكلمات وبكل الصيغ المفقطة الممكنة له على الأقل مرة (ستة مائة - ست مئة - ستمئة - ستمائة).

**Prompt Translation (En):**  
Write a short story about the number 600. Within the story, the number should be spelled out in Arabic in all possible written forms at least once each ("ستة مائة", "ست مئة", "ستمئة", "ستمائة").

**Instructions to follow:**  
Your response must explicitly include the following Arabic spellings at least once each:
- ستة
- مائة
- ست
- مئة
- ستمئة
- ستمائة

**Example JSON Format:**
```json
{
  "key": 4768,
  "prompt": "اكتب قصة قصيرة عن الرقم 600، على أن يكتب الرقم في القصة بالكلمات وبكل الصيغ المفقطة الممكنة له على الأقل مرة (ستة مائة - ست مئة - ستمئة - ستمائة).",
  "instruction_id_list": [
    "keywords:frequency",
    "keywords:frequency",
    "keywords:frequency",
    "keywords:frequency",
    "keywords:frequency",
    "keywords:frequency"
  ],
  "kwargs": [
    {"relation": "at least", "keyword": "ستة", "frequency": 1},
    {"relation": "at least", "keyword": "مائة", "frequency": 1},
    {"relation": "at least", "keyword": "ست", "frequency": 1},
    {"relation": "at least", "keyword": "مئة", "frequency": 1},
    {"relation": "at least", "keyword": "ستمئة", "frequency": 1},
    {"relation": "at least", "keyword": "ستمائة", "frequency": 1}
  ],
  "lang": ["ar"]
}
```
</details>

### Evaluation Methodology & Metrics 

To evaluate the models, we adopted a comprehensive methodology combining both explicit and implicit evaluation techniques. Explicit evaluation involved using automated scripts to  assess whether instructions were strictly followed, focusing on elements such as correct formatting and specific word usage. Implicit evaluation addressed more nuanced linguistic expectations, such as maintaining the intended response language and avoiding repetitive patterns.

Additionally, we utilized scoring metrics introduced by Google in the IFEval framework, applying these metrics at both prompt-level and instruction-level granularity. These metrics were measured using strict criteria accuracy, which requires adherence to the provided instructions. The prompt-level score is notably harder, reflecting the user's viewpoint by asking, "Did I get everything I requested?" If a prompt included multiple requirements, failing to meet any single requirement would mean the user's request was not fully satisfied. In contrast, the instruction-level score is more lenient, allowing us to evaluate partial compliance.

In our analysis, we will emphasize the prompt-level strict accuracy as it provides the most rigorous assessment of a model's instruction-following capabilities.

### Results & Analysis 

We evaluated a broad range of LLMs on both the English IFEval benchmark and our newly introduced Arabic IFEval. This encompassed closed-source models (such as OpenAI's GPT series and Anthropic's Claude models) and open-source alternatives (including the Jais series, Meta’s LLaMA-2 variants, and various open bilingual models). Below, we present a summary of results for a representative subset of these models, comparing their prompt-level accuracy on both English and Arabic IFEval. Accuracy is reported using both strict and loose criteria, with values expressed as the percentage of prompts successfully completed.


<details>
  <summary>Instruction Following Leaderboard Sample</summary>
    
**Table 5. Sample Scores from Instruction Following Benchmark** 
| Rank | Model Name                         | Arabic Prompt-lvl (%) | English Prompt-lvl (%) |
|------|------------------------------------|-----------------------|------------------------|
| 1    | claude-3.5-sonnet                  | 72.5                  | 84.7                   |
| 2    | gpt-4o-2024-08-06                  | 70.8                  | 79.4                   |
| 3    | gpt-4o-mini-2024-07-18             | 68.1                  | 76.9                   |
| 4    | claude-3.5-haiku                   | 67.1                  | 78.2                   |
| 5    | Qwen/Qwen2.5-72B-Instruct          | 67.3                  | 83.5                   |
| 6    | Qwen/Qwen2.5-32B-Instruct          | 60.4                  | 77.6                   |
| 7    | google/gemma-2-27b-it              | 59.4                  | 76.1                   |
| 8    | CohereForAI/aya-expanse-32b        | 56.7                  | 65.1                   |
| 9    | CohereForAI/c4ai-command-r7b-12-2024 | 56.4                  | 74.9                   |
| 10   | meta-llama/Llama-3.3-70B-Instruct  | 58.2                  | 88.2                   |

</details>


## Upcoming Work

As part of our work we will be adding and updating more leaderboards to the Arabic-Leaderboards Space as our internal work progress.
In the upcoming releases, we expect to put online a leaderbaord for visual question-answering across multiple tasks powered by camel-bench and kitab from our collaborators at MBZUAI.
