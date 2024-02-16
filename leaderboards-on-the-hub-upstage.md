---
title: "Introducing the Open Ko-LLM Leaderboard: Leading the Korean LLM Evaluation Ecosystem"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_upstage.png
authors:
- user: Chanjun
  guest: true
- user: hunkim
  guest: true
- user: clefourrier
---

# Introducing the Open Ko-LLM Leaderboard: Leading the Korean LLM Evaluation Ecosystem
In the fast-evolving landscape of Large Language Models (LLMs), building an “ecosystem” has never been more important. This trend is evident in several major developments like Hugging Face's democratizing NLP and Naver's building a Generative AI ecosystem. 

Inspired by these industry milestones, in September of 2023, at Upstage we initiated the Open Ko-LLM Leaderboard. Our goal was to quickly develop and introduce an evaluation ecosystem for Korean LLM data, aligning with the global movement towards open and collaborative AI development.

Our vision for the Open Ko-LLM Leaderboard is to cultivate a vibrant Korean LLM evaluation ecosystem, fostering transparency by enabling researchers to share their results and uncover hidden talents in the LLM field. In essence, we're striving to expand the playing field for Korean LLMs. 
To that end, we've developed an open platform where individuals can register their Korean LLM and engage in competitions with other models.
Additionally, we aimed to create a leaderboard that captures the unique characteristics and culture of the Korean language. To achieve this goal, we made sure that our translated benchmark datasets such as Ko-MMLU reflect the distinctive attributes of Korean.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="upstage/open-ko-llm-leaderboard"></gradio-app>

## Leaderboard design choices: creating a new private test set for fairness

The Open Ko-LLM Leaderboard is characterized by its unique approach to benchmarking, particularly:
- its adoption of Korean language datasets, as opposed to the prevalent use of English-based benchmarks. 
- the non-disclosure of test sets, contrasting with the open test sets of most leaderboards: we decided to construct entirely new datasets dedicated to Open Ko-LLM and maintain them as private, to prevent test set contamination and ensure a more equitable comparison framework.

While acknowledging the potential for broader impact and utility to the research community through open benchmarks, the decision to maintain a closed test set environment was made with the intention of fostering a more controlled and fair comparative analysis.

## Evaluation Tasks
The Open Ko-LLM Leaderboard adopts the following five types of evaluation methods:
- **Ko-ARC** (AI2 Reasoning Challenge): Ko-ARC is a multiple-choice test designed to assess scientific thinking and understanding. It measures the reasoning ability required to solve scientific problems, evaluating complex reasoning, problem-solving skills, and the understanding of scientific knowledge. The evaluation metric focuses on accuracy rates, reflecting how often the model selects the correct answer from a set of options, thereby gauging its ability to navigate and apply scientific principles effectively.
- **Ko-HellaSwag**: Ko-HellaSwag evaluates situational comprehension and prediction ability, either in a generative format or as a multiple-choice setup. It tests the capacity to predict the most likely next scenario given a situation, serving as an indicator of the model's understanding and reasoning abilities about situations. Metrics include accuracy assessing the quality of predictions, depending on whether it is approached as a multiple-choice.
- **Ko-MMLU** (Massive Multitask Language Understanding): Ko-MMLU assesses language comprehension across a wide range of topics and fields in a multiple-choice format. This broad test demonstrates how well a model functions across various domains, showcasing its versatility and depth in language understanding. Overall accuracy across tasks and domain-specific performance are key metrics, highlighting strengths and weaknesses in different areas of knowledge.
- **Ko-Truthful QA**: Ko-Truthful QA is actually a multiple-choice benchmark designed to evaluate the model's truthfulness and factual accuracy. Unlike a generative format where the model freely generates responses, in this multiple-choice setting, the model is tasked with selecting the most accurate and truthful answer from a set of options. This approach emphasizes the model's ability to discern truthfulness and accuracy within a constrained choice framework. The primary metric for Ko-Truthful QA focuses on the accuracy of the model's selections, assessing its consistency with known facts and its ability to identify the most truthful response among the provided choices.
- **Ko-CommonGEN V2**: A newly made benchmark for the Open Ko-LLM Leaderboard assesses whether LLMs can generate outputs that align with Korean common sense given certain conditions, testing the model’s capacity to produce contextually and culturally relevant outputs in the Korean language.

## A leaderboard in action: the barometer of Ko-LLM
The Open Ko-LLM Leaderboard has exceeded expectations, with over 1,000 models submitted. In comparison, the English Open LLM Leaderboard now hosts over 4,000 models. Within just five months of its launch, we've achieved a quarter of that number, a highly encouraging feat. We're incredibly grateful for this widespread participation, which shows the vibrant interest in Korean LLM development.

Of particular note is the diverse competition, encompassing individual researchers, corporations, and academic institutions such as KT, Lotte Information & Communication, Yanolja, MegaStudy Maum AI, 42Maru, the Electronics and Telecommunications Research Institute (ETRI), KAIST, and Korea University. 
One standout submission is KT's Mi:dm 7B model, which not only topped the rankings among models with 7B parameters or fewer but also became accessible for public use, marking a significant milestone.

We also observed that more generally two types of models demonstrate strong performance on the leaderboard:
- models which underwent cross-lingual transfer or fine-tuning in Korean (like Upstage’s SOLAR)
- models fine-tuned from LLaMa2, Yi, and Mistral, emphasizing the importance of leveraging solid foundational models for finetuning.

Managing such a big leaderboard did not come without its own challenged. The Open Ko-LLM Leaderboard aims to closely align with the Open LLM Leaderboard’s philosophy, especially in integrating with the Hugging Face model ecosystem. This strategy ensures that the leaderboard is accessible, making it easier for participants to take part, a crucial factor in its operation. Nonetheless, there are limitations due to the infrastructure, which relies on 16 A100 80GB GPUs. This setup faces challenges, particularly when running models larger than 30 billion parameters as they require an excessive amount of compute. This leads to prolonged pending states for many submissions. Addressing these infrastructure challenges is essential for future enhancements of the Open Ko-LLM Leaderboard.

## What’s next?
We recognize several limitations in current leaderboard models when considered in real-world contexts:
- Outdated Data: Datasets like SQUAD and KLEU become outdated over time. Data evolves and transforms continuously, but existing leaderboards remain fixed in a specific timeframe, making them less reflective of the current moment as hundreds of new data points are generated daily.
- Failure to Reflect the Real World: In B2B and B2C services, data is constantly accumulated from users or industries, and edge cases or outliers continuously arise. True competitive advantage lies in responding well to these challenges, yet current leaderboard systems lack the means to measure this capability. Real-world data is perpetually generated, changing, and evolving.
- Questionable Meaningfulness of Competition: Many models are specifically tuned to perform well on the test sets, potentially leading to another form of overfitting within the test set. Thus, the current leaderboard system operates in a leaderboard-centric manner rather than being real-world-centric.

Charting the course ahead, we therefore aim to expand the range of tasks. Although we currently offer five tasks, there is a need for more precise evaluations concerning ethical aspects and factual grounding. We plan to further broaden the scope of tasks in collaboration with various corporations and academic institutions. If you’d like to collaborate with us on this, we’ll be delighted to talk with you!

Next, we intend to extend the range of evaluation subjects. Beyond Korean, incorporating other languages and considering the growing importance of code, we are planning to include evaluations for code language models on a separate leaderboard.

Last but not least, rather than introducing entirely new plans, we believe in the importance of continuity and will continue to internally deliberate on sustainable approaches.

In conclusion, similar to Chatbot Arena, we believe that evaluations in a real online environment, as opposed to benchmark-based evaluations, are highly meaningful. Even within benchmark-based evaluations, there is a need for benchmarks to be updated monthly or for the benchmarks to more specifically assess domain-specific aspects.

## Many thanks to our partners
The journey of Open Ko-LLM Leaderboard began with a collaboration agreement to develop a Korean-style leaderboard, in partnership with Upstage and the National Information Society Agency (NIA), a key national institution in Korea. This partnership marked the starting signal, and within just a month, we were able to launch the leaderboard. 
To validate common-sense reasoning, we collaborated with Professor Heuiseok Lim's research team at Korea University to incorporate KoCommonGen V2 as an additional task for the leaderboard.
Building a robust infrastructure was crucial for success. To that end, we are grateful to Korea Telecom (KT) for their generous support of GPU resources and to Hugging Face for their recent CPU upgrade support. It's encouraging that Open Ko-LLM Leaderboard has established a direct line of communication with Hugging Face, a global leader in natural language processing, and we're in continuous discussion to push new initiatives forward.
Moreover, the Open Ko-LLM Leaderboard boasts a prestigious consortium of credible partners: the National Information Society Agency (NIA), Upstage, KT, and Korea University. The participation of these institutions, especially the inclusion of a national agency, lends significant authority and trustworthiness to the endeavor, underscoring its potential as a cornerstone in the academic and practical exploration of language models.
