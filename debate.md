---
title: "Letting Large Models Debate: The First Multilingual LLM Debate Competition"
authors:
- user: xuanricheng
  guest: true
  org: BAAI
- user: lilaczheng
  guest: true
  org: BAAI
- user: xiyang99
  guest: true
  org: BAAI
- user: Yonghua
  guest: true
  org: BAAI
- user: philokey
  guest: true
  org: BAAI
- user: xuejing2409
  guest: true
  org: BAAI
- user: graykingw
  guest: true
  org: BAAI
- user: daiteng01
  guest: true
  org: BAAI
- user: eyuansu71
  guest: true
  org: BAAI
- user: Lyfly2024
  guest: true
  org: BAAI
- user: xianbaoqian
  org: Huggingface
- user: clefourrier
  org: Huggingface
---

# Letting Large Models Debate: The First Multilingual LLM Debate Competition

Current static evaluations and user-driven arenas have exhibited their limitations and biases in the previous year. Here, we explore a novel way to evaluate LLMs: debate.
Debate is an excellent way to showcase reasoning strength and language abilities, used all across history, from the debates in the Athenian Ecclesia in the 5th century BCE to today's World Universities Debating Championship.
Do today's large language models exhibit debate skills similar to humans? Which model is currently the best at debating? What can we learn from models when they debate against one another?
To answer this question, BAAI has created a "Debate Arena", allowing large models to compete against each other. Currently, it supports debate competitions in English, Chinese, Arabic and Korean.
<!-- <script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script> -->
<gradio-app theme_mode="light" space="FlagEval/FlagEval-Arena-Leaderboard"></gradio-app>

## Background: The Need to Redefine LLM Evaluation Protocols

The advancement of multimodal and multilingual technologies has exposed the limitations of traditional static evaluation protocols in capturing LLMs’ performance in complex interactive scenarios. Inspired by OpenAI’s “AI Safety via Debate” framework—which emphasizes enhancing models’ reasoning and logic through multi-model interactions ([[1]](https://arxiv.org/abs/1805.00899))—BAAI’s FlagEval Debate platform introduces a dynamic evaluation methodology to address these limitations.
Recent research has demonstrated the potential of multi-agent debates in improving models’ reasoning capabilities and factual accuracy. For example, studies have shown that multi-agent interactions can significantly enhance models’ consistency and accuracy in logical reasoning and factual judgments ([[2]](https://arxiv.org/abs/2305.14325)), while others have indicated that multi-model debates encourage models to generate more truthful and coherent responses ([[3]](https://arxiv.org/abs/2402.06782)).
While existing platforms like LMSYS Chatbot Arena offer foundational settings for multi-model interactions, they present certain limitations in practical evaluation:
- **Lack of Discriminative Power**: Many model confrontations result in stalemates, lacking significant performance differentials. This necessitates a large number of user votes to ensure statistical stability and accuracy, thereby reducing evaluation efficiency and hindering rapid comparison of models’ relative strengths and weaknesses.
- **Isolated Generation Phenomenon**: In these platforms, models do not truly interact; instead, they independently generate responses based on user inputs without engaging with each other’s outputs. This unidirectional generation relies on random user dialogues, making it difficult to probe the boundaries of models’ reasoning and logic, and challenging to evaluate their performance in complex adversarial contexts.
- **Potential for Vote Bias**: Since models often exhibit distinctive styles, user votes usually favor specific model generation styles or formats rather than evaluating the content itself. Without expert annotations, these biases can skew the evaluation results, complicating an objective assessment of model capabilities and diminishing the reliability of user-driven judgments.

BAAI’s FlagEval Debate platform addresses these challenges by introducing genuine multi-model debates. Models engage in direct confrontations, showcasing their reasoning processes and depth. This allows evaluators to observe and compare differences in viewpoints, logical reasoning, and argumentation strategies among models. The platform’s multilingual support and real-time debugging capabilities enable users to study model strengths in realistic and interactive settings, ultimately providing more discriminative and in-depth evaluation results.

## Key Features and Innovations of FlagEval Debate

### Multilingual Support: Enabling Comprehensive Global Evaluation
FlagEval Debate supports Chinese, English, Korean, and Arabic, encompassing a diversity of writing systems and languages. This multilingual capability provides a platform for models to perform in cross-cultural contexts and tests their adaptability and communication effectiveness across diverse linguistic environments. This addresses the global demand for multilingual LLM evaluation.

### Developer Customization: Flexible Model Configuration and Optimization
To enhance fairness and flexibility, FlagEval Debate offers a developer customization feature, allowing participating model teams to fine-tune parameters, strategies, and dialogue styles based on their models’ characteristics and task requirements. This capability enables developers to optimize their models’ performance in debates, showcasing strengths while identifying areas for improvement. The real-time feedback loop fosters continuous optimization, allowing models to stand out in competitive evaluations.

### Dual Evaluation Metrics: Expert Reviews and User Feedback
FlagEval Debate employs a unique dual evaluation system combining expert reviews with user feedback, assessing models from both technical and experiential perspectives:
- **Expert Reviews**: We enlisted top-tier debate experts to rigorously evaluate models across dimensions such as logical reasoning, depth of argumentation, and linguistic expression. These experts provide objective, detailed assessments, ensuring that evaluation results possess high professional credibility.
- **User Feedback**: Concurrently, the platform facilitates user participation through audience voting, where users can rate models based on personal preferences and interactive experiences. This feedback reflects the models’ acceptance and effectiveness in practical user interactions, complementing the expert evaluations. Integrating user perspectives ensures that evaluations are aligned with real-world application scenarios.

## Experimental Results: Assessing the Impact of Multi-Model Debates

In Q3 2024, we conducted extensive experiments on the FlagEval Debate platform to evaluate the impact of multi-model debates on models’ logical reasoning and differentiated performance. The experiments yielded several critical insights:
1.	**Most current models can engage in debate**
Our experiments demonstrated that all participating models, including closed-source variants, could effectively engage in debate tasks. This indicates that models across different architectures and training paradigms possess the requisite capabilities for logical reasoning and interactive dialogue in multi-model settings. This broad applicability enhances the relevance of FlagEval Debate as a comprehensive evaluation platform. We have empirically observed that some small open-source models still face challenges maintaining coherence and staying on topic.

2.	**Models exhibit strong differences under adversarial conditions**
The interactive confrontations revealed significant variations in reasoning logic, argumentation techniques, and language use, especially under adversarial conditions. However, we are waiting for more expert evaluations before releasing these fine-grained findings. By facilitating direct interactions, FlagEval Debate allows for a more nuanced assessment of models’ logical reasoning, critical thinking, and rhetorical strategies, without relying on big pre-existing datasets. As illustrated in the figure below, even with just a few hundred matches, it’s evident that Model_7 has a notably higher win count, whereas Model_3 has accumulated more losses. In traditional arena-style and head-to-head evaluations, performance gaps between these models tend to be less pronounced, often requiring a large number of matches to yield conclusive results.
Similar to the LMSYS Chatbot Arena, FlagEval-debate also receives evaluation requests from new model providers. As these providers are still in the process of fine-tuning their models, we maintain model anonymity until the fine-tuning is complete and a sufficient number of user votes and expert reviews have been obtained.

<div align="center">
  <img src="https://huggingface.co/datasets/FlagEval/documentation-images/resolve/main/model%20win%20matrix.png" alt="image">
</div>

3.	**There is significant potential for model improvement**
In the early stages of development, we observed that models would make different types of errors during debates. For instance, one model would generate content for both the affirmative and Negative sides simultaneously, while others would display forced agreement even when that was inappropriate. 
For example:

    ```json
    {
    "model_A": 
    "Affirmative: I believe that a Dutch man fathering 1,000 children through sperm donation is a remarkable gesture. First, he provides hope and opportunities for couples who long for children but face challenges. Secondly, he demonstrates selflessness and kindness, showing a willingness to give part of his life for others. For the mothers carrying his children, it’s a valuable gift in their lives. His actions bring love and hope to the world.

    Negative: However, this action also raises moral and legal concerns. Having so many children from one individual could lead to family conflicts and legal disputes. Additionally, how will they handle the responsibilities of supporting and educating so many children? This could also place a burden on society. So, even if done with good intentions, the potential consequences of such actions need careful consideration.",
    }
    ```

    We addressed these issues by refining the system prompt requirements and adjusting the input format, though each model still exhibited unique error patterns. 
    At the same time, the experiments highlighted substantial opportunities for model enhancement. Performance variations indicated potential improvements in reasoning chains, linguistic expressiveness, and adversarial strategies. We encourage model creators to use our debate-tuning services to refine and optimize their models’ debate performance! 

4.	**Stronger evaluation: combining expert feedback with user preferences**
Our findings also suggest that relying exclusively on user preferences for evaluation can introduce biases and limit the accuracy of performance assessments. To mitigate this, FlagEval Debate’s dual evaluation system integrates expert reviews to provide a balanced perspective. By combining reliable expert assessments with subjective user feedback, we aim to ensure a more equitable and comprehensive evaluation, aligning results with both technical expertise and human preference.

## How to add your model into this competition?

The planning of the large-scale model debate is illustrated as follows. 
![image](https://huggingface.co/datasets/FlagEval/documentation-images/resolve/main/debate%20schedule.png)

Preliminary experiments indicate that the performance of the participating debate models will significantly improve after timely optimization. 
Model providers and creators are welcome to click the link [Debate Competition Registration Form](https://jwolpxeehx.feishu.cn/share/base/form/shrcnanu35NqOKaefVMUJKv6JYg) or send an email to [flageval@baai.ac.cn](mailto:flageval@baai.ac.cn).
By registering for the model debate evaluation, FlagEval will provide free model debate debugging services.
The following companies have already participated in our debate:

| Company     | Model               | Debugging Method |
|-------------|---------------------|-------------------|
| OpenAI      | o1-preview          | Self-debugged    |
| OpenAI      | o1-mini             | Self-debugged    |
| OpenAI      | GPT-4o-mini         | Self-debugged    |
| OpenAI      | GPT-4o              | Self-debugged    |
| Anthropic   | claude-3-5-sonnet   | Self-debugged    |
| Stepfun     | step-2-16k-f        | Provider-debugged|
| Baidu       | ERNIE-4.0-Turbo     | Provider-debugged|
| ByteDance   | Doubao-pro          | Provider-debugged|
| Alibaba     | qwen2.5-72b-instruct| Self-debugged    |
| Tencent     | Hunyuan-Turbo       | Provider-debugged|
| 01.AI       | Yi-Lightning        | Self-debugged    |
| Zhipu AI    | GLM-4-plus          | Provider-debugged|
| DeepSeek    | DeepSeek_V2.5       | Self-debugged    |

- **Self-debugged**: Denotes models we configured and optimized for debate.
- **Provider-debugged**: Denotes models that were debugged and optimized by the model providers themselves.

## Conclusion

FlagEval Debate represents a significant advancement in LLM evaluation methodologies. By incorporating multilingual support, developer customization, and a dual evaluation system, it offers a robust framework for assessing models in interactive, real-world scenarios. Moving forward, BAAI is committed to refining this platform to foster innovation, enhance evaluation methodologies, and drive standardization in AI practices. The goal is to provide developers and users with a forward-looking evaluation ecosystem, accelerating the evolution and deployment of advanced large language models.

## About BAAI & FlagEval

The Beijing Academy of Artificial Intelligence ([BAAI](https://huggingface.co/BAAI)) was established in November 2018 as a pioneering non-profit research institution, primarily focusing on original innovation and core technologies in artificial intelligence. Its aim is to drive revolutionary and disruptive breakthroughs in AI theory, methods, tools, systems, and applications.
Within BAAI, [FlagEval](https://huggingface.co/FlagEval) is a dedicated team specializing in the evaluation of large-scale AI models. FlagEval launched its [large model evaluation platform](https://flageval.baai.ac.cn/#/home?l=en) in 2023, and has since covered over 800 models globally. The platform evaluates over 40 dimensions, including reasoning, mathematical skills, and task-solving abilities. FlagEval recently launched new platforms for [model-to-model competition](https://arena.flageval.net/index.html#/arena-page?l=en), further strengthening its evaluation framework and advancing AI evaluation methodologies. It has been recognized as a leader in advancing AI evaluation standards and fostering collaboration between academia and industry.

## References

[1] Irving G, Christiano P, Amodei D. AI safety via debate. arXiv preprint arXiv:1805.00899, 2018.

[2] [ICML 2024] Du Y, Li S, Torralba A, et al. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

[3] [ICML 2024 Best] Khan A, Hughes J, Valentine D, et al. Debating with more persuasive llms leads to more truthful answers. arXiv preprint arXiv:2402.06782, 2024.