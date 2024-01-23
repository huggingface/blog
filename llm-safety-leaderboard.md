---
title: "An Introduction to Hugging Face LLM Safety Leaderboard"
thumbnail: /blog/assets/llm-leaderboard/leaderboard-thumbnail.png
authors:
- user: danielz01
- user: alphapav
- user: Cometkmt
- user: chejian
- user: BoLi-aisecure
---
Hugging Face’s [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) (originally created by Ed Beeching and Lewis Tunstall, and maintained by Nathan Habib and Clémentine Fourrier) is well known for tracking the functional performance of open source LLMs, comparing their performance in a variety of tasks, such as [TruthfulQA](https://github.com/sylinrl/TruthfulQA) or [HellaSwag](https://rowanzellers.com/hellaswag/).

This has been of tremendous value to the open-source community, as it provides a way for practitioners to keep track of the best open-source models.

In the meantime, given the wide adoption of LLMs, it is critical to understand their safety and risks in different scenarios before large deployments in the real world. In particular, the US Whitehouse has published an executive order on safe, secure, and trustworthy AI; the EU AI Act has emphasized the mandatory requirements for high-risk AI systems. Together with regulations, it is important to provide technical solutions to assess the risks of AI systems, enhance their safety, and potentially provide safe and aligned AI systems with guarantees.

Thus, in 2023, at Secure Learning Lab, we introduced DecodingTrust, the first comprehensive and unified evaluation platform for the trustworthiness of LLMs. This paper has won the Outstanding Paper Award at NeurIPS 2023. DecodingTrust provides eight trustworthiness perspectives, including toxicity, stereotype bias, adversarial robustness, OOD robustness, robustness on adversarial demonstrations, ethics, fairness, and privacy. In particular, DecodingTrust 1) contains comprehensive trustworthiness perspectives, 2) provides novel red-teaming algorithms for each perspective to test LLMs in-depth, 3) can be easily installed on different cloud environments, 4) provides a thorough leaderboard for existing open and close models based on their trustworthiness, 5) provides failure examples for transparency, 6) provides an end-to-end demo and model evaluation report for practical usage.

Today, we are happy to announce the release of the new LLM Safety Leaderboard, focusing on safety evaluation for LLMs and powered by the HF leaderboard template.

Concretely, DecodingTrust provides several novel red-teaming methodologies for each evaluation perspective for stress tests. 

For instance, for the privacy perspective, different levels of evaluation are provided, including 1) privacy leakage from pertaining data, 2) privacy leakage during conversations, and 3) privacy-related words and events understanding of LLMs. In particular, for 1) we have designed different approaches to perform privacy attacks. For instance, we provide different formats of content to LLMs to guide them to output sensitive information such as email addresses and credit card numbers.

For the adversarial robustness perspective, we have designed several adversarial attack algorithms to generate stealthy perturbations added to the inputs to mislead the model outputs.

For the OOD robustness perspective, we have designed different style transformations, knowledge transformations, etc, to evaluate the model performance when 1) the style of the inputs is transformed to other relatively rare styles such as Shakspares and Poem, or 2) the knowledge required to answer the question is not contained in the training data of LLMs.


# Summary
Based on the evaluations DecodingTrust, we have provided key findings for each trustworthiness perspective. Overall, we find that 1) GPT-4 is more vulnerable than GPT-3.5, 2) there is no single model that can dominate others on all trustworthiness perspectives, 3) there are tradeoffs between different trustworthiness perspectives, 4) LLMs have different capabilities in understanding different privacy-related words. For instance, if the model is prompted with "in confidence," it may not leak private information, while it may leak information if prompted with "confidently." 5) LLMs are vulnerable to adversarial or misleading prompts or instructions under different trustworthiness perspectives.


# Citation

If you find our evaluations useful, please consider citing our results.

```
@article{wang2023decodingtrust,
  title={DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models},
  author={Wang, Boxin and Chen, Weixin and Pei, Hengzhi and Xie, Chulin and Kang, Mintong and Zhang, Chenhui and Xu, Chejian and Xiong, Zidi and Dutta, Ritik and Schaeffer, Rylan and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

