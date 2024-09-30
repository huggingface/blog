---
title: "Introducing the Open Arabic LLM Leaderboard"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_arabic.png
authors:
- user: alielfilali01
  guest: true
  org: 2A2I
- user: Hamza-Alobeidli
  guest: true
  org: tiiuae
- user: rcojocaru
  guest: true
  org: tiiuae
- user: Basma-b
  guest: true
  org: tiiuae
- user: clefourrier
---


# Introducing the Open Arabic LLM Leaderboard

The Open Arabic LLM Leaderboard (OALL) is designed to address the growing need for specialized benchmarks in the Arabic language processing domain. As the field of Natural Language Processing (NLP) progresses, the focus often remains heavily skewed towards English, leaving a significant gap in resources for other languages. The OALL aims to balance this by providing a platform specifically for evaluating and comparing the performance of Arabic Large Language Models (LLMs), thus promoting research and development in Arabic NLP.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.4.0/gradio.js"> </script>
<gradio-app theme_mode="light" space="OALL/Open-Arabic-LLM-Leaderboard"></gradio-app>

This initiative is particularly significant given that it directly serves over 380 million Arabic speakers worldwide. By enhancing the ability to accurately evaluate and improve Arabic LLMs, we hope the OALL will play a crucial role in developing models and applications that are finely tuned to the nuances of the Arabic language, culture and heritage.

## Benchmarks, Metrics & Technical setup

### Benchmark Datasets
The Open Arabic LLM Leaderboard (OALL) utilizes an extensive and diverse collection of robust datasets to ensure comprehensive model evaluation. 

- [AlGhafa benchmark](https://aclanthology.org/2023.arabicnlp-1.21): created by the TII LLM team with the goal of evaluating models on a range of abilities including reading comprehension, sentiment analysis, and question answering. It was initially introduced with 11 native Arabic datasets and was later extended to include an additional 11 datasets that are translations of other widely adopted benchmarks within the English NLP community. 
- ACVA and AceGPT benchmarks: feature 58 datasets from the paper ["AceGPT, Localizing Large Language Models in Arabic"](https://arxiv.org/abs/2309.12053), and translated versions of the MMLU and EXAMS benchmarks to broaden the evaluation spectrum and cover a comprehensive range of linguistic tasks. These benchmarks are meticulously curated and feature various subsets that precisely capture the complexities and subtleties of the Arabic language.

### Evaluation Metrics
Given the nature of the tasks, which include multiple-choice and yes/no questions, the leaderboard primarily uses normalized log likelihood accuracy for all tasks. This metric was chosen for its ability to provide a clear and fair measurement of model performance across different types of questions.


### Technical setup
The technical setup for the Open Arabic LLM Leaderboard (OALL) uses: 
- front- and back-ends inspired by the [`demo-leaderboard`](https://huggingface.co/demo-leaderboard-backend), with the back-end running locally on the TII cluster
- the `lighteval` library to run the evaluations. Significant contributions have been made to integrate the Arabic benchmarks discussed above into `lighteval`, to support out-of-the-box evaluations of Arabic models for the community (see [PR #44](https://github.com/huggingface/lighteval/pull/44) and [PR #95](https://github.com/huggingface/lighteval/pull/95) on GitHub for more details).


## Future Directions

We have many ideas about expanding the scope of the Open Arabic LLM Leaderboard. Plans are in place to introduce additional leaderboards under various categories, such as one for evaluating Arabic LLMs in Retrieval Augmented Generation (RAG) scenarios and another as a chatbot arena that calculates the ELO scores of different Arabic chatbots based on user preferences. 

Furthermore, we aim to extend our benchmarks to cover more comprehensive tasks by developing the OpenDolphin benchmark, which will include about 50 datasets and will be an open replication of the work done by Nagoudi et al. in the paper titled [“Dolphin: A Challenging and Diverse Benchmark for Arabic NLG”](https://arxiv.org/abs/2305.14989). For those interested in adding their benchmarks or collaborating on the OpenDolphin project, please contact us through the discussion tab or at this [email address](mailto:elfilali.ali01@gmail.com).

We’d love to welcome your contribution on these points! We encourage the community to contribute by submitting models, suggesting new benchmarks, or participating in discussions. We also encourage the community to make use of the top models of the current leaderboard to create new models through finetuning or any other techniques that might help your model to climb the ranks to the first place! You can be the next Arabic Open Models Hero! 

We hope the OALL will encourage technological advancements and highlight the unique linguistic and cultural characteristics inherent to the Arabic language, and that our technical setup and learnings from deploying a large-scale, language-specific leaderboard can be helpful for similar initiatives in other underrepresented languages. This focus will help bridge the gap in resources and research, traditionally dominated by English-centric models, enriching the global NLP landscape with more diverse and inclusive tools, which is crucial as AI technologies become increasingly integrated into everyday life around the world. 


## Submit Your Model !

### Model Submission Process

To ensure a smooth evaluation process, participants must adhere to specific guidelines when submitting models to the Open Arabic LLM Leaderboard:

1. **Ensure Model Precision Alignment:** It is critical that the precision of the submitted models aligns with that of the original models. Discrepancies in precision may result in the model being evaluated but not properly displayed on the leaderboard.

2. **Pre-Submission Checks:**
   - **Load Model and Tokenizer:** Confirm that your model and tokenizer can be successfully loaded using AutoClasses. Use the following commands:
     ```python
     from transformers import AutoConfig, AutoModel, AutoTokenizer
     config = AutoConfig.from_pretrained("your model name", revision=revision)
     model = AutoModel.from_pretrained("your model name", revision=revision)
     tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
     ```
     If you encounter errors, address them by following the error messages to ensure your model has been correctly uploaded.

   - **Model Visibility:** Ensure that your model is set to public visibility. Additionally, note that if your model requires `use_remote_code=True`, this feature is not currently supported but is under development.

3. **Convert Model Weights to Safetensors:**
   - Convert your model weights to safetensors, a safer and faster format for loading and using weights. This conversion also enables the inclusion of the model's parameter count in the `Extended Viewer`.

4. **License and Model Card:**
   - **Open License:** Verify that your model is openly licensed. This leaderboard promotes the accessibility of open LLMs to ensure widespread usability.
   - **Complete Model Card:** Populate your model card with detailed information. This data will be automatically extracted and displayed alongside your model on the leaderboard.

### In Case of Model Failure

If your model appears in the 'FAILED' category, this indicates that execution was halted. Review the steps outlined above to troubleshoot and resolve any issues. Additionally, test the following [script](https://gist.github.com/alielfilali01/d486cfc962dca3ed4091b7c562a4377f) on your model locally to confirm its functionality before resubmitting.


##  Acknowledgements

We extend our gratitude to all contributors, partners, and sponsors, particularly the Technology Innovation Institute and Hugging Face for their substantial support in this project. TII has provided generously the essential computational resources, in line with their commitment to supporting community-driven projects and advancing open science within the Arabic NLP field, whereas Hugging Face has assisted with the integration and customization of their new evaluation framework and leaderboard template.

We would also like to express our thanks to Upstage for their work on the Open Ko-LLM Leaderboard, which served as a valuable reference and source of inspiration for our own efforts. Their pioneering contributions have been instrumental in guiding our approach to developing a comprehensive and inclusive Arabic LLM leaderboard.


## Citations and References
```
@misc{OALL,
  author = {Elfilali, Ali and Alobeidli, Hamza and Fourrier, Clémentine and Boussaha, Basma El Amel and Cojocaru, Ruxandra and Habib, Nathan and Hacid, Hakim},
  title = {Open Arabic LLM Leaderboard},
  year = {2024},
  publisher = {OALL},
  howpublished = "\url{https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard}"
}

@inproceedings{almazrouei-etal-2023-alghafa,
    title = "{A}l{G}hafa Evaluation Benchmark for {A}rabic Language Models",
    author = "Almazrouei, Ebtesam  and
      Cojocaru, Ruxandra  and
      Baldo, Michele  and
      Malartic, Quentin  and
      Alobeidli, Hamza  and
      Mazzotta, Daniele  and
      Penedo, Guilherme  and
      Campesan, Giulia  and
      Farooq, Mugariya  and
      Alhammadi, Maitha  and
      Launay, Julien  and
      Noune, Badreddine",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.21",
    doi = "10.18653/v1/2023.arabicnlp-1.21",
    pages = "244--275",
    abstract = "Recent advances in the space of Arabic large language models have opened up a wealth of potential practical applications. From optimal training strategies, large scale data acquisition and continuously increasing NLP resources, the Arabic LLM landscape has improved in a very short span of time, despite being plagued by training data scarcity and limited evaluation resources compared to English. In line with contributing towards this ever-growing field, we introduce AlGhafa, a new multiple-choice evaluation benchmark for Arabic LLMs. For showcasing purposes, we train a new suite of models, including a 14 billion parameter model, the largest monolingual Arabic decoder-only model to date. We use a collection of publicly available datasets, as well as a newly introduced HandMade dataset consisting of 8 billion tokens. Finally, we explore the quantitative and qualitative toxicity of several Arabic models, comparing our models to existing public Arabic LLMs.",
}
@misc{huang2023acegpt,
      title={AceGPT, Localizing Large Language Models in Arabic}, 
      author={Huang Huang and Fei Yu and Jianqing Zhu and Xuening Sun and Hao Cheng and Dingjie Song and Zhihong Chen and Abdulmohsen Alharthi and Bang An and Ziche Liu and Zhiyi Zhang and Junying Chen and Jianquan Li and Benyou Wang and Lian Zhang and Ruoyu Sun and Xiang Wan and Haizhou Li and Jinchao Xu},
      year={2023},
      eprint={2309.12053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.3.0},
  url = {https://github.com/huggingface/lighteval}
}
```

