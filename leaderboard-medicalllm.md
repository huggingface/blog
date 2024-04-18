---
title: "The Open Medical-LLM Leaderboard: Benchmarking Large Language Models in Healthcare"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_medicalllm.png
authors:
- user: aaditya
  guest: true
- user: pminervini
  guest: true
- user: clefourrier
---

# The Open Medical-LLM Leaderboard: Benchmarking Large Language Models in Healthcare

![Image source : https://arxiv.org/pdf/2311.05112.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/3a38d183-b6f4-4c23-a560-1bdd708e7af1/Screenshot_2024-04-12_at_9.45.49_AM.png)

Over the years, Large Language Models (LLMs) have emerged as a groundbreaking technology with immense potential to revolutionize various aspects of healthcare. These models, such as [GPT-3](https://arxiv.org/abs/2005.14165), [GPT-4](https://arxiv.org/abs/2303.08774) and [Med-PaLM 2](https://arxiv.org/abs/2305.09617) have demonstrated remarkable capabilities in understanding and generating human-like text, making them valuable tools for tackling complex medical tasks and improving patient care. They have notably shown promise in various medical applications, such as medical question-answering (QA), dialogue systems, and text generation. Moreover, with the exponential growth of electronic health records (EHRs), medical literature, and patient-generated data, LLMs could help healthcare professionals extract valuable insights and make informed decisions.

However, despite the immense potential of Large Language Models (LLMs) in healthcare, there are significant and specific challenges that need to be addressed. 

While model errors may be of minimal consequence in casual conversation or other contexts, wrong explanation and answers in the medical domain can have severe consequences for patient care and outcomes. The accuracy and reliability of information provided by language models can be a matter of life or death, as it could potentially affect healthcare decisions, diagnosis, and treatment plans. 

For example, when given a medical query, GPT-3 provided a wrong answer (as shown below) which could have dreadful consequences if taken at face value. To give some context, although the explanation correctly stated that tetracycline is contraindicated in pregnancy due to potential harmful effects on the fetus, GPT-3.5 still incorrectly selected it as the answer, but if tetracycline were administered based on this incorrect information, it could lead to bone growth problems in the baby.

![Image source : [https://arxiv.org/pdf/2311.05112.pdf](https://arxiv.org/abs/2307.15343)](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/4ea13c24-88a0-40b8-880f-e7bc9888b997/Screenshot_2024-04-12_at_1.18.05_PM.png)


To fully utilize the power of LLMs in healthcare, it is crucial to develop and benchmark models **in a setup** specifically designed for the medical domain, taking into account the unique characteristics and requirements of healthcare data and applications. The development of methods to evaluate the Medical-LLM is not just of academic interest but of practical importance, given the real-life risks they pose in the healthcare sector.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/4.20.1/gradio.js"> </script>
<gradio-app theme_mode="light" space="openlifescienceai/open_medical_llm_leaderboard"></gradio-app>


The Open Medical-LLM Leaderboard aims to address these challenges and limitations by providing a standardized platform for evaluating and comparing the performance of various large language models on a diverse range of medical tasks and datasets. By offering a comprehensive assessment of each model's medical knowledge and question-answering capabilities, the leaderboard aims to foster the development of more effective and reliable medical LLMs. 

This platform enables researchers and practitioners to identify the strengths and weaknesses of different approaches, drive further advancements in the field, and ultimately contribute to better patient care and outcomes

## Datasets, Tasks, and Evaluation Setup 

The Medical-LLM Leaderboard includes a variety of tasks, and uses accuracy as its primary evaluation metric (accuracy measures the percentage of correct answers provided by a language model across the various medical QA datasets).

### MedQA (USMLE)

The [MedQA](https://arxiv.org/abs/2009.13081) dataset consists of multiple-choice questions from the United States Medical Licensing Examination (USMLE). It covers general medical knowledge and includes 11,450 questions in the development set and 1,273 questions in the test set. Each question has 4 or 5 answer choices, and the dataset is designed to assess the medical knowledge and reasoning skills required for medical licensure in the United States.

![Screenshot 2024-04-12 at 11.51.47 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/4eca4c09-e52e-4852-8f14-f26fa22c7f1d/Screenshot_2024-04-12_at_11.51.47_PM.png)

### MedMCQA

[MedMCQA](https://proceedings.mlr.press/v174/pal22a.html) is a large-scale multiple-choice QA dataset derived from Indian medical entrance examinations (AIIMS/NEET). It covers 2.4k healthcare topics and 21 medical subjects, with over 187,000 questions in the development set and 6,100 questions in the test set. Each question has 4 answer choices and is accompanied by an explanation. MedMCQA evaluates a model's general medical knowledge and reasoning capabilities.

![Screenshot 2024-04-12 at 11.53.18 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/7b2860c6-4ae1-4e77-877f-f77a1ecb06d9/Screenshot_2024-04-12_at_11.53.18_PM.png)

### PubMedQA

[PubMedQA](https://aclanthology.org/D19-1259/) is a closed-domain QA dataset, In which each question can be answered by looking at an associated context (PubMed abstract). It is consists of 1,000 expert-labeled question-answer pairs. Each question is accompanied by a PubMed abstract as context, and the task is to provide a yes/no/maybe answer based on the information in the abstract. The dataset is split into 500 questions for development and 500 for testing. PubMedQA assesses a model's ability to comprehend and reason over scientific biomedical literature.

![Screenshot 2024-04-13 at 2.46.17 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/b35a8500-a3b2-48af-b6fb-0907cdcd8e77/Screenshot_2024-04-13_at_2.46.17_PM.png)

### MMLU Subsets (Medicine and Biology)

The [MMLU benchmark](https://arxiv.org/abs/2009.03300) (Measuring Massive Multitask Language Understanding) includes multiple-choice questions from various domains. For the Open Medical-LLM Leaderboard, we focus on the subsets most relevant to medical knowledge:

- Clinical Knowledge: 265 questions assessing clinical knowledge and decision-making skills.
- Medical Genetics: 100 questions covering topics related to medical genetics.
- Anatomy: 135 questions evaluating the knowledge of human anatomy.
- Professional Medicine: 272 questions assessing knowledge required for medical professionals.
- College Biology: 144 questions covering college-level biology concepts.
- College Medicine: 173 questions assessing college-level medical knowledge.

Each MMLU subset consists of multiple-choice questions with 4 answer options and is designed to evaluate a model's understanding of specific medical and biological domains.

![Screenshot 2024-04-13 at 2.47.28 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/1db10348-5bbf-4d91-be67-d5387356db54/Screenshot_2024-04-13_at_2.47.28_PM.png)

The Open Medical-LLM Leaderboard offers a robust assessment of a model's performance across various aspects of medical knowledge and reasoning.


## Insights and Analysis

The Open Medical-LLM Leaderboard evaluates the performance of various large language models (LLMs) on a diverse set of medical question-answering tasks. Here are our key findings:

- Commercial models like GPT-4-base and Med-PaLM-2 consistently achieve high accuracy scores across various medical datasets, demonstrating strong performance in different medical domains.
- Open-source models, such as [Starling-LM-7B](https://huggingface.co/Nexusflow/Starling-LM-7B-beta), [gemma-7b](https://huggingface.co/google/gemma-7b), Mistral-7B-v0.1, and [Hermes-2-Pro-Mistral-7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B), show competitive performance on certain datasets and tasks, despite having smaller sizes of around 7 billion parameters.
- Both commercial and open-source models perform well on tasks like comprehension and reasoning over scientific biomedical literature (PubMedQA) and applying clinical knowledge and decision-making skills (MMLU Clinical Knowledge subset).

![Image source : [https://arxiv.org/pdf/2311.05112.pdf](https://arxiv.org/abs/2402.07023)](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/174711e7-e27b-431f-88a8-cc3fbcc64c33/a00c20bc-f459-4f5a-b236-afd77f3557c2_.png)


Google's model, [Gemini Pro](https://arxiv.org/abs/2312.11805) demonstrates strong performance in various medical domains, particularly excelling in data-intensive and procedural tasks like Biostatistics, Cell Biology, and Obstetrics & Gynecology. However, it shows moderate to low performance in critical areas such as Anatomy, Cardiology, and Dermatology, revealing gaps that require further refinement for comprehensive medical application.

![Image source : [https://arxiv.org/pdf/2311.05112.pdf](https://arxiv.org/abs/2402.07023)](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/dbc10ab2-e1b1-4f80-ba8d-25d5a37203e5/Screenshot_2024-04-13_at_12.29.29_AM.png)


## Submitting Your Model for Evaluation

To submit your model for evaluation on the Open Medical-LLM Leaderboard, follow these steps:

**1. Convert Model Weights to Safetensors Format**

First, convert your model weights to the safetensors format. Safetensors is a new format for storing weights that is safer and faster to load and use. Converting your model to this format will also allow the leaderboard to display the number of parameters of your model in the main table.

**2. Ensure Compatibility with AutoClasses**

Before submitting your model, make sure you can load your model and tokenizer using the AutoClasses from the Transformers library. Use the following code snippet to test the compatibility:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained(MODEL_HUB_ID)
model = AutoModel.from_pretrained("your model name")
tokenizer = AutoTokenizer.from_pretrained("your model name")

```

If this step fails, follow the error messages to debug your model before submitting it. It's likely that your model has been improperly uploaded.

**3. Make Your Model Public**

Ensure that your model is publicly accessible. The leaderboard cannot evaluate models that are private or require special access permissions.

**4. Remote Code Execution (Coming Soon)**

Currently, the Open Medical-LLM Leaderboard does not support models that require `use_remote_code=True`. However, the leaderboard team is actively working on adding this feature, so stay tuned for updates.

**5. Submit Your Model via the Leaderboard Website**

Once your model is in the safetensors format, compatible with AutoClasses, and publicly accessible, you can submit it for evaluation using the "Submit here!" panel on the Open Medical-LLM Leaderboard website. Fill out the required information, such as the model name, description, and any additional details, and click the submit button.

The leaderboard team will process your submission and evaluate your model's performance on the various medical QA datasets. Once the evaluation is complete, your model's scores will be added to the leaderboard, allowing you to compare its performance with other submitted models.

## What's next? Expanding the Open Medical-LLM Leaderboard

The Open Medical-LLM Leaderboard is committed to expanding and adapting to meet the evolving needs of the research community and healthcare industry. Key areas of focus include:

1. Incorporating a wider range of medical datasets covering diverse aspects of healthcare, such as radiology, pathology, and genomics, through collaboration with researchers, healthcare organizations, and industry partners.
2. Enhancing evaluation metrics and reporting capabilities by exploring additional performance measures beyond accuracy, such as Pointwise score and domain-specific metrics that capture the unique requirements of medical applications.
3. A few efforts are already underway in this direction. If you are interested in collaborating on the next benchmark we are planning to propose, please join our [Discord community](https://discord.gg/A5Fjf5zC69) to learn more and get involved. We would love to collaborate and brainstorm ideas!

If you're passionate about the intersection of AI and healthcare, building models for the healthcare domain, and care about safety and hallucination issues for medical LLMs, we invite you to join our vibrant [community on Discord](https://discord.gg/A5Fjf5zC69).

## Credits and Acknowledgments

![Screenshot 2024-04-13 at 3.02.48 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/fbc16be2-424f-4e04-9574-e76c2baf5178/Screenshot_2024-04-13_at_3.02.48_PM.png)

Special thanks to all the people who helped make this possible, including Clémentine Fourrier and the Hugging Face team. I would like to thank Andreas Motzfeldt, Aryo Gema, & Logesh Kumar Umapathi for their discussion and feedback on the leaderboard during development. Sincere gratitude to Prof. Pasquale Minervini for his time, technical assistance, and for providing GPU support from the University of Edinburgh.

## About Open Life Science AI

Open Life Science AI is a project that aims to revolutionize the application of Artificial intelligence in the life science and healthcare domains. It serves as a central hub for list of medical models, datasets, benchmarks, and tracking conference deadlines, fostering collaboration, innovation, and progress in the field of AI-assisted healthcare.  We strive to establish Open Life Science AI as the premier destination for anyone interested in the intersection of AI and healthcare. We provide a platform for researchers, clinicians, policymakers, and industry experts to engage in dialogues, share insights, and explore the latest developments in the field.

![Screenshot 2024-04-13 at 3.02.10 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e749ee15-500e-4660-b028-a1069816cfa3/7d48f302-53da-40bd-bd9e-23aa8eb83414/Screenshot_2024-04-13_at_3.02.10_PM.png)

## Citation

If you find our evaluations useful, please consider citing our work

**Medical-LLM Leaderboard**
```
@misc{Medical-LLM Leaderboard,
author = {Ankit Pal, Pasquale Minervini and Andreas Geert Motzfeldt},
title = {openlifescienceai/open_medical_llm_leaderboard},
year = {2024},
publisher = {Hugging Face},
howpublished = "\url{https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard}"
}
```

