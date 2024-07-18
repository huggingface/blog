---
title: "Docmatix - an instruct dataset for Document Visual Question Answering" 
thumbnail: /blog/assets/182_finetune-florence/thumbnail.png
authors:
- user: andito
- user: HugoLaurencon
---

# Docmatix - An instruct dataset for Document Visual Question Answering


During our work on Idefics2, we developed [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), an extensive collection of 50 datasets for the fine-tuning of Vision-Language Model (VLM). Through this process, we identified a significant gap in the availability of large-scale Document Visual Question Answering (DocVQA) datasets. The primary dataset we relied on was DocVQA, which contains 10,000 images and 39,000 question-answer (Q/A) pairs.
To address this limitation, we are excited to introduce Docmatix, a DocVQA dataset featuring 2.4 million images and 9.5 million Q/A pairs derived from 1.3 million PDF documents. This marks a 240X increase in scale compared to previous datasets.

<div align="center">

| Dataset              | # images | # Q/A pairs | # tokens   |
|----------------------|----------|-------------|------------|
| **Docmatix**         | **2,444,750**| **9,500,000**   | **390,000,000**|
| DocVQA               | 10,189   | 39,463      | 337,829    |
| VisualMRC            | 3,027    | 11,988      | 168,828    |
| InfoVQA              | 2,118    | 10,074      | 61,048     |

</div>


Docmatix is generated from PDFA, an extensive OCR dataset containing 2.1 million PDFs. We took the transcriptions from PDFA and employed a Phi-3-small model to generate Q/A pairs. To ensure the dataset's quality, we filtered the generations, discarding 15% of the Q/A pairs identified as hallucinations. To do so, we used regular expressions to detect code and removed answers that contained the keyword “unanswerable”. 
The dataset contains a row for each PDF. We converted the PDFs to images at a resolution of 150 dpi, and uploaded the processed images to the Hugging Face Hub for easy access. 
All the original PDFs in Docmatix can be traced back to the original PDFA dataset, providing transparency and reliability. Still, we uploaded the processed images for convenience because converting many PDFs to images can be resource-intensive.

<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/docmatix_processing.png" alt="Processing for Docmatix" style="width: 90%; height: auto;"><br>
 <em>Processing pipeline to generate Docmatix</em>
</p>

After processing the first small batch of the dataset, we performed several ablation studies to optimize the prompts. We aimed to generate around four pairs of Q/A per page. Too many pairs indicate a large overlap between them, while too few pairs suggest a lack of detail.
Additionally, we aimed for answers to be human-like, avoiding excessively short or long responses. We also prioritized diversity in the questions, ensuring minimal repetition. Interestingly, this last objective was easily accomplished once our prompts guided the Phi3 model to ask questions based on the specific information it received (e.g., "What are the titles of John Doe?").
The following plot presents some key statistics from our analysis:


<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/docmatix_prompt_analysis.png" alt="Prompt analysis Docmatix" style="width: 90%; height: auto;"><br>
 <em>Analysis of Docmatix per prompt</em>
</p>

To evaluate Docmatix's performance, we conducted ablation studies using the Florence-2 model. We trained two versions of the model for comparison. The first version was trained over several epochs on the DocVQA dataset. The second version was trained for one epoch on Docmatix (20% of the images and 4% of the Q/A pairs), followed by one epoch on DocVQA to ensure the model produced the correct format for DocVQA evaluation.
The results were notable: training on this small portion of Docmatix yielded a relative improvement of almost 20%. Additionally, the 0.7B Florence-2 model performed only 5% worse than the 8B Idefics2 model trained on a mixture of datasets.



<div align="center">

| Dataset                              | ANSL on DocVQA |
|--------------------------------------|----------------|
| Florence 2 fine-tuned on DocVQA      | 60.1           | 
| Florence 2 fine-tuned on Docmatix    | 71,4           |
| Idefics2                             | 74,0           | 

</div>

<iframe
  src="https://huggingface.co/datasets/HuggingFaceM4/Docmatix/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>


