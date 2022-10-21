---
title: "Accelerating Document AI" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
---

# Accelerating Document AI

<div class="blog-metadata">
    <small>Published September 02, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/decision-transformers-train.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/rajistics"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1652986473945-60f2e74cadf471cbdf8bb663.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>rajistics</code>
            <span class="fullname">Rajiv Shah</span>
        </div>
    </a>
    <a href="/nielsr"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1608042047613-5f1158120c833276f61f1a84.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>nielsr</code>
            <span class="fullname">Niels Rogge</span>
        </div>
    </a>
</div>

Enterprises are full of documents containing knowledge that isn't accessible by digital workflows. These documents can vary from letters, invoices, forms, reports, to receipts. With the improvements in text, vision, and multimodal AI, it's now possible to unlock that information. This post shows you how your teams can use open-source models to build custom solutions for free!

Document AI includes many data science tasks from [image classification](https://huggingface.co/tasks/image-classification), [image to text,](https://huggingface.co/tasks/image-to-text) [document question answering](https://huggingface.co/tasks/document-question-answering), [table question answering](https://huggingface.co/tasks/table-question-answering), and [visual question answering](https://huggingface.co/tasks/visual-question-answering). This post starts with a taxonomy of use cases within Document AI and the best open-source models for those use cases. Next, the post focuses on licensing, data preparation, and modeling. Throughout this post, there are links to web demos, documentation, and models. 

### Use Cases

There are at least six general use cases for building document AI solutions. These use cases differ in the kind of document inputs and outputs. A combination of approaches is often necessary when solving enterprise Document AI problems.

**Optical Character Recognition (OCR)**

Turning typed, handwritten, or printed text into machine-encoded text is known as Optical Character Recognition (OCR). It's a widely studied problem with many well-established open-source and commercial offerings. The figure shows an example of converting handwriting into text.

[https://lh3.googleusercontent.com/0pvA_Wgk2umfNAyDfIjQ_ELkv3qZ2umbC0OLVR7lXVoQUxegfBoB0O3X0KOehiI3nCDuDC6UCA5ImlKFzd-phpLou-ltQh8CLU_33UdOzO44x7I455B2OVsm4SvaTdN0Nkpz5VpmBxk3pgZXSVBD87mYwHajdcjnuKp_KixJsoXV3TUmoS52szSZvw](https://lh3.googleusercontent.com/0pvA_Wgk2umfNAyDfIjQ_ELkv3qZ2umbC0OLVR7lXVoQUxegfBoB0O3X0KOehiI3nCDuDC6UCA5ImlKFzd-phpLou-ltQh8CLU_33UdOzO44x7I455B2OVsm4SvaTdN0Nkpz5VpmBxk3pgZXSVBD87mYwHajdcjnuKp_KixJsoXV3TUmoS52szSZvw)

OCR is a backbone of Document AI use cases as it's essential to transform the text into something readable by a computer. Some widely available OCR models that operate at the document level are [EasyOCR](https://huggingface.co/spaces/tomofi/EasyOCR) or [PaddleOCR](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR). There are also models like [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/docs/transformers/model_doc/trocr), which runs on single-text line images. This model works with a text detection model like CRAFT which first identifies the individual "pieces" of text in a document in the form of bounding boxes. The relevant metrics for OCR are Character Error Rate (CER) and word-level precision, recall, and F1. Check out [this Space](https://huggingface.co/spaces/tomofi/CRAFT-TrOCR) to see a demonstration of CRAFT and TrOCR.  

**Document Image Classification**

Classifying documents into the appropriate category, such as forms, invoices, or letters, is known as document image classification. Classification may use either one or both of the document's image and text. The recent addition of multimodal models that use the visual structure and the underlying text has dramatically increased classifier performance.

A basic approach (and an excellent starting point) is applying OCR on a document image, after which a [BERT](https://huggingface.co/docs/transformers/model_doc/bert)-like model is used for classification. However, relying on only a BERT model doesn't take any layout or visual information into account. The examples below show that the visual structure is often quite different between different document types.

[https://lh4.googleusercontent.com/JvVI3Hbu1lDKUntocHYTQQ76F9scwJ9-fpG9XKnoc8XHzPt6cNW78M3hZH2oAeJXiku4KA-DFGq6Iw4iIXzjpXZIgSKjEDISmxYgxFuCGe8nVK8mBGOcObxNhldwlN5TrYQM2kUFGF7YpxfRDi9w_-rEZ2VC8dLItXmkVFOczF--HwRXb9gKi8YVzQ](https://lh4.googleusercontent.com/JvVI3Hbu1lDKUntocHYTQQ76F9scwJ9-fpG9XKnoc8XHzPt6cNW78M3hZH2oAeJXiku4KA-DFGq6Iw4iIXzjpXZIgSKjEDISmxYgxFuCGe8nVK8mBGOcObxNhldwlN5TrYQM2kUFGF7YpxfRDi9w_-rEZ2VC8dLItXmkVFOczF--HwRXb9gKi8YVzQ)

That's where models like [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) and [Donut](https://huggingface.co/docs/transformers/model_doc/donut) come into play. By incorporating not only text but also visual information, these models can dramatically increase accuracy. For comparison, on [RVL-CDIP](https://huggingface.co/datasets/rvl_cdip), an important benchmark for document image classification, a BERT-base model achieves 89% accuracy by using the text. A [DiT](https://huggingface.co/docs/transformers/main/en/model_doc/dit) (Document Image Transformer) is a pure vision model (i.e., it does not take text as input) and can reach 92% accuracy. But models like [LayoutLMv3](https://huggingface.co/docs/transformers/main/en/model_doc/layoutlmv3) and [Donut](https://huggingface.co/docs/transformers/model_doc/donut), which use the text and visual information together using a multimodal Transformer, can achieve 95% accuracy! These multimodal models are changing how practitioners solve Document AI use cases.

**Document layout analysis**

Document layout analysis is the task of determining the physical structure of a document, i.e., identifying the individual building blocks that make up a document, like text segments, headers, and tables. This task is often solved by framing it as an image segmentation/object detection problem. The model outputs a set of segmentation masks/bounding boxes, along with class names.

Models that are currently state-of-the-art for document layout analysis are [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) and [DiT](https://huggingface.co/docs/transformers/model_doc/dit) (Document Image Transformer). Both models use the classic [Mask R-CNN](https://arxiv.org/abs/1703.06870) framework for object detection as a backbone. This [document layout analysis](https://huggingface.co/spaces/nielsr/dit-document-layout-analysis) Space illustrates how DiT can be used to identify text segments, titles, and tables in documents.

[https://lh4.googleusercontent.com/fF8_8Bwcn7uNHRotxMNCbCbfFvJJv2Vj8N-kSK5qtP7ut7nRRTwAv63STdSMucgFYwMARTYuF07T4B2UqE28e8m7IclkfjXJO5oHkpcihI6_NZcGObrHxcaXXYP95U5FYnqra1A_B5bQ1Y4vGK-bU2pn3cEDJcxtbWx4zQ1vzG1GN1wYzzh0q0ZsIA](https://lh4.googleusercontent.com/fF8_8Bwcn7uNHRotxMNCbCbfFvJJv2Vj8N-kSK5qtP7ut7nRRTwAv63STdSMucgFYwMARTYuF07T4B2UqE28e8m7IclkfjXJO5oHkpcihI6_NZcGObrHxcaXXYP95U5FYnqra1A_B5bQ1Y4vGK-bU2pn3cEDJcxtbWx4zQ1vzG1GN1wYzzh0q0ZsIA)

[https://lh3.googleusercontent.com/3ZPB5m42VQwt4uWqUG9ErpHeogb9QTAFs0MgdPHijKtzFp5VvLaxTyq_SWRz0daZgKIzLJY9HnV7L7DgcZopMlPL_W01aUg1tDFhW9Aohlc1IeEPmf3mufvIWA3rilaVz9N3PoBFpnSWgOGWnwsiWd13m82BPgJzBOtUiLZxLmC3sQjAG9tW2Tho8Q](https://lh3.googleusercontent.com/3ZPB5m42VQwt4uWqUG9ErpHeogb9QTAFs0MgdPHijKtzFp5VvLaxTyq_SWRz0daZgKIzLJY9HnV7L7DgcZopMlPL_W01aUg1tDFhW9Aohlc1IeEPmf3mufvIWA3rilaVz9N3PoBFpnSWgOGWnwsiWd13m82BPgJzBOtUiLZxLmC3sQjAG9tW2Tho8Q)

Document layout analysis with DiT.

Document layout analysis typically uses the mAP (mean average-precision) metric, often used for evaluating object detection models. An important benchmark for layout analysis is the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset. LayoutLMv3, the state-of-the-art at the time of writing, achieves an overall mAP score of 0.951 ([source](https://paperswithcode.com/sota/document-layout-analysis-on-publaynet-val)).

**Document parsing**

A step beyond layout analysis is document parsing. Document parsing is identifying and extracting key information from a document, such as names, items, and totals from an invoice form. This [LayoutLMv2 Space](https://huggingface.co/spaces/nielsr/LayoutLMv2-FUNSD) shows to parse a document to recognize questions, answers, and headers.

The first version of LayoutLM (now known as LayoutLMv1) was released in 2020 and dramatically improved over existing benchmarks, and it's still one of the most popular models on the Hugging Face Hub for Document AI. LayoutLMv2 and LayoutLMv3 incorporate visual features during pre-training, which provides an improvement. The LayoutLM family produced a step change in Document AI performance. For example, on the [FUNSD](https://guillaumejaume.github.io/FUNSD/) benchmark dataset, a BERT model has an F1 score of 60%, but with LayoutLM, it is possible to get to 90%! 

LayoutLMv1 now has many successors. [Donut](https://huggingface.co/docs/transformers/model_doc/donut) builds on LayoutLM but can take the image as input, so it doesn't require a separate OCR engine. For multilingual use cases, there are multilingual variants of LayoutLM, like [LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm) and [LiLT](https://huggingface.co/docs/transformers/main/en/model_doc/lilt).

Data scientists are finding document layout analysis and extraction as key use cases for enterprises. The existing commercial solutions typically cannot handle the diversity of most enterprise data, in content and structure. Consequently, data science teams can often surpass commercial tools by fine-tuning their own models.

[https://lh3.googleusercontent.com/YEwL9DOP2wGk6V2Z4Uix-BM9I4s8BhPXSTwJgJcs_RJq3Pm7xr0D-0407AGuOwsqbURzM7yspyJgtaAaCJSmvnFAZqfPOmDFctyxwZxJrNxuu7zIFxQAy5Tlt_ff2DUFkfj4a4L3QKBPw7P9R7tu5GisjI6hVbWXlwgLdzj7CSgcdpZAa7trx19SNw](https://lh3.googleusercontent.com/YEwL9DOP2wGk6V2Z4Uix-BM9I4s8BhPXSTwJgJcs_RJq3Pm7xr0D-0407AGuOwsqbURzM7yspyJgtaAaCJSmvnFAZqfPOmDFctyxwZxJrNxuu7zIFxQAy5Tlt_ff2DUFkfj4a4L3QKBPw7P9R7tu5GisjI6hVbWXlwgLdzj7CSgcdpZAa7trx19SNw)

**Table detection, extraction, and table structure recognition**

Documents often contain tables, and most OCR tools don't work incredibly well out-of-the-box on tabular data. Table detection is the task of identifying where tables are located, and table extraction creates a structured representation of that information. Table structure recognition is the task of identifying the individual pieces that make up a table, like rows, columns, and cells. Table functional analysis (FA) is the task of recognizing the keys and values of the table. The figure below illustrates the difference between the various subtasks.

[https://lh5.googleusercontent.com/3QCNUwmlzix6u5zGmp2iaxYmhCn-p5oHpVT0QTUC0v3s78Kb9YrDmhnSl4zR68mIXJTdGlILfW2wBt3FhvPvdNkHZeAQLwU-lJvwdeV-RxZpwM8ktb-Hc48hxaaWPjU3u0CLuGvQqvrvVLN2lI1TRKnxVFzkNJftCkG_HQMzfHqeBJ4YEipYdtpLzQ](https://lh5.googleusercontent.com/3QCNUwmlzix6u5zGmp2iaxYmhCn-p5oHpVT0QTUC0v3s78Kb9YrDmhnSl4zR68mIXJTdGlILfW2wBt3FhvPvdNkHZeAQLwU-lJvwdeV-RxZpwM8ktb-Hc48hxaaWPjU3u0CLuGvQqvrvVLN2lI1TRKnxVFzkNJftCkG_HQMzfHqeBJ4YEipYdtpLzQ)

The approach for table detection and structure recognition is similar to document layout analysis in using object detection models that output a set of bounding boxes and corresponding classes.

The latest approaches, like [Table Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer), can enable table detection and table structure recognition with the same model. The Table Transformer is a [DETR](https://huggingface.co/docs/transformers/model_doc/detr)-like object detection model, trained on [PubTables-1M](https://arxiv.org/abs/2110.00061) (a dataset comprising one million tables). Evaluation for table detection and structure recognition typically uses the average precision (AP) metric. The Table Transformer performance is reported as having an AP of 0.966 for table detection and an AP of 0.912 for table structure recognition + functional analysis on PubTables-1M.

Table detection and extraction is an exciting approach, but the results may be different on your data. In our experience, the quality and formatting of tables vary widely and can affect how well the models perform. Additional fine-tuning on some custom data will greatly improve the performance.

**Document question answering (DocVQA)**

Question answering on documents has dramatically changed how people interact with AI. Recent advancements have made it possible to ask models to answer questions about an image - this is known as document visual question answering, or DocVQA for short. After being given a question, the model analyzes the image and responds with an answer. An example of this is shown in the figure below. The user asks," Mention the ZIP code written? " and the model responds with the answer.

[https://lh6.googleusercontent.com/58W8Q5EeD_Pen5XcD9H6RGdw9RysuVd3n9-R7sLEHrsuJGsby_wWIUFDHAhs64Q4fvZE3j1rwNM4DwH7EgOTMeLQkRSd4_w0f6ruc5yR12rsRTUbJ7WzZhH8SG3bs76kre55oshfClTammEqq4dVV3V9pil6SyKTW06_BXG8Gea95HtzWlidYqzDXA](https://lh6.googleusercontent.com/58W8Q5EeD_Pen5XcD9H6RGdw9RysuVd3n9-R7sLEHrsuJGsby_wWIUFDHAhs64Q4fvZE3j1rwNM4DwH7EgOTMeLQkRSd4_w0f6ruc5yR12rsRTUbJ7WzZhH8SG3bs76kre55oshfClTammEqq4dVV3V9pil6SyKTW06_BXG8Gea95HtzWlidYqzDXA)

In the past, building a DocVQA system would often require multiple models working together. There could be separate models for analyzing the document layout, performing OCR, extracting entities, and then answering a question. The latest DocVQA models enable question-answering in an end-to-end manner, comprising only a single (multimodal) model.

DocVQA is typically evaluated using the Average Normalized Levenshtein Similarity (ANLS) metric. For more details regarding this metric, we refer to [this guide](https://rrc.cvc.uab.es/?ch=11&com=tasks). The current state-of-the-art on the DocVQA benchmark that is open-source is [LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) which achieves an ANLS score of 83.37. However, this model consists of a pipeline of OCR + multimodal Transformer. [Donut](https://huggingface.co/docs/transformers/model_doc/donut) solves the task in an end-to-end manner using a single encoder-decoder Transformer, not relying on OCR. Donut doesn't provide state-of-the-art accuracy but shows the great potential of the end-to-end approach using a generative T5-like model. Impira hosts an [exciting Space](https://huggingface.co/spaces/impira/docquery) that illustrates LayoutLM and Donut for DocVQA.

Visual question answering is compelling; however, there are many considerations for successfully using it. Having accurate training data, evaluation metrics, and post-processing is vital. For teams taking on this use case, be aware that DocVQA can be challenging to work properly. In some cases, responses can be unpredictable, and the model can “hallucinate” by giving an answer that doesn't appear within the document. Ensuring proper model setup and post-processing is integral to building a successful DocVQA solution.

### Licensing Issues

Industry and academia make enormous contributions to advancing Document AI. There are a wide assortment of models and datasets available for data scientists to use. However, licensing can be a non-starter for building an enterprise solution. Some well-known models have restrictive licenses that forbid the model from being used for commercial purposes. Most notably, Microsoft has decided that LayoutLMv2 and LayoutLMv3 cannot be used commercially. When you start a project, we advise carefully evaluating the license of prospective models. Knowing which models you want to use is essential at the outset, since that may affect data collection and annotation.

### Data Preparation Issues

Data preparation for Document AI is critical and challenging. It's crucial to have properly annotated data. Here are some lessons we have learned along with the way around data preparation.

First, machine learning depends on the scale and quality of your data. If the image quality of your documents is poor, you can't expect AI to be able to read these documents magically. Similarly, if your training data is small with many classes, your performance may be poor. Document AI is like other problems in machine learning where larger data will generally provide greater performance.

Second, be flexible in your approaches. You may need to test several different methodologies to find the best solution. A great example is OCR, in which you can use an open-source product like Tesseract, a commercial solution like Cloud Vision API, or the OCR capability inside an open-source multimodal model like Donut.

Third, start small with annotating data and pick your tools wisely. In our experience, you can get good results with several hundred documents. So start small and carefully evaluate your performance. Once you have narrowed your overall approach, you can begin to scale up the data to maximize your predictive accuracy. When annotating, remember that some tasks like layout identification and document extraction require identifying a specific region within a document. You will want to ensure your annotation tool supports bounding boxes.

### Modeling Issues

The flexibility of building your models leads to many options for data scientists. Our strong recommendation for teams is to start with the pre-trained open-source models. These models can be fine-tuned to your specific documents, and this is generally the quickest way to a good model.

For teams considering building their own pre-trained model, be aware this can involve millions of documents and can easily take several weeks to train a model. Building a pre-trained model requires significant effort and is not recommended for most data science teams. Instead, start with fine-tuning one, but ask yourself these questions first.

Do you want the model to handle the OCR? For example, Donut doesn't require the document to be OCRed and directly works on full-resolution images, so there is no need for OCR before modeling. However, depending on your problem setup, it may be simpler to get OCR separately.

Should you use higher-resolution images? When using images with LayoutLMv2, it downscales them to 224 by 224, whereas Donut uses the full high-resolution image. However, using the full high-resolution image dramatically increases the memory required for training and inference.

How are you evaluating the model? Watch out for misaligned bounding boxes. You should ensure bounding boxes provided by the OCR engine of your choice align with the model processor. Verifying this can save you from unexpectedly poor results. Second, let your project requirements guide your evaluation metrics. For example, in some tasks like token classification or question answering, a 100% match may not be the best metric. A metric like partial match could allow for many more potential tokens to be considered, such as “Acme” and “inside Acme” as a match.

### Next Steps

Are you seeing the possibilities of Document AI? Every day we work with enterprises to unlock valuable data using state-of-the-art vision and language models. We included links to various demos throughout this post, so use them as a starting point. The last section of the post contains resources for starting to code up your own models, such as visual question answering. Once you are ready to start building your solutions, the Hugging Face public hub is a great starting point. It hosts a vast array of Document AI models.

If you want to accelerate your Document AI efforts, Hugging Face can help. Through our Enterprise Acceleration Program (EAP) we partner with enterprises to provide guidance on AI use cases. For Document AI, this could involve helping build a pre-train model, improving accuracy on a fine-tuning task, or providing overall guidance on tackling your first Document AI use case.

### Resources

Tutorial notebooks for many Document AI models are in [Niels Github](https://github.com/NielsRogge/Transformers-Tutorials) repository: [https://github.com/NielsRogge/Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials).

### Popular Open-Source Models for Document AI

A table of the currently available Transformers models achieving state-of-the-art performance on Document AI tasks. This was last updated in October 2022.

| model | paper | license | checkpoints |
| --- | --- | --- | --- |
| Donut | arxiv | MIT | huggingface |
| LayoutLM | arxiv | MIT | huggingface |
| LayoutXLM | arxiv | CC BY-NC-SA 4.0 | huggingface |
| LayoutLMv2 | arxiv | CC BY-NC-SA 4.0 | huggingface |
| LayoutLMv3 | arxiv | CC BY-NC-SA 4.0 | huggingface |
| DiT | arxiv | CC BY-NC-SA 4.0 | huggingface |
| TrOCR | arxiv | MIT | huggingface |
| Table Transformer | arxiv | MIT | huggingface |
| LiLT | arxiv | MIT | huggingface |

### Metrics and Datasets for Document AI

A table of the common metrics and datasets for command Document AI tasks. This was last updated in October 2022.

| task | typical metrics | benchmark datasets |
| --- | --- | --- |
| Optical Character Recognition | Character Error Rate (CER) |  |
| Document Image Classification | Accuracy, F1 | RVL-CDIP |
| Document layout analysis | mAP (mean average precision) | PubLayNet, ‣ (Forms) |
| Document parsing | Accuracy, F1 | FUNSD, SROIE (receipts), CORD |
| Table Detection and Extraction | mAP (mean average precision) | PubTables-1M |
| Document visual question answering | Average Normalized Levenshtein Similarity (ANLS) | DocVQA |
|  |  |  |
