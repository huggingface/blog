---
title: "Supercharge your OCR Pipelines with Open Models"
thumbnail: /blog/assets/ocr-open-models/thumbnail.png
authors:
- user: merve
- user: ariG23498
- user: davanstrien
- user: hynky
- user: andito
- user: reach-vb
- user: pcuenq
---

# Supercharge your OCR Pipelines with Open Models

TL;DR: The rise of powerful vision-language models has transformed document AI. Each model comes with unique strengths, making it tricky to choose the right one. Open-weight models offer better cost efficiency and privacy. To help you get started with them, we’ve put together this guide.

In this guide, you’ll learn:

* The landscape of current models and their capabilities  
* When to fine-tune models vs. use models out-of-the-box  
* Key factors to consider when selecting a model for your use case  
* How to move beyond OCR with multimodal retrieval and document QA

By the end, you’ll know how to choose the right OCR model, start building with it, and gain deeper insights into document AI. Let’s go\!

## Table-of-Contents 

- [Supercharge your OCR Pipelines with Open Models](#supercharge-your-ocr-pipelines-with-open-models)
  - [Brief Introduction to Modern OCR](#brief-introduction-to-modern-ocr)
    - [Model Capabilities](#model-capabilities)
      - [Transcription](#transcription)
      - [Handling complex components in documents](#handling-complex-components-in-documents)
      - [Output formats](#output-formats)
      - [Locality Awareness in OCR](#locality-awareness-in-ocr)
      - [Model Prompting](#model-prompting)
  - [Cutting-edge Open OCR Models](#cutting-edge-open-ocr-models)
    - [Comparing Latest Models](#comparing-latest-models)
    - [Evaluating Models](#evaluating-models)
      - [Benchmarks](#benchmarks)
      - [Cost-efficiency](#cost-efficiency)
      - [Open OCR Datasets](#open-ocr-datasets)
  - [Tools to Run Models](#tools-to-run-models)
    - [Locally](#locally)
    - [Remotely](#remotely)
  - [Going Beyond OCR](#going-beyond-ocr)
    - [Visual Document Retrievers](#visual-document-retrievers)
    - [Using Vision Language Models for Document Question Answering](#using-vision-language-models-for-document-question-answering)
  - [Wrapping up](#wrapping-up)

## Brief Introduction to Modern OCR 

Optical Character Recognition (OCR) is one of the earliest and longest running challenges in computer vision.  Many of AI’s first practical applications focused on turning printed text into digital form.

With the surge of [vision-language models](https://huggingface.co/blog/vlms) (VLMs), OCR has advanced significantly. Recently, many OCR models have been developed by fine-tuning existing VLMs. But today’s capabilities extend far beyond OCR: you can retrieve documents by query or answer questions about them directly. Thanks to stronger vision features, these models can also handle low-quality scans, interpret complex elements like tables, charts, and images, and fuse text with visuals to answer open-ended questions across documents.

### Model Capabilities

#### Transcription
Recent models transcribe texts into a machine-readable format.   
The input can include: 

- Handwritten text   
- Various scripts like Latin, Arabic, and Japanese characters  
- Mathematical expressions   
- Chemical formulas  
- Image/Layout/Page number tags

	  
OCR models convert them into machine-readable text that comes in many different formats like HTML, Markdown and more.  
	

#### Handling complex components in documents

On top of text, some models can also recognize:

- Images  
- Charts  
- Tables

Some models know where images are inside the document, extract their coordinates, and insert them appropriately between texts. Other models generate captions for images and insert them where they appear. This is especially useful if you are feeding the machine-readable output into an LLM. Example models are [OlmOCR by AllenAI](https://huggingface.co/allenai/olmOCR-7B-0825), or [PaddleOCR-VL by PaddlePaddle](https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

Models use different machine-readable output formats, such as **DocTags**, **HTML** or **Markdown** (explained in the next section *Output Formats*). The way a model handles tables and charts often depends on the output format they are using. Some models treat charts like images: they are kept as is. Other models convert charts into markdown tables or JSON, e.g., a bar chart can be converted as follows. 

![Chart Rendering](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/chart-rendering.png)

Similarly for tables, cells are converted into a machine-readable format while retaining context from headings and columns. 

![Table Rendering](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/table-rendering.png)

#### Output formats
Different OCR models have different output formats. Briefly, here are the common output formats used by modern models.   
**DocTag:** DocTag is an XML-like format for documents that expresses location, text format, component-level information, and more. Below is an illustration of a paper parsed into DocTags. This format is employed by the open Docling models.  

![DocTags](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/doctags_v2.png)  

- **HTML:** HTML is one of the most popular output formats used for document parsing as it properly encodes structure and hierarchical information.   
- **Markdown:** Markdown is the most human-readable format. It’s simpler than HTML but not as expressive. For example, it can’t represent split-column tables.  
- **JSON:** JSON is not a format that models use for the entire output, but it can be used to represent information in tables or charts.

The right model depends on how you plan to use its outputs:

* **Digital reconstruction**: To reconstruct documents digitally, choose a model with a layout-preserving format (e.g., DocTags or HTML).  
* **LLM input or Q\&A**: If the use case involves passing outputs to LLM, pick a model that outputs Markdown and image captions, since they’re closer to natural language.  
* **Programmatic use**: If you want to pass your outputs to a program (like data analysis), opt for a model that generates structured outputs like JSON.

#### Locality Awareness 

Documents can have complex structures, like multi-column text blocks and floating figures. Older OCR models handled these documents by detecting words and then the layout of pages manually in post-processing to have the text rendered in reading order, which is brittle.  Modern OCR models, on the other hand, incorporate layout metadata to help preserve reading order and accuracy. This metadata is called “anchor”, it can come in bounding boxes. This process is also called as “grounding/anchoring” because it helps with reducing hallucination.


#### Model Prompting

OCR models can either take in images and an optional text prompt, this depends on the model architecture and the pre-training setup.   
Some OCR models support prompt-based task switching, e.g. [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M) can parse an entire page with the prompt “Convert this page to Docling” while it can also take prompts like “Convert this formula to LaTeX” along with a page full of formulas.   
Other models, however, are trained only for parsing entire pages, and they are conditioned to do this through a system prompt.   
For instance, [OlmOCR by AllenAI](https://huggingface.co/collections/allenai/olmocr-67af8630b0062a25bf1b54a1) takes a long conditioning prompt. Like many others, OlmOCR is technically an OCR fine-tuned version of a VLM (Qwen2.5VL in this case), so you can prompt for other tasks, but its performance will not be on par with the OCR capabilities. 

## Cutting-edge Open OCR Models

We’ve seen an incredible wave of new models this past year. Because so much work is happening in the open, these players build on and benefit from each other’s work. A great example is AllenAI’s release of OlmOCR, which not only released a model but also the dataset used to train it. With these, others can build upon them in new directions. The field is incredibly active, but it’s not always obvious which model to use. 

### Comparing Latest Models

To make things a bit easier, we’re putting together a non-exhaustive comparison of some of our current favorite models. All of the models below are layout-aware and can parse tables, charts, and math equations. The full list of languages each model supports are detailed in their model cards, so make sure to check them if you’re interested.

| Model Name | Output formats | Features | Model Size | Multilingual? |
| :---- | :---- | :---- | :---- | :---- |
| [Nanonets-OCR2-3B](https://huggingface.co/collections/nanonets/nanonets-ocr2-68ed207f17ee6c31d226319e) | structured Markdown with semantic tagging (plus HTML tables, etc.) | Captions images in the documents<br>Signature & watermark extraction<br>Handles checkboxes, flowcharts, and handwriting | 4B | ✅Supports English, Chinese, French, Arabic and more. |
| [PaddleOCR-VL](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl-68f0db852483c7af0bc86849) | Markdown, JSON, HTML tables and charts | Handles handwriting, old documents<br>Allows prompting<br>Converts tables & charts to HTML<br>Extracts and inserts images directly | 0.9B | ✅Supports 109 languages |
| [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | Markdown, JSON | Grounding<br>Extracts and inserts images<br>Handles handwriting | 3B | ✅Multilingual with language info not available |
| [OlmOCR](https://huggingface.co/allenai/olmOCR-7B-0825) | Markdown, HTML, LaTeX | Grounding<br>Optimized for large-scale batch processing | 8B | ❎English-only |
| [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | DocTags | Prompt-based task switching<br>Ability to prompt element locations with location tokens<br>Rich output | 258M | ✅Supports English, Japanese, Arabic and Chinese. |
| [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | Markdown + HTML | Supports general visual understanding<br>Can parse and re-render all charts, tables, and more into HTML<br>Handles handwriting<br>Memory-efficient, solves text through image | 3B | ✅Supports nearly 100 languages |

Here’s a [small demo](https://prithivMLmods-Multimodal-OCR3.hf.space) for you to try some of the latest models and compare their outputs.   
<iframe  
    src="https://prithivMLmods-Multimodal-OCR3.hf.space"  
    frameborder="0"  
    width="850"  
    height="450"

></iframe>

### Evaluating Models

#### Benchmarks

There’s no single best model, as every problem has different needs. Should tables be rendered in Markdown or HTML? Which elements should we extract? How should we quantify text accuracy and error rates? 👀  
While there are many evaluation datasets and tools, many don’t answer these questions. So we suggest using the following benchmarks:

1. [**OmniDocBenchmark**](https://huggingface.co/datasets/opendatalab/OmniDocBench)**:** This widely used benchmark stands out for its diverse document types: books, magazines, and textbooks. Its evaluation criteria are well designed, accepting tables in both HTML and Markdown formats. A novel matching algorithm evaluates the reading order, and formulas are normalized before evaluation. Most metrics rely on edit distance or tree edit distance (tables). Notably, the annotations used for evaluation are not solely human-generated but are acquired through SoTA VLMs or conventional OCR methods.  
2. [**OlmOCR-Bench**](https://huggingface.co/datasets/allenai/olmOCR-bench): OlmOCR-Bench takes a different approach: they treat the evaluation as a set of unit tests. For example, table evaluation is done by checking the relation between selected cells of a given table. They use PDFs from public sources, and annotations are done using a wide range of closed-source VLMs. This benchmark is quite successful to evaluate on the English language.  
3. [**CC-OCR (Multilingual)**:](https://huggingface.co/datasets/wulipc/CC-OCR) Compared to the previous benchmarks, CC-OCR is less preferred when picking models, due to lower document quality and diversity. However, it’s the only benchmark that contains evaluation beyond English and Chinese\! While the evaluation is far from perfect (images are photos with few words), it’s still the best you can do for multilingual evaluation.

When testing different OCR models, we've found that the performance across different document types, languages, etc., varies a lot. Your domain may not be well represented in existing benchmarks\! To make effective use of this new generation of VLM-based OCR models we suggest aiming to collect a dataset of representative examples of your task domain and testing a few different models to compare their performance. 

#### Cost-efficiency

Most OCR models are small, having between 3B and 7B parameters; you can even find models with fewer than 1B parameters, like PaddleOCR-VL. However, the cost also depends on the availability of optimized implementations  inference frameworks. For example, OlmOCR comes with vLLM and SGLang implementations, and the cost per million pages is 190 dollars (assuming on H100 for $2.69/hour). DeepSeek-OCR can process 200k+ pages per day on a single A100 with 40GB VRAM.  With napkin math, we see that the cost per million pages is more or less similar to OlmOCR (although it depends on your A100 provider). If your use case remains unaffected, you can also opt for quantized versions of the models. The cost of running open-source models heavily depends on the hourly cost of the instance and the optimizations the model includes, but it’s guaranteed to be cheaper than many closed-source models out there on a larger scale.

#### Open OCR Datasets 

While the past year has seen a surge in open OCR models, this hasn't been matched by as many open training and evaluation datasets. An exception is AllenAI's [olmOCR-mix-0225](https://huggingface.co/datasets/allenai/olmOCR-mix-0225), which has been used to train at least [72 models on the Hub](https://huggingface.co/models?dataset=dataset:allenai/olmOCR-mix-0225) – likely more, since not all models document their training data.

Sharing more datasets could unlock even greater advances in open OCR models. There are several promising approaches for creating these datasets:

- **Synthetic data generation** (e.g., [isl_synthetic_ocr](https://huggingface.co/datasets/Sigurdur/isl_synthetic_ocr))  
- **VLM-generated transcriptions** filtered manually or through heuristics  
- **Using existing OCR models** to generate training data for new, potentially more efficient models in specific domains  
- **Leveraging existing corrected datasets** like the [Medical History of British India Dataset](https://huggingface.co/NationalLibraryOfScotland), which contains extensively human-corrected OCR for historic documents

It's worth noting that many such datasets exist but remain unused. Making them more readily available as 'training-ready' datasets carries a considerable potential for the open-source community.

## Tools to Run Models

We have received many questions about getting started with OCR models, so here are a few ways you can use local inference tools and host remotely with Hugging Face.

### Locally

Most cutting-edge models come with vLLM support and transformers implementation. You can get more info about how to serve each from the models’ own cards. For convenience, we show how to infer locally using vLLM here. The code below can differ from model to model, but for most models it looks like the following. 

```shell
vllm serve nanonets/Nanonets-OCR2-3B
```

And then you can query as follows using e.g. OpenAI client. 

```py
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1")

model = "nanonets/Nanonets-OCR2-3B"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def infer(img_base64):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

img_base64 = encode_image(your_img_path)
print(infer(img_base64))
```

**Transformers**

Transformers provides standard model definitions for easy inference and fine-tuning. Models available in transformers come with either official transformers implementation (model definitions within the library) or “remote code” implementations. Latter is defined by the model owners to enable easy loading of models into transformers interface, so you don’t have to go through the model implementation. Below is an example loading Nanonets model using transformers implementation.

```py
# make sure to install flash-attn and transformers
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "nanonets/Nanonets-OCR2-3B", 
    torch_dtype="auto", 
    device_map="auto", 
    attn_implementation="flash_attention_2"
)
model.eval()
processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR2-3B")

def infer(image_url, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

result = infer(image_path, model, processor, max_new_tokens=15000)
print(result)
```

**MLX**  
MLX is an open-source machine learning framework for Apple Silicon. [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) is built on top of MLX to serve vision language models easily. You can explore all the OCR models available in MLX format [here](https://huggingface.co/models?sort=trending&search=ocr). They also come in quantized versions.  
You can install MLX-VLM as follows.

```
pip install -U mlx-vlm
```

```
wget https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/throughput_smolvlm.png

python -m mlx_vlm.generate --model ibm-granite/granite-docling-258M-mlx --max-tokens 4096 --temperature 0.0 --prompt "Convert this chart to JSON." --image throughput_smolvlm.png 

```

### Remotely

**Inference Endpoints for Managed Deployment**  
You can deploy OCR models compatible with vLLM or SGLang on Hugging Face Inference Endpoints, either from a model repository “Deploy” option or directly through [Inference Endpoints interface](https://endpoints.huggingface.co/). Inference Endpoints serve the cutting-edge models in a fully managed environment with GPU acceleration, auto-scaling, and monitoring without manually managing the infrastructure.  
   
Here is a simple method of deploying `nanonets` using vLLM as the inference engine.

1. Navigate to the model repository [`nanonets/Nanonets-OCR2-3B`](https://huggingface.co/nanonets/Nanonets-OCR2-3B)  
2. Click on the “Deploy” button and select the “HF Inference Endpoints”

![Inference Endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/IE.png)

3. Configure the deployment setup within seconds

![Inference Endpoints](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ocr/IE2.png)

4. After the endpoint is created, you can consume it using the OpenAI client snippet we provided in the previous section.

You can learn more about it [here](https://huggingface.co/docs/inference-endpoints/engines/vllm).

**Hugging Face Jobs for Batch Inference** 

For many OCR applications, you want to do efficient batch inference, i.e., running a model across thousands of images as cheaply and efficiently as possible. A good approach is to use vLLM's offline inference mode. As discussed above, many recent VLM-based OCR models are supported by vLLM, which efficiently batches images and generates OCR outputs at scale.

To make this even easier, we've created [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr), a collection of ready-to-run OCR scripts that work with Hugging Face Jobs. These scripts let you run OCR on any dataset without needing your own GPU. Simply point the script at your input dataset, and it will:

- Process all images in a dataset column using many different open OCR models  
- Add OCR results as a new markdown column to the dataset  
- Push the updated dataset with OCR results to the Hub

For example, to run OCR on 100 images:

```bash  
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
  your-input-dataset your-output-dataset \
  --max-samples 100
```

The scripts handle all the vLLM configuration and batching automatically, making batch OCR accessible without infrastructure setup.

### Going Beyond OCR

If you are interested in document AI, not just OCR, here are some of our recommendations. 

#### Visual Document Retrievers
Visual document retrieval is to retrieve the most relevant top-k documents when given a text query. If you have previously worked with retriever models, the difference is that you search directly on a stack of PDFs. Aside from using them standalone, you can also build multimodal RAG pipelines by combining them with a vision language model (find how to do so [here](https://huggingface.co/merve/smol-vision/blob/main/ColPali\_%2B\_Qwen2\_VL.ipynb)). You can find [all of them on Hugging Face Hub](https://huggingface.co/models?pipeline\_tag=visual-document-retrieval\&sort=trending).

There are two types of visual document retrievers, single-vector and multi-vector models. Single-vector models are more memory efficient and less performant; meanwhile, multi-vector models are more memory hungry and more performant. Most of these models often come with vLLM and transformers integrations, so you can index documents using them and then do a search easily using a vector DB.

#### Using Vision Language Models for Document Question Answering
If you have a task at hand that only requires answering questions based on documents, you can use some of the vision language models that had document tasks in their training tasks. We’ve observed users trying to convert documents into text and passing the output to LLMs, but if your document has a complex layout, and your converted document outputs charts and so on in HTML, or images are captioned incorrectly, the LLM will miss out. Instead, feed your document and query to one of the advanced vision language models like [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe) not to miss out on any context. 

## Wrapping up

In this blog post, we wanted to give you an overview of how to pick your OCR model, existing cutting-edge models and capabilities, and the tools to get you started with OCR.   
If you want to learn more about OCR and vision language models, we encourage you to read the resources below. 

- [Vision Language Models Explained](https://huggingface.co/blog/vlms)  
- [Vision Language Models 2025 Update](https://huggingface.co/blog/vlms-2025)  
- [PP-OCR-v5](https://huggingface.co/blog/baidu/ppocrv5)  
- [SOTA OCR on-device with Core ML and dots.ocr](https://huggingface.co/blog/dots-ocr-ne)

