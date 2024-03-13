---
title: "Fast Embeddings with 🤗 Optimum Intel and fastRAG"
thumbnail: /blog/assets/optimum_intel/thumbnail.png
authors:
- user: peterizsak
  guest: true
- user: mber
  guest: true
- user: danf
  guest: true
- user: echarlaix
- user: mfuntowicz
- user: moshew
  guest: true
---


# Fast Embeddings with 🤗 Optimum Intel and fastRAG

## Introduction


Embedding models play a vital role in information retrieval by encoding textual data into vector representations that possess both semantic and contextual meaning. These models effectively capture the intricate relationships between words and documents, thereby enabling more accurate and efficient information searching. Moreover, by utilizing dense vector representations, embedding models allow for straightforward similarity calculations such as cosine similarity.

Embedding models provide us with semantic (dense) encoding. Semantic and sparse information retrieval are two distinct approaches to retrieving information from an index. Sparse retrieval involves searching through a large collection by matching terms or phrases, whereas semantic retrieval focuses on understanding the context and meaning behind the words in a document. Semantic retrieval can provide more accurate results by considering the overall meaning of a sentence or paragraph, whereas sparse retrieval may miss important information if it only focuses on specific keywords. However, dense retrieval is far more compute-heavy than traditional lexical-based matching algorithms (e.g., BM25).

Embedding models are primarily utilized during the retrieval phase of retrieval augmented generation (RAG) use cases. This involves encoding documents when constructing a semantic or dense index, encoding queries in real-time for querying a dense index, and re-ordering (reranking) a set of documents to find the best similarity between the query vector and the retrieved documents.
Embedding models are useful for many applications such as retrieval, reranking, clustering, classification, etc. The [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a widely adapted benchmark for evaluating embedding models in different tasks and on multiple datasets. For evaluating embedding models for RAG, the “reranking” and “retrieval” tasks of MTEB are good indicators for embedding model performance.

The embedding models research community has witnessed significant advancements in recent years, leading to substantial enhancements in the quality of open-source embedding models, such as the BGE, GTE, and E5 family of models. As a result, these models have become increasingly competitive with proprietary services that offer similar capabilities such as Open AI and Cohere. These models are usually lightweight (100-350M parameters) compared to LLMs and use an encoder architecture, which makes them ideal candidates for optimization and utilization on CPU backends running RAG pipelines.


 
## Optimize your model with Optimum Intel

[Optimum Intel](https://github.com/huggingface/optimum-intel/) is an open-source library that accelerates end-to-end pipelines built with the Huggingface Transformers library on Intel Hardware. The library includes several techniques to accelerate models such as low-bit quantization, model weight pruning, distillation, and an accelerated runtime.

The runtime and optimizations included in Optimum Intel take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs to accelerate models.

Optimizing pre-trained models can be done easily with Optimum Intel many simple examples can be found [here](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc).




## Optimizing BGE-embedders with Optimum Intel


In our evaluation, we focus on a family of embedding models (denoted as BGE) created by researchers at the [Beijing Academy of Artificial Intelligence](https://arxiv.org/pdf/2309.07597.pdf), as their models show competitive results on the widely adopted [MTEB leaderboard](https://github.com/embeddings-benchmark/mteb).

### BGE Embedders Technical Details

Bi-encoder models are Transformer-based encoders that were trained to minimize a similarity metric, such as cosine-similarity, between two semantically similar texts as vectors. For example, popular embedders use a BERT model as a base pre-trained model and fine-tune it for embedding documents. The vector representing the encoded text can be anything from the output layer, for example, it could be the [CLS] token vector or a mean of all document tokens.

Unlike more complex embedding architectures, bi-encoders encode only single documents, thus they lack contextual interaction between encoded entities such as query-document and document-document. However, state-of-the-art bi-encoder embedders present competitive performance and are extremely fast due to its simple architecture.

We focus on 3 BGE models: [small](https://huggingface.co/BAAI/bge-small-en-v1.5), [base](https://huggingface.co/BAAI/bge-base-en-v1.5), and [large](https://huggingface.co/BAAI/bge-large-en-v1.5) consisting of 45M, 110M, and 355M parameters encoding to 384/768/1024 sized embedding vectors, respectively.

We note that the optimization process we showcase below is generic and can be applied to other embedding models (including bi-encoder, cross-encoder, and such).


### Optimization by Quantization
We present a step-by-step guide for enhancing the performance of embedders, focusing on reducing latency (with a batch size of 1) and increasing throughput (measured in documents encoded per second). This recipe utilizes optimum-intel and [Intel Neural Compressor](https://github.com/intel/neural-compressor) to quantize the model and use [IPEX](https://intel.github.io/intel-extension-for-pytorch/#introduction) for optimized runtime on Intel-based hardware.

### Installing the Packages

To install optimum-intel and IPEX run the following command:

```bash
pip install -U optimum[neural-compressor] intel-extension-for-transformers
```

### Post-training Static Quantization

Post-training static quantization requires a calibration set to determine the dynamic range of weights and activations. The calibration is done by running a representative set through the model, collecting statistics, and then quantizing the model based on the gathered info to minimize accuracy loss.

The following snippet shows a sample code for quantization:


```python
def quantize(model_name: str, output_path: str, calibration_set: Dataset):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

    vectorized_ds = calibration_set.map(preprocess_function, num_proc=10)
    vectorized_ds = vectorized_ds.remove_columns(["text"])

    quantizer = INCQuantizer.from_pretrained(model)
    quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", domain="nlp")
    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=vectorized_ds,
        save_directory=output_path,
        batch_size=1,
    )
    tokenizer.save_pretrained(output_path)
```

In our calibration process we use a subset of the [qasper](https://huggingface.co/datasets/allenai/qasper) dataset.


## Usage


Once the model is quantized it is easy to integrate it into an inference pipeline.

Loading a quantized model by doing the following:


```python
from optimum.intel import IPEXModel

model = IPEXModel.from_pretrained("Intel/bge-small-en-v1.5-rag-int8-static")
```

And encode sentences using the [Transformers](https://github.com/huggingface/transformers) library :

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Intel/bge-small-en-v1.5-rag-int8-static")
inputs = tokenizer(sentences, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # get the [CLS] token
    embeddings = outputs[0][:, 0]
```

We provide additional and important details on how to configure the CPU-backend setup in the evaluation section below (correct machine setup).

## MTEB Evaluation

Quantizing the models weights to a lower precision introduces accuracy loss as we lose precision moving from float32 weights to int8. Therefore, we aim to validate the accuracy of the optimized models by comparing them to the original models with two MTEB’s tasks:
1. **Retrieval** - where a corpus is encoded and ranked lists are created by searching the index given a query
2. **Reranking** – reranking the retrieval’s results for better relevancy given a query

The table below shows the average accuracy (on multiple datasets)  of each task type (MAP for Reranking, NDCG@10 for Retrieval), where fp8 is our quantized model and fp32 is the original model (results taken from the official MTEB leaderboard). We can see that the optimization process maintained accuracy as close as 0.24% loss in the Reranking task and at most 1.55% in the Retrieval tasks, proving that the optimization process marginally changed the embedders' capabilities in RAG related tasks.



---
|           |  int8  |  fp32  |  diff  |
| --------- | ------ | ------ | ------ |

| BGE-small | 0.5823 | 0.5836 | -0.22% |
| BGE-base  | 0.5887 | 0.5886 | +0.02% |
| BGE-large | 0.5988 | 0.6003 | -0.24% |

Table 1: Reranking

---


---
|           |  int8  |  fp32  |  diff  |
| --------- | ------ | ------ | ------ |

| BGE-small | 0.514  | 0.5168 | -0.58% |
| BGE-base  | 0.524  | 0.5325 | -1.55% |
| BGE-large | 0.535  | 0.5429 | -1.53% |

Table 2: Retrieval

---



## Speed and Latency Evaluations


We compare the performance of our models with two other common methods of usage of models:

1. Vanilla usage with PyTorch and Huggingface’s Transformers library with bf16.
2. Intel extension for PyTorch (IPEX) runtime with bf16, AMP, and torchscript.

Experimental setup notes:

- The vanilla model evaluation used all cores in the system (no additional environment flags).
- IPEX/Optimum setups were run with ipexrun, on 1 socket, and cores ranging from 22-56.
- Hardware (CPU):  4th gen Intel Xeon 8480+ with 2 sockets, 56 cores per socket.
- TCMalloc was installed and defined as an environment variable.

### How did we run the evaluation?

1. Baseline PyTorch and Hugging Face:


```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

@torch.inference_mode()
def encode_text():
    outputs = model(inputs)

with torch.cpu.amp.autocast(dtype=torch.bfloat16):
    encode_text()
```

2. IPEX torchscript and bf16:


```python
import torch
from transformers import AutoModel
import intel_extension_for_pytorch as ipex


model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
model = ipex.optimize(model, dtype=torch.bfloat16)

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
d = torch.randint(vocab_size, size=[batch_size, seq_length])
model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
model = torch.jit.freeze(model)

@torch.inference_mode()
def encode_text():
    outputs = model(inputs)

with torch.cpu.amp.autocast(dtype=torch.bfloat16):
    encode_text()
```


3. Optimum-intel with IPEX and int8 model:



```python
import torch
from optimum.intel import IPEXModel

model = IPEXModel.from_pretrained("Intel/bge-small-en-v1.5-rag-int8-static")

@torch.inference_mode()
def encode_text():
    outputs = model(inputs)

encode_text()
```




### Latency performance

In this evaluation, we aim to measure how fast the models respond. This is an example use case for encoding queries in RAG pipelines.

The batch size is set to 1 and we measure document different lengths.

We can see that the quantized model has the best latency overall, under 10 ms for the small and base models and <20 ms for the large model. Compared to the vanilla model, the quantized model shows x6 - x12 speedup in latency.




<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/178_intel_ipex_quantization/latency.png" alt="IN8 AG" style="width: 70%; height: auto;"><br>
<em>Figure 1. Latency for BGE models.</em>
</p>




### Throughput Performance

In our throughput evaluation, we aim to search for the peak performance of the model in terms of encoded documents per second. This experiment highlights the capability of the hardware and model when encoding large quantities of text (when creating an index).

We set the encoded text lengths to 256 tokens as it serves as a good estimate of an average passage in RAG pipelines. We run the evaluation with batch sizes of 4, 8, 16, 32, 64, 128, 256.

Results show that the quantized models running with the optimized backend have the best throughput over all other options and reach peak throughput at batch size 128. Overall, for all model sizes, the quantized model has x4-x5 higher throughput than the vanilla model at the peak performance batch size.




<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/178_intel_ipex_quantization/throughput_small.png" alt="IN8 AG" style="width: 70%; height: auto;"><br>
<em>Figure 2. Throughput for BGE small.</em>
</p>



<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/178_intel_ipex_quantization/throughput_base.png" alt="IN8 AG" style="width: 70%; height: auto;"><br>
<em>Figure 2. Throughput for BGE base.</em>
</p>



<p align="center">
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/178_intel_ipex_quantization/throughput_large.png" alt="IN8 AG" style="width: 70%; height: auto;"><br>
<em>Figure 3. Throughput for BGE large.</em>
</p>


