---
title: "Tokenization in Transformers v5: Simpler, Clearer, and More Modular"
thumbnail: /blog/assets/tokenizers/thumbnail.png
authors:
- user: itazap
- user: ariG23498
- user: ArthurZ
- user: sergiopaniego
- user: merve
- user: pcuenq
---

# Tokenization in Transformers v5: Simpler, Clearer, and More Modular

![thumbnail](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tokenizers/thumbnail.png)

[Transformers v5](https://huggingface.co/blog/transformers-v5) redesigns how tokenizers work. The [big tokenizers reformat](https://github.com/huggingface/transformers/pull/40936/files) separates tokenizer design from trained vocabulary (much like how PyTorch separates neural network architecture from learned weights). The result is tokenizers you can *inspect*, *customize*, and *train* from scratch with far less friction.

> [!NOTE]
> TL;DR: This blog explains how tokenization works in Transformers and why v5 is a major redesign, with clearer internals, a clean class hierarchy, and a single fast backend. It’s a practical guide for anyone who wants to understand, customize, or train model-specific tokenizers instead of treating them as black boxes.

## Table of Contents

- [What is Tokenization?](#what-is-tokenization)
- [The Tokenization Pipeline](#the-tokenization-pipeline)
- [Tokenization Algorithms](#tokenization-algorithms)
- [Accessing `tokenizers` through `transformers`](#accessing-tokenizers-through-transformers)
- [The Tokenizer Class Hierarchy in `transformers`](#the-tokenizer-class-hierarchy-in-transformers)
- [`AutoTokenizer` Automatically Selects the Correct Tokenizer Class](#autotokenizer-automatically-selects-the-correct-tokenizer-class)
- [v5 Separates Tokenizer Architecture from Trained Vocab](#v5-separates-tokenizer-architecture-from-trained-vocab)
- [Summary](#summary)

> [!TIP]
> For experts: If you're already familiar with the concepts and want to understand the changes in v5, go to [v5 Separates Tokenizer Architecture from Trained Vocab](#v5-separates-tokenizer-architecture-from-trained-vocab)

Before diving into the changes, let's quickly cover what tokenization does and how the pieces fit together.

## What is tokenization?

<iframe
	src="https://xenova-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

Language models don't read raw text. They consume sequences of integers usually called **token IDs or input IDs**. Tokenization is the process of converting raw text into these token IDs.

Tokenization is a broad concept used across natural language processing and text processing generally. This post focuses specifically on tokenization for Large Language Models (LLMs) using the [`transformers`](https://github.com/huggingface/transformers) and [`tokenizers`](https://github.com/huggingface/tokenizers) libraries.

```py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

text = "Hello world"
tokens = tokenizer(text)

print(tokens["input_ids"])
# [9906, 1917]

print(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
# ['Hello', 'Ġworld']
```

> [!NOTE]
>  `Ġworld` (above) is a single token that represents the character sequence " world" (with the space).

A **token** is the smallest string unit the model sees. It can be a character, word, or subword chunk like "play" or "##ing" ("##" is a pattern, don't worry if you don't completely understand it now 🤗). The **vocabulary** maps each unique token to the token ID.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
print(tokenizer.vocab)

# {'ÎĹÎľ': 106502, 'ĠPeel': 89694, '.languages': 91078, ...}
```

A good tokenizer *compresses* text into the smallest amount of tokens. Fewer tokens means more usable context without increasing model size. Training a tokenizer boils down to finding the best compression rules for your datasets. For example, if you work on Chinese corpus you can sometimes find [very nice surprises 😉](https://x.com/suchenzang/status/1697862650053660721).

## The tokenization pipeline

Tokenization happens in stages. Each stage transforms text before passing it to the next:

| Stage | Purpose | Example |
| :---: | :---: | :---: |
| **Normalizer** | Standardizes text (lowercasing, unicode normalization, whitespace cleanup) | `"HELLO World"` → `"hello world"` |
| **Pre-tokenizer** | Splits text into preliminary chunks | `"hello world"` → `["hello", " world"]` |
| **Model** | Applies the tokenization algorithm (BPE, Unigram, etc.) | `["hello", " world"]` → `[9906, 1917]` |
| **Post-processor** | Adds special tokens (BOS, EOS, padding) | `[9906, 1917]` → `[1, 9906, 1917, 2]` |
| **Decoder** | Converts token IDs back to text | `[9906, 1917]` → `"hello world"` |

Each component is *independent*. You can swap [normalizers](https://huggingface.co/docs/tokenizers/en/api/normalizers) or change the [algorithm](https://huggingface.co/docs/tokenizers/en/api/models) without rewriting everything else.

> [!NOTE]
> You can access the rust based tokenizer through `_tokenizer`. We go in more depth about it in [this section](#tokenizersbackend-wraps-the-tokenizers-library)

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

print(f"{tokenizer._tokenizer.normalizer=}")
# Replace(...)

print(f"{tokenizer._tokenizer.pre_tokenizer=}")
# Split(...)

print(f"{tokenizer._tokenizer.model=}")
# BPE(...)

print(f"{tokenizer._tokenizer.post_processor=}")
# TemplateProcessing(...)

print(f"{tokenizer._tokenizer.decoder=}")
# Sequence(decoders=[Replace(...), ByteFallback(), Fuse()])
```

## Tokenization algorithms

The following algorithms dominate modern language model tokenizers:

1. **Byte Pair Encoding (BPE)** iteratively merges the most frequent character pairs. This algorithm is deterministic and widely used. (Read more about [BPE](https://huggingface.co/learn/llm-course/en/chapter6/5))

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
print(tokenizer._tokenizer.model)

# BPE(...)
```

2. **Unigram** takes a probabilistic approach, selecting the most likely segmentation from a large initial vocabulary. This is more flexible than the strict BPE. (Read more about [Unigram](https://huggingface.co/learn/llm-course/en/chapter6/7))

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
print(tokenizer._tokenizer.model)

# Unigram(...)
```

3. **WordPiece** resembles BPE but uses different merge criteria based on likelihood. (Read more about [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6))

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer._tokenizer.model)

# WordPiece(...)
```

## Accessing tokenizers through transformers

The [`tokenizers`](https://github.com/huggingface/tokenizers) library is a Rust-based tokenization engine. It is fast, efficient, and completely language model agnostic. The library handles the mechanics of converting text into token IDs and back. The `tokenizers` library is a general-purpose tool that implements the tokenization algorithms, but does not implement the conventions that connect those algorithms to specific language models.

Consider what happens when you use `tokenizers` directly with the [`SmolLM3-3B`](http://hf.co/HuggingFaceTB/SmolLM3-3B) model:

```py
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
text = "Hello world"
encodings = tokenizer.encode(text)

print(encodings.ids)
# [9906, 1917]
print(encodings.tokens)
# ['Hello', 'Ġworld']
```

The output is raw tokenization. You get token IDs and the string pieces they correspond to. Nothing more.

Now consider what's missing. The `SmolLM3-3B` is a *conversational model*. When you interact with it, you typically structure your input as a conversation with roles like "user" and "assistant". The language model expects special formatting tokens to indicate these roles. The raw `tokenizers` library has no concept of any of this.

### How do you bridge the gap between raw tokenization and model requirements?

The `transformers` library bridges this gap. The library is primarily known as a model definition library, but it also provides a tokenizer abstraction layer that wraps the raw `tokenizers` backend and adds model-aware functionality.

Here's the same tokenization with the `transformers` wrapper:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

# Format a conversation using the model's chat template
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(text)

# <|im_start|>system
# ...
# <|im_start|>user
# Give me a brief explanation of gravity in simple terms.<|im_end|>
# <|im_start|>assistant

model_inputs = tokenizer([text], return_tensors="pt")
```

Notice how the special tokens like `<|im_start>` and `<|im_end>` are applied to the prompt before tokenizing. This is useful for the model to learn where a new sequence starts and ends.

The `transformers` tokenizer adds everything the raw library lacks:

* **Chat template application.** The `apply_chat_template` method formats conversations according to the model's expected format, inserting the correct special tokens and delimiters.  
* **Automatic special token insertion.** Beginning-of-sequence and end-of-sequence tokens are added where the model expects them.  
* **Truncation to context length.** You can specify `truncation=True` and the tokenizer will respect the model's maximum sequence length.  
* **Batch encoding with padding.** Multiple inputs can be padded to the same length with the correct padding token and direction.  
* **Return format options.** You can request PyTorch tensors (`return_tensors="pt"`), NumPy arrays and others.

> [!NOTE]
> `transformers` implements the tokenization API that is most commonly used in the entire ML community (`encode`, `decode`, `convert_tokens_to_ids`, etc.)

## The tokenizer class hierarchy in transformers

The `transformers` library organizes tokenizers into a class hierarchy. At the top sits a base class that defines the common interface. Below it, backend classes handle the actual tokenization using different engines. At the bottom, model-specific classes configure the backends for particular models.

| ![class hierarchy](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tokenizers/hierarchy.png) |
| :--: |
| The class hierarchy for tokenizers inside transformers |

### `PreTrainedTokenizerBase` defines the common interface for all tokenizers

[`PreTrainedTokenizerBase`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/tokenization_utils_base.py#L964C7) is the abstract base class for all tokenizers in `transformers`. It defines the interface that every tokenizer must implement.

The base class handles functionality that doesn't depend on the tokenization backend:

* **Special token properties.** Properties like `bos_token`, `eos_token`, `pad_token`, and `unk_token` are defined here. These properties provide access to the special tokens that models use to mark sequence boundaries and handle unknown inputs.  
* **Encoding interface.** The `__call__` method, `encode`, and `encode_plus` methods are defined here. These methods accept text input and return token IDs along with attention masks and other metadata.  
* **Decoding interface.** The `decode` and `batch_decode` methods convert token IDs back to text.  
* **Serialization.** The `save_pretrained` and `from_pretrained` methods handle downloading the correct files, reading information, saving tokenizers to disk etc.  
* **Chat template support.** The `apply_chat_template` method lives here, formatting conversations according to Jinja templates stored in the tokenizer configuration.

Every tokenizer in `transformers` ultimately inherits from `PreTrainedTokenizerBase`. The base class ensures consistent behavior across all tokenizers, regardless of which backend they use for the actual tokenization.

### `TokenizersBackend` wraps the `tokenizers` library

[`TokenizersBackend`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/tokenization_utils_tokenizers.py#L80C7) is the primary backend class for most modern tokenizers. It inherits from `PreTrainedTokenizerBase` and wraps the Rust-based `tokenizers` library.

The class stores the Rust tokenizer object internally:

```py
class TokenizersBackend(PreTrainedTokenizerBase):
    def __init__(self, tokenizer_object, ...):
        self._tokenizer = tokenizer_object  # The Rust tokenizer
        ...
```

When you call encoding methods on a `TokenizersBackend` tokenizer, the class delegates the actual tokenization to the Rust backend:

```py
def _batch_encode_plus(self, batch_text_or_text_pairs, ...):
    encodings = self._tokenizer.encode_batch(batch_text_or_text_pairs, ...)
    ...
```

The Rust backend performs computationally intensive work, while the Python wrapper adds the model-aware features on top.

Many model-specific tokenizers inherit from `TokenizersBackend`, examples include:

* `LlamaTokenizer`  
* `GemmaTokenizer`

These model-specific classes configure the backend with the correct vocabulary, merge rules, special tokens, and normalization settings for their respective models.

### `PythonBackend` provides a pure-Python mixin

[`PythonBackend`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/tokenization_python.py#L400) inherits from `PreTrainedTokenizerBase` and implements tokenization in pure Python. The class is aliased as [`PreTrainedTokenizer`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/tokenization_python.py#L1400C1).

The pure-Python backend exists for several reasons:

* **Custom tokenization logic.** Some models require tokenization behavior that doesn't fit the standard `tokenizers` pipeline.  
* **Legacy compatibility.** Older model implementations may rely on Python-specific behavior.

> [!NOTE]
> The Python backend is slower than the Rust backend. For most use cases, the Rust-backed `TokenizersBackend` is preferred.

Model-specific tokenizers that inherit from `PythonBackend` (or its alias `PreTrainedTokenizer`) include some older or specialized models, like:

* `CTRLTokenizer`  
* `CanineTokenizer`

### `SentencePieceBackend` handles SentencePiece models

[`SentencePieceBackend`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/tokenization_utils_sentencepiece.py#L46) inherits from `PythonBackend` and provides integration with Google's [SentencePiece](https://github.com/google/sentencepiece) library. SentencePiece is a standalone tokenization library that many models use, particularly those trained by Google.

The backend wraps a SentencePiece processor:

```py
class SentencePieceBackend(PythonBackend):
    def __init__(self, vocab_file, ...):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        ...
```

Models that use SentencePiece tokenization inherit from this backend. Examples include:

* `SiglipTokenizer`  
* `BartphoTokenizer`

The SentencePiece backend inherits from `PythonBackend` rather than directly from `PreTrainedTokenizerBase` because it shares much of the same interface and padding/truncation logic.

## AutoTokenizer automatically selects the correct tokenizer class

[`AutoTokenizer`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/models/auto/tokenization_auto.py#L531) is the recommended entry point for loading tokenizers. It automatically determines which tokenizer class to use for a given model and returns an instance of that class.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Behind the scenes, `AutoTokenizer` performs these steps:

1. **Download the tokenizer configuration.** The `from_pretrained` method fetches `tokenizer_config.json` from the Hub (or from a local directory).  
2. **Identify the model type.** The configuration contains metadata that [identifies the model type](https://huggingface.co/openai-community/gpt2/blob/main/config.json#L12) (e.g., "gpt2", "llama", "bert").  
3. **Look up the tokenizer class.** `AutoTokenizer` maintains a mapping called [`TOKENIZER_MAPPING_NAMES`](https://github.com/huggingface/transformers/blob/7f52a2a4ea8ab49b7f069df7fac58a5b280d4919/src/transformers/models/auto/tokenization_auto.py#L64) that maps model types to tokenizer class names:

```py
TOKENIZER_MAPPING_NAMES = {
    "gpt2": "GPT2Tokenizer",
    "llama": "LlamaTokenizer",
    "bert": "BertTokenizer",
    ...
}
```

4. **Instantiate the correct class.** `AutoTokenizer` imports the appropriate tokenizer class and calls its `from_pretrained` method.  
5. **Return the configured tokenizer.** You receive a fully configured, model-specific tokenizer ready for use.

> [!NOTE]
> The benefit of `AutoTokenizer` is that you don't need to know which tokenizer class a model uses. Whether a model uses `LlamaTokenizer`, `GPT2Tokenizer`, or `BertTokenizer`, the same `AutoTokenizer.from_pretrained("model-name")` call works.

The tokenizer system in `transformers` forms a layered architecture:

| Layer | Component | Responsibility |
| :---: | :---: | :---: |
| Entry Point | `AutoTokenizer` | Automatically selects and instantiates the correct tokenizer class |
| Model-Specific | `LlamaTokenizer`, `GPT2Tokenizer`, etc. | Configures the backend with model-specific architecture of normalizer, pre tokenizer, etc, special tokens, and settings |
| Backend | `TokenizersBackend`, `PythonBackend`, `SentencePieceBackend` | Implements the actual tokenization using a specific engine |
| Base | `PreTrainedTokenizerBase` | Defines the common interface and shared functionality |
| Engine | `tokenizers` (Rust), SentencePiece, Pure Python | Performs raw tokenization |

## v5 Separates Tokenizer Architecture from Trained Vocab

The most significant change in Transformers v5 is a philosophical shift in how tokenizers are defined. **Tokenizers now work like PyTorch's `nn.Module`**: you define the architecture first, then fill it with learned parameters.

### The problem with v4: tokenizers were opaque and tightly coupled

In v4, tokenizers were black boxes tied to pretrained checkpoint files. If you loaded `LlamaTokenizerFast`, you couldn't easily answer basic questions about it:

* Is it BPE or Unigram?  
* How does it normalize text?  
* What pre-tokenization strategy does it use?  
* What are the special tokens and their positions?

The `__init__` method gave no clues. You had to dig through serialized files or external documentation to understand what the tokenizer actually did.

| ![v4 llama](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tokenizers/v4-llama.png) |
| :--: |
| `LlamaTokenizerFast` as seen in v4 `transformers` |

v4 also maintained two parallel implementations for every model:

1. a "slow" Python tokenizer (`LlamaTokenizer` inheriting from `PreTrainedTokenizer`) and  
2. a "fast" Rust-backed tokenizer (`LlamaTokenizerFast` inheriting from `PreTrainedTokenizerFast`).

This meant:

* **Two files per model** (e.g., `tokenization_llama.py` and `tokenization_llama_fast.py`)  
* **Code duplication** across hundreds of models  
* **Behavioral discrepancies** between slow and fast versions, leading to subtle bugs  
* **A growing test suite** dedicated to verifying that slow and fast tokenizers produced identical outputs  
* **User confusion** about which tokenizer to use and when

Worst of all, you couldn't create an empty tokenizer architecture. If you wanted to train a LLaMA-style tokenizer on your own data, there was no clean way to instantiate a "blank" LLaMA tokenizer and fill it with your vocabulary and merges. Tokenizers existed only as loaded checkpoints, not as configurable templates.

### The v5 solution: architecture and parameters are now separate

v5 treats tokenizer architecture (normalizer, pre-tokenizer, model type, post-processor, decoder) as distinct from trained parameters (vocabulary, merges). This mirrors how PyTorch separates model architecture from learned weights.

**With `nn.Module`, you define layers first:**

```py
from torch import nn

model = nn.Sequential(
    nn.Embedding(vocab_size, embed_dim),
    nn.Linear(embed_dim, hidden_dim),
)
# Architecture defined; weights initialized randomly or loaded later
```

**V5 tokenizers follow the same pattern:**

```py
from transformers import LlamaTokenizer

# Instantiate the architecture
tokenizer = LlamaTokenizer()

# Train on your own data to fill in vocab and merges
tokenizer.train(files=["my_corpus.txt"])
```

The tokenizer class now explicitly declares its structure. Looking at `LlamaTokenizer` in v5, you can immediately see:

* [It uses **BPE**](https://github.com/huggingface/transformers/blob/0a8465420eecbac1c6d7dd9f45c08dd96b8c5027/src/transformers/models/llama/tokenization_llama.py#L92) as its tokenization model
* It may add a **prefix space** before text  
* Its special tokens (`unk`, `bos`, `eos`) sit at specific vocabulary positions  
* [It does **not normalize**](https://github.com/huggingface/transformers/blob/0a8465420eecbac1c6d7dd9f45c08dd96b8c5027/src/transformers/models/llama/tokenization_llama.py#L121) input text  
* [Its decoder](https://github.com/huggingface/transformers/blob/0a8465420eecbac1c6d7dd9f45c08dd96b8c5027/src/transformers/models/llama/tokenization_llama.py#L122) replaces the metaspace character `▁` with spaces

| ![v5 llama](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tokenizers/v5-llama.png) |
| :--: |
| `LlamaTokenizer` as seen in v5 `transformers` |

This transparency was impossible in v4, where the same information was buried in serialized files.

### One file, one backend, one recommended path

v5 consolidates the two-file system *into a single file per model*. `LlamaTokenizer` now inherits from `TokenizersBackend`, which wraps the Rust-based tokenizer that was previously exposed as the “fast” implementation and is now the default.

The former “slow” Python implementation lives explicitly behind `PythonBackend`, and `SentencePieceBackend` remains for models that require it, but **Rust-backed tokenization is the preferred default**.

This change eliminates:

* Duplicate code across slow/fast implementations  
* The confusing `Tokenizer` vs `TokenizerFast` naming convention  
* Test suites dedicated to checking slow-fast parity

Users now have one clear entry point. Advanced users who need to customize can still access lower-level components, but the library no longer forces everyone to navigate two parallel implementations.

### You can now train model specific tokenizers from scratch

Suppose you want a tokenizer that behaves exactly like LLaMA's – same normalization, same pre-tokenization, same BPE model type – but trained on a domain-specific corpus (medical text, legal documents, a new language). In v4, this required manually reconstructing the tokenizer pipeline from low-level `tokenizers` library primitives. In v5, you can instantiate the architecture directly and call `train`:

```py
from transformers import LlamaTokenizer
from datasets import load_dataset

# Initialize blank tokenizer
tokenizer = LlamaTokenizer()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def get_training_corpus():
    batch = 1000
    for i in range(0, len(dataset), batch):
        yield dataset[i : i + batch]["text"]

trained_tokenizer = tokenizer.train_new_from_iterator(
    text_iterator=get_training_corpus(),
    vocab_size=32000,
    length=len(dataset),
    show_progress=True,
)

trained_tokenizer.push_to_hub("my_custom_tokenizer")

tokenizer = LlamaTokenizer.from_pretrained("my_custom_tokenizer")
```

The resulting tokenizer will have your custom vocabulary and merge rules, but will process text identically to how a standard LLaMA tokenizer would with the same whitespace handling, same special token conventions, same decoding behavior.

| Aspect | V4 | V5 |
| :---: | :---: | :---: |
| Files per model | Two (`tokenization_X.py`, `tokenization_X_fast.py`) | One (`tokenization_X.py`) |
| Default backend | Split between Python and Rust | Rust (`TokenizersBackend`) preferred |
| Architecture visibility | Hidden in serialized files | Explicit in class definition |
| Training from scratch | Required manual pipeline construction | `tokenizer.train(files=[...])` |
| Component inspection | Difficult, undocumented | Direct properties (`tokenizer.normalizer`, etc.) |
| Parent classes | `PreTrainedTokenizer`, `PreTrainedTokenizerFast` | `TokenizersBackend` (or `SentencePieceBackend`, `PythonBackend`) |

The shift from "tokenizers as loaded checkpoints" to "tokenizers as configurable architectures" makes the library more modular, more transparent, and more aligned with how practitioners think about building ML systems.

## Summary

Transformers v5 brings three improvements to tokenization:

1. **One file per model** instead of separate slow/fast implementations  
2. **Visible architecture** so you can inspect normalizers, pre-tokenizers, and decoders  
3. **Trainable templates** that let you create custom tokenizers matching any model's design

The wrapper layer between `tokenizers` and Transformers remains essential. It adds model awareness, context lengths, chat templates, special tokens, that raw tokenization doesn't provide. V5 just makes that layer clearer and more customizable.

If you are looking to learn more about tokenization here are some resources:
- [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=ZAfCjZjpyPHsnyfF)
- [Gotchas in Tokenizer Behavior Every Developer Should Know](https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior)
- [Chat Templates](https://huggingface.co/blog/chat-templates)
- [A list of resoruces we have gathered from the community!](https://x.com/ariG23498/status/1999058214906888237)