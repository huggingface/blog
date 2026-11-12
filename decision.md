---
title: "Decision Models in Hugging Face"
thumbnail: /blog/assets/decision/thumbnail.png
authors:
  - user: ariG23498
  - user: lysandre
  - user: pcuenq
  - user: tomaarsen
---

# Decision Models in Hugging Face

Large Language Models (LLMs) are exceptionally good at generating text. We use them to answer questions, write code, reason about problems, and so much more. *But not every problem requires generating text.*

Consider a customer support system receiving the following message:

"I was charged twice for my subscription. I need a refund today."

The system needs to determine which team should handle the request, how urgent it is, and whether the customer is requesting a refund.

We could ask an LLM to return a JSON object containing these answers.

```bash
llama cli -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0

> We have the following customer support ticket, "I was charged twice for my subscription. I need a refund today.", which team should handle the request, how urgent it is, and whether the customer is requesting a refund. Give the answers in JSON.

[Start thinking]
...
[end thinking]
```json
{
  "team_to_handle": "Billing",
  "urgency": "High",
  "request_refund": true
}
```

But what if the model could directly evaluate the available options and return a probability distribution over them, *without generating any text*?

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decision = decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Bugs and technical issues",
                    "sales": "Pricing and subscriptions"
                }
            },
        }
    }
)

# {
#   'department': {
#       'type': 'choice',
#       'choice': 'billing',
#       'confidence': 0.9974000453948975,
#       'probabilities': {
#         'billing': 0.9982666969299316,
#         'technical': 0.0010246571619063616,
#         'sales': 0.0007087149424478412
#       }
#     }
# }

```

This is the key idea behind **decision models**.

## What are decision models?

A decision model takes some input context (state), evaluates a question with a set of predefined possible answers (primitives), and returns a structured decision along with probabilities. Instead of asking a model to generate an answer token by token, we ask it to assign probabilities to the answers that our application already understands.

![state and primitives for the decision models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/decision/decision-models.png)

[TypeSafe AI introduced Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), describing it as a [*System One model*](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow). The name draws inspiration from the distinction between fast, intuitive thinking (System 1) and slower, deliberative thinking (System 2).

> The community has converged to the term **decision model**, because it describes the model's computational role rather than invoking a particular theory of cognition. In the rest of the post we will refer to the models as "decision models" instead of "system one models". ([A poll on twitter deciding what we should be calling such models](https://x.com/ariG23498/status/2113220513158009022))

For a decision model, the fundamental interface is:

```text
State + Questions + Allowed Answers → Decisions + Probabilities
```

## Why not use LLMs for everything?

We could simply prompt an LLM to return a JSON object containing our decisions. In fact, this is how many applications already use language models for classification, routing, and other structured tasks.

However, there are a few problems with this approach.

* LLMs are not guaranteed to follow the requested schema: Even with explicit instructions, a model might produce invalid JSON, introduce unexpected keys, or return an answer outside the allowed choices.
* Constrained decoding helps, but does not eliminate the computational overhead: Libraries such as [Outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar) can enforce grammars or JSON schemas by restricting which tokens the model is allowed to generate at each step. This ensures the output follows the supported structural constraints, but introduces additional work.
* Why generate an entire response: LLMs are designed for open-ended generation, but when our application only needs to choose between predefined answers, autoregressive decoding can be unnecessary work.

## How are decision models different from classification models?

Before understanding the interface of a decision model, it is useful to revisit two closely related approaches namely traditional classification and zero-shot classification.

### Traditional classification

Suppose we train a classifier that routes customer-support requests to three departments:

* Billing
* Technical support
* Sales

A conventional classification model learns a mapping from an input to a predefined set of labels. During training, the model learns to predict these labels from examples. A classification head typically produces one logit per class, and softmax turns those logits into a probability distribution.

The limitation here is that ***the output classes are baked into the model's classification head***.

If we introduce a fourth department, such as Legal, the existing classifier does not automatically acquire a corresponding output class. We may need to modify the head, collect training data, and **retrain** or adapt the model.

### Zero-shot classification

[Zero-shot classification](https://huggingface.co/tasks/zero-shot-classification) addresses an important part of this limitation. Instead of requiring the classifier to be trained specifically on every output category, we supply the candidate labels at inference time.

For example:

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

classifier(
    "My payment was deducted twice.",
    candidate_labels=["billing", "technical support", "sales"],
)

# {'sequence': 'My payment was deducted twice.',
#  'labels': ['billing', 'sales', 'technical support'],
#  'scores': [0.8466812372207642, 0.0962134450674057, 0.05710534751415253]}
```

Unlike a conventional fixed-label classifier, the categories can change from one request to another. Let’s add another department, such as “Legal”, and see the model outputs.

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

classifier(
    "My payment was deducted twice.",
    candidate_labels=["billing", "technical support", "sales", "legal"],
)

# {'sequence': 'My payment was deducted twice.',
#  'labels': ['billing', 'sales', 'technical support', 'legal'],
#  'scores': [0.8185835480690002, 0.09302053600549698, 0.05521026626229286, 0.03318570926785469]}
```

One common implementation uses a model trained for natural language inference (NLI). Each candidate label is converted into a hypothesis, and the model measures how strongly the input entails that hypothesis.

This already sounds quite similar to a decision model. And in many ways, it is.

### Where decision models differ

Decision models extend this idea beyond simply assigning an input to a category. A decision model can accept several named questions about the same state, with different answer types. The types can be choosing between alternatives (choice), scoring an ordered scale (score), or estimating the probability that a statement is true (noul).

More importantly, its interface is designed around the **probability distribution over possible decisions**.

Consider these three approaches:

| Property | Classification | Zero-shot classification | Decision model |
| --- | --- | --- | --- |
| Probability over choices | Yes | Yes | Yes |
| Labels defined at inference | No | Yes | Yes |
| Multiple typed questions | No | No | Yes |
| Ordered scoring and yes/no decisions | Task-specific | Possible with adaptation | Native primitives |

The advantage of decision models is not that they invented runtime-defined labels or probabilistic classification. It is that they organize these capabilities into a consistent interface, potentially backed by architectures and training procedures optimized specifically for decisions.

## The decision model interface

A decision model request has the following two components:

1. state
2. questions

### State

The state is the information the model should evaluate. It may be a string, a structured JSON object, or, other modalities for supported omni-modal models.

For example:

```json
{
  "customer_message": "I was charged twice. Please refund me today.",
  "subscription": "pro",
  "previous_tickets": 2
}
```

### Questions and their primitives

Questions describe what we want to know about the state. In the decision models interface, three fundamental primitives represent different forms of decisions:

1. **Choice:** Select one of several possible answers.
2. **Score:** Evaluate something on an ordered scale.
3. **Noul:** Answer a yes/no question probabilistically.

Let's go over them one by one.

#### Choice: A probability distribution over options

Choice is the most intuitive primitive and the closest to traditional classification.

Suppose we ask:

> Which department should handle this customer request?

The available options are Billing, Technical support, and Sales.

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decision = decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Bugs and technical issues",
                    "sales": "Pricing and subscriptions"
                }
            },
        }
    }
)

# {
#   'department': {
#     'type': 'choice',
#     'choice': 'billing',
#     'confidence': 0.9974000453948975,
#     'probabilities': {
#       'billing': 0.9982666969299316,
#       'technical': 0.0010246571619063616,
#       'sales': 0.0007087149424478412
#     }
#   }
# }
```

The model assigns a logit to each option. For logits \\(z_1,\ldots,z_K\\), (with \\(K\\) classes) we can obtain the probabilities using softmax:

$$
p_i=\frac{\exp(z_i)}{\sum_{j=1}^{K}\exp(z_j)}
$$

Here, \\(K\\) is the number of choices.

Consider this illustrative distribution:

| Choice | Probability |
| --- | --- |
| Billing | 0.998 |
| Technical support | 0.001 |
| Sales | 0.001 |

The probabilities sum to one:

$$
\sum_{i=1}^{K}p_i=1.
$$

The model chooses the option with the largest probability:

$$
\hat y=\arg\max_i p_i
$$

In this case, the choice is Billing, with the probability of 0.998, but the remaining probabilities are useful too. They tell us how the model distributes its belief among the alternatives.

#### Score: The expectation of an ordered distribution

Choice works when there are several distinct alternatives. But what if we want to measure something on an ordered scale?

Consider the question:

> How urgent is this customer request?

We define three levels:

* 0: Low urgency
* 1: Medium urgency
* 2: High urgency

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decision = decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is the request?",
                "criteria": ["low", "medium", "high"]
            },
        }
    }
)

# {
#   'urgency': {
#     'type': 'score',
#     'score': 1.7053890228271484,
#     'legend': {
#       '0': 'low',
#       '1': 'medium',
#       '2': 'high'
#     },
#     'confidence': 0.5580834252759814,
#     'probabilities': {
#       '0': 0.014908230863511562,
#       '1': 0.26479458808898926,
#       '2': 0.7202972173690796
#     }
#   }
# }
```

The model again produces a probability distribution:

| Level | Probability |
| :--: | :--: |
| 0 — Low | 0.01 |
| 1 — Medium | 0.26 |
| 2 — High | 0.72 |

Rather than returning only the most likely level, the Score primitive computes the **expected value of the level index**.

For a random variable \\(X\\) representing the level, the expectation is:

$$
\mathbb{E}[X]=\sum_{i=0}^{K-1}ip_i
$$

For our example:

$$
\mathbb{E}[X] = 0(0.01)+1(0.26)+2(0.72)=1.7
$$

The reported score is therefore `1.7`.

This does not mean that the model selected a fictional level halfway between medium and high. It means that the expectation of its probability distribution is 1.7.

#### Noul: A Bernoulli decision

The third primitive is Noul, which represents a binary question.

For example:

> Is the customer requesting a refund?

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decision = decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is the customer requesting a refund?"
            }
        }
    }
)

# {
#   'refund': {
#     'type': 'noul',
#     'noul': 0.9947046637535095
#   }
# }
```

Rather than defining a list of choices, we ask the model to return the probability that the answer is true. We can understand this using the Bernoulli distribution (noul comes from Bernoulli).

Let \\(Y\\) represent a binary outcome:

$$
Y\sim\operatorname{Bernoulli}(p)
$$

where

$$
P(Y=1)=p,\qquad P(Y=0)=1-p.
$$

If the model returns:

```json
{"type": "noul", "noul": 0.92}
```

it assigns a probability of `0.92` to Yes and `0.08` to No.

The model is not randomly sampling a binary answer here. It returns the probability parameter itself, leaving the application to decide how to act on it.

#### Confidence: How certain is the model?

So far, we have obtained probability distributions from the model. But the decision model interface also returns a field called `confidence` for Choice and Score.

The full distribution might contain several possible answers, and it is useful to summarize how concentrated the model's belief is. With `confidence` we achieve just that.

For a Choice with \\(K\\) options, the confidence is:

$$
\frac{p_{\max}-\frac{1}{K}}{1-\frac{1}{K}}
$$

where \\(p_{\max}\\) is the largest predicted probability.

The intuition here is:

* A uniform distribution has confidence 0. \\(p_{\max} = \frac{1}{K}\\). This means the model estimates that all choices are equally likely.
* A distribution concentrated entirely on one option has confidence 1. \\(p_{\max} = 1\\)
* The larger the winning probability relative to a uniform distribution, the higher the confidence.

In our previous three-choice example, Billing had probability 0.998:

$$
C=\frac{3(0.998)-1}{2}=0.997
$$

Notice that the winning probability is 0.998, but the reported confidence is 0.997. The confidence normalizes the largest probability relative to the number of available options. In contrast, the ordered Score example only had a confidence of 0.558.

## Decision models in the Hugging Face ecosystem

Decision models are a young and fast-moving space. Each one makes decisions differently under the hood (heads that score the whole schema jointly, separate readouts on top of LLM backbones, edge-sized and multimodal variants, etc). This variety is healthy, but it comes at a cost. If every model exposes its own interface, every tool around it needs model-specific adaptations.

Our goal at Hugging Face is to agree on a common **interface** without forcing a common **implementation**. A decision model should be free to work however it likes internally, as long as it accepts the same state and questions and returns the same typed decisions. Once that contract exists, the rest of the ecosystem can build on it once:

* inference engines like llama.cpp can serve any decision model through the same API
* the Hub can package and surface them consistently
* and training tools can target shared primitives instead of individual architectures

The rest of this section walks through each of these pieces.

### 1. Serving with Transformers

Transformers now supports the `DecisionPipeline`, `AutoModelForDecision`, and the `DecisionConfig` classes. These follow the same overall structure of the pipeline and model classes in `transformers` but are tailor made for decision models.

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
print(decide)

# DecisionPipeline: {'model': 'JevForDecision', 'dtype': 'bfloat16', 'device': 'cuda', 'input_modalities': 'text'}
```

One can also inspect the configuration and the model like so:

```python
from transformers import AutoModelForDecision, DecisionConfig

model_id = "autotrust/GEV-26B-Decide"
config = DecisionConfig.from_pretrained(model_id)
model = AutoModelForDecision.from_pretrained(model_id)
```

[Transformers also supports the `/v1/systemone` endpoint](https://huggingface.co/docs/transformers/main/en/serve-cli/serving#v1systemone).

We can start a local server using the serving dependencies:

```bash
pip install "transformers[serving]"
transformers serve
```

Using a compatible version of Transformers, we can then send a decision request:

```json
curl http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "autotrust/GEV-26B-Decide",
    "state": "I was charged twice. Please refund me today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Bugs and technical issues",
          "sales": "Pricing and subscriptions"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["low", "medium", "high"]
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?"
      }
    }
  }'
```

The response contains three named answers with the corresponding typed results and probabilities.

```bash
{
  "model":"autotrust/GEV-26B-Decide",
  "answers":{
    "department":{
      "type":"choice",
      "choice":"billing",
      "confidence":0.9974000453948975,
      "probabilities":{
        "billing":0.9982666969299316,
        "technical":0.0010246571619063616,
        "sales":0.0007087149424478412
        }
    },
    "urgency":{
      "type":"score",
      "score":1.7525686174631119,
      "legend":{"0":"low","1":"medium","2":"high"},
      "confidence":0.6288529792800546,
      "probabilities":{
        "0":0.014000383205711842,
        "1":0.2194305807352066,
        "2":0.7665690183639526
      }
    },
    "refund":{
      "type":"noul",
      "noul":0.9953224062919617
    }
  },
  "usage":{"input_tokens":148,"output_tokens":0}
}
```

> [!NOTE]
> An interesting detail is that the above APIs works with compatible autoregressive language models too. Transformers constructs the decision prompts, evaluates the answer label logits in a batched forward pass, and applies softmax without generating response text. This means we can experiment with the decision interface **without** necessarily training a new model.

### 2. Serving with llama.cpp

[llama.cpp also supports decision models](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp) using the same `/v1/systemone` API.

For example, we can launch a GGUF decision model from the Hub:

```bash
llama serve -hf ggml-org/Kev-4B-GGUF
```

The default server runs on port 9931.

We can then send a request containing the same state and questions, using `http://localhost:9931/v1/systemone`. With the model selected by the server, we can omit the `model` field from the request.

The useful property is interface compatibility.

```python
curl http://localhost:9931/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ggml-org/Kev-4B-GGUF",
    "state": "I was charged twice. Please refund me today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Bugs and technical issues",
          "sales": "Pricing and subscriptions"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["low", "medium", "high"]
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?"
      }
    }
  }'
```

### 3. Discovering models on the Hugging Face Hub

Hugging Face introduced a dedicated [`decision-model` tag](https://huggingface.co/models?other=decision-model&sort=trending) for discovering compatible models. This is useful because decision models do not have to share a single backbone or architecture.

For example, the ecosystem includes models based on encoder architectures such as ModernBERT, as well as models derived from larger generative language-model backbones.

### 4. Training decision models using Hugging Face tooling

Once we understand inference, the next question is whether we can adapt a decision model. Since decision models are integrated in transformers, we can fine-tune them with the `Trainer` API.

Below is a toy example that fine-tunes `JEV-9B` to route support requests. It is intentionally small, but it shows the parts that matter:

* each example is a decision request (a `state` plus `questions`) paired with the gold answer.
* `model.decision_config.prepare_inputs` turns the request into model inputs, the same way it does at inference time.
* the model returns one logit per option, so a `choice` question is trained with a plain cross-entropy loss over the options.

```python
import os; os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForDecision, Trainer, TrainingArguments

checkpoint = "autotrust/JEV-9B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForDecision.from_pretrained(checkpoint)

examples = [
    ("I was charged twice. Please refund me.", "refund"),
    ("My package has not arrived.", "shipping"),
    ("The application crashes when I open it.", "technical"),
    ("I want my money back.", "refund"),
    ("Where is my delivery?", "shipping"),
    ("I cannot log into my account.", "technical"),
]

options = {
    "refund": "Billing issues and refunds",
    "shipping": "Orders and deliveries",
    "technical": "Software and account issues",
}

def preprocess(text, label):
    request = {
        "state": text,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Classify the support request.",
                "criteria": options,
            }
        },
    }
    (inputs, questions), = model.decision_config.prepare_inputs(
        request, tokenizer
    )
    inputs["marks"] = inputs["marks"].unsqueeze(0)
    inputs["num_options"] = torch.tensor([[len(questions[0].options)]])
    inputs["question_type_ids"] = torch.tensor([[1]])  # choice
    inputs["labels"] = torch.tensor(
        [questions[0].options.index(label)]
    )
    return inputs

dataset = [preprocess(text, label) for text, label in examples]


class DecisionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        logits = model(**inputs)

        logits = logits.unsqueeze(0)  # [3] -> [1, 3]
        loss = F.cross_entropy(logits.float(), labels)

        return (loss, logits) if return_outputs else loss


trainer = DecisionTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./decision-toy",
        max_steps=30,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    ),
    train_dataset=dataset,
    data_collator=lambda batch: batch[0],
)

trainer.train()
```

First-class support for training decision models is also coming to TRL ([huggingface/trl#8040](https://github.com/huggingface/trl/pull/8040)), so we won't have to write this plumbing by hand. At the time of writing, the PR is still open.

If we want to go further and turn a generative LLM into a decision model, [Unsloth provides a recipe](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) that fine-tunes models such as Qwen with LoRA and a decision-scoring head.

## Final thoughts

Decision models are interesting not because classification is new, but because they make **decisions and probabilities first-class outputs for software**.

Traditional classifiers already produce probabilities, and zero-shot classifiers already allow runtime-defined categories. Decision models bring these ideas into a typed interface with multiple question primitives, non-generative scoring, and the possibility of specialized training for reliable, calibrated decisions.
