---
title: "Decision Models in the Hugging Face Ecosystem"
thumbnail: /blog/assets/decision/thumbnail.png
authors:
  - user: ariG23498
---

# Decision Models in the Hugging Face Ecosystem

Large Language Models (LLMs) are exceptionally good at generating text. We use them to answer questions, write code, reason about problems, and so much more. But not every problem requires generating text.

Consider a customer support system receiving the following message:

"I was charged twice for my subscription. I need a refund today."

The system needs to determine which team should handle the request, how urgent it is, and whether the customer is requesting a refund.

We could ask an LLM to return a JSON object containing these answers. But what if the model could directly evaluate the available options and return a probability distribution over them, *without generating any text*?

This is the key idea behind **decision models**.

## What are decision models?

A decision model takes some input context (state), evaluates a question with a set of predefined possible answers (primitives), and returns a structured decision along with probabilities. Instead of asking a model to generate an answer token by token, we ask it to assign probabilities to the answers that our application already understands.

![image.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/decision/decision-models.png)

[TypeSafe AI introduced Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), describing it as a [*System One model*](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow). The name draws inspiration from the distinction between fast, intuitive thinking (System 1) and slower, deliberative thinking (System 2).

> We prefer the term **decision model**, because it describes the model's computational role rather than invoking a particular theory of cognition. ([A poll on twitter deciding what we should be calling such models](https://x.com/ariG23498/status/2113220513158009022))

For a decision model, the fundamental interface is:

```text
State + Questions + Allowed Answers → Decisions + Probabilities
```

## Why not use LLMs for everything?

We could simply prompt an LLM to return a JSON object containing our decisions. In fact, this is how many applications already use language models for classification, routing, and other structured tasks.

However, there are three problems with this approach.

1. **LLMs are not guaranteed to follow the requested schema:** Even with explicit instructions, a model might produce invalid JSON, introduce unexpected keys, or return an answer outside the allowed choices.
2. **Constrained decoding helps, but does not eliminate the computational overhead:** Libraries such as [Outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar) can enforce grammars or JSON schemas by restricting which tokens the model is allowed to generate at each step. This ensures the output follows the supported structural constraints, but introduces additional work for grammar compilation.
3. **Why generate an entire response:** LLMs are designed for open-ended generation, but when our application only needs to choose between predefined answers, autoregressive decoding can be unnecessary work.

## How are decision models different from classification models?

Before understanding their interface, it is useful to revisit two closely related approaches. Traditional classification and Zero-shot classification.

### Traditional classification

Suppose we train a classifier that routes customer-support requests to three departments:

- Billing
- Technical support
- Sales

A conventional classification model learns a mapping from an input to a predefined set of labels.

During training, the model learns to predict these labels from examples. A classification head typically produces one logit per class, and softmax turns those logits into a probability distribution.

The limitation here is that ***the output classes are baked into the model's classification head***.

If we introduce a fourth department, such as Legal, the existing classifier does not automatically acquire a corresponding output class. We may need to modify the head, collect training data, and **retrain** or adapt the model.

### Zero-shot classification

[Zero-shot classification](https://huggingface.co/tasks/zero-shot-classification) addresses an important part of this limitation.

Instead of requiring the classifier to be trained specifically on every output category, we supply the candidate labels at inference time.

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
#  'scores': [0.8185835480690002,
#   0.09302053600549698,
#   0.05521026626229286,
#   0.03318570926785469]}
```

One common implementation uses a model trained for natural language inference (NLI). Each candidate label is converted into a hypothesis, and the model measures how strongly the input entails that hypothesis.

This already sounds quite similar to a decision model. And in many ways, it is.

### Where decision models differ

Decision models extend this idea beyond simply assigning an input to a category.

A decision model can accept several named questions about the same state, with different answer types. The types can be choosing between alternatives (choice), scoring an ordered scale (score), or assessing whether a statement is true (noul).

More importantly, its interface is designed around the **probability distribution over possible decisions**.

Consider these three approaches:

| Property | Classification | Zero-shot classification | Decision model |
| --- | --- | --- | --- |
| Probability over choices | Yes | Yes | Yes |
| Labels defined at inference | No | Yes | Yes |
| Multiple typed questions | No | No | Yes |
| Ordered scoring and yes/no decisions | Task-specific | Possible with adaptation | Native primitives |

The advantage of decision models is therefore not that they invented runtime-defined labels or probabilistic classification. It is that they organize these capabilities into a consistent, typed interface, potentially backed by architectures and training procedures optimized specifically for decisions.

## The decision-model interface

A decision-model request has two important components, **state** and **questions**.

### State

The state is the information the model should evaluate. It may be a string, a structured JSON object, or, for supported multimodal models, text and images.

For example:

```json
{
  "customer_message": "I was charged twice. Please refund me today.",
  "subscription": "pro",
  "previous_tickets": 2
}
```

A useful analogy is to think of state as the shared context supplied to the model, somewhat like system message (shared context) in an LLM conversation.

However, state is not literally equivalent to a system message. It is primarily the task data being evaluated. The questions and their instructions define what the model should do with that data.

### Questions and their primitives

Questions describe what we want to know about the state. In the decision models interface, three fundamental primitives represent different forms of decisions:

1. **Choice:** Select one of several possible answers.
2. **Score:** Evaluate something on an ordered scale.
3. **Noul:** Answer a yes/no question probabilistically.

Let's understand them using probability theory.

## Choice: A probability distribution over options

Choice is the most intuitive primitive and the closest to traditional classification.

Suppose we ask:

> Which department should handle this customer request?

The available options are Billing, Technical support, and Sales.

The model assigns a logit, to each option. For logits ($z_1,\ldots,z_K$), (with K classes) we can obtain the probabilities using softmax:

$$
p_i=\frac{\exp(z_i/T)}{\sum_{j=1}^{K}\exp(z_j/T)}

$$

Here, (K) is the number of choices and (T) is a temperature parameter. For (T=1), this reduces to the usual softmax.

Consider this illustrative distribution:

| Choice | Probability |
| --- | --- |
| Billing | 0.80 |
| Technical support | 0.15 |
| Sales | 0.05 |

The probabilities sum to one:

$$
\sum_{i=1}^{K}p_i=1.
$$

The model chooses the option with the largest probability:

$$
\hat y=\arg\max_i p_i
$$

In this case, the answer is Billing. But the remaining probabilities are useful too. They tell us how the model distributes its belief among the alternatives. 

## Score: The expectation of an ordered distribution

Choice works when there are several distinct alternatives. But what if we want to measure something on an ordered scale?

Consider the question:

> How urgent is this customer request?

We define three levels:

- 0: Low urgency
- 1: Medium urgency
- 2: High urgency

The model again produces a probability distribution:

| Level | Probability |
| --- | --- |
| 0 — Low | 0.10 |
| 1 — Medium | 0.30 |
| 2 — High | 0.60 |

Rather than returning only the most likely level, the Score primitive computes the **expected value of the level index**.

For a random variable (X) representing the level, the expectation is:

$$
\mathbb{E}[X]=\sum_{i=0}^{K-1}ip_i
$$

For our example:

$$
\mathbb{E}[X] = 0(0.10)+1(0.30)+2(0.60)=1.50
$$

The reported score is therefore `1.5`.

This does not mean that the model selected a fictional level halfway between medium and high. It means that the expectation of its probability distribution is 1.5.

## Noul: A Bernoulli decision

The third primitive is Noul, which represents a binary question.

For example:

> Is the customer requesting a refund?

Rather than defining a list of choices, we ask the model to return the probability that the answer is true. We can understand this using the Bernoulli distribution (noul comes from Bernoulli).

Let (Y) represent a binary outcome:

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

## Confidence: How certain is the model?

So far, we have obtained probability distributions from the model. But the decision model interface also returns a field called `confidence` for Choice and Score.

The full distribution might contain several possible answers, and it is useful to summarize how concentrated the model's belief is. With `confidence` we achieve just that.

For a Choice with (K) options, the confidence is:

$$
\frac{p_{\max}-\frac{1}{K}}{1-\frac{1}{K}}
$$

where ($p_{\max}$) is the largest predicted probability.

The intuition is straightforward:

- A uniform distribution has confidence 0. ($p_{\max} = \frac{1}{K}$)
- A distribution concentrated entirely on one option has confidence 1. ($p_{\max} = 1$)
- The larger the winning probability relative to a uniform distribution, the higher the confidence.

In our previous three-choice example, Billing had probability 0.80:

$$
C=\frac{3(0.80)-1}{2}=0.70
$$

Notice that the winning probability is 0.80, but the reported confidence is 0.70. The confidence normalizes the largest probability relative to the number of available options.

## Decision models in the Hugging Face ecosystem

The interesting development is that decision models are becoming accessible through the same ecosystem used to distribute, serve, and train language models.

### 1. Serving with Transformers

[Transformers supports the `/v1/systemone` endpoint](https://huggingface.co/docs/transformers/main/en/serve-cli/serving#v1systemone).

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
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
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
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.9999295473098755,
        "technical": 0.00005144123861100525,
        "sales": 0.000018924172763945535
      },
      "confidence": 0.9998943209648132
    },
    "urgency": {
      "type": "score",
      "score": 1.8186043351888657,
      "legend": {
        "0": "low",
        "1": "medium",
        "2": "high"
      },
      "probabilities": {
        "0": 0.009196032769978046,
        "1": 0.16300354897975922,
        "2": 0.8278003931045532
      },
      "confidence": 0.727906578220427
    },
    "refund": {
      "type": "noul",
      "noul": 0.9241418242454529
    }
  },
  "usage": {
    "input_tokens": 213,
    "output_tokens": 0
  }
}
```

One particularly interesting detail is that this works with compatible autoregressive language models. Transformers constructs the decision prompts, evaluates the answer-label logits in a batched forward pass, and applies softmax without generating response text.

This means we can experiment with the decision interface **without** necessarily training a new model.

There is an important difference, however, between exposing an LLM through a decision-scoring interface and loading a specialized decision model with its own architecture and scoring head. Transformers' generic next-token-scoring implementation is not automatically equivalent to the native inference path of every specialized model.

```python
curl http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Cloudflare/clef",
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

```bash
{
  "model": "Cloudflare/clef",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 1,
        "technical": 2.689087431021033e-20,
        "sales": 4.974311602001936e-21
      },
      "confidence": 1
    },
    "urgency": {
      "type": "score",
      "score": 2.0000000000000044,
      "legend": {
        "0": "low",
        "1": "medium",
        "2": "high"
      },
      "probabilities": {
        "0": 2.497478712930836e-18,
        "1": 4.3766183914300226e-15,
        "2": 1
      },
      "confidence": 0.9999999999999934
    },
    "refund": {
      "type": "noul",
      "noul": 1
    }
  },
  "usage": {
    "input_tokens": 168,
    "output_tokens": 0
  }
}
```

### 2. Serving with llama.cpp

[llama.cpp also supports decision models](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp) using the same `/v1/systemone` API.

For example, we can launch a GGUF decision model from the Hub:

```bash
llama serve -hf ggml-org/Kev-4B-GGUF
```

The default server runs on port 8080.

We can then send a request containing the same state and questions, using `http://localhost:8080/v1/systemone`. With the model selected by the server, we can omit the `model` field from the request.

The useful property is interface compatibility.

```python
curl http://localhost:8080/v1/systemone \
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

We can experiment with different models or backends while retaining the same decision schema.

Of course, interface compatibility does not imply identical answers, calibration, or performance. These depend on the model and its inference implementation.

### 3. Discovering models on the Hugging Face Hub

Hugging Face introduced a dedicated [`decision-model` tag](https://huggingface.co/models?other=decision-model&sort=trending) for discovering compatible models. This is useful because decision models do not have to share a single backbone or architecture.

For example, the ecosystem includes models based on encoder architectures such as ModernBERT, as well as models derived from larger generative language-model backbones.

### 4. Training decision models using Hugging Face tooling

Once we understand inference, the next question is whether we can train a model to make decisions on a particular domain.

[Unsloth provides a training recipe](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) that fine-tunes language models such as Qwen into decision models. This training pipeline also demonstrates how reusable components from the Hugging Face ecosystem fit together.

A decision-model training dataset can contain three fields:

- `state`: The input context.
- `questions`: The questions and allowed answers.
- `gold`: The correct answers for the training example.

A minimal example might look like this:

```json
example = {
    "state": "I was charged twice. Please refund me.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Bugs and errors",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Is a refund being requested?",
        },
    },
    "gold": {
        "department": "billing",
        "refund": True,
    },
}
```

We can obtain existing datasets from the Hugging Face Hub using `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset(
    "LocalLLaMA/typed-decisions",
    "all",
    split="train",
)
```

This gives us a starting point for working with examples in the decision-model format.

Unsloth's recipe uses parameter-efficient training with LoRA and a decision-scoring head. It handles the training procedure and provides a way to evaluate predictions on held-out examples.

We also provide a training [example](https://github.com/huggingface/trl/pull/7784) inside our TRL repository. At the time of writing, it is still open and being worked on.

## Final thoughts

Decision models are interesting not because classification is new, but because they make **decisions and probabilities first-class outputs for software**.

Traditional classifiers already produce probabilities, and zero-shot classifiers already allow runtime-defined categories. Decision models bring these ideas into a typed interface with multiple question primitives, non-generative scoring, and the possibility of specialized training for reliable, calibrated decisions.
