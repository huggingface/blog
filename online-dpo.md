---
title: "Online DPO"
thumbnail: /blog/assets/online-dpo/thumbnail.png
authors:
- user: TODO
---

# Online DPO

...

## Online preference generation with judge

The online nature of the method implies that preferences must be generated during learning. This can be done using a _judge_, i.e. a model whose purpose is to evaluate the quality of the generated completion. TRL provides a simple interface for the use of different types of judge. In the case of preference optimization, a pairwise judge is used. Such a judge is based on a binary classification model that takes two completions as input and indicates which is the better one. For example:

```python
from trl import HfPairwiseJudge

judge = HfPairwiseJudge()
judge.judge(
    prompts=["What is the capital of France?", "What is the largest planet in the solar system?"]
    completions=[["Paris", "Lyon"], ["Saturn", "Jupiter"]],
) # Outputs: [0, 1]
```

Under the hood, HfPairwiseJudge uses [Llama 3 70B Instruct](https://huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct)` via the [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference).
