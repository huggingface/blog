---
title: 'Preference Tuning LLMs with Direct Preference Optimization Methods'
thumbnail: /blog/assets/pref-tuning/thumbnail.png
authors:
- user: edbeeching
- users: kashif
- users: lewtun
- users: lvwerra
- users: osanseviero
---

# Preference Tuning LLMs with Direct Preference Optimization Methods

**TL;DR**

We evaluate three promising methods to align language models without reinforcement learning (or preference tuning) on a number of models and hyperparameter settings. In particular we train:
*  Direct Preference Optimization (DPO)
* Identity Preference Optimisation (IPO)
* Kahneman-Taversky Optimisation (KTO) 
and find that...

## Alignment without Reinforcement Learning 

|![Image from the DPO paper ([https://arxiv.org/abs/2305.18290](https://arxiv.org/pdf/2305.18290.pdf))](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/dpo.png)|
|:--:|
|Image from the DPO paper ([https://arxiv.org/abs/2305.18290](https://arxiv.org/pdf/2305.18290.pdf))|

[Direct Preference Optimization (DPO)](https://huggingface.co/papers/2305.18290) has emerged as a promising alternative for aligning Large Language Models (LLMs) to human or AI preferences. Unlike [traditional alignment methods](https://www.notion.so/Aligning-LLMs-with-Direct-Preference-Optimization-Methods-517ba7f77356497ab0a5a91394898a3c?pvs=21), which are based on reinforcement learning, DPO recasts the alignment formulation as a simple loss function that can be optimised directly on a dataset of preferences \\( \{(x, y_w, y_l)\} \\), where \\(x\\) is a prompt and \\(y_w,y_l\\) are the preferred and dispreferred responses. This makes DPO simple to use in practice and has been applied with success to train models like [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) and [Intel’s NeuralChat](https://huggingface.co/Intel/neural-chat-7b-v3-3).

The success of DPO has prompted researchers to develop new loss functions that generalise the method in two main directions:
* **Robustness**: One shortcoming of DPO is that it tends to quickly overfit on the preference dataset. To avoid this, researchers at Google DeepMind introduced [Identity Preference Optimisation (IPO)](https://huggingface.co/papers/2310.12036), which adds a regularisation term to the DPO loss and enables one to train models to convergence without requiring tricks like early stopping.
* **Dispensing with paired preference data altogether**: Like most alignment methods, DPO requires a dataset of paired preferences \(( \{(x, y_w, y_l)\}\\), where annotators label which response is better according to a set of criteria like helpfulness or harmfulness. In practice, creating these datasets is a time consuming and costly endeavour. ContextualAI recently proposed an interesting alternative called [Kahneman-Taversky Optimisation (KTO)](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf), which defines the loss function entirely in terms of individual examples that have been labelled as "good" or "bad" (for example, the 👍 or 👎 icons one sees in chat UIs). These labels are much easier to acquire in practice and KTO is a promising way to continually update chat models running in production environments.

At the same time, these various methods come with hyperparameters, the most important one being \((\beta \)), which controls how much to weight the preference of the reference model. With these alternatives now available in the practitioner’s arsenal through libraries like 🤗 [TRL](https://github.com/huggingface/trl), a natural question then becomes which of these methods and hyperparameters produce the best chat model?

This post aims to answer this question by performing an empirical analysis of the three methods. We will sweep over key hyperparameters such as \\(\beta\\) and training steps, then evaluate the resulting models’ performance via [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench), which is a common benchmark to measure chat model capabilities.

Let’s get started!

## Links

Here are the important links associated with our analysis:

- 🏋️‍♀️ Code to align models with 🤗 TRL: [https://gist.github.com/kashif/372f101771c7ddc065d2799f2839ed0a](https://gist.github.com/kashif/372f101771c7ddc065d2799f2839ed0a)
- 👩‍⚖️ Code to evaluate models on MT-Bench: [https://gist.github.com/kashif/cd8a966e5e624eacbd248034996ab316](https://gist.github.com/kashif/cd8a966e5e624eacbd248034996ab316)
- 📚 The collection of dataset and models we used: [https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e](https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e)

## Experimental Setup

There are two main ingredients needed for the alignment methods we consider:

- **An initial chat model to optimise.** We use [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) as it’s one of the best 7B parameter chat models that hasn’t been subject to any alignment techniques.
- **A dataset of human or AI preferences.** We use Intel’s `orca_dpo_pairs` [dataset](https://huggingface.co/datasets/Intel/orca_dpo_pairs), which consists of 13k prompts where the chosen response is generated by GPT-4, and the undesired response is generated by Llama-Chat 13b. This is the dataset behind NeuralChat and NeuralHermes-2.5-Mistral-7B. Since KTO doesn’t require pairwise preferences per se, we simply treat the GPT-4 responses as “good” labels and the Llama-Chat 13b ones as “bad”.

More specifically, we will use the following to setup the dataset (where the `chatml_format` is defined in the gist):

```python
from datasets import load_dataset

dataset = load_dataset("Intel/orca_dpo_pairs")["train"]

model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

original_columns = dataset.column_names
dataset = dataset.map(
    chatml_format, # <- check the gist where this is defined
    remove_columns=original_columns,
    fn_kwargs={"tokenizer": tokenizer},
)
```

For fine-tuning, we will, instead of fine tuning the whole model, instead use the following LoRA based fine-tuning with the configuration:

```python
# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "k_proj",
        "gate_proj",
        "v_proj",
        "up_proj",
        "q_proj",
        "o_proj",
        "down_proj",
    ],
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    load_in_4bit=True,
)
model.config.use_cache = False # if doing gradient check-pointing
```

The main logic of our experiment will be reflected in the training arguments and arguments to the `DPOTrainer`:

```python
# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    max_steps=FLAGS.max_steps, # <- the different training steps: 200 and 800
    save_strategy="no",
    logging_steps=1,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
    run_name=f"{model_name}-{FLAGS.loss_type}-beta-{FLAGS.beta}-steps-{FLAGS.max_steps}",
    output_dir=f"final_checkpoint-{FLAGS.loss_type}-beta-{FLAGS.beta}-steps-{FLAGS.max_steps}",
    overwrite_output_dir=True,
)

# Create DPO trainer, where the model without the lora weights is the ref model
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=FLAGS.beta,           # <- the different betas: 0.1, ..., 0.9
    max_prompt_length=1024,
    max_length=1536,
    loss_type=FLAGS.loss_type, # <- the different loss types: ipo, kto, sigmoid
)
```

## Hyperparameter Sweep

We will, as mentioned, train the `DPO`, `IPO` and `KTO` methods via the `loss_type` argument to the `DPOTrainer` with the `beta` going from `0.1`, `0.2`, …, `0.9`. We will train for `200` steps (which is approximately a quarter of the dataset) and then also train for `800` steps of SGD. All other hyperparameters are kept the same during each run, including the data splits via the random seed. 

Once the models have been trained, the resulting LoRA weights can be merged with the original base model and evaluated using the 3 steps of the `llm_judge` MT-Bench benchmark. For merging the models, we use the following snippet.

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Merge base model with the adapter
model = PeftModel.from_pretrained(
    base_model, 
    f"final_checkpoint-{FLAGS.loss_type}-beta-{FLAGS.beta}-steps-{FLAGS.max_steps}"
)
model = model.merge_and_unload()

# Evaluate model
...
```

## Results


| ![Zephyr comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/Zephyr-comp.png) |
|:--:|
| Caption |

| ![OpenHermes comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/openhermes-comp.png) |
|:--:|
| Caption |

| ![Zephyr scan](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/zephyr_scan.png) |
|:--:|
| Caption |

| ![OpenHermes scan](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/openhermes_scan.png) |
|:--:|
| Caption |

