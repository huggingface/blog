---
title: "使用 DPO 微调 Llama 2" 
thumbnail: /blog/assets/157_dpo_trl/dpo_thumbnail.png
authors:
- user: kashif
- user: ybelkada
- user: lvwerra
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# 使用 DPO 微调 Llama 2


## 简介

基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback，RLHF) 事实上已成为 GPT-4 或 Claude 等 LLM 训练的最后一步，它可以确保语言模型的输出符合人类在闲聊或安全性等方面的期望。然而，它也给 NLP 引入了一些 RL 相关的复杂性: 既要构建一个好的奖励函数，并训练一个模型用以估计每个状态的价值 (value) ; 又要注意最终生成的 LLM 不能与原始模型相差太远，如果太远的话会使得模型容易产生乱码而非有意义的文本。该过程非常复杂，涉及到许多复杂的组件，而这些组件本身在训练过程中又是动态变化的，因此把它们料理好并不容易。

Rafailov、Sharma、Mitchell 等人最近发表了一篇论文 [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)，论文提出将现有方法使用的基于强化学习的目标转换为可以通过简单的二元交叉熵损失直接优化的目标，这一做法大大简化了 LLM 的提纯过程。

本文介绍了直接偏好优化 (Direct Preference Optimization，DPO) 法，该方法现已集成至 [TRL 库](https://github.com/lvwerra/trl) 中。同时，我们还展示了如何在 [stack-exchange preference](https://huggingface.co/datasets/lvwerra/stack-exchange-paired) 数据集上微调最新的 Llama v2 7B 模型， `stack-exchange preference` 数据集中包含了各个 `stack-exchange` 门户上的各种问题及其排序后的回答。

## DPO 与 PPO

在通过 RL 优化人类衍生偏好时，一直以来的传统做法是使用一个辅助奖励模型来微调目标模型，以通过 RL 机制最大化目标模型所能获得的奖励。直观上，我们使用奖励模型向待优化模型提供反馈，以促使它多生成高奖励输出，少生成低奖励输出。同时，我们使用冻结的参考模型来确保输出偏差不会太大，且继续保持输出的多样性。这通常需要在目标函数设计时，除了奖励最大化目标外再添加一个相对于参考模型的 KL 惩罚项，这样做有助于防止模型学习作弊或钻营奖励模型。

DPO 绕过了建模奖励函数这一步，这源于一个关键洞见: 从奖励函数到最优 RL 策略的分析映射。这个映射直观地度量了给定奖励函数与给定偏好数据的匹配程度。有了它，作者就可与将基于奖励和参考模型的 RL 损失直接转换为仅基于参考模型的损失，从而直接在偏好数据上优化语言模型！因此，DPO 从寻找最小化 RLHF 损失的最佳方案开始，通过改变参量的方式推导出一个 _仅需_ 参考模型的损失！

有了它，我们可以直接优化该似然目标，而不需要奖励模型或繁琐的强化学习优化过程。

## 如何使用 TRL 进行训练

如前所述，一个典型的 RLHF 流水线通常包含以下几个环节:

1. 有监督微调 (supervised fine-tuning，SFT)
2. 用偏好标签标注数据
3. 基于偏好数据训练奖励模型
4. RL 优化

TRL 库包含了所有这些环节所需的工具程序。而 DPO 训练直接消灭了奖励建模和 RL 这两个环节 (环节 3 和 4)，直接根据标注好的偏好数据优化 DPO 目标。

使用 DPO，我们仍然需要执行环节 1，但我们仅需在 TRL 中向 `DPOTrainer` 提供环节 2 准备好的偏好数据，而不再需要环节 3 和 4。标注好的偏好数据需要遵循特定的格式，它是一个含有以下 3 个键的字典:

- `prompt` : 即推理时输入给模型的提示
- `chosen` : 即针对给定提示的较优回答
- `rejected` :  即针对给定提示的较劣回答或非给定提示的回答

例如，对于 `stack-exchange preference` 数据集，我们可以通过以下工具函数将数据集中的样本映射至上述字典格式并删除所有原始列:

```python
def return_prompt_and_responses(samples) -> Dict[str, str, str]:
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"], # rated better than k
        "rejected": samples["response_k"], # rated worse than j
    }

dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split="train",
    data_dir="data/rl"
)
original_columns = dataset.column_names

dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)
```

一旦有了排序数据集，DPO 损失其实本质上就是一种有监督损失，其经由参考模型获得隐式奖励。因此，从上层来看，`DPOTrainer` 需要我们输入待优化的基础模型以及参考模型:

```python
dpo_trainer = DPOTrainer(
    model, # 经 SFT 的基础模型
    model_ref, # 一般为经 SFT 的基础模型的一个拷贝
    beta=0.1, # DPO 的温度超参
    train_dataset=dataset, # 上文准备好的数据集
    tokenizer=tokenizer, # 分词器
    args=training_args, # 训练参数，如: batch size, 学习率等
)
```

其中，超参 `beta` 是 DPO 损失的温度，通常在 `0.1` 到 `0.5` 之间。它控制了我们对参考模型的关注程度，`beta` 越小，我们就越忽略参考模型。对训练器初始化后，我们就可以简单调用以下方法，使用给定的 `training_args` 在给定数据集上进行训练了:

```python
dpo_trainer.train()
```

## 基于 Llama v2 进行实验

在 TRL 中实现 DPO 训练器的好处是，人们可以利用 TRL 及其依赖库 (如 Peft 和 Accelerate) 中已有的 LLM 相关功能。有了这些库，我们甚至可以使用 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 库提供的 [QLoRA 技术](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 来训练 Llama v2 模型。

### 有监督微调

如上文所述，我们先用 TRL 的 `SFTTrainer` 在 SFT 数据子集上使用 [QLoRA](https://arxiv.org/abs/2305.14314) 对 7B Llama v2 模型进行有监督微调:

```python
# load the base model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, # "meta-llama/Llama-2-7b-hf"
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

# add LoRA layers on top of the quantized base model
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
...
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args, # HF Trainer arguments
)
trainer.train()
```

### DPO 训练

SFT 结束后，我们保存好生成的模型。接着，我们继续进行 DPO 训练，我们把 SFT 生成的模型作为 DPO 的基础模型和参考模型，并在上文生成的 `stack-exchange preference` 数据上，以 DPO 为目标函数训练模型。我们选择对模型进行 LoRa 微调，因此我们使用 Peft 的 `AutoPeftModelForCausalLM` 函数加载模型:

```python
model = AutoPeftModelForCausalLM.from_pretrained(
    script_args.model_name_or_path, # location of saved SFT model
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    is_trainable=True,
)
model_ref = AutoPeftModelForCausalLM.from_pretrained(
    script_args.model_name_or_path, # same model as the main one
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
...
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=script_args.beta,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
dpo_trainer.train()
dpo_trainer.save_model()
```

可以看出，我们以 4 比特的方式加载模型，然后通过 `peft_config` 参数选择 QLora 方法对其进行训练。训练器还会用评估数据集评估训练进度，并报告一些关键指标，例如可以选择通过 WandB 记录并显示隐式奖励。最后，我们可以将训练好的模型推送到 HuggingFace Hub。

## 总结

SFT 和 DPO 训练脚本的完整源代码可在该目录 [examples/stack_llama_2](https://github.com/lvwerra/trl/tree/main/examples/research_projects/stack_llama_2) 处找到，训好的已合并模型也已上传至 HF Hub (见 [此处](https://huggingface.co/kashif/stack-llama-2))。

你可以在 [这儿](https://wandb.ai/krasul/huggingface/runs/c54lmder) 找到我们的模型在训练过程的 WandB 日志，其中包含了 `DPOTrainer` 在训练和评估期间记录下来的以下奖励指标:

- `rewards/chosen (较优回答的奖励) ` : 针对较优回答，策略模型与参考模型的对数概率二者之差的均值，按 `beta` 缩放。
- `rewards/rejected (较劣回答的奖励) ` : 针对较劣回答，策略模型与参考模型的对数概率二者之差的均值，按 `beta` 缩放。
- `rewards/accuracy (奖励准确率) ` : 较优回答的奖励大于相应较劣回答的奖励的频率的均值
- `rewards/margins (奖励余裕值) ` : 较优回答的奖励与相应较劣回答的奖励二者之差的均值。

直观上讲，在训练过程中，我们希望余裕值增加并且准确率达到 1.0，换句话说，较优回答的奖励高于较劣回答的奖励 (或余裕值大于零)。随后，我们还可以在评估数据集上计算这些指标。

我们希望我们代码的发布可以降低读者的入门门槛，让大家可以在自己的数据集上尝试这种大语言模型对齐方法，我们迫不及待地想看到你会用它做哪些事情！如果你想试试我们训练出来的模型，可以玩玩这个 space: [trl-lib/stack-llama](https://huggingface.co/spaces/trl-lib/stack-llama)。