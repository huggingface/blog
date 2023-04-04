---
title: "StackLlama: A hands-on guide to train LlaMa with RLHF" 
thumbnail: /blog/assets/120_rlhf/thumbnail.png <-UPDATE!
authors:
- user: edbeeching
- user: kashif
- user: younesbelkada
- user: lewtun
- user: lvwerra
---

# StackLlama: A hands-on guide to train LlaMa with RLHF

<!-- {blog_metadata} -->
<!-- {authors} -->

Models such as [ChatGPT]([https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt)), [GPT-4]([https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)), and [Claude]([https://www.anthropic.com/index/introducing-claude](https://www.anthropic.com/index/introducing-claude)) are powerful language models that have been fine-tuned using a method called Reinforcement Learning from Human Feedback to be better aligned with how we expect them to behave and would like to use them.

In this blogpost, we show all the steps involved in training a LlaMa model to answer StackExchange questions with RLHF, through a combination of:

- Supervised Fine Tuning (SFT)
- Reward / preference modeling (RM)
- Reinforcement Learning from Human Feedback (RLHF)

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/instructGPT.png)
*From InstructGPT paper: Ouyang, Long, et al. "Training language models to follow instructions with human feedback." arXiv preprint arXiv:2203.02155 (2022).*

**TODO: add figure and reference**

By combining these approaches, we create the StackLlama model. The model is available on the [🤗 hub](https://huggingface.co/trl-lib/llama-se-rl-peft). We also open-source [the entire training pipeline](https://huggingface.co/docs/trl/index) as part of the trl library.

## The LlaMa model

When doing RLHF it is important to start with a capable model: the RLHF step is only a fine-tuning step to align the model with the way we want to interact with it and how we expect it to respond.  Therefore, we choose to use the recently introduced and performant Llama models. The Llama models are the latest large language models developed by Meta AI. They come in sizes ranging from 7B to 65B parameters were trained on between 1T and 1.4T tokens which makes them very capable. We use the 7B model as the base for all the following steps!

## StackExchange dataset

Gathering human feedback is a complex and expensive endeavour. In order to bootstrap the process for this example while still building a useful model we make use of the [StackExchange dataset](HuggingFaceH4/stack-exchange-preferences). The dataset includes questions and their corresponding answers from the StackExchange platform (including StackOverflow for code as well as many other topic). What makes it so attractive for this use-case is that the answers come together with the number of upvotes and a label for the accepted answer. 

We follow the approach described in [Askell et al. 2021]([https://arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861)) and assign each answer with a score:

`score = log2 (1 + upvotes) rounded to the nearest integer, plus 1 if the answer was accepted by the questioner (we assign a score of −1 if the number of upvotes is negative).`

For the reward model we will always need two answers per question which we can compare as we’ll see later. Some questions have dozens of answers which leads to a lot of possible pairs. To limit the number of data points per question we sample at most 10 answer pairs per question. Finally, we cleanup formatting by converting HTML to Markdown to make the model’s outputs more readable. You can find the dataset as well as the processing notebook [here]([https://huggingface.co/datasets/lvwerra/stack-exchange-paired](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)).


## Efficient Training Strategies

Even training the smallest LlaMa model requires an enormous amount of memory. Some quick math: in fp32 every parameter uses 2 bytes in addition to 8 bytes used e.g. in the Adam optimizer (see the [performance docs]([https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer](https://huggingface.co/docs/transformers/perf_train_gpu_one#optimizer)) in Transformers for more info). So a 7B parameter model would use `(2+8)*7B=70GB` just to fit in memory and would likely need more when you compute intermediate values such as attention scores. So you couldn’t train the model even on a single 80GB A100 like that. You can use some tricks, like more efficient optimizers of half precision training, to squeeze a bit more into memory but sooner or later you’ll run out.

Another option is to use Parameter-Efficient Fine-Tuning (PEFT), such as the [`peft`](https://github.com/huggingface/peft) library that can perform low rank adaption (LoRA) on a model loaded in 8-bit. 

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/lora-animated.gif)
*Low rank adaption of linear layers: extra parameters (in orange) are added next to the frozen layer (in blue), and the resulting encoded hidden states are added together with the hidden states of the frozen layer.*

Low rank adaption of linear layers: extra parameters (in orange) are added next to the frozen layer (in blue), and the resulting encoded hidden states are added together with the hidden states of the frozen layer.

Loading the model in 8bit reduces the memory footprint drastically since you only need one byte per parameter for the weights (e.g. 7B LlaMa is 7GB in memory). Instead of training the original weights directly LoRA adds small adapter layers on top some specific layers (usually the attention layers), thus the number of trainable parameters is drastically reduced.

In this scenario, a rule of thumb is to allocate ~1.2-1.4GB per billion parameter (depending on the batch size and sequence length) to fit the entire fine-tuning setup. As detailed in the blogpost attached above, this enables fine-tuning larger models (up to 50-60B scale models on a NVIDIA A100 80GB) at low cost. 

These techniques have enabled fine-tuning large models on consumer devices and on Google Colab. Notable demos being fine-tuning `facebook/opt-6.7b` (13GB in `float16` ) , and `openai/whisper-large` on Google Colab (15GB GPU RAM). To learn more about using `peft`, refer to our [github repo](https://github.com/huggingface/peft) or the [previous blogpost]([https://huggingface.co/blog/trl-peft](https://huggingface.co/blog/trl-peft)) on training 20b parameter models on consumer hardware. 

Now we can fit very large models into a single GPU, but the training might still be still very slow. The simplest strategy in this scenario is data parallelism: we replicate the same training setup into separate GPUs and pass different batches to each GPU. With this you can parallelize the forward/backward passes of the model and scale with the number of GPUs. 

![chapter10_ddp.png](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/chapter10_ddp.png)

We use either the `transformers.Trainer` or `accelerate` which both support data parallelism without any code changes, by simply passing arguments when calling the scripts with `torchrun` or `accelerate launch`.

 [MAYBE ADD EXAMPLES HERE?]

## Supervised Fine-tuning

Before we start training reward models and tuning our model with RL it helps a lot if the model is already good at the domain we are interested in. In our case we want it to answer questions, while for other use-cases we might want it to follow instructions, in which case instruction tuning is a great idea. The easiest way to achieve this is by continuing to train the language model with the language modeling objective on texts from the domain or task. The [StackExchange dataset](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) is enormous (over 10million instructions), so we can easily train the language model on a subset of it.

There is nothing special about the way we fine-tune the model before doing RLHF - it’s just the causal language modeling objective from pretraining that we apply here. In order to use the data efficiently we use a technique called packing: instead of having one text per sample in the batch and then padding to either the longest text or the maximal context of the model we concatenate a lot of texts with a EOS token in between and cut chunks of the context size to fill the batch without any padding.

![chapter10_preprocessing-clm.png](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/chapter10_preprocessing-clm.png)

The packing is handled by the `ConstantLengthDataset` and we can then use the `Trainer` after loading the model with `peft`. First we load the model in int8 and prepare it for training and then add the LoRA adapters on top.

```python
# load model in 8bit
model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        device_map={"": Accelerator().local_process_index}
    )
model = prepare_model_for_int8_training(model)

# add LoRA to model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
```

We train the model for a few thousand steps with the causal language modeling objective and save the model. Since we will tune the model again with different objectives we merge the adapter weights with the original model weights.

**Disclaimer:** due to Llama’s license we release only the adapter weights for this and the model checkpoints in the following sections.

Now that we have fine-tuned the model for the task we are ready to train a reward model.

## Reward modeling and human preferences

In principle we could fine-tune the model using RLHF directly with the human annotations. However, this would require us to send some samples to humans for rating after each optimization iteration, which is both expensive and slow due to the number of training samples needed for convergence and the inherent latency of human reading and annotator speed.

A trick that works well instead of direct feedback is to train a reward model on human annotations that are collected before the RL loop. The goal of the reward model is to imitate how a human would rate a text. There are several possible strategies to build a reward model: the simplest way would be to simply predict the annotation (e.g. a rating score or a binary value for “good”/”bad”). In practice, what works better is to predict the ranking of two examples, where reward model is presented with two candidates \\( (y_k, y_j) \\) for a given prompt \\( x \\) and has to predict which one would be rated higher by a human annotator.

This can be translated into the following loss function:

\\( \operatorname{loss}(\theta)=- E_{\left(x, y_j, y_k\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_j\right)-r_\theta\left(x, y_k\right)\right)\right)\right] \\)

where \\( r \\) is the model’s score and \\( y_j \\) is the preferred candidate.

With the StackExchange dataset we can infer which of the two answers was preferred by the users based on the score. With that information and the loss as defined above we can then proceed to modify the `transformers.Trainer` by adding a custom loss function. 

```python
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"],  attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

We utilize a subset of a 100,000 pair of candidates and evaluate on a held-out set of 50,000. With a modest training batch size of 4 we train the Llama model using the LoRA `peft` adapter for a single epoch using the Adam optimizer with BF16 precision. Our LoRA configuration is:

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
```

The training is logged via [Weights & Biases](https://wandb.ai/krasul/huggingface/runs/wmd8rvq6?workspace=user-krasul) and took a few hours on 8-A100 GPUs using the 🤗 research cluster and the model achieves a final **accuracy of 67%**. Note that although this sounds like a low score the task is also very hard even for human annotators.

The resulting adapter can be merged into the frozen model and saved for further downstream use, as detailed in the next section.

## Reinforcement Learning from Human feedback

With the fine-tuned language model and the reward model at hand we are now ready to run the RL loop. It follows roughly three steps:

1. Generate responses from prompts
2. Rate the responses with the reward model
3. Run a reinforcement learning policy-optimization step with the ratings

![Untitled](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/trl_loop.png)

The Query and Response prompts are templated as follows before being tokenized and passed to the model:

```bash
Question: <Query>

Answer: <Response>
```

The same template was used for SFT, RM and RLHF stages.

A common issue with training language model with RL is that they can learn to exploit the reward model by generating complete gibberish which causes the reward model to assign high rewards. To balance this we add a penalty to the reward: we keep a reference of the model that we don’t train and compare the new model’s generation to the reference one by computing the KL-divergence:

\\( \operatorname{R}(x, y)=\operatorname{r}(x, y)- \beta \operatorname{KL}(x, y) \\)

where \\( r \\) is the reward from the reward model and  \\( \operatorname{KL}(x,y) \\) is the KL-divergence between the current  policy and the reference model. 

Once more, we utilize `peft` for memory-efficient training, which offers an extra advantage in the RLHF context. Here, the reference model and policy share the same base, which we load in 8-bit and freeze during training. We exclusively optimize the policy's LoRA weights using PPO, while sharing the base model's weights.

```python
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]
		
		# sample from the policy and to generate reponses
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
		# Log stats to Wandb
    ppo_trainer.log_stats(stats, batch, rewards)
```

We train for 30 hours on 8 A100-80GB GPUs, using the 🤗 research cluster. All the training statistics of the training run are available on [Weights & Biases](https://wandb.ai/edbeeching/trl/runs/5xgr1ifp?workspace=user-edbeeching).

![Per batch reward at each step during training. The model’s performance plateaus after around 1000 steps.](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/wandb_reward.png)
*_Per batch reward at each step during training. The model’s performance plateaus after around 1000 steps.*

## Interactive demo

In addition to hosting the StackLlama adapter weights on the 🤗 hub. We share a Space where you can try out the model for yourself.

[LEWIS’S DEMO]

## Conclusion

In this post we went through the entire training cycle for RLHF starting at preparing a dataset with human annotations, adapting the language model to the domain, training a reward model, and finally training a model with RL. 

By using `peft` anyone can run our example on a single GPU! If you find training is too slow you can use data parallelism with no code changes and scale training by adding more GPUs.

For a real use-case this is just the first step! Once you have a model trained you need to evaluate it and compare against other models to see how good it is. This can for example be done by ranking generations of different model versions similar to how we built the reward dataset. 

Once you added the evaluation step the fun begins: you can start iterating on your dataset and  model training setup to see if there are ways to improve the model. You could add other datasets to the mix or apply better filters to the existing one. On the other hand you could try different model sizes and architecture for the reward model or train for longer.

We are actively improving TRL to make all steps involved in RLHF more accessible and are excited see the things people build with it!  Check out the [issues on GitHub](https://github.com/lvwerra/trl/issues) if you're interested in contributing.

## Acknowledgements

We thank Philipp Schmid for sharing his wonderful Space of streaming text generation upon which our demo was based. We also thank Nathan Lambert and Nazneen Rajani for feedback on early drafts of the post and project. 





